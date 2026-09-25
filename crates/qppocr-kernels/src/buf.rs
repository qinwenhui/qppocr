//! 池化的张量缓冲（`pool.hpp` + `buf.hpp` ）。
//!
//! ## 为什么需要它（ 注释的实测结论，照抄）
//!
//! 图节点的输出缓冲：分配→写满→被几个消费者读→释放，每节点一次、每次
//! 推理几百次。每次都还回 CRT 再要新的，代价是一对 VirtualFree/
//! VirtualAlloc 加上——真正贵的部分——新块每 4 KB 一次 soft page fault
//! 和内核清零。同一 sgemm 写 fresh 47 MB 输出比 warm 的多 9.6 ms
//!（62%）。**成本是 fault，不是 memset**：warm 块上的 memset 便宜，
//! 所以只有「块永远不还给 OS」才能把 fault 拿掉。
//!
//! 块按请求字节数的最高位分桶（一桶服务到两倍尺寸的请求）。把 warm 块
//! 过度分配几乎免费（没碰的页只花地址空间不花时间），而尺寸类 miss 每
//! 4 KB 一次 fault——这个方向的取舍值得。
//!
//! ## Rust 侧的安全边界
//!
//! [`F32Buf::with_uninit`] / [`F32Buf::resize_uninit`] 跳过清零——
//! **调用方必须在任何读之前写满每个元素**，否则读到的是池里上一任的
//! 陈旧数据（位模式上可能是 NaN 陷阱值）。这个契约与 `resize_uninit`
//! 相同，也是它只能住在 kernels crate（唯一 unsafe 边界）的原因。

// 本模块是进程级内存池原语：每个 unsafe 块的合约都写在其所属函数的
// `# Safety` 段（块本身重复同样的话没有信息量）。
#![allow(clippy::undocumented_unsafe_blocks)]

use std::sync::Mutex;

const MAX_CLASS: usize = 34; // 16 GB 上限

/// 桶里的空闲块指针。池按拥有者独占语义借出/回收，指针本身只在
/// Mutex 内移动。
#[derive(Clone, Copy)]
struct FreeBlockWrap(*mut u8);
// SAFETY: 块指针只在锁内入桶/出桶，借出后由 F32Buf 独占持有。
unsafe impl Send for FreeBlockWrap {}

struct PoolInner {
    free: Vec<Vec<FreeBlockWrap>>, // 每类一个桶
    held_bytes: usize,
    // 上限（保守默认； 256 MB 是为单机跑分调的，
    // Rust 版默认保守，让调用方显式开大）
    cap_bytes: usize,
    per_class_cap: [usize; 3], // small(<64KB) / mid(<4MB) / big
}

/// 块按 2^k 字节分桶回收。全局、永不销毁（OS 在进程退出时收走）。
impl PoolInner {
    /// 全局池（OnceLock 惰性初始化）。
    fn get() -> &'static Mutex<PoolInner> {
        static POOL: std::sync::OnceLock<Mutex<PoolInner>> = std::sync::OnceLock::new();
        POOL.get_or_init(|| {
            Mutex::new(PoolInner {
                free: (0..=MAX_CLASS).map(|_| Vec::new()).collect(),
                held_bytes: 0,
                cap_bytes: 256 << 20, // 对齐 基准的 256 MB：det 的 concat/中间张量 95 MB 级，64 MB 上限会把最大块全部挤出池（实测端到端慢 1.4x 的主因之一）
                per_class_cap: [16, 8, 8],
            })
        })
    }
}

/// 字节数的尺寸类（最高位位置；block_size(k) = 2^k ≥ bytes）。
fn class_of(bytes: usize) -> usize {
    usize::BITS as usize - (bytes - 1).leading_zeros() as usize
}

/// 从池取一块（或 malloc 新块）。
///
/// SAFETY：返回的指针来自 malloc 或池的空闲桶；caller 负责最终用
/// [`dealloc`] 归还同一指针与同一尺寸类。
unsafe fn alloc(bytes: usize) -> *mut u8 {
    let bytes = bytes.max(1);
    let k = class_of(bytes);
    if k > MAX_CLASS {
        return unsafe { libc_malloc(bytes) };
    }
    let mut g = PoolInner::get().lock().unwrap();
    if let Some(FreeBlockWrap(p)) = g.free[k].pop() {
        g.held_bytes -= 1 << k;
        return p;
    }
    unsafe { libc_malloc(1 << k) }
}

/// 归还一块到池（超限则 free）。
///
/// SAFETY：`p` 必须来自 [`alloc`]，且此后不再使用。
unsafe fn dealloc(p: *mut u8, bytes: usize) {
    if p.is_null() {
        return;
    }
    let bytes = bytes.max(1);
    let k = class_of(bytes);
    if k > MAX_CLASS {
        unsafe { libc_free(p) };
        return;
    }
    let mut g = PoolInner::get().lock().unwrap();
    let blk = 1usize << k;
    let cap_idx = if blk >= 4 << 20 {
        2
    } else if blk >= 64 << 10 {
        1
    } else {
        0
    };
    if g.free[k].len() >= g.per_class_cap[cap_idx] || g.held_bytes + blk > g.cap_bytes {
        unsafe { libc_free(p) };
        return;
    }
    g.free[k].push(FreeBlockWrap(p));
    g.held_bytes += blk;
}

// 不引 libc 依赖：直接声明 CRT 的两个符号（MSVC/GNU 都导出）。
unsafe extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(p: *mut u8);
}

/// SAFETY：malloc 的契约。
unsafe fn libc_malloc(size: usize) -> *mut u8 {
    unsafe { malloc(size) }
}
/// SAFETY：free 的契约。
unsafe fn libc_free(p: *mut u8) {
    unsafe { free(p) }
}

/// f32 缓冲：内存借自池，drop 归还。
///
/// 大多数地方当 `&[f32]` / `&mut [f32]` 用（Deref）。
#[derive(Debug)]
pub struct F32Buf {
    ptr: std::ptr::NonNull<f32>,
    len: usize,
    cap_bytes: usize,
}

// 池在线程间共享；F32Buf 的所有权转移即线程间转移。
// SAFETY: 指针指向堆块，池的互斥锁保护桶管理，数据本身按拥有者独占访问。
unsafe impl Send for F32Buf {}
// SAFETY: 同上——数据访问由 &mut 借用规则约束。
unsafe impl Sync for F32Buf {}

impl F32Buf {
    /// 空缓冲。
    pub fn new() -> Self {
        F32Buf {
            // NonNull::dangling：对齐的悬垂指针，len=0 时永不解引用
            ptr: std::ptr::NonNull::dangling(),
            len: 0,
            cap_bytes: 0,
        }
    }

    /// n 个清零元素。
    pub fn with_zeroed(n: usize) -> Self {
        let mut b = Self::new();
        b.resize_zeroed(n);
        b
    }

    /// n 个**未初始化**元素。
    ///
    /// # Safety
    ///
    /// 调用方合约：任何读（Deref 到的切片）之前必须写满全部元素；否则读到池里
    /// 上一任的陈旧位模式。 `resize_uninit` 同款合约。
    pub unsafe fn with_uninit(n: usize) -> Self {
        let mut b = Self::new();
        unsafe { b.resize_uninit(n) };
        b
    }

    /// 从 Vec 拷贝。
    ///
    /// 不做零拷贝接管：Vec 的分配尺寸未必是 2 的幂，直接把它记成池块
    /// 会让 dealloc 归进「块可能比桶标称小」的桶，破坏「桶里的块 ≥ 桶
    /// 标称」的不变量。调用方（executor 的权重装载）只跑一次，拷贝可接受。
    pub fn from_vec(v: &[f32]) -> Self {
        let mut b = Self::with_zeroed(v.len());
        b.as_mut_slice().copy_from_slice(v);
        b
    }

    /// 长度。
    pub fn len(&self) -> usize {
        self.len
    }
    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// 拷贝成 Vec（测试与序列化）。
    pub fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }
    /// 切片视图。
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: len 只有 with_zeroed/resize_zeroed（清零）或调用方写满
        // uninit 后才被外界读到；指向的块在 drop 前有效。
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    /// 可变切片视图。
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: 同上；&mut 保证独占。
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// 扩到 n：新元素清零（与 `Vec::resize(n, 0.)` 等价）。
    pub fn resize_zeroed(&mut self, n: usize) {
        let old = self.len;
        self.reserve(n);
        self.len = n;
        if n > old {
            // SAFETY: [old, n) 刚保留、尚未暴露。
            unsafe { std::ptr::write_bytes(self.ptr.as_ptr().add(old), 0, n - old) };
        }
    }

    /// 扩到 n：**不碰新元素**。
    ///
    /// # Safety
    ///
    /// 调用方合约：读之前写满（同 [`Self::with_uninit`]）。
    pub unsafe fn resize_uninit(&mut self, n: usize) {
        self.reserve(n);
        self.len = n;
    }

    /// 保证容量 ≥ n×4 字节（池块重分配：新块 + memcpy 旧内容 + 归还旧块）。
    fn reserve(&mut self, n: usize) {
        let need = n * 4;
        if need <= self.cap_bytes {
            return;
        }
        // SAFETY: alloc 返回有效块或 null。null 只在 malloc 失败时——
        // 让它 panic（内存耗尽没有恢复路径）。
        let np = unsafe { alloc(need) };
        assert!(!np.is_null(), "F32Buf: out of memory ({need} bytes)");
        let np = std::ptr::NonNull::new(np.cast::<f32>()).unwrap();
        if self.len > 0 && self.cap_bytes > 0 {
            // SAFETY: 旧块至少 len 个元素（此前 reserve 过）。
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), np.as_ptr(), self.len);
            }
            // SAFETY: 旧块来自 alloc。
            unsafe { dealloc(self.ptr.as_ptr().cast::<u8>(), self.cap_bytes) };
        }
        self.ptr = np;
        // cap_bytes 记「向池申请的字节数」，dealloc 按它分桶。
        // alloc 实际给的块 ≥ need（桶按 2^ceil(log2 need))——两个尺寸
        // 进同一桶，记 need 与记块大小等价。
        self.cap_bytes = need;
    }
}

impl Default for F32Buf {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for F32Buf {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        self.as_slice()
    }
}
impl std::ops::DerefMut for F32Buf {
    fn deref_mut(&mut self) -> &mut [f32] {
        self.as_mut_slice()
    }
}

impl Clone for F32Buf {
    fn clone(&self) -> Self {
        let mut b = F32Buf::with_zeroed(self.len);
        b.as_mut_slice().copy_from_slice(self.as_slice());
        b
    }
}

impl PartialEq for F32Buf {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl Eq for F32Buf {}

impl Drop for F32Buf {
    fn drop(&mut self) {
        if self.cap_bytes > 0 {
            // SAFETY: 块来自 alloc（reserve 的唯一写入路径），按记名的
            // cap_bytes 归还到同一尺寸类。
            unsafe { dealloc(self.ptr.as_ptr().cast::<u8>(), self.cap_bytes) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buf_basic() {
        let mut b = F32Buf::with_zeroed(100);
        assert_eq!(b.len(), 100);
        assert!(b.iter().all(|&v| v == 0.0));
        b[5] = 3.5;
        assert_eq!(b[5], 3.5);
        b.resize_zeroed(200);
        assert_eq!(b.len(), 200);
        assert_eq!(b[5], 3.5); // 扩容保留旧内容
        // 池回收：drop 后再分配同尺寸应命中桶（行为验证——不崩溃即可）
        {
            let _ = F32Buf::with_zeroed(200);
        }
        let c = b.clone();
        assert_eq!(c, b);
    }

    #[test]
    fn buf_growth_patterns() {
        // 反复增长（模拟 tile 的 cols 复用）
        let mut b = F32Buf::new();
        for round in 0..10 {
            let n = 1000 * (round + 1);
            b.resize_zeroed(n);
            b[n - 1] = round as f32;
            assert_eq!(b[n - 1], round as f32);
        }
        // 清零语义
        b.resize_zeroed(10);
        assert_eq!(b.len(), 10);
    }
}
