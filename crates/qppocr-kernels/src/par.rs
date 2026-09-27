//! 并行调度与 fork 阈值。
//!
//! 并行调度与 fork 阈值。
//!
//! 调度器是 [`crate::pool`]——一套自研 fork-join 池：批量唤醒、主线程参与、
//! join 自旋、参与者计数屏障。多线程扩展从 rayon 的 1.86x 追到 2.07x 靠的
//! 就是这套语义，不是内核本身。嵌套 `parallel_for` 是死锁；`sgemm_serial` /
//! 串行 im2col 这些「调用方已在并行区」的串行路径因此是硬性的（它们还有
//! 位级一致的作用，见 `gemm.rs`）。
//!
//! 池的构造与所有 worker 线程的诞生都发生在 [`crate::pool`] 里。
//!
//! ## 阈值不重调
//!
//! 下面这些门控每一条都是量出来的（不是拍脑袋），实测见 CHANGELOG：
//!
//! - `fork_min_macs = 4e6`：16 线程 fork 一次约 97 us，低于 ~4M MAC 的算子
//!   并行收益盖不过 fork 本身。rec 的 depthwise conv 曾经在 ~10 us 的单元上
//!   fork，花掉 6 倍于工作量本身的开销。
//! - `gemm_par_min = 2e5`：GEMM 的 flops 门槛。
//! - `elem_fork_min_bytes = 1<<20`：纯搬运算子的字节门槛——单线程 memcpy 约
//!   12 GB/s，1 MB ≈ 一次 fork 的工作量。
//!
//! ## FTZ/DAZ
//!
//! x86 的 FMA 单元在非正规数上严重降速（实测 19 倍），推理中间量很容易踩中。
//! [`enable_flush_denormals`] 必须在**任何 worker 线程诞生之前**于主线程调用
//! ——MXCSR 是每线程的，子线程继承创建者的标志位；池在构造函数里做同样的事。

/// 每线程任务块数的乘子：原子领票的动态负载均衡，不是 rayon 的静态微任务。
/// rotated 语料实测 4/8/16/32 在噪声内，**1 是离群值**（16 线程 -8%）。
#[cfg(feature = "parallel")]
const TP_CHUNKS: usize = 8;

/// 并行门槛（默认值见 [`Thresholds::default`]，实测标定）。
#[derive(Clone, Copy, Debug)]
pub struct Thresholds {
    /// 低于这么多 MAC 的算子不并行（`fork_min_macs`）。
    pub fork_min_macs: f64,
    /// GEMM 的并行 flops 门槛（`gemm_par_min`）。
    pub gemm_par_min: f64,
    /// 逐元素算子的字节门槛（`elem_fork_min_bytes`）。
    pub elem_fork_min_bytes: usize,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            fork_min_macs: 4e6,
            gemm_par_min: 2e5,
            elem_fork_min_bytes: 1 << 20,
        }
    }
}

static THRESHOLDS: std::sync::OnceLock<Thresholds> = std::sync::OnceLock::new();

/// 当前生效的并行门槛。
pub fn thresholds() -> &'static Thresholds {
    THRESHOLDS.get_or_init(Thresholds::default)
}

/// 工作量（MAC 数）是否值得 fork。
pub fn worth_forking(macs: f64) -> bool {
    macs >= thresholds().fork_min_macs && threads() > 1
}

/// 搬运量（字节数）是否值得 fork。
pub fn worth_forking_bytes(bytes: usize) -> bool {
    threads() > 1 && bytes >= thresholds().elem_fork_min_bytes
}

/// 本线程所在池的参与者数（含调用线程）。分片线程下小于 [`threads`]。
///
/// 只用于**负载均衡**类的决策（切几块）；切分轴/分块数这类进位的东西
/// 一律用全局的 [`threads`]。
#[cfg(feature = "parallel")]
pub fn pool_threads() -> usize {
    crate::pool::pool_thread_count()
}
#[cfg(not(feature = "parallel"))]
#[inline]
pub fn pool_threads() -> usize {
    1
}

/// 可用线程数（池大小，含主线程）。`parallel` feature 关闭时恒为 1。
pub fn threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        crate::pool::thread_count()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// 池的总线程数（含主线程）。CLI 的批量 `--workers` 按它折算每进程
/// 线程数（`pool / worker 数`），对齐  `threads_each`。
#[cfg(feature = "parallel")]
pub fn pool_thread_count() -> usize {
    crate::pool::thread_count()
}

/// 本线程后续的 `parallel_for` 走**串行执行**（不发布任务、不唤醒
/// worker，直接在本线程跑完全部 chunk）。
///
/// 切分决策不变——`threads()` 仍报池的真实大小，因此各内核的分块与
/// 累加序与并行路径完全一致（逐位可验证）。用途是「逐行并行」：每行
/// 一个线程、行内串行，避免行内的 fork 争用 `fork_mu` 互相排队。
pub fn set_serial_execution(on: bool) {
    #[cfg(feature = "parallel")]
    crate::pool::set_serial_exec(on);
    #[cfg(not(feature = "parallel"))]
    let _ = on;
}

/// 本线程是否处于串行执行模式（见 [`set_serial_execution`]）。
pub fn serial_execution() -> bool {
    #[cfg(feature = "parallel")]
    {
        crate::pool::serial_exec()
    }
    #[cfg(not(feature = "parallel"))]
    {
        false
    }
}

/// 请求池的线程数（含主线程，0 = 自动）。必须在**第一次并行算子**之前
/// 调用——池在首用时定容，之后请求不再生效（「最早调用者定容」）。
/// 显式数字不设 16 上限。
///
/// 环境变量 `QPPOCR_THREADS` 等价（优先级低于本函数）。
#[cfg(feature = "parallel")]
pub fn set_threads(n: usize) {
    crate::pool::request_threads(n);
}

/// 按当前配置算线程数但**不建池**（给「先请求布局再使用」的调用方）。
#[cfg(feature = "parallel")]
pub fn planned_threads() -> usize {
    crate::pool::planned_threads()
}
#[cfg(not(feature = "parallel"))]
#[inline]
pub fn planned_threads() -> usize {
    1
}

/// 请求池布局：`sizes[0]` 是默认池（det 与所有非分片调用走它），
/// 其余是**分片池**，供两级并行用——外层每个线程占一个，其算子再在本池内
/// fork，互不排队。必须在第一次 fork 之前调用（池首用定容）。
///
/// 全部线程一次性创建；空闲的池只在 condvar 上等，不占 CPU。
#[cfg(feature = "parallel")]
pub fn request_pools(sizes: &[usize]) {
    crate::pool::request_pools(sizes);
}

/// 本线程用哪个池（0 = 默认池）。仅对**当前线程**生效，且只影响它自己发起的
/// `fork_join`；分片线程由调用方显式设置。
#[cfg(feature = "parallel")]
pub fn set_pool_slot(i: usize) {
    crate::pool::set_pool_slot(i);
}

/// 在 x86 上打开 FTZ/DAZ（刷新非正规数）。
///
/// 必须在 worker 线程诞生前于主线程调用；`qppocr-core` 在引擎构造时做这件事。
/// 这是**逐位可复现**的前提之一：FTZ 改变非正规数结果的位模式。
///
/// （`_mm_getcsr`/`_mm_setcsr` 内联函数已被 std 标记 deprecated，这里按其
/// 建议改用内联汇编——语义就是 STMXCSR / LDMXCSR 两条指令。）
pub fn enable_flush_denormals() {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: STMXCSR 把当前 MXCSR 存到栈上的 u32、LDMXCSR 从中读回。
        // 只影响当前线程的 MXCSR（FTZ|DAZ 位），无其他内存副作用。
        // 两条指令的操作数都是 m32，所以经指针间接传（借用各自独立建立，
        // 编译器能看到 word 的数据流）。
        let mut word: u32 = 0;
        unsafe {
            // SAFETY: 见上方注释。
            std::arch::asm!("stmxcsr [{0}]", in(reg) &mut word, options(nostack));
            word |= 0x8040; // FTZ | DAZ
            std::arch::asm!("ldmxcsr [{0}]", in(reg) &word, options(nostack));
        }
    }
}

/// fork-join 的 `fn(begin, end)` over `[0, n)`，主线程参与。
///
/// `min_grain`：n 低于它就不 fork，直接串行跑（对齐  `parallel_for`
/// 的默认 grain 256；单元本身就是重活的调用方传 1）。
///
/// **不保证**负载均衡的最优切分，只保证覆盖：每个 `begin..end`
/// 恰好执行一次，联合覆盖 `[0, n)`。输出不依赖切分（每个元素的累加顺序固定）。
pub fn parallel_for<F>(n: usize, min_grain: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    if n == 0 {
        return;
    }
    let t = threads(); // 全局：只做「有没有并行」的粗判
    let tp = pool_threads(); // 本地：决定切几块
    if t == 1 || tp == 1 || n < min_grain.max(2) {
        f(0, n);
        return;
    }
    #[cfg(feature = "parallel")]
    {
        let nchunk = (tp.saturating_mul(TP_CHUNKS)).min(n);
        let chunk = n.div_ceil(nchunk);
        let base = chunk; // 每 chunk 的步长
        let total = n;
        crate::pool::fork_join(nchunk, &|i, _e| {
            let b = i * base;
            let e = ((i + 1) * base).min(total);
            if b < e {
                f(b, e);
            }
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (t, min_grain);
        f(0, n);
    }
}

/// 单元本身就是重活的 fork-join（整通道平面、整输出行、GEMM 面板）。
///
/// 设计里这是 `parallel_for_units`：默认 256 的 grain 是按**元素数**标定的，
/// 会把 v6 的 N*C（16..320）判成「不值得并行」，让 pool2d/resize/GAP
/// 单核跑完全场。
pub fn parallel_for_units<F>(n: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    if n < 2 {
        if n == 1 {
            f(0, 1);
        }
        return;
    }
    parallel_for(n, 2, f);
}

/// 与 `parallel_for_elems` 对应：`units` 是工作单元数（决定切分），
/// `elems` 是真实搬运量（决定**要不要** fork）。
///
/// 两者必须分开：广播路径的单元数远小于元素数（batchnorm 的单元是通道平面，
/// reduce_mean 的单元是输出元素）；按单元数门控会把 `[N,T,C] += [C]`
/// 这种 240 单元 / 1.66M 元素的节点按回单线程。
pub fn parallel_for_elems<F>(units: usize, elems: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    if pool_threads() == 1 || !worth_forking_bytes(elems.saturating_mul(4)) {
        f(0, units);
        return;
    }
    parallel_for(units, 2, f);
}

/// 裸指针的 `Send + Sync` 包装。
///
/// 并行内核写的是「不相交但无法用 `split_at_mut` 表达」的区间（同一些行的
/// 不同列段、按行切块的输出）。安全性**不**来自这个类型——它来自每个调用点
/// 的区间不相交论证（见各内核的 SAFETY 注释）。这个类型只是让裸指针能过
/// `fork_join` 闭包的线程安全检查。
///
/// SAFETY（使用方责任）：通过 [`SyncPtr::get`] 取得的指针，其解引用与写入
/// 必须与所有并发访问者不相交。
pub struct SyncPtr<T>(*mut T);

// SAFETY: SyncPtr 只是裸指针的标记包装。线程安全**不**来自这里——来自
// 每个使用点论证其区间与其他并发访问者不相交（见各内核的 SAFETY 注释）。
unsafe impl<T> Send for SyncPtr<T> {}
// SAFETY: 同上；共享的是指针值，不是它指向的数据。
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> Clone for SyncPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    /// 包装一个裸指针。
    pub fn new(p: *mut T) -> Self {
        SyncPtr(p)
    }
    /// 取回裸指针。
    pub fn get(self) -> *mut T {
        self.0
    }
    /// 指针偏移（不叫 `add` 是为了避免与 `std::ops::Add::add` 混淆）。
    pub fn offset(self, off: usize) -> Self
    where
        T: Sized,
    {
        // SAFETY: 指针算术；偏移合法性由使用方保证（原指针指向分配区内）。
        unsafe { SyncPtr::new(self.0.add(off)) }
    }
    /// 从指针与长度构造可变切片。
    ///
    /// # Safety
    ///
    /// 区间必须与所有并发访问者不相交，且指针指向有效分配。
    pub unsafe fn slice(self, len: usize) -> &'static mut [T] {
        // SAFETY: 使用方已论证（见各内核的 SAFETY 注释）。
        unsafe { std::slice::from_raw_parts_mut(self.0, len) }
    }
}
