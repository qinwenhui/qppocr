//! sgemm 与 im2col。
//!
//! C[M,N] = A[M,K] * B[K,N]，行主序。并行单元：N 宽时按 N 面板（32 列）切，
//! 否则按 M 行切。微内核：4 行 × 4 个 AVX 向量、k 内层 broadcast+FMA
//! （每步 16 路独立 FMA）。
//!
//! # 位级细节（★ 不要"修"，）
//!
//! 1. **没有清 C 步，也不需要**：每条路径都在寄存器里算满输出再 store，
//!    没有任何路径先读 C。 曾经 memset C（det 的 1x1 conv 输出 47 MB，
//!    ~4 ms/节点的无效零填充），已删。
//! 2. **bias 折进 store 阶段**，省掉最常见的一次 Add 的读改写。
//! 3. **bias 的加法位置按路径不同**——浮点加法不可结合，
//!    两种顺序低位不同；「中间张量逐位一致」的判据依赖这套分支结构：
//!    - 4 行块、整面板（`nn == 32`）尾行、窄 N 路径 32 列主体：FMA 链从 0
//!      累加，**最后**加 bias；
//!    - 非整面板尾行、窄 N 路径 N%32 尾列：以 bias **起种**再走 FMA 链。
//!
//! # 并行与裸指针
//!
//! 面板切分写的是同一些行的不相交列区间，M 行切分写的是不相交行区间——
//! 两者都无法用 `split_at_mut` 表达成多个 `&mut`，这正是内核层用裸指针的
//! 原因。每个输出元素恰好由一个并行块写一次，
//! 区间两两不相交，无数据竞争。

use crate::activation::{Activation, apply_act};
use crate::par;

/// C[M,N] = A[M,K] * B[K,N]，行主序。
///
/// - `ldc`：C 的行距（列块场景下 C 是更宽行的列块）；通常传 `n`。
/// - `bias`：可选，每行一个值，折进 store 阶段。
/// - `act`：可选的激活，作用在**已写出的输出**上（不是累加器上——那会把
///   erf 的除法和常数拖进占满 16 个 YMM 的循环，det_small 实测慢 5.3%）。
///
/// 输出元素的累加顺序固定（k 升序），与并行切分无关。
#[allow(clippy::too_many_arguments)]
pub fn sgemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    bias: Option<&[f32]>,
    act: &Activation,
) {
    sgemm_impl(a, b, c, m, n, k, ldc, bias, act, false, None);
}

/// sgemm 的残差融合版（Conv + ResidualAdd）：见 `finish_res`。
#[allow(clippy::too_many_arguments)]
pub fn sgemm_res(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    bias: Option<&[f32]>,
    act: &Activation,
    residual: Option<&[f32]>,
) {
    sgemm_impl(a, b, c, m, n, k, ldc, bias, act, false, residual);
}

/// implicit-GEMM 的串行入口：`B` 不是稠密矩阵，而是**每个 k 一行一个指针**
/// （行内步长 `sw`，1 = 连续、2 = 隔一个取一个）。于是卷积不再需要先把
/// patch 矩阵 materialize 出来——那是输入的 kh·kw 倍。
///
/// 与 [`sgemm_serial`] 走**同一条面板路径、同一个 k 序**（ch → ky → kx），
/// 输出逐位一致。
#[allow(clippy::too_many_arguments)]
pub fn sgemm_bptrs_serial(
    a: &[f32],
    bptrs: &[*const f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    bias: Option<&[f32]>,
    act: &Activation,
    sw: usize,
) {
    assert_eq!(bptrs.len(), k, "sgemm_bptrs: 指针数应等于 k");
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    assert!(ldc >= n, "sgemm_bptrs: ldc < n");
    assert!(a.len() >= m * k, "sgemm_bptrs: A too small");
    assert!(c.len() >= (m - 1) * ldc + n, "sgemm_bptrs: C too small");
    let np = n.div_ceil(32);
    let bp = bias.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    // SAFETY: 形状已断言；面板区间 [p*32, p*32+nn) 两两不相交；B 的指针由
    // 调用方保证指向可读的 slab（含右侧 slack）。
    unsafe {
        if sw == 2 {
            crate::arch::sgemm_panel_bptrs::<2>(
                a.as_ptr(),
                bptrs.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                ldc,
                bp,
                0,
                np,
            );
        } else {
            crate::arch::sgemm_panel_bptrs::<1>(
                a.as_ptr(),
                bptrs.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                ldc,
                bp,
                0,
                np,
            );
        }
    }
    finish(c, m, n, ldc, act);
}

/// 从并行区里调用的 sgemm：**固定走串行面板路径**。
///
/// 设计里这是 `serial=true`，有两重作用：一是嵌套 `parallel_for` 在
/// fork-join 池里是死锁（[`crate::pool`] 与 同款限制）；二是面板分支
/// 与 M 行分支对同一元素的 bias 加法位置不同（见本文件顶部的位级细节），
/// **路径选择本身进位**。并行区内的调用必须走同一条分支，输出才能
/// 逐位一致。
#[allow(clippy::too_many_arguments)]
pub fn sgemm_serial(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    bias: Option<&[f32]>,
    act: &Activation,
) {
    sgemm_impl(a, b, c, m, n, k, ldc, bias, act, true, None);
}

/// [`sgemm_serial`] 的残差融合版（并行区里逐行调用，保持串行面板路径）。
#[allow(clippy::too_many_arguments)]
pub fn sgemm_serial_res(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    bias: Option<&[f32]>,
    act: &Activation,
    residual: Option<&[f32]>,
) {
    sgemm_impl(a, b, c, m, n, k, ldc, bias, act, true, residual);
}

#[allow(clippy::too_many_arguments)]
fn sgemm_impl(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    bias: Option<&[f32]>,
    act: &Activation,
    serial: bool,
    residual: Option<&[f32]>,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    assert!(ldc >= n, "sgemm: ldc < n");
    assert!(a.len() >= m * k, "sgemm: A too small");
    assert!(b.len() >= k * n, "sgemm: B too small");
    assert!(c.len() >= (m - 1) * ldc + n, "sgemm: C too small");

    let flops = (m * n * k) as f64;
    let np = n.div_ceil(32); // n panels
    if let Some(b) = bias {
        assert!(b.len() >= m, "sgemm: bias too small");
    }
    let bptr = |bias: Option<&[f32]>| bias.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());

    // 小 GEMM 留在单线程——但仍然走同一内核。 这里曾退化为标量三重循环，
    // 让角度分类器和 FPN neck 里的每个小 conv 比算术该有的慢几个数量级。
    if serial || flops < par::thresholds().gemm_par_min || par::threads() == 1 {
        // SAFETY: 串行调用，无并发访问；形状已在上面的 assert 校验。
        unsafe {
            crate::arch::sgemm_panel(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                ldc,
                bptr(bias),
                0,
                np,
            )
        };
        finish_res(c, residual, m, n, ldc, act);
        return;
    }

    // 切分轴。自动规则（「N 面板够多就用面板」）对至少一个真实形状是错的：
    // det 的 768->384 k1x1 @23x31 是 M=384 N=713 K=768、NP=23，单线程 12.7 ms、
    // 两线程 167 ms、四线程 148 ms——并行比串行慢最多 13 倍；而它的镜像
    // （384->768，同样 FLOPs、同样面板数）正常扩展。
    if np >= par::threads() {
        let cp = par::SyncPtr::new(c.as_mut_ptr());
        par::parallel_for(np, 1, |pb, pe| {
            // SAFETY: 各面板写不相交的列区间 [p*32, p*32+nn)，元素两两不重叠。
            unsafe {
                crate::arch::sgemm_panel(
                    a.as_ptr(),
                    b.as_ptr(),
                    cp.get(),
                    m,
                    n,
                    k,
                    ldc,
                    bptr(bias),
                    pb,
                    pe,
                )
            };
        });
    } else {
        // 窄 N：按 M 行并行，K 上外积（B 整体流过一遍）
        let cp = par::SyncPtr::new(c.as_mut_ptr());
        par::parallel_for(m, 1, |mb, me| {
            // SAFETY: 各块写不相交的行区间 [mb, me)。
            unsafe {
                crate::arch::sgemm_mrows(
                    a.as_ptr(),
                    b.as_ptr(),
                    cp.get(),
                    m,
                    n,
                    k,
                    ldc,
                    bptr(bias),
                    mb,
                    me,
                )
            };
        });
    }
    finish_res(c, residual, m, n, ldc, act);
}

/// 激活收尾：作用在已写出的输出上，读的是刚写的数据（cache 命中）。
/// det 的 1x1 conv 一次调用 47 MB 输出，串行扫比把激活留成独立节点还慢，
/// 所以这里自己 fork。
fn finish(c: &mut [f32], m: usize, n: usize, ldc: usize, act: &Activation) {
    finish_res(c, None, m, n, ldc, act);
}

/// finish 的残差融合版：先施加激活、再加 residual（与被融合掉的独立 Add
/// 节点同值同序——Add 读的就是激活后的值）。读的是刚写出的热数据，省掉
/// 独立 Add 的整趟冷读冷写与一个中间张量。residual 与 c 同形同布局。
fn finish_res(
    c: &mut [f32],
    residual: Option<&[f32]>,
    m: usize,
    n: usize,
    ldc: usize,
    act: &Activation,
) {
    if !act.is_on() && residual.is_none() {
        return;
    }
    let cp = par::SyncPtr::new(c.as_mut_ptr());
    let rp = residual.map(|r| par::SyncPtr::new(r.as_ptr() as *mut f32));
    par::parallel_for_elems(m, m * n, |b, e| {
        for row in b..e {
            // SAFETY: 行区间 [b, e) 两两不相交，每块只动自己的行。
            let row_slice = unsafe { cp.offset(row * ldc).slice(n) };
            if act.is_on() {
                apply_act(row_slice, act);
            }
            if let Some(r) = &rp {
                // SAFETY: residual 与 c 同形同布局，行区间本块独占。
                let rr = unsafe { r.offset(row * ldc).slice(n) };
                for (d, sv) in row_slice.iter_mut().zip(rr) {
                    *d += *sv;
                }
            }
        }
    });
}

/// 面板体（标量判据）：`[pb, pe)` 号 N 面板，每面板 32 列。
///
/// SIMD 后端（AVX2/NEON）与它逐位对拍；`arch` 分发层的标量分支也走它。
///
/// # Safety
///
/// `c` 的列区间 `[p*32, p*32+nn)`（所有行）必须与并发调用者不相交。
///
/// 尾部面板只算它实际拥有的列：在 B 的行尾读满 32 个 float 会越过页边界
/// 段错误——间歇性地（原注释）。`nn` 是循环不变量，分支完美预测。
///
/// SAFETY: `c` 的列区间 `[p*32, p*32+nn)`（对所有行）必须与并发调用者
/// 的区间不相交。
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn panel_body(
    a: &[f32],
    b: &[f32],
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    bias: Option<&[f32]>,
    pb: usize,
    pe: usize,
) {
    for p in pb..pe {
        let n0 = p * 32;
        let nn = 32.min(n - n0);
        let full = nn == 32;
        // 4 行块：FMA 链从 0，bias 后置
        let mut m0 = 0;
        while m0 + 4 <= m {
            for i in 0..4 {
                let row = m0 + i;
                let arow = &a[row * k..];
                for j in 0..nn {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        // acc = fma(a, b, acc)：与 AVX2 的 FMA 链一致
                        acc = arow[kk].mul_add(b[kk * n + n0 + j], acc);
                    }
                    if let Some(bs) = bias {
                        acc += bs[row];
                    }
                    // SAFETY: 函数级契约（见签名）：本面板的列区间不相交。
                    unsafe { *c.add(row * ldc + n0 + j) = acc };
                }
            }
            m0 += 4;
        }
        // 尾行（M%4 余数）：
        // - 整面板：1-row 内核语义——FMA 链从 0，bias 后置；
        // - 非整面板： 标量尾巴以 bias 起种（位级差异，原样保留）。
        for row in m0..m {
            let arow = &a[row * k..];
            let bv = bias.map(|bs| bs[row]).unwrap_or(0.0f32);
            for j in 0..nn {
                let v = if full {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        acc = arow[kk].mul_add(b[kk * n + n0 + j], acc);
                    }
                    if let Some(bs) = bias {
                        acc += bs[row];
                    }
                    acc
                } else {
                    let mut s = bv;
                    for kk in 0..k {
                        s = arow[kk].mul_add(b[kk * n + n0 + j], s);
                    }
                    s
                };
                // SAFETY: 同上（函数级契约）。
                unsafe { *c.add(row * ldc + n0 + j) = v };
            }
        }
    }
}

/// 窄 N 路径（标量判据）：整行计算，K 上外积。32 列主体 bias 后置，
/// N%32 尾列 bias 起种。
///
/// # Safety
///
/// 行区间 `[mb, me)` 必须与并发调用者不相交。
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn m_rows_body(
    a: &[f32],
    b: &[f32],
    c: *mut f32,
    _m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    bias: Option<&[f32]>,
    mb: usize,
    me: usize,
) {
    let n32 = n - n % 32;
    for row in mb..me {
        let arow = &a[row * k..];
        let bv = bias.map(|bs| bs[row]).unwrap_or(0.0f32);
        for j in 0..n32 {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc = arow[kk].mul_add(b[kk * n + j], acc);
            }
            if let Some(bs) = bias {
                acc += bs[row];
            }
            // SAFETY: 函数级契约：本块的行区间不相交。
            unsafe { *c.add(row * ldc + j) = acc };
        }
        for j in n32..n {
            let mut s = bv;
            for kk in 0..k {
                s = arow[kk].mul_add(b[kk * n + j], s);
            }
            // SAFETY: 同上。
            unsafe { *c.add(row * ldc + j) = s };
        }
    }
}

/// 非整面板的 M%4 尾行回退（标量，bias 起种）：SIMD 后端共用的尾巴。
/// 纯标量 `mul_add` 链，无架构内联函数，各后端逐位同值。
///
/// # Safety
///
/// 面板 `p` 的列区间 `[p*32, p*32+nn)`（行 `[row_start, m)`）与并发调用者
/// 不相交；`bptrs`（BP 时）指向 k 个可读行。
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_op_in_unsafe_fn)] // 裸指针循环：前置条件见 # Safety 段
pub(crate) unsafe fn panel_tail_scalar<const BP: bool, const SW: usize>(
    a: *const f32,
    b: *const f32,
    bptrs: *const *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    bias: *const f32,
    p: usize,
    row_start: usize,
) {
    let n0 = p * 32;
    let nn = 32.min(n - n0);
    let has_bias = !bias.is_null();
    for row in row_start..m {
        let ar = a.add(row * k);
        let bv = if has_bias { *bias.add(row) } else { 0.0 };
        let cp = c.add(row * ldc + n0);
        for j in 0..nn {
            // bias 起种 + fma 链（与 panel_body 的非整面板分支逐位相同）
            let mut s = bv;
            for kk in 0..k {
                let bv1 = if BP {
                    (*bptrs.add(kk)).add((n0 + j) * SW).read()
                } else {
                    b.add(kk * n + n0 + j).read()
                };
                s = ar.add(kk).read().mul_add(bv1, s);
            }
            cp.add(j).write(s);
        }
    }
}

/// implicit-GEMM 面板体的标量判据：B 每 k 一行一个指针、行内步长 `SW`。
/// 每元素的运算串与 [`panel_body`] 相同（k 升序 FMA 链、bias 位置按分支），
/// 只是 B 的取数经指针间接。
///
/// # Safety
///
/// 同 [`panel_body`]；`bptrs` 指向 k 个可读行（含右侧按 SW 计的触达范围）。
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_op_in_unsafe_fn)] // 裸指针循环：前置条件见 # Safety 段
pub(crate) unsafe fn sgemm_panel_bptrs_scalar<const SW: usize>(
    a: *const f32,
    bptrs: *const *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    bias: *const f32,
    pb: usize,
    pe: usize,
) {
    let has_bias = !bias.is_null();
    for p in pb..pe {
        let n0 = p * 32;
        let nn = 32.min(n - n0);
        let full = nn == 32;
        let mut m0 = 0;
        while m0 + 4 <= m {
            for i in 0..4 {
                let row = m0 + i;
                let ar = a.add(row * k);
                for j in 0..nn {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        let bv = (*bptrs.add(kk)).add((n0 + j) * SW).read();
                        acc = ar.add(kk).read().mul_add(bv, acc);
                    }
                    if has_bias {
                        acc += *bias.add(row);
                    }
                    c.add(row * ldc + n0 + j).write(acc);
                }
            }
            m0 += 4;
        }
        for row in m0..m {
            let ar = a.add(row * k);
            let bv = if has_bias { *bias.add(row) } else { 0.0 };
            for j in 0..nn {
                let v = if full {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        let bvv = (*bptrs.add(kk)).add((n0 + j) * SW).read();
                        acc = ar.add(kk).read().mul_add(bvv, acc);
                    }
                    if has_bias {
                        acc += *bias.add(row);
                    }
                    acc
                } else {
                    let mut s = bv;
                    for kk in 0..k {
                        let bvv = (*bptrs.add(kk)).add((n0 + j) * SW).read();
                        s = ar.add(kk).read().mul_add(bvv, s);
                    }
                    s
                };
                c.add(row * ldc + n0 + j).write(v);
            }
        }
    }
}

/// im2col：X[N=1, C, H, W]（单批次）-> cols[K=C*kh*kw, rows*ow]。
///
/// 只为输出行 `[oy0, oy0+rows)` 建补丁矩阵，让缓冲保持 cache 大小，
/// 而不是随整张特征图扩张（v6 的 2x2 conv 在 992x752 的图上要 ~190 MB）。
///
/// `cols` 布局：`cols[(c*kh*kw + ky*kw + kx) * (rows*ow) + r*ow + ox]`。
#[allow(clippy::too_many_arguments)]
/// 全进程累计的 **im2col 搬运字节数**（写 patch + 读输入）。诊断用：
/// 这是节点级统计看不见的一块——patch 矩阵比输入大 k 倍。
pub static IM2COL_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 全进程累计的**深度卷积补零平面**字节数（写 slab + 读原图）。
pub static DW_PAD_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 把 patch 矩阵摊成 `[k][n]`（行主序）。
#[allow(clippy::too_many_arguments)]
pub fn im2col(
    x: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    ow: usize,
    oy0: usize,
    rows: usize,
    cols: &mut [f32],
) {
    let nout = rows * ow;
    assert!(cols.len() >= c * kh * kw * nout, "im2col: cols too small");
    // 写 patch（c·kh·kw·nout）+ 读输入（约一半的 patch 量，边界有零填充）
    IM2COL_BYTES.fetch_add(
        (c * kh * kw * nout * 4 + c * kh * nout * 4) as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    let colsp = par::SyncPtr::new(cols.as_mut_ptr());
    let body = |ob: usize, oe: usize| {
        for r in ob..oe {
            let oy = oy0 + r; // 全局输出行
            for ch in 0..c {
                let xc = &x[ch * h * w..];
                for ky in 0..kh {
                    let iy = oy as isize * sh as isize - ph as isize + ky as isize;
                    // (c, ky) 固定时 cols 里是一段连续 nout 中的 r*ow..r*ow+ow
                    let base = (ch * kh * kw + ky * kw) * nout + r * ow;
                    // SAFETY: 行 r 写的段 [base, base+ow) 与并行其他块的段
                    // 不相交（cols 的每段按 r 切开，互不重叠）。
                    // SAFETY: 见上方块级注释（行 r 的段两两不相交）。
                    unsafe {
                        if iy < 0 || iy as usize >= h {
                            // 整行在 padding 里：零填充。★ 每个 kx 一段
                            //（不是 `for kx: memset`）——只填 ky*kw 那一段
                            // 会给 kx>0 的段留下复用缓冲里的陈旧数据。
                            for kx in 0..kw {
                                std::ptr::write_bytes(colsp.get().add(base + kx * nout), 0, ow);
                            }
                            continue;
                        }
                        let xr = &xc[iy as usize * w..(iy as usize + 1) * w];
                        for kx in 0..kw {
                            let d0 = (ch * kh * kw + ky * kw + kx) * nout + r * ow;
                            let dst = colsp.get().add(d0);
                            // ★ 有效 ox 是**一段连续区间**：ix = ox*sw - pw + kx 落在
                            //   [0, w) 等价于 ox ∈ [ceil((pw-kx)/sw), floor((w-1+pw-kx)/sw)]。
                            //   旧写法逐元素判断越界，是纯搬运里最贵的一种写法——
                            //   实测 im2col 只有 6-10 GB/s，而它占 k3x3 卷积的 50-65%。
                            let lo = ((pw as isize - kx as isize + sw as isize - 1)
                                .div_euclid(sw as isize))
                            .max(0) as usize;
                            let hi = ((w as isize - 1 + pw as isize - kx as isize)
                                .div_euclid(sw as isize)
                                + 1)
                            .clamp(0, ow as isize) as usize;
                            let lo = lo.min(ow);
                            let hi = hi.max(lo);
                            if sw == 1 {
                                // 连续段：两端补零 + 中间一次 memcpy
                                std::ptr::write_bytes(dst, 0, lo);
                                std::ptr::copy_nonoverlapping(
                                    xr.as_ptr().add(lo + kx - pw),
                                    dst.add(lo),
                                    hi - lo,
                                );
                                std::ptr::write_bytes(dst.add(hi), 0, ow - hi);
                            } else {
                                // 带步长：仍逐元素，但没有逐元素的越界分支
                                std::ptr::write_bytes(dst, 0, lo);
                                std::ptr::write_bytes(dst.add(hi), 0, ow - hi);
                                let mut ox = lo;
                                while ox < hi {
                                    // ★ 加法在前：`ox*sw - pw + kx` 的左结合会先算
                                    //   `ox*sw - pw`（lo=0、kx>0 时 usize 下溢）。
                                    //   release 的回绕恰好绕回正确值，debug 直接
                                    //   panic——这正是 CI 三次挂掉的那个错。
                                    *dst.add(ox) = *xr.as_ptr().add(ox * sw + kx - pw);
                                    ox += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    };
    // ★ 串行执行，调用方负责并行。conv 的 tile_body 调用它时恒为串行
    // ——tile 本身已在 `parallel_for` 里，嵌套 fork 在 fork-join 池里是
    // 死锁（见 `crate::pool` 的模块注释）。
    body(0, rows);
}
