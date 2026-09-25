//! sgemm 与 im2col：设计文档 GEMM 部分。
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
//! 3. **bias 的加法位置按路径不同**—— 原样如此，浮点加法不可结合，
//!    两种顺序低位不同；阶段 2「中间张量与基准值逐位一致」依赖这套分支结构：
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
    sgemm_impl(a, b, c, m, n, k, ldc, bias, act, false);
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
    sgemm_impl(a, b, c, m, n, k, ldc, bias, act, true);
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
    #[cfg(target_arch = "x86_64")]
    let bptr = |bias: Option<&[f32]>| bias.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());

    // AVX2 与标量共用同一套分发；内核选择见 panel_body/m_rows_body 的注释。

    // 小 GEMM 留在单线程——但仍然走同一内核。 这里曾退化为标量三重循环，
    // 让角度分类器和 FPN neck 里的每个小 conv 比算术该有的慢几个数量级。
    if serial || flops < par::thresholds().gemm_par_min || par::threads() == 1 {
        // SAFETY: 串行调用，无并发访问；形状已在上面的 assert 校验。
        unsafe {
            crate::arch_dispatch!(
                crate::x86::sgemm_panel_avx2(
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
                ),
                panel_body(a, b, c.as_mut_ptr(), m, n, k, ldc, bias, 0, np)
            )
        };
        finish(c, m, n, ldc, act);
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
                crate::arch_dispatch!(
                    crate::x86::sgemm_panel_avx2(
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
                    ),
                    panel_body(a, b, cp.get(), m, n, k, ldc, bias, pb, pe)
                )
            };
        });
    } else {
        // 窄 N：按 M 行并行，K 上外积（B 整体流过一遍）
        let cp = par::SyncPtr::new(c.as_mut_ptr());
        par::parallel_for(m, 1, |mb, me| {
            // SAFETY: 各块写不相交的行区间 [mb, me)。
            unsafe {
                crate::arch_dispatch!(
                    crate::x86::sgemm_mrows_avx2(
                        a.as_ptr(),
                        b.as_ptr(),
                        cp.get(),
                        n,
                        k,
                        ldc,
                        bptr(bias),
                        mb,
                        me,
                    ),
                    m_rows_body(a, b, cp.get(), m, n, k, ldc, bias, mb, me)
                )
            };
        });
    }
    finish(c, m, n, ldc, act);
}

/// 激活收尾：作用在已写出的输出上，读的是刚写的数据（cache 命中）。
/// det 的 1x1 conv 一次调用 47 MB 输出，串行扫比把激活留成独立节点还慢，
/// 所以这里自己 fork。
fn finish(c: &mut [f32], m: usize, n: usize, ldc: usize, act: &Activation) {
    if !act.is_on() {
        return;
    }
    let cp = par::SyncPtr::new(c.as_mut_ptr());
    par::parallel_for_elems(m, m * n, |b, e| {
        for row in b..e {
            // SAFETY: 行区间 [b, e) 两两不相交，每块只动自己的行。
            let row_slice = unsafe { cp.offset(row * ldc).slice(n) };
            apply_act(row_slice, act);
        }
    });
}

/// 面板体：`[pb, pe)` 号 N 面板，每面板 32 列。
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
unsafe fn panel_body(
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

/// 窄 N 路径：整行计算，K 上外积。32 列主体 bias 后置，N%32 尾列 bias 起种。
///
/// # Safety
///
/// 行区间 `[mb, me)` 必须与并发调用者不相交。
#[allow(clippy::too_many_arguments)]
unsafe fn m_rows_body(
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

/// im2col：X[N=1, C, H, W]（单批次）-> cols[K=C*kh*kw, rows*ow]。
///
/// 只为输出行 `[oy0, oy0+rows)` 建补丁矩阵，让缓冲保持 cache 大小，
/// 而不是随整张特征图扩张（v6 的 2x2 conv 在 992x752 的图上要 ~190 MB）。
///
/// `cols` 布局：`cols[(c*kh*kw + ky*kw + kx) * (rows*ow) + r*ow + ox]`。
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
                            //（基准是 `for kx: memset`）——只填 ky*kw 那一段
                            // 会给 kx>0 的段留下复用缓冲里的陈旧数据。
                            for kx in 0..kw {
                                std::ptr::write_bytes(colsp.get().add(base + kx * nout), 0, ow);
                            }
                            continue;
                        }
                        let xr = &xc[iy as usize * w..(iy as usize + 1) * w];
                        for kx in 0..kw {
                            let d0 = (ch * kh * kw + ky * kw + kx) * nout + r * ow;
                            for ox in 0..ow {
                                let ix = ox as isize * sw as isize - pw as isize + kx as isize;
                                let v = if ix >= 0 && (ix as usize) < w {
                                    xr[ix as usize]
                                } else {
                                    0.0
                                };
                                *colsp.get().add(d0 + ox) = v;
                            }
                        }
                    }
                }
            }
        }
    };
    // ★ 串行执行，调用方负责并行。基准的 im2col 带 serial 参数，conv 的
    // tile_body 总是传 true——因为 tile 本身已在 parallel_for 里，嵌套
    // fork 在 fork-join 池里是死锁（ 池与 `crate::pool` 同款限制）。
    // 这里直接执行与 基准的实际行为一致。
    body(0, rows);
}
