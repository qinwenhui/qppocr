//! sgemm 与 im2col。
//!
//! C[M,N] = A[M,K] * B[K,N]，行主序。并行单元：N 宽时按 N 面板（32 列）切，
//! 否则按 M 行切。微内核：4 行 × 列向量组、k 内层 broadcast+FMA（向量宽度按后端）
//! （每步 16 路独立 FMA）。
//!
//! # 位级细节
//!
//! 1. **没有清 C 步，也不需要**：每条路径都在寄存器里算满输出再 store，
//!    没有任何路径先读 C（大输出的逐节点 memset 是纯浪费）。
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
///   erf 的除法和常数拖进占满向量寄存器的循环，挤占 GEMM 累加器）。
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
/// `QPPOCR_GEMM_PANELS_PER_CHUNK`（≥2 生效；OnceLock 缓存——进程内不重复
/// 查环境，CLI 的 A/B 靠独立进程切换）。
fn panel_chunk_env() -> Option<usize> {
    static ENV: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| {
        std::env::var("QPPOCR_GEMM_PANELS_PER_CHUNK")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v >= 2)
    })
}

/// A 侧分块的块字节预算（`QPPOCR_GEMM_MB`，0 = 关）。默认 512 KB。
///
/// **为什么分块**：面板内核每面板全量重读 A（`sgemm_panel` 的 `for p` 外层
/// 对每个 32 列面板走一遍全部 m 行）。A 不超过 L2 时这是免费的；A 一旦
/// 超过（rec CSP 的 1×1 权重 2-9 MB、NHWC 的激活 4.5 MB），重读落 DRAM。
/// gemm_shape_probe 的单线程形状律（2026-11-09，逐形状线性回归）：
/// **耗时 ≈ A 重流量 / ~5.2 GB/s**——`1024×2184×512`（A=2 MB、69 面板
/// → 138 MB 重流）26.8 ms；A 减半耗时减半；`1536³`（A=9.4 MB、48 面板
/// → 452 MB）146.9 ms。单线程算术峰值的 60% 都不到，瓶颈全在 A 重流。
///
/// 分块后 M 切成 `ceil(m·k·4 / 预算)` 行块，A 块 ≤ 预算（L2 常驻），面板
/// 循环搬到块内——A 每块只冷读一次，重流量从 `np × A` 降到 `nmb × A`
/// （nmb = 块数，典型 4-8 vs np = N/32，典型 32-585）。B 侧 slab 反过来
/// 被块数遍重读，但 B ≤ 4.5 MB 落 L3，代价低一个量级。
///
/// 位级：每个输出元素的 k 升序 FMA 链与所属 (行块, 面板) 无关——分块只改
/// 循环次序，不改任何元素的累加序列 → 与未分块逐位一致（同 ppc 判例）。
fn mb_budget() -> usize {
    static ENV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| {
        std::env::var("QPPOCR_GEMM_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(512 * 1024)
    })
}

/// B 侧补齐开关（`QPPOCR_GEMM_BPAD`，=1 开，默认关）。见 `sgemm_impl`
/// 内注释：动机是 load8 的跨线罚金，实测判定默认关。
fn bpad_on() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| {
        std::env::var("QPPOCR_GEMM_BPAD")
            .ok()
            .as_deref()
            .map(|v| v.trim() == "1")
            .unwrap_or(false)
    })
}

#[allow(clippy::too_many_arguments)] // 与公开 sgemm 系同款形状参数面
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

    // B 侧补齐（QPPOCR_GEMM_BPAD=1；np≥2 且行距非 64B 倍数才有意义）：
    // N%16≠0 时稠密 B 的行距 n·4 破坏 64B 对齐，面板内核每 k 步的 load8
    // 跨缓存线（bpack_probe 三臂：N=975 档 pad16 +18-23%，引擎真实 N 档
    // +0-16%，本机分钟级噪声大——引擎级同窗 A/B 是最终裁判）。做法：把 B
    // 拷进 64B 对齐、行距 align32(n) 的池化 scratch——所有 load 回到线内
    // 且不越界（面板读取最多触达 n0+32 = align32(n)）。位级：死列读到的
    // 零只进死累加器 lane，store 侧 cnt 掩码不动 → 逐位一致。np=1 的
    // mrows 路径不补：2026-11-04 的 np=1 右填充实验已证持平（「缓存冷墙」
    // 效应），此处沿用该判决。
    let mut bpad_buf: Option<crate::buf::F32Buf> = None;
    let nb = if bpad_on() && np >= 2 && !n.is_multiple_of(16) {
        let nb = (n + 31) & !31;
        // 池化 scratch：+16 浮点余量做 64B 基址取整；with_zeroed 写满全部
        // 元素（死列清零），下方只覆盖真实列 → 无未初始化读。
        let buf = crate::buf::F32Buf::with_zeroed(k * nb + 16);
        let base = (buf.as_slice().as_ptr() as usize + 63) & !63;
        let dst = base as *mut f32;
        let src = b.as_ptr();
        for kk in 0..k {
            // SAFETY: 目标行 [kk·nb, kk·nb+n) 落在 buf 的 k·nb 区间内且行间
            // 不相交；源行 [kk·n, kk·n+n) 由上方 b.len() ≥ k·n 断言覆盖。
            unsafe { std::ptr::copy_nonoverlapping(src.add(kk * n), dst.add(kk * nb), n) };
        }
        bpad_buf = Some(buf);
        nb
    } else {
        n
    };
    let b_eff = bpad_buf
        .as_ref()
        .map(|buf| ((buf.as_slice().as_ptr() as usize + 63) & !63) as *const f32)
        .unwrap_or_else(|| b.as_ptr());

    // 小 GEMM 留在单线程——但仍然走同一内核。 这里曾退化为标量三重循环，
    // 让角度分类器和 FPN neck 里的每个小 conv 比算术该有的慢几个数量级。
    if serial || flops < par::thresholds().gemm_par_min || par::threads() == 1 {
        // A 侧分块（A 超预算才分；见 mb_budget 的测量律说明）。
        // mb 向下取整到 4 的倍数：块边界若落在 4 行块内部，会把这些行
        // 推进「M%4 尾行」分支（bias 起种，位级与整块不同）。
        let budget = mb_budget();
        let mb = if m.saturating_mul(k) * 4 > budget {
            ((budget / (k * 4)).max(4) / 4) * 4
        } else {
            m
        };
        let bp = bptr(bias);
        // SAFETY: 串行调用，无并发访问；形状已在上面的 assert 校验。
        unsafe {
            for m0 in (0..m).step_by(mb) {
                let mh = (m - m0).min(mb);
                crate::arch::sgemm_panel(
                    a.as_ptr().add(m0 * k),
                    b_eff,
                    c.as_mut_ptr().add(m0 * ldc),
                    mh,
                    n,
                    nb,
                    k,
                    ldc,
                    if bp.is_null() { bp } else { bp.add(m0) },
                    0,
                    np,
                )
            }
        };
        finish_res(c, residual, m, n, ldc, act);
        return;
    }

    // 切分轴。规则**只看形状，不看线程数**——这是刻意的，也是本条最重要的
    // 不变量（见下）。
    //
    // 旧规则是 `np >= par::threads()`，它把线程数混进了**算法路径的选择**，
    // 代价是灾难性的：`sgemm_panel` 是 4 行 × 列向量组的 4×4 微内核，
    // 而 `sgemm_mrows` 逐行广播标量（每次 k 只喂 4 列），同一个形状慢
    // 2.6~6.4×。于是线程数越大，越多窄 N 的 GEMM 掉进 M 行路径。
    //
    // 实测（本机 M4、大杂图 medium、869 行 rec；把所有 fork 降级为就地
    // 串行执行以剔除调度干扰，于是**只剩 `threads()` 在变**）：
    //   threads=2  rec 16.2 s（`np>=2` ⇒ 只有 n≤32 走 M 行）
    //   threads=10 rec 164.3 s（`np>=10` ⇒ n≤320 全走 M 行）
    // 端到端 95.3 s → 31.7 s（3.0×），框数与文本逐字不变。
    //
    // 逐形状代价表（threads=10、同一形状两臂各测一遍，单位 ms/次）：
    //   m=512 n=132 k=1024  面板 4.60  M 行 14.32  （3.1×）
    //   m=512 n=156 k=1024  面板 4.56  M 行 24.74  （5.4×）
    //   m=512 n=222 k=1024  面板 6.86  M 行 43.57  （6.4×）
    // rec 全部 984 个 (形状,路径) 组合**无一例外**面板更快；全场合计
    // M 行 316.5 s vs 面板 82.4 s。
    //
    // 取 `np >= 2`（n ≥ 33）而不是「永远面板」：`np == 1` 时面板轴只有
    // 一个单元、**根本无法并行**，而 M 行路径至少能用满 m 路并行度。
    // 实测 det 上这一档值 ~1.1 s（7620→8760 ms），而 rec 侧 np=2 与
    // np=1 门槛无差别（22.4 vs 22.8 s）——白拿的 det 余量。
    //
    // ★ 副作用是**修复**而非破坏：旧规则下 T=3 及以上的输出路径与 T=1/T=2
    //   就已经不同了。新规则对 T ≤ 2 是逐位等同（两时都算 `np >= 2`），
    //   因此 T≥3 的数值被拉回 T=2 的路径，而不是反过来。
    //
    // ⚠ 已知反例，**未收口，也不属这次改动**：det 的 768→384 k1×1 @23x31
    //   （M=384 N=713 K=768、np=23）单线程 12.7 ms、四线程 148 ms——面板路径
    //   在**这个**形状上并行是负收益。旧规则 `23 >= threads()` 对 threads ≤ 23
    //   恒真、新规则 `23 >= 2` 亦真，两时都走面板，故非本次引入；端到端 det
    //   实测无退化（这形状被 det 其它大头盖住了）。要治它得按形状挑切分轴，
    //   而不是按路径。
    const PANEL_MIN_NP: usize = 2;
    if np >= PANEL_MIN_NP {
        let cp = par::SyncPtr::new(c.as_mut_ptr());
        // 面板大块化实验（2026-11-04，QPPOCR_GEMM_PANELS_PER_CHUNK）：
        // 默认动态领票把 np 切成 ~threads×8 份，每 worker 每次调用重流
        // A 侧 ~6-8 遍；4 shard 并发跑不同层时 A 从 DRAM 反复重流（
        // 引擎内 ~6-8 GB/s = DRAM 饱和，det conv 对照 8.8）。每份多拿
        // 几个连续面板 = A 只扫 1-2 遍，代价是 per-worker 工作集变大、
        // 负载均衡变粗。份数下限 = 本池参与者数（不闲置 worker）。
        // 切分只改「谁跑哪段列区间」，每元素的 k 序不变 → 逐位相同。
        let ppc = panel_chunk_env();
        let nchunk = match ppc {
            Some(ppc) => np.div_ceil(ppc).max(par::pool_threads()).min(np),
            None => np.min(par::pool_threads().saturating_mul(8)),
        };
        // A 侧分块（同串行分支的 mb_budget；A 超预算才分）。分块后并行
        // 单元 = 行块 × 面板段，票号 **mb 主序**（低维是面板段）：worker
        // 顺序领票会在同一行块上连扫多个面板段，A 块在 L2 里被本 worker
        // 反复命中；换块才冷读一次。B 侧 slab 每块重读一遍，落 L3。
        // mb 同样取整到 4 的倍数（同串行分支的位级理由）。
        let budget = mb_budget();
        let mb = if m.saturating_mul(k) * 4 > budget {
            ((budget / (k * 4)).max(4) / 4) * 4
        } else {
            m
        };
        let nmb = m.div_ceil(mb);
        // b_eff（B 或补齐 scratch）进闭包要过 Send：包一层 SyncPtr。
        let bpp = par::SyncPtr::new(b_eff as *mut f32);
        if nmb <= 1 {
            par::parallel_for_chunks(np, nchunk, |pb, pe| {
                // SAFETY: 各面板写不相交的列区间 [p*32, p*32+nn)，元素两两不重叠。
                unsafe {
                    crate::arch::sgemm_panel(
                        a.as_ptr(),
                        bpp.get(),
                        cp.get(),
                        m,
                        n,
                        nb,
                        k,
                        ldc,
                        bptr(bias),
                        pb,
                        pe,
                    )
                };
            });
        } else {
            // 面板段的切法与 parallel_for_chunks(np, nchunk) 完全一致
            //（chunk 步长 + min 截断），只是套进 (mb, pc) 二维票里。
            let pstep = np.div_ceil(nchunk);
            let units = nmb * nchunk;
            let uchunk = units.min(par::pool_threads().saturating_mul(8)).max(1);
            let bp = bptr(bias);
            let ap = par::SyncPtr::new(a.as_ptr() as *mut f32);
            let bpz = par::SyncPtr::new(bp as *mut f32);
            par::parallel_for_chunks(units, uchunk, |u0, u1| {
                for u in u0..u1 {
                    let (mbi, pc) = (u / nchunk, u % nchunk);
                    let m0 = mbi * mb;
                    let mh = (m - m0).min(mb);
                    let pb = pc * pstep;
                    let pe = (pb + pstep).min(np);
                    if pb >= pe {
                        continue;
                    }
                    let bpo = bpz.get();
                    // SAFETY: 行块 [m0, m0+mh) × 列区间 [pb*32, pe*32) 与其他
                    // 调用者两两不相交（mbi 与 pc 划分笛卡尔积）。
                    unsafe {
                        crate::arch::sgemm_panel(
                            (ap.get() as *const f32).add(m0 * k),
                            bpp.get(),
                            cp.get().add(m0 * ldc),
                            mh,
                            n,
                            nb,
                            k,
                            ldc,
                            if bpo.is_null() { bpo } else { bpo.add(m0) },
                            pb,
                            pe,
                        )
                    };
                }
            });
        }
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
/// 段错误——间歇性地。`nn` 是循环不变量，分支完美预测。
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
    nb: usize,
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
                        acc = arow[kk].mul_add(b[kk * nb + n0 + j], acc);
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
                        acc = arow[kk].mul_add(b[kk * nb + n0 + j], acc);
                    }
                    if let Some(bs) = bias {
                        acc += bs[row];
                    }
                    acc
                } else {
                    let mut s = bv;
                    for kk in 0..k {
                        s = arow[kk].mul_add(b[kk * nb + n0 + j], s);
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
    nb: usize,
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
                    b.add(kk * nb + n0 + j).read()
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
