//! aarch64 NEON + FMLA 微内核。
//!
//! f32 NEON 是 aarch64 的基线指令集（ARMv8-A 起强制），无需运行时探测，
//! 也无需 `target_feature` 标注；内核仍是 `unsafe fn`——前置条件（区间
//! 相交性、形状、指针长度）由 [`crate::arch`] 分发层保证。
//!
//! 数值上这些内核与 `x86` 版、标量版**逐位一致**：每元素的运算串逐条镜像
//! ——GEMM 的 k 升序 FMA 链、bias 的加法位置（4 行块/整面板尾行后置，
//! 非整面板尾行起种）、softmax 水平归约的结合树
//! `((v0+v4)+(v1+v5)) + ((v2+v6)+(v3+v7))`、erf/exp 多项式的系数与
//! 舍入形态（fused 的 vfmaq/vfmsq 对应 AVX2 的 fmadd/fnmadd）。
//! `tests/bitexact.rs` 在 aarch64 上自动选 NEON 侧逐位对拍。
//!
//! 向量宽度是 128-bit（4 lane）：列方向的分组与 AVX2 版不同（16 列一组、
//! 一面板两组），但每个元素的累加链只依赖 k 序，与列分组无关——位级
//! 结果不受影响。水平归约类（softmax 的和）严格按 8-lane 结构镜像，
//! 用两个 q 寄存器凑满一组。

#![allow(unsafe_op_in_unsafe_fn)] // 见模块注释：安全性由分发层契约承担
#![allow(clippy::approx_constant)] // 系数照抄多项式原文：位级一致的要求

use std::arch::aarch64::*;

// ================================================================ 向量数学

/// exp 的 4-lane 版：range-reduce 到 k·ln2 + r，r 上多项式，按 2^k 缩放。
///
/// 系数与运算顺序与 `x86::exp256_ps` / `activation::exp1` 相同（含 ln2
/// 两段拆分的固有误差），round 用 FRINTN（ties-to-even）。
#[inline]
pub unsafe fn expq_ps(x: float32x4_t) -> float32x4_t {
    let ln2_hi = vdupq_n_f32(0.693_147_2);
    let ln2_lo = vdupq_n_f32(-2.980_232_2e-8);
    let inv_ln2 = vdupq_n_f32(1.442_695);
    let one = vdupq_n_f32(1.0);
    let x = vminq_f32(x, vdupq_n_f32(88.0));
    let x = vmaxq_f32(x, vdupq_n_f32(-88.0));
    let kf = vrndnq_f32(vmulq_f32(x, inv_ln2));
    // fnmadd 镜像：x − kf·ln2 两段都保持融合舍入（vfmsq = a − b·c）
    let r = vfmsq_f32(x, kf, ln2_hi);
    let r = vfmsq_f32(r, kf, ln2_lo);
    let mut p = vdupq_n_f32(1.0 / 720.0);
    p = vfmaq_f32(vdupq_n_f32(1.0 / 120.0), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5), p, r);
    p = vfmaq_f32(one, p, r);
    p = vfmaq_f32(one, p, r);
    let ki = vcvtq_s32_f32(kf);
    let ki = vaddq_s32(ki, vdupq_n_s32(127));
    let ki = vshlq_n_s32::<23>(ki);
    vmulq_f32(p, vreinterpretq_f32_s32(ki))
}

/// erf 的 4-lane 版：A&S 7.1.26，branch-free。erf(-x) = -erf(x)，
/// 取 |x| 算、末尾还原符号。
#[inline]
#[allow(clippy::excessive_precision)] // 原字面值照抄
pub unsafe fn erfq_ps(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let ax = vabsq_f32(x);
    let t = vdivq_f32(one, vfmaq_f32(one, vdupq_n_f32(0.327_591_1), ax));
    let mut p = vdupq_n_f32(1.061_405_4);
    p = vfmaq_f32(vdupq_n_f32(-1.453152027), p, t);
    p = vfmaq_f32(vdupq_n_f32(1.421413741), p, t);
    p = vfmaq_f32(vdupq_n_f32(-0.284496736), p, t);
    p = vfmaq_f32(vdupq_n_f32(0.254_829_6), p, t);
    p = vmulq_f32(p, t);
    let e = expq_ps(vmulq_f32(ax, vsubq_f32(vdupq_n_f32(0.0), ax)));
    // 1 − p·e 融合（fnmadd 镜像）
    let r = vfmsq_f32(one, p, e);
    // 位运算按整型类型走（float 无 vand/vorr），经 reinterpret 往返
    let sign = vandq_s32(
        vreinterpretq_s32_f32(x),
        vdupq_n_s32((-0.0f32).to_bits() as i32),
    );
    vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(r), sign))
}

// ================================================================ sgemm

/// sgemm 面板微内核的统一实现。`BP = false` 走稠密 B（行距 `n`）；
/// `BP = true` 走 implicit-GEMM 的 B：每个 k 一行一个指针、行内步长
/// `SW`（1 = 连续、2 = 隔一个取一个）。
///
/// 结构：4 行 × 4 向量（16 列一组，一面板最多两组），k 内层
/// broadcast+FMLA。累加顺序逐 k 不变，输出与 `x86` 版 / 标量版逐位一致。
///
/// # Safety
///
/// 与 `crate::gemm` 里的标量 `panel_body` 相同：`c` 的列区间
/// `[p*32, p*32+nn)` 必须与并发调用者不相交；`a`/`b`/`bias` 长度已由
/// 分发层校验；`bptrs`（BP 时）各行的可读范围含右侧按 SW 计的触达。
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_panel_impl<const BP: bool, const SW: usize>(
    a: *const f32,
    b: *const f32,
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
    // 第 kk 个 k 行、第 n0 列起的指针。稠密时是 b + kk·n + n0；implicit 时
    // 是指针数组里的那一行 + n0·SW。
    macro_rules! brow {
        ($kk:expr, $n0:expr) => {
            if BP {
                (*bptrs.add($kk)).add($n0 * SW)
            } else {
                b.add($kk * n + $n0)
            }
        };
    }
    // 载入 4 个连续 n 的 B 值。SW=2 时读 [p, p+8) 再抽偶数位（vuzp1）。
    macro_rules! load4 {
        ($p:expr) => {
            if SW == 1 {
                vld1q_f32($p)
            } else {
                let _p = $p;
                vuzp1q_f32(vld1q_f32(_p), vld1q_f32(_p.add(4)))
            }
        };
    }
    let has_bias = !bias.is_null();
    for p in pb..pe {
        let n0 = p * 32;
        let nn = 32.min(n - n0);
        let full = nn == 32;
        let mut m0 = 0;

        // 4 行块：FMA 链从 0，bias 后置（位级规则与 x86 版一致）
        while m0 + 4 <= m {
            let a0 = a.add(m0 * k);
            let a1 = a.add((m0 + 1) * k);
            let a2 = a.add((m0 + 2) * k);
            let a3 = a.add((m0 + 3) * k);
            let mut g = 0usize; // 16 列组
            while g * 16 < nn {
                let cn = n0 + g * 16;
                let w = 16.min(nn - g * 16);
                let (v1, v2, v3) = (w > 4, w > 8, w > 12);
                let (mut c00, mut c01, mut c02, mut c03) = (
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                );
                let (mut c10, mut c11, mut c12, mut c13) = (
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                );
                let (mut c20, mut c21, mut c22, mut c23) = (
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                );
                let (mut c30, mut c31, mut c32, mut c33) = (
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                );
                for kk in 0..k {
                    let bp = brow!(kk, cn);
                    let bv0 = load4!(bp);
                    let bv1 = if v1 { load4!(bp.add(4 * SW)) } else { bv0 };
                    let bv2 = if v2 { load4!(bp.add(8 * SW)) } else { bv0 };
                    let bv3 = if v3 { load4!(bp.add(12 * SW)) } else { bv0 };
                    let mut av = vdupq_n_f32(*a0.add(kk));
                    c00 = vfmaq_f32(c00, av, bv0);
                    if v1 {
                        c01 = vfmaq_f32(c01, av, bv1);
                    }
                    if v2 {
                        c02 = vfmaq_f32(c02, av, bv2);
                    }
                    if v3 {
                        c03 = vfmaq_f32(c03, av, bv3);
                    }
                    av = vdupq_n_f32(*a1.add(kk));
                    c10 = vfmaq_f32(c10, av, bv0);
                    if v1 {
                        c11 = vfmaq_f32(c11, av, bv1);
                    }
                    if v2 {
                        c12 = vfmaq_f32(c12, av, bv2);
                    }
                    if v3 {
                        c13 = vfmaq_f32(c13, av, bv3);
                    }
                    av = vdupq_n_f32(*a2.add(kk));
                    c20 = vfmaq_f32(c20, av, bv0);
                    if v1 {
                        c21 = vfmaq_f32(c21, av, bv1);
                    }
                    if v2 {
                        c22 = vfmaq_f32(c22, av, bv2);
                    }
                    if v3 {
                        c23 = vfmaq_f32(c23, av, bv3);
                    }
                    av = vdupq_n_f32(*a3.add(kk));
                    c30 = vfmaq_f32(c30, av, bv0);
                    if v1 {
                        c31 = vfmaq_f32(c31, av, bv1);
                    }
                    if v2 {
                        c32 = vfmaq_f32(c32, av, bv2);
                    }
                    if v3 {
                        c33 = vfmaq_f32(c33, av, bv3);
                    }
                }
                // 逐行加 bias、store。经 t 暂存再拷 w 列（尾组的残列只写
                // 属于本面板的部分）。
                let rows = [
                    (m0, [c00, c01, c02, c03]),
                    (m0 + 1, [c10, c11, c12, c13]),
                    (m0 + 2, [c20, c21, c22, c23]),
                    (m0 + 3, [c30, c31, c32, c33]),
                ];
                for (row, regs) in rows {
                    let cp = c.add(row * ldc + cn);
                    let vb = vdupq_n_f32(*bias.add(row));
                    let mut t = [0f32; 16];
                    for (jj, reg) in regs.iter().enumerate() {
                        let val = if has_bias { vaddq_f32(*reg, vb) } else { *reg };
                        vst1q_f32(t.as_mut_ptr().add(jj * 4), val);
                    }
                    std::ptr::copy_nonoverlapping(t.as_ptr(), cp, w);
                }
                g += 1;
            }
            m0 += 4;
        }
        // M%4 尾行：整面板走 1 行向量内核（bias 后置）；非整面板标量
        //（bias 起种）——位级语义与 x86 版一致，标量尾巴两边共用。
        for row in m0..m {
            if !full {
                crate::gemm::panel_tail_scalar::<BP, SW>(
                    a, b, bptrs, c, m, n, k, ldc, bias, p, row,
                );
                continue;
            }
            let ar = a.add(row * k);
            let vb = vdupq_n_f32(*bias.add(row));
            let mut g = 0usize;
            while g * 16 < nn {
                let cn = n0 + g * 16;
                let w = 16.min(nn - g * 16);
                let (v1, v2, v3) = (w > 4, w > 8, w > 12);
                let (mut c0, mut c1, mut c2, mut c3) = (
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                );
                for kk in 0..k {
                    let bp = brow!(kk, cn);
                    let av = vdupq_n_f32(*ar.add(kk));
                    c0 = vfmaq_f32(c0, av, load4!(bp));
                    if v1 {
                        c1 = vfmaq_f32(c1, av, load4!(bp.add(4 * SW)));
                    }
                    if v2 {
                        c2 = vfmaq_f32(c2, av, load4!(bp.add(8 * SW)));
                    }
                    if v3 {
                        c3 = vfmaq_f32(c3, av, load4!(bp.add(12 * SW)));
                    }
                }
                let cp = c.add(row * ldc + cn);
                let mut t = [0f32; 16];
                for (jj, reg) in [c0, c1, c2, c3].iter().enumerate() {
                    let val = if has_bias { vaddq_f32(*reg, vb) } else { *reg };
                    vst1q_f32(t.as_mut_ptr().add(jj * 4), val);
                }
                std::ptr::copy_nonoverlapping(t.as_ptr(), cp, w);
                g += 1;
            }
        }
    }
}

/// 稠密 B 的面板内核。
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_panel_neon(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    bias: *const f32,
    pb: usize,
    pe: usize,
) {
    sgemm_panel_impl::<false, 1>(a, b, std::ptr::null(), c, m, n, k, ldc, bias, pb, pe)
}

/// implicit-GEMM 的面板内核：B 每 k 一行一个指针，行内步长 `SW`。
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_panel_bptrs_neon<const SW: usize>(
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
    sgemm_panel_impl::<true, SW>(a, std::ptr::null(), bptrs, c, m, n, k, ldc, bias, pb, pe)
}

/// 窄 N 路径：整行计算，K 上外积。32 列主体 bias 后置，N%32 尾列 bias 起种。
///
/// # Safety
///
/// 行区间 `[mb, me)` 必须与并发调用者不相交。
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_mrows_neon(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    n: usize,
    k: usize,
    ldc: usize,
    bias: *const f32,
    mb: usize,
    me: usize,
) {
    let has_bias = !bias.is_null();
    let n32 = n - n % 32;
    for row in mb..me {
        let ar = a.add(row * k);
        let cp = c.add(row * ldc);
        let bv = if has_bias { *bias.add(row) } else { 0.0 };
        let vbv = vdupq_n_f32(bv);
        let mut j = 0usize;
        // 32 列主体（n32 是 32 的倍数，4 向量整除，无残列）
        while j + 4 <= n32 {
            let mut acc = vdupq_n_f32(0.0);
            for kk in 0..k {
                let av = vdupq_n_f32(*ar.add(kk));
                acc = vfmaq_f32(acc, av, vld1q_f32(b.add(kk * n + j)));
            }
            let val = if has_bias { vaddq_f32(acc, vbv) } else { acc };
            vst1q_f32(cp.add(j), val);
            j += 4;
        }
        // N%32 尾列：bias 起种的标量链
        for j in n32..n {
            let mut s = bv;
            for kk in 0..k {
                s = (*ar.add(kk)).mul_add(*b.add(kk * n + j), s);
            }
            cp.add(j).write(s);
        }
    }
}

// ================================================================ 激活（区间）

/// relu：`max(v, 0)`，就地。NaN→0（FMAX 返回非 NaN 操作数，与 MAXPS
/// 的 src2 语义在「常量在第二位」时一致）。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交。
pub unsafe fn relu_vec(t: *mut f32, b: usize, e: usize) {
    let z = vdupq_n_f32(0.0);
    let mut i = b;
    while i + 4 <= e {
        let p = t.add(i);
        vst1q_f32(p, vmaxq_f32(vld1q_f32(p), z));
        i += 4;
    }
    crate::activation::relu_seg_scalar(t, i, e);
}

/// hardsigmoid：`clip(0, 1, fma(x, a, b))`，就地。clamp 顺序与标量一致。
/// （v NaN 时 FMIN/FMAX 返回非 NaN 操作数，与标量比较链在 NaN 输入上
/// 不同——正常数据逐位一致，同 `x86::clip_vec` 的既定偏差说明。）
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交；`x`/`y` 等长。
#[allow(clippy::too_many_arguments)]
pub unsafe fn hardsigmoid_vec(
    x: *const f32,
    y: *mut f32,
    b: usize,
    e: usize,
    alpha: f32,
    beta: f32,
) {
    let va = vdupq_n_f32(alpha);
    let vb = vdupq_n_f32(beta);
    let vz = vdupq_n_f32(0.0);
    let vo = vdupq_n_f32(1.0);
    let mut i = b;
    while i + 4 <= e {
        let v = vfmaq_f32(vb, vld1q_f32(x.add(i)), va);
        vst1q_f32(y.add(i), vmaxq_f32(vz, vminq_f32(vo, v)));
        i += 4;
    }
    crate::activation::hardsigmoid_seg_scalar(x, y, i, e, alpha, beta);
}

/// sigmoid：`1 / (1 + expq(-x))`。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交；`x`/`y` 等长。
pub unsafe fn sigmoid_vec(x: *const f32, y: *mut f32, b: usize, e: usize) {
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);
    let mut i = b;
    while i + 4 <= e {
        let v = vld1q_f32(x.add(i));
        let en = expq_ps(vsubq_f32(zero, v));
        vst1q_f32(y.add(i), vdivq_f32(one, vaddq_f32(one, en)));
        i += 4;
    }
    crate::activation::sigmoid_seg_scalar(x, y, i, e);
}

/// gelu：`c3·x·(erfq(x·(1/c1)) + c2)`，就地。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交。
#[allow(clippy::too_many_arguments)]
pub unsafe fn gelu_vec(t: *mut f32, b: usize, e: usize, c1: f32, c2: f32, c3: f32) {
    let inv_c1 = 1.0f32 / c1;
    let vs = vdupq_n_f32(inv_c1);
    let vc2 = vdupq_n_f32(c2);
    let vc3 = vdupq_n_f32(c3);
    let mut i = b;
    while i + 4 <= e {
        let p = t.add(i);
        let x = vld1q_f32(p);
        let er = erfq_ps(vmulq_f32(x, vs));
        vst1q_f32(p, vmulq_f32(vmulq_f32(vc3, x), vaddq_f32(er, vc2)));
        i += 4;
    }
    crate::activation::gelu_seg_scalar(t, i, e, c1, c2, c3);
}

/// clip：`min(hi, max(lo, v))`，就地。NaN→lo（FMAX 返回非 NaN 操作数，
/// 与 MAXPS(v, vlo) 的 src2 语义一致）。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交。
pub unsafe fn clip_vec(t: *mut f32, b: usize, e: usize, lo: f32, hi: f32) {
    let vlo = vdupq_n_f32(lo);
    let vhi = vdupq_n_f32(hi);
    let mut i = b;
    while i + 4 <= e {
        let p = t.add(i);
        vst1q_f32(p, vminq_f32(vhi, vmaxq_f32(vld1q_f32(p), vlo)));
        i += 4;
    }
    crate::activation::clip_seg_scalar(t, i, e, lo, hi);
}

/// softmax 行内向量相：max/exp/归约。8 路用两个 q 寄存器凑满一组，
/// 水平归约严格镜像 `x86::hsum256_ps` 的结合树。
/// 前置条件：`inner >= 8`（由分发层保证）。
///
/// # Safety
///
/// `row` 指向 `inner` 个可读写元素，且该行与其他并行块不相交。
pub unsafe fn softmax_row_neon(row: *mut f32, inner: usize) {
    // max：向量化部分逐 lane，标量尾巴单独并入（max 顺序无关）
    let mut vmax_lo = vld1q_f32(row);
    let mut vmax_hi = vld1q_f32(row.add(4));
    let mut i = 8usize;
    while i + 8 <= inner {
        vmax_lo = vmaxq_f32(vmax_lo, vld1q_f32(row.add(i)));
        vmax_hi = vmaxq_f32(vmax_hi, vld1q_f32(row.add(i + 4)));
        i += 8;
    }
    let mut m2 = vmaxvq_f32(vmaxq_f32(vmax_lo, vmax_hi));
    while i < inner {
        if *row.add(i) > m2 {
            m2 = *row.add(i);
        }
        i += 1;
    }
    let vmx = vdupq_n_f32(m2);
    let mut vsum_lo = vdupq_n_f32(0.0);
    let mut vsum_hi = vdupq_n_f32(0.0);
    let mut i3 = 0usize;
    while i3 + 8 <= inner {
        let ev_lo = expq_ps(vsubq_f32(vld1q_f32(row.add(i3)), vmx));
        vst1q_f32(row.add(i3), ev_lo);
        let ev_hi = expq_ps(vsubq_f32(vld1q_f32(row.add(i3 + 4)), vmx));
        vst1q_f32(row.add(i3 + 4), ev_hi);
        vsum_lo = vaddq_f32(vsum_lo, ev_lo);
        vsum_hi = vaddq_f32(vsum_hi, ev_hi);
        i3 += 8;
    }
    // 水平归约：((v0+v4)+(v1+v5)) + ((v2+v6)+(v3+v7))——t[j] = v[j]+v[j+4]
    // 先做两两配对，再按树结合。
    let t = vaddq_f32(vsum_lo, vsum_hi);
    let mut ta = [0f32; 4];
    vst1q_f32(ta.as_mut_ptr(), t);
    let mut sum = (ta[0] + ta[1]) + (ta[2] + ta[3]);
    while i3 < inner {
        let ev = crate::activation::exp1(*row.add(i3) - m2);
        *row.add(i3) = ev;
        sum += ev;
        i3 += 1;
    }
    let inv = 1.0f32 / sum;
    let vinv = vdupq_n_f32(inv);
    let mut i4 = 0usize;
    while i4 + 4 <= inner {
        let p = row.add(i4);
        vst1q_f32(p, vmulq_f32(vld1q_f32(p), vinv));
        i4 += 4;
    }
    while i4 < inner {
        *row.add(i4) *= inv;
        i4 += 1;
    }
}

// ================================================================ 二元（向量）

/// 同形平坦就地：`dst[i] op= src[i]`。op 码：0..3 = + - * /。
/// 尾巴复用标量判据（pow 的 op=4 不进向量路径）。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交；`dst`/`src` 等长。
pub unsafe fn binary_flat_inplace_vec(dst: *mut f32, src: *const f32, b: usize, e: usize, op: u8) {
    let mut i = b;
    while i + 4 <= e {
        let d = vld1q_f32(dst.add(i));
        let s = vld1q_f32(src.add(i));
        let r = match op {
            0 => vaddq_f32(d, s),
            1 => vsubq_f32(d, s),
            2 => vmulq_f32(d, s),
            _ => vdivq_f32(d, s),
        };
        vst1q_f32(dst.add(i), r);
        i += 4;
    }
    crate::elementwise::binary_flat_inplace_scalar(dst, src, i, e, op);
}

/// run 路径就地：`dst[o*runlen + j] op= bsrc[ib + j]`（稠密）或 `op= cv`。
///
/// # Safety
///
/// 输出 run 与并发调用者不相交；b 稠密时 `bsrc + ib` 起有 `runlen` 个元素。
#[allow(clippy::too_many_arguments)]
pub unsafe fn binary_run_inplace_vec(
    dst: *mut f32,
    bsrc: *const f32,
    o: usize,
    runlen: usize,
    ib: usize,
    b_dense_run: bool,
    cv: f32,
    op: u8,
) {
    let vc = vdupq_n_f32(cv);
    let base = dst.add(o * runlen);
    let mut j = 0usize;
    while j + 4 <= runlen {
        let d = vld1q_f32(base.add(j));
        let r = if b_dense_run {
            let s = vld1q_f32(bsrc.add(ib + j));
            match op {
                0 => vaddq_f32(d, s),
                1 => vsubq_f32(d, s),
                2 => vmulq_f32(d, s),
                _ => vdivq_f32(d, s),
            }
        } else {
            match op {
                0 => vaddq_f32(d, vc),
                1 => vsubq_f32(d, vc),
                2 => vmulq_f32(d, vc),
                _ => vdivq_f32(d, vc),
            }
        };
        vst1q_f32(base.add(j), r);
        j += 4;
    }
    crate::elementwise::binary_run_inplace_seg(base, bsrc, j, runlen, ib, b_dense_run, cv, op);
}

/// 单元素广播：`py[i] = op(sv, v[i])` 或 `op(v[i], sv)`。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交；`v` 同长。
pub unsafe fn binary_scalar_vec(
    v: *const f32,
    sv: f32,
    a_scalar: bool,
    py: *mut f32,
    b: usize,
    e: usize,
    op: u8,
) {
    let vs = vdupq_n_f32(sv);
    let mut i = b;
    while i + 4 <= e {
        let vv = vld1q_f32(v.add(i));
        let r = match (op, a_scalar) {
            (0, true) => vaddq_f32(vs, vv),
            (0, false) => vaddq_f32(vv, vs),
            (1, true) => vsubq_f32(vs, vv),
            (1, false) => vsubq_f32(vv, vs),
            (2, _) => vmulq_f32(vv, vs),
            (_, true) => vdivq_f32(vs, vv),
            (_, false) => vdivq_f32(vv, vs),
        };
        vst1q_f32(py.add(i), r);
        i += 4;
    }
    crate::elementwise::binary_scalar_bcast_scalar(v, sv, a_scalar, py, i, e, op);
}

/// run 路径分配：`py[o*runlen + j] = op(a 值, b[j] 或 cv)`；a 恒常量。
///
/// # Safety
///
/// 输出 run 与并发调用者不相交；b 稠密时 `bsrc + ib` 起有 `runlen` 个元素。
#[allow(clippy::too_many_arguments)]
pub unsafe fn binary_run_alloc_vec(
    py: *mut f32,
    pa_const: *const f32,
    bsrc: *const f32,
    o: usize,
    runlen: usize,
    ib: usize,
    b_dense_run: bool,
    cv: f32,
    a_const: bool,
    op: u8,
) {
    let dst = py.add(o * runlen);
    let vc = vdupq_n_f32(cv);
    let dense = bsrc.add(ib);
    let mut j = 0usize;
    while j + 4 <= runlen {
        let bv = if b_dense_run {
            vld1q_f32(dense.add(j))
        } else {
            vc
        };
        let av = if a_const {
            vdupq_n_f32(*pa_const)
        } else {
            vld1q_f32(dst.add(j)) // 不该发生（a_const 必 true 才走 run alloc）
        };
        let r = match op {
            0 => vaddq_f32(av, bv),
            1 => vsubq_f32(av, bv),
            2 => vmulq_f32(av, bv),
            _ => vdivq_f32(av, bv),
        };
        vst1q_f32(dst.add(j), r);
        j += 4;
    }
    crate::elementwise::binary_run_alloc_seg(
        dst,
        pa_const,
        bsrc,
        j,
        runlen,
        ib,
        b_dense_run,
        cv,
        a_const,
        op,
    );
}

/// 同形平坦：`y[i] = a[i] op b[i]`（写新缓冲）。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交；`pa`/`pb`/`py` 等长。
pub unsafe fn binary_flat_vec(
    pa: *const f32,
    pb: *const f32,
    py: *mut f32,
    b: usize,
    e: usize,
    op: u8,
) {
    let mut i = b;
    while i + 4 <= e {
        let va = vld1q_f32(pa.add(i));
        let vb = vld1q_f32(pb.add(i));
        let r = match op {
            0 => vaddq_f32(va, vb),
            1 => vsubq_f32(va, vb),
            2 => vmulq_f32(va, vb),
            _ => vdivq_f32(va, vb),
        };
        vst1q_f32(py.add(i), r);
        i += 4;
    }
    crate::elementwise::binary_flat_scalar(pa, pb, py, i, e, op);
}

// ================================================================ conv 相关

/// 深度卷积一个 (n, channel) 输出平面，输入是已补零平面（无边界判断）。
/// 累加顺序 ky 外层、kx 内层、从 0 起，bias 最后加——与 `x86` 版/标量版
/// 逐位一致。
///
/// # Safety
///
/// `yc` 指向 `oh*ow` 个可写元素；`xp` 指向
/// `((oh-1)*sh+kh) * pwidth` 个可读元素；`wc` 指向 `kh*kw` 个可读元素。
#[allow(clippy::too_many_arguments)]
pub unsafe fn depthwise_plane_padded_neon(
    yc: *mut f32,
    xp: *const f32,
    wc: *const f32,
    oh: usize,
    ow: usize,
    pwidth: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    bias: f32,
) {
    let vbias = vdupq_n_f32(bias);
    let vec_ok = (sw == 1 || sw == 2) && ow >= 4;
    for oy in 0..oh {
        let yrow = yc.add(oy * ow);
        let base = xp.add(oy * sh * pwidth);
        if !vec_ok {
            for ox in 0..ow {
                let mut acc = 0.0f32;
                for ky in 0..kh {
                    let row = base.add(ky * pwidth);
                    for kx in 0..kw {
                        acc = (*wc.add(ky * kw + kx)).mul_add(*row.add(ox * sw + kx), acc);
                    }
                }
                *yrow.add(ox) = acc + bias;
            }
            continue;
        }
        // 一发 4 个输出。尾部用重叠的一发收掉（写在 [ow-4, ow)，与已写
        // 列重叠但值相同，幂等），[0, ow) 全覆盖、无标量残列。
        let emit = |ox: usize| {
            let mut acc = vdupq_n_f32(0.0);
            for ky in 0..kh {
                let row = base.add(ky * pwidth);
                for kx in 0..kw {
                    let wv = vdupq_n_f32(*wc.add(ky * kw + kx));
                    let p = row.add(ox * sw + kx);
                    let xv = if sw == 1 {
                        vld1q_f32(p)
                    } else {
                        // 读 [p, p+8)，vuzp1 抽偶数位 → p0,p2,p4,p6。
                        // SAFETY: 调用方按 `ow*sw + kw + 8` 保证 pwidth 够读。
                        vuzp1q_f32(vld1q_f32(p), vld1q_f32(p.add(4)))
                    };
                    acc = vfmaq_f32(acc, wv, xv);
                }
            }
            vst1q_f32(yrow.add(ox), vaddq_f32(acc, vbias));
        };
        let mut ox = 0usize;
        while ox + 4 <= ow {
            emit(ox);
            ox += 4;
        }
        if ox < ow {
            emit(ow - 4);
        }
    }
}

/// ConvTranspose 的 4 宽 interleave 内核：`j..j+4` 个输入产生 8 个连续
/// 输出。`a`/`b` 是两个 kx 奇偶的 c 累加；vzip1/vzip2 交错。
///
/// # Safety
///
/// `xr` 起有 `c*ch_plane` 个可读元素；`w0`/`w1` 各 `c` 个；`orow` 起有
/// `j*2+8` 个可写元素。
pub unsafe fn convt_row_neon(
    xr: *const f32,
    ch_plane: usize,
    w0: *const f32,
    w1: *const f32,
    c: usize,
    j: usize,
    orow: *mut f32,
) {
    let mut a = vdupq_n_f32(0.0);
    let mut b = vdupq_n_f32(0.0);
    for ch in 0..c {
        let xv = vld1q_f32(xr.add(ch * ch_plane + j));
        a = vfmaq_f32(a, xv, vdupq_n_f32(*w0.add(ch)));
        b = vfmaq_f32(b, xv, vdupq_n_f32(*w1.add(ch)));
    }
    // a -> 偶数输出列、b -> 奇数：[a0,b0,a1,b1] 与 [a2,b2,a3,b3]
    vst1q_f32(orow.add(j * 2), vzip1q_f32(a, b));
    vst1q_f32(orow.add(j * 2 + 4), vzip2q_f32(a, b));
}

// ================================================================ 池化 / 双线性 / 膨胀

/// 池化 2x2 s1 的行内相：`out[j] = max(max(r0[j], r1[j]), max(r0[j+1],
/// r1[j+1]))`；`r1` 为 null 表示最后一行（钳制）。`vm` 不使用（单趟）。
///
/// # Safety
///
/// `r0`/`r1`（非空时）至少 `w` 个可读元素；`out` 至少 `w` 个可写元素。
pub unsafe fn pool2x2_row_neon(
    r0: *const f32,
    r1: *const f32,
    _vm: *mut f32,
    out: *mut f32,
    w: usize,
) {
    if w == 0 {
        return;
    }
    // 右边界：最后一列没有 j+1，直接取本列的两行 max（输入非负、无 NaN，
    // 与 -inf 哨兵写法结果相同）。
    let last = w - 1;
    let mut ox = 0usize;
    if r1.is_null() {
        // 末行：out[j] = max(r0[j], r0[j+1])
        while ox + 4 <= last {
            let a = vld1q_f32(r0.add(ox));
            let b = vld1q_f32(r0.add(ox + 1));
            vst1q_f32(out.add(ox), vmaxq_f32(a, b));
            ox += 4;
        }
        while ox < last {
            *out.add(ox) = (*r0.add(ox)).max(*r0.add(ox + 1));
            ox += 1;
        }
        *out.add(last) = *r0.add(last);
        return;
    }
    while ox + 4 <= last {
        let v0 = vmaxq_f32(vld1q_f32(r0.add(ox)), vld1q_f32(r1.add(ox)));
        let v1 = vmaxq_f32(vld1q_f32(r0.add(ox + 1)), vld1q_f32(r1.add(ox + 1)));
        vst1q_f32(out.add(ox), vmaxq_f32(v0, v1));
        ox += 4;
    }
    while ox < last {
        let v0 = (*r0.add(ox)).max(*r1.add(ox));
        let v1 = (*r0.add(ox + 1)).max(*r1.add(ox + 1));
        *out.add(ox) = v0.max(v1);
        ox += 1;
    }
    *out.add(last) = (*r0.add(last)).max(*r1.add(last));
}

/// resize_bilinear 的行内积：4 个输出列一批。NEON 无 gather 指令，
/// 源下标逐列标量取数后拼向量；权重结合顺序固定（首项两乘，其余 fma）。
///
/// # Safety
///
/// 输出行区间与其他并行块不相交；映射表长度覆盖 `[ox0, ox1)`，源列界内。
#[allow(clippy::too_many_arguments)]
pub unsafe fn bilinear_row_neon(
    r0: *const f32,
    r1: *const f32,
    ix0: *const i32,
    ix1: *const i32,
    fx: *const f32,
    ly: f32,
    dst: *mut f32,
    ox0: usize,
    ox1: usize,
) {
    let only = vdupq_n_f32(1.0);
    let vly = vdupq_n_f32(ly);
    let mut ox = ox0;
    while ox + 4 <= ox1 {
        let j = ox - ox0;
        let mut l0 = [0f32; 4];
        let mut l1 = [0f32; 4];
        let mut h0 = [0f32; 4];
        let mut h1 = [0f32; 4];
        for q in 0..4 {
            let i0 = *ix0.add(j + q) as usize;
            let i1 = *ix1.add(j + q) as usize;
            l0[q] = *r0.add(i0);
            l1[q] = *r0.add(i1);
            h0[q] = *r1.add(i0);
            h1[q] = *r1.add(i1);
        }
        let vfx = vld1q_f32(fx.add(j));
        let invx = vsubq_f32(only, vfx);
        let invy = vsubq_f32(only, vly);
        let t = vmulq_f32(vmulq_f32(vld1q_f32(l0.as_ptr()), invx), invy);
        let t = vfmaq_f32(t, vmulq_f32(vld1q_f32(l1.as_ptr()), vfx), invy);
        let t = vfmaq_f32(t, vmulq_f32(vld1q_f32(h0.as_ptr()), invx), vly);
        let r = vfmaq_f32(t, vmulq_f32(vld1q_f32(h1.as_ptr()), vfx), vly);
        vst1q_f32(dst.add(ox), r);
        ox += 4;
    }
    crate::resize::bilinear_row_scalar(r0, r1, ix0, ix1, fx, ly, dst, ox, ox1);
}

/// 2×2 最大值膨胀的行内核（`x ∈ [1, w)`）：16 字节一批取 max。
///
/// # Safety
///
/// `cur` 至少 `w` 字节；`up` 非空时至少 `w` 字节；`out` 至少 `w` 字节。
pub unsafe fn dilate2x2_row_neon(cur: &[u8], up: Option<&[u8]>, out: *mut u8, w: usize) {
    let mut x = 1usize;
    while x + 16 <= w {
        let a = vld1q_u8(cur.as_ptr().add(x));
        let b = vld1q_u8(cur.as_ptr().add(x - 1));
        let mut v = vmaxq_u8(a, b);
        if let Some(u) = up {
            let c = vld1q_u8(u.as_ptr().add(x));
            let d = vld1q_u8(u.as_ptr().add(x - 1));
            v = vmaxq_u8(v, vmaxq_u8(c, d));
        }
        vst1q_u8(out.add(x), v);
        x += 16;
    }
    for xx in x..w {
        let mut v = (*cur.as_ptr().add(xx - 1)).max(*cur.as_ptr().add(xx));
        if let Some(u) = up {
            v = v.max(*u.as_ptr().add(xx - 1)).max(*u.as_ptr().add(xx));
        }
        *out.add(xx) = v;
    }
}
