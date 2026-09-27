//! AVX2 + FMA 内核：设计文档 SIMD 部分。
//!
//! 微内核逐句镜像 基准的结构——循环顺序、分块、寄存器分配意图。**不要**
//! 改写成更 Rust 的形式：这些内核的性能靠裸指针 + 精确的寄存器分配拿到
//! ，`#[target_feature]` 函数内的指针运算不越界由
//! 分发层的形状校验保证。
//!
//! 数值上这些内核与 `scalar` 版**逐位一致**：标量版就是按这里的表达式树
//! （`mul_add` 序列、round-ties-even、hsum 结合顺序）写的。对拍测试
//! `tests/bitexact.rs` 逐算子验证这一点。

#![allow(clippy::missing_safety_doc)] // 见模块注释：安全性由分发层契约承担
#![allow(clippy::approx_constant)]
// 系数照抄  字面值：位级一致的要求
// 本模块全部是 #[target_feature] unsafe 内核，前置条件由各函数的
// `# Safety` 段声明、由分发层保证；函数体内不再逐操作包 unsafe。
#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;

// ================================================================ 向量数学

/// exp 的 8-lane 版：range-reduce 到 k·ln2 + r，r 上多项式，按 2^k 缩放。
///
/// 系数与运算顺序固定（含 ln2 两段拆分的固有
/// 误差——见 `scalar::exp1` 的注释，照抄不改）。
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn exp256_ps(x: __m256) -> __m256 {
    let ln2_hi = _mm256_set1_ps(0.693_147_2);
    let ln2_lo = _mm256_set1_ps(-2.980_232_2e-8);
    let inv_ln2 = _mm256_set1_ps(1.442_695);
    let one = _mm256_set1_ps(1.0);
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.0));
    let x = _mm256_max_ps(x, _mm256_set1_ps(-88.0));
    let kf = _mm256_round_ps(
        _mm256_mul_ps(x, inv_ln2),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    // ★ GCC -ffp-contract=fast 把源码里的 sub(x, mul(kf,·)) 两处都收缩成
    // vfnmadd（汇编实证：vroundps 后跟两条 vfnmadd231ps）。源码形态
    // `_mm256_sub_ps(x, _mm256_mul_ps(...))` 在 GCC 下**不是**分立指令——
    // 逐位对齐  必须显式 fnmadd。
    let r = _mm256_fnmadd_ps(kf, ln2_hi, x);
    let r = _mm256_fnmadd_ps(kf, ln2_lo, r);
    let mut p = _mm256_set1_ps(1.0 / 720.0);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 120.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 24.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 6.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
    p = _mm256_fmadd_ps(p, r, one);
    p = _mm256_fmadd_ps(p, r, one);
    let mut ki = _mm256_cvtps_epi32(kf);
    ki = _mm256_add_epi32(ki, _mm256_set1_epi32(127));
    ki = _mm256_slli_epi32::<23>(ki);
    _mm256_mul_ps(p, _mm256_castsi256_ps(ki))
}

/// erf 的 8-lane 版：A&S 7.1.26，branch-free。erf(-x) = -erf(x)，
/// 取 |x| 算、末尾还原符号。
#[inline]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::excessive_precision)] //  原字面值照抄
pub unsafe fn erf256_ps(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let ax = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
    let t = _mm256_div_ps(one, _mm256_fmadd_ps(_mm256_set1_ps(0.327_591_1), ax, one));
    let mut p = _mm256_set1_ps(1.061_405_4);
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-1.453152027));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.421413741));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-0.284496736));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(0.254_829_6));
    p = _mm256_mul_ps(p, t);
    let e = exp256_ps(_mm256_mul_ps(ax, _mm256_sub_ps(_mm256_setzero_ps(), ax)));
    // ★ 同上：1 − p·e 被 GCC 收缩成 vfnmadd132ps（汇编实证）
    let r = _mm256_fnmadd_ps(p, e, one);
    _mm256_or_ps(r, _mm256_and_ps(x, _mm256_set1_ps(-0.0)))
}

/// GELU 的 8-lane 版（`gelu256_ps`）：与 `gelu1` 相同的表达式、相同的顺序。
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu256_ps(x: __m256, vs: __m256, vc2: __m256, vc3: __m256) -> __m256 {
    let er = erf256_ps(_mm256_mul_ps(x, vs));
    _mm256_mul_ps(_mm256_mul_ps(vc3, x), _mm256_add_ps(er, vc2))
}

/// 8-lane 水平求和（`hsum256_ps`）：`((v0+v4)+(v1+v5)) + ((v2+v6)+(v3+v7))`。
/// 结合顺序被 `scalar::softmax` 镜像——动这里必须同时动那边。
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn hsum256_ps(v: __m256) -> f32 {
    let mut lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    _mm_cvtss_f32(lo)
}

/// 8-lane 水平最大（`hmax256_ps`）。max 逐位与顺序无关。
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn hmax256_ps(v: __m256) -> f32 {
    let mut lo = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_shuffle_ps::<1>(lo, lo));
    _mm_cvtss_f32(lo)
}

// ================================================================ sgemm

/// sgemm 面板体的 AVX2 版：`[pb, pe)` 号 N 面板。
///
/// 4 行 × 4 向量、k 内层 broadcast+FMA（16 路独立 FMA 打满流水线）。
/// 尾部面板只加载实际拥有的向量——在 B 的行尾读满 32 个 float 会越页
/// 段错误（原注释：间歇性地）。非整面板经 `t[32]` 暂存再拷 nn 列。
///
/// # Safety
///
/// 与 `crate::gemm` 里的标量 `panel_body` 相同：`c` 的列区间
/// `[p*32, p*32+nn)` 必须与并发调用者不相交；`a`/`b`/`bias` 长度已由
/// 分发层校验。
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sgemm_panel_avx2(
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
    let has_bias = !bias.is_null();
    for p in pb..pe {
        let n0 = p * 32;
        let nn = 32.min(n - n0);
        let full = nn == 32;
        let mut m0 = 0;
        while m0 + 4 <= m {
            // 16 个累加器 = 4 行 × 4 向量：k 内层每步 16 路独立 FMA
            let (mut c00, mut c01, mut c02, mut c03) = (
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
            );
            let (mut c10, mut c11, mut c12, mut c13) = (
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
            );
            let (mut c20, mut c21, mut c22, mut c23) = (
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
            );
            let (mut c30, mut c31, mut c32, mut c33) = (
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
            );
            let a0 = a.add(m0 * k);
            let a1 = a.add((m0 + 1) * k);
            let a2 = a.add((m0 + 2) * k);
            let a3 = a.add((m0 + 3) * k);
            for kk in 0..k {
                let bp = b.add(kk * n + n0);
                let bv0 = _mm256_loadu_ps(bp);
                let bv1 = if nn > 8 {
                    _mm256_loadu_ps(bp.add(8))
                } else {
                    _mm256_setzero_ps()
                };
                let bv2 = if nn > 16 {
                    _mm256_loadu_ps(bp.add(16))
                } else {
                    _mm256_setzero_ps()
                };
                let bv3 = if nn > 24 {
                    _mm256_loadu_ps(bp.add(24))
                } else {
                    _mm256_setzero_ps()
                };
                let mut av = _mm256_broadcast_ss(&*a0.add(kk));
                c00 = _mm256_fmadd_ps(av, bv0, c00);
                c01 = _mm256_fmadd_ps(av, bv1, c01);
                c02 = _mm256_fmadd_ps(av, bv2, c02);
                c03 = _mm256_fmadd_ps(av, bv3, c03);
                av = _mm256_broadcast_ss(&*a1.add(kk));
                c10 = _mm256_fmadd_ps(av, bv0, c10);
                c11 = _mm256_fmadd_ps(av, bv1, c11);
                c12 = _mm256_fmadd_ps(av, bv2, c12);
                c13 = _mm256_fmadd_ps(av, bv3, c13);
                av = _mm256_broadcast_ss(&*a2.add(kk));
                c20 = _mm256_fmadd_ps(av, bv0, c20);
                c21 = _mm256_fmadd_ps(av, bv1, c21);
                c22 = _mm256_fmadd_ps(av, bv2, c22);
                c23 = _mm256_fmadd_ps(av, bv3, c23);
                av = _mm256_broadcast_ss(&*a3.add(kk));
                c30 = _mm256_fmadd_ps(av, bv0, c30);
                c31 = _mm256_fmadd_ps(av, bv1, c31);
                c32 = _mm256_fmadd_ps(av, bv2, c32);
                c33 = _mm256_fmadd_ps(av, bv3, c33);
            }
            // 逐行加 bias、store。非整面板先落 t[32] 再拷 nn 列。
            let rows = [
                (m0, [c00, c01, c02, c03]),
                (m0 + 1, [c10, c11, c12, c13]),
                (m0 + 2, [c20, c21, c22, c23]),
                (m0 + 3, [c30, c31, c32, c33]),
            ];
            for (row, regs) in rows {
                let cp = c.add(row * ldc + n0);
                let mut r = regs;
                if has_bias {
                    let bv = _mm256_set1_ps(*bias.add(row));
                    for v in r.iter_mut() {
                        *v = _mm256_add_ps(*v, bv);
                    }
                }
                if full {
                    for (j, v) in r.iter().enumerate() {
                        _mm256_storeu_ps(cp.add(j * 8), *v);
                    }
                } else {
                    let mut t = [0f32; 32];
                    for (j, v) in r.iter().enumerate() {
                        _mm256_storeu_ps(t.as_mut_ptr().add(j * 8), *v);
                    }
                    std::ptr::copy_nonoverlapping(t.as_ptr(), cp, nn);
                }
            }
            m0 += 4;
        }
        // M%4 尾行：整面板走 1-row 向量内核；非整面板走标量（bias 起种，
        //  位级语义）。标量路径与 crate::gemm 的 panel_body 相同。
        for row in m0..m {
            let cp = c.add(row * ldc + n0);
            let bv = if has_bias { *bias.add(row) } else { 0.0 };
            if full {
                let (mut c0, mut c1, mut c2, mut c3) = (
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                );
                let ar = a.add(row * k);
                for kk in 0..k {
                    let bp = b.add(kk * n + n0);
                    let av = _mm256_broadcast_ss(&*ar.add(kk));
                    c0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp), c0);
                    if nn > 8 {
                        c1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp.add(8)), c1);
                    }
                    if nn > 16 {
                        c2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp.add(16)), c2);
                    }
                    if nn > 24 {
                        c3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp.add(24)), c3);
                    }
                }
                if has_bias {
                    let bvv = _mm256_set1_ps(bv);
                    c0 = _mm256_add_ps(c0, bvv);
                    c1 = _mm256_add_ps(c1, bvv);
                    c2 = _mm256_add_ps(c2, bvv);
                    c3 = _mm256_add_ps(c3, bvv);
                }
                _mm256_storeu_ps(cp, c0);
                _mm256_storeu_ps(cp.add(8), c1);
                _mm256_storeu_ps(cp.add(16), c2);
                _mm256_storeu_ps(cp.add(24), c3);
            } else {
                // 非整面板尾行： 在 AVX2 构建里这段本来就是标量
                // （bias 起种 + GCC 收缩的 FMA）。逐位语义两边共用。
                return_to_scalar_tail(a, b, c, m, n, k, ldc, bias, p, row);
            }
        }
    }
}

/// 非整面板的 M%4 尾行回退：直接执行标量逻辑（bias 起种）。
///
///  在 AVX2 构建里这段本来就是标量循环（GCC 收缩成 FMA），Rust 版
/// 把它放在 `gemm.rs::panel_tail_scalar` 里两边共用。
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn return_to_scalar_tail(
    a: *const f32,
    b: *const f32,
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
            // bias 起种 + fma 链（与标量 panel_body 的非整面板分支逐位相同）
            let mut s = bv;
            for kk in 0..k {
                s = ar.add(kk).read().mul_add(b.add(kk * n + n0 + j).read(), s);
            }
            cp.add(j).write(s);
        }
    }
}

/// 窄 N 路径的 AVX2 版：整行计算，K 上外积。
///
/// # Safety
///
/// 行区间 `[mb, me)` 必须与并发调用者不相交（与标量 `m_rows_body` 相同）。
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sgemm_mrows_avx2(
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
        let mut n_idx = 0;
        while n_idx + 32 <= n {
            let (mut c0, mut c1, mut c2, mut c3) = (
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
            );
            for kk in 0..k {
                let bp = b.add(kk * n + n_idx);
                let av = _mm256_broadcast_ss(&*ar.add(kk));
                c0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp), c0);
                c1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp.add(8)), c1);
                c2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp.add(16)), c2);
                c3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp.add(24)), c3);
            }
            if has_bias {
                let bvv = _mm256_set1_ps(bv);
                c0 = _mm256_add_ps(c0, bvv);
                c1 = _mm256_add_ps(c1, bvv);
                c2 = _mm256_add_ps(c2, bvv);
                c3 = _mm256_add_ps(c3, bvv);
            }
            _mm256_storeu_ps(cp.add(n_idx), c0);
            _mm256_storeu_ps(cp.add(n_idx + 8), c1);
            _mm256_storeu_ps(cp.add(n_idx + 16), c2);
            _mm256_storeu_ps(cp.add(n_idx + 24), c3);
            n_idx += 32;
        }
        // N%32 尾列：bias 起种的标量链（ 位级语义，与标量版共用）
        for j in n32..n {
            let mut sv = bv;
            for kk in 0..k {
                sv = ar.add(kk).read().mul_add(b.add(kk * n + j).read(), sv);
            }
            cp.add(j).write(sv);
        }
    }
}

// ================================================================ 激活（向量）

/// relu：`max(v, 0)`，就地。NaN→0（MAXPS 语义：任一 NaN 返回第二操作数）。
/// 与标量的 `v > 0 ? v : 0` 一致。
#[target_feature(enable = "avx2")]
pub unsafe fn relu_vec(t: *mut f32, b: usize, e: usize) {
    let z = _mm256_setzero_ps();
    let mut i = b;
    while i + 8 <= e {
        let p = t.add(i);
        _mm256_storeu_ps(p, _mm256_max_ps(_mm256_loadu_ps(p), z));
        i += 8;
    }
    while i < e {
        *t.add(i) = if *t.add(i) > 0.0 { *t.add(i) } else { 0.0 };
        i += 1;
    }
}

/// hardsigmoid：`clip(0, 1, fma(x, a, b))`，就地。clamp 顺序与标量一致。
#[target_feature(enable = "avx2,fma")]
pub unsafe fn hardsigmoid_vec(
    x: *const f32,
    y: *mut f32,
    b: usize,
    e: usize,
    alpha: f32,
    beta: f32,
) {
    let va = _mm256_set1_ps(alpha);
    let vb = _mm256_set1_ps(beta);
    let vz = _mm256_setzero_ps();
    let vo = _mm256_set1_ps(1.0);
    let mut i = b;
    while i + 8 <= e {
        let v = _mm256_fmadd_ps(_mm256_loadu_ps(x.add(i)), va, vb);
        _mm256_storeu_ps(y.add(i), _mm256_max_ps(vz, _mm256_min_ps(vo, v)));
        i += 8;
    }
    while i < e {
        let v = (*x.add(i)).mul_add(alpha, beta);
        *y.add(i) = if v < 0.0 {
            0.0
        } else if v > 1.0 {
            1.0
        } else {
            v
        };
        i += 1;
    }
}

/// sigmoid：`1 / (1 + exp256(-x))`。
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sigmoid_vec(x: *const f32, y: *mut f32, b: usize, e: usize) {
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let mut i = b;
    while i + 8 <= e {
        let v = _mm256_loadu_ps(x.add(i));
        let en = exp256_ps(_mm256_sub_ps(zero, v));
        _mm256_storeu_ps(y.add(i), _mm256_div_ps(one, _mm256_add_ps(one, en)));
        i += 8;
    }
    while i < e {
        let xv = *x.add(i);
        *y.add(i) = 1.0f32 / (1.0f32 + crate::activation::exp1(-xv));
        i += 1;
    }
}

/// gelu：`c3·x·(erf256(x·(1/c1)) + c2)`，就地（融合 GELU 与 apply_act 共用）。
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gelu_vec(t: *mut f32, b: usize, e: usize, c1: f32, c2: f32, c3: f32) {
    let inv_c1 = 1.0f32 / c1;
    let vs = _mm256_set1_ps(inv_c1);
    let vc2 = _mm256_set1_ps(c2);
    let vc3 = _mm256_set1_ps(c3);
    let mut i = b;
    while i + 8 <= e {
        let p = t.add(i);
        let x = _mm256_loadu_ps(p);
        _mm256_storeu_ps(p, gelu256_ps(x, vs, vc2, vc3));
        i += 8;
    }
    while i < e {
        let v = *t.add(i);
        *t.add(i) = c3 * v * (crate::activation::erf1(v * inv_c1) + c2);
        i += 1;
    }
}

/// clip：`min(hi, max(lo, v))`，就地。NaN 语义与标量版对齐（穿透）：
/// MAXPS/MINPS 的 NaN 处理是「返回第二操作数」，这里 lo/hi 在第二位，
/// NaN 输入返回 lo/hi——与标量的比较链不同，故 NaN 只在输入为 NaN 时
/// 出现差异；正常数据逐位一致。
#[target_feature(enable = "avx2")]
pub unsafe fn clip_vec(t: *mut f32, b: usize, e: usize, lo: f32, hi: f32) {
    let vlo = _mm256_set1_ps(lo);
    let vhi = _mm256_set1_ps(hi);
    let mut i = b;
    while i + 8 <= e {
        let p = t.add(i);
        _mm256_storeu_ps(
            p,
            _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(p), vlo), vhi),
        );
        i += 8;
    }
    while i < e {
        let m = if *t.add(i) < lo { lo } else { *t.add(i) };
        *t.add(i) = if hi < m { hi } else { m };
        i += 1;
    }
}

// ================================================================ 二元（向量）

/// 同形平坦 pass：`dst[i] op= src[i]`（就地）。op 码：0..3 = + - * /。
#[target_feature(enable = "avx2")]
pub unsafe fn binary_flat_inplace_vec(dst: *mut f32, src: *const f32, b: usize, e: usize, op: u8) {
    let mut i = b;
    while i + 8 <= e {
        let d = _mm256_loadu_ps(dst.add(i));
        let s = _mm256_loadu_ps(src.add(i));
        let r = match op {
            0 => _mm256_add_ps(d, s),
            1 => _mm256_sub_ps(d, s),
            2 => _mm256_mul_ps(d, s),
            _ => _mm256_div_ps(d, s),
        };
        _mm256_storeu_ps(dst.add(i), r);
        i += 8;
    }
    while i < e {
        let d = *dst.add(i);
        let s = *src.add(i);
        *dst.add(i) = match op {
            0 => d + s,
            1 => d - s,
            2 => d * s,
            3 => d / s,
            _ => d.powf(s),
        };
        i += 1;
    }
}

/// run 路径：`dst[j] op= cv`（b 侧常量广播）或 `dst[j] op= bsrc[j]`
///（b 侧稠密）。a_const=false 时为就地版（dst 是 a）。
#[allow(clippy::too_many_arguments)] // 内核入参镜像
#[target_feature(enable = "avx2")]
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
    let vc = _mm256_set1_ps(cv);
    let mut j = 0usize;
    while j + 8 <= runlen {
        let d = _mm256_loadu_ps(dst.add(o * runlen + j));
        let r = if b_dense_run {
            let s = _mm256_loadu_ps(bsrc.add(ib + j));
            match op {
                0 => _mm256_add_ps(d, s),
                1 => _mm256_sub_ps(d, s),
                2 => _mm256_mul_ps(d, s),
                _ => _mm256_div_ps(d, s),
            }
        } else {
            match op {
                0 => _mm256_add_ps(d, vc),
                1 => _mm256_sub_ps(d, vc),
                2 => _mm256_mul_ps(d, vc),
                _ => _mm256_div_ps(d, vc),
            }
        };
        _mm256_storeu_ps(dst.add(o * runlen + j), r);
        j += 8;
    }
    while j < runlen {
        let d = *dst.add(o * runlen + j);
        let bv = if b_dense_run { *bsrc.add(ib + j) } else { cv };
        *dst.add(o * runlen + j) = match op {
            0 => d + bv,
            1 => d - bv,
            2 => d * bv,
            3 => d / bv,
            _ => d.powf(bv),
        };
        j += 1;
    }
}

// ================================================================ conv 相关

/// depthwise 的 sw==1 内层：`yd[i] = fma(wv, xd[i], yd[i])`，len 个元素。
#[target_feature(enable = "avx2,fma")]
pub unsafe fn depthwise_fma_vec(yd: *mut f32, xd: *const f32, len: usize, wv: f32) {
    let vw = _mm256_set1_ps(wv);
    let mut i = 0usize;
    while i + 8 <= len {
        let p = yd.add(i);
        _mm256_storeu_ps(
            p,
            _mm256_fmadd_ps(vw, _mm256_loadu_ps(xd.add(i)), _mm256_loadu_ps(p)),
        );
        i += 8;
    }
    while i < len {
        *yd.add(i) = wv.mul_add(*xd.add(i), *yd.add(i));
        i += 1;
    }
}

/// ConvTranspose 的 8 宽 interleave 内核：j..j+8 个输入产生 16 个连续输出。
/// `a`/`b` 是两个 kx 奇偶的 c 累加；unpacklo/unpackhi + permute2f128 交错。
#[target_feature(enable = "avx2,fma")]
pub unsafe fn convt_row_vec(
    xr: *const f32,
    ch_plane: usize, // 每通道平面跨度 = h*w
    w0: *const f32,
    w1: *const f32,
    c: usize,
    j: usize,
    orow: *mut f32,
) {
    let mut a = _mm256_setzero_ps();
    let mut b = _mm256_setzero_ps();
    for ch in 0..c {
        let xv = _mm256_loadu_ps(xr.add(ch * ch_plane + j));
        a = _mm256_fmadd_ps(xv, _mm256_set1_ps(*w0.add(ch)), a);
        b = _mm256_fmadd_ps(xv, _mm256_set1_ps(*w1.add(ch)), b);
    }
    let lo = _mm256_unpacklo_ps(a, b);
    let hi = _mm256_unpackhi_ps(a, b);
    _mm256_storeu_ps(orow.add(j * 2), _mm256_permute2f128_ps::<0x20>(lo, hi));
    _mm256_storeu_ps(orow.add(j * 2 + 8), _mm256_permute2f128_ps::<0x31>(lo, hi));
}

/// softmax 行内向量相：max/exp/归一的 8-lane 循环（语义与标量镜像版一致，
/// 标量版按这里的 lane 结构与 hsum 结合顺序写的）。
/// 前置条件：`inner >= 8`（不足时由分发层走标量路径）。
#[target_feature(enable = "avx2,fma")]
pub unsafe fn softmax_row_vec(row: *mut f32, inner: usize) {
    // max：向量化部分逐 lane，标量尾巴单独并入（max 顺序无关）
    let mut vmax = _mm256_loadu_ps(row);
    let mut i = 8usize;
    while i + 8 <= inner {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row.add(i)));
        i += 8;
    }
    let mut m2 = hmax256_ps(vmax);
    while i < inner {
        if *row.add(i) > m2 {
            m2 = *row.add(i);
        }
        i += 1;
    }
    let vmx = _mm256_set1_ps(m2);
    let mut vsum = _mm256_setzero_ps();
    let mut i3 = 0usize;
    while i3 + 8 <= inner {
        let ev = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(row.add(i3)), vmx));
        _mm256_storeu_ps(row.add(i3), ev);
        vsum = _mm256_add_ps(vsum, ev);
        i3 += 8;
    }
    let mut sum = hsum256_ps(vsum);
    while i3 < inner {
        let ev = crate::activation::exp1(*row.add(i3) - m2);
        *row.add(i3) = ev;
        sum += ev;
        i3 += 1;
    }
    let inv = 1.0f32 / sum;
    let vinv = _mm256_set1_ps(inv);
    let mut i4 = 0usize;
    while i4 + 8 <= inner {
        let p = row.add(i4);
        _mm256_storeu_ps(p, _mm256_mul_ps(_mm256_loadu_ps(p), vinv));
        i4 += 8;
    }
    while i4 < inner {
        *row.add(i4) *= inv;
        i4 += 1;
    }
}

/// 池化 2x2 s1 的行内相：垂直 max 落 vm、水平 max 落 out（可分）。
/// vm 长度 W+1，vm[W] 是 -inf 哨兵。`r1` 为 null 表示最后一行（钳制：
/// vm 就是 r0 的拷贝——绝不能读 r0+W，那是**下一个通道**的数据）。
#[target_feature(enable = "avx2")]
pub unsafe fn pool2x2_row_vec(
    r0: *const f32,
    r1: *const f32,
    vm: *mut f32,
    out: *mut f32,
    w: usize,
) {
    let neg_inf = -f32::MAX;
    if r1.is_null() {
        std::ptr::copy_nonoverlapping(r0, vm, w);
    } else {
        let mut j = 0usize;
        while j + 8 <= w {
            _mm256_storeu_ps(
                vm.add(j),
                _mm256_max_ps(_mm256_loadu_ps(r0.add(j)), _mm256_loadu_ps(r1.add(j))),
            );
            j += 8;
        }
        while j < w {
            *vm.add(j) = (*r0.add(j)).max(*r1.add(j));
            j += 1;
        }
    }
    *vm.add(w) = neg_inf;
    let mut ox = 0usize;
    while ox + 8 <= w {
        _mm256_storeu_ps(
            out.add(ox),
            _mm256_max_ps(_mm256_loadu_ps(vm.add(ox)), _mm256_loadu_ps(vm.add(ox + 1))),
        );
        ox += 8;
    }
    while ox < w {
        *out.add(ox) = (*vm.add(ox)).max(*vm.add(ox + 1));
        ox += 1;
    }
}

// ================================================================ 二元（分配路径）

/// 同形平坦：`y[i] = a[i] op b[i]`（写新缓冲，非 inplace）。op 码同 。
#[target_feature(enable = "avx2")]
pub unsafe fn binary_flat_vec(
    pa: *const f32,
    pb: *const f32,
    py: *mut f32,
    b: usize,
    e: usize,
    op: u8,
) {
    let mut i = b;
    while i + 8 <= e {
        let va = _mm256_loadu_ps(pa.add(i));
        let vb = _mm256_loadu_ps(pb.add(i));
        let r = match op {
            0 => _mm256_add_ps(va, vb),
            1 => _mm256_sub_ps(va, vb),
            2 => _mm256_mul_ps(va, vb),
            _ => _mm256_div_ps(va, vb),
        };
        _mm256_storeu_ps(py.add(i), r);
        i += 8;
    }
    while i < e {
        let a = *pa.add(i);
        let bv = *pb.add(i);
        *py.add(i) = match op {
            0 => a + bv,
            1 => a - bv,
            2 => a * bv,
            3 => a / bv,
            _ => a.powf(bv),
        };
        i += 1;
    }
}

/// 单元素广播：`y[i] = op(sv, v[i])` 或 `op(v[i], sv)`（a_scalar 决定方向）。
#[target_feature(enable = "avx2")]
pub unsafe fn binary_scalar_vec(
    v: *const f32,
    sv: f32,
    a_scalar: bool,
    py: *mut f32,
    b: usize,
    e: usize,
    op: u8,
) {
    let vs = _mm256_set1_ps(sv);
    let mut i = b;
    while i + 8 <= e {
        let vv = _mm256_loadu_ps(v.add(i));
        let r = match (op, a_scalar) {
            (0, true) => _mm256_add_ps(vs, vv),
            (0, false) => _mm256_add_ps(vv, vs),
            (1, true) => _mm256_sub_ps(vs, vv),
            (1, false) => _mm256_sub_ps(vv, vs),
            (2, _) => _mm256_mul_ps(vv, vs),
            (_, true) => _mm256_div_ps(vs, vv),
            (_, false) => _mm256_div_ps(vv, vs),
        };
        _mm256_storeu_ps(py.add(i), r);
        i += 8;
    }
    while i < e {
        let x = *v.add(i);
        let (va, vb) = if a_scalar { (sv, x) } else { (x, sv) };
        *py.add(i) = match op {
            0 => va + vb,
            1 => va - vb,
            2 => va * vb,
            3 => va / vb,
            _ => va.powf(vb),
        };
        i += 1;
    }
}

/// run 路径：`y[o*runlen + j] = op(a值, b[j] 或 cv)`。b 稠密时 bsrc，否则 cv 广播。
#[allow(clippy::too_many_arguments)] // 内核入参镜像
#[target_feature(enable = "avx2")]
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
    let vc = _mm256_set1_ps(cv);
    let dense = bsrc.add(ib);
    let mut j = 0usize;
    while j + 8 <= runlen {
        let bv = if b_dense_run {
            _mm256_loadu_ps(dense.add(j))
        } else {
            vc
        };
        let av = if a_const {
            _mm256_set1_ps(*pa_const)
        } else {
            _mm256_loadu_ps(dst.add(j)) // 不该发生（a_const 必 true 才走 run alloc）
        };
        let r = match op {
            0 => _mm256_add_ps(av, bv),
            1 => _mm256_sub_ps(av, bv),
            2 => _mm256_mul_ps(av, bv),
            _ => _mm256_div_ps(av, bv),
        };
        _mm256_storeu_ps(dst.add(j), r);
        j += 8;
    }
    while j < runlen {
        let bv = if b_dense_run { *dense.add(j) } else { cv };
        let av = if a_const { *pa_const } else { *dst.add(j) };
        *dst.add(j) = match op {
            0 => av + bv,
            1 => av - bv,
            2 => av * bv,
            3 => av / bv,
            _ => av.powf(bv),
        };
        j += 1;
    }
}

// ================================================================ 双线性（f32 Resize 算子）

/// resize_bilinear 的行内积向量化：8 个输出列一批。
/// `ix0/ix1/fx` 是预计算的源列映射（每输出列一对），`r0/r1` 是上下两行。
/// 权重结合顺序与基准一致（首项两乘，其余 fma）。
#[allow(clippy::too_many_arguments)] // 内核入参：两行 + 三张映射表 + 权重 + 目标 + 范围
#[target_feature(enable = "avx2,fma")]
pub unsafe fn bilinear_row_vec(
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
    let only = _mm256_set1_ps(1.0);
    let vly = _mm256_set1_ps(ly);
    let mut ox = ox0;
    // fx 的补集一次算好：inv = 1 - fx
    while ox + 8 <= ox1 {
        let j = ox - ox0;
        let lix0 = _mm256_i32gather_ps(r0, _mm256_loadu_si256(ix0.add(j) as *const __m256i), 4);
        let lix1 = _mm256_i32gather_ps(r0, _mm256_loadu_si256(ix1.add(j) as *const __m256i), 4);
        let hix0 = _mm256_i32gather_ps(r1, _mm256_loadu_si256(ix0.add(j) as *const __m256i), 4);
        let hix1 = _mm256_i32gather_ps(r1, _mm256_loadu_si256(ix1.add(j) as *const __m256i), 4);
        let vfx = _mm256_loadu_ps(fx.add(j));
        let invx = _mm256_sub_ps(only, vfx);
        let invy = _mm256_sub_ps(only, vly);
        let t = _mm256_mul_ps(_mm256_mul_ps(lix0, invx), invy);
        let t = _mm256_fmadd_ps(_mm256_mul_ps(lix1, vfx), invy, t);
        let t = _mm256_fmadd_ps(_mm256_mul_ps(hix0, invx), vly, t);
        let r = _mm256_fmadd_ps(_mm256_mul_ps(hix1, vfx), vly, t);
        _mm256_storeu_ps(dst.add(ox), r);
        ox += 8;
    }
    while ox < ox1 {
        let j = ox - ox0;
        let lx = *fx.add(j);
        let i0 = *ix0.add(j) as usize;
        let i1 = *ix1.add(j) as usize;
        let t = *r0.add(i0) * (1.0 - lx) * (1.0 - ly);
        let t = (*r0.add(i1) * lx).mul_add(1.0 - ly, t);
        let t = (*r1.add(i0) * (1.0 - lx)).mul_add(ly, t);
        *dst.add(ox) = (*r1.add(i1) * lx).mul_add(ly, t);
        ox += 1;
    }
}
