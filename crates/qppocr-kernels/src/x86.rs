//! AVX2 + FMA 内核：`ops.cpp` SIMD 部分的移植。
//!
//! 微内核逐句镜像 C++ 的结构——循环顺序、分块、寄存器分配意图。**不要**
//! 改写成更 Rust 的形式：这些内核的性能靠裸指针 + 精确的寄存器分配拿到
//! （DESIGN.md §3.2），`#[target_feature]` 函数内的指针运算不越界由
//! 分发层的形状校验保证。
//!
//! 数值上这些内核与 `scalar` 版**逐位一致**：标量版就是按这里的表达式树
//! （`mul_add` 序列、round-ties-even、hsum 结合顺序）写的。对拍测试
//! `tests/bitexact.rs` 逐算子验证这一点。

#![allow(clippy::missing_safety_doc)]
// 见模块注释：安全性由分发层契约承担
// 本模块全部是 #[target_feature] unsafe 内核，前置条件由各函数的
// `# Safety` 段声明、由分发层保证；函数体内不再逐操作包 unsafe。
#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;

// ================================================================ 向量数学

/// exp 的 8-lane 版：range-reduce 到 k·ln2 + r，r 上多项式，按 2^k 缩放。
///
/// 系数与运算顺序与 C++ `exp256_ps` 逐字相同（含 ln2 两段拆分的固有
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
    let r = _mm256_sub_ps(
        _mm256_sub_ps(x, _mm256_mul_ps(kf, ln2_hi)),
        _mm256_mul_ps(kf, ln2_lo),
    );
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
pub unsafe fn erf256_ps(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let ax = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
    let t = _mm256_div_ps(one, _mm256_fmadd_ps(_mm256_set1_ps(0.327_591_1), ax, one));
    let mut p = _mm256_set1_ps(1.061_405_4);
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-1.453_152));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.421_413_7));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-0.284_496_74));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(0.254_829_6));
    p = _mm256_mul_ps(p, t);
    let e = exp256_ps(_mm256_mul_ps(ax, _mm256_sub_ps(_mm256_setzero_ps(), ax)));
    let r = _mm256_sub_ps(one, _mm256_mul_ps(p, e));
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
/// 段错误（C++ 注释原话：间歇性地）。非整面板经 `t[32]` 暂存再拷 nn 列。
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
            let a0 = a.add((m0 + 0) * k);
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
                (m0 + 0, [c00, c01, c02, c03]),
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
        // C++ 位级语义）。标量路径与 crate::gemm 的 panel_body 相同。
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
                // 非整面板尾行：C++ 在 AVX2 构建里这段本来就是标量
                // （bias 起种 + GCC 收缩的 FMA）。逐位语义两边共用。
                return_to_scalar_tail(a, b, c, m, n, k, ldc, bias, p, row);
            }
        }
    }
}

/// 非整面板的 M%4 尾行回退：直接执行标量逻辑（bias 起种）。
///
/// C++ 在 AVX2 构建里这段本来就是标量循环（GCC 收缩成 FMA），Rust 版
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
        // N%32 尾列：bias 起种的标量链（C++ 位级语义，与标量版共用）
        for j in n32..n {
            let mut sv = bv;
            for kk in 0..k {
                sv = ar.add(kk).read().mul_add(b.add(kk * n + j).read(), sv);
            }
            cp.add(j).write(sv);
        }
    }
}
