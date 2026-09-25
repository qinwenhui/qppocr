//! 激活与逐元素数值内核：设计文档 激活部分。
//!
//! # ★ 多项式近似——系数与运算顺序逐字照抄，不要换成 libm
//!
//!  用手写的向量多项式（`erf256_ps` / `exp256_ps`），这既影响速度也影响
//! **数值逐位一致**：
//!
//! - `erf`：Abramowitz & Stegun 7.1.26，|err| < 1.5e-7，branch-free。
//!   erf(-x) = -erf(x)，取 |x| 算、末尾还原符号。
//! - `exp`：range-reduce 到 k·ln2 + r，r 上多项式，按 2^k 缩放。
//!   ±88 截断保证两端饱和与 libm 路径一致（1/(1+inf) == 0）。
//!
//! 标量版 [`erf1`] / [`exp1`] 是 AVX2 版的**逐 lane 镜像**（同样的
//! `mul_add` 序列、同样的 round-ties-even），因此标量与 SIMD 逐位一致——
//! 这是把 scalar 当判据的前提。基准的标量尾巴调 libm（`erff`/`exp`），
//! 与其向量版本本来就不逐位一致；我们不沿用那个尾巴。
//!
//! ⚠ 已声明的与 基准的偏差：仅在 `inner % 8` 的标量尾巴上， 用 libm、
//! 我们用同一多项式，差异 ≤1 ulp。文本级对拍（阶段 3 判据）不受影响；
//! 若阶段 2 的中间张量对拍在这里翻车，再局部处理。

/// `exp(x)`：`exp256_ps` 的标量镜像（f32）。
///
/// 运算序列：截断到 ±88 → k = round_ties_even(x·inv_ln2) →
/// r = x − k·ln2_hi − k·ln2_lo（两段拆开保精度）→ 7 项多项式（Horner，
/// mul_add）→ 乘 2^k（指数位直接拼）。
#[inline]
#[allow(clippy::approx_constant)] //  字面值照抄：换 std 常数会破坏逐位一致
pub fn exp1(x: f32) -> f32 {
    const LN2_HI: f32 = 0.693_147_2;
    const LN2_LO: f32 = -2.980_232_2e-8;
    const INV_LN2: f32 = 1.442_695;
    let x = x.min(88.0).max(-88.0);
    let kf = (x * INV_LN2).round_ties_even();
    // fnmadd 镜像（GCC 对 x86::exp256_ps 的收缩形态，见那里的注释）
    let r = (-kf).mul_add(LN2_HI, x);
    let r = (-kf).mul_add(LN2_LO, r);
    let mut p = 1.0f32 / 720.0;
    p = p.mul_add(r, 1.0 / 120.0);
    p = p.mul_add(r, 1.0 / 24.0);
    p = p.mul_add(r, 1.0 / 6.0);
    p = p.mul_add(r, 0.5);
    p = p.mul_add(r, 1.0);
    p = p.mul_add(r, 1.0);
    let ki = (kf as i32).wrapping_add(127);
    // 2^ki：直接拼指数位。kf ∈ [-127, 127]（±88 截断后），ki ∈ [0, 254]。
    f32::from_bits((ki as u32) << 23) * p
}

/// `erf(x)`：`erf256_ps` 的标量镜像（f32），A&S 7.1.26。
#[inline]
#[allow(clippy::approx_constant)] // 同上：0.3275911 等系数是 A&S 7.1.26 原文
#[allow(clippy::excessive_precision)] //  原字面值照抄：f32 舍入位级锁定
pub fn erf1(x: f32) -> f32 {
    let ax = f32::from_bits(x.to_bits() & 0x7fff_ffff); // andnot(-0.0)
    let t = 1.0f32 / 0.327_591_1f32.mul_add(ax, 1.0);
    let mut p = 1.061_405_4f32;
    p = p.mul_add(t, -1.453152027);
    p = p.mul_add(t, 1.421413741);
    p = p.mul_add(t, -0.284496736);
    p = p.mul_add(t, 0.254_829_6);
    p *= t;
    let e = exp1(ax * (0.0 - ax));
    // fnmadd 镜像（同上）
    let r = (-p).mul_add(e, 1.0);
    f32::from_bits(r.to_bits() | (x.to_bits() & 0x8000_0000)) // 还原符号
}

/// 内核可以折进自己输出的激活（对应 基准的 `Activation`，
/// 让图优化能删掉紧随其后的激活节点）。
///
/// 它作用在**已写出的输出**上，不是累加器上——变换累加器会把 erf 的除法和
/// 常数拖进占满 16 个 YMM 的循环（det_small 慢 5.3%）；读回刚 store 的数据
/// 不占 GEMM 寄存器，而且还在 cache 里。
#[derive(Clone, Copy, Debug)]
pub struct Activation {
    /// 激活种类。
    pub kind: ActKind,
    /// GELU 的除数系数（c1，默认 √2）。
    pub c1: f32,
    /// GELU 的加数（c2，默认 1）。
    pub c2: f32,
    /// GELU 的乘数（c3，默认 0.5）。
    pub c3: f32,
}

/// 激活种类。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActKind {
    /// 无。
    None,
    /// `c3·x·(erf(x/c1) + c2)`。
    Gelu,
}

impl Default for Activation {
    #![allow(clippy::approx_constant)] //  ops.hpp 的默认值照抄
    fn default() -> Self {
        Self {
            kind: ActKind::None,
            c1: 1.414_213_5,
            c2: 1.0,
            c3: 0.5,
        }
    }
}

impl Activation {
    /// 是否有激活要做。
    pub fn is_on(&self) -> bool {
        self.kind != ActKind::None
    }
    /// 构造一个 GELU 激活（v6 图里出现的那组常数）。
    pub fn gelu(c1: f32, c2: f32, c3: f32) -> Self {
        Self {
            kind: ActKind::Gelu,
            c1,
            c2,
            c3,
        }
    }
}

/// 就地施加 `act` 到 `p`（`apply_act`）。
///
/// 与 `gelu_inplace` 用同一组表达式、同一顺序——这里成立的唯一理由是它
/// **精确**：读回的值就是刚写出的值。
pub fn apply_act(p: &mut [f32], act: &Activation) {
    if !act.is_on() || p.is_empty() {
        return;
    }
    gelu_inplace(p, act.c1, act.c2, act.c3);
}

/// `y = max(0, x)`，就地。`v > 0 ? v : 0` 的写法保留 基准的 NaN→0 语义。
pub fn relu_inplace(t: &mut [f32]) {
    let tp = par::SyncPtr::new(t.as_mut_ptr());
    let n = t.len();
    par::parallel_for_elems(n, n, |b, e| {
        #[cfg(target_arch = "x86_64")]
        #[cfg(target_arch = "x86_64")]
        if crate::use_avx2() {
            // SAFETY: 区间 [b, e) 与其他并行块不相交。
            unsafe { crate::x86::relu_vec(tp.get(), b, e) };
            return;
        }
        // SAFETY: 元素区间 [b, e) 与其他并行块不相交。
        let seg = unsafe { tp.offset(b).slice(e - b) };
        for v in seg.iter_mut() {
            *v = if *v > 0.0 { *v } else { 0.0 };
        }
    });
}

/// `y = clip(0, 1, x·alpha + beta)`。PP-OCR 的 HardSigmoid（mobile 版，
/// alpha=1/6、beta=1/2）走这里。
pub fn hardsigmoid(x: &[f32], alpha: f32, beta: f32, y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "hardsigmoid: size mismatch");
    let yp = par::SyncPtr::new(y.as_mut_ptr());
    let n = x.len();
    par::parallel_for_elems(n, n, |b, e| {
        #[cfg(target_arch = "x86_64")]
        #[cfg(target_arch = "x86_64")]
        if crate::use_avx2() {
            // SAFETY: 区间 [b, e) 与其他并行块不相交；x/y 等长已断言。
            unsafe { crate::x86::hardsigmoid_vec(x.as_ptr(), yp.get(), b, e, alpha, beta) };
            return;
        }
        // SAFETY: 元素区间 [b, e) 与其他并行块不相交。
        let seg = unsafe { yp.offset(b).slice(e - b) };
        for (i, d) in seg.iter_mut().enumerate() {
            // v = fma(x, alpha, beta)；clamp 顺序与 AVX2 一致：max(0, min(1, v))
            let v = x[b + i].mul_add(alpha, beta);
            *d = if v < 0.0 {
                0.0
            } else if v > 1.0 {
                1.0
            } else {
                v
            };
        }
    });
}

/// `y = 1 / (1 + exp(-x))`（整张量，拷贝语义）。
///
/// det 的 sigmoid 面是 1x1x1504x1984（3M 元素）： 标量版逐元素调 libm
/// 实测 17 ms，向量版 ~2 ms。
pub fn sigmoid_tensor(x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "sigmoid: size mismatch");
    let yp = par::SyncPtr::new(y.as_mut_ptr());
    let n = x.len();
    par::parallel_for_elems(n, n, |b, e| {
        #[cfg(target_arch = "x86_64")]
        #[cfg(target_arch = "x86_64")]
        if crate::use_avx2() {
            // SAFETY: 区间 [b, e) 与其他并行块不相交；x/y 等长已断言。
            unsafe { crate::x86::sigmoid_vec(x.as_ptr(), yp.get(), b, e) };
            return;
        }
        // SAFETY: 元素区间 [b, e) 与其他并行块不相交。
        let seg = unsafe { yp.offset(b).slice(e - b) };
        for (i, d) in seg.iter_mut().enumerate() {
            *d = 1.0f32 / (1.0f32 + exp1(-x[b + i]));
        }
    });
}

/// `t = c3·t·(erf(t/c1) + c2)`，就地。融合 GELU：图导出器会把它拆成
/// Div/Erf/Add/Mul/Mul 五个节点，这里一趟做完。
pub fn gelu_inplace(t: &mut [f32], c1: f32, c2: f32, c3: f32) {
    let inv_c1 = 1.0f32 / c1;
    let tp = par::SyncPtr::new(t.as_mut_ptr());
    let n = t.len();
    par::parallel_for_elems(n, n, |b, e| {
        #[cfg(target_arch = "x86_64")]
        #[cfg(target_arch = "x86_64")]
        if crate::use_avx2() {
            // SAFETY: 区间 [b, e) 与其他并行块不相交。
            unsafe { crate::x86::gelu_vec(tp.get(), b, e, c1, c2, c3) };
            return;
        }
        // SAFETY: 元素区间 [b, e) 与其他并行块不相交。
        let seg = unsafe { tp.offset(b).slice(e - b) };
        for v in seg.iter_mut() {
            *v = c3 * *v * (erf1(*v * inv_c1) + c2);
        }
    });
}

/// `t = erf(t)`，就地（未融合路径用）。
pub fn erf_inplace(t: &mut [f32]) {
    let tp = par::SyncPtr::new(t.as_mut_ptr());
    let n = t.len();
    par::parallel_for_elems(n, n, |b, e| {
        // SAFETY: 元素区间 [b, e) 与其他并行块不相交。
        let seg = unsafe { tp.offset(b).slice(e - b) };
        for v in seg.iter_mut() {
            *v = erf1(*v);
        }
    });
}

/// `t = clip(t, lo, hi)`，就地。比较写法与标量版一致（NaN 穿透）。
pub fn clip_inplace(t: &mut [f32], lo: f32, hi: f32) {
    let tp = par::SyncPtr::new(t.as_mut_ptr());
    let n = t.len();
    par::parallel_for_elems(n, n, |b, e| {
        #[cfg(target_arch = "x86_64")]
        #[cfg(target_arch = "x86_64")]
        if crate::use_avx2() {
            // SAFETY: 区间 [b, e) 与其他并行块不相交。
            unsafe { crate::x86::clip_vec(tp.get(), b, e, lo, hi) };
            return;
        }
        // SAFETY: 元素区间 [b, e) 与其他并行块不相交。
        let seg = unsafe { tp.offset(b).slice(e - b) };
        for v in seg.iter_mut() {
            *v = {
                let m = if *v < lo { lo } else { *v }; // max(v, lo)
                if hi < m { hi } else { m } // min(m, hi)
            };
        }
    });
}

/// `y = clip(x, lo, hi)`（拷贝语义）。
pub fn clip_tensor(x: &[f32], lo: f32, hi: f32, y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "clip: size mismatch");
    y.copy_from_slice(x);
    clip_inplace(y, lo, hi);
}

/// 8-lane 求和，**严格镜像 `hsum256_ps` 的结合顺序**：
/// `((v0+v4)+(v1+v5)) + ((v2+v6)+(v3+v7))`。
///
/// 浮点加法不可结合，softmax 的和必须与 AVX2 版同序才能逐位一致。
#[inline]
fn hsum8(v: &[f32; 8]) -> f32 {
    let t = [v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7]];
    (t[0] + t[1]) + (t[2] + t[3])
}

/// 最后一维 softmax，就地。`shape` 是 `t` 的形状。
///
/// 单元是整行（`inner` 个元素），不是元素——行数才是 grain。
pub fn softmax_last_dim(t: &mut [f32], inner: usize) {
    assert!(t.len() % inner == 0, "softmax: size not multiple of inner");
    let outer = t.len() / inner;
    let tp = par::SyncPtr::new(t.as_mut_ptr());
    #[cfg(target_arch = "x86_64")]
    let use_vec = inner >= 8 && crate::use_avx2();
    par::parallel_for_units(outer, |b, e| {
        for o in b..e {
            #[cfg(target_arch = "x86_64")]
            if use_vec {
                // SAFETY: 行 o 与其他并行块不相交。
                unsafe { crate::x86::softmax_row_vec(tp.get().add(o * inner), inner) };
                continue;
            }
            // SAFETY: 行 o 与其他并行块不相交。
            let row = unsafe { tp.offset(o * inner).slice(inner) };
            // 行最大值。max 逐位与顺序无关（NaN 除外，这里不会出现）。
            // ⚠ 行必须凑满一个向量再首次 load：不满时多余的 lane 会读到
            // *下一行*，mx 偏大、每个 exp 下溢、结果是 0·inf。角度分类器
            // softmax 的是 4 个方向分——这个 bug 曾让它静默把每张图都判错方向。
            let mut i = 0;
            let mut acc = [0f32; 8];
            let mut mx = if inner >= 8 {
                let mut vmax = [row[0]; 8];
                // 镜像向量循环：逐 lane max
                while i + 8 <= inner {
                    for j in 0..8 {
                        if row[i + j] > vmax[j] {
                            vmax[j] = row[i + j];
                        }
                    }
                    i += 8;
                }
                let mut m = vmax[0];
                for j in 1..8 {
                    if vmax[j] > m {
                        m = vmax[j];
                    }
                }
                m
            } else {
                i = 1;
                row[0]
            };
            while i < inner {
                if row[i] > mx {
                    mx = row[i];
                }
                i += 1;
            }
            // exp 阶段：逐 lane 累加（vsum），随后按 hsum8 的结合顺序归约。
            let mut sum = 0.0f32;
            let mut i2 = 0;
            if inner >= 8 {
                while i2 + 8 <= inner {
                    for j in 0..8 {
                        let ev = exp1(row[i2 + j] - mx);
                        row[i2 + j] = ev;
                        acc[j] += ev;
                    }
                    i2 += 8;
                }
                sum = hsum8(&acc);
            }
            while i2 < inner {
                row[i2] = exp1(row[i2] - mx);
                sum += row[i2];
                i2 += 1;
            }
            // 除法拆成乘 1/sum（与基准一致：inv = 1.f/sum 然后逐元素乘）
            let inv = 1.0f32 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    });
}

use crate::par;

/// 任意 axis 的 softmax 通用路径（`softmax_axis` 的非末维分支）。
///
/// 段布局：`seg(m) = t[(o·mid + m)·inner + i]`，m 是归约维。
/// core 是 `forbid(unsafe_code)` 的，所以这条带裸指针并行路径住在 kernels。
/// ★  原版这里用 libm exp；我们用 `exp1`（同一多项式），低 位差异
/// ≤1 ulp——见模块头「已声明的偏差」。末维 softmax 别走这条，
/// 用 [`softmax_last_dim`]（有向量内核）。
pub fn softmax_axis_generic(t: &mut [f32], outer: i64, mid: i64, inner: i64) {
    let tp = par::SyncPtr::new(t.as_mut_ptr());
    par::parallel_for(outer.min(1 << 30) as usize, 1, |o0, o1| {
        for o in o0 as i64..o1 as i64 {
            for i in 0..inner {
                // SAFETY: 每段跨 mid、步长 inner，与其他并行块不相交；
                // 范围由 outer×mid×inner == t.len() 保证。
                unsafe {
                    let base = tp.get().offset(((o * mid) * inner + i) as isize);
                    let mut mx = *base;
                    for m in 1..mid {
                        let v = *base.offset((m * inner) as isize);
                        if v > mx {
                            mx = v;
                        }
                    }
                    let mut sum = 0f32;
                    for m in 0..mid {
                        let p = base.offset((m * inner) as isize);
                        let ev = exp1(*p - mx);
                        *p = ev;
                        sum += ev;
                    }
                    let inv = 1.0f32 / sum;
                    for m in 0..mid {
                        let p = base.offset((m * inner) as isize);
                        *p *= inv;
                    }
                }
            }
        }
    });
}
/// `t = sqrt(t)`，就地。IEEE 精确（硬件指令），与 基准的 std::sqrt 逐位同。
/// 放 kernels：core forbid(unsafe)，而并行就地遍历这里需要 SyncPtr。
pub fn sqrt_inplace(t: &mut [f32]) {
    let tp = par::SyncPtr::new(t.as_mut_ptr());
    let n = t.len();
    par::parallel_for(n, 256, |b, e| {
        // SAFETY: 区间 [b, e) 与其他并行块不相交。
        let seg = unsafe { tp.offset(b).slice(e - b) };
        for v in seg.iter_mut() {
            *v = v.sqrt();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 多项式近似的精度与形态。
    ///
    /// erf 的解析误差界是 |err| ≤ 1.5e-7（A&S 7.1.26），但 f32 中间量舍入
    /// 再叠 1~2 ulp（中段值在 1 附近，ulp≈6e-8），实测总绝对误差到 ~2.7e-7
    /// ——容差放 5e-7。这个测试同时是系数抄录的守卫：抄错一位会差出
    /// 数量级。exp 与 f64 参考比相对误差。
    #[test]
    fn erf_exp_polynomial_accuracy() {
        // erf 的已知值（双精度参考）
        let known: &[(f32, f64)] = &[
            (0.0, 0.0),
            (0.1, 0.112_462_916_018_284_9),
            (0.25, 0.276_326_390_168_236_9),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (1.5, 0.966_105_146_475_310_8),
            (2.0, 0.995_322_265_018_952_7),
            (3.0, 0.999_977_909_503_001_4),
            (4.0, 0.999_999_984_582_742_1),
            (6.0, 0.999_999_999_998_462_6),
        ];
        for &(x, r) in known {
            let got = erf1(x);
            assert!((got as f64 - r).abs() < 5e-7, "erf({x}) = {got}, ref {r}");
        }
        // exp：f64 参考。⚠ 误差是**系统性**的，三项来源（随 |kf| 线性增长，
        // |x|=68 实测 5.5e-6，|x|=88 上界 ~7.5e-6）：
        // 1. ln2 两段拆分不精确：hi=0.693147182 + lo=-2.98e-8（=2^-25，不是
        //    这个 hi 对应的余项）拼起来与 ln2 差 ~2.8e-8；
        // 2. kf·hi 是 f32 乘法，自身半个 ulp 的舍入（kf≈100 时 ~3e-6）——
        //    hi 是满精度 f32，kf·hi 并不精确（Cody-Waite 会选少位数的 hi）；
        // 3. 多项式截断 + Horner 舍入 ~1e-7。
        // ★ 照抄不改：修掉任何一项都会破坏与 基准的逐位一致。
        // 容差按最坏情况设 1e-5。
        let mut x = -80.0f32;
        while x <= 80.0 {
            let got = exp1(x);
            let refv = (x as f64).exp();
            let rel = ((got as f64) - refv).abs() / refv.max(1e-30);
            assert!(rel < 1e-5, "exp({x}) = {got}, rel {rel}");
            x += 0.07;
        }
        // 截断边界：±88 饱和行为
        assert_eq!(exp1(100.0), exp1(88.0));
        assert_eq!(exp1(-100.0), exp1(-88.0));
        // erf 奇函数与符号还原（位级：erf(-x) 与 -erf(x) 逐位相同）
        for v in [0.5f32, -1.25, 3.75, -5.5, 0.0] {
            assert_eq!(erf1(-v).to_bits(), (-erf1(v)).to_bits(), "erf odd at {v}");
        }
    }
}
