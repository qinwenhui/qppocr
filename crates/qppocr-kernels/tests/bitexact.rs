//! AVX2 ↔ 标量逐位对拍（DESIGN.md §9 阶段 1 判据二）。
//!
//! 标量版按 AVX2 内核的表达式树编写（同样的 `mul_add` 序列、同样的
//! round-ties-even、hsum 的结合顺序），所以两版的输出必须**逐位相同**——
//! 不允许多一个 ulp。任何不一致都是某一侧转写出错。

use qppocr_kernels::activation::Activation;
use qppocr_kernels::gemm::{sgemm, sgemm_serial};
use qppocr_kernels::{Backend, force_backend};

struct Rng(u64);
impl Rng {
    fn new() -> Self {
        Rng(0x9E37_79B9_7F4A_7C15)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (((v >> 40) as i64 - (1 << 23)) as f64 / (1i64 << 23) as f64) as f32
    }
    fn fill(&mut self, v: &mut [f32]) {
        for e in v.iter_mut() {
            *e = self.next_f32();
        }
    }
}

fn assert_bits_eq(name: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{name}: size");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{name}: bit mismatch at {i}: scalar={x} avx2={y}"
        );
    }
}

/// 两侧后端各跑一遍 sgemm，比对位。
#[test]
fn sgemm_bitexact() {
    // 形状覆盖：M%4 尾行、N%32 尾列/整面板、窄 N（m-rows 路径）、K 大小
    let shapes: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (8, 8, 8),
        (17, 713, 77), // M%4 尾行 + N%32 尾列（两条位级分支同时踩）
        (5, 37, 13),
        (64, 64, 576),   // 面板路径
        (3136, 64, 576), // det 的 1x1 形状
        (1024, 256, 256),
        (512, 96, 129), // N<128：m-rows 路径候选
        (23, 713, 384), // det 颈部形状（NP=23）
    ];
    let mut rng = Rng::new();
    for &(m, n, k) in shapes {
        let mut a = vec![0f32; m * k];
        rng.fill(&mut a);
        let mut b = vec![0f32; k * n];
        rng.fill(&mut b);
        for use_bias in [false, true] {
            let bias: Vec<f32> = (0..m).map(|_| rng.next_f32()).collect();
            let bias = if use_bias { Some(&bias[..]) } else { None };

            for serial in [false, true] {
                let mut c_scalar = vec![0f32; m * n];
                let mut c_avx2 = vec![0f32; m * n];
                force_backend(Some(Backend::Scalar));
                if serial {
                    sgemm_serial(
                        &a,
                        &b,
                        &mut c_scalar,
                        m,
                        n,
                        k,
                        n,
                        bias,
                        &Activation::default(),
                    );
                } else {
                    sgemm(
                        &a,
                        &b,
                        &mut c_scalar,
                        m,
                        n,
                        k,
                        n,
                        bias,
                        &Activation::default(),
                    );
                }
                force_backend(Some(Backend::Avx2));
                if serial {
                    sgemm_serial(
                        &a,
                        &b,
                        &mut c_avx2,
                        m,
                        n,
                        k,
                        n,
                        bias,
                        &Activation::default(),
                    );
                } else {
                    sgemm(
                        &a,
                        &b,
                        &mut c_avx2,
                        m,
                        n,
                        k,
                        n,
                        bias,
                        &Activation::default(),
                    );
                }
                force_backend(None);
                assert_bits_eq(
                    &format!("sgemm m={m} n={n} k={k} bias={use_bias} serial={serial}"),
                    &c_scalar,
                    &c_avx2,
                );
            }
        }
    }
}

/// 融合 GELU 的位级一致（apply_act 路径——当前只有标量版，此测试为
/// 后续接入向量版后立即生效的守卫）。
#[test]
fn sgemm_gelu_bitexact() {
    let (m, n, k) = (64usize, 100, 77);
    let mut rng = Rng::new();
    let mut a = vec![0f32; m * k];
    rng.fill(&mut a);
    let mut b = vec![0f32; k * n];
    rng.fill(&mut b);
    let bias: Vec<f32> = (0..m).map(|_| rng.next_f32()).collect();
    let act = Activation::gelu(1.4142135, 1.0, 0.5);

    let mut c1 = vec![0f32; m * n];
    let mut c2 = vec![0f32; m * n];
    force_backend(Some(Backend::Scalar));
    sgemm(&a, &b, &mut c1, m, n, k, n, Some(&bias), &act);
    force_backend(Some(Backend::Avx2));
    sgemm(&a, &b, &mut c2, m, n, k, n, Some(&bias), &act);
    force_backend(None);
    assert_bits_eq("sgemm+gelu", &c1, &c2);
}
