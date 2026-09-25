//! AVX2 ↔ 标量逐位对拍。
//!
//! 标量版按 AVX2 内核的表达式树编写（同样的 `mul_add` 序列、同样的
//! round-ties-even、hsum 的结合顺序），所以两版的输出必须**逐位相同**——
//! 不允许多一个 ulp。任何不一致都是某一侧转写出错。

#![allow(clippy::approx_constant, clippy::type_complexity)]

use qppocr_kernels::activation::Activation;
use qppocr_kernels::buf::F32Buf;
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

/// 激活类算子的 AVX2 ↔ 标量逐位对拍。
#[test]
fn activation_bitexact() {
    use qppocr_kernels::activation::{
        clip_inplace, erf_inplace, gelu_inplace, hardsigmoid, relu_inplace, sigmoid_tensor,
        softmax_last_dim,
    };
    let mut rng = Rng::new();
    // 覆盖 8 的倍数与余数尾巴
    for n in [8usize, 13, 64, 4097] {
        let mut x = vec![0f32; n];
        rng.fill(&mut x);
        // 拉宽值域让 erf/exp 的多项式路径吃满
        for (i, v) in x.iter_mut().enumerate() {
            *v *= if i % 3 == 0 { 40.0 } else { 1.0 };
        }

        // 直接两遍跑（force_backend 切换）
        macro_rules! cmp_inplace {
            ($name:expr, $f:expr) => {{
                let mut t1 = x.clone();
                let mut t2 = x.clone();
                force_backend(Some(Backend::Scalar));
                $f(&mut t1);
                force_backend(Some(Backend::Avx2));
                $f(&mut t2);
                force_backend(None);
                assert_bits_eq($name, &t1, &t2);
            }};
        }
        cmp_inplace!("relu", |t: &mut [f32]| relu_inplace(t));
        cmp_inplace!("gelu", |t: &mut [f32]| gelu_inplace(t, 1.4142135, 1.0, 0.5));
        cmp_inplace!("erf", |t: &mut [f32]| erf_inplace(t));
        cmp_inplace!("clip", |t: &mut [f32]| clip_inplace(t, -0.3, 0.7));

        // sigmoid / hardsigmoid（拷贝语义）
        let mut y1 = vec![0f32; n];
        let mut y2 = vec![0f32; n];
        force_backend(Some(Backend::Scalar));
        sigmoid_tensor(&x, &mut y1);
        force_backend(Some(Backend::Avx2));
        sigmoid_tensor(&x, &mut y2);
        force_backend(None);
        assert_bits_eq("sigmoid", &y1, &y2);

        force_backend(Some(Backend::Scalar));
        hardsigmoid(&x, 1.0 / 6.0, 0.5, &mut y1);
        force_backend(Some(Backend::Avx2));
        hardsigmoid(&x, 1.0 / 6.0, 0.5, &mut y2);
        force_backend(None);
        assert_bits_eq("hardsigmoid", &y1, &y2);

        // softmax（行宽覆盖 <8 与 >8）
        for inner in [4usize, 13, 64] {
            if n % inner != 0 {
                continue;
            }
            let mut t1 = x.clone();
            let mut t2 = x.clone();
            force_backend(Some(Backend::Scalar));
            softmax_last_dim(&mut t1, inner);
            force_backend(Some(Backend::Avx2));
            softmax_last_dim(&mut t2, inner);
            force_backend(None);
            assert_bits_eq(&format!("softmax inner={inner}"), &t1, &t2);
        }
    }
}

/// 二元算子（就地各路径）的 AVX2 ↔ 标量逐位对拍。
#[test]
fn binary_inplace_bitexact() {
    use qppocr_kernels::elementwise::BinOp;
    use qppocr_kernels::elementwise::binary_op_inplace;
    let mut rng = Rng::new();
    let cases: &[(Vec<i64>, Vec<i64>, BinOp)] = &[
        (vec![1, 16, 8, 8], vec![1, 16, 8, 8], BinOp::Add), // 同形平坦
        (vec![1, 16, 8, 8], vec![1, 16, 1, 1], BinOp::Mul), // 常量 run
        (vec![1, 8, 4, 13], vec![13], BinOp::Add),          // 稠密 run + 13 余数
        (vec![2, 3, 40], vec![3, 1], BinOp::Add),           // k==0 走法（纯标量）
    ];
    for (a_shape, b_shape, op) in cases {
        let mut a = vec![0f32; a_shape.iter().product::<i64>() as usize];
        rng.fill(&mut a);
        let mut b = vec![0f32; b_shape.iter().product::<i64>() as usize];
        rng.fill(&mut b);
        let mut a2 = a.clone();
        force_backend(Some(Backend::Scalar));
        binary_op_inplace(&mut a, a_shape, &b, b_shape, *op);
        force_backend(Some(Backend::Avx2));
        binary_op_inplace(&mut a2, a_shape, &b, b_shape, *op);
        force_backend(None);
        assert_bits_eq(
            &format!("inplace {:?} {:?} {:?}", a_shape, b_shape, op),
            &a,
            &a2,
        );
    }
}

/// conv 三分支 / convT / pool2x2 快路径的 AVX2 ↔ 标量逐位对拍。
#[test]
fn conv_bitexact() {
    #![allow(clippy::approx_constant, clippy::type_complexity)]

    use qppocr_kernels::activation::Activation;
    use qppocr_kernels::conv::{ConvParams, conv2d, convtranspose2d};
    use qppocr_kernels::pool2d::pool2d;
    let mut rng = Rng::new();
    let cases: &[(
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    )] = &[
        // (n, c, m, h, w, kh, kw, sh, sw, ph, pw)
        (1, 16, 24, 32, 32, 1, 1, 1, 1, 0, 0), // 1x1 → sgemm
        (1, 16, 16, 64, 64, 3, 3, 1, 1, 1, 1), // im2col（N%32 尾列）
        (1, 16, 16, 63, 61, 3, 3, 1, 1, 1, 1), // im2col（奇数宽）
        (1, 16, 16, 64, 64, 3, 3, 1, 1, 1, 1), // depthwise 用例在下面单独加
        (1, 3, 4, 10, 11, 3, 3, 1, 2, 1, 1),   // s[1,2]
    ];
    for &(n, c, m, h, w, kh, kw, sh, sw, ph, pw) in cases {
        let mut x = vec![0f32; n * c * h * w];
        rng.fill(&mut x);
        let mut wt = vec![0f32; m * c * kh * kw];
        rng.fill(&mut wt);
        let bias: Vec<f32> = (0..m).map(|_| rng.next_f32()).collect();
        let p = ConvParams {
            sh,
            sw,
            ph,
            pw,
            peh: ph,
            pew: pw,
            dh: 1,
            dw: 1,
            group: 1,
        };
        let mut y1 = F32Buf::new();
        let mut y2 = F32Buf::new();
        force_backend(Some(Backend::Scalar));
        conv2d(
            &x,
            &[n as i64, c as i64, h as i64, w as i64],
            &wt,
            &[m as i64, c as i64, kh as i64, kw as i64],
            Some(&bias),
            &p,
            &Activation::default(),
            &mut y1,
        );
        force_backend(Some(Backend::Avx2));
        conv2d(
            &x,
            &[n as i64, c as i64, h as i64, w as i64],
            &wt,
            &[m as i64, c as i64, kh as i64, kw as i64],
            Some(&bias),
            &p,
            &Activation::default(),
            &mut y2,
        );
        force_backend(None);
        assert_bits_eq(&format!("conv2d {m}->{m} k{kh}x{kw} s{sh}x{sw}"), &y1, &y2);
    }

    // depthwise（m == c）
    {
        let (n, c, h, w, kh, kw) = (1usize, 16, 64, 64, 3, 3);
        let mut x = vec![0f32; n * c * h * w];
        rng.fill(&mut x);
        let mut wt = vec![0f32; c * kh * kw];
        rng.fill(&mut wt);
        let bias: Vec<f32> = (0..c).map(|_| rng.next_f32()).collect();
        let p = ConvParams {
            sh: 1,
            sw: 1,
            ph: 1,
            pw: 1,
            peh: 1,
            pew: 1,
            dh: 1,
            dw: 1,
            group: c,
        };
        let mut y1 = F32Buf::new();
        let mut y2 = F32Buf::new();
        force_backend(Some(Backend::Scalar));
        conv2d(
            &x,
            &[n as i64, c as i64, h as i64, w as i64],
            &wt,
            &[c as i64, 1, kh as i64, kw as i64],
            Some(&bias),
            &p,
            &Activation::default(),
            &mut y1,
        );
        force_backend(Some(Backend::Avx2));
        conv2d(
            &x,
            &[n as i64, c as i64, h as i64, w as i64],
            &wt,
            &[c as i64, 1, kh as i64, kw as i64],
            Some(&bias),
            &p,
            &Activation::default(),
            &mut y2,
        );
        force_backend(None);
        assert_bits_eq("depthwise 3x3 s1", &y1, &y2);
    }

    // convT 2x2 s2（W 含 8 的倍数与余数）
    for w in [16usize, 21] {
        let (n, c, h, m) = (1usize, 7, 5, 3);
        let mut x = vec![0f32; n * c * h * w];
        rng.fill(&mut x);
        let mut wt = vec![0f32; c * m * 4];
        rng.fill(&mut wt);
        let mut y1 = F32Buf::new();
        let mut y2 = F32Buf::new();
        force_backend(Some(Backend::Scalar));
        convtranspose2d(
            &x,
            &[n as i64, c as i64, h as i64, w as i64],
            &wt,
            &[c as i64, m as i64, 2, 2],
            2,
            2,
            0,
            0,
            &mut y1,
        );
        force_backend(Some(Backend::Avx2));
        convtranspose2d(
            &x,
            &[n as i64, c as i64, h as i64, w as i64],
            &wt,
            &[c as i64, m as i64, 2, 2],
            2,
            2,
            0,
            0,
            &mut y2,
        );
        force_backend(None);
        assert_bits_eq(&format!("convT w={w}"), &y1, &y2);
    }

    // pool 2x2 s1 SAME_UPPER（快路径）
    {
        let (n, c, h, w) = (1usize, 3, 20, 30);
        let mut x = vec![0f32; n * c * h * w];
        rng.fill(&mut x);
        let mut y1 = F32Buf::new();
        let mut y2 = F32Buf::new();
        force_backend(Some(Backend::Scalar));
        pool2d(&x, n, c, h, w, 2, 2, 1, 1, 0, 0, 1, 1, true, &mut y1);
        force_backend(Some(Backend::Avx2));
        pool2d(&x, n, c, h, w, 2, 2, 1, 1, 0, 0, 1, 1, true, &mut y2);
        force_backend(None);
        assert_bits_eq("pool 2x2 s1", &y1, &y2);
    }
}
