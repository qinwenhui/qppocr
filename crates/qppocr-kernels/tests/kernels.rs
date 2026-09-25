//! 内核单元测试：形状随机的用例集，对拍朴素参考实现。
//!
//! 这是公开的、自包含的测试——不需要模型、不需要语料。
//! 判据：相对误差 < 1e-5（长 f32 归约的绝对误差会超过 1e-4，
//! 但相对量级是精确的）。
//!
//! 补充覆盖：softmax / pool / resize /
//! transpose / slice / concat / matmul / reduce_mean / gelu / sigmoid /
//! hardsigmoid / convtranspose 的朴素参考测试。

#![allow(
    clippy::needless_range_loop,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::approx_constant // 参考/被测双方的常数都照抄  字面值
)]

use qppocr_kernels::activation::*;
use qppocr_kernels::buf::F32Buf;
use qppocr_kernels::conv::{ConvParams, conv2d, convtranspose2d};
use qppocr_kernels::elementwise::{BinOp, binary_can_inplace, binary_op, binary_op_inplace};
use qppocr_kernels::gemm::sgemm;
use qppocr_kernels::pool2d::{global_avg_pool, pool2d};
use qppocr_kernels::resize::{resize_bilinear, resize_nearest};
use qppocr_kernels::shape::*;

// ---------------------------------------------------------------- 基础设施

/// 确定性随机源（数值不同没关系：参考值在测试内算）。
struct Rng(u64);
impl Rng {
    fn new() -> Self {
        Rng(0x243F_6A88_85A3_08D3)
    }
    fn next_f32(&mut self) -> f32 {
        // xorshift64* → [-1, 1)
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

/// 相对误差检查（ `check` ）。返回 Err 描述首个超差位置。
fn check(name: &str, got: &[f32], ref_: &[f32]) {
    assert_eq!(
        got.len(),
        ref_.len(),
        "{name}: size {} vs {}",
        got.len(),
        ref_.len()
    );
    let mut mx = 0f64;
    let mut scale = 0f64;
    let mut arg = 0usize;
    for (i, (&a, &b)) in got.iter().zip(ref_).enumerate() {
        let d = (a as f64 - b as f64).abs();
        scale = scale.max(b.abs() as f64);
        if d > mx {
            mx = d;
            arg = i;
        }
    }
    let rel = mx / scale.max(1e-6);
    assert!(
        rel < 1e-5,
        "{name}: max diff {mx:.3e} at {arg}: ours={:.6} ref={:.6} (rel {rel:.1e})",
        got[arg],
        ref_[arg]
    );
}

fn bits_eq(name: &str, got: &[f32], ref_: &[f32]) {
    assert_eq!(got.len(), ref_.len(), "{name}: size");
    for (i, (&a, &b)) in got.iter().zip(ref_).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{name}: bit mismatch at {i}: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------- conv2d

/// 朴素 NCHW conv 参考（f64 累加）
#[allow(clippy::too_many_arguments)]
fn ref_conv(
    x: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    w_: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    group: usize,
    y: &mut [f32],
    oh: usize,
    ow: usize,
) {
    let mg = m / group;
    let cg = c / group;
    for bn in 0..n {
        for mm in 0..m {
            for oy in 0..oh {
                for oxx in 0..ow {
                    let g = mm / mg;
                    let mut acc = bias.map(|b| b[mm] as f64).unwrap_or(0.0);
                    for ch in 0..cg {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy as isize * sh as isize - ph as isize + ky as isize;
                                let ix = oxx as isize * sw as isize - pw as isize + kx as isize;
                                if iy < 0 || iy as usize >= h || ix < 0 || ix as usize >= w {
                                    continue;
                                }
                                let ci = g * cg + ch;
                                let xv =
                                    x[((bn * c + ci) * h + iy as usize) * w + ix as usize] as f64;
                                let wv = w_[((mm * cg + ch) * kh + ky) * kw + kx] as f64;
                                acc += xv * wv;
                            }
                        }
                    }
                    y[((bn * m + mm) * oh + oy) * ow + oxx] = acc as f32;
                }
            }
        }
    }
}

struct Case {
    name: &'static str,
    n: usize,
    c: usize,
    m: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    group: usize,
    bias: bool,
}

#[test]
fn conv2d_kernel_tests() {
    let cases = [
        Case {
            name: "1x1 conv (gemm path)",
            n: 1,
            c: 16,
            m: 24,
            h: 32,
            w: 32,
            kh: 1,
            kw: 1,
            sh: 1,
            sw: 1,
            ph: 0,
            pw: 0,
            group: 1,
            bias: true,
        },
        Case {
            name: "3x3 s1 p1 (im2col)",
            n: 1,
            c: 3,
            m: 16,
            h: 64,
            w: 64,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            ph: 1,
            pw: 1,
            group: 1,
            bias: true,
        },
        Case {
            name: "3x3 s2 p1 (im2col)",
            n: 1,
            c: 3,
            m: 16,
            h: 65,
            w: 65,
            kh: 3,
            kw: 3,
            sh: 2,
            sw: 2,
            ph: 1,
            pw: 1,
            group: 1,
            bias: true,
        },
        Case {
            name: "5x5 s1 p2 (im2col)",
            n: 1,
            c: 8,
            m: 16,
            h: 33,
            w: 33,
            kh: 5,
            kw: 5,
            sh: 1,
            sw: 1,
            ph: 2,
            pw: 2,
            group: 1,
            bias: false,
        },
        Case {
            name: "5x5 s2 p2 (im2col)",
            n: 1,
            c: 8,
            m: 16,
            h: 64,
            w: 64,
            kh: 5,
            kw: 5,
            sh: 2,
            sw: 2,
            ph: 2,
            pw: 2,
            group: 1,
            bias: true,
        },
        Case {
            name: "depthwise 3x3 s1 p1",
            n: 1,
            c: 16,
            m: 16,
            h: 64,
            w: 64,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            ph: 1,
            pw: 1,
            group: 16,
            bias: true,
        },
        Case {
            // ★ 回归：1×1 输入的 depthwise。ph=pw=1 时输出恰好 1×1（合法），
            //   极小输入走的是同一份快路径代码。真正带 kx 越界的场景见下方
            //   tiny_depthwise_end_pad 测试（Case 表表达不了 end padding）。
            name: "depthwise 3x3 s1 p1 on 1x1 (tiny input)",
            n: 1,
            c: 4,
            m: 4,
            h: 1,
            w: 1,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            ph: 1,
            pw: 1,
            group: 4,
            bias: false,
        },
        Case {
            name: "depthwise 3x3 s2 p1",
            n: 1,
            c: 32,
            m: 32,
            h: 65,
            w: 65,
            kh: 3,
            kw: 3,
            sh: 2,
            sw: 2,
            ph: 1,
            pw: 1,
            group: 32,
            bias: true,
        },
        Case {
            name: "depthwise 5x5 s1 p2",
            n: 1,
            c: 24,
            m: 24,
            h: 40,
            w: 40,
            kh: 5,
            kw: 5,
            sh: 1,
            sw: 1,
            ph: 2,
            pw: 2,
            group: 24,
            bias: true,
        },
        Case {
            name: "depthwise 5x5 s2 p2",
            n: 1,
            c: 24,
            m: 24,
            h: 40,
            w: 40,
            kh: 5,
            kw: 5,
            sh: 2,
            sw: 2,
            ph: 2,
            pw: 2,
            group: 24,
            bias: false,
        },
        Case {
            name: "depthwise 3x3 s[1,2] p1",
            n: 1,
            c: 32,
            m: 32,
            h: 40,
            w: 40,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 2,
            ph: 1,
            pw: 1,
            group: 32,
            bias: true,
        },
        Case {
            name: "depthwise 3x3 s[2,1] p1",
            n: 1,
            c: 32,
            m: 32,
            h: 40,
            w: 40,
            kh: 3,
            kw: 3,
            sh: 2,
            sw: 1,
            ph: 1,
            pw: 1,
            group: 32,
            bias: true,
        },
        Case {
            name: "depthwise 5x5 s[1,2] p2",
            n: 1,
            c: 24,
            m: 24,
            h: 40,
            w: 40,
            kh: 5,
            kw: 5,
            sh: 1,
            sw: 2,
            ph: 2,
            pw: 2,
            group: 24,
            bias: true,
        },
        Case {
            name: "3x3 s[1,2] p1 (im2col)",
            n: 1,
            c: 8,
            m: 16,
            h: 40,
            w: 40,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 2,
            ph: 1,
            pw: 1,
            group: 1,
            bias: true,
        },
        Case {
            name: "3x3 s[2,1] p1 (im2col)",
            n: 1,
            c: 8,
            m: 16,
            h: 40,
            w: 40,
            kh: 3,
            kw: 3,
            sh: 2,
            sw: 1,
            ph: 1,
            pw: 1,
            group: 1,
            bias: true,
        },
        Case {
            name: "depthwise 7x7 s1 p3",
            n: 1,
            c: 32,
            m: 32,
            h: 40,
            w: 40,
            kh: 7,
            kw: 7,
            sh: 1,
            sw: 1,
            ph: 3,
            pw: 3,
            group: 32,
            bias: true,
        },
        Case {
            name: "depthwise 7x7 s2 p3",
            n: 1,
            c: 32,
            m: 32,
            h: 40,
            w: 40,
            kh: 7,
            kw: 7,
            sh: 2,
            sw: 2,
            ph: 3,
            pw: 3,
            group: 32,
            bias: true,
        },
        Case {
            name: "grouped g=4 (im2col)",
            n: 1,
            c: 16,
            m: 24,
            h: 32,
            w: 32,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            ph: 1,
            pw: 1,
            group: 4,
            bias: true,
        },
        Case {
            name: "grouped g=2 5x5 s2 (im2col)",
            n: 1,
            c: 6,
            m: 12,
            h: 33,
            w: 33,
            kh: 5,
            kw: 5,
            sh: 2,
            sw: 2,
            ph: 2,
            pw: 2,
            group: 2,
            bias: true,
        },
        Case {
            name: "batch=2 3x3 (im2col)",
            n: 2,
            c: 8,
            m: 16,
            h: 24,
            w: 24,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            ph: 1,
            pw: 1,
            group: 1,
            bias: true,
        },
        // 非 4 倍数的 M：M%4 尾行（位级分支覆盖）
        Case {
            name: "M=17 N=713 (tail rows)",
            n: 1,
            c: 8,
            m: 17,
            h: 30,
            w: 24,
            kh: 1,
            kw: 1,
            sh: 1,
            sw: 1,
            ph: 0,
            pw: 0,
            group: 1,
            bias: true,
        },
        Case {
            name: "M=3 3x3 (all tail)",
            n: 1,
            c: 4,
            m: 3,
            h: 16,
            w: 16,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            ph: 1,
            pw: 1,
            group: 1,
            bias: true,
        },
    ];
    let mut rng = Rng::new();
    for c in cases {
        let mut x = vec![0f32; c.n * c.c * c.h * c.w];
        rng.fill(&mut x);
        let mut wt = vec![0f32; c.m * (c.c / c.group) * c.kh * c.kw];
        rng.fill(&mut wt);
        let mut b = vec![0f32; c.m];
        rng.fill(&mut b);
        let oh = (c.h + 2 * c.ph - c.kh) / c.sh + 1;
        let ow = (c.w + 2 * c.pw - c.kw) / c.sw + 1;
        let mut ref_ = vec![0f32; c.n * c.m * oh * ow];
        ref_conv(
            &x,
            c.n,
            c.c,
            c.h,
            c.w,
            &wt,
            if c.bias { Some(&b) } else { None },
            c.m,
            c.kh,
            c.kw,
            c.sh,
            c.sw,
            c.ph,
            c.pw,
            c.group,
            &mut ref_,
            oh,
            ow,
        );

        let params = ConvParams {
            sh: c.sh,
            sw: c.sw,
            ph: c.ph,
            pw: c.pw,
            peh: c.ph,
            pew: c.pw,
            dh: 1,
            dw: 1,
            group: c.group,
        };
        let mut y = F32Buf::new();
        let out_shape = conv2d(
            &x,
            &[c.n as i64, c.c as i64, c.h as i64, c.w as i64],
            &wt,
            &[c.m as i64, (c.c / c.group) as i64, c.kh as i64, c.kw as i64],
            if c.bias { Some(&b) } else { None },
            &params,
            &Activation::default(),
            &mut y,
        );
        assert_eq!(
            out_shape,
            [c.n as i64, c.m as i64, oh as i64, ow as i64],
            "{}: shape",
            c.name
        );
        check(c.name, &y.to_vec(), &ref_);
    }
}

/// ★ 回归：1×1 输入 + 尾部 padding 的 depthwise —— `kx > wdim+pw` 的列输入
///   完全在图外，快路径的 `ox1 = ow.min(wdim + pw - kx)` 若不饱和会在 usize
///   上下溢 panic。真实事故（2026-09-25）：1688 防盗链占位图 spaceball.gif
///   是 1×1 透明 GIF，det 网络在它上面直接把整批图片识别打断
///   （生产 18 张图全部失败）。Case 表表达不了 end padding，这里单独构造：
///   `w=1, pw=0, pew=2, kw=3, sw=1` → ow=(1+0+2-3)+1=1 合法，而 kx=2 时
///   `wdim+pw-kx = -1`。
#[test]
fn tiny_depthwise_end_pad_tests() {
    // 固定数字，diff 直接暴露内核实际算了哪些 tap：
    // x[0]=1.0；bias[och]=och；wt 全部 0，唯独每个核的 (0,0) tap = 10.0+och。
    // 期望输出 = bias + (10+och)*1 —— 任何别的 tap 被算进来都会立刻显形。
    let (n, c, h, w) = (1usize, 4, 1, 1);
    let (m, kh, kw) = (4usize, 3, 3);
    let x = vec![1.0f32; n * c * h * w];
    let mut wt = vec![0f32; m * kh * kw];
    for och in 0..m {
        wt[och * kh * kw] = 10.0 + och as f32;
    }
    let bias: Vec<f32> = (0..m).map(|i| i as f32).collect();

    let params = ConvParams {
        sh: 1,
        sw: 1,
        ph: 0,
        pw: 0,
        peh: 2,
        pew: 2,
        dh: 1,
        dw: 1,
        group: c,
    };
    let mut y = F32Buf::new();
    let out_shape = conv2d(
        &x,
        &[n as i64, c as i64, h as i64, w as i64],
        &wt,
        &[m as i64, 1, kh as i64, kw as i64],
        Some(&bias),
        &params,
        &Activation::default(),
        &mut y,
    );
    assert_eq!(out_shape, [n as i64, m as i64, 1, 1], "shape");

    let ref_: Vec<f32> = (0..m).map(|och| och as f32 + 10.0 + och as f32).collect();
    check("tiny depthwise end pad", &y.to_vec(), &ref_);
}

// ---------------------------------------------------------------- sgemm

#[test]
fn sgemm_tests() {
    let mut rng = Rng::new();
    let shapes: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (64, 64, 64),
        (1, 500, 300),
        (17, 33, 129),
        (256, 96, 384),
        (1, 368, 368 * 9),
        // M%4 尾行 + N%32 尾列（bias 位级分支）
        (17, 713, 77),
        (5, 37, 13),
    ];
    for &(m, n, k) in shapes {
        let mut a = vec![0f32; m * k];
        rng.fill(&mut a);
        let mut b = vec![0f32; k * n];
        rng.fill(&mut b);
        let bias: Vec<f32> = (0..m).map(|_| rng.next_f32()).collect();
        let mut cbuf = vec![0f32; m * n];
        let mut ref_ = vec![0f32; m * n];
        for mm in 0..m {
            for nn in 0..n {
                let mut s = 0f64;
                for kk in 0..k {
                    s += a[mm * k + kk] as f64 * b[kk * n + nn] as f64;
                }
                ref_[mm * n + nn] = (s + bias[mm] as f64) as f32;
            }
        }
        sgemm(
            &a,
            &b,
            &mut cbuf,
            m,
            n,
            k,
            n,
            Some(&bias),
            &Activation::default(),
        );
        check(&format!("gemm {m}x{n}x{k}"), &cbuf, &ref_);
    }
}

// ---------------------------------------------------------------- batchnorm

#[test]
fn batchnorm_tests() {
    let mut rng = Rng::new();
    for n in [1usize, 2, 5] {
        let (c, h, w) = (8usize, 6, 7);
        let mut x = vec![0f32; n * c * h * w];
        rng.fill(&mut x);
        let mut mk = || {
            let mut v = vec![0f32; c];
            for e in v.iter_mut() {
                *e = rng.next_f32() * 0.5 + 1.0;
            }
            v
        };
        let (sc, bi, me, va) = (mk(), mk(), mk(), mk());
        let mut y = F32Buf::new();
        batchnorm(
            &x,
            &[n as i64, c as i64, h as i64, w as i64],
            &sc,
            &bi,
            &me,
            &va,
            1e-5,
            &mut y,
        );
        let mut ref_ = vec![0f32; x.len()];
        let plane = (h * w) as i64;
        for bn in 0..n {
            for ch in 0..c {
                let a = sc[ch] / (va[ch] + 1e-5).sqrt();
                let b = bi[ch] - me[ch] * a;
                for i in 0..plane {
                    let idx = ((bn * c + ch) as i64 * plane + i) as usize;
                    ref_[idx] = x[idx] * a + b;
                }
            }
        }
        check(&format!("batchnorm N={n} C={c}"), &y.to_vec(), &ref_);
    }
}

// ---------------------------------------------------------------- binary_op 广播

struct BCase {
    name: &'static str,
    a: Vec<i64>,
    b: Vec<i64>,
    op: BinOp,
}

/// 朴素右对齐广播参考。
fn naive_broadcast(a: &[f32], a_shape: &[i64], b: &[f32], b_shape: &[i64], op: BinOp) -> Vec<f32> {
    let r = a_shape.len().max(b_shape.len());
    let fdim = |s: &[i64], i: usize| -> i64 {
        let pad = r - s.len();
        if i >= pad { s[i - pad] } else { 1 }
    };
    let mut fs = vec![1i64; r];
    for i in 0..r {
        fs[i] = fdim(a_shape, i).max(fdim(b_shape, i));
    }
    let fstride = |s: &[i64]| -> Vec<i64> {
        let mut st = vec![0i64; r];
        let mut f = vec![1i64; r];
        for i in 0..s.len() {
            f[r - s.len() + i] = s[i];
        }
        let mut acc = 1i64;
        for i in (0..r).rev() {
            st[i] = if f[i] == 1 { 0 } else { acc };
            acc *= f[i];
        }
        st
    };
    let (sa, sb) = (fstride(a_shape), fstride(b_shape));
    let total: i64 = fs.iter().product();
    let mut out = vec![0f32; total as usize];
    let mut idx = vec![0i64; r];
    for (lin, o) in out.iter_mut().enumerate() {
        let (mut ia, mut ib) = (0i64, 0i64);
        for i in 0..r {
            ia += idx[i] * sa[i];
            ib += idx[i] * sb[i];
        }
        *o = op.apply(a[ia as usize], b[ib as usize]);
        // C 序 odometer（最外维最慢）
        for i in (0..r).rev() {
            idx[i] += 1;
            if idx[i] < fs[i] {
                break;
            }
            idx[i] = 0;
        }
        let _ = lin;
    }
    out
}

#[test]
fn binary_op_broadcast_tests() {
    let mut rng = Rng::new();
    let cases = [
        BCase {
            name: "[1] x [1,16,368,368] mul",
            a: vec![1],
            b: vec![1, 16, 368, 368],
            op: BinOp::Mul,
        },
        BCase {
            name: "rank0 x [1,16,8,8] mul",
            a: vec![],
            b: vec![1, 16, 8, 8],
            op: BinOp::Mul,
        },
        BCase {
            name: "[1,16,1,1] x [1,16,8,8] mul",
            a: vec![1, 16, 1, 1],
            b: vec![1, 16, 8, 8],
            op: BinOp::Mul,
        },
        BCase {
            name: "[16,1,1] x [1,16,8,8] mul",
            a: vec![16, 1, 1],
            b: vec![1, 16, 8, 8],
            op: BinOp::Mul,
        },
        BCase {
            name: "[1,16,8,8] x [1,16,8,8] add",
            a: vec![1, 16, 8, 8],
            b: vec![1, 16, 8, 8],
            op: BinOp::Add,
        },
        BCase {
            name: "[1,16,8,8] / [1,16,8,8]",
            a: vec![1, 16, 8, 8],
            b: vec![1, 16, 8, 8],
            op: BinOp::Div,
        },
        BCase {
            name: "[368,368] + [1,16,368,368]",
            a: vec![368, 368],
            b: vec![1, 16, 368, 368],
            op: BinOp::Add,
        },
        BCase {
            name: "[1] + [1]",
            a: vec![1],
            b: vec![1],
            op: BinOp::Add,
        },
        BCase {
            name: "[1,8,1,1] + [1,8,4,4]",
            a: vec![1, 8, 1, 1],
            b: vec![1, 8, 4, 4],
            op: BinOp::Add,
        },
        // 附加：Pow 与负轴
        BCase {
            name: "[2,3,40] ^ [2,1,1]",
            a: vec![2, 3, 40],
            b: vec![2, 1, 1],
            op: BinOp::Pow,
        },
    ];
    for bc in cases {
        let mut a = vec![0f32; bc.a.iter().product::<i64>() as usize];
        rng.fill(&mut a);
        // 幂的底数取正避免复数域
        if bc.op == BinOp::Pow {
            for v in a.iter_mut() {
                *v = v.abs();
            }
        }
        let mut b = vec![0f32; bc.b.iter().product::<i64>() as usize];
        rng.fill(&mut b);
        if bc.op == BinOp::Pow {
            for v in b.iter_mut() {
                *v = v.abs() * 2.0;
            }
        }
        let (y, ys) = binary_op(&a, &bc.a, &b, &bc.b, bc.op);
        let r = bc.a.len().max(bc.b.len());
        let fdim = |s: &[i64], i: usize| -> i64 {
            let pad = r - s.len();
            if i >= pad { s[i - pad] } else { 1 }
        };
        // fdim 是右对齐的前向维访问器，(0..r).map 已是前向序，不能再 rev
        let want: Vec<i64> = (0..r).map(|i| fdim(&bc.a, i).max(fdim(&bc.b, i))).collect();
        assert_eq!(ys, want, "{}: out shape", bc.name);
        let ref_ = naive_broadcast(&a, &bc.a, &b, &bc.b, bc.op);
        check(bc.name, &y, &ref_);
    }
}

// ---------------------------------------------------------------- binary_op_inplace

struct ICase {
    name: &'static str,
    a: Vec<i64>,
    b: Vec<i64>,
    op: BinOp,
}

#[test]
fn binary_op_inplace_tests() {
    let mut rng = Rng::new();
    let cases = [
        ICase {
            name: "[1,16,8,8] += [1,16,8,8]",
            a: vec![1, 16, 8, 8],
            b: vec![1, 16, 8, 8],
            op: BinOp::Add,
        },
        ICase {
            name: "[1,16,8,8] *= [1,16,8,8]",
            a: vec![1, 16, 8, 8],
            b: vec![1, 16, 8, 8],
            op: BinOp::Mul,
        },
        ICase {
            name: "[1,16,8,8] += [1,16,1,1]",
            a: vec![1, 16, 8, 8],
            b: vec![1, 16, 1, 1],
            op: BinOp::Add,
        },
        ICase {
            name: "[1,16,8,8] *= [1,16,1,1]",
            a: vec![1, 16, 8, 8],
            b: vec![1, 16, 1, 1],
            op: BinOp::Mul,
        },
        ICase {
            name: "[2,3,40,6906] += [6906]",
            a: vec![2, 3, 40, 6906],
            b: vec![6906],
            op: BinOp::Add,
        }, // dense run
        ICase {
            name: "[1,8,4,4] += [4]",
            a: vec![1, 8, 4, 4],
            b: vec![4],
            op: BinOp::Add,
        }, // dense run
        ICase {
            name: "[1,8,4,4] *= [4]",
            a: vec![1, 8, 4, 4],
            b: vec![4],
            op: BinOp::Mul,
        },
        ICase {
            name: "[1,8,4,4] -= [4]",
            a: vec![1, 8, 4, 4],
            b: vec![4],
            op: BinOp::Sub,
        },
        ICase {
            name: "[1,8,4,4] /= [4]",
            a: vec![1, 8, 4, 4],
            b: vec![4],
            op: BinOp::Div,
        },
        ICase {
            name: "[2,3,40] += [3,1]",
            a: vec![2, 3, 40],
            b: vec![3, 1],
            op: BinOp::Add,
        }, // strided walk
        ICase {
            name: "[2,3,40] *= [1,40]",
            a: vec![2, 3, 40],
            b: vec![1, 40],
            op: BinOp::Mul,
        }, // dense run
        ICase {
            name: "[2,3,40] -= [3,40]",
            a: vec![2, 3, 40],
            b: vec![3, 40],
            op: BinOp::Sub,
        }, // k == 0 strided
        ICase {
            name: "[5,7] += [7]",
            a: vec![5, 7],
            b: vec![7],
            op: BinOp::Add,
        },
        ICase {
            name: "[5,7] *= [5,1]",
            a: vec![5, 7],
            b: vec![5, 1],
            op: BinOp::Mul,
        },
    ];
    for ic in cases {
        let mut a = vec![0f32; ic.a.iter().product::<i64>() as usize];
        rng.fill(&mut a);
        let mut b = vec![0f32; ic.b.iter().product::<i64>() as usize];
        rng.fill(&mut b);
        let mut ref_ = a.clone();
        // 朴素参考：C 序走 a，取 b 右对齐下标处的元素
        {
            let r = ic.a.len().max(ic.b.len());
            let fdim = |s: &[i64], i: usize| -> i64 {
                let pad = r - s.len();
                if i >= pad { s[i - pad] } else { 1 }
            };
            let mut fs = vec![1i64; r];
            for i in 0..r {
                fs[i] = fdim(&ic.a, i).max(fdim(&ic.b, i));
            }
            let fstride = |s: &[i64]| -> Vec<i64> {
                let mut st = vec![0i64; r];
                let mut f = vec![1i64; r];
                for i in 0..s.len() {
                    f[r - s.len() + i] = s[i];
                }
                let mut acc = 1i64;
                for i in (0..r).rev() {
                    st[i] = if f[i] == 1 { 0 } else { acc };
                    acc *= f[i];
                }
                st
            };
            let sb = fstride(&ic.b);
            let mut idx = vec![0i64; r];
            for v in ref_.iter_mut() {
                let mut ib = 0i64;
                for i in 0..r {
                    ib += idx[i] * sb[i];
                }
                *v = ic.op.apply(*v, b[ib as usize]);
                for i in (0..r).rev() {
                    idx[i] += 1;
                    if idx[i] < fs[i] {
                        break;
                    }
                    idx[i] = 0;
                }
            }
        }
        assert!(
            binary_can_inplace(&ic.a, &ic.b),
            "{}: not in-place eligible",
            ic.name
        );
        binary_op_inplace(&mut a, &ic.a, &b, &ic.b, ic.op);
        bits_eq(ic.name, &a, &ref_);
    }
}

// ---------------------------------------------------------------- 激活

#[test]
fn activation_tests() {
    let mut rng = Rng::new();
    let mut x = vec![0f32; 4096];
    rng.fill(&mut x);

    // relu
    let mut t = x.clone();
    relu_inplace(&mut t);
    for (i, &v) in t.iter().enumerate() {
        let want = if x[i] > 0.0 { x[i] } else { 0.0 };
        assert_eq!(v.to_bits(), want.to_bits(), "relu at {i}");
    }

    // sigmoid（f64 参考）
    let mut y = vec![0f32; x.len()];
    sigmoid_tensor(&x, &mut y);
    for (i, &v) in y.iter().enumerate() {
        let want = 1.0 / (1.0 + (-(x[i] as f64)).exp());
        assert!(
            (v as f64 - want).abs() < 1e-6,
            "sigmoid at {i}: {v} vs {want}"
        );
    }

    // hardsigmoid（v6 mobile 常数：alpha=1/6, beta=1/2）
    let (alpha, beta) = (1.0f32 / 6.0, 0.5f32);
    hardsigmoid(&x, alpha, beta, &mut y);
    for (i, &v) in y.iter().enumerate() {
        let t = x[i] as f64 * alpha as f64 + beta as f64;
        let want = t.clamp(0.0, 1.0);
        assert!((v as f64 - want).abs() < 1e-6, "hardsigmoid at {i}");
    }

    // gelu（f64 参考，用同一 erf 多项式双精度化不合适——用容差 2e-6）
    let mut t = x.clone();
    gelu_inplace(&mut t, 1.4142135, 1.0, 0.5);
    for (i, &v) in t.iter().enumerate() {
        // f64 erf 参考：A&S 7.1.26 双精度。★ 参数是 x/c1（c1=√2），
        // 与内核的 erf(x·inv_c1) 一致
        let xf = x[i] as f64;
        let ax = (xf / 1.4142135f64).abs();
        let tt = 1.0 / (0.3275911 * ax + 1.0);
        let poly = tt
            * ((((1.061405429 * tt - 1.453152027) * tt + 1.421413741) * tt - 0.284496736) * tt
                + 0.254829592);
        let er = 1.0 - poly * (-ax * ax).exp();
        let er = if xf < 0.0 { -er } else { er };
        let want = 0.5 * xf * (er + 1.0);
        assert!((v as f64 - want).abs() < 3e-7, "gelu at {i}: {v} vs {want}");
    }

    // erf（奇函数 + 与 gelu 一致性）
    let mut t = x.clone();
    erf_inplace(&mut t);
    for (i, &v) in t.iter().enumerate() {
        assert_eq!(v.to_bits(), erf1(x[i]).to_bits(), "erf at {i}");
    }

    // clip
    let mut t = x.clone();
    clip_inplace(&mut t, -0.3, 0.7);
    for (i, &v) in t.iter().enumerate() {
        let want = if x[i] < -0.3 {
            -0.3
        } else if 0.7 < x[i] {
            0.7
        } else {
            x[i]
        };
        assert_eq!(v.to_bits(), want.to_bits(), "clip at {i}");
    }

    // softmax：行和为 1、与 f64 参考一致
    let (outer, inner) = (37usize, 13usize);
    let mut t = vec![0f32; outer * inner];
    rng.fill(&mut t);
    softmax_last_dim(&mut t, inner);
    for o in 0..outer {
        let row = &t[o * inner..(o + 1) * inner];
        let sum: f64 = row.iter().map(|&v| v as f64).sum();
        assert!((sum - 1.0).abs() < 1e-4, "softmax row sum {sum}");
    }
    // cls 形状（inner=4 < 8，纯标量路径）
    let mut t4 = vec![0f32; 4 * 4];
    rng.fill(&mut t4);
    let orig = t4.clone();
    softmax_last_dim(&mut t4, 4);
    for o in 0..4 {
        let row = &orig[o * 4..o * 4 + 4];
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f64> = row.iter().map(|&v| (v as f64 - mx as f64).exp()).collect();
        let s: f64 = exps.iter().sum();
        for j in 0..4 {
            let want = exps[j] / s;
            assert!(
                (t4[o * 4 + j] as f64 - want).abs() < 1e-6,
                "softmax4 at {o},{j}"
            );
        }
    }
}

// ---------------------------------------------------------------- 池化 / resize

#[test]
fn pool_and_resize_tests() {
    let mut rng = Rng::new();
    let (n, c, h, w) = (2usize, 3, 20, 30);

    // max pool 2x2 s1 SAME_UPPER（v6 det 的形状）：ph=pw=0, peh=pew=1
    let mut x = vec![0f32; n * c * h * w];
    rng.fill(&mut x);
    let mut y = F32Buf::new();
    let (oh, ow) = pool2d(&x, n, c, h, w, 2, 2, 1, 1, 0, 0, 1, 1, true, &mut y);
    assert_eq!((oh, ow), (h, w));
    for nc in 0..n * c {
        for oy in 0..h {
            for oxx in 0..w {
                let mut mx = f32::NEG_INFINITY;
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let iy = (oy + dy).min(h - 1);
                    let ix = (oxx + dx).min(w - 1);
                    mx = mx.max(x[nc * h * w + iy * w + ix]);
                }
                let got = y[nc * h * w + oy * w + oxx];
                assert_eq!(got, mx, "maxpool at {nc},{oy},{oxx}");
            }
        }
    }

    // avg pool 3x3 s2 p1
    let (oh, ow) = pool2d(&x, n, c, h, w, 3, 3, 2, 2, 1, 1, 1, 1, false, &mut y);
    assert_eq!((oh, ow), ((20 + 2 - 3) / 2 + 1, (30 + 2 - 3) / 2 + 1));
    for nc in 0..n * c {
        for oy in 0..oh {
            for oxx in 0..ow {
                let mut s = 0f64;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = oy as isize * 2 - 1 + ky as isize;
                        let ix = oxx as isize * 2 - 1 + kx as isize;
                        if iy < 0 || iy as usize >= h || ix < 0 || ix as usize >= w {
                            continue;
                        }
                        s += x[nc * h * w + iy as usize * w + ix as usize] as f64;
                    }
                }
                let want = (s / 9.0) as f32;
                let got = y[nc * oh * ow + oy * ow + oxx];
                assert!((got - want).abs() < 1e-5, "avgpool at {nc},{oy},{oxx}");
            }
        }
    }

    // global avg pool（f64 累加）
    let mut g = F32Buf::new();
    global_avg_pool(&x, n, c, &mut g);
    for nc in 0..n * c {
        let s: f64 = x[nc * h * w..(nc + 1) * h * w]
            .iter()
            .map(|&v| v as f64)
            .sum();
        assert_eq!(g[nc], (s / (h * w) as f64) as f32, "gap at {nc}");
    }

    // resize nearest
    let (oh, ow) = (33, 47);
    resize_nearest(&x, n, c, h, w, oh, ow, &mut y);
    for nc in 0..n * c {
        for oy in 0..oh {
            let iy = (((oy as f32 * h as f32) / oh as f32) as usize).min(h - 1);
            for oxx in 0..ow {
                let ix = (((oxx as f32 * w as f32) / ow as f32) as usize).min(w - 1);
                assert_eq!(
                    y[nc * oh * ow + oy * ow + oxx],
                    x[nc * h * w + iy * w + ix],
                    "resize nn at {nc},{oy},{oxx}"
                );
            }
        }
    }

    // resize bilinear（f64 参考）
    for align in [false, true] {
        resize_bilinear(&x, n, c, h, w, oh, ow, align, &mut y);
        for nc in 0..n * c {
            for oy in 0..oh {
                let v = if align && oh > 1 {
                    oy as f64 * (h - 1) as f64 / (oh - 1) as f64
                } else {
                    ((oy as f64 + 0.5) * h as f64 / oh as f64 - 0.5).max(0.0)
                };
                let iy0 = v.floor().clamp(0.0, (h - 1) as f64) as usize;
                let iy1 = (iy0 + 1).min(h - 1);
                let fy = v - iy0 as f64;
                for oxx in 0..ow {
                    let v = if align && ow > 1 {
                        oxx as f64 * (w - 1) as f64 / (ow - 1) as f64
                    } else {
                        ((oxx as f64 + 0.5) * w as f64 / ow as f64 - 0.5).max(0.0)
                    };
                    let ix0 = v.floor().clamp(0.0, (w - 1) as f64) as usize;
                    let ix1 = (ix0 + 1).min(w - 1);
                    let fx = v - ix0 as f64;
                    let p = |i: usize, j: usize| x[nc * h * w + i * w + j] as f64;
                    let want = p(iy0, ix0) * (1.0 - fx) * (1.0 - fy)
                        + p(iy0, ix1) * fx * (1.0 - fy)
                        + p(iy1, ix0) * (1.0 - fx) * fy
                        + p(iy1, ix1) * fx * fy;
                    let got = y[nc * oh * ow + oy * ow + oxx];
                    assert!(
                        (got as f64 - want).abs() < 1e-4,
                        "resize bl at {nc},{oy},{oxx}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------- 形状类 / matmul

#[test]
fn shape_op_tests() {
    let mut rng = Rng::new();

    // transpose [2,3,4,5] → perm [0,2,3,1]
    let shape = [2i64, 3, 4, 5];
    let total = 120usize;
    let mut x = vec![0f32; total];
    rng.fill(&mut x);
    let (y, ys) = transpose_tensor(PayloadRef::F32(&x), &shape, &[0, 2, 3, 1]);
    assert_eq!(ys, vec![2, 4, 5, 3]);
    let Payload::F32(yd) = y else { panic!() };
    // 朴素参考
    for i0 in 0..2 {
        for i1 in 0..3 {
            for i2 in 0..4 {
                for i3 in 0..5 {
                    let src = ((i0 * 3 + i1) * 4 + i2) * 5 + i3;
                    // 输出下标 (i0, i2, i3, i1)
                    let dst = ((i0 * 4 + i2) * 5 + i3) * 3 + i1;
                    assert_eq!(yd[dst], x[src], "transpose at {i0},{i1},{i2},{i3}");
                }
            }
        }
    }

    // transpose 反转（默认 perm）
    let (y2, ys2) = transpose_tensor(PayloadRef::F32(&x), &shape, &[]);
    assert_eq!(ys2, vec![5, 4, 3, 2]);
    let Payload::F32(y2d) = y2 else { panic!() };
    for i in 0..total {
        // 反转 = 线性下标位反转
        let lin = i as i64;
        let i0 = lin / 60;
        let mut r = lin % 60;
        let i1 = r / 20;
        r %= 20;
        let i2 = r / 5;
        let i3 = r % 5;
        let dlin = ((i3 * 4 + i2) * 3 + i1) * 2 + i0; // 输出 [5,4,3,2]：步长 24,6,2,1
        assert_eq!(y2d[dlin as usize], x[i], "rev transpose at {i}");
    }

    // slice：整维 + 步长 + 负索引
    let (sy, ss) = slice_tensor(
        PayloadRef::F32(&x),
        &shape,
        &[1, -2],
        &[3, i64::MAX],
        &[1, 3],
        &[1, 2],
    );
    // dim3: start=-2→3, end=MAX→5, step 2 → 只有下标 3（5 越界），长度 1
    assert_eq!(ss, vec![2, 2, 4, 1]);
    let Payload::F32(syd) = sy else { panic!() };
    for i0 in 0..2 {
        for i1 in 0..2 {
            for i2 in 0..4 {
                for i3 in 0..1 {
                    let src = ((i0 * 3 + (1 + i1)) * 4 + i2) * 5 + (3 + 2 * i3);
                    let dst = (i0 * 2 + i1) * 4 + i2; // dim3 长度 1
                    assert_eq!(syd[dst], x[src], "slice at {i0},{i1},{i2},{i3}");
                }
            }
        }
    }

    // concat（f32，axis=1）
    let xs_shape: Vec<&[i64]> = vec![&[2, 3, 4], &[2, 5, 4]];
    let mut xa = vec![0f32; 24];
    let mut xb = vec![0f32; 40];
    rng.fill(&mut xa);
    rng.fill(&mut xb);
    let mut out = Payload::F32(F32Buf::new());
    let payloads = [PayloadRef::F32(&xa), PayloadRef::F32(&xb)];
    let cshape = concat_any(&payloads, &xs_shape, 1, &mut out);
    assert_eq!(cshape, vec![2, 8, 4]);
    let Payload::F32(cd) = out else { panic!() };
    for o in 0..2 {
        for ch in 0..8 {
            for i in 0..4 {
                let src = if ch < 3 { &xa } else { &xb };
                let chl = if ch < 3 { ch } else { ch - 3 };
                assert_eq!(
                    cd[(o * 8 + ch) * 4 + i],
                    src[(o * (if ch < 3 { 3 } else { 5 }) + chl) * 4 + i],
                    "concat at {o},{ch},{i}"
                );
            }
        }
    }

    // reduce_mean keepdims
    let mut rm = F32Buf::new();
    let rs = reduce_mean(&x, &shape, &[2, 3], true, &mut rm);
    assert_eq!(rs, vec![2, 3, 1, 1]);
    for i0 in 0..2 {
        for i1 in 0..3 {
            let mut s = 0f64;
            for i2 in 0..4 {
                for i3 in 0..5 {
                    s += x[((i0 * 3 + i1) * 4 + i2) * 5 + i3] as f64;
                }
            }
            assert_eq!(
                rm[i0 * 3 + i1],
                (s / 20.0) as f32,
                "reduce_mean at {i0},{i1}"
            );
        }
    }

    // matmul：批次广播（A [2,3,40,80] × B [80,101]）。
    // 相对误差按整张量最大幅度归一（check 的语义）：逐元素归一会在
    // 相消到 ~0 的点上放大 f32/f64 差异（实测 1.5e-5 的假阳性）。
    let a_shape = [2i64, 3, 40, 80];
    let b_shape = [80i64, 101];
    let mut a = vec![0f32; (2 * 3 * 40 * 80) as usize];
    let mut b = vec![0f32; 80 * 101];
    rng.fill(&mut a);
    rng.fill(&mut b);
    let mut mm = F32Buf::new();
    let ms = matmul(&a, &a_shape, &b, &b_shape, &mut mm);
    assert_eq!(ms, vec![2, 3, 40, 101]);
    let mut ref_ = vec![0f32; 6 * 40 * 101];
    let mut scale = 0f64;
    for bi in 0..6usize {
        for row in 0..40usize {
            for j in 0..101usize {
                let mut s = 0f64;
                for kk in 0..80usize {
                    s += a[bi * 40 * 80 + row * 80 + kk] as f64 * b[kk * 101 + j] as f64;
                }
                ref_[bi * 40 * 101 + row * 101 + j] = s as f32;
                scale = scale.max(s.abs());
            }
        }
    }
    let mut mx = 0f64;
    let mut arg = 0usize;
    for i in 0..ref_.len() {
        let d = (mm[i] as f64 - ref_[i] as f64).abs();
        if d > mx {
            mx = d;
            arg = i;
        }
    }
    assert!(
        mx / scale.max(1e-6) < 1e-5,
        "matmul: max diff {mx:.3e} at {arg} (rel {:.1e})",
        mx / scale
    );

    // reshape：-1 推断与 0 保留
    assert_eq!(reshape_shape(&[2, 3, 4, 5], &[0, -1, 10]), vec![2, 6, 10]); // 0 保留 dim0=2，known=2*10，推断 120/20=6
    assert_eq!(reshape_shape(&[2, 3, 4, 5], &[-1]), vec![120]);
    assert_eq!(reshape_shape(&[2, 3, 4, 5], &[5, 24]), vec![5, 24]);
}

// ---------------------------------------------------------------- convtranspose

#[test]
fn convtranspose_tests() {
    let mut rng = Rng::new();
    let (n, c, h, w) = (1usize, 7, 5, 6);
    let m = 3usize;
    let mut x = vec![0f32; n * c * h * w];
    rng.fill(&mut x);
    // ONNX 布局 [C_in, C_out, 2, 2]
    let mut wt = vec![0f32; c * m * 4];
    rng.fill(&mut wt);
    let mut y = F32Buf::new();
    let ys = convtranspose2d(
        &x,
        &[n as i64, c as i64, h as i64, w as i64],
        &wt,
        &[c as i64, m as i64, 2, 2],
        2,
        2,
        0,
        0,
        &mut y,
    );
    assert_eq!(ys, [n as i64, m as i64, (h * 2) as i64, (w * 2) as i64]);
    let (oh, ow) = (h * 2, w * 2);
    for bn in 0..n {
        for co in 0..m {
            for oy in 0..oh {
                for oxx in 0..ow {
                    // out[oy][ox] = Σ_c x[c][oy/2][ox/2]·w[c][co][oy%2][ox%2]
                    let mut s = 0f64;
                    for ch in 0..c {
                        let (iy, ky) = (oy / 2, oy % 2);
                        let (ix, kx) = (oxx / 2, oxx % 2);
                        s += x[((bn * c + ch) * h + iy) * w + ix] as f64
                            * wt[((ch * m + co) * 2 + ky) * 2 + kx] as f64;
                    }
                    let want = s as f32;
                    let got = y[((bn * m + co) * oh + oy) * ow + oxx];
                    assert!(
                        (got - want).abs() / want.abs().max(1e-6) < 1e-5,
                        "convT at {co},{oy},{oxx}: {got} vs {want}"
                    );
                }
            }
        }
    }
}
