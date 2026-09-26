//! im2col 与 GEMM 的**分账**：同一个 conv shape，分别只跑 im2col、只跑
//! sgemm（喂一整块建好的 patch）、以及完整 conv2d。
//!
//! 目的：判断 k3x3 的普通卷积到底是「搬运 patch 矩阵」贵，还是 GEMM 贵。
//! 这决定了下一步是去做 implicit GEMM 还是去调 GEMM。
//!
//! cargo run --release -p qppocr-kernels --example im2col_split

use qppocr_kernels::activation::Activation;
use qppocr_kernels::buf::F32Buf;
use qppocr_kernels::conv::{ConvParams, conv2d};
use qppocr_kernels::gemm::{im2col, sgemm_serial};

/// (名字, c_in, c_out, 输出 h, 输出 w, k, stride)
const CASES: &[(&str, usize, usize, usize, usize, usize, usize)] = &[
    ("tiny  det 64->16 k3x3 @200x240 s1", 64, 16, 200, 240, 3, 1),
    ("tiny  det 32->16 k3x3 @400x480 s2", 32, 16, 400, 480, 3, 2),
    ("tiny  det  3->16 k3x3 @800x960 s2", 3, 16, 800, 960, 3, 2),
    ("tiny  det 16->8  k2x2 @400x480 s1", 16, 8, 400, 480, 2, 1),
    ("small det 96->24 k3x3 @200x240 s1", 96, 24, 200, 240, 3, 1),
];

fn main() {
    qppocr_kernels::par::set_threads(1);
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        let v = rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (((v >> 40) as i64 - (1 << 23)) as f64 / (1i64 << 23) as f64) as f32
    };

    for &(name, cin, cout, oh, ow, k, s) in CASES {
        let p = (k - 1) / 2;
        let h = (oh - 1) * s + k - 2 * p;
        let w = (ow - 1) * s + k - 2 * p;
        let x: Vec<f32> = (0..cin * h * w).map(|_| next()).collect();
        let wt: Vec<f32> = (0..cout * cin * k * k).map(|_| next()).collect();
        let bias: Vec<f32> = (0..cout).map(|_| next()).collect();
        let params = ConvParams {
            sh: s,
            sw: s,
            ph: p,
            pw: p,
            peh: p,
            pew: p,
            dh: 1,
            dw: 1,
            group: 1,
        };
        let kk = cin * k * k;

        // ---- (a) 只跑 im2col，按 conv2d 单线程时的 tile 规则 ----
        let col_budget = 1usize << 19;
        let tile = (col_budget / (kk * ow).max(1)).max(1).min(oh);
        let ntiles = oh.div_ceil(tile);
        let mut cols = vec![0f32; kk * tile * ow];
        let mut best_i = f64::MAX;
        for _ in 0..9 {
            let t0 = std::time::Instant::now();
            for t in 0..ntiles {
                let oy0 = t * tile;
                let rows = tile.min(oh - oy0);
                im2col(&x, cin, h, w, k, k, s, s, p, p, ow, oy0, rows, &mut cols);
            }
            best_i = best_i.min(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // ---- (b) 只跑 sgemm，喂一整块建好的 patch ----
        //      ld(b) 取 oh*ow 让每次读的仍是稠密 B；C 的行距也是 oh*ow。
        let big: Vec<f32> = vec![0.0; kk * oh * ow];
        let mut y = F32Buf::new();
        let mut best_g = f64::MAX;
        for _ in 0..9 {
            let t0 = std::time::Instant::now();
            // SAFETY: sgemm 写满每个输出元素。
            unsafe { y.resize_uninit(cout * oh * ow) };
            sgemm_serial(
                &wt,
                &big,
                y.as_mut_slice(),
                cout,
                oh * ow,
                kk,
                oh * ow,
                Some(&bias),
                &Activation::default(),
            );
            best_g = best_g.min(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // ---- (b2) 按 conv2d 的方式切 tile 调 sgemm（同样的 n/ldc）----
        let mut best_g2 = f64::MAX;
        for _ in 0..9 {
            let t0 = std::time::Instant::now();
            // SAFETY: sgemm 写满每个输出元素。
            unsafe { y.resize_uninit(cout * oh * ow) };
            for t in 0..ntiles {
                let oy0 = t * tile;
                let rows = tile.min(oh - oy0);
                let ysub =
                    &mut y.as_mut_slice()[oy0 * ow..oy0 * ow + (cout - 1) * oh * ow + rows * ow];
                sgemm_serial(
                    &wt,
                    &big[..kk * rows * ow],
                    ysub,
                    cout,
                    rows * ow,
                    kk,
                    oh * ow,
                    Some(&bias),
                    &Activation::default(),
                );
            }
            best_g2 = best_g2.min(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // ---- (c) 完整 conv2d ----
        let mut best_c = f64::MAX;
        for _ in 0..9 {
            let t0 = std::time::Instant::now();
            conv2d(
                &x,
                &[1, cin as i64, h as i64, w as i64],
                &wt,
                &[cout as i64, cin as i64, k as i64, k as i64],
                Some(&bias),
                &params,
                &Activation::default(),
                &mut y,
            );
            best_c = best_c.min(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let macs = (cout * cin * k * k * oh * ow) as f64;
        let carry = 100.0 * best_i / (best_i + best_g);
        println!(
            "{name}
    im2col {best_i:7.3} ms   sgemm {best_g:7.3} ms   合计 {:7.3}                conv2d {best_c:7.3} ms   patch {:.0} MB
    GEMM {:5.1} GMAC/s   搬运占 {carry:.0}%",
            best_i + best_g,
            (kk * oh * ow * 4) as f64 / 1e6,
            macs / best_g / 1e6,
        );
        println!(
            "    切 tile 后只跑 GEMM {best_g2:7.3} ms（n={} ldc={ohw} 共 {ntiles} 次）",
            tile * ow,
            ohw = oh * ow,
        );
    }
}
