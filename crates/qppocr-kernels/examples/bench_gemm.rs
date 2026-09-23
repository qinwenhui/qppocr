//! GEMM 性能对拍入口（tools/migration-bench 的 Rust 侧孪生）。
// GEMM A/B — Rust 侧（qppocr-kernels 的 sgemm）。
// 与 bench.cpp 同形状、同数据、同「7 轮取最好」口径；交错运行对比。
// 构建：rustc -O -C target-feature=+avx2,+fma -o bench_rust bench_rs_qppocr.rs
//       --extern qppocr_kernels=<path-to-libqppocr_kernels.rlib> -L <deps>
// （或直接用 cargo 里的 benches/下例程；此文件保持与 bench.cpp 对应的独立性）

use std::time::Instant;

use qppocr_kernels::activation::Activation;
use qppocr_kernels::gemm::sgemm_serial;
use qppocr_kernels::{Backend, force_backend};

fn main() {
    force_backend(Some(Backend::Avx2));
    let shapes: &[(usize, usize, usize)] = &[
        (3136, 64, 576),  // det 的 3x3 conv（im2col 后）
        (1024, 256, 256), // 中等 conv
        (512, 256, 576),
        (64, 64, 64), // 小特征图，fork 阈值以下
    ];
    for &(m, n, k) in shapes {
        let mut a = vec![0.001f32; m * k];
        let mut b = vec![0.002f32; k * n];
        let mut c = vec![0f32; m * n];
        let mut i = 0;
        while i < a.len() {
            a[i] = 0.5;
            i += 7;
        }
        let mut i = 0;
        while i < b.len() {
            b[i] = 0.25;
            i += 11;
        }
        let mut best = f64::MAX;
        for _ in 0..7 {
            let t0 = Instant::now();
            sgemm_serial(&a, &b, &mut c, m, n, k, n, None, &Activation::default());
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if ms < best {
                best = ms;
            }
        }
        let gflops = 2.0 * (m * n * k) as f64 / (best * 1e6);
        println!(
            "rust              M={m:<5} N={n:<4} K={k:<4}  best {best:8.3} ms  {gflops:7.2} GFLOPS   chk={:.4}",
            c[m / 2 * n + n / 2]
        );
    }
    force_backend(None);
}
