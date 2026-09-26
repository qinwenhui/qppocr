//! 纯计算扩展探针：N 个**完全独立**的线程各跑一份同样的 GEMM，
//! 量「N 线程相对 1 线程能拿到几倍」。
//!
//! 这是判断「并行效率上不去」是**我们的调度问题**还是**机器本身**的尺子
//! ——本机是 6P+8E 的混合核，重载下降频 + E 核慢，N 大了单线程吞吐会掉。
//!
//! cargo run --release -p qppocr-kernels --example scaling_probe

use qppocr_kernels::activation::Activation;
use qppocr_kernels::buf::F32Buf;
use qppocr_kernels::gemm::sgemm_serial;

fn main() {
    // 一次 GEMM：M=N=K=384，约 56.6M MAC —— 与 rec 里最大的那些 1x1 同量级。
    const D: usize = 384;
    let a: Vec<f32> = (0..D * D).map(|i| (i % 17) as f32 * 1e-3).collect();
    let b: Vec<f32> = (0..D * D).map(|i| (i % 13) as f32 * 1e-3).collect();
    let macs = (D * D * D) as f64;

    println!(
        "{:>8}  {:>10}  {:>10}  {:>8}",
        "线程", "墙钟 ms", "每份 ms", "扩展比"
    );
    let mut base = 0.0f64;
    for n in [1usize, 2, 3, 4, 6, 8, 12, 16] {
        // 预热
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            std::thread::scope(|s| {
                for _ in 0..n {
                    s.spawn(|| {
                        let mut y = F32Buf::new();
                        // SAFETY: 每个线程自己的 y。
                        unsafe { y.resize_uninit(D * D) };
                        sgemm_serial(
                            &a,
                            &b,
                            y.as_mut_slice(),
                            D,
                            D,
                            D,
                            D,
                            None,
                            &Activation::default(),
                        );
                        std::hint::black_box(y[0]);
                    });
                }
            });
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            best = best.min(ms);
        }
        let per = best / n as f64;
        if n == 1 {
            base = per;
        }
        println!(
            "{n:>8}  {best:>10.3}  {per:>10.3}  {:>8.2}x   ({:.1} GMAC/s/线程)",
            base / per,
            macs / per / 1e6
        );
    }
}
