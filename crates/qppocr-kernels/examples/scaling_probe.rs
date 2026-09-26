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

/// 量 fork_join 的协调成本：连续 fork N 次、每次只做一点点工作。
fn fork_cost() {
    const N: usize = 2000;
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            qppocr_kernels::par::parallel_for(64, 1, |b, e| {
                std::hint::black_box(b + e);
            });
        }
        best = best.min(t0.elapsed().as_secs_f64() * 1e6 / N as f64);
    }
    println!(
        "
一次 parallel_for(64 单元) 的平均成本（{} 线程）：{best:.1} us",
        qppocr_kernels::par::threads()
    );
}

/// 量内存带宽：N 个线程各在自己的大缓冲上做读+写（`dst[i] = src[i] * 1.5`）。
/// 用来判断那些「并行度上不去」的搬运类算子是不是撞了带宽墙。
fn mem_bw(nthreads: usize) {
    const MB: usize = 8;
    let n = MB * 1024 * 1024 / 4;
    let src: Vec<f32> = vec![1.0; n];
    let mut dst: Vec<f32> = vec![0.0; n];
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        let chunk = n.div_ceil(nthreads);
        std::thread::scope(|s| {
            for (i, d) in dst.chunks_mut(chunk).enumerate() {
                let sr = &src[i * chunk..i * chunk + d.len()];
                s.spawn(move || {
                    for (o, v) in d.iter_mut().zip(sr) {
                        *o = *v * 1.5;
                    }
                });
            }
        });
        best = best.min(t0.elapsed().as_secs_f64());
    }
    // 读 8 MB + 写 8 MB
    let gbs = (n * 4 * 2) as f64 / best / 1e9;
    println!("  {nthreads:>2} 线程   读+写 {:.1} GB/s", gbs);
    std::hint::black_box(&dst);
}

/// det 里那个 MaxPool 的形状（3x3 s1 p1，1x16x400x480）单独计时。
fn maxpool_probe() {
    let (n, c, h, w) = (1usize, 16usize, 400usize, 480usize);
    let x: Vec<f32> = (0..n * c * h * w).map(|i| (i % 251) as f32).collect();
    let mut y = qppocr_kernels::buf::F32Buf::new();
    let mut best = f64::MAX;
    for _ in 0..9 {
        let t0 = std::time::Instant::now();
        qppocr_kernels::pool2d::pool2d(&x, n, c, h, w, 2, 2, 1, 1, 0, 0, 1, 1, true, &mut y);
        best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
    }
    // 每轮新建输出缓冲（模拟整图里「上一个算子写出来、下一个读」的冷块）
    let mut cold = f64::MAX;
    for _ in 0..9 {
        let mut y2 = qppocr_kernels::buf::F32Buf::new();
        let t0 = std::time::Instant::now();
        qppocr_kernels::pool2d::pool2d(&x, n, c, h, w, 2, 2, 1, 1, 0, 0, 1, 1, true, &mut y2);
        cold = cold.min(t0.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(&y2);
    }
    println!("    热输出缓冲 {best:7.3} ms   每轮新建输出 {cold:7.3} ms");
    let bytes = (n * c * h * w * 4 * 2) as f64;
    println!(
        "  MaxPool [1,16,400,480] k2x2 s1 SAME_UPPER   {} 线程：{best:7.3} ms   {:.1} GB/s",
        qppocr_kernels::par::threads(),
        bytes / best / 1e6
    );
    std::hint::black_box(&y);
}

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
    if std::env::args().nth(1).as_deref() == Some("pool") {
        maxpool_probe();
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("bw") {
        println!("内存带宽（8 MB 缓冲，读写各一遍）：");
        for t in [1usize, 2, 4, 8, 16] {
            mem_bw(t);
        }
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("fork") {
        // 池首用定容，改线程数必须换进程——用 QPPOCR_THREADS 环境变量。
        fork_cost();
        return;
    }
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
