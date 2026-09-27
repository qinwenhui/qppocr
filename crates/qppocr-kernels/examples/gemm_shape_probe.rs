//! sgemm 的**形状探针**：同一套内核，只换 (m, n, k)，量单线程 GMAC/s。
//!
//! 存在的理由：逐算子账里 `320->160 k1x1 @3x249` 跑 54.6 GMAC/s、同一个
//! `@3x60` 只有 9.2——**差 6 倍，而且是同一个内核**。要判断该改内核还是
//! 该改调度，就得先知道「效率是从哪个形状开始掉的」。
//!
//! 用法：`cargo run --release -p qppocr-kernels --example gemm_shape_probe`
//! 或带自定义形状：`... --example gemm_shape_probe -- 160x180x320 160x747x320`
//! （`MxNxK`，M = 输出通道、N = 空间、K = 输入通道——即 1x1 卷积的 sgemm 形参）
//!
//! 全串行（`sgemm_serial`），不受并行调度干扰；先跑一次热身。
use qppocr_kernels::activation::Activation;
use qppocr_kernels::gemm::sgemm_serial;
use std::hint::black_box;
use std::time::Instant;

fn run(m: usize, n: usize, k: usize, reps: usize) -> f64 {
    let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 * 0.01).collect();
    let mut c = vec![0f32; m * n];
    let act = Activation::default();
    // 热身：让 A/B/C 进 cache、让分支预测收敛
    sgemm_serial(&a, &b, &mut c, m, n, k, n, None, &act);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        sgemm_serial(&a, &b, &mut c, m, n, k, n, None, &act);
        let dt = t.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    black_box(&c);
    (m * n * k) as f64 / best / 1e9
}

/// 并发档：`T` 个线程各跑同一形状，报**聚合** GMAC/s。
///
/// 这是判别「并发膨胀是流水线的错还是内存子系统的错」的尺子：同样的
/// 内核、同样的形状，只改同时跑几个。若聚合吞吐随 T 平掉，那 rec 的
/// 1.15x 就不是 qppocr 的调度问题。
fn mt(m: usize, n: usize, k: usize, t: usize, secs: f64, churn: bool) {
    let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 * 0.01).collect();
    let act = Activation::default();
    let done = std::sync::atomic::AtomicBool::new(false);
    let counts: Vec<std::sync::atomic::AtomicU64> =
        (0..t).map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for c in &counts {
            s.spawn(|| {
                let mut cc = vec![0f32; m * n];
                // churn 档：每次迭代从全局池借/还一个缓冲（模拟执行器每节点
                // 一次 alloc+dealloc），量互斥锁在 T 线程并发下的代价。
                let mut buf = if churn {
                    Some(qppocr_kernels::buf::F32Buf::with_zeroed(m * n))
                } else {
                    None
                };
                while !done.load(std::sync::atomic::Ordering::Relaxed) {
                    if churn {
                        buf.take();
                        buf = Some(qppocr_kernels::buf::F32Buf::with_zeroed(m * n));
                    }
                    sgemm_serial(&a, &b, &mut cc, m, n, k, n, None, &act);
                    c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                black_box(&cc);
            });
        }
        std::thread::sleep(std::time::Duration::from_secs_f64(secs));
        done.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    let total: u64 = counts.iter().map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).sum();
    let dt = t0.elapsed().as_secs_f64();
    let macs = total as f64 * (m * n * k) as f64;
    println!(
        "  {t:>3} 线程  {:<16} 聚合 {:>7.1} GMAC/s   （单线程 {:>5.1}）",
        format!("{m}x{n}x{k}"),
        macs / dt / 1e9,
        macs / dt / 1e9 / t as f64
    );
}

fn main() {
    if let Some(a) = std::env::args().nth(1) {
        if let Some(rest) = a.strip_prefix("mt:") {
            let (m, n, k) = {
                let v: Vec<usize> = rest.split('x').filter_map(|t| t.parse().ok()).collect();
                (v[0], v[1], v[2])
            };
            let secs: f64 = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(1.5);
            let churn = std::env::args().any(|v| v == "churn");
            println!("并发档：{m}x{n}x{k}，每档 {secs}s，池扰动={churn}");
            for t in [1usize, 2, 4, 6, 8, 11, 16] {
                mt(m, n, k, t, secs, churn);
            }
            return;
        }
    }
    let shapes: Vec<(usize, usize, usize)> = std::env::args()
        .skip(1)
        .filter_map(|s| {
            let v: Vec<usize> = s.split('x').filter_map(|t| t.parse().ok()).collect();
            if v.len() == 3 {
                Some((v[0], v[1], v[2]))
            } else {
                None
            }
        })
        .collect();
    // 默认：rec 的 1x1 卷积（M=160/320、K=160/320、N=3·W）+ 一个 det 的大 N 参照
    let shapes = if shapes.is_empty() {
        vec![
            (160, 180, 320),
            (160, 213, 320),
            (160, 249, 320),
            (160, 360, 320),
            (160, 747, 320),
            (320, 180, 160),
            (320, 747, 160),
            (160, 180, 160),
            (96, 360, 192),
            (64, 49920, 32),
        ]
    } else {
        shapes
    };
    println!("{:>5} {:>6} {:>6}  {:>9}  {:>10}", "M", "N", "K", "GMAC/s", "ms/次");
    let mut acc = 0.0;
    for (m, n, k) in shapes {
        let reps = (2_000_000 / (m * n * k).max(1)).clamp(3, 200);
        let g = run(m, n, k, reps);
        acc += g;
        println!(
            "{m:>5} {n:>6} {k:>6}  {g:>9.1}  {:>10.4}",
            (m * n * k) as f64 / g / 1e6
        );
    }
    let _ = acc;
}
