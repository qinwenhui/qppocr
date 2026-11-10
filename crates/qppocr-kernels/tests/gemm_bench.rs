//! sgemm 面板微基准（诊断用，默认 ignore）：
//!   cargo test -p qppocr-kernels --test gemm_bench -- --ignored --nocapture
//! 形状取自真实工作负载热点（QPPOCR_PROF 实测）：
//!   rec  1×1: (M=522, N=1024, K=512) ×6 与 (M=522, N=512, K=1024) ×6
//!   det  1×1: (M=2280, N=512, K=1024) 等同族
use qppocr_kernels::gemm::{sgemm, sgemm_serial};

fn bench(name: &str, m: usize, n: usize, k: usize, rounds: usize) {
    let mut a = vec![0f32; m * k];
    let mut b = vec![0f32; k * n];
    let mut c = vec![0f32; m * n];
    for (i, x) in a.iter_mut().enumerate() {
        *x = (i % 7) as f32 * 0.01 - 0.03;
    }
    for (i, x) in b.iter_mut().enumerate() {
        *x = (i % 11) as f32 * 0.02 - 0.1;
    }
    // 预热 + 计时
    sgemm(&a, &b, &mut c, m, n, k, n, None, &Default::default());
    let t0 = std::time::Instant::now();
    for _ in 0..rounds {
        sgemm(&a, &b, &mut c, m, n, k, n, None, &Default::default());
    }
    let par_ms = t0.elapsed().as_secs_f64() * 1000.0 / rounds as f64;
    let t0 = std::time::Instant::now();
    for _ in 0..rounds.max(1) {
        sgemm_serial(&a, &b, &mut c, m, n, k, n, None, &Default::default());
    }
    let ser_ms = t0.elapsed().as_secs_f64() * 1000.0 / rounds.max(1) as f64;
    let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;
    println!(
        "{name:<22} M={m} N={n} K={k}: 并行 {par_ms:8.2} ms ({:6.1} GFLOP/s) | 串行 {ser_ms:8.2} ms ({:6.1} GFLOP/s)",
        gflop / par_ms * 1000.0,
        gflop / ser_ms * 1000.0,
    );
    let _ = std::hint::black_box(&c);
}

#[test]
#[ignore]
fn gemm_hot_shapes() {
    bench("rec 512->1024", 522, 1024, 512, 20);
    bench("rec 1024->512", 522, 512, 1024, 20);
    bench("rec 768->1536", 261, 1536, 768, 20);
    bench("det 128->256", 2280, 256, 128, 10);
    bench("det 1024->512", 2280, 512, 1024, 10);
    bench("det 256->64", 2280, 64, 256, 10);
}

/// rec 1x1 热点的两种形态对决（2026-11-04，预打包立项判决）：
///   A 现行：C[M=1536, N=30] = W[1536,768] × X[768,30] —— np=1 窄面板
///   B 转置：C[R=510, N=1536] = Xᵀ[510,768] × Wᵀ[768,1536] —— np=48 宽面板
/// R = n·ohw（批 17 × 30 空间位）。判决门：B 显著快才动引擎。
#[test]
#[ignore]
fn rec_dense_transposed_shootout() {
    use qppocr_kernels::activation::Activation;
    let k = 768usize;
    // A：现行批内路径（M=1536 权重行 × N=30，串行面板内核，模拟一个 chunk≈M/2）
    let a_w = vec![0.01f32; 1536 * k];
    let a_x = vec![0.02f32; k * 30];
    let mut a_c = vec![0f32; 1536 * 30];
    sgemm_serial(
        &a_w,
        &a_x,
        &mut a_c,
        1536,
        30,
        k,
        30,
        None,
        &Activation::default(),
    );
    let t0 = std::time::Instant::now();
    for _ in 0..20 {
        sgemm_serial(
            &a_w,
            &a_x,
            &mut a_c,
            1536,
            30,
            k,
            30,
            None,
            &Activation::default(),
        );
    }
    let a_ms = t0.elapsed().as_secs_f64() * 1000.0 / 20.0;
    let flop_a = 2.0 * 1536.0 * 30.0 * k as f64;
    println!(
        "A 现行  M=1536 N=30 K=768: {a_ms:8.2} ms ({:7.1} GFLOP/s)",
        flop_a / a_ms
    );

    // B：转置宽面板（并行，np=48）
    let b_x = vec![0.02f32; 510 * k];
    let b_wt = vec![0.01f32; k * 1536];
    let mut b_c = vec![0f32; 510 * 1536];
    sgemm(
        &b_x,
        &b_wt,
        &mut b_c,
        510,
        1536,
        k,
        1536,
        None,
        &Activation::default(),
    );
    let t0 = std::time::Instant::now();
    for _ in 0..20 {
        sgemm(
            &b_x,
            &b_wt,
            &mut b_c,
            510,
            1536,
            k,
            1536,
            None,
            &Activation::default(),
        );
    }
    let b_ms = t0.elapsed().as_secs_f64() * 1000.0 / 20.0;
    let flop_b = 2.0 * 510.0 * 1536.0 * k as f64;
    println!(
        "B 转置  R=510 N=1536 K=768: {b_ms:8.2} ms ({:7.1} GFLOP/s)",
        flop_b / b_ms
    );

    // B'：转置但串行（面板内核单线程，对照并行收益）
    let t0 = std::time::Instant::now();
    for _ in 0..20 {
        sgemm_serial(
            &b_x,
            &b_wt,
            &mut b_c,
            510,
            1536,
            k,
            1536,
            None,
            &Activation::default(),
        );
    }
    let bs_ms = t0.elapsed().as_secs_f64() * 1000.0 / 20.0;
    println!(
        "B' 转置串行:              {bs_ms:8.2} ms ({:7.1} GFLOP/s)",
        flop_b / bs_ms
    );
    let _ = std::hint::black_box((&a_c, &b_c));
}

/// 转置形态在 4 路并行（模拟 shard 池参与者数）下的效率——
/// 决定「分片内肥批次」设计的可行性。set_threads 必须先于池构建。
#[test]
#[ignore]
fn transposed_at_shard_width() {
    use qppocr_kernels::activation::Activation;
    qppocr_kernels::par::set_threads(4);
    let k = 768usize;
    // 肥批次（grain 32 行）：R = 32×30 = 960
    for r in [510usize, 960usize] {
        let x = vec![0.02f32; r * k];
        let wt = vec![0.01f32; k * 1536];
        let mut c = vec![0f32; r * 1536];
        sgemm(
            &x,
            &wt,
            &mut c,
            r,
            1536,
            k,
            1536,
            None,
            &Activation::default(),
        );
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            sgemm(
                &x,
                &wt,
                &mut c,
                r,
                1536,
                k,
                1536,
                None,
                &Activation::default(),
            );
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
        let flop = 2.0 * r as f64 * 1536.0 * k as f64;
        println!(
            "R={r} 4路并行: {ms:8.2} ms ({:7.1} GFLOP/s)",
            flop / ms / 1000.0
        );
        let _ = std::hint::black_box(&c);
    }
}

/// np=1 形状的 mrows vs 填充面板对决（2026-11-04）：rec 热点在默认
/// batch=1 下是 M=1536 N=30 K=768 —— np=1 落 mrows 标量。把 X 右填充到
/// 64 列（np=2 干净面板）后走面板内核：浪费 ≤2× FLOPs，前 30 列逐位
/// 相同。4 路并行 = shard 池参与者数。
#[test]
#[ignore]
fn np1_pad_shootout() {
    use qppocr_kernels::activation::Activation;
    qppocr_kernels::par::set_threads(4);
    let (m, n, k) = (1536usize, 30usize, 768usize);
    let a = vec![0.01f32; m * k];
    let b = vec![0.02f32; k * n];
    let mut c = vec![0f32; m * n];
    sgemm(&a, &b, &mut c, m, n, k, n, None, &Activation::default());
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        sgemm(&a, &b, &mut c, m, n, k, n, None, &Activation::default());
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!(
        "mrows  n=30: {ms:8.2} ms ({:6.1} GFLOP/s)",
        2.0 * m as f64 * n as f64 * k as f64 / 1e9 / ms
    );

    let b64 = vec![0.02f32; k * 64];
    let mut c64 = vec![0f32; m * 64];
    sgemm(
        &a,
        &b64,
        &mut c64,
        m,
        64,
        k,
        64,
        None,
        &Activation::default(),
    );
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        sgemm(
            &a,
            &b64,
            &mut c64,
            m,
            64,
            k,
            64,
            None,
            &Activation::default(),
        );
    }
    let ms64 = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!(
        "面板   n=64: {ms64:8.2} ms ({:6.1} GFLOP/s)  [含 34 列填充浪费]",
        2.0 * m as f64 * 64.0 * k as f64 / 1e9 / ms64
    );
    let _ = std::hint::black_box((&c, &c64));
}
