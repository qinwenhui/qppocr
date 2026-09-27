//! 纯寄存器 FMA 峰值：不碰内存，量本机单核 AVX2+FMA 的理论上限。
//! 用来判断 sgemm 离峰值还有多远（若 sgemm 只有峰值的一半，那是内核问题）。
use std::arch::x86_64::*;

#[target_feature(enable = "avx2,fma")]
unsafe fn loop_fma(iters: usize) -> f32 {
    let mut a = _mm256_set1_ps(1.000001);
    let mut b = _mm256_set1_ps(0.999999);
    let mut c = _mm256_set1_ps(1.000002);
    let mut d = _mm256_set1_ps(0.999998);
    let mut e = _mm256_set1_ps(1.000003);
    let mut f = _mm256_set1_ps(0.999997);
    let mut g = _mm256_set1_ps(1.000004);
    let mut h = _mm256_set1_ps(0.999996);
    let k = _mm256_set1_ps(1e-9);
    for _ in 0..iters {
        a = std::hint::black_box(_mm256_fmadd_ps(a, k, a));
        b = _mm256_fmadd_ps(b, k, b);
        c = _mm256_fmadd_ps(c, k, c);
        d = _mm256_fmadd_ps(d, k, d);
        e = _mm256_fmadd_ps(e, k, e);
        f = _mm256_fmadd_ps(f, k, f);
        g = _mm256_fmadd_ps(g, k, g);
        h = _mm256_fmadd_ps(h, k, h);
    }
    let s = _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d));
    let t = _mm256_add_ps(_mm256_add_ps(e, f), _mm256_add_ps(g, h));
    _mm256_cvtss_f32(_mm256_add_ps(s, t))
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000_000);
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        // SAFETY: loop_fma 只做纯算术，无内存副作用。
        let v = unsafe { loop_fma(n) };
        let s = t0.elapsed().as_secs_f64();
        best = best.min(s);
        std::hint::black_box(v);
    }
    // 每次迭代 8 条 FMA × 8 lane × 2 flop = 128 flop
    let flops = (n as f64) * 128.0;
    println!(
        "单核 AVX2+FMA 峰值 ≈ {:.1} GFLOPS = {:.1} GMAC/s（{:.2} GHz 等效）",
        flops / best / 1e9,
        flops / best / 2e9,
        flops / best / 32e9
    );
    println!("对照：sgemm 单线程实测 42 GMAC/s（bench_gemm 82-97 GFLOPS）");
}
