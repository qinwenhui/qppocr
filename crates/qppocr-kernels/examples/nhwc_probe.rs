//! NHWC 1x1 卷积 vs NCHW 1x1 卷积：同一个形状、同一份数据，只换内存布局。
//!
//! NHWC 形式 `Y[p][co] = Σ_ci X[p][ci] · Wt[ci][co]` 正好是 sgemm 的
//! `C[m][n] = A[m][k]·B[k][n]`（m=像素块、n=co、k=ci），A 的 k 维连续。
//! NCHW 形式 `Y[co][p] = Σ_ci W[co][ci]·X[ci][p]` 里 B 的 k 维按 `plane`
//! 步长跳——每个 k 一个页，TLB 压力大。
use qppocr_kernels::activation::Activation;
use qppocr_kernels::buf::F32Buf;
use qppocr_kernels::gemm::sgemm_serial;

/// (名字, ci, co, 像素数)
const CASES: &[(&str, usize, usize, usize)] = &[
    ("det 16->32  @240x216", 16, 32, 240 * 216),
    ("det 32->64  @240x216", 32, 64, 240 * 216),
    ("det 64->16  @240x216", 64, 16, 240 * 216),
    ("det 128->64 @60x54", 128, 64, 60 * 54),
    ("det 384->384 @30x27", 384, 384, 30 * 27),
    ("rec 384->192 @12x264", 384, 192, 12 * 264),
    ("rec 192->384 @12x264", 192, 384, 12 * 264),
    ("rec 96->192 @12x64", 96, 192, 12 * 64),
];

fn main() {
    let threads: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    qppocr_kernels::par::set_threads(threads);
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
        let v = rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (((v >> 40) as i64 - (1 << 23)) as f64 / (1i64 << 23) as f64) as f32
    };
    const BLK: usize = 128; // 一次处理的像素块（NCHW 侧是 n 面板块，NHWC 侧是 m 块）
    println!("threads = {threads}   像素块 = {BLK}\n");
    println!("{:<24}{:>10}{:>10}{:>8}", "形状", "NCHW", "NHWC", "提速");
    let (mut sa, mut sb) = (0.0f64, 0.0f64);
    for &(name, ci, co, npix) in CASES {
        // NCHW：X[ci][npix]，权重 W[co][ci]
        let x_nchw: Vec<f32> = (0..ci * npix).map(|_| next()).collect();
        let w: Vec<f32> = (0..co * ci).map(|_| next()).collect();
        let bias: Vec<f32> = (0..co).map(|_| next()).collect();
        let mut y_nchw = F32Buf::new();

        let mut best_nchw = f64::MAX;
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            // SAFETY: sgemm 写满每个输出元素。
            unsafe { y_nchw.resize_uninit(co * npix) };
            let mut p0 = 0;
            while p0 < npix {
                let w_n = BLK.min(npix - p0);
                sgemm_serial(
                    &w, &x_nchw[p0..], y_nchw.as_mut_slice(), co, w_n, ci, npix,
                    Some(&bias), &Activation::default(),
                );
                p0 += BLK;
            }
            best_nchw = best_nchw.min(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // NHWC：X[npix][ci]，权重转置成 Wt[ci][co]
        let x_nhwc: Vec<f32> = (0..npix * ci).map(|_| next()).collect();
        let mut wt = vec![0f32; ci * co];
        for c in 0..ci {
            for o in 0..co {
                wt[c * co + o] = w[o * ci + c];
            }
        }
        let mut y_nhwc = F32Buf::new();
        let mut best_nhwc = f64::MAX;
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            // SAFETY: sgemm 写满每个输出元素。
            unsafe { y_nhwc.resize_uninit(npix * co) };
            let mut p0 = 0;
            while p0 < npix {
                let m = BLK.min(npix - p0);
                // ⚠ sgemm 的 bias 是按**行**加的；NHWC 的 bias 按列（输出通道），
                //   所以这里传 None，卷积后单独补一趟——这一趟的成本要算进去。
                sgemm_serial(
                    &x_nhwc[p0 * ci..],
                    &wt,
                    &mut y_nhwc.as_mut_slice()[p0 * co..],
                    m, co, ci, co, None, &Activation::default(),
                );
                let seg = &mut y_nhwc.as_mut_slice()[p0 * co..(p0 + m) * co];
                for row in seg.chunks_mut(co) {
                    for (v, &bv) in row.iter_mut().zip(&bias) {
                        *v += bv;
                    }
                }
                p0 += BLK;
            }
            best_nhwc = best_nhwc.min(t0.elapsed().as_secs_f64() * 1000.0);
        }
        sa += best_nchw; sb += best_nhwc;
        println!("{name:<24}{best_nchw:10.3}{best_nhwc:10.3}{:8.2}", best_nchw / best_nhwc);
    }
    println!("\n{:<24}{sa:10.2}{sb:10.2}{:8.2}", "—— 合计", sa / sb);
}
