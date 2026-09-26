//! conv 独立基准：**模型里真实存在的那些形状**，固定数据、「9 轮取最好」。
//!
//! cargo run --release -p qppocr-kernels --example conv_bench [-- <线程数>]
//!
//! 形状取自 PP-OCRv6 tiny/small 的 det（`QPPOCR_PROF=1` 的逐算子表）。
//! **单线程数字才是内核效率的判据**——多线程下算子互相争核，逐算子耗时
//! 没有可比性。默认 1 线程。

use qppocr_kernels::activation::Activation;
use qppocr_kernels::buf::F32Buf;
use qppocr_kernels::conv::{ConvParams, conv2d};

struct Case {
    name: &'static str,
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    s: usize,
    group: usize,
}

/// 输入尺寸由**输出**尺寸反推：`oh = (h + 2p - k)/s + 1`，`p=(k-1)/2`。
const fn case(
    name: &'static str,
    c: usize,
    oh: usize,
    ow: usize,
    k: usize,
    s: usize,
    group: usize,
) -> Case {
    let p = (k - 1) / 2;
    Case {
        name,
        c,
        h: (oh - 1) * s + k - 2 * p,
        w: (ow - 1) * s + k - 2 * p,
        k,
        s,
        group,
    }
}

fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    qppocr_kernels::par::set_threads(threads);

    #[rustfmt::skip]
    let cases = [
        // ---- tiny det 深度卷积（输出尺寸写在名字里）----
        case("dw 32->32  k3x3 @200x240 s1", 32, 200, 240, 3, 1, 32),
        case("dw 32->32  k3x3 @200x240 s2", 32, 200, 240, 3, 2, 32),
        case("dw 48->48  k3x3 @100x120 s1", 48, 100, 120, 3, 1, 48),
        case("dw 48->48  k3x3 @100x120 s2", 48, 100, 120, 3, 2, 48),
        case("dw 64->64  k3x3 @ 50x 60 s1", 64,  50,  60, 3, 1, 64),
        case("dw 64->64  k3x3 @ 50x 60 s2", 64,  50,  60, 3, 2, 64),
        case("dw 160->160 k3x3 @ 25x 30 s1", 160, 25, 30, 3, 1, 160),
        case("dw 64->64  k5x5 @200x240 s1", 64, 200, 240, 5, 1, 64),
        case("dw 64->64  k5x5 @100x120 s1", 64, 100, 120, 5, 1, 64),
        // ---- small det 深度卷积 ----
        case("dw 48->48  k3x3 @200x240 s1", 48, 200, 240, 3, 1, 48),
        case("dw 96->96  k3x3 @100x120 s1", 96, 100, 120, 3, 1, 96),
        case("dw 96->96  k3x3 @100x120 s2", 96, 100, 120, 3, 2, 96),
        case("dw 192->192 k3x3 @ 50x 60 s1", 192, 50, 60, 3, 1, 192),
        case("dw 192->192 k3x3 @ 50x 60 s2", 192, 50, 60, 3, 2, 192),
        case("dw 384->384 k3x3 @ 25x 30 s1", 384, 25, 30, 3, 1, 384),
        case("dw 96->96  k7x7 @200x240 s1", 96, 200, 240, 7, 1, 96),
        case("dw 96->96  k7x7 @ 50x 60 s1", 96,  50,  60, 7, 1, 96),
        // ---- 窄 N 的普通卷积（tiny det 最大的一项）----
        case("g1 64->16   k3x3 @200x240 s1", 64, 200, 240, 3, 1, 1),
    ];

    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        let v = rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (((v >> 40) as i64 - (1 << 23)) as f64 / (1i64 << 23) as f64) as f32
    };

    println!("threads = {threads}\n");
    let mut grand = 0.0;
    for c in &cases {
        let oh = c.h.div_ceil(c.s);
        let ow = c.w.div_ceil(c.s);
        let out_c = if c.group == 1 { c.c } else { c.c };
        let x: Vec<f32> = (0..c.c * c.h * c.w).map(|_| next()).collect();
        let w: Vec<f32> = (0..out_c * (c.c / c.group) * c.k * c.k)
            .map(|_| next())
            .collect();
        let bias: Vec<f32> = (0..out_c).map(|_| next()).collect();
        let params = ConvParams {
            sh: c.s,
            sw: c.s,
            ph: (c.k - 1) / 2,
            pw: (c.k - 1) / 2,
            peh: (c.k - 1) / 2,
            pew: (c.k - 1) / 2,
            dh: 1,
            dw: 1,
            group: c.group,
        };
        let mut y = F32Buf::new();
        let mut best = f64::MAX;
        for _ in 0..9 {
            let t0 = std::time::Instant::now();
            conv2d(
                &x,
                &[1, c.c as i64, c.h as i64, c.w as i64],
                &w,
                &[out_c as i64, (c.c / c.group) as i64, c.k as i64, c.k as i64],
                Some(&bias),
                &params,
                &Activation::default(),
                &mut y,
            );
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let macs = (out_c * c.k * c.k * oh * ow) as f64;
        // chk 只用来发现「输出全一样 / 压根没写」这类低级错误
        let chk = y[y.len() / 2];
        grand += best;
        println!(
            "{:<30} {best:8.3} ms  {:7.2} GMAC/s   chk={chk:.4}",
            c.name,
            macs / best / 1e6
        );
    }
    println!("\n{:<30} {grand:8.3} ms", "—— 合计");
}
