//! conv 双边对拍：Rust 侧（与 /tmp/conv_diff.cpp 同 shape 同数据同 9 轮取最好）。
//! cargo run --release -p qppocr-kernels --example conv_bench -- <case-idx>

use qppocr_kernels::activation::Activation;
use qppocr_kernels::buf::F32Buf;
use qppocr_kernels::conv::{ConvParams, conv2d};

struct Case {
    name: &'static str,
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
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let idx: usize = args[1].parse().unwrap();
    let cases = [
        Case {
            name: "dw 32->32 k3x3 @80x240 g32",
            c: 32,
            m: 32,
            h: 80,
            w: 240,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            ph: 1,
            pw: 1,
            group: 32,
        },
        Case {
            name: "dw 48->48 k3x3 @40x120 g48",
            c: 48,
            m: 48,
            h: 40,
            w: 120,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            ph: 1,
            pw: 1,
            group: 48,
        },
        Case {
            name: "dw 64->64 k5x5 @80x240 g64",
            c: 64,
            m: 64,
            h: 80,
            w: 240,
            kh: 5,
            kw: 5,
            sh: 1,
            sw: 1,
            ph: 2,
            pw: 2,
            group: 64,
        },
        Case {
            name: "im2col 3->16 k3x3 @320x960 s2 p1",
            c: 3,
            m: 16,
            h: 320,
            w: 960,
            kh: 3,
            kw: 3,
            sh: 2,
            sw: 2,
            ph: 1,
            pw: 1,
            group: 1,
        },
        Case {
            name: "im2col 64->16 k3x3 @80x240 p1",
            c: 64,
            m: 16,
            h: 80,
            w: 240,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            ph: 1,
            pw: 1,
            group: 1,
        },
        Case {
            name: "im2col 32->16 k3x3 @160x480 s2 p1",
            c: 32,
            m: 16,
            h: 160,
            w: 480,
            kh: 3,
            kw: 3,
            sh: 2,
            sw: 2,
            ph: 1,
            pw: 1,
            group: 1,
        },
    ];
    let c = &cases[idx];

    // mt19937(777) + uniform_real_distribution<float>(-1,1) 的位级复刻做不到，
    // 但 bench 不对数值——数据任意即可（两边各自随机）。
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        let v = rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (((v >> 40) as i64 - (1 << 23)) as f64 / (1i64 << 23) as f64) as f32
    };
    let x: Vec<f32> = (0..c.c * c.h * c.w).map(|_| next()).collect();
    let w: Vec<f32> = (0..c.m * (c.c / c.group) * c.kh * c.kw)
        .map(|_| next())
        .collect();
    let bias: Vec<f32> = vec![0.0; c.m];

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
    let mut best = f64::MAX;
    for _ in 0..9 {
        let t0 = std::time::Instant::now();
        conv2d(
            &x,
            &[1, c.c as i64, c.h as i64, c.w as i64],
            &w,
            &[c.m as i64, (c.c / c.group) as i64, c.kh as i64, c.kw as i64],
            Some(&bias),
            &params,
            &Activation::default(),
            &mut y,
        );
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        best = best.min(ms);
    }
    let chk = y[y.len() / 2];
    println!("rust {:<36} best {best:8.3} ms  chk={chk:.6}", c.name);
}
