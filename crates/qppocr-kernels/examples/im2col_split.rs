//! im2col 路径分解计时：im2col 写 cols vs sgemm 读 cols。
//! cargo run --release -p qppocr-kernels --example im2col_split

use qppocr_kernels::activation::Activation;
use qppocr_kernels::buf::F32Buf;
use qppocr_kernels::conv::{ConvParams, conv2d};
use qppocr_kernels::gemm::{im2col, sgemm_serial};

fn main() {
    // case: 3->16 k3x3 @320x960 s2 p1（1.93x 差距的那个）
    let (c, m, h, w, kh, kw, sh, sw, ph, pw) = (3usize, 16usize, 320, 960, 3, 3, 2, 2, 1, 1);
    let (oh, ow) = ((h + 2 * ph - kh) / sh + 1, (w + 2 * pw - kw) / sw + 1);
    let kk = c * kh * kw;
    let ohw = oh * ow;

    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        let v = rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (((v >> 40) as i64 - (1 << 23)) as f64 / (1i64 << 23) as f64) as f32
    };
    let x: Vec<f32> = (0..c * h * w).map(|_| next()).collect();
    let wt: Vec<f32> = (0..m * kk).map(|_| next()).collect();

    // 复刻 conv.rs 的 tile 划分
    let col_budget = 1usize << 19;
    let nchunks = 1usize.max(qppocr_kernels::par::threads().min(oh.max(1)));
    let mut tile = 1usize.max(col_budget / (kk * ow).max(1) / nchunks);
    tile = tile.min(oh);
    {
        let ng = 1;
        let want = 1usize.max(qppocr_kernels::par::threads().div_ceil(ng));
        tile = tile.min(1usize.max(oh.div_ceil(want)));
    }
    let ntiles = oh.div_ceil(tile);
    println!(
        "tile={tile} ntiles={ntiles} threads={}",
        qppocr_kernels::par::threads()
    );

    // 整体（对照）
    let params = ConvParams {
        sh,
        sw,
        ph,
        pw,
        peh: ph,
        pew: pw,
        dh: 1,
        dw: 1,
        group: 1,
    };
    let mut y = F32Buf::new();
    let mut best = f64::MAX;
    for _ in 0..9 {
        let t0 = std::time::Instant::now();
        conv2d(
            &x,
            &[1, c as i64, h as i64, w as i64],
            &wt,
            &[m as i64, c as i64, kh as i64, kw as i64],
            None,
            &params,
            &Activation::default(),
            &mut y,
        );
        best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
    }
    println!("conv2d total: {best:.3} ms");

    // im2col-only（全部 tiles 的 cols 写入）
    let mut cols: Vec<f32> = vec![0.0; kk * tile * ow];
    let mut best = f64::MAX;
    for _ in 0..9 {
        let t0 = std::time::Instant::now();
        for t in 0..ntiles {
            let oy0 = t * tile;
            let rows = tile.min(oh - oy0);
            im2col(
                &x, c, h, w, kh, kw, sh, sw, ph, pw, ow, oy0, rows, &mut cols,
            );
        }
        best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
    }
    println!("im2col all tiles (serial per-tile): {best:.3} ms");

    // sgemm-only（用一份现成 cols；输出写到能容纳整个图的缓冲）
    let mut yfull = F32Buf::with_zeroed(m * ohw);
    let act = Activation::default();
    let mut best = f64::MAX;
    for _ in 0..9 {
        let t0 = std::time::Instant::now();
        for t in 0..ntiles {
            let oy0 = t * tile;
            let rows = tile.min(oh - oy0);
            let len = (m - 1) * ohw + rows * ow;
            sgemm_serial(
                &wt,
                &cols,
                &mut yfull[..len],
                m,
                rows * ow,
                kk,
                ohw,
                None,
                &act,
            );
        }
        best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
    }
    println!("sgemm all tiles: {best:.3} ms");
}
