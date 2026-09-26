//! 图像级双线性缩放（u8 RGB）微基准。
//!
//! det 输入准备是 848×816 → 832×832 的全图缩放，曾经是 6.9 ms 的串行热点；
//! 现在走 `resize_bilinear_rgb_u8` 的逐行并行版。
//!
//! 运行：cargo run --release -p qppocr-kernels --example resize_bench

fn main() {
    let (sw, sh) = (848usize, 816usize);
    let src: Vec<u8> = (0..sw * sh * 3).map(|i| (i % 251) as u8).collect();
    for &(dw, dh, tag) in &[
        (832usize, 832usize, "det 输入 848x816->832x832"),
        (400, 48, "rec 单行 ->48x400"),
    ] {
        let mut dst = vec![0u8; dw * dh * 3];
        let mut best = f64::MAX;
        for i in 0..30 {
            let t0 = std::time::Instant::now();
            qppocr_kernels::resize::resize_bilinear_rgb_u8(&src, sw, sh, &mut dst, dw, dh);
            let el = t0.elapsed().as_secs_f64() * 1000.0;
            if i >= 5 {
                best = best.min(el);
            }
        }
        println!("{tag}: 最快 {best:.3} ms/次");
    }
}
