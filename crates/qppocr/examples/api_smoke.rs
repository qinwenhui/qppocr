//! 门面 API 冒烟：三行 API + builder + 预设 + SHA 校验（需要本地模型）。
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t0 = std::time::Instant::now();
    let engine = qppocr::Engine::new(qppocr::Tier::Tiny, "models/")?;
    println!("load: {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);
    let t0 = std::time::Instant::now();
    let img = std::env::args()
        .nth(1)
        .expect("用法: api_smoke <图片路径>")
        .parse::<std::path::PathBuf>()
        .unwrap();
    let out = engine.run_image_file(img)?;
    println!("run: {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);
    for line in &out.lines {
        println!("{:.2}\t{}", line.confidence, line.text);
    }
    println!(
        "config: rec_height={} det_thresh={}",
        engine.config().rec_height,
        engine.config().det_thresh
    );
    // builder + preset
    let e2 = qppocr::Engine::builder()
        .tier(qppocr::Tier::Tiny)
        .preset(qppocr::Preset::Speed)
        .threads(8)
        .advanced(|a| a.det_thresh = 0.25)
        .build("models/")?;
    println!(
        "speed preset: rec_height={} retry={} det_thresh={}",
        e2.config().rec_height,
        e2.config().retry_conf,
        e2.config().det_thresh
    );
    Ok(())
}
