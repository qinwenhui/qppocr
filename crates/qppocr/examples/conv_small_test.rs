//! 转换版 small 模型 × 本引擎：与参考实现同模型同旋钮的 A/B。
fn main() {
    let img = qppocr::decode_file(
        std::env::args()
            .nth(1)
            .expect("用法: conv_small_test <图片路径> [模型目录（默认 models）]"),
    )
    .unwrap();
    let models_dir = std::env::args().nth(2).unwrap_or_else(|| "models".into());
    for &(tag, perp, margin, retry, win, cls) in &[
        ("监控预设+cls ", 0.5, 0.45, 0.0, false, true),
        ("监控预设+开窗", 0.5, 0.45, 0.0, true, true),
        ("监控预设+关cls", 0.5, 0.45, 0.0, false, false),
    ] {
        let eng = qppocr::Engine::builder()
            .tier(qppocr::Tier::Small)
            .detect_orientation(cls)
            .advanced(move |a| {
                a.unclip_perp = perp;
                a.unclip_margin_thresh = margin;
                a.retry_conf = retry;
                a.cls_window = win;
            })
            .verify_sha256(false)
            .build(&models_dir)
            .unwrap();
        let r = eng.run(&img).unwrap();
        println!("{tag} 框: {:?}", r.lines[0].pts);
        println!(
            "{tag}: {:?} conf={:.2} decluttered={} retried={}",
            r.lines.iter().map(|l| l.text.clone()).collect::<Vec<_>>(),
            r.lines.first().map(|l| l.confidence).unwrap_or(0.0),
            r.num_decluttered,
            r.num_det_retried
        );
        println!(
            "flipped={} rotation={}",
            r.num_flipped,
            r.lines.first().map(|l| l.rotation).unwrap_or(0)
        );
    }
}
