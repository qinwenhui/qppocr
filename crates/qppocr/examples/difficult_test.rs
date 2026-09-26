//! 难图.png A/B:默认 vs 监控截图参数(unclip_perp=0.5, unclip_margin_thresh=0.45)
fn main() {
    let img_path = std::env::args()
        .nth(1)
        .expect("用法: difficult_test <图片路径>");
    let img = qppocr::decode_file(&img_path).unwrap();
    // 增加 small 档测试 + rec_height 变体
    for &(tier_s, tag2, perp, margin) in &[
        ("tiny", "监控tiny  ", 0.5, 0.45),
        ("small", "监控small", 0.5, 0.45),
    ] {
        let tier = if tier_s == "tiny" {
            qppocr::Tier::Tiny
        } else {
            qppocr::Tier::Small
        };
        let eng = qppocr::Engine::builder()
            .tier(tier)
            .advanced(move |a| {
                a.unclip_perp = perp;
                a.unclip_margin_thresh = margin;
            })
            .build("models")
            .unwrap();
        let r = eng.run(&img).unwrap();
        println!("{}: lines={}", tag2, r.lines.len());
        for l in &r.lines {
            println!("   {:?} conf={:.2}", l.text, l.confidence);
        }
    }
    let cases: &[(&str, f32, f32, i32, u32, bool)] = &[
        ("A 默认          perp1.0 marg0  ", 1.0, 0.0, 1, 960, false),
        ("B 仅perp0.5     marg0          ", 0.5, 0.0, 1, 960, false),
        ("C 仅marg.45     perp1.0        ", 1.0, 0.45, 1, 960, false),
        ("D 两者          perp0.5 marg.45", 0.5, 0.45, 1, 960, false),
        ("E perp0.0(只撑长) marg0        ", 0.0, 0.0, 1, 960, false),
    ];
    for &(tag, perp, margin, up, msl, eh) in cases {
        let eng = qppocr::Engine::builder()
            .tier(qppocr::Tier::Tiny)
            .advanced(move |a| {
                a.unclip_perp = perp;
                a.unclip_margin_thresh = margin;
                a.upscale = up;
                a.max_side_len = msl as i32;
                a.enhance_contrast = eh;
            })
            .build("models")
            .unwrap();
        let r = eng.run(&img).unwrap();
        println!("{}: boxes={} lines={}", tag, r.num_boxes, r.lines.len());
        for l in &r.lines {
            println!("   {:?} conf={:.2}", l.text, l.confidence);
        }
    }
}
