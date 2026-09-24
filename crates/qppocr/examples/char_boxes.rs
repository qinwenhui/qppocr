//! 每字坐标（TextLine::chars）的几何验证：
//! 1. 每行 chars 数 == text 字符数（含插入空格）
//! 2. 字框落在行框的外扩包围盒内（原图坐标系）
//! 3. 横排行的字框沿阅读方向单调递增；翻转行（rotation=180）同样（映射已还原）
//!
//! 运行：cargo run --release -p qppocr --example char_boxes -- <img>...
fn main() {
    let eng = qppocr::Engine::new(qppocr::Tier::Tiny, "models").unwrap();
    let mut bad = 0usize;
    for path in std::env::args().skip(1) {
        let r = eng.run_image_file(&path).unwrap();
        let name = std::path::Path::new(&path)
            .file_name()
            .unwrap()
            .to_string_lossy();
        println!("== {name} ({} 行)", r.lines.len());
        for l in &r.lines {
            let n = l.text.chars().count();
            if n == 0 {
                continue;
            }
            if l.chars.len() != n {
                println!("  [X] 数量不符: text {n} vs chars {}", l.chars.len());
                bad += 1;
                continue;
            }
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for p in &l.pts {
                x0 = x0.min(p[0]);
                y0 = y0.min(p[1]);
                x1 = x1.max(p[0]);
                y1 = y1.max(p[1]);
            }
            let m = (y1 - y0).max(1.0) * 0.3;
            let mut inside = true;
            for cs in &l.chars {
                for p in &cs.pts {
                    if !(x0 - m..=x1 + m).contains(&p[0]) || !(y0 - m..=y1 + m).contains(&p[1]) {
                        inside = false;
                    }
                }
            }
            if !inside {
                println!("  [X] 字框越界: \"{}\"", l.text);
                bad += 1;
            }
            // 横排文本 = 行框的宽边（TL→TR）长于高边（TL→BL）；竖排不查单调
            let w = (l.pts[0][0] - l.pts[1][0]).hypot(l.pts[0][1] - l.pts[1][1]);
            let h = (l.pts[0][0] - l.pts[3][0]).hypot(l.pts[0][1] - l.pts[3][1]);
            let horiz = w > h;
            if horiz && n >= 2 {
                // 沿阅读方向（TL→TR 单位向量）投影首角点，斜线也该单调
                let (dx, dy) = (l.pts[1][0] - l.pts[0][0], l.pts[1][1] - l.pts[0][1]);
                let len = dx.hypot(dy).max(1e-6);
                let (ux, uy) = (dx / len, dy / len);
                let lefts: Vec<f32> = l
                    .chars
                    .iter()
                    .map(|c| c.pts[0][0] * ux + c.pts[0][1] * uy)
                    .collect();
                // 180° 翻转行的阅读顺序在原图里从 pts[1] 向 pts[0]（倒置文本
                // 的首字符在几何右侧）——方向取反
                let mono = if l.rotation == 180 {
                    lefts.windows(2).all(|w| w[1] <= w[0] + 1.0)
                } else {
                    lefts.windows(2).all(|w| w[1] >= w[0] - 1.0)
                };
                if !mono {
                    println!("  [X] 非单调: \"{}\" {:?}", l.text, lefts);
                    bad += 1;
                }
            }
            let joined: String = l
                .chars
                .iter()
                .map(|c| c.text.clone())
                .collect::<Vec<_>>()
                .join("|");
            println!(
                "  [{}] rot={} \"{}\" 首字({:.0},{:.0}) 尾字({:.0},{:.0})",
                if bad == 0 { "ok" } else { "??" },
                l.rotation,
                joined.chars().take(15).collect::<String>(),
                l.chars[0].pts[0][0],
                l.chars[0].pts[0][1],
                l.chars[n - 1].pts[0][0],
                l.chars[n - 1].pts[0][1],
            );
        }
    }
    if bad > 0 {
        println!("FAILED: {bad}");
        std::process::exit(1);
    }
    println!("ALL-OK");
}
