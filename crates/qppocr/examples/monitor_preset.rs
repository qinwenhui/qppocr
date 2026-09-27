//! 监控截图预设完整配方：白字压在栏杆/栅栏纹理上的场景。
//!
//! - `unclip_perp=0.5`：收紧 unclip 的垂直扩张，不把背景纹理吃进裁剪框
//! - `unclip_margin_thresh=0.45`：框内文字区之外的边跟能量超阈值时重建框
//! - `detect_orientation(false)`：此类时间戳/编号行本就正立，绕开方向
//!   分类器在极端长宽比裁剪上的误判（详见 example 内注释与 docs）
//!
//! 运行：cargo run --release -p qppocr --example monitor_preset -- <图> [模型目录]
fn main() {
    let img_path = std::env::args()
        .nth(1)
        .expect("用法: monitor_preset <图片路径> [模型目录]");
    let models = std::env::args().nth(2).unwrap_or_else(|| "models".into());
    let img = qppocr::decode_file(&img_path).unwrap();
    let eng = qppocr::Engine::builder()
        .tier(qppocr::Tier::Small)
        .detect_orientation(false)
        .advanced(|a| {
            a.unclip_perp = 0.5;
            a.unclip_margin_thresh = 0.45;
        })
        .build(&models)
        .unwrap();
    let r = eng.run(&img).unwrap();
    for l in &r.lines {
        println!("{:?} conf={:.2} rot={}", l.text, l.confidence, l.rotation);
    }
}
