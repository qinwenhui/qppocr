//! 引擎侧修复的端到端验证（serde / cls 跳过 / PipelineConfig 导出 / Debug）。
//! 运行：cargo run --release -p qppocr --features serde --example api_verify
use qppocr::{Advanced, Config, Engine, PipelineConfig, Preset, Tier};

fn main() -> anyhow_lite::Result<()> {
    // 1) PipelineConfig 可命名 + Debug
    let engine = Engine::new(Tier::Tiny, "models")?;
    let cfg: &PipelineConfig = engine.config();
    println!("1 PipelineConfig 导出+可命名: upscale={}", cfg.upscale);
    println!("1 Engine Debug 摘要: {:?}", &engine as &dyn std::fmt::Debug);

    // 2) detect_orientation 默认 true：倒置翻正路径可用
    let img: std::path::PathBuf = std::env::args()
        .nth(1)
        .expect("用法: api_verify <图片路径>")
        .parse()
        .unwrap();
    let r = engine.run_image_file(&img)?;
    println!(
        "2 默认 detect_orientation=true: cls_ms={:.2} lines={}",
        r.timings.cls_ms,
        r.lines.len()
    );

    // 3) serde：结果往返
    let json = serde_json::to_string(&r)?;
    let back: qppocr::OcrResult = serde_json::from_str(&json)?;
    assert_eq!(back.lines.len(), r.lines.len());
    assert_eq!(back.lines[0].text, r.lines[0].text);
    println!("3 OcrResult serde 往返: {} bytes, 行数一致", json.len());

    // 4) Advanced 部分 JSON = 预设覆盖语义（缺省字段取基准值）
    let adv: Advanced = serde_json::from_str(r#"{"upscale": 2}"#)?;
    assert_eq!(adv.upscale, 2);
    assert_eq!(adv.det_thresh, 0.2); // 未给 → 默认
    println!(
        "4 Advanced 部分 JSON: upscale={} det_thresh={}",
        adv.upscale, adv.det_thresh
    );

    // 5) Tier/Preset 小写字符串
    assert_eq!(serde_json::to_string(&Tier::Small)?, "\"small\"");
    assert_eq!(serde_json::to_string(&Preset::Balanced)?, "\"balanced\"");
    println!("5 Tier/Preset 序列化为 \"small\"/\"balanced\"");

    // 6) Config 反序列化（Option 字段缺省）
    let c: Config = serde_json::from_str(r#"{"threads": 4}"#)?;
    assert_eq!(c.threads, 4);
    assert!(c.detect_orientation); // Default 提供
    println!("6 Config 部分 JSON: threads=4, detect_orientation 默认 true");

    // 7) detect_orientation(false) = 真跳过 cls（cls_ms 必为 0）
    let off = Engine::builder()
        .tier(Tier::Tiny)
        .detect_orientation(false)
        .build("models")?;
    let r2 = off.run_image_file(&img)?;
    assert_eq!(r2.timings.cls_ms, 0.0);
    assert_eq!(r2.num_flipped, 0);
    println!("7 关方向分类: cls_ms=0, num_flipped=0（阶段真跳过）");
    println!("ALL-OK");
    Ok(())
}
// 本地 mini-result（避免 example 引 anyhow）
mod anyhow_lite {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
}
