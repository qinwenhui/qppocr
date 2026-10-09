//! 量「权重存成 fp16」的精度代价：同一份图、同一套旋钮，**只换模型目录**。
//!
//! 用法： bench_f16w <模型目录> <tiny|small|medium> <图>...
//!
//! 输出每行一条 TSV，便于两次运行直接 diff：
//!   <图名>\t<行号>\tconf\trot\t文本
//!
//! 之所以走库 API 而不是 CLI：产物模型的 SHA-256 必然对不上官方原件，
//! CLI 没有跳过校验的开关（这是对的，不该为一次测量在生产代码上开口子），
//! 而 `EngineBuilder::verify_sha256(false)` 本来就是给这种场合留的。

use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let models = args
        .next()
        .expect("用法: bench_f16w <模型目录> <tier> <图>...");
    let tier = match args.next().as_deref() {
        Some("tiny") => qppocr::Tier::Tiny,
        Some("small") => qppocr::Tier::Small,
        Some("medium") => qppocr::Tier::Medium,
        other => panic!("未知档位 {other:?}（tiny|small|medium）"),
    };
    let imgs: Vec<String> = args.collect();
    assert!(!imgs.is_empty(), "至少要给一张图");

    let eng = qppocr::Engine::builder()
        .tier(tier)
        .verify_sha256(false)
        .build(&models)
        .expect("建引擎失败");

    for p in &imgs {
        let name = Path::new(p)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.clone());
        let r = match eng.run_image_file(p) {
            Ok(r) => r,
            Err(e) => {
                println!("{name}\tERR\t{e}");
                continue;
            }
        };
        // 空文本也要打印（行数本身就是判据之一），但 TSV 里留个占位。
        for (i, l) in r.lines.iter().enumerate() {
            let t = if l.text.is_empty() { "<空>" } else { &l.text };
            println!("{name}\t{i}\t{:.6}\t{}\t{}", l.confidence, l.rotation, t);
        }
        println!("{name}\t=\tlines={}", r.lines.len());
    }
}
