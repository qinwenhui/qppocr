//! 量「权重存成 fp16」的精度代价：同一份图、同一套旋钮，**只换一个自变量**。
//!
//! 用法： bench_f16w <模型目录> <tiny|small|medium> <图>...
//!
//! 输出每行一条 TSV，便于两次运行直接 diff：
//!   <图名>\t<行号>\tconf\trot\t文本
//!
//! 自变量现在有两条路，都靠**跑两次、只动一处**来隔离：
//!
//! 1. **换模型目录**——`bench/gpuhealth/mk_f16w.py` 把权重整体改写成
//!    fp16（零长度变化）。这条路 CPU 也能跑，量的是「所有初始值都窄
//!    一半」的**上界**。
//! 2. **`QPPOCR_METAL_WF16=1`**——生产旋钮（Metal-only），只把走
//!    `n_conv8` 的层压成 fp16。这是**真实会发货**的那条路，必须在
//!    `QPPOCR_BENCH_DEVICE=metal` 下才有效（CPU 会话读不到这个旋钮，
//!    两条路会逐字节相同——这正是它容易被误当成「没代价」的地方）。
//!
//! `QPPOCR_BENCH_DEVICE`：`cpu`（默认，保持历史读数可比）| `metal` | `gpu`。
//!
//! 之所以走库 API 而不是 CLI：产物模型的 SHA-256 必然对不上官方原件，
//! CLI 没有跳过校验的开关（这是对的，不该为一次测量在生产代码上开口子），
//! 而 `EngineBuilder::verify_sha256(false)` 本来就是给这种场合留的。

use std::path::Path;

/// 计算设备：`QPPOCR_BENCH_DEVICE` = `cpu`（默认）| `metal` | `gpu`。
///
/// 默认**保持 CPU**：本工具的历史读数（2026-10-08 的精度三档）都是在
/// CPU 上取的，默认值一改，那些数就不能再和新数直接比了。
fn bench_device() -> qppocr::DeviceChoice {
    match std::env::var("QPPOCR_BENCH_DEVICE").as_deref() {
        Ok("metal") => qppocr::DeviceChoice::Gpu {
            api: qppocr::GpuApi::Metal,
            index: None,
        },
        Ok("gpu") => qppocr::DeviceChoice::gpu(),
        Ok("cpu") | Err(_) => qppocr::DeviceChoice::default(),
        Ok(other) => panic!("QPPOCR_BENCH_DEVICE={other:?}（cpu|metal|gpu）"),
    }
}

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
        .device(bench_device())
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
