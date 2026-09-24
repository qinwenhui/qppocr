//! 端到端 OCR 驱动：`cargo run --release -p qppocr-core --example run_ocr --
//!              <image> [--models <dir>] [--dict <file>] [--json]`
//!
//! 默认模型目录 workspace 根的 models/（tiny 档上游原件 + 外部字典）。
//! `--dict` 缺省时按 `models/dict_tiny.txt` 或 rec 模型内嵌字典尝试。

use qppocr_core::pipeline::{Dictionary, Engine, PipelineConfig};

// 解码只用于 example：dev-dependency 的 image crate（产品 API 的解码
// 在门面 crate feature 门控）。
fn decode_any(path: &std::path::Path) -> Option<qppocr_core::pipeline::image::Image> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Some(qppocr_core::pipeline::image::Image {
        w: w as i32,
        h: h as i32,
        c: 3,
        orig_w: w as i32,
        orig_h: h as i32,
        data: rgb.into_raw(),
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!(
            "usage: run_ocr <image>... [--models <dir>] [--dict <file>] [--json] [--det-only]"
        );
        std::process::exit(2);
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let mut models_dir = root.join("models");
    let mut dict_path: Option<std::path::PathBuf> = None;
    let mut json = false;
    let mut det_only = false;
    let mut images: Vec<std::path::PathBuf> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--models" => {
                i += 1;
                models_dir = args[i].clone().into();
            }
            "--dict" => {
                i += 1;
                dict_path = Some(args[i].clone().into());
            }
            "--json" => json = true,
            "--det-only" => det_only = true,
            a => images.push(a.into()),
        }
        i += 1;
    }

    // 模型布局：models/{tiny,small,medium}/{det,rec}.onnx + cls.onnx；
    // 兼容基准实现的平铺布局（PP-OCRv6_*_tiny.onnx + ppocr_cls.onnx）
    let tier_dir = if models_dir.join("tiny/det.onnx").exists() {
        models_dir.join("tiny")
    } else {
        models_dir.clone()
    };
    let det_p = tier_dir.join("det.onnx");
    let det_p = if det_p.exists() {
        det_p
    } else {
        models_dir.join("PP-OCRv6_det_tiny.onnx")
    };
    let rec_p = tier_dir.join("rec.onnx");
    let rec_p = if rec_p.exists() {
        rec_p
    } else {
        models_dir.join("PP-OCRv6_rec_tiny.onnx")
    };
    let mut cls_p = models_dir.join("cls.onnx");
    if !cls_p.exists() {
        cls_p = models_dir.join("ppocr_cls.onnx");
    }
    let cls_p = if cls_p.exists() { Some(cls_p) } else { None };

    // 字典：显式 > tiny 外部字典 > 模型内嵌
    let dict = if let Some(p) = dict_path {
        Dictionary::Text(std::fs::read_to_string(p).expect("read dict"))
    } else if let Ok(t) = std::fs::read_to_string(models_dir.join("dict_tiny.txt")) {
        Dictionary::Text(t)
    } else if let Ok(t) = std::fs::read_to_string(models_dir.join("dict_small_medium.txt")) {
        Dictionary::Text(t)
    } else {
        Dictionary::Embedded
    };

    let t0 = std::time::Instant::now();
    let engine = Engine::open(
        &det_p,
        &rec_p,
        cls_p.as_deref(),
        dict,
        PipelineConfig::default(),
    )
    .expect("open engine");
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

    for img_path in &images {
        let img = match decode_any(img_path) {
            Some(im) => im,
            None => {
                eprintln!("cannot decode {}", img_path.display());
                continue;
            }
        };
        let t0 = std::time::Instant::now();
        let res = if det_only {
            engine.run_det_only(&img).expect("run")
        } else {
            engine.run(&img).expect("run")
        };
        let wall = t0.elapsed().as_secs_f64() * 1000.0;
        if json {
            println!("{{");
            println!("  \"file\": {:?},", img_path.display().to_string());
            println!("  \"lines\": [");
            for (i, l) in res.lines.iter().enumerate() {
                let comma = if i + 1 < res.lines.len() { "," } else { "" };
                println!(
                    "    {{\"text\": {:?}, \"conf\": {:.4}, \"rotation\": {}}}{}",
                    l.text, l.confidence, l.rotation, comma
                );
            }
            println!("  ],");
            println!(
                "  \"timings\": {{\"load_ms\": {load_ms:.1}, \"total_ms\": {:.1}, \"det_infer_ms\": {:.1}, \"rec_infer_ms\": {:.1}, \"cls_ms\": {:.1}}},",
                res.timings.total_ms,
                res.timings.det_infer_ms,
                res.timings.rec_infer_ms,
                res.timings.cls_ms
            );
            println!(
                "  \"counts\": {{\"boxes\": {}, \"lines\": {}, \"merged\": {}, \"unread\": {}, \"flipped\": {}, \"retried\": {}}},",
                res.num_boxes,
                res.lines.len(),
                res.num_merged,
                res.num_unread,
                res.num_flipped,
                res.num_det_retried
            );
            println!(
                "  \"det_input\": [{}, {}], \"work\": [{}, {}]",
                res.det_input_h, res.det_input_w, res.work_h, res.work_w
            );
            println!("}}");
        } else {
            for l in &res.lines {
                println!("{:.2}\t{}", l.confidence, l.text);
            }
            eprintln!(
                "[{:?}] lines={} boxes={} merged={} unread={} flipped={} retried={} | load {:.0}ms run {:.0}ms (det {:.0} rec {:.0} cls {:.0})",
                img_path.file_name().unwrap().to_string_lossy(),
                res.lines.len(),
                res.num_boxes,
                res.num_merged,
                res.num_unread,
                res.num_flipped,
                res.num_det_retried,
                load_ms,
                wall,
                res.timings.det_infer_ms,
                res.timings.rec_infer_ms,
                res.timings.cls_ms
            );
        }
    }
}
