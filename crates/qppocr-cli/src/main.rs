//! `qppocr` 参考命令行（对应 C++ 参考实现的 `main.cpp`，取其单进程子集：
//! 多进程 worker 池是上层的事，DESIGN.md §7——引擎核心只做单实例）。

use std::io::Write as _;
use std::path::PathBuf;

use qppocr::{Engine, Error, Preset, Tier};

struct Options {
    images: Vec<PathBuf>,
    models_dir: PathBuf,
    tier: Tier,
    preset: Preset,
    threads: usize,
    det_only: bool,
    show_boxes: bool,
    json: bool,
    quiet: bool,
    bench: usize,
    no_cls: bool,
}

fn print_usage() {
    println!(
        "qppocr — 纯 Rust 的 PP-OCRv6 推理引擎（上游官方模型）

usage: qppocr <image> [<image> ...] [options]

options:
  --models <dir>    model directory   (default: models/)
  --tier <t>        tiny | small | medium   (default: small)
  --preset <p>      speed | balanced | accuracy   (default: balanced)
  --threads <n>     worker threads (default: auto)
  --det-only        only detect boxes, skip recognition
  --boxes           also print box coordinates
  --json            emit JSON
  --bench <n>       run n times per image and report the best
  --no-cls          turn off the 0/180 direction classifier
  --quiet           suppress the per-line listing
  -h, --help        this help"
    );
}

fn parse_args() -> Option<Options> {
    let mut o = Options {
        images: Vec::new(),
        models_dir: "models".into(),
        tier: Tier::Small,
        preset: Preset::Balanced,
        threads: 0,
        det_only: false,
        show_boxes: false,
        json: false,
        quiet: false,
        bench: 1,
        no_cls: false,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "-h" | "--help" => {
                print_usage();
                return None;
            }
            "--models" => {
                i += 1;
                o.models_dir = args.get(i)?.clone().into();
            }
            "--tier" => {
                i += 1;
                o.tier = match args.get(i)?.as_str() {
                    "tiny" => Tier::Tiny,
                    "small" => Tier::Small,
                    "medium" => Tier::Medium,
                    _ => {
                        eprintln!("--tier: tiny | small | medium");
                        return None;
                    }
                };
            }
            "--preset" => {
                i += 1;
                o.preset = match args.get(i)?.as_str() {
                    "speed" => Preset::Speed,
                    "balanced" => Preset::Balanced,
                    "accuracy" => Preset::Accuracy,
                    _ => {
                        eprintln!("--preset: speed | balanced | accuracy");
                        return None;
                    }
                };
            }
            "--threads" => {
                i += 1;
                o.threads = args.get(i)?.parse().ok()?;
            }
            "--det-only" => o.det_only = true,
            "--boxes" => o.show_boxes = true,
            "--json" => o.json = true,
            "--bench" => {
                i += 1;
                o.bench = args.get(i)?.parse().ok()?;
            }
            "--no-cls" => o.no_cls = true,
            "--quiet" => o.quiet = true,
            _ => o.images.push(a.clone().into()),
        }
        i += 1;
    }
    if o.images.is_empty() {
        print_usage();
        return None;
    }
    Some(o)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn main() {
    let Some(o) = parse_args() else {
        std::process::exit(0);
    };
    if let Err(e) = run(&o) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(o: &Options) -> Result<(), Error> {
    let t0 = std::time::Instant::now();
    let mut builder = Engine::builder()
        .tier(o.tier)
        .preset(o.preset)
        .detect_orientation(!o.no_cls);
    if o.threads > 0 {
        builder = builder.threads(o.threads);
    }
    let engine = builder.build(&o.models_dir)?;
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if !o.json {
        eprintln!("model: {} (load {:.0} ms)", o.tier.dir_name(), load_ms);
    }

    let mut json_out = String::new();
    if o.json {
        json_out.push('[');
    }
    for (idx, path) in o.images.iter().enumerate() {
        let img = qppocr::decode_file(path)?;
        let decode_ms = 0.0; // decode_file 内含在 load 阶段外，此处近似

        let mut best_ms = f64::MAX;
        let mut result = None;
        for _ in 0..o.bench.max(1) {
            let t0 = std::time::Instant::now();
            let r = if o.det_only {
                engine.run_det_only(&img)?
            } else {
                engine.run(&img)?
            };
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if ms < best_ms {
                best_ms = ms;
            }
            result = Some(r);
        }
        let res = result.unwrap();

        if o.json {
            if idx > 0 {
                json_out.push(',');
            }
            json_out.push_str(&format!(
                "{{\"image\": \"{}\", \"width\": {}, \"height\": {}, \"lines\": [",
                json_escape(&path.display().to_string()),
                img.w,
                img.h
            ));
            for (i, l) in res.lines.iter().enumerate() {
                if i > 0 {
                    json_out.push(',');
                }
                json_out.push_str(&format!(
                    "{{\"text\": \"{}\", \"confidence\": {:.4}, \"rotation\": {}, \"box\": [{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}]}}",
                    json_escape(&l.text),
                    l.confidence,
                    l.rotation,
                    l.pts[0][0], l.pts[0][1],
                    l.pts[1][0], l.pts[1][1],
                    l.pts[2][0], l.pts[2][1],
                    l.pts[3][0], l.pts[3][1],
                ));
            }
            json_out.push_str(&format!(
                "], \"timing\": {{\"decode_ms\": {decode_ms:.2}, \"total_ms\": {best_ms:.2}}}}}"
            ));
        } else {
            if !o.quiet {
                for l in &res.lines {
                    if o.show_boxes {
                        println!(
                            "[{:.2}] {}  ({:.0},{:.0})-({:.0},{:.0})",
                            l.confidence,
                            l.text,
                            l.pts[0][0],
                            l.pts[0][1],
                            l.pts[2][0],
                            l.pts[2][1]
                        );
                    } else {
                        println!("[{:.2}] {}", l.confidence, l.text);
                    }
                }
            }
            eprintln!(
                "  {}: {} lines, {:.1} ms{}",
                path.display(),
                res.lines.len(),
                best_ms,
                if o.bench > 1 {
                    format!(" (best of {})", o.bench)
                } else {
                    String::new()
                }
            );
        }
    }
    if o.json {
        json_out.push(']');
        let mut stdout = std::io::stdout().lock();
        let _ = writeln!(stdout, "{json_out}");
    }
    Ok(())
}
