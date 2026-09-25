//! `qppocr` 参考命令行（对应 基准实现的 设计文档）。
//!
//! 批量多图走**进程内 worker 线程共享一个引擎**：基准的多进程 worker 池
//! （`proc_pool.hpp`）是被迫的——它的 `ThreadPool` 并发 `parallel_for`
//! 会挂死；我们的池用 `fork_mu` 把并发 fork 串行化（`pool.rs` 模块注释），
//! 同进程并发 `run` 是安全的，还省下 W 份权重内存与全部 IPC。

use std::io::Write as _;
use std::path::{Path, PathBuf};

use qppocr::{Engine, Error, Preset, Tier};

struct Options {
    images: Vec<PathBuf>,
    models_dir: PathBuf,
    tier: Tier,
    preset: Preset,
    threads: usize,
    /// 批量并发 worker 数，0 = 自动（对齐  `--workers`）。
    workers: usize,
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
  --workers <n>     concurrent images for a multi-image batch (0 = auto)
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
        workers: 0,
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
            "--workers" => {
                i += 1;
                o.workers = args.get(i)?.parse().ok()?;
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

/// preset 枚举回 CLI 字符串（worker 参数回传用）。
fn preset_name(p: Preset) -> &'static str {
    match p {
        Preset::Speed => "speed",
        Preset::Balanced => "balanced",
        Preset::Accuracy => "accuracy",
    }
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

/// 批量并发的 worker 数：按图幅与核数自动定。
///
/// - 显式 `--workers N`：`min(N, 图数)`；
/// - 自动：头部探测平均面积（按 max_side_len=960 的帽折算——两档默认
///   相同）——大图（≥1.5 MP）一张就吃满线程，cap 2；小图 cap 8；
///   再受核数减半约束（ 实测：同 16 线程摊到更多 worker 优于集中）。
///
/// 与 基准的差异：**没有按内存收紧**。多进程每个 worker 是整份引擎
/// （多进程方案预算 512 MB/worker）；进程内共享引擎的增量只是图像缓冲 +
/// arena（大图几十 MB 级），8 个并发也在单份引擎的量级内。
fn auto_workers(o: &Options) -> usize {
    if o.images.len() < 2 {
        return 1;
    }
    if o.workers > 0 {
        return o.workers.min(o.images.len());
    }
    let cores = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(4)
        .min(16);
    let mut mp_sum = 0.0f64;
    let mut probed = 0usize;
    for p in &o.images {
        if let Ok((mut w, mut h)) = qppocr::probe_dimensions(p) {
            let m = w.max(h);
            if m > 960 {
                let s = 960.0 / m as f32;
                w = (w as f32 * s) as u32;
                h = (h as f32 * s) as u32;
            }
            mp_sum += w as f64 * h as f64 / 1e6;
            probed += 1;
        }
    }
    let avg_mp = if probed > 0 {
        mp_sum / probed as f64
    } else {
        0.0
    };
    let cap = if avg_mp >= 1.5 { 2 } else { 8 };
    o.images.len().min((cores / 2).max(1)).min(cap).max(1)
}

/// 单张图的处理结果（并行模式下先收集、按输入序输出）。
struct Outcome {
    /// json 模式下该图的对象片段。
    json: String,
    /// 非 json 模式的 stdout 行（空 = quiet 或无行）。
    stdout_text: String,
    /// stderr 的进度摘要行。
    summary: String,
}

fn process_one(engine: &Engine, o: &Options, path: &Path) -> Result<Outcome, Error> {
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

    let t = &res.timings;
    let (det_pre, det_inf, det_post, crop, cls, rec_pre, rec_inf, rec_post) = (
        t.det_pre_ms,
        t.det_infer_ms,
        t.det_post_ms,
        t.crop_ms,
        t.cls_ms,
        t.rec_pre_ms,
        t.rec_infer_ms,
        t.rec_post_ms,
    );
    let mut json = format!(
        "{{\"image\": \"{}\", \"width\": {}, \"height\": {}, \"lines\": [",
        json_escape(&path.display().to_string()),
        img.w,
        img.h
    );
    for (i, l) in res.lines.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
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
    json.push_str(&format!(
        "], \"timing\": {{\"decode_ms\": {decode_ms:.2}, \"total_ms\": {best_ms:.2}, \"det_pre_ms\": {det_pre:.2}, \"det_infer_ms\": {det_inf:.2}, \"det_post_ms\": {det_post:.2}, \"crop_ms\": {crop:.2}, \"cls_ms\": {cls:.2}, \"rec_pre_ms\": {rec_pre:.2}, \"rec_infer_ms\": {rec_inf:.2}, \"rec_post_ms\": {rec_post:.2}}}}}"
    ));

    let mut stdout_text = String::new();
    if !o.json && !o.quiet {
        for l in &res.lines {
            if o.show_boxes {
                stdout_text.push_str(&format!(
                    "[{:.2}] {}  ({:.0},{:.0})-({:.0},{:.0})\n",
                    l.confidence, l.text, l.pts[0][0], l.pts[0][1], l.pts[2][0], l.pts[2][1]
                ));
            } else {
                stdout_text.push_str(&format!("[{:.2}] {}\n", l.confidence, l.text));
            }
        }
    }
    let summary = format!(
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
    Ok(Outcome {
        json,
        stdout_text,
        summary,
    })
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

    let w = auto_workers(o);
    if w > 1 && !o.json {
        eprintln!("[batch] {} 张图，{} 个 worker 进程", o.images.len(), w);
    }

    let mut json_out = String::new();
    if o.json {
        json_out.push('[');
    }

    if w <= 1 {
        // ---- 顺序：边跑边输出（单图 / --workers 1 的原路径）----
        for (idx, path) in o.images.iter().enumerate() {
            let out = process_one(&engine, o, path)?;
            if o.json {
                if idx > 0 {
                    json_out.push(',');
                }
                json_out.push_str(&out.json);
            } else {
                print!("{}", out.stdout_text);
                eprintln!("{}", out.summary);
            }
        }
    } else {
        // ---- 并行：W 个子进程各持一份引擎（ proc_pool 的静态切分版）。
        // 为什么不是进程内多线程共享引擎：池的 fork_mu 把并发 fork 串行化，
        // 多图的并行算子排队——串行段可以重叠（实测 100 图 14→7.4 s），
        // 但独立小池才能真并发（实测子进程 5.8 s）。gemm 分轴规则还依赖
        // 线程数（路径选择本身进位），进程内收窄宽度会破坏位级一致性。
        // 连续切块（不是交错）：子进程按序拼接 = 输入序，无需解析合并。
        let threads_each = if o.threads > 0 {
            o.threads
        } else {
            (qppocr::thread_count() / w).max(1)
        };
        let exe = std::env::current_exe()?;
        let n = o.images.len();
        let chunk = n.div_ceil(w);
        let mut children = Vec::new();
        for (ci, part) in o.images.chunks(chunk).enumerate() {
            if part.is_empty() {
                break;
            }
            let mut cmd = std::process::Command::new(&exe);
            cmd.args(part)
                .arg("--workers")
                .arg("1")
                .arg("--threads")
                .arg(threads_each.to_string());
            for (flag, on) in [
                ("--det-only", o.det_only),
                ("--boxes", o.show_boxes),
                ("--json", o.json),
                ("--no-cls", o.no_cls),
                ("--quiet", o.quiet),
            ] {
                if on {
                    cmd.arg(flag);
                }
            }
            if o.bench > 1 {
                cmd.arg("--bench").arg(o.bench.to_string());
            }
            cmd.arg("--models").arg(&o.models_dir);
            // tier/preset 从字符串回传（枚举无反向 API，这两处字符串是唯一来源）
            cmd.arg("--tier")
                .arg(o.tier.dir_name())
                .arg("--preset")
                .arg(preset_name(o.preset));
            let child = cmd
                .stdout(std::process::Stdio::piped())
                .spawn()
                .map_err(|e| Error::Io(format!("启动 worker {ci}: {e}")))?;
            children.push(child);
        }
        // 逐子进程收 stdout（生成序 = 输入序）；stderr 继承（进度实时）
        for child in children {
            let out = child
                .wait_with_output()
                .map_err(|e| Error::Io(format!("等待 worker: {e}")))?;
            if !out.status.success() {
                return Err(Error::Io(format!("worker 退出码 {:?}", out.status.code())));
            }
            let text = String::from_utf8_lossy(&out.stdout);
            if o.json {
                let inner = text.trim().trim_start_matches('[').trim_end_matches(']');
                if !json_out.ends_with('[') {
                    json_out.push(',');
                }
                json_out.push_str(inner.trim());
            } else {
                print!("{}", text);
            }
        }
    }
    if o.json {
        json_out.push(']');
        let mut stdout = std::io::stdout().lock();
        let _ = writeln!(stdout, "{json_out}");
    }
    Ok(())
}
