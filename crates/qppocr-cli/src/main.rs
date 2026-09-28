//! `qppocr` 参考命令行。
//!
//! 批量多图走**进程内 worker 线程共享一个引擎**：池用 `fork_mu` 把并发
//! fork 串行化（见 `qppocr-kernels` 的 `pool.rs` 模块注释），同进程并发
//! `run` 是安全的，比多进程扇出省下 W 份权重内存与全部 IPC。

use std::io::Write as _;
use std::path::{Path, PathBuf};

use qppocr::{DeviceChoice, Engine, Error, GpuApi, Preset, Tier};

#[derive(Clone)]
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
    no_retry: bool,
    rec_height: u32,
    rec_shards: usize,
    /// 计算设备（`--device`）。
    device: DeviceChoice,
    /// `bench` 子命令模式。
    bench_mode: bool,
    /// bench 的测量轮数（另有 1 轮 warmup 丢弃）。
    rounds: usize,
    /// bench 的扫描轴：`key=v1,v2,...`，逐值与基准配置交错对比。
    sweep: Option<(String, Vec<String>)>,
}

/// `--device` 值解析：`cpu | gpu | vulkan[:N] | cuda[:N]`。
fn parse_device(s: &str) -> Option<DeviceChoice> {
    let (api, idx) = match s.split_once(':') {
        Some((a, i)) => (a, Some(i.parse().ok()?)),
        None => (s, None),
    };
    match api {
        "cpu" if idx.is_none() => Some(DeviceChoice::Cpu),
        "gpu" => Some(DeviceChoice::Gpu {
            api: GpuApi::Auto,
            index: idx,
        }),
        "vulkan" => Some(DeviceChoice::Gpu {
            api: GpuApi::Vulkan,
            index: idx,
        }),
        "cuda" => Some(DeviceChoice::Gpu {
            api: GpuApi::Cuda,
            index: idx,
        }),
        _ => None,
    }
}

/// DeviceChoice 回 CLI 字符串（worker 参数回传 / bench 报告用）。
fn device_name(d: &DeviceChoice) -> String {
    match d {
        DeviceChoice::Cpu => "cpu".into(),
        // DeviceChoice 是 non_exhaustive：未来变体（如指定设备名）落到
        // 这里，字符串形式的往返保真交给那一版再扩。
        DeviceChoice::Gpu { api, index } => {
            let a = match api {
                GpuApi::Auto => "gpu",
                GpuApi::Vulkan => "vulkan",
                GpuApi::Cuda => "cuda",
            };
            match index {
                Some(i) => format!("{a}:{i}"),
                None => a.into(),
            }
        }
        _ => "cpu".into(),
    }
}

fn print_usage() {
    println!(
        "qppocr — 纯 Rust 的 PP-OCRv6 推理引擎（上游官方模型）

usage: qppocr <image> [<image> ...] [options]
       qppocr bench <image> [<image> ...] [options] [--rounds n] [--sweep k=v1,v2]

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
  --rec-shards <n>  rec 两级并行的外层分片数（0 = 关，不传 = 按档位自动）
  --device <d>      cpu | gpu | vulkan[:N] | cuda[:N]   (default: cpu)
  --quiet           suppress the per-line listing
  -h, --help        this help

bench 子命令（测量纪律内建）：
  同进程跑完整语料（不逐图起进程），1 轮 warmup 丢弃；多配置时逐轮
  交错、各自取中位，报告探测到的内核后端与生效配置。
  --rounds <n>      测量轮数（默认 3）
  --sweep <k=vs>    扫一个轴：preset=balanced,speed | rec-height=40,48,56 |
                    threads=4,8,16 | rec-shards=0,4,12 | device=cpu,gpu。
                    第一个值之外还能用 'base' 引用命令行给的基准配置。"
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
        no_retry: false,
        rec_height: 0,
        // `usize::MAX` = 自动（按档位），`0` = 关，其余 = 显式分片数。
        rec_shards: usize::MAX,
        device: DeviceChoice::Cpu,
        bench_mode: false,
        rounds: 3,
        sweep: None,
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
            "bench" if i == 0 && !o.bench_mode => o.bench_mode = true,
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
            "--no-retry" => o.no_retry = true,
            "--rec-height" => {
                i += 1;
                o.rec_height = args.get(i)?.parse().ok()?;
            }
            "--rec-shards" => {
                i += 1;
                o.rec_shards = args.get(i)?.parse().ok()?;
            }
            "--device" => {
                i += 1;
                let s = args.get(i)?;
                o.device = parse_device(s).or_else(|| {
                    eprintln!("--device: cpu | gpu | vulkan[:N] | cuda[:N]");
                    None
                })?;
            }
            "--rounds" => {
                i += 1;
                o.rounds = args.get(i)?.parse().ok()?;
            }
            "--sweep" => {
                i += 1;
                let s = args.get(i)?;
                let (k, vs) = s.split_once('=')?;
                let k = k.trim().to_ascii_lowercase();
                if !matches!(
                    k.as_str(),
                    "preset" | "rec-height" | "threads" | "rec-shards" | "device"
                ) {
                    eprintln!("--sweep: preset | rec-height | threads | rec-shards | device");
                    return None;
                }
                let vs: Vec<String> = vs.split(',').map(|v| v.trim().to_string()).collect();
                if vs.is_empty() || vs.iter().any(|v| v.is_empty()) {
                    eprintln!("--sweep: 值列表不能为空");
                    return None;
                }
                o.sweep = Some((k, vs));
            }
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
    let r = if o.bench_mode { run_bench(&o) } else { run(&o) };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// 批量并发的 worker 数：按图幅与核数自动定。
///
/// - 显式 `--workers N`：`min(N, 图数)`；
/// - 自动：头部探测平均面积（按 max_side_len=960 的帽折算——两档默认
///   相同）——大图（≥1.5 MP）一张就吃满线程，cap 2；小图 cap 8；
///   再受核数减半约束（实测：同 16 线程摊到更多 worker 优于集中）。
///
/// 线程数**不按内存收紧**：进程内共享同一个引擎，每个 worker 的增量只是
/// 图像缓冲 + arena（大图几十 MB 级），8 个并发也在单份引擎的量级内。
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
    // ★ 分阶段数字必须来自**最快的那一次**。以前只留最后一次的 timings，
    //   于是总时间是最快的、分项是碰运气的那次，九项加起来对不上总数——
    //   读的人会以为有没统计到的开销。
    let mut best_t = None;
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
            best_t = Some(r.timings);
        }
        result = Some(r);
    }
    let res = result.unwrap();
    let best_t = best_t.expect("bench 至少跑一次");

    let t = &best_t;
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
        "], \"timing\": {{\"decode_ms\": {decode_ms:.2}, \"total_ms\": {best_ms:.2}, \"det_pre_ms\": {det_pre:.2}, \"det_infer_ms\": {det_inf:.2}, \"det_post_ms\": {det_post:.2}, \"crop_ms\": {crop:.2}, \"cls_ms\": {cls:.2}, \"rec_pre_ms\": {rec_pre:.2}, \"rec_infer_ms\": {rec_inf:.2}, \"rec_post_ms\": {rec_post:.2}}}"
    ));
    // 统计量单列一段：与大括号转义混在一起写容易数错（{} 后的字面 }
    // 要写 `}}`），拆开就一目了然。
    json.push_str(&format!(
        ", \"num_boxes\": {}, \"num_merged\": {}, \"num_decluttered\": {}, \
         \"num_det_retried\": {}, \"num_flipped\": {}, \"num_unread\": {}",
        res.num_boxes,
        res.num_merged,
        res.num_decluttered,
        res.num_det_retried,
        res.num_flipped,
        res.num_unread,
    ));
    json.push('}');

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

/// 从 Options 构造引擎（run 与 bench 共用；sweep 的每个配置各建一份）。
fn build_engine(o: &Options) -> Result<Engine, Error> {
    let mut builder = Engine::builder()
        .tier(o.tier)
        .preset(o.preset)
        .device(o.device.clone())
        .detect_orientation(!o.no_cls);
    if o.threads > 0 {
        builder = builder.threads(o.threads);
    }
    // ⚠ 这里曾经写成 `if o.rec_shards > 0 { ... }`，于是**显式的 `--rec-shards 0`
    // 也被忽略**、默认值落到自动分片——「关分片」的对照实验全程在测同一个配置，
    // 得出来的「无差别」是假的。显式值必须原样透传，`0` 就是「关」。
    let n = o.rec_shards;
    let no_retry = o.no_retry;
    let rec_height = o.rec_height;
    builder = builder.advanced(move |a| {
        a.rec_shards = n;
        if no_retry {
            a.retry_conf = 0.0;
        }
        if rec_height > 0 {
            a.rec_height = rec_height as i32;
        }
    });
    builder.build(&o.models_dir)
}

fn run(o: &Options) -> Result<(), Error> {
    let t0 = std::time::Instant::now();
    let engine = build_engine(o)?;
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
            // ★ device 必须回传：这里漏掉的话子进程静默跑 CPU，bench 数字
            //   全部失真且无任何报错（--tier/--preset 曾是同一类遗漏高发区）。
            cmd.arg("--device").arg(device_name(&o.device));
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

// ================================================================ bench 子命令

/// 一轮语料测量的累计（引擎阶段时间不含图像解码）。
struct RoundStats {
    total: f64,
    det: f64,
    line: f64,
    wall: f64,
}

/// 跑一遍完整语料（同进程、顺序；每图引擎计时累加，墙钟含解码）。
fn corpus_round(engine: &Engine, o: &Options) -> Result<RoundStats, Error> {
    let t0 = std::time::Instant::now();
    let (mut total, mut det, mut line) = (0.0f64, 0.0f64, 0.0f64);
    for path in &o.images {
        let img = qppocr::decode_file(path)?;
        let r = if o.det_only {
            engine.run_det_only(&img)?
        } else {
            engine.run(&img)?
        };
        let t = &r.timings;
        total += t.total_ms;
        det += t.det_pre_ms + t.det_infer_ms + t.det_post_ms;
        line += t.cls_ms + t.rec_pre_ms + t.rec_infer_ms + t.rec_post_ms;
    }
    Ok(RoundStats {
        total,
        det,
        line,
        wall: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

/// 中位数（偶数个取中间两数的均值）。
fn med(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// 把 sweep 的一个值写进配置副本（`base` 引用命令行的基准值）。
fn apply_sweep(o: &mut Options, key: &str, val: &str) -> Result<String, String> {
    let resolve = |v: &str| -> Result<String, String> {
        if v == "base" {
            Ok(match key {
                "preset" => preset_name(o.preset).to_string(),
                "rec-height" => o.rec_height.to_string(),
                "threads" => o.threads.to_string(),
                "device" => device_name(&o.device),
                _ => o.rec_shards.to_string(),
            })
        } else {
            Ok(v.to_string())
        }
    };
    let v = resolve(val)?;
    match key {
        "preset" => {
            o.preset = match v.as_str() {
                "speed" => Preset::Speed,
                "balanced" => Preset::Balanced,
                "accuracy" => Preset::Accuracy,
                _ => return Err(format!("preset: {v}")),
            };
        }
        "rec-height" => {
            o.rec_height = v.parse().map_err(|_| format!("rec-height: {v} 不是整数"))?;
        }
        "threads" => {
            o.threads = v.parse().map_err(|_| format!("threads: {v} 不是整数"))?;
        }
        "rec-shards" => {
            o.rec_shards = v.parse().map_err(|_| format!("rec-shards: {v} 不是整数"))?;
        }
        "device" => {
            o.device = parse_device(&v).ok_or_else(|| format!("device: {v}"))?;
        }
        _ => unreachable!("parse_args 已限定了 sweep 键"),
    }
    Ok(match key {
        "rec-height" => format!("rec_height={v}"),
        _ => format!("{key}={v}"),
    })
}

/// `qppocr bench`：测量纪律内建的基准报告。
///
/// - **同进程跑完整语料**：逐图起进程会把冷启动摊到每张上；
/// - **1 轮 warmup 丢弃**：首轮含权重加载后的缓存冷态；
/// - **多配置逐轮交错**：配置间受同样的频率/热态影响，比值才是配置差异；
/// - **取中位**：单轮极值不进场。
fn run_bench(o: &Options) -> Result<(), Error> {
    if o.images.is_empty() {
        return Err(Error::Io("bench 需要至少一张图".into()));
    }
    let rounds = o.rounds.max(1);

    // 配置列表：基准 + sweep 值（第一个 sweep 值通常就是 base）。
    let mut configs: Vec<(String, Options)> = Vec::new();
    let eff = format!(
        "preset={} rec_h={} threads={} shards={} device={}",
        preset_name(o.preset),
        if o.rec_height > 0 {
            o.rec_height.to_string()
        } else {
            "preset".into()
        },
        if o.threads > 0 {
            o.threads.to_string()
        } else {
            "auto".into()
        },
        match o.rec_shards {
            usize::MAX => "auto".into(),
            0 => "off".into(),
            n => n.to_string(),
        },
        device_name(&o.device),
    );
    configs.push(("base".to_string(), o.clone()));
    if let Some((key, vals)) = &o.sweep {
        for v in vals {
            let mut c = o.clone();
            let label = apply_sweep(&mut c, key, v).map_err(Error::Io)?;
            // 与基准完全相同的配置不重复测。
            let dup = configs.iter().any(|(_, e)| {
                e.preset == c.preset
                    && e.rec_height == c.rec_height
                    && e.threads == c.threads
                    && e.rec_shards == c.rec_shards
                    && e.device == c.device
            });
            if !dup {
                configs.push((label, c));
            }
        }
    }

    // 每个配置一份引擎（构造期即生效各自的线程布局与分片规则）。
    let mut engines = Vec::with_capacity(configs.len());
    for (label, c) in &configs {
        let t0 = std::time::Instant::now();
        let e = build_engine(c)?;
        eprintln!(
            "[bench] engine ready: {label} ({:.0} ms)",
            t0.elapsed().as_secs_f64() * 1000.0
        );
        engines.push(e);
    }

    let backend = match qppocr::detect_backend() {
        qppocr::Backend::Avx2 => "avx2",
        qppocr::Backend::Neon => "neon",
        qppocr::Backend::Scalar => "scalar",
    };
    eprintln!(
        "[bench] {} 张图 | tier={} | {} / {} | 后端 {} | {eff} | 轮数 {}（+1 warmup）| 阶段时间不含图像解码",
        o.images.len(),
        o.tier.dir_name(),
        std::env::consts::ARCH,
        std::env::consts::OS,
        backend,
        rounds,
    );

    // warmup：每个配置各跑一遍，丢弃。
    for (ci, (label, _)) in configs.iter().enumerate() {
        corpus_round(&engines[ci], o)?;
        eprintln!("[bench] warmup done: {label}");
    }

    // 逐轮交错测量。
    let mut stats: Vec<Vec<RoundStats>> = (0..configs.len()).map(|_| Vec::new()).collect();
    for r in 0..rounds {
        for (ci, _) in configs.iter().enumerate() {
            stats[ci].push(corpus_round(&engines[ci], o)?);
        }
        eprintln!(
            "[bench] round {}/{}: {}",
            r + 1,
            rounds,
            configs
                .iter()
                .enumerate()
                .map(|(ci, (l, _))| format!("{l} {:.0}", stats[ci][r].total))
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }

    // 报告（stdout）：中位 + 相对第一个配置的比值。
    let n_img = o.images.len() as f64;
    let base_total = med(&stats[0].iter().map(|s| s.total).collect::<Vec<_>>());
    println!(
        "config                          total/img   det/img  line/img  wall/img   img/s  ratio"
    );
    for (ci, (label, _)) in configs.iter().enumerate() {
        let (t, d, l, w) = (
            med(&stats[ci].iter().map(|s| s.total).collect::<Vec<_>>()),
            med(&stats[ci].iter().map(|s| s.det).collect::<Vec<_>>()),
            med(&stats[ci].iter().map(|s| s.line).collect::<Vec<_>>()),
            med(&stats[ci].iter().map(|s| s.wall).collect::<Vec<_>>()),
        );
        println!(
            "{:<30} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>7.1} {:>6.2}x",
            label,
            t / n_img,
            d / n_img,
            l / n_img,
            w / n_img,
            1000.0 * n_img / w,
            base_total / t.max(1e-9),
        );
    }
    // 逐轮明细（判断噪声量级用）。
    for (ci, (label, _)) in configs.iter().enumerate() {
        let rounds_str = stats[ci]
            .iter()
            .map(|s| format!("{:.0}", s.total))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  rounds[{label}]: {rounds_str}");
    }
    Ok(())
}
