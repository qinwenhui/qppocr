#![allow(clippy::needless_range_loop, clippy::identity_op)] // 形状字面量按维度展开写

//! 真实模型解析验证（阶段 2）。
//!
//! 对照 / 附录 D 的已知事实：
//! - 上游 det：opset **14**、242 节点、169 权重（转换版才是 opset 11/464/213）
//! - rec：opset 11、219 节点；tiny rec 自带 `character` 元数据（6904 字典）
//! - small/medium rec 的字典 18708 项；medium rec **无内嵌字典**
//!
//! 模型不在仓库里：缺失时跳过（CI 无模型），本机有则全查。
//! 位置：workspace 根的 models/（.gitignore 排除）。

use qppocr_core::onnx::load_onnx;
use qppocr_core::tensor::{DType, Tensor};

fn models_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn check_model(file: &str) -> Option<qppocr_core::onnx::Graph> {
    let p = models_dir().join(file);
    if !p.exists() {
        eprintln!("skip (no model): {}", p.display());
        return None;
    }
    match load_onnx(&p) {
        Ok(g) => Some(g),
        Err(e) => panic!("{file}: parse failed: {e}"),
    }
}

#[test]
fn parse_upstream_models() {
    // ---- 上游原件 det：opset 14、优化前 242 节点、169 权重 ----
    if let Some(g) = check_model("tiny/det.onnx") {
        assert_eq!(g.opset_version, 14, "上游 det 是 opset 14（§4.4）");
        // node_count 是优化后的；优化前 = node_count + 各 pass 删掉的节点
        let pre = g.node_count + g.dropped_identity;
        // §4.4 的 242 是优化前节点数；drop_identity 在 Constant/Identity 消除后计数
        println!(
            "det_tiny: opset={} nodes(pre-const-fold≈{}) post={} fused_gelu={} folded_bias={} dropped_tp={} fused_act={} params={}",
            g.opset_version,
            pre,
            g.node_count,
            g.fused_gelu,
            g.folded_bias,
            g.dropped_identity,
            g.fused_conv_act,
            g.param_count
        );
    }

    // ---- 上游 rec tiny：opset 11、219 节点、**不带**字典元数据 ----
    //（§4.4：上游原件不带元数据，转换版才内嵌 character——字典是独立文件）
    if let Some(g) = check_model("tiny/rec.onnx") {
        assert_eq!(g.opset_version, 11, "上游 rec 是 opset 11");
        assert!(
            !g.metadata.contains_key("character"),
            "上游 rec 不应带 character 元数据（那是转换版的特征）"
        );
        println!(
            "rec_tiny: opset={} nodes={} metadata_keys={:?}",
            g.opset_version,
            g.node_count,
            g.metadata.keys().collect::<Vec<_>>()
        );
    }

    // ---- medium rec：无内嵌字典（§4.4 硬要求的事实） ----
    if let Some(g) = check_model("medium/rec.onnx") {
        assert!(
            !g.metadata.contains_key("character") && !g.metadata.contains_key("ppocr_character"),
            "medium rec 不带字典（§4.4）"
        );
        println!(
            "rec_medium: opset={} nodes={}",
            g.opset_version, g.node_count
        );
    }

    // ---- cls ----
    if let Some(g) = check_model("cls.onnx") {
        println!(
            "cls: opset={} nodes={} inputs={:?} outputs={:?}",
            g.opset_version, g.node_count, g.inputs, g.outputs
        );
    }
}

/// 权重字节账：图里到底装了多少 f32 权重、有多少**没有任何节点引用**。
///
/// 动机（Metal 侧实测）：medium rec 的计划静态区 = **148.1 MB**，而
/// `medium/rec.onnx` 整个文件才 **76.6 MB** ⇒ 1.93×。det 同口径是
/// 61.8 MB 对 62.0 MB（**1.00×**，零膨胀）。两条读数各自自洽，所以
/// 差额只能落在「rec 图的权重本身就比文件大」或「Metal 侧重排把它放大
/// 了」这两者之一 —— 本测试分开量前一半（纯 CPU，零 GPU 风险）。
///
/// 判读：
/// * `init f32 字节 ≈ file_bytes` ⇒ 膨胀在**重排**侧（Metal/Vulkan 的
///   `repack_*`），按 blob 逐个查；
/// * `init f32 字节 ≈ 2 × file_bytes` ⇒ 文件本身**就是**紧凑存的（比如
///   fp16/量化），解析时展开成 f32，那 2× 是解析器的选择而不是 bug。
#[test]
fn weight_bytes_report() {
    use std::collections::HashSet;
    for f in [
        "tiny/det.onnx",
        "tiny/rec.onnx",
        "medium/det.onnx",
        "medium/rec.onnx",
        "cls.onnx",
    ] {
        let Some(g) = check_model(f) else { continue };
        let mut items: Vec<(i64, &String)> =
            g.initializers.iter().map(|(k, t)| (t.numel(), k)).collect();
        items.sort_by_key(|b| std::cmp::Reverse(b.0));
        let total: i64 = items.iter().map(|(n, _)| *n).sum();
        let used: HashSet<&str> = g
            .nodes
            .iter()
            .flat_map(|nd| nd.inputs.iter().map(|s| s.as_str()))
            .collect();
        let (nu, nu_bytes) = items
            .iter()
            .filter(|(_, k)| !used.contains(k.as_str()))
            .fold((0usize, 0i64), |(c, b), (n, _)| (c + 1, b + n * 4));
        println!(
            "== {f}: file={:.1} MB · init {} 块 / {} elem / {:.1} MB(f32) = {:.2}×file · 未被引用 {} 块 / {:.1} MB",
            g.file_bytes as f64 / 1e6,
            items.len(),
            total,
            total as f64 * 4.0 / 1e6,
            total as f64 * 4.0 / g.file_bytes as f64,
            nu,
            nu_bytes as f64 / 1e6
        );
        for (n, k) in items.iter().take(10) {
            println!("     {:>10.2} MB  {k}  ({n} elem)", *n as f64 * 4.0 / 1e6);
        }
    }
}

/// rec tiny 前向一遍：固定输入、确定性输出形状 + 基本健全性。
/// 逐位对拍在 dump 对比测试里做（需要  侧黄金值）。
#[test]
fn run_rec_tiny_forward() {
    let p = models_dir().join("tiny/rec.onnx");
    if !p.exists() {
        eprintln!("skip (no model): {}", p.display());
        return;
    }
    let sess = qppocr_core::executor::Session::open(&p).expect("open rec_tiny");
    // 输入名从图里取；形状 [1,3,48,320]（rec 标准画布）
    let in_name = sess
        .graph
        .inputs
        .iter()
        .find(|n| !n.is_empty())
        .unwrap()
        .clone();
    let mut rng: u64 = 0xDEADBEEF;
    let mut data = vec![0f32; 1 * 3 * 48 * 320];
    for v in data.iter_mut() {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        *v = ((rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as i64 - (1 << 23)) as f32
            / (1 << 23) as f32;
    }
    let t0 = std::time::Instant::now();
    let out = sess
        .run(vec![(
            in_name,
            Tensor {
                name: "x".into(),
                shape: vec![1, 3, 48, 320],
                dtype: DType::F32,
                f32: qppocr_kernels::buf::F32Buf::from_vec(&data),
                i64: Vec::new(),
            },
        )])
        .expect("run rec_tiny");
    let dt = t0.elapsed();
    assert!(!out.is_empty(), "rec 应有输出");
    let o = &out[0];
    println!(
        "rec_tiny forward: out shape={:?} dtype={:?} in {:?}",
        o.shape, o.dtype, dt
    );
    // CTC 输出：[1, T, C]，C = 字典+1。tiny 字典 6904 → C = 6906
    assert_eq!(o.rank(), 3, "rec 输出 [1,T,C]");
    assert_eq!(o.shape[2], 6906, "tiny 字典 6904 + blank");
    assert!(
        o.f32.as_slice().iter().all(|v| v.is_finite()),
        "输出应全有限"
    );
}

/// det tiny 前向一遍：[1,3,H,W] → [1,1,H',W'] 概率图。
#[test]
fn run_det_tiny_forward() {
    let p = models_dir().join("tiny/det.onnx");
    if !p.exists() {
        eprintln!("skip (no model): {}", p.display());
        return;
    }
    let sess = qppocr_core::executor::Session::open(&p).expect("open det_tiny");
    let in_name = sess
        .graph
        .inputs
        .iter()
        .find(|n| !n.is_empty())
        .unwrap()
        .clone();
    let mut rng: u64 = 0xC0FFEE;
    let mut data = vec![0f32; 1 * 3 * 736 * 960];
    for v in data.iter_mut() {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        *v = ((rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as i64 - (1 << 23)) as f32
            / (1 << 23) as f32;
    }
    let t0 = std::time::Instant::now();
    let out = sess
        .run(vec![(
            in_name,
            Tensor {
                name: "x".into(),
                shape: vec![1, 3, 736, 960],
                dtype: DType::F32,
                f32: qppocr_kernels::buf::F32Buf::from_vec(&data),
                i64: Vec::new(),
            },
        )])
        .expect("run det_tiny");
    println!("det_tiny forward: {:?} in {:?}", out[0].shape, t0.elapsed());
    assert_eq!(out[0].shape.len(), 4);
    // det 输出与输入同分辨率（上采样回来）
    assert_eq!(out[0].shape[2], 736);
    assert_eq!(out[0].shape[3], 960);
    // 概率图：sigmoid 之后值域 [0,1]
    let mx = out[0].f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mn = out[0].f32.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        (0.0..=1.0).contains(&mx) && (0.0..=1.0).contains(&mn),
        "概率图值域 [{mn},{mx}]"
    );
}

/// MatMul 权重的名字/形状，以及「谁引用了它」——查 148 MB 静态区里那
/// 71.6 MB 的死重（见 `weight_bytes_report` 的动机）。
///
/// 结论（2026-10-10）：9 块 MatMul 权重**全部只被 MatMul 引用**，而
/// `plan.rs` 的 MatMul 分支自己会走 `n_weight_off` 重排一份紧凑的
/// `[n4][k4]` blob ⇒ `init_to_nhwc` 那 9 份（被 `[1,1,C,N]` 顶成 cpad4(1)=4，
/// 放大 4×）是**死重**：71.64 MB 的白占，与文件里那 17.91 MB 一一对应。
/// 死重不参与每次 run 的读，所以只吃内存；**换形状重拷那一半**已由
/// `Layout` 的两段分离修掉（见 plan.rs 的 `Layout`）。
///
/// **已落地**：`plan.rs` 的 `matmul_w_names` 现在预环跳过这些块（与
/// `conv_w_names` 同款），medium rec 的静态区 148.1 → **76.5 MB**。
/// 实测**性能中性**（`ab.sh` 中位 Δtotal +0.15 ms，det 对照 −0.40 混号；
/// 逐形状墙钟逐条不变，例如 [1,3,48,408] 改前后都是 10.79 ms）——正
/// 因为死重本来就不被读，省下来的是常驻内存不是时间。当初曾反推
/// 「静态区每 run 被读一遍 ⇒ 2.9 ms/run 的固定成本是权重流量」，
/// 这次实测把这个假设证伪了：固定成本与静态区大小无关。
///
/// 下面把那个前提**断言下来**：`plan.rs` 的跳过门槛是「只被 MatMul
/// 引用」，模型哪天多一个算子共用同一块权重，`off_of` 就会落空——
/// 那会是一次运行时崩溃，不如在这里先炸。
#[test]
fn matmul_weight_owners() {
    let Some(g) = check_model("medium/rec.onnx") else {
        return;
    };
    use std::collections::HashMap;
    let mut users: HashMap<&str, Vec<&str>> = HashMap::new();
    for n in &g.nodes {
        for i in &n.inputs {
            if !i.is_empty() {
                users
                    .entry(i.as_str())
                    .or_default()
                    .push(n.op_type.as_str());
            }
        }
    }
    let mut tot = 0i64;
    let mut shared: Vec<&str> = Vec::new();
    for n in &g.nodes {
        if n.op_type != "MatMul" || n.inputs.len() < 2 {
            continue;
        }
        let Some(t) = g.initializers.get(&n.inputs[1]) else {
            println!("  {} B={} 不是 initializer（动态）", n.name, n.inputs[1]);
            continue;
        };
        let numel = t.numel();
        tot += numel;
        println!(
            "  {} B={} {:?} {} elem {:.2} MB · 引用者 {:?}",
            n.name,
            n.inputs[1],
            t.shape,
            numel,
            numel as f64 * 4.0 / 1e6,
            users.get(n.inputs[1].as_str())
        );
        if let Some(u) = users.get(n.inputs[1].as_str()) {
            if u.iter().any(|o| *o != "MatMul") && !shared.contains(&n.inputs[1].as_str()) {
                shared.push(n.inputs[1].as_str());
            }
        }
    }
    assert!(
        shared.is_empty(),
        "MatMul 权重被别的算子共用了：{shared:?} —— `plan.rs::matmul_w_names` \
         会跳过它们的 NHWC 区，共用方的 `off_of` 将落空"
    );
    println!("MatMul 权重合计 {:.2} MB", tot as f64 * 4.0 / 1e6);
    // 反向：哪些 initializer 只被 MatMul 引用（⇒ 若 MatMul 走自己的重排，
    // 那份 init_to_nhwc 拷贝就是死的）
    let mut only_mm = 0i64;
    let mut only_mm_n = 0;
    for (name, t) in &g.initializers {
        if let Some(u) = users.get(name.as_str()) {
            if !u.is_empty() && u.iter().all(|o| *o == "MatMul") {
                only_mm += t.numel();
                only_mm_n += 1;
            }
        }
    }
    println!(
        "只被 MatMul 引用的 initializer：{only_mm_n} 块 / {:.2} MB（f32）",
        only_mm as f64 * 4.0 / 1e6
    );
}
