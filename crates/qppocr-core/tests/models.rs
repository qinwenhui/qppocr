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
