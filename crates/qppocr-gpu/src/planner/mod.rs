//! planner：装载期的图分析（后端无关，Vulkan/CUDA 共用）。
//!
//! core 没有装载期形状推理——executor 的形状是运行期逐节点推的。
//! GPU 会话必须装载期知道全部缓冲尺寸（权重上传、缓冲规划、内核
//! 分组），所以这里做两件事：
//!
//! 1. **静态形状推理**（[`shape`]）：对给定输入形状，逐节点推出全部
//!    中间张量的形状与 dtype。语义**逐字复刻** executor 的实现
//!    （输出公式、resolve_pads/auto_pad、右对齐广播、slice 钳位、
//!    0/-1 reshape、右对齐批次 matmul、keepdims 归约）——判据不是
//!    「像」，是 tests/plan.rs 用真实 det 模型对着 `QPPOCR_DUMP_DIR`
//!    落盘的逐节点形状做 oracle 对拍。
//! 2. **常量折叠**：`Shape/Cast/Slice/Concat/Transpose/Unsqueeze` 的
//!    i64 链在装载期求值（Resize 的 sizes、Reshape 的目标形状、
//!    Squeeze/ReduceMean 的 axes 都来自这些链）。
//!
//! Phase F 在此之上做缓冲规划与融合分组。

pub(crate) mod shape;

use std::collections::HashMap;

use qppocr_core::error::{Error, Result};
use qppocr_core::onnx::model::{Graph, Node};
use qppocr_core::tensor::{DType, Tensor};

/// 装载期常量（折叠出来的小张量，不进设备）。
#[derive(Clone, Debug)]
pub enum Const {
    /// i64 载荷（形状链等）。
    I64(Vec<i64>),
    /// f32 载荷（scales 等小常量）。
    F32(Vec<f32>),
}

impl Const {
    /// executor 的 `as_i64` 语义：哪个缓冲有值读哪个，f32 取 round。
    pub fn as_i64(&self) -> Vec<i64> {
        match self {
            Const::I64(v) => v.clone(),
            Const::F32(v) => v.iter().map(|x| x.round() as i64).collect(),
        }
    }

    /// executor 的 f32 读取语义（scales）。
    pub fn as_f32(&self) -> Vec<f32> {
        match self {
            Const::I64(v) => v.iter().map(|x| *x as f32).collect(),
            Const::F32(v) => v.clone(),
        }
    }
}

/// 一个张量的装载期信息。
#[derive(Clone, Debug)]
pub struct Value {
    /// 形状。
    pub shape: Vec<i64>,
    /// 元素类型。
    pub dtype: DType,
    /// 装载期已知的值（形状链/权重常量）；`None` = 运行期才定。
    pub konst: Option<Const>,
}

/// 全图推理结果：输出名 → 值信息。
pub struct ShapeTable {
    /// 全部张量（初始值 + 逐节点输出）。
    pub values: HashMap<String, Value>,
}

/// 对给定输入做静态形状推理 + 常量折叠。
///
/// 图按 `graph.nodes` 顺序单趟扫描（ONNX 图即拓扑序，与 executor 的
/// 调度等价——形状只依赖数据依赖，与执行顺序无关）。不支持的算子
/// 带节点名报错：GPU 路径在装载期点名，不做「跑一半才发现」。
pub fn infer_shapes(
    graph: &Graph,
    initializers: &HashMap<String, Tensor>,
    inputs: &[(String, Vec<i64>, DType)],
) -> Result<ShapeTable> {
    let mut values: HashMap<String, Value> = HashMap::new();
    // 初始值有两类：initializer（权重/常量，值已知）与图输入（形状已知）。
    for (name, t) in initializers {
        let konst = match t.dtype {
            DType::I64 => Some(Const::I64(t.i64.clone())),
            DType::F32 => Some(Const::F32(t.f32.to_vec())),
        };
        values.insert(
            name.clone(),
            Value {
                shape: t.shape.clone(),
                dtype: t.dtype,
                konst,
            },
        );
    }
    for (name, dims, dtype) in inputs {
        values.insert(
            name.clone(),
            Value {
                shape: dims.clone(),
                dtype: *dtype,
                konst: None,
            },
        );
    }

    for n in &graph.nodes {
        let out = shape::infer_node(n, &values)
            .map_err(|e| Error::Graph(format!("{} (node {}): {e}", n.op_type, n.name)))?;
        values.insert(n.outputs[0].clone(), out);
    }

    Ok(ShapeTable { values })
}

/// planner 侧的 `get_i`（镜像 executor 同名小函数）。
pub fn get_i(a: Option<&qppocr_core::onnx::model::Attribute>, dflt: i64) -> i64 {
    a.filter(|x| x.has_i).map(|x| x.i).unwrap_or(dflt)
}

/// planner 侧的 `get_f`。
pub fn get_f(a: Option<&qppocr_core::onnx::model::Attribute>, dflt: f32) -> f32 {
    a.filter(|x| x.has_f).map(|x| x.f).unwrap_or(dflt)
}

/// planner 侧的 `axes_from`：axes 来自属性（opset ≤ 12）或常量输入
/// （opset 13+，折叠后的 i64 链）。
pub(crate) fn axes_from(n: &Node, values: &HashMap<String, Value>) -> Result<Vec<i64>> {
    if let Some(a) = n.attr("axes") {
        if !a.ints.is_empty() {
            return Ok(a.ints.clone());
        }
    }
    if n.inputs.len() >= 2 && !n.inputs[1].is_empty() {
        if let Some(v) = values.get(&n.inputs[1]).and_then(|v| v.konst.as_ref()) {
            return Ok(v.as_i64());
        }
        return Err(Error::Graph(format!(
            "axes 输入 {} 非装载期常量",
            n.inputs[1]
        )));
    }
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::shape;
    use qppocr_core::executor::Session;
    use qppocr_core::tensor::Tensor;
    use qppocr_kernels::buf::F32Buf;
    use std::io::Read as _;
    use std::path::PathBuf;

    /// models/tiny/det.onnx（缺则跳过——CI 无模型）。
    fn tiny_det() -> Option<PathBuf> {
        let p = PathBuf::from("../../models/tiny/det.onnx");
        p.is_file().then_some(p)
    }

    /// dump 的 .f32 文件头：i32 rank + i64×rank 形状。
    fn read_dump_shape(path: &std::path::Path) -> std::io::Result<Vec<i64>> {
        let mut f = std::fs::File::open(path)?;
        let mut buf = [0u8; 4];
        f.read_exact(&mut buf)?;
        let rank = i32::from_le_bytes(buf) as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            let mut b = [0u8; 8];
            f.read_exact(&mut b)?;
            shape.push(i64::from_le_bytes(b));
        }
        Ok(shape)
    }

    /// `s{id}_{idx:06}.f32` 文件名 → (id, idx)。
    fn parse_dump_name(name: &str) -> Option<(u64, usize)> {
        let stem = name.strip_suffix(".f32")?;
        let (s, i) = stem.split_once('_')?;
        Some((s.strip_prefix('s')?.parse().ok()?, i.parse().ok()?))
    }

    /// oracle：真实 det 模型，planner 的静态推理 vs executor 的运行期
    /// 形状逐节点一致。首个分歧带两侧输入形状定位根因。
    /// 三个输入尺寸（方形/小方形/长宽比≠1）压满 padding/池化/插值的公式。
    #[test]
    fn det_shapes_match_executor() {
        for &(h, w) in &[(960i64, 960i64), (640, 640), (960, 512)] {
            det_oracle(h, w);
        }
    }

    fn det_oracle(h: i64, w: i64) {
        let Some(det) = tiny_det() else {
            eprintln!("[plan] models/tiny/det.onnx 不存在，跳过 oracle 对拍");
            return;
        };
        let dump_dir =
            std::env::temp_dir().join(format!("qppocr-plan-oracle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dump_dir);
        std::fs::create_dir_all(&dump_dir).unwrap();
        unsafe {
            // SAFETY: edition 2024 的 set_var 是 unsafe；本测试独占该环境
            // 变量（只有 Session::run 读它），本文件仅此一测。
            std::env::set_var("QPPOCR_DUMP_DIR", &dump_dir);
        }

        let bytes = std::fs::read(&det).unwrap();
        let session = Session::from_memory(&bytes, "tiny.det").expect("加载 det");
        let input_name = session
            .graph
            .inputs
            .iter()
            .find(|s| !s.is_empty())
            .cloned()
            .unwrap_or_default();
        session
            .run(vec![(
                input_name.clone(),
                Tensor {
                    name: input_name.clone(),
                    shape: vec![1, 3, h, w],
                    dtype: DType::F32,
                    f32: F32Buf::with_zeroed(3 * (h * w) as usize),
                    i64: Vec::new(),
                },
            )])
            .expect("CPU 前向（为 dump 落盘）");

        // manifest → executor 的逐节点输出形状。dump 的会话 id 全局递增
        //（本函数可能被多次调用），从文件名解析出本次的 s{id}_ 前缀。
        let manifest = std::fs::read_to_string(dump_dir.join("manifest.tsv")).unwrap();
        let mut max_sid: Option<u64> = None;
        for e in std::fs::read_dir(&dump_dir).unwrap().flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            if let Some((sid, _idx)) = parse_dump_name(&name) {
                max_sid = Some(max_sid.map_or(sid, |m: u64| m.max(sid)));
            }
        }
        let sid = max_sid.expect("dump 目录里没有任何 s{id}_*.f32");
        let mut exec: HashMap<String, Vec<i64>> = HashMap::new();
        for line in manifest.lines() {
            let mut it = line.split('\t');
            let (Some(idx), Some(_op), Some(out)) = (it.next(), it.next(), it.next()) else {
                continue;
            };
            // idx 必须按整数补零（&str 的 {:06} 是空格填充，文件名对不上）
            let Ok(idx) = idx.parse::<usize>() else {
                continue;
            };
            if let Ok(shape) = read_dump_shape(&dump_dir.join(format!("s{sid}_{idx:06}.f32"))) {
                exec.insert(out.to_string(), shape);
            }
        }

        // planner 逐步走（与 infer_shapes 同序），首个分歧/错误即报，
        // 错误时带两侧输入形状——分歧的根因几乎总在报错节点的上游。
        let (graph, initializers) = session.into_parts();
        let mut values: HashMap<String, Value> = initializers
            .iter()
            .map(|(name, t)| {
                let konst = match t.dtype {
                    DType::I64 => Some(Const::I64(t.i64.clone())),
                    DType::F32 => Some(Const::F32(t.f32.to_vec())),
                };
                (
                    name.clone(),
                    Value {
                        shape: t.shape.clone(),
                        dtype: t.dtype,
                        konst,
                    },
                )
            })
            .collect();
        values.insert(
            input_name.clone(),
            Value {
                shape: vec![1, 3, h, w],
                dtype: DType::F32,
                konst: None,
            },
        );

        let mut checked = 0usize;
        let mut first_bad: Option<String> = None;
        for n in &graph.nodes {
            let out_name = &n.outputs[0];
            match shape::infer_node(n, &values) {
                Ok(v) => {
                    if let Some(want) = exec.get(out_name) {
                        if &v.shape != want {
                            let inputs: Vec<String> = n
                                .inputs
                                .iter()
                                .map(|nm| {
                                    let pv = values
                                        .get(nm)
                                        .map(|v| format!("{:?}", v.shape))
                                        .unwrap_or_else(|| "<无>".into());
                                    let ev = exec
                                        .get(nm)
                                        .map(|s| format!("{s:?}"))
                                        .unwrap_or_else(|| "未落盘".into());
                                    format!("  {nm}: planner={pv} executor={ev}")
                                })
                                .collect();
                            first_bad = Some(format!(
                                "节点 {} ({}): planner {:?} != executor {:?}\n{}",
                                n.name,
                                n.op_type,
                                v.shape,
                                want,
                                inputs.join("\n")
                            ));
                        } else {
                            checked += 1;
                        }
                    }
                    values.insert(out_name.clone(), v);
                }
                Err(e) => {
                    let inputs: Vec<String> = n
                        .inputs
                        .iter()
                        .map(|nm| {
                            let pv = values
                                .get(nm)
                                .map(|v| format!("{:?}", v.shape))
                                .unwrap_or_else(|| "<无>".into());
                            let ev = exec
                                .get(nm)
                                .map(|s| format!("{s:?}"))
                                .unwrap_or_else(|| "未落盘".into());
                            format!("  {nm}: planner={pv} executor={ev}")
                        })
                        .collect();
                    first_bad = Some(format!(
                        "节点 {} ({}): 推理失败 {e}\n{}",
                        n.name,
                        n.op_type,
                        inputs.join("\n")
                    ));
                }
            }
            if first_bad.is_some() {
                break;
            }
        }
        let _ = std::fs::remove_dir_all(&dump_dir);
        unsafe {
            // SAFETY: 同上——本测试独占该环境变量。
            std::env::remove_var("QPPOCR_DUMP_DIR");
        }
        if let Some(msg) = first_bad {
            panic!("首个分歧：\n{msg}");
        }
        assert!(checked > 50, "对拍节点数异常地少：{checked}");
        eprintln!("[plan] det@{h}x{w}: {checked} 个节点形状与 executor 逐节点一致");
    }
}

#[cfg(test)]
mod extra_tests {
    use super::*;
    use qppocr_core::executor::Session;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    /// det 图的算子/属性直方图——内核覆盖清单（写内核前先看真实分布）。
    #[test]
    fn det_op_histogram() {
        let p = PathBuf::from("../../models/tiny/det.onnx");
        if !p.is_file() {
            eprintln!("[plan] 无模型，跳过");
            return;
        }
        let session = Session::from_memory(&std::fs::read(&p).unwrap(), "tiny.det").unwrap();
        let (graph, _) = session.into_parts();
        let mut ops: BTreeMap<String, usize> = BTreeMap::new();
        let mut conv_act: BTreeMap<String, usize> = BTreeMap::new();
        let mut conv_group: BTreeMap<i64, usize> = BTreeMap::new();
        let mut conv_k: BTreeMap<(i64, i64), usize> = BTreeMap::new();
        for n in &graph.nodes {
            *ops.entry(n.op_type.clone()).or_default() += 1;
            if n.op_type == "Conv" {
                let act = n.attr("act").filter(|a| a.has_i).map(|a| a.i).unwrap_or(0);
                *conv_act.entry(format!("act={act}")).or_default() += 1;
                *conv_group.entry(get_i(n.attr("group"), 1)).or_default() += 1;
                // 权重形状不在图里——内核形状在 planner 推理里，这里先记属性
                let ks = n
                    .attr("kernel_shape")
                    .map(|a| a.ints.clone())
                    .unwrap_or_default();
                if ks.len() == 2 {
                    *conv_k.entry((ks[0], ks[1])).or_default() += 1;
                }
            }
        }
        eprintln!("[plan] det 算子直方图: {ops:?}");
        eprintln!(
            "[plan] Conv act: {conv_act:?} | group: {conv_group:?} | kernel_shape: {conv_k:?}"
        );
    }
}
