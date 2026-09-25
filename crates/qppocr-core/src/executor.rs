//! 执行器（设计文档 ）：带引用计数释放的 arena 逐节点执行。
//!
//! 就绪判定 = 输入都在 arena 里；按序扫描 + 最多 8 趟重试。
//! `take0`：输入是最后消费者时**移动**而不是拷贝——in-place 算子直接吃
//! 原缓冲，省一次多 MB 的分配和一整趟内存。
//!
//! ## 逐节点落盘（阶段 2 判据的对拍机制）
//!
//! `QPPOCR_DUMP_DIR=<dir>` 时每个节点输出按  `QPPOCR_DUMP_DIR` 相同的
//! 格式落盘（`%06d.f32`：i32 rank + i64×rank 形状 + f32 数据，manifest.tsv），
//! 两侧目录逐文件 diff 即「中间张量逐位一致」的判据。

use std::collections::HashMap;
use std::io::Write;

use crate::error::{Error, Result};
use crate::onnx::model::{Attribute, Graph, Node};
use crate::tensor::{DType, Tensor};
use qppocr_kernels::buf::F32Buf;

fn get_i(a: Option<&Attribute>, dflt: i64) -> i64 {
    a.filter(|x| x.has_i).map(|x| x.i).unwrap_or(dflt)
}

fn get_f(a: Option<&Attribute>, dflt: f32) -> f32 {
    a.filter(|x| x.has_f).map(|x| x.f).unwrap_or(dflt)
}

/// 形状类输入（Reshape/Unsqueeze/Slice 的目标）规范上是 int64，但 Cast
/// 可能留下 float，有些导出器也直接那么发。哪个缓冲有值读哪个。
fn as_i64(t: &Tensor) -> Vec<i64> {
    if !t.i64.is_empty() {
        return t.i64.clone();
    }
    t.f32.iter().map(|v| v.round() as i64).collect()
}

/// axes 可能是属性（opset ≤ 12）或输入张量（opset 13+）。
fn axes_from(
    n: &Node,
    arena: &HashMap<String, Tensor>,
    initializers: &HashMap<String, Tensor>,
) -> Vec<i64> {
    if let Some(a) = n.attr("axes") {
        if !a.ints.is_empty() {
            return a.ints.clone();
        }
    }
    if n.inputs.len() >= 2 {
        if let Some(t) = arena
            .get(&n.inputs[1])
            .or_else(|| initializers.get(&n.inputs[1]))
        {
            return as_i64(t);
        }
    }
    Vec::new()
}

/// 任意 axis 的 softmax（外维并行；exp 占大头，中等张量也值得并行）。
/// axis 是最后一维时委托给向量化内核（rec 注意力的每个 softmax 都是）；
/// 通用路径是逐元素标量。★ 通用路径  用 libm exp、我们用同一多项式
/// （`exp1`），低 位差异 ≤1 ulp——声明过的偏差，对拍若在此翻车有据可查。
fn softmax_axis(t: &mut Tensor, axis: isize) {
    let r = t.rank() as isize;
    let axis = if axis < 0 { axis + r } else { axis };
    let inner_len = *t.shape.last().unwrap_or(&1);
    if axis == r - 1 && r >= 1 && inner_len > 0 {
        let inner = t.shape[(r - 1) as usize] as usize;
        qppocr_kernels::activation::softmax_last_dim(&mut t.f32, inner);
        return;
    }
    let (mut outer, mut mid, mut inner) = (1i64, 1i64, 1i64);
    for (i, &d) in t.shape.iter().enumerate() {
        let i = i as isize;
        if i < axis {
            outer *= d;
        } else if i == axis {
            mid = d;
        } else {
            inner *= d;
        }
    }
    qppocr_kernels::activation::softmax_axis_generic(&mut t.f32, outer, mid, inner);
}

/// ONNX auto_pad：导出器可能整个省掉 `pads`、让运行时从输入尺寸推导
///（SAME_UPPER/SAME_LOWER）。end padding 可以超过 begin，内核按两个都传
/// 处理。
#[allow(clippy::too_many_arguments)]
fn resolve_pads(
    n: &Node,
    in_h: i64,
    in_w: i64,
    kh: i64,
    kw: i64,
    sh: i64,
    sw: i64,
    dh: i64,
    dw: i64,
) -> (i64, i64, i64, i64) {
    let (mut ph, mut pw, mut peh, mut pew) = (0i64, 0, 0, 0);
    let ap = n.attr("auto_pad").filter(|a| a.has_s).map(|a| a.s.clone());
    if let Some(a) = n.attr("pads") {
        if a.ints.len() >= 4 {
            ph = a.ints[0];
            pw = a.ints[1];
            peh = a.ints[2];
            pew = a.ints[3];
        }
    }
    if ap.as_deref() != Some("SAME_UPPER") && ap.as_deref() != Some("SAME_LOWER") {
        return (ph, pw, peh, pew); // NOTSET / VALID 保留 pads
    }
    let (eff_kh, eff_kw) = ((kh - 1) * dh + 1, (kw - 1) * dw + 1);
    let oh = in_h.div_euclid(sh) + i64::from(in_h % sh != 0); // ceil
    let ow = in_w.div_euclid(sw) + i64::from(in_w % sw != 0);
    let th = ((oh - 1) * sh + eff_kh - in_h).max(0);
    let tw = ((ow - 1) * sw + eff_kw - in_w).max(0);
    if ap.as_deref() == Some("SAME_UPPER") {
        ph = th / 2;
        peh = th - ph;
        pw = tw / 2;
        pew = tw - pw;
    } else {
        peh = th / 2;
        ph = th - peh;
        pew = tw / 2;
        pw = tw - pew;
    }
    (ph, pw, peh, pew)
}

/// 一次加载、多次 run 的会话。权重只读，`&self` 跑图。
pub struct Session {
    /// 解析后的图（已优化）。
    pub graph: Graph,
    /// 权重 arena 的输入（run 时 clone 进运行 arena——后续缓冲池会接管）。
    initializers: HashMap<String, Tensor>,
}

impl Session {
    /// 从内存打开模型。
    pub fn from_memory(data: &[u8], display_name: &str) -> Result<Self> {
        let mut graph = crate::onnx::load_onnx_memory(data, display_name)?;
        // ★ 权重单份持有（take 而非 clone）：clone 让 det+rec+cls 的全部
        // 权重双份常驻（small 档 ≈33 MB × 2），PeakWS 实测多 ~19 MB。
        // graph.initializers 留空——运行期读者只有下面的 run（已改读
        // self.initializers）；optimize 系列都在 load 期、take 之前跑完。
        let initializers = std::mem::take(&mut graph.initializers);
        Ok(Self {
            graph,
            initializers,
        })
    }

    /// 从文件打开模型。
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let mut graph = crate::onnx::load_onnx(path)?;
        let initializers = std::mem::take(&mut graph.initializers);
        Ok(Self {
            graph,
            initializers,
        })
    }

    /// 跑一遍图。`inputs` 是 (名字, 张量) 列表；返回按 `graph.outputs`
    /// 顺序排列的输出。
    pub fn run(&self, inputs: Vec<(String, Tensor)>) -> Result<Vec<Tensor>> {
        // FTZ/DAZ：调用线程可能不是池的创建者
        qppocr_kernels::par::enable_flush_denormals();

        let g = &self.graph;
        // ★ 权重不进 arena：每次 run 克隆全部 initializer（det 445K floats +
        // 每个权重的 String 键）在 100ms 级的图上是纯浪费。arena 只放中间
        // 张量；查找两级（arena → initializers），释放跳过 initializers
        //（refs 计数那里已有该判断）。
        let mut arena: HashMap<String, Tensor> = HashMap::new();
        for (name, t) in inputs {
            arena.insert(name, t);
        }
        // 两级查找：先中间张量、再权重。借用生命周期要显式统一到 arena。
        macro_rules! get {
            ($arena:expr, $name:expr) => {
                match $arena.get($name) {
                    Some(t) => Some(t),
                    None => self.initializers.get($name),
                }
            };
        }

        // 对拍落盘（QPPOCR_DUMP_DIR 同格式）
        // 按会话分目录（s0_/s1_/s2_）：det/cls/rec 三个会话的节点索引都从
        // 0 起，共用目录会互相覆盖——对拍时无法区分归属。
        let dump_dir = std::env::var("QPPOCR_DUMP_DIR").ok().map(|d| {
            let _ = std::fs::create_dir_all(&d);
            static SESSION: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let id = SESSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            (d, id)
        });

        // per-op profile（QPPOCR_PROF=1）：按算子累计内核时间。
        let prof = std::env::var("QPPOCR_PROF").is_ok();
        let mut prof_acc: std::collections::HashMap<String, (f64, u32)> =
            std::collections::HashMap::new();
        let prof_t0 = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 消费者引用计数
        let mut refs: HashMap<&str, i64> = HashMap::new();
        for n in &g.nodes {
            for inn in &n.inputs {
                if !inn.is_empty() {
                    *refs.entry(inn.as_str()).or_insert(0) += 1;
                }
            }
        }

        let mut done = vec![false; g.nodes.len()];
        let mut remaining = g.nodes.len();
        let mut passes = 0;
        while remaining > 0 && passes < 8 {
            passes += 1;
            for i in 0..g.nodes.len() {
                if done[i] {
                    continue;
                }
                let n = &g.nodes[i];
                let mut ready = true;
                for inn in &n.inputs {
                    if !inn.is_empty() && get!(&arena, inn).is_none() {
                        ready = false;
                        break;
                    }
                }
                if !ready {
                    continue;
                }

                // inputs[idx] 是否只被本节点消费（可移动进输出）
                let takeable = |idx: usize| -> bool {
                    if idx >= n.inputs.len() {
                        return false;
                    }
                    let nm = &n.inputs[idx];
                    if nm.is_empty() {
                        return false;
                    }
                    match refs.get(nm.as_str()) {
                        Some(&c) if c == 1 => {}
                        _ => return false,
                    }
                    if self.initializers.contains_key(nm) {
                        return false;
                    }
                    if g.is_graph_output(nm) {
                        return false;
                    }
                    true
                };

                let op = n.op_type.as_str();
                let op_t0 = if prof {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let prof_key = if prof && op == "Conv" {
                    // ★ 极端输入下张量可能缺失（不该发生，但缺了要报错而不是 panic）：
                    //   整个 run 的错误契约是 Err(Graph)，panic 会让整批识别中断。
                    let x = get!(&arena, &n.inputs[0]).ok_or_else(|| {
                        Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                    })?;
                    let w = get!(&arena, &n.inputs[1]).ok_or_else(|| {
                        Error::Graph(format!("{op}: missing input {}", n.inputs[1]))
                    })?;
                    format!(
                        "Conv {}->{} k{}x{} @{}x{} g{}",
                        x.shape[1],
                        w.shape[0],
                        w.shape[2],
                        w.shape[3],
                        x.shape[2],
                        x.shape[3],
                        get_i(n.attr("group"), 1)
                    )
                } else if prof {
                    op.to_string()
                } else {
                    String::new()
                };
                let take0 = |arena: &mut HashMap<String, Tensor>| -> Result<Tensor> {
                    if takeable(0) {
                        arena.remove(&n.inputs[0]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                        })
                    } else {
                        // 权重或共享输入：从两级里拷贝
                        match arena.get(&n.inputs[0]) {
                            Some(t) => Ok(t.clone()),
                            None => self
                                .initializers
                                .get(&n.inputs[0])
                                .ok_or_else(|| {
                                    Error::Graph(format!("{op}: missing weight {}", n.inputs[0]))
                                })
                                .cloned(),
                        }
                    }
                };

                match op {
                    "Conv" => {
                        let x = get!(&arena, &n.inputs[0]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                        })?;
                        let w = get!(&arena, &n.inputs[1]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[1]))
                        })?;
                        let bias = if n.inputs.len() > 2 {
                            get!(&arena, &n.inputs[2]).map(|t| t.f32.as_slice())
                        } else {
                            None
                        };
                        let (sh, sw) = strides_of(n);
                        let (dh, dw) = dilations_of(n);
                        let group = get_i(n.attr("group"), 1) as usize;
                        let (ph, pw, peh, pew) = resolve_pads(
                            n, x.shape[2], x.shape[3], w.shape[2], w.shape[3], sh, sw, dh, dw,
                        );
                        // 融合激活挂在 Conv 节点上（graph_opt）：内核对刚写出的
                        // 输出施加它，独立节点（和它的分叉）消失。
                        let mut act = qppocr_kernels::activation::Activation::default();
                        if let Some(aa) = n.attr("act").filter(|a| a.has_i) {
                            if aa.i == 1 {
                                act = qppocr_kernels::activation::Activation::gelu(
                                    get_f(n.attr("act_c1"), 1.414_213_5),
                                    get_f(n.attr("act_c2"), 1.0),
                                    get_f(n.attr("act_c3"), 0.5),
                                );
                            }
                        }
                        let mut y = F32Buf::new();
                        let x4 = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
                        let w4 = [w.shape[0], w.shape[1], w.shape[2], w.shape[3]];
                        let os = qppocr_kernels::conv::conv2d(
                            &x.f32,
                            &x4,
                            &w.f32,
                            &w4,
                            bias,
                            &qppocr_kernels::conv::ConvParams {
                                sh: sh as usize,
                                sw: sw as usize,
                                ph: ph as usize,
                                pw: pw as usize,
                                peh: peh as usize,
                                pew: pew as usize,
                                dh: dh as usize,
                                dw: dw as usize,
                                group,
                            },
                            &act,
                            &mut y,
                        );
                        arena.insert(
                            n.outputs[0].clone(),
                            Tensor {
                                name: n.outputs[0].clone(),
                                shape: os.to_vec(),
                                dtype: DType::F32,
                                f32: y,
                                i64: Vec::new(),
                            },
                        );
                    }
                    "ConvTranspose" => {
                        let x = get!(&arena, &n.inputs[0]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                        })?;
                        let w = get!(&arena, &n.inputs[1]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[1]))
                        })?;
                        let (sh, sw) = strides_of(n);
                        let pads = pads_of(n);
                        let mut y = F32Buf::new();
                        let x4 = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
                        let w4 = [w.shape[0], w.shape[1], w.shape[2], w.shape[3]];
                        let os = qppocr_kernels::conv::convtranspose2d(
                            &x.f32,
                            &x4,
                            &w.f32,
                            &w4,
                            sh as usize,
                            sw as usize,
                            pads.0 as usize,
                            pads.1 as usize,
                            &mut y,
                        );
                        arena.insert(
                            n.outputs[0].clone(),
                            Tensor {
                                name: n.outputs[0].clone(),
                                shape: os.to_vec(),
                                dtype: DType::F32,
                                f32: y,
                                i64: Vec::new(),
                            },
                        );
                    }
                    "BatchNormalization" => {
                        // 输入是最后消费者时就地：batchnorm 先 `y = x` 再覆写
                        // 每个元素，能拿就别拷。
                        let (sc, bi, me, va) = (
                            get!(&arena, &n.inputs[1])
                                .ok_or_else(|| {
                                    Error::Graph(format!("{op}: missing input {}", n.inputs[1]))
                                })?
                                .f32
                                .to_vec(),
                            get!(&arena, &n.inputs[2])
                                .ok_or_else(|| {
                                    Error::Graph(format!("{op}: missing input {}", n.inputs[2]))
                                })?
                                .f32
                                .to_vec(),
                            get!(&arena, &n.inputs[3])
                                .ok_or_else(|| {
                                    Error::Graph(format!("{op}: missing input {}", n.inputs[3]))
                                })?
                                .f32
                                .to_vec(),
                            get!(&arena, &n.inputs[4])
                                .ok_or_else(|| {
                                    Error::Graph(format!("{op}: missing input {}", n.inputs[4]))
                                })?
                                .f32
                                .to_vec(),
                        );
                        let mut y = take0(&mut arena)?;
                        let mut out = F32Buf::new();
                        qppocr_kernels::shape::batchnorm(
                            &y.f32,
                            &y.shape,
                            &sc,
                            &bi,
                            &me,
                            &va,
                            get_f(n.attr("epsilon"), 1e-5),
                            &mut out,
                        );
                        y.f32 = out;
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "Relu" => {
                        let mut y = take0(&mut arena)?;
                        qppocr_kernels::activation::relu_inplace(&mut y.f32);
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "Sigmoid" => {
                        let mut y = take0(&mut arena)?;
                        // 原地拷贝语义：sigmoid(x) 写满新缓冲
                        let src = y.f32.clone();
                        let mut out = F32Buf::with_zeroed(src.len());
                        qppocr_kernels::activation::sigmoid_tensor(&src, &mut out);
                        y.f32 = out;
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "HardSigmoid" => {
                        let mut y = take0(&mut arena)?;
                        let src = y.f32.clone();
                        let mut out = F32Buf::with_zeroed(src.len());
                        qppocr_kernels::activation::hardsigmoid(
                            &src,
                            get_f(n.attr("alpha"), 0.2),
                            get_f(n.attr("beta"), 0.5),
                            &mut out,
                        );
                        y.f32 = out;
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "FusedGelu" => {
                        let mut y = take0(&mut arena)?;
                        let (c1, c2, c3) = (
                            get_f(n.attr("c1"), 1.414_213_5),
                            get_f(n.attr("c2"), 1.0),
                            get_f(n.attr("c3"), 0.5),
                        );
                        qppocr_kernels::activation::gelu_inplace(&mut y.f32, c1, c2, c3);
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "Erf" => {
                        let mut y = take0(&mut arena)?;
                        qppocr_kernels::activation::erf_inplace(&mut y.f32);
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "Sqrt" => {
                        let mut y = take0(&mut arena)?;
                        qppocr_kernels::activation::sqrt_inplace(&mut y.f32);
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "Clip" => {
                        let mut lo = -3.4e38f32;
                        let mut hi = 3.4e38f32;
                        if let Some(a) = n.attr("min").filter(|a| a.has_f) {
                            lo = a.f;
                        }
                        if let Some(a) = n.attr("max").filter(|a| a.has_f) {
                            hi = a.f;
                        }
                        if n.inputs.len() > 1 && !n.inputs[1].is_empty() {
                            if let Some(t) = get!(&arena, &n.inputs[1]) {
                                if t.numel() > 0 && !t.f32.is_empty() {
                                    lo = t.f32[0];
                                }
                            }
                        }
                        if n.inputs.len() > 2 && !n.inputs[2].is_empty() {
                            if let Some(t) = get!(&arena, &n.inputs[2]) {
                                if t.numel() > 0 && !t.f32.is_empty() {
                                    hi = t.f32[0];
                                }
                            }
                        }
                        let mut y = take0(&mut arena)?;
                        qppocr_kernels::activation::clip_inplace(&mut y.f32, lo, hi);
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "Add" | "Sub" | "Mul" | "Div" | "Pow" => {
                        let binop = match op {
                            "Add" => qppocr_kernels::elementwise::BinOp::Add,
                            "Sub" => qppocr_kernels::elementwise::BinOp::Sub,
                            "Mul" => qppocr_kernels::elementwise::BinOp::Mul,
                            "Div" => qppocr_kernels::elementwise::BinOp::Div,
                            _ => qppocr_kernels::elementwise::BinOp::Pow,
                        };
                        // 操作数 0 在本节点死亡且已持有结果形状 → 就地：
                        // 省一次多 MB 分配和一整趟额外内存
                        let can_ip = takeable(0)
                            && qppocr_kernels::elementwise::binary_can_inplace(
                                get!(&arena, &n.inputs[0])
                                    .ok_or_else(|| {
                                        Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                                    })?
                                    .shape
                                    .as_slice(),
                                get!(&arena, &n.inputs[1])
                                    .ok_or_else(|| {
                                        Error::Graph(format!("{op}: missing input {}", n.inputs[1]))
                                    })?
                                    .shape
                                    .as_slice(),
                            );
                        let out = if can_ip {
                            // can_ip 已含 takeable(0)（存在性），缺了就显式报错
                            let mut acc = arena.remove(&n.inputs[0]).ok_or_else(|| {
                                Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                            })?;
                            qppocr_kernels::elementwise::binary_op_inplace(
                                &mut acc.f32,
                                &acc.shape,
                                get!(&arena, &n.inputs[1])
                                    .ok_or_else(|| {
                                        Error::Graph(format!("{op}: missing input {}", n.inputs[1]))
                                    })?
                                    .f32
                                    .as_slice(),
                                get!(&arena, &n.inputs[1])
                                    .ok_or_else(|| {
                                        Error::Graph(format!("{op}: missing input {}", n.inputs[1]))
                                    })?
                                    .shape
                                    .as_slice(),
                                binop,
                            );
                            acc
                        } else {
                            let a = get!(&arena, &n.inputs[0]).ok_or_else(|| {
                                Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                            })?;
                            let b = get!(&arena, &n.inputs[1]).ok_or_else(|| {
                                Error::Graph(format!("{op}: missing input {}", n.inputs[1]))
                            })?;
                            let (data, shape) = qppocr_kernels::elementwise::binary_op(
                                &a.f32, &a.shape, &b.f32, &b.shape, binop,
                            );
                            Tensor {
                                name: String::new(),
                                shape,
                                dtype: DType::F32,
                                f32: F32Buf::from_vec(&data),
                                i64: Vec::new(),
                            }
                        };
                        arena.insert(n.outputs[0].clone(), out);
                    }
                    "Softmax" => {
                        let mut y = take0(&mut arena)?;
                        softmax_axis(&mut y, get_i(n.attr("axis"), -1) as isize);
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "GlobalAveragePool" => {
                        let x = get!(&arena, &n.inputs[0]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                        })?;
                        let (nn, c) = (x.shape[0] as usize, x.shape[1] as usize);
                        let mut out = F32Buf::new();
                        qppocr_kernels::pool2d::global_avg_pool(&x.f32, nn, c, &mut out);
                        arena.insert(
                            n.outputs[0].clone(),
                            Tensor {
                                name: n.outputs[0].clone(),
                                shape: vec![nn as i64, c as i64, 1, 1],
                                dtype: DType::F32,
                                f32: out,
                                i64: Vec::new(),
                            },
                        );
                    }
                    "AveragePool" | "MaxPool" => {
                        let ks = n
                            .attr("kernel_shape")
                            .filter(|a| a.ints.len() == 2)
                            .ok_or_else(|| Error::Graph(format!("{op}: kernel_shape required")))?;
                        let st = strides_of(n);
                        let x = get!(&arena, &n.inputs[0]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                        })?;
                        let (nn, c, h, w) = (
                            x.shape[0] as usize,
                            x.shape[1] as usize,
                            x.shape[2] as usize,
                            x.shape[3] as usize,
                        );
                        let (ph, pw, peh, pew) = resolve_pads(
                            n, x.shape[2], x.shape[3], ks.ints[0], ks.ints[1], st.0, st.1, 1, 1,
                        );
                        let mut out = F32Buf::new();
                        let (oh, ow) = qppocr_kernels::pool2d::pool2d(
                            &x.f32,
                            nn,
                            c,
                            h,
                            w,
                            ks.ints[0] as usize,
                            ks.ints[1] as usize,
                            st.0 as usize,
                            st.1 as usize,
                            ph as usize,
                            pw as usize,
                            peh as usize,
                            pew as usize,
                            op == "MaxPool",
                            &mut out,
                        );
                        arena.insert(
                            n.outputs[0].clone(),
                            Tensor {
                                name: n.outputs[0].clone(),
                                shape: vec![nn as i64, c as i64, oh as i64, ow as i64],
                                dtype: DType::F32,
                                f32: out,
                                i64: Vec::new(),
                            },
                        );
                    }
                    "Resize" => {
                        let x = get!(&arena, &n.inputs[0]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                        })?;
                        let (oh, ow): (usize, usize);
                        if n.inputs.len() >= 4 && !n.inputs[3].is_empty() {
                            let sizes = get!(&arena, &n.inputs[3]).ok_or_else(|| {
                                Error::Graph("Resize: missing sizes input".into())
                            })?;
                            // 按**扁平数组末两位**取 H/W（基准是 i64.size()-2），
                            // 不是按 rank——rank-1 的 sizes 长度 4 很常见
                            let sv = as_i64(sizes);
                            oh = sv[sv.len() - 2] as usize;
                            ow = sv[sv.len() - 1] as usize;
                        } else if n.inputs.len() >= 3 && !n.inputs[2].is_empty() {
                            let sc = get!(&arena, &n.inputs[2]).ok_or_else(|| {
                                Error::Graph("Resize: missing scales input".into())
                            })?;
                            let sh_ = sc.f32[sc.f32.len() - 2];
                            let sw_ = sc.f32[sc.f32.len() - 1];
                            oh = (x.shape_at(2) as f32 * sh_) as usize;
                            ow = (x.shape_at(3) as f32 * sw_) as usize;
                        } else {
                            return Err(Error::Graph("Resize: no scales/sizes".into()));
                        }
                        let mode = n
                            .attr("mode")
                            .filter(|a| a.has_s)
                            .map(|a| a.s.clone())
                            .unwrap_or_else(|| "nearest".into());
                        let ctm = n
                            .attr("coordinate_transformation_mode")
                            .filter(|a| a.has_s)
                            .map(|a| a.s.clone())
                            .unwrap_or_else(|| "half_pixel".into());
                        let (nn, c, h, w) = (
                            x.shape[0] as usize,
                            x.shape[1] as usize,
                            x.shape[2] as usize,
                            x.shape[3] as usize,
                        );
                        let mut out = F32Buf::new();
                        let shape = vec![nn as i64, c as i64, oh as i64, ow as i64];
                        match mode.as_str() {
                            "nearest" => qppocr_kernels::resize::resize_nearest(
                                &x.f32, nn, c, h, w, oh, ow, &mut out,
                            ),
                            "linear" => qppocr_kernels::resize::resize_bilinear(
                                &x.f32,
                                nn,
                                c,
                                h,
                                w,
                                oh,
                                ow,
                                ctm == "align_corners",
                                &mut out,
                            ),
                            m => return Err(Error::Graph(format!("Resize mode {m} unsupported"))),
                        }
                        arena.insert(
                            n.outputs[0].clone(),
                            Tensor {
                                name: n.outputs[0].clone(),
                                shape,
                                dtype: DType::F32,
                                f32: out,
                                i64: Vec::new(),
                            },
                        );
                    }
                    "Concat" => {
                        let axis = get_i(n.attr("axis"), 0);
                        use qppocr_kernels::shape::PayloadRef;
                        let xs: Vec<&Tensor> = n
                            .inputs
                            .iter()
                            .map(|inn| {
                                get!(&arena, inn).ok_or_else(|| {
                                    Error::Graph(format!("{op}: missing input {inn}"))
                                })
                            })
                            .collect::<Result<Vec<&Tensor>>>()?;
                        let xs_shape: Vec<&[i64]> = xs.iter().map(|t| t.shape.as_slice()).collect();
                        let payloads: Vec<PayloadRef<'_>> = xs
                            .iter()
                            .map(|t| match t.dtype {
                                DType::F32 => PayloadRef::F32(&t.f32),
                                DType::I64 => PayloadRef::I64(&t.i64),
                            })
                            .collect();
                        let mut out = match xs[0].dtype {
                            DType::F32 => qppocr_kernels::shape::Payload::F32(F32Buf::new()),
                            DType::I64 => qppocr_kernels::shape::Payload::I64(Vec::new()),
                        };
                        let shape =
                            qppocr_kernels::shape::concat_any(&payloads, &xs_shape, axis, &mut out);
                        let y = match out {
                            qppocr_kernels::shape::Payload::F32(v) => Tensor {
                                name: n.outputs[0].clone(),
                                shape,
                                dtype: DType::F32,
                                f32: v,
                                i64: Vec::new(),
                            },
                            qppocr_kernels::shape::Payload::I64(v) => Tensor {
                                name: n.outputs[0].clone(),
                                shape,
                                dtype: DType::I64,
                                f32: F32Buf::new(),
                                i64: v,
                            },
                        };
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "Slice" => {
                        // opset 13+/11：starts/ends/axes/steps 是输入；
                        // opset ≤9：starts/ends/axes 是**属性**（上游 cls
                        // opset 7 正是这种—— 参考在这里越界读直接段错误，
                        // 这是它跑不了上游 cls 的根因）。
                        let (starts, ends, axes) = if n.inputs.len() >= 3
                            && !n.inputs[1].is_empty()
                            && !n.inputs[2].is_empty()
                        {
                            // starts/ends/axes 通常是折叠进 initializers 的
                            // 常量（「权重零克隆」后不进 arena，查两级）
                            let grab = |nm: &str| -> Result<Vec<i64>> {
                                let t = arena
                                    .get(nm)
                                    .or_else(|| self.initializers.get(nm))
                                    .ok_or_else(|| Error::Graph(format!("Slice: missing {nm}")))?;
                                Ok(as_i64(t))
                            };
                            let axes = if n.inputs.len() > 3 && !n.inputs[3].is_empty() {
                                grab(&n.inputs[3])?
                            } else {
                                Vec::new()
                            };
                            (grab(&n.inputs[1])?, grab(&n.inputs[2])?, axes)
                        } else {
                            let attr_ints = |nm: &str| -> Vec<i64> {
                                n.attr(nm).map(|a| a.ints.clone()).unwrap_or_default()
                            };
                            let starts = attr_ints("starts");
                            let ends = attr_ints("ends");
                            if starts.is_empty() || ends.is_empty() {
                                return Err(Error::Graph(format!(
                                    "Slice: neither inputs nor starts/ends attributes present (node {})",
                                    n.name
                                )));
                            }
                            (starts, ends, attr_ints("axes"))
                        };
                        let steps = if n.inputs.len() > 4 && !n.inputs[4].is_empty() {
                            let t = arena
                                .get(&n.inputs[4])
                                .or_else(|| self.initializers.get(&n.inputs[4]))
                                .ok_or_else(|| Error::Graph("Slice: missing steps".into()))?;
                            as_i64(t)
                        } else {
                            Vec::new()
                        };
                        let x = get!(&arena, &n.inputs[0]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                        })?;
                        use qppocr_kernels::shape::PayloadRef;
                        let (payload, shape) = match x.dtype {
                            DType::F32 => qppocr_kernels::shape::slice_tensor(
                                PayloadRef::F32(&x.f32),
                                &x.shape,
                                &starts,
                                &ends,
                                &axes,
                                &steps,
                            ),
                            DType::I64 => qppocr_kernels::shape::slice_tensor(
                                PayloadRef::I64(&x.i64),
                                &x.shape,
                                &starts,
                                &ends,
                                &axes,
                                &steps,
                            ),
                        };
                        let y = match payload {
                            qppocr_kernels::shape::Payload::F32(v) => Tensor {
                                name: n.outputs[0].clone(),
                                shape,
                                dtype: DType::F32,
                                f32: v,
                                i64: Vec::new(),
                            },
                            qppocr_kernels::shape::Payload::I64(v) => Tensor {
                                name: n.outputs[0].clone(),
                                shape,
                                dtype: DType::I64,
                                f32: F32Buf::new(),
                                i64: v,
                            },
                        };
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "Transpose" => {
                        let perm = n.attr("perm").map(|a| a.ints.clone()).unwrap_or_default();
                        let x = get!(&arena, &n.inputs[0]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                        })?;
                        use qppocr_kernels::shape::PayloadRef;
                        let (payload, shape) = match x.dtype {
                            DType::F32 => qppocr_kernels::shape::transpose_tensor(
                                PayloadRef::F32(&x.f32),
                                &x.shape,
                                &perm,
                            ),
                            DType::I64 => qppocr_kernels::shape::transpose_tensor(
                                PayloadRef::I64(&x.i64),
                                &x.shape,
                                &perm,
                            ),
                        };
                        let y = match payload {
                            qppocr_kernels::shape::Payload::F32(v) => Tensor {
                                name: n.outputs[0].clone(),
                                shape,
                                dtype: DType::F32,
                                f32: v,
                                i64: Vec::new(),
                            },
                            qppocr_kernels::shape::Payload::I64(v) => Tensor {
                                name: n.outputs[0].clone(),
                                shape,
                                dtype: DType::I64,
                                f32: F32Buf::new(),
                                i64: v,
                            },
                        };
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "Reshape" | "Squeeze" => {
                        // 只改形状：行主序元素序一致，无数据可搬。能拿就拿，
                        // 让赋值变自赋值。Squeeze 的 axes 可能在第二个输入。
                        let axes = if op == "Squeeze" {
                            axes_from(n, &arena, &self.initializers)
                        } else {
                            Vec::new()
                        };
                        let shp = if op == "Reshape" {
                            let t = get!(&arena, &n.inputs[1]).ok_or_else(|| {
                                Error::Graph("Reshape: missing shape input".into())
                            })?;
                            as_i64(t) // 按值取：take0 之后原槽位要失效
                        } else {
                            Vec::new()
                        };
                        let mut y = take0(&mut arena)?;
                        y.shape = if op == "Reshape" {
                            qppocr_kernels::shape::reshape_shape(&y.shape, &shp)
                        } else {
                            squeeze_shape(&y.shape, &axes)
                        };
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "Unsqueeze" => {
                        // 同上：只形状。先拿输入再算，避免读到移动后的槽位。
                        let axes = axes_from(n, &arena, &self.initializers);
                        let mut y = take0(&mut arena)?;
                        let xsh = y.shape.clone();
                        // 输出秩 = 输入秩 + axes 数。插入掩码按 max(axes)+1
                        // 尺寸会悄悄丢尾维（axes=[1] 落在中间时）。
                        let rr = xsh.len() + axes.len();
                        let mut ins = vec![false; rr];
                        for &a in &axes {
                            let idx = if a < 0 { a + rr as i64 } else { a } as usize;
                            if idx >= rr {
                                return Err(Error::Graph("Unsqueeze: axis out of range".into()));
                            }
                            ins[idx] = true;
                        }
                        y.shape.clear();
                        let mut k = 0usize;
                        for i in 0..rr {
                            y.shape.push(if ins[i] {
                                1
                            } else {
                                let v = xsh[k];
                                k += 1;
                                v
                            });
                        }
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "ReduceMean" => {
                        let axes = axes_from(n, &arena, &self.initializers);
                        let x = get!(&arena, &n.inputs[0]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                        })?;
                        let mut out = F32Buf::new();
                        let shape = qppocr_kernels::shape::reduce_mean(
                            &x.f32,
                            &x.shape,
                            &axes,
                            get_i(n.attr("keepdims"), 1) != 0,
                            &mut out,
                        );
                        arena.insert(
                            n.outputs[0].clone(),
                            Tensor {
                                name: n.outputs[0].clone(),
                                shape,
                                dtype: DType::F32,
                                f32: out,
                                i64: Vec::new(),
                            },
                        );
                    }
                    "MatMul" => {
                        let a = get!(&arena, &n.inputs[0]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                        })?;
                        let b = get!(&arena, &n.inputs[1]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[1]))
                        })?;
                        let mut out = F32Buf::new();
                        let shape = qppocr_kernels::shape::matmul(
                            &a.f32, &a.shape, &b.f32, &b.shape, &mut out,
                        );
                        arena.insert(
                            n.outputs[0].clone(),
                            Tensor {
                                name: n.outputs[0].clone(),
                                shape,
                                dtype: DType::F32,
                                f32: out,
                                i64: Vec::new(),
                            },
                        );
                    }
                    "Shape" => {
                        let x = get!(&arena, &n.inputs[0]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                        })?;
                        arena.insert(
                            n.outputs[0].clone(),
                            Tensor {
                                name: n.outputs[0].clone(),
                                shape: vec![x.rank() as i64],
                                dtype: DType::I64,
                                f32: F32Buf::new(),
                                i64: x.shape.clone(),
                            },
                        );
                    }
                    "Cast" => {
                        let to = get_i(n.attr("to"), 1);
                        let x = match arena.get(&n.inputs[0]) {
                            Some(t) => t.clone(),
                            // ★ 缺键曾是 Index panic —— 换显式 Err（极端输入的
                            //   健壮性契约：任何图错误都走 Err(Graph) 而不是 panic）
                            None => self
                                .initializers
                                .get(&n.inputs[0])
                                .ok_or_else(|| {
                                    Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                                })?
                                .clone(),
                        };
                        let mut y = Tensor {
                            name: n.outputs[0].clone(),
                            shape: x.shape.clone(),
                            ..Default::default()
                        };
                        if to == 1 || to == 10 {
                            // to float
                            y.dtype = DType::F32;
                            y.f32 = match x.dtype {
                                DType::F32 => x.f32.clone(),
                                DType::I64 => {
                                    let v: Vec<f32> = x.i64.iter().map(|&v| v as f32).collect();
                                    F32Buf::from_vec(&v)
                                }
                            };
                        } else {
                            // to int（lround：四舍五入远离零）
                            y.dtype = DType::I64;
                            y.i64 = match x.dtype {
                                DType::F32 => x.f32.iter().map(|&v| v.round() as i64).collect(),
                                DType::I64 => x.i64.clone(),
                            };
                        }
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "Identity" => {
                        // ★ 与 Cast 同款：缺键曾是 Index panic。解析器折叠
                        //   Identity 后执行器不该再见到它，见到就带名报错。
                        let t = match arena.get(&n.inputs[0]) {
                            Some(t) => t.clone(),
                            None => match self.initializers.get(&n.inputs[0]) {
                                Some(w) => w.clone(),
                                None => {
                                    return Err(Error::Graph(format!(
                                        "{op}: missing input {}",
                                        n.inputs[0]
                                    )));
                                }
                            },
                        };
                        arena.insert(n.outputs[0].clone(), t);
                    }
                    "Constant" => {
                        // 解析器应该已折叠
                        return Err(Error::Graph("Constant node reached executor".into()));
                    }
                    other => {
                        return Err(Error::Graph(format!(
                            "unsupported op: {other} (node {})",
                            n.name
                        )));
                    }
                }

                if let Some(t0) = op_t0 {
                    let e = prof_acc.entry(prof_key).or_insert((0.0, 0));
                    e.0 += t0.elapsed().as_secs_f64() * 1000.0;
                    e.1 += 1;
                }
                #[allow(unused_assignments)]
                {
                    let _ = &op_t0;
                }
                // 对拍落盘（ QPPOCR_DUMP_DIR 同格式：%06d.f32 + manifest.tsv）
                if let Some(dir) = &dump_dir {
                    if let Some(t) = arena.get(&n.outputs[0]) {
                        if t.dtype == DType::F32 {
                            let (ddir, sid) = dir;
                            let path = format!("{}/s{}_{:06}.f32", ddir, sid, i);
                            if let Ok(file) = std::fs::File::create(&path) {
                                let mut f = std::io::BufWriter::new(file);
                                let _ = f.write_all(&(t.rank() as i32).to_le_bytes());
                                for d in &t.shape {
                                    let _ = f.write_all(&d.to_le_bytes());
                                }
                                let _ = write_f32_le(&mut f, &t.f32);
                            }
                            let mf = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(format!("{}/manifest.tsv", ddir));
                            if let Ok(mut mf) = mf {
                                let _ = writeln!(mf, "{}\t{}\t{}", i, op, n.outputs[0]);
                            }
                        }
                    }
                }

                // 释放已消费的输入
                for inn in &n.inputs {
                    if inn.is_empty() {
                        continue;
                    }
                    let nm = inn.as_str();
                    if let Some(cnt) = refs.get_mut(nm) {
                        *cnt -= 1;
                        if *cnt <= 0 {
                            let is_init = g.initializers.contains_key(nm);
                            if !is_init && !g.is_graph_output(nm) {
                                arena.remove(nm);
                            }
                        }
                    }
                }

                done[i] = true;
                remaining -= 1;
            }
        }
        if remaining > 0 {
            let mut missing = String::new();
            for n in &g.nodes {
                if missing.len() >= 400 {
                    break;
                }
                let need: Vec<&str> = n
                    .inputs
                    .iter()
                    .filter(|inn| !inn.is_empty() && !arena.contains_key(*inn))
                    .map(|s| s.as_str())
                    .collect();
                if !need.is_empty() {
                    missing.push_str(&format!(" [{} needs {}]", n.op_type, need.join(" ")));
                }
            }
            return Err(Error::Graph(format!(
                "graph has unresolvable nodes:{missing}"
            )));
        }

        if prof {
            let wall = prof_t0
                .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let kernel: f64 = prof_acc.values().map(|(ms, _)| *ms).sum();
            let mut v: Vec<_> = prof_acc.into_iter().collect();
            v.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
            eprintln!(
                "--- per-op profile [{}] (kernel {kernel:.1} ms, wall {wall:.1} ms, executor {:.1} ms = {:.1}%) ---",
                g.model_path,
                wall - kernel,
                100.0 * (wall - kernel) / wall.max(0.001)
            );
            for (op, (ms, cnt)) in v {
                eprintln!("  {op:<16} {ms:8.1} ms  x{cnt}");
            }
        }

        let mut outputs = Vec::with_capacity(g.outputs.len());
        for o in &g.outputs {
            let t = arena
                .remove(o)
                .ok_or_else(|| Error::Graph(format!("missing graph output {o}")))?;
            outputs.push(t);
        }
        Ok(outputs)
    }
}

fn strides_of(n: &Node) -> (i64, i64) {
    match n.attr("strides").filter(|a| !a.ints.is_empty()) {
        Some(a) => (a.ints[0], if a.ints.len() > 1 { a.ints[1] } else { 1 }),
        None => (1, 1),
    }
}

fn dilations_of(n: &Node) -> (i64, i64) {
    match n.attr("dilations").filter(|a| !a.ints.is_empty()) {
        Some(a) => (a.ints[0], if a.ints.len() > 1 { a.ints[1] } else { 1 }),
        None => (1, 1),
    }
}

fn pads_of(n: &Node) -> (i64, i64) {
    match n.attr("pads").filter(|a| !a.ints.is_empty()) {
        Some(a) => (a.ints[0], if a.ints.len() > 1 { a.ints[1] } else { 0 }),
        None => (0, 0),
    }
}

fn squeeze_shape(x_shape: &[i64], axes: &[i64]) -> Vec<i64> {
    let r = x_shape.len();
    if axes.is_empty() {
        return x_shape.iter().copied().filter(|&d| d != 1).collect();
    }
    let mut drop = vec![false; r];
    for &a in axes {
        let a = if a < 0 { a + r as i64 } else { a } as usize;
        drop[a] = true;
    }
    x_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !drop[*i])
        .map(|(_, &d)| d)
        .collect()
}

/// f32 逐元素小端写出（对拍落盘用；core 无 unsafe，不引 bytemuck）。
fn write_f32_le(f: &mut impl Write, v: &[f32]) -> std::io::Result<()> {
    for x in v {
        f.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}
