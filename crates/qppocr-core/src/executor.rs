//! 执行器：带引用计数释放的 arena 逐节点执行。
//!
//! 就绪判定 = 一个节点的入边全部就绪（见下方调度注释）。
//! `take0`：输入是最后消费者时**移动**而不是拷贝——in-place 算子直接吃
//! 原缓冲，省一次多 MB 的分配和一整趟内存。
//!
//! ## 逐节点落盘（位级一致的判据）
//!
//! `QPPOCR_DUMP_DIR=<dir>` 时每个节点输出按 `%06d.f32` 的
//! 格式落盘（i32 rank + i64×rank 形状 + f32 数据，另写 manifest.tsv），
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
    if let Some(a) = n.attr("axes")
        && !a.ints.is_empty()
    {
        return a.ints.clone();
    }
    if n.inputs.len() >= 2
        && let Some(t) = arena
            .get(&n.inputs[1])
            .or_else(|| initializers.get(&n.inputs[1]))
    {
        return as_i64(t);
    }
    Vec::new()
}

/// 任意 axis 的 softmax（外维并行；exp 占大头，中等张量也值得并行）。
/// axis 是最后一维时委托给向量化内核（rec 注意力的每个 softmax 都是）；
/// 通用路径是逐元素标量，exp 用 `exp1`（与向量版同一多项式），与 libm
/// 参考有 ≤1 ulp 量级的差异。
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
    if let Some(a) = n.attr("pads")
        && a.ints.len() >= 4
    {
        ph = a.ints[0];
        pw = a.ints[1];
        peh = a.ints[2];
        pew = a.ints[3];
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
    /// 逐节点 dump 目录。显式指定优先；否则回退 `QPPOCR_DUMP_DIR`
    /// 环境变量（CLI/示例用）。测试内必须显式指定——并行测试线程
    /// 共享进程环境，靠环境变量会让同时段其它 `run` 读到同一目录互踩。
    dump_dir: Option<String>,
}

impl Session {
    /// 从内存打开模型。
    pub fn from_memory(data: &[u8], display_name: &str) -> Result<Self> {
        let (graph, initializers) = Self::parts_from_memory(data, display_name)?;
        Ok(Self::from_parts(graph, initializers))
    }

    /// 从文件打开模型。
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let (graph, initializers) = Self::parts_from_path(path)?;
        Ok(Self::from_parts(graph, initializers))
    }

    /// 解析 ONNX 文件并把权重 take 成独立一份（装载前半程，pipeline 的
    /// 设备路径共用；之后 `graph.initializers` 为空）。
    pub(crate) fn parts_from_path(
        path: &std::path::Path,
    ) -> Result<(Graph, HashMap<String, Tensor>)> {
        let mut graph = crate::onnx::load_onnx(path)?;
        // ★ 权重单份持有（take 而非 clone）：见 [`Self::from_parts`]。
        let initializers = std::mem::take(&mut graph.initializers);
        Ok((graph, initializers))
    }

    /// 解析 ONNX 字节并把权重 take 成独立一份（同 [`Self::parts_from_path`]）。
    pub(crate) fn parts_from_memory(
        data: &[u8],
        display_name: &str,
    ) -> Result<(Graph, HashMap<String, Tensor>)> {
        let mut graph = crate::onnx::load_onnx_memory(data, display_name)?;
        // ★ 权重单份持有（take 而非 clone）：clone 让 det+rec+cls 的全部
        // 权重双份常驻（small 档 ≈33 MB × 2），PeakWS 实测多 ~19 MB。
        // graph.initializers 留空——运行期读者只有下面的 run（已改读
        // self.initializers）；optimize 系列都在 load 期、take 之前跑完。
        let initializers = std::mem::take(&mut graph.initializers);
        Ok((graph, initializers))
    }

    /// 从已解析的图与权重组装（图优化须已在装载期跑完）。
    ///
    /// 契约（设备工厂同样适用）：权重全在 `initializers` 参数，
    /// `graph.initializers` 必须为空——装载链路 take 之后就空了。
    /// `run` 的释放判定读 `graph.initializers`，混入权重会让 arena 里
    /// 同名张量永不释放。
    pub fn from_parts(graph: Graph, initializers: HashMap<String, Tensor>) -> Self {
        debug_assert!(
            graph.initializers.is_empty(),
            "from_parts: 权重应在 initializers 参数里，graph.initializers 须为空"
        );
        Self {
            graph,
            initializers,
            dump_dir: None,
        }
    }

    /// 显式指定逐节点 dump 目录（优先于 `QPPOCR_DUMP_DIR` 环境变量）。
    pub fn set_dump_dir(&mut self, dir: impl Into<String>) {
        self.dump_dir = Some(dir.into());
    }

    /// 拆回 (图, 权重)，所有权移出（供设备工厂重组用）。
    pub fn into_parts(self) -> (Graph, HashMap<String, Tensor>) {
        (self.graph, self.initializers)
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
        // 0 起，共用目录会互相覆盖——对拍时无法区分归属。计数器在
        // device 模块全局共享：混合部署（det-GPU + rec-CPU）下各持一个
        // 会互相覆盖对拍目录。
        let dump_dir = self
            .dump_dir
            .clone()
            .or_else(|| std::env::var("QPPOCR_DUMP_DIR").ok())
            .map(|d| {
                let _ = std::fs::create_dir_all(&d);
                let id = crate::device::next_dump_session_id();
                (d, id)
            });

        // per-op profile（QPPOCR_PROF=1）：按算子累计内核时间。
        let prof = std::env::var("QPPOCR_PROF").is_ok();
        let mut prof_acc: std::collections::HashMap<String, (f64, u32)> =
            std::collections::HashMap::new();
        let mut prof_b: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
        let prof_t0 = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut prof_bytes: u64 = 0;

        // 消费者引用计数
        let mut refs: HashMap<&str, i64> = HashMap::new();
        for n in &g.nodes {
            for inn in &n.inputs {
                if !inn.is_empty() {
                    *refs.entry(inn.as_str()).or_insert(0) += 1;
                }
            }
        }

        // ★ 就绪调度改成「未满足入边计数 + 队列」。
        //
        // 旧写法是每轮扫全表、逐个节点问「输入齐了吗」：一条 53 节点的
        // 顺序链要扫 53 轮 × 53 节点 ≈ 2800 次探测。cls 的图正好是这种
        // 小节点长链，profile 里 executor 开销 1.0-1.9 ms、占该次前向的
        // 20%——前向本身才 1.3 ms。改成计数后每个节点只在入边满足时被
        // 入队一次，复杂度从 O(n²) 降到 O(V+E)。
        //
        // 执行顺序只在「同时就绪」的节点之间变化；这些节点互不依赖，
        // 每个算子的输出只由自己的输入决定，因此结果逐位不变。
        let mut pending: Vec<u32> = vec![0; g.nodes.len()];
        let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, n) in g.nodes.iter().enumerate() {
            for inn in &n.inputs {
                if inn.is_empty() {
                    continue;
                }
                let nm = inn.as_str();
                if arena.contains_key(nm) || self.initializers.contains_key(nm) {
                    continue;
                }
                pending[i] += 1;
                consumers.entry(nm).or_default().push(i);
            }
        }
        // 用小顶堆而不是 FIFO：**始终执行下标最小的就绪节点**，这与旧写法
        // 「按下标扫、碰上就绪就执行」的执行顺序一致（ONNX 图按下标就是
        // 拓扑序）。顺序重要——按内存序推进时生产者刚写出就被消费者读，
        // 缓存是热的；换 FIFO 实测 cls 阶段反而慢 6%（4.62 → 4.92 ms/图），
        // 因为中间张量同时在世的数量变多、缓冲池命中率下降。
        let mut ready: std::collections::BinaryHeap<std::cmp::Reverse<usize>> = (0..g.nodes.len())
            .filter(|&i| pending[i] == 0)
            .map(std::cmp::Reverse)
            .collect();

        // Concat 零拷贝别名（P2）：计划按执行图静态判定，QPPOCR_NO_CONCAT_ALIAS
        // 可整体关闭（A/B 与逃生门）。
        let mut alias = ConcatAliasCtx::build(
            std::env::var_os("QPPOCR_NO_CONCAT_ALIAS").is_none(),
            g,
            &self.initializers,
        );

        let mut remaining = g.nodes.len();
        while let Some(std::cmp::Reverse(i)) = ready.pop() {
            {
                let n = &g.nodes[i];

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
                    // strides 缺省是 (1,1)。深度卷积在不同 stride 下走的是
                    // 完全不同的分支（s1 有向量化 x 向快路径、s≠1 退回逐元素），
                    // 不带 stride 的键会把两者混成一个数。
                    // NHWC 岛：稠密权重已预打包成 2D [K,M]、激活物理
                    // [n,h,w,c]——按物理形状取键，键名带 nhwc 前缀区分。
                    let st = strides_of(n);
                    if x.shape.len() == 4 && w.shape.len() == 2 {
                        format!(
                            "nhwc Conv {}->{} k1x1 @{}x{} g1 s{},{}",
                            x.shape[3], w.shape[1], x.shape[1], x.shape[2], st.0, st.1
                        )
                    } else if x.shape.len() == 4 && w.shape.len() == 4 {
                        format!(
                            "Conv {}->{} k{}x{} @{}x{} g{} s{},{}",
                            x.shape[1],
                            w.shape[0],
                            w.shape[2],
                            w.shape[3],
                            x.shape[2],
                            x.shape[3],
                            get_i(n.attr("group"), 1),
                            st.0,
                            st.1
                        )
                    } else {
                        format!(
                            "Conv r{} x{} w{}",
                            x.shape.len(),
                            x.shape.first().unwrap_or(&0),
                            w.shape.len()
                        )
                    }
                } else if prof {
                    // 池化/转置这类「形状即行为」的算子，属性不打印就分不清
                    // 走的是哪条分支（MaxPool 的 2x2s1 快路径与 3x3 通用路径
                    // 差 10 倍以上）。
                    match op {
                        "MaxPool" | "AveragePool" | "GlobalAveragePool" => {
                            let ks = n
                                .attr("kernel_shape")
                                .map(|a| a.ints.clone())
                                .unwrap_or_default();
                            let st = n
                                .attr("strides")
                                .map(|a| a.ints.clone())
                                .unwrap_or_default();
                            let pd = n.attr("pads").map(|a| a.ints.clone()).unwrap_or_default();
                            format!("{op} k{ks:?} s{st:?} p{pd:?}")
                        }
                        _ => op.to_string(),
                    }
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
                        // 残差融合（fuse_conv_residual）：inputs[3] 是运行期
                        // 残差张量，在激活之后逐元素加上。图优化保证了
                        // 有它就必有 inputs[2]（bias），无歧义。
                        let residual = if n.inputs.len() > 3 {
                            Some(get!(&arena, &n.inputs[3]).ok_or_else(|| {
                                Error::Graph(format!("{op}: missing residual {}", n.inputs[3]))
                            })?)
                        } else {
                            None
                        };
                        // NHWC 岛分支（rec_nhwc_island 标记）：张量按
                        // channels-last 物理序存放，稠密 1×1 的权重已被装载
                        // 期预打包成 2D [K,M]——**必须在 NCHW 前奏之前分流**，
                        // 前奏的 resolve_pads/x4 索引假定 4D NCHW。
                        let nhwc = n
                            .attr("qpn_nhwc")
                            .filter(|a| a.has_i)
                            .map(|a| a.i == 1)
                            .unwrap_or(false);
                        if nhwc {
                            let (sh, sw) = strides_of(n);
                            // 融合激活挂在 Conv 节点上（graph_opt）。
                            let mut act = qppocr_kernels::activation::Activation::default();
                            if let Some(aa) = n.attr("act").filter(|a| a.has_i) {
                                match aa.i {
                                    1 => {
                                        act = qppocr_kernels::activation::Activation::gelu(
                                            get_f(n.attr("act_c1"), 1.414_213_5),
                                            get_f(n.attr("act_c2"), 1.0),
                                            get_f(n.attr("act_c3"), 0.5),
                                        )
                                    }
                                    2 => act = qppocr_kernels::activation::Activation::relu(),
                                    _ => {}
                                }
                            }
                            let mut y = F32Buf::new();
                            if n.attr("qpn_dw").is_some() {
                                // depthwise：权重已被装载期预打包成 tap-major
                                // [kh*kw, c]（两种预打包都是 2D，先按标记分流）；
                                // 核形状从节点属性取（w 已是 2D）。
                                let (dh, dw) = dilations_of(n);
                                let group = get_i(n.attr("group"), 1) as usize;
                                let ks = n
                                    .attr("kernel_shape")
                                    .map(|a| a.ints.clone())
                                    .unwrap_or_default();
                                let kh_ = ks.first().copied().unwrap_or(3).max(0) as usize;
                                let kw_ = ks.get(1).copied().unwrap_or(kh_ as i64).max(0) as usize;
                                let (ph_, pw_, peh_, pew_) = resolve_pads(
                                    n, x.shape[1], x.shape[2], kh_ as i64, kw_ as i64, sh, sw, dh,
                                    dw,
                                );
                                let x4 = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
                                let p = qppocr_kernels::conv::ConvParams {
                                    sh: sh as usize,
                                    sw: sw as usize,
                                    ph: ph_ as usize,
                                    pw: pw_ as usize,
                                    peh: peh_ as usize,
                                    pew: pew_ as usize,
                                    dh: dh as usize,
                                    dw: dw as usize,
                                    group,
                                };
                                let os = qppocr_kernels::conv::depthwise_nhwc_packed(
                                    &x.f32,
                                    &x4,
                                    &w.f32,
                                    kh_,
                                    kw_,
                                    &p,
                                    bias,
                                    &act,
                                    residual.map(|t| t.f32.as_slice()),
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
                            } else if w.shape.len() == 2 {
                                // 稠密 1×1：x=[n,h,w,k]、w=[k,m]。
                                if sh != 1 || sw != 1 {
                                    return Err(Error::Graph("NHWC 1x1: stride must be 1".into()));
                                }
                                let (h, wd, k) = (
                                    x.shape[1] as usize,
                                    x.shape[2] as usize,
                                    x.shape[3] as usize,
                                );
                                let m = w.shape[1] as usize;
                                let r = x.shape[0] as usize * h * wd;
                                qppocr_kernels::conv::conv1x1_nhwc(
                                    &x.f32,
                                    &w.f32,
                                    r,
                                    k,
                                    m,
                                    bias,
                                    &act,
                                    residual.map(|t| t.f32.as_slice()),
                                    &mut y,
                                );
                                arena.insert(
                                    n.outputs[0].clone(),
                                    Tensor {
                                        name: n.outputs[0].clone(),
                                        shape: vec![x.shape[0], h as i64, wd as i64, m as i64],
                                        dtype: DType::F32,
                                        f32: y,
                                        i64: Vec::new(),
                                    },
                                );
                            } else {
                                return Err(Error::Graph(
                                    "NHWC 1x1: weight not prepacked (missing qpn_dw and 2D shape)"
                                        .into(),
                                ));
                            }
                        } else {
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
                                match aa.i {
                                    1 => {
                                        act = qppocr_kernels::activation::Activation::gelu(
                                            get_f(n.attr("act_c1"), 1.414_213_5),
                                            get_f(n.attr("act_c2"), 1.0),
                                            get_f(n.attr("act_c3"), 0.5),
                                        )
                                    }
                                    2 => act = qppocr_kernels::activation::Activation::relu(),
                                    _ => {}
                                }
                            }
                            let mut y = F32Buf::new();
                            let x4 = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
                            let w4 = [w.shape[0], w.shape[1], w.shape[2], w.shape[3]];
                            let cp = qppocr_kernels::conv::ConvParams {
                                sh: sh as usize,
                                sw: sw as usize,
                                ph: ph as usize,
                                pw: pw as usize,
                                peh: peh as usize,
                                pew: pew as usize,
                                dh: dh as usize,
                                dw: dw as usize,
                                group,
                            };
                            let os_pre = qppocr_kernels::conv::conv2d_out_shape(&x4, &w4, &cp);
                            let need_pre = os_pre.iter().product::<i64>().max(0) as usize;
                            let hw_pre = (os_pre[2] * os_pre[3]) as usize;
                            let res = residual.map(|t| t.f32.as_slice());
                            let mut held = alias.take(
                                n.outputs[0].as_str(),
                                x4[0] as usize,
                                w4[0] as usize,
                                hw_pre,
                            );
                            // 别名路径：内核直接写共享缓冲的槽位切片；own 路径走
                            // 安全分配版 conv2d_res（uninit 收在 kernels 里）。
                            let os = match &mut held {
                                Some((_, b, off)) => qppocr_kernels::conv::conv2d_res_into(
                                    &x.f32,
                                    &x4,
                                    &w.f32,
                                    &w4,
                                    bias,
                                    &cp,
                                    &act,
                                    os_pre,
                                    &mut b[*off..*off + need_pre],
                                    res,
                                ),
                                None => qppocr_kernels::conv::conv2d_res(
                                    &x.f32, &x4, &w.f32, &w4, bias, &cp, &act, &mut y, res,
                                ),
                            };
                            let was_alias = alias.finish(held);
                            arena.insert(
                                n.outputs[0].clone(),
                                if was_alias {
                                    alias_marker(n.outputs[0].clone(), os.to_vec())
                                } else {
                                    Tensor {
                                        name: n.outputs[0].clone(),
                                        shape: os.to_vec(),
                                        dtype: DType::F32,
                                        f32: y,
                                        i64: Vec::new(),
                                    }
                                },
                            );
                        }
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
                        if n.inputs.len() > 1
                            && !n.inputs[1].is_empty()
                            && let Some(t) = get!(&arena, &n.inputs[1])
                            && t.numel() > 0
                            && !t.f32.is_empty()
                        {
                            lo = t.f32[0];
                        }
                        if n.inputs.len() > 2
                            && !n.inputs[2].is_empty()
                            && let Some(t) = get!(&arena, &n.inputs[2])
                            && t.numel() > 0
                            && !t.f32.is_empty()
                        {
                            hi = t.f32[0];
                        }
                        let mut y = take0(&mut arena)?;
                        qppocr_kernels::activation::clip_inplace(&mut y.f32, lo, hi);
                        arena.insert(n.outputs[0].clone(), y);
                    }
                    "MulAddScale" => {
                        // 图优化把 `Add(Mul(特征, 门), 残差)` 坍缩成这一个节点。
                        // 门是 [1,C,1,1] 时走逐通道标量的快路径（读三写一，
                        // 省掉中间那趟「写 12 MB + 读 12 MB」）；形状不匹配就
                        // 退回「先 Mul 再 Add」——与坍缩前逐位相同。
                        let f = get!(&arena, &n.inputs[0]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                        })?;
                        let gm = get!(&arena, &n.inputs[1]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[1]))
                        })?;
                        let rs = get!(&arena, &n.inputs[2]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[2]))
                        })?;
                        let shape = f.shape.clone();
                        let fast = f.shape.len() == 4
                            && gm.f32.len() == f.shape[1] as usize
                            && rs.shape == f.shape
                            && !f.f32.is_empty();
                        // 别名只挂 fast 路径（逐通道标量、输出与特征同形）；
                        // fallback 走通用二元路径，Concat 装配时照常拷入。
                        let mut held = if fast {
                            let c = f.shape[1] as usize;
                            let hw = f.f32.len() / c;
                            alias.take(n.outputs[0].as_str(), f.shape[0] as usize, c, hw)
                        } else {
                            None
                        };
                        let mut own = F32Buf::new();
                        let was_alias;
                        if let Some((_, b, off)) = &mut held {
                            let (c, hw) = (f.shape[1] as usize, f.f32.len() / f.shape[1] as usize);
                            let s = &mut b[*off..*off + c * hw];
                            qppocr_kernels::elementwise::mul_add_scale(
                                &f.f32, &gm.f32, &rs.f32, s, c, hw,
                            );
                            was_alias = true;
                        } else if fast {
                            let (c, hw) = (f.shape[1] as usize, f.f32.len() / f.shape[1] as usize);
                            own = qppocr_kernels::elementwise::mul_add_scale_alloc(
                                &f.f32, &gm.f32, &rs.f32, c, hw,
                            );
                            was_alias = false;
                        } else {
                            let (mut t, shp) = qppocr_kernels::elementwise::binary_op(
                                &f.f32,
                                &f.shape,
                                &gm.f32,
                                &gm.shape,
                                qppocr_kernels::elementwise::BinOp::Mul,
                            );
                            qppocr_kernels::elementwise::binary_op_inplace(
                                &mut t,
                                &shp,
                                &rs.f32,
                                &rs.shape,
                                qppocr_kernels::elementwise::BinOp::Add,
                            );
                            own = t;
                            was_alias = false;
                        }
                        alias.finish(held);
                        arena.insert(
                            n.outputs[0].clone(),
                            if was_alias {
                                alias_marker(n.outputs[0].clone(), shape)
                            } else {
                                Tensor {
                                    name: n.outputs[0].clone(),
                                    shape,
                                    dtype: DType::F32,
                                    f32: own,
                                    i64: Vec::new(),
                                }
                            },
                        );
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
                        let (oh, ow) = qppocr_kernels::pool2d::pool2d_out_hw(
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
                        );
                        let mut out = F32Buf::new();
                        let need = nn * c * oh * ow;
                        let mut held = alias.take(n.outputs[0].as_str(), nn, c, oh * ow);
                        // 别名路径：池化内核直接写共享缓冲的槽位切片；own
                        // 路径走安全分配版 pool2d（uninit 收在 kernels 里）。
                        match &mut held {
                            Some((_, b, off)) => qppocr_kernels::pool2d::pool2d_into(
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
                                oh,
                                ow,
                                &mut b[*off..*off + need],
                            ),
                            None => {
                                qppocr_kernels::pool2d::pool2d(
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
                            }
                        }
                        let was_alias = alias.finish(held);
                        arena.insert(
                            n.outputs[0].clone(),
                            if was_alias {
                                alias_marker(
                                    n.outputs[0].clone(),
                                    vec![nn as i64, c as i64, oh as i64, ow as i64],
                                )
                            } else {
                                Tensor {
                                    name: n.outputs[0].clone(),
                                    shape: vec![nn as i64, c as i64, oh as i64, ow as i64],
                                    dtype: DType::F32,
                                    f32: out,
                                    i64: Vec::new(),
                                }
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
                            // 按**扁平数组末两位**取 H/W（等价于 i64 的 size()-2），
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
                        let need = nn * c * oh * ow;
                        let mut held = alias.take(n.outputs[0].as_str(), nn, c, oh * ow);
                        // 别名路径：缩放内核直接写共享缓冲的槽位切片；own
                        // 路径走安全分配版（uninit 收在 kernels 里）。
                        match &mut held {
                            Some((_, b, off)) => match mode.as_str() {
                                "nearest" => qppocr_kernels::resize::resize_nearest_into(
                                    &x.f32,
                                    nn,
                                    c,
                                    h,
                                    w,
                                    oh,
                                    ow,
                                    &mut b[*off..*off + need],
                                ),
                                "linear" => qppocr_kernels::resize::resize_bilinear_into(
                                    &x.f32,
                                    nn,
                                    c,
                                    h,
                                    w,
                                    oh,
                                    ow,
                                    ctm == "align_corners",
                                    &mut b[*off..*off + need],
                                ),
                                m => {
                                    return Err(Error::Graph(format!(
                                        "Resize mode {m} unsupported"
                                    )));
                                }
                            },
                            None => match mode.as_str() {
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
                                m => {
                                    return Err(Error::Graph(format!(
                                        "Resize mode {m} unsupported"
                                    )));
                                }
                            },
                        }
                        let was_alias = alias.finish(held);
                        arena.insert(
                            n.outputs[0].clone(),
                            if was_alias {
                                alias_marker(n.outputs[0].clone(), shape)
                            } else {
                                Tensor {
                                    name: n.outputs[0].clone(),
                                    shape,
                                    dtype: DType::F32,
                                    f32: out,
                                    i64: Vec::new(),
                                }
                            },
                        );
                    }
                    "Concat" => {
                        let axis = get_i(n.attr("axis"), 0);
                        // 别名装配：axis==1 且计划命中（全部输入通道数已静态
                        // 推断）。运行期校验形状与 dtype；任何不符都置
                        // assembled=false，走下面的通用拷贝路径——正确性
                        // 不依赖别名成立。
                        let mut assembled = false;
                        let alias_ent = if axis == 1 {
                            alias.slots.get(n.outputs[0].as_str())
                        } else {
                            None
                        };
                        if let Some((slots, total_c)) = alias_ent {
                            let mut dims: Option<(i64, i64, i64)> = None; // (n, h, w)
                            let mut valid = n.inputs.len() == slots.len();
                            for (inn, s) in n.inputs.iter().zip(slots.iter()) {
                                let Some(t) = get!(&arena, inn) else {
                                    valid = false;
                                    break;
                                };
                                if t.dtype != DType::F32 || t.shape.len() != 4 || t.shape[1] != s.c
                                {
                                    valid = false;
                                    break;
                                }
                                let d = (t.shape[0], t.shape[2], t.shape[3]);
                                match dims {
                                    None => dims = Some(d),
                                    Some(prev) if prev == d => {}
                                    Some(_) => {
                                        valid = false;
                                        break;
                                    }
                                }
                            }
                            if valid {
                                let (nb, h0, w0) = dims.unwrap();
                                let hwd = (h0 * w0) as usize;
                                // 兜底缓冲按**全批**分配：nb>1 时生产者全部
                                // 退回自有缓冲（take 只认 n==1），装配要把
                                // 所有批平面拷进来。漏乘 nb 会让 b≥1 的写址
                                // 越界 panic；上轮遗留的 nb=1 缓冲则被下面
                                // 的长度守卫拒掉、走通用路径。
                                let total = nb as usize * *total_c as usize * hwd;
                                let buf =
                                    alias.bufs.remove(n.outputs[0].as_str()).unwrap_or_else(|| {
                                        // 合约见 kernels::buf：装配循环保证进
                                        // arena 前每个槽位都被写满（别名在位
                                        // 或按平面拷入），此前无读。
                                        F32Buf::with_uninit_for_concat_alias(total)
                                    });
                                if buf.len() == total {
                                    let mut buf = buf;
                                    for (inn, s) in n.inputs.iter().zip(slots.iter()) {
                                        let t = get!(&arena, inn).unwrap();
                                        if t.f32.is_empty() {
                                            continue; // 别名槽：生产者已写到位
                                        }
                                        concat_copy_planes(
                                            &t.f32,
                                            &mut buf,
                                            nb as usize,
                                            s.c as usize,
                                            s.prefix_c as usize,
                                            *total_c as usize,
                                            hwd,
                                        );
                                    }
                                    arena.insert(
                                        n.outputs[0].clone(),
                                        Tensor {
                                            name: n.outputs[0].clone(),
                                            shape: vec![nb, *total_c, h0, w0],
                                            dtype: DType::F32,
                                            f32: buf,
                                            i64: Vec::new(),
                                        },
                                    );
                                    assembled = true;
                                }
                            }
                        }
                        // 通用路径（axis≠1 / 计划未命中 / 运行期形状不符）。
                        if !assembled {
                            use qppocr_kernels::shape::PayloadRef;
                            // 标记物化：装配校验失败退回这里时，别名标记
                            // （空 f32）的真实数据还在共享缓冲的槽位里——
                            // 按标记自身形状拷出（标记的通道数已由 take 与
                            // 计划核对过），绝不让空 f32 流进 concat_any。
                            let mut owned: Vec<Option<Tensor>> = Vec::with_capacity(n.inputs.len());
                            for inn in &n.inputs {
                                let t = get!(&arena, inn).ok_or_else(|| {
                                    Error::Graph(format!("{op}: missing input {inn}"))
                                })?;
                                if t.dtype == DType::F32 && t.numel() > 0 && t.f32.is_empty() {
                                    let s = alias.dump_slice(inn, &t.shape).ok_or_else(|| {
                                        Error::Graph(format!(
                                            "Concat: alias marker {inn} without buffer"
                                        ))
                                    })?;
                                    let mut b = F32Buf::with_zeroed(s.len());
                                    b.as_mut_slice().copy_from_slice(s);
                                    owned.push(Some(Tensor {
                                        name: inn.clone(),
                                        shape: t.shape.clone(),
                                        dtype: DType::F32,
                                        f32: b,
                                        i64: Vec::new(),
                                    }));
                                } else {
                                    owned.push(None);
                                }
                            }
                            let xs: Vec<&Tensor> = owned
                                .iter()
                                .zip(n.inputs.iter())
                                .map(|(o, inn)| match o {
                                    Some(t) => Ok(t),
                                    None => get!(&arena, inn).ok_or_else(|| {
                                        Error::Graph(format!("{op}: missing input {inn}"))
                                    }),
                                })
                                .collect::<Result<Vec<&Tensor>>>()?;
                            let xs_shape: Vec<&[i64]> =
                                xs.iter().map(|t| t.shape.as_slice()).collect();
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
                            let shape = qppocr_kernels::shape::concat_any(
                                &payloads, &xs_shape, axis, &mut out,
                            );
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
                        let mut axes = axes_from(n, &arena, &self.initializers);
                        let x = get!(&arena, &n.inputs[0]).ok_or_else(|| {
                            Error::Graph(format!("{op}: missing input {}", n.inputs[0]))
                        })?;
                        // NHWC 岛：图里的 axes 是逻辑 NCHW 下标，物理是
                        // channels-last。逻辑→物理是入口转置 perm [0,2,3,1]
                        // 的**逆**映射：逻辑 2=H → 物理 1，逻辑 3=W → 物理 2。
                        if n.attr("qpn_nhwc").is_some() {
                            let rank = x.rank() as i64;
                            // 逆 perm：INV[l] = p 使 perm[p] == l。
                            const INV: [usize; 4] = [0, 3, 1, 2];
                            for a in axes.iter_mut() {
                                let l = if *a < 0 { *a + rank } else { *a };
                                if !(0..4).contains(&l) {
                                    return Err(Error::Graph(format!(
                                        "ReduceMean: NHWC axis {a} out of range"
                                    )));
                                }
                                *a = INV[l as usize] as i64;
                            }
                        }
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
                    // 非 Conv 也带上输出形状：逐元素/搬运算子的成本由**形状**
                    // 决定（同一个 Mul 在 1.97M 元素上是 0.7 ms、在 24K 元素上
                    // 是 3 us），只有算子名聚不出有意义的账。Conv 的键里已有
                    // @HxW，不重复。
                    let key = if op == "Conv" {
                        prof_key.clone()
                    } else {
                        match arena.get(&n.outputs[0]) {
                            Some(t) => format!(
                                "{prof_key} [{}]",
                                t.shape
                                    .iter()
                                    .map(|d| d.to_string())
                                    .collect::<Vec<_>>()
                                    .join(",")
                            ),
                            None => prof_key.clone(),
                        }
                    };
                    let key2 = key.clone();
                    let e = prof_acc.entry(key).or_insert((0.0, 0));
                    e.0 += t0.elapsed().as_secs_f64() * 1000.0;
                    e.1 += 1;
                    let mut this = 0u64;
                    if let Some(t) = arena.get(&n.outputs[0]) {
                        let esz = if t.dtype == DType::I64 { 8 } else { 4 };
                        this += (t.f32.len() + t.i64.len()) as u64 * esz;
                        for inn in &n.inputs {
                            if let Some(it) = get!(&arena, inn) {
                                let iesz = if it.dtype == DType::I64 { 8 } else { 4 };
                                this += (it.f32.len() + it.i64.len()) as u64 * iesz;
                            }
                        }
                    }
                    prof_bytes += this;
                    *prof_b.entry(key2).or_insert(0) += this;
                }
                #[allow(unused_assignments)]
                {
                    let _ = &op_t0;
                }
                // 对拍落盘（ QPPOCR_DUMP_DIR 同格式：%06d.f32 + manifest.tsv）
                if let Some(dir) = &dump_dir
                    && let Some(t) = arena.get(&n.outputs[0])
                    && t.dtype == DType::F32
                {
                    let (ddir, sid) = dir;
                    // 别名标记（空 f32）的数据在共享缓冲槽位里，
                    // 取真实切片落盘，保证 dump 口径不变。
                    let payload: &[f32] = if t.f32.is_empty() {
                        alias.dump_slice(&n.outputs[0], &t.shape).unwrap_or(&[])
                    } else {
                        &t.f32
                    };
                    let path = format!("{}/s{}_{:06}.f32", ddir, sid, i);
                    if let Ok(file) = std::fs::File::create(&path) {
                        let mut f = std::io::BufWriter::new(file);
                        let _ = f.write_all(&(t.rank() as i32).to_le_bytes());
                        for d in &t.shape {
                            let _ = f.write_all(&d.to_le_bytes());
                        }
                        let _ = write_f32_le(&mut f, payload);
                    }
                    let mf = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(format!("{}/manifest.tsv", ddir));
                    if let Ok(mut mf) = mf {
                        let _ = writeln!(mf, "{}\t{}\t{}", i, op, n.outputs[0]);
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

                remaining -= 1;

                // 本节点的输出就绪 → 递减下游的未满足入边计数
                for out in &n.outputs {
                    if let Some(cs) = consumers.get(out.as_str()) {
                        for &c in cs {
                            pending[c] -= 1;
                            if pending[c] == 0 {
                                ready.push(std::cmp::Reverse(c));
                            }
                        }
                    }
                }
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
                    .filter(|inn| {
                        !inn.is_empty()
                            && !arena.contains_key(*inn)
                            && !self.initializers.contains_key(inn.as_str())
                    })
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
            // ★ 搬运量：每个节点「读全部输入 + 写输出」的字节数之和。
            //   这是判断「算子慢」还是「内存墙」的尺子——有效带宽 = 搬运量
            //   / 墙钟。若它贴着实测带宽上限，那再怎么调内核也没用。
            let im2col =
                qppocr_kernels::gemm::IM2COL_BYTES.swap(0, std::sync::atomic::Ordering::Relaxed);
            let dwpad =
                qppocr_kernels::gemm::DW_PAD_BYTES.swap(0, std::sync::atomic::Ordering::Relaxed);
            let internal = (im2col + dwpad) as f64 / 1e6;
            eprintln!(
                "--- 搬运 {:.1} MB（节点级）+ {:.1} MB（im2col {:.1} / 补零 {:.1}）= {:.1} MB，有效带宽 {:.1} GB/s",
                prof_bytes as f64 / 1e6,
                internal,
                im2col as f64 / 1e6,
                dwpad as f64 / 1e6,
                (prof_bytes as f64 / 1e6) + internal,
                ((prof_bytes + im2col + dwpad) as f64) / wall.max(0.001) / 1e6
            );
            let mut v: Vec<_> = prof_acc.into_iter().collect();
            v.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
            eprintln!(
                "--- per-op profile [{}] (kernel {kernel:.1} ms, wall {wall:.1} ms, executor {:.1} ms = {:.1}%) ---",
                g.model_path,
                wall - kernel,
                100.0 * (wall - kernel) / wall.max(0.001)
            );
            for (op, (ms, cnt)) in v {
                let mb = prof_b.get(&op).copied().unwrap_or(0) as f64 / 1e6;
                eprintln!("  {op:<16} {ms:8.1} ms  {mb:7.1} MB  x{cnt}");
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

// ─────────────────────────────────────────────────────────────────────
// Concat 零拷贝别名（P2）
//
// C 轴拼接的输出布局 = 各输入的通道平面依次落座。与其「每个生产者写
// 自己的缓冲、Concat 再整块读+写一遍」，不如让生产者**直接写进共享缓冲
// 的槽位**、Concat 退化成装配：别名槽零搬运，未别名槽（常量输入、批次
// 退化）按平面拷入。数值不变：同一批内核、同一输入、同一累加序，只有
// 最终 store 的落址不同——逐位可验证（QPPOCR_DUMP_DIR 对拍）。
//
// 别名的唯一根：真正分配输出缓冲的节点。计划期按执行图（Identity 已
// 消解、激活已折进 Conv）静态判定：生产者是 Conv/池化/Resize/MulAddScale
// 之一、输出只被本 Concat 消费、且全部输入的通道数可静态推断。
// ─────────────────────────────────────────────────────────────────────

/// 单个 Concat 输入的槽位信息（与节点输入顺序对齐）。
#[derive(Clone)]
struct ConcatAliasSlot {
    /// 输入名（= 生产者输出名；常量输入则为 initializer 名）。
    input: String,
    /// 该输入的通道数（计划期推断）。
    c: i64,
    /// 它之前所有输入的通道之和（输出里的起始通道）。
    prefix_c: i64,
    /// 生产者是否直接写进共享缓冲。
    aliased: bool,
}

/// 每次推理独立的别名运行时：计划（roots/slots）+ 共享缓冲（bufs）。
struct ConcatAliasCtx {
    /// 生产者输出名 -> (Concat 输出名, 通道前缀, 本槽静态通道数, 总通道数)。
    roots: HashMap<String, (String, i64, i64, i64)>,
    /// Concat 输出名 -> (槽位表, 总通道数)。
    slots: HashMap<String, (Vec<ConcatAliasSlot>, i64)>,
    /// 共享缓冲：首生产者创建，Concat 节点收走插进 arena。
    bufs: HashMap<String, F32Buf>,
}

/// 静态推断一个张量的通道数（C 轴拼接的偏移前提）。
///
/// 只走「通道原样传递」的算子链；碰上推断不了的（含 ConvTranspose、
/// 别名根集合外的生产者、i64/非 4D 常量）返回 None——整个 Concat 退回
/// 原拷贝路径。推断只是**提示**：装配期逐输入校验 `t.shape[1] == c`，
/// 推错只会导致兜底，不会算错。
///
/// 权重查 `weights`（装载时 `std::mem::take` 之后 `g.initializers` 恒空，
/// 会话权重全在 `self.initializers`）。
fn infer_concat_channels(
    g: &crate::onnx::model::Graph,
    weights: &HashMap<String, Tensor>,
    producer_of: &HashMap<&str, usize>,
    name: &str,
    depth: usize,
) -> Option<i64> {
    if depth > 16 {
        return None;
    }
    if let Some(t) = weights.get(name) {
        if t.dtype == DType::F32 && t.shape.len() == 4 {
            return Some(t.shape[1]);
        }
        return None;
    }
    let &pi = producer_of.get(name)?;
    match g.nodes[pi].op_type.as_str() {
        // 权重 [m, k, kh, kw]（2D 预打包的 NHWC 岛 conv 不参与——保守）。
        "Conv" => {
            let wn = g.nodes[pi].inputs.get(1)?;
            let w = weights.get(wn)?;
            if w.dtype == DType::F32 && w.shape.len() == 4 {
                Some(w.shape[0])
            } else {
                None
            }
        }
        // decoder 的上采样链常以 backbone 的拼接输出为源：总通道 = 各
        // 输入通道之和（axis≠1 时退出）。
        "Concat" => {
            if get_i(g.nodes[pi].attr("axis"), 0) != 1 {
                return None;
            }
            let mut s = 0i64;
            for inn in &g.nodes[pi].inputs {
                if inn.is_empty() {
                    return None;
                }
                s += infer_concat_channels(g, weights, producer_of, inn, depth + 1)?;
            }
            Some(s)
        }
        // 通道原样传递的池化/缩放算子（特征在 inputs[0]）。刻意**不含**
        // Add/Sub/Mul/Div：广播能把输出通道变宽（[1,1,H,W]+[1,C,1,1]），
        // 传递推断不保真——推断只是提示，但少一类不保真就少一类兜底。
        "MaxPool" | "AveragePool" | "Resize" | "MulAddScale" => infer_concat_channels(
            g,
            weights,
            producer_of,
            g.nodes[pi].inputs.first()?,
            depth + 1,
        ),
        _ => None,
    }
}

impl ConcatAliasCtx {
    /// 从执行图构建别名计划。`enabled`=false（QPPOCR_NO_CONCAT_ALIAS）
    /// 时返回空计划——所有算子走原路径。`weights` 是会话权重
    /// （self.initializers——装载后 g.initializers 恒空）。
    fn build(
        enabled: bool,
        g: &crate::onnx::model::Graph,
        weights: &HashMap<String, Tensor>,
    ) -> Self {
        let mut me = Self {
            roots: HashMap::new(),
            slots: HashMap::new(),
            bufs: HashMap::new(),
        };
        if !enabled {
            return me;
        }
        let mut producer_of: HashMap<&str, usize> = HashMap::new();
        let mut ncons: HashMap<&str, i64> = HashMap::new();
        for (i, nd) in g.nodes.iter().enumerate() {
            for o in &nd.outputs {
                producer_of.insert(o.as_str(), i);
            }
            for inn in &nd.inputs {
                if !inn.is_empty() {
                    *ncons.entry(inn.as_str()).or_insert(0) += 1;
                }
            }
        }
        for nd in g.nodes.iter().filter(|n| n.op_type == "Concat") {
            if get_i(nd.attr("axis"), 0) != 1 || nd.outputs.len() != 1 {
                continue;
            }
            let mut prefix = 0i64;
            let mut slots: Vec<ConcatAliasSlot> = Vec::with_capacity(nd.inputs.len());
            let mut ok = true;
            let mut any = false;
            for inn in &nd.inputs {
                if inn.is_empty() {
                    ok = false;
                    break;
                }
                let Some(c) = infer_concat_channels(g, weights, &producer_of, inn, 0) else {
                    ok = false;
                    break;
                };
                let aliased = match producer_of.get(inn.as_str()) {
                    Some(&pi) => {
                        let supported = matches!(
                            g.nodes[pi].op_type.as_str(),
                            "Conv" | "MaxPool" | "AveragePool" | "Resize" | "MulAddScale"
                        );
                        supported
                            && ncons.get(inn.as_str()).copied().unwrap_or(0) == 1
                            && !g.is_graph_output(inn)
                    }
                    None => false,
                };
                any |= aliased;
                slots.push(ConcatAliasSlot {
                    input: inn.clone(),
                    c,
                    prefix_c: prefix,
                    aliased,
                });
                prefix += c;
            }
            if !ok || !any {
                continue;
            }
            let co = nd.outputs[0].clone();
            for s in &slots {
                if s.aliased {
                    me.roots
                        .insert(s.input.clone(), (co.clone(), s.prefix_c, s.c, prefix));
                }
            }
            me.slots.insert(co, (slots, prefix));
        }
        me
    }

    /// 生产者侧入口：本输出若是别名根，取出共享缓冲（没有则按总通道数
    /// 创建），返回 (Concat 输出名, 缓冲, 元素偏移)。
    ///
    /// 返回 `None` = 不是别名根，或运行期形状对不上（批次 n≠1、与共享
    /// 缓冲长度不符——都意味着这张图不是计划假设的形态）——调用方退回
    /// 自有缓冲路径，Concat 装配时会看到非空张量并按平面拷入。正确性
    /// 永远不依赖这里成功。
    fn take(
        &mut self,
        out_name: &str,
        n: usize,
        c: usize,
        hw: usize,
    ) -> Option<(String, F32Buf, usize)> {
        if n != 1 {
            return None;
        }
        let (co, prefix_c, c_static, total_c) = self.roots.get(out_name)?.clone();
        // 运行期通道数必须与计划一致：不一致说明静态推断猜错了（这张图
        // 不是计划假设的形态），本生产者退回自有缓冲——标记不落 arena，
        // 装配按普通张量把它拷入。其余槽位不受影响（各自独立校验）。
        if c as i64 != c_static {
            return None;
        }
        let total = total_c as usize * hw;
        let off = prefix_c as usize * hw;
        // SAFETY: 别名契约——共享缓冲在进 arena（被读）之前，每个槽位都
        // 会被写满：别名生产者的内核全量写出自己的槽位，其余槽位由
        // Concat 装配按平面拷入。没有先读。
        let buf = self
            .bufs
            .remove(&co)
            .unwrap_or_else(|| F32Buf::with_uninit_for_concat_alias(total));
        if buf.len() != total || off + c * hw > total {
            self.bufs.insert(co.clone(), buf);
            return None;
        }
        Some((co, buf, off))
    }

    /// 生产者侧收尾：把共享缓冲放回，返回别名是否成功。
    fn finish(&mut self, held: Option<(String, F32Buf, usize)>) -> bool {
        match held {
            Some((co, b, _)) => {
                self.bufs.insert(co, b);
                true
            }
            None => false,
        }
    }

    /// 别名标记（空 f32）的真实数据切片：按标记自身形状从共享缓冲的
    /// 槽位取。dump 落盘与通用拼接路径的标记物化共用。
    fn dump_slice(&self, out_name: &str, shape: &[i64]) -> Option<&[f32]> {
        if shape.len() != 4 {
            return None;
        }
        let (co, prefix_c, c_static, _total) = self.roots.get(out_name)?;
        if shape[1] != *c_static {
            return None; // 推断失真：标记本不该存在（take 已挡），保底退出
        }
        let hw = (shape[2] * shape[3]) as usize;
        let off = *prefix_c as usize * hw;
        let need = shape[1] as usize * hw;
        let b = self.bufs.get(co)?;
        b.get(off..off + need)
    }
}

/// 别名标记张量：形状齐全、数据在共享缓冲里（f32 为空即「数据在别处」
/// 信号，Concat 装配与 dump 都认这个约定）。
fn alias_marker(name: String, shape: Vec<i64>) -> Tensor {
    Tensor {
        name,
        shape,
        dtype: DType::F32,
        f32: F32Buf::new(),
        i64: Vec::new(),
    }
}

/// 把一个输入的通道平面拷进共享缓冲的槽位（未别名/退化输入用）。
/// n==1 时就是一段连续拷贝；n>1 按批次平面散拷。
fn concat_copy_planes(
    src: &[f32],
    dst: &mut [f32],
    nb: usize,
    ci: usize,
    prefix_c: usize,
    total_c: usize,
    hw: usize,
) {
    for b in 0..nb {
        let s = &src[b * ci * hw..(b + 1) * ci * hw];
        let d = &mut dst[b * total_c * hw + prefix_c * hw..][..ci * hw];
        d.copy_from_slice(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 合成「Conv → MaxPool(1×1 别名根) / Conv → Concat(axis=1)」图：
    /// 语义上 y[.., 0:5] = x[..,0] + (10+c)、y[.., 5:12] = 2·x[..,1] + (20+c')。
    /// 权重每通道只挑一个输入通道（其余为 0），保证求和次序无关、逐位确定。
    fn synth_graph() -> (Graph, HashMap<String, Tensor>) {
        let mk_w = |co: usize, pick: usize, k: f32| {
            let mut v = vec![0f32; co * 3];
            for c in 0..co {
                v[c * 3 + pick] = k;
            }
            Tensor {
                name: String::new(),
                shape: vec![co as i64, 3, 1, 1],
                dtype: DType::F32,
                f32: F32Buf::from_vec(&v),
                i64: Vec::new(),
            }
        };
        let mk_b = |base: f32, co: usize| {
            let v: Vec<f32> = (0..co).map(|c| base + c as f32).collect();
            Tensor {
                name: String::new(),
                shape: vec![co as i64],
                dtype: DType::F32,
                f32: F32Buf::from_vec(&v),
                i64: Vec::new(),
            }
        };
        let conv = |name: &str, w: &str, b: &str, inp: &str, out: &str| Node {
            op_type: "Conv".into(),
            name: name.into(),
            inputs: vec![inp.into(), w.into(), b.into()],
            outputs: vec![out.into()],
            attrs: Vec::new(),
        };
        // MaxPool 1×1/stride 1 = 恒等；它是 Concat 别名计划认定的生产者。
        let pool = Node {
            op_type: "MaxPool".into(),
            name: "p1".into(),
            inputs: vec!["h1".into()],
            outputs: vec!["h1p".into()],
            attrs: vec![Attribute {
                name: "kernel_shape".into(),
                ints: vec![1, 1],
                ..Default::default()
            }],
        };
        let concat = Node {
            op_type: "Concat".into(),
            name: "cat".into(),
            inputs: vec!["h1p".into(), "h2".into()],
            outputs: vec!["y".into()],
            attrs: vec![Attribute {
                name: "axis".into(),
                i: 1,
                has_i: true,
                ..Default::default()
            }],
        };
        let g = Graph {
            nodes: vec![
                conv("c1", "w1", "b1", "x", "h1"),
                pool,
                conv("c2", "w2", "b2", "x", "h2"),
                concat,
            ],
            initializers: HashMap::new(),
            inputs: vec!["x".into()],
            input_dims: vec![vec![0, 3, 0, 0]],
            outputs: vec!["y".into()],
            ..Default::default()
        };
        let mut inits = HashMap::new();
        inits.insert("w1".to_string(), mk_w(5, 0, 1.0));
        inits.insert("b1".to_string(), mk_b(10.0, 5));
        inits.insert("w2".to_string(), mk_w(7, 1, 2.0));
        inits.insert("b2".to_string(), mk_b(20.0, 7));
        (g, inits)
    }

    /// 回归守卫：Concat 别名装配的**全批**语义。
    ///
    /// nb=1 走别名槽位（MaxPool 生产者直写共享缓冲）+ 未别名平面拷入的
    /// 混合路径；nb>1 时生产者全部退回自有缓冲（`take` 只认 n==1），
    /// 装配要靠兜底缓冲把**所有批**的平面拷进来——该缓冲曾漏乘批数，
    /// b≥1 的写址直接越界 panic（公开 API `Session::run` 传 nb>1 即触发）。
    #[test]
    fn concat_alias_assembly_all_batches() {
        let (h, w) = (4usize, 6usize);
        let hw = h * w;
        for nb in [1usize, 8] {
            let (g, inits) = synth_graph();
            let s = Session::from_parts(g, inits);
            let mut xv = vec![0f32; nb * 3 * hw];
            for (i, v) in xv.iter_mut().enumerate() {
                *v = ((i * 37 + 13) % 97) as f32 / 97.0 - 0.5;
            }
            let x = Tensor {
                name: "x".into(),
                shape: vec![nb as i64, 3, h as i64, w as i64],
                dtype: DType::F32,
                f32: F32Buf::from_vec(&xv),
                i64: Vec::new(),
            };
            let out = s.run(vec![("x".to_string(), x)]).expect("run");
            assert_eq!(out.len(), 1);
            let y = &out[0];
            assert_eq!(y.shape, vec![nb as i64, 12, h as i64, w as i64]);
            let yv = y.f32.as_slice();
            for b in 0..nb {
                for c in 0..12usize {
                    for i in 0..hw {
                        let got = yv[(b * 12 + c) * hw + i];
                        let x0 = xv[b * 3 * hw + i];
                        let x1 = xv[b * 3 * hw + hw + i];
                        let want = if c < 5 {
                            x0 + (10.0 + c as f32)
                        } else {
                            2.0 * x1 + (20.0 + (c - 5) as f32)
                        };
                        assert_eq!(got, want, "nb={nb} b={b} c={c} i={i}");
                    }
                }
            }
        }
    }
}
