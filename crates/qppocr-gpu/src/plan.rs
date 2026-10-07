//! 整图计划模型（n_ 内核族路径）：图 + 输入形状 → dispatch 列表 +
//! arena 布局，**后端无关、不持有任何设备资源**。
//!
//! Vulkan 会话（`vulkan::session`）与 Metal 会话（`qppocr-metal`）共同
//! 消费本模块：装载期把（已优化的）图经 [`PlanBuilder::build`] 编排成
//! 一次可执行的 dispatch 序列（内核名 / 参数块偏移 / 网格），区域布局
//! 与权重上传清单一并产出；各后端只负责分配真实缓冲、写入静态数据、
//! 按 dispatch 列表编码/录制命令。
//!
//! 设计约定（与内核头注释一一对应）：
//! - 激活按 NHWC f32 存储，C 补到 %4，区域大小一律 **u32 word**计；
//! - 单一大 arena：偏移寻址，参数块与数据同块（u32 视图间接寻址，
//!   `params[0]` 是 real_n 全局槽——word 0，构造上恒为 0）；
//! - 静态内容（权重/参数块）**只增不复用**（alloc_static）——命令缓冲
//!   每次重放/重编码都会重读它；激活区按活性分析 free-复用；
//! - 本模块是历史 bug 沉淀最厚的层（区域复用踩写、门广播批维、定向
//!   搅拌、填充不变性……），修改任何分支前先读各分支内的注释。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use qppocr_core::error::{Error, Result};
use qppocr_core::onnx::model::{Graph, Node};
use qppocr_core::tensor::{DType, Tensor};

use crate::nhwc;
use crate::planner;

/// 「无此输入」的哨兵偏移（conv 的 bias/residual 等）。
pub const OFF_NONE: u32 = u32::MAX;

/// push constant 块（与各内核的 PC 布局一一对应；float 元素偏移）。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PcParams {
    /// 参数块在 arena 里的 **u32 下标**。
    pub p_off: u32,
}

impl PcParams {
    /// 裸字节视图（与 pipeline 的 pc_bytes! 同风格；Metal 侧直接
    /// set_bytes u32，不走这里）。
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: repr(C) 定长 POD 的连续字节读取，无内部指针。
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

/// 参数块写入器：形状参数太多（conv 20+ 项）装不进 128 B 的 Vulkan
/// push constant，写进 arena 一小块，内核经 u32 视图读取。装载期写
/// 一次、随命令复用——零每帧成本。
#[derive(Default)]
pub struct ParamBlock {
    words: Vec<u32>,
}

impl ParamBlock {
    /// 空参数块。
    pub fn new() -> Self {
        Self { words: Vec::new() }
    }

    /// 追加一个 u32。
    pub fn u(&mut self, v: u32) -> &mut Self {
        self.words.push(v);
        self
    }

    /// 字数（布局分配用）。
    pub fn len_words(&self) -> u32 {
        self.words.len() as u32
    }

    /// 只读视图（会话直接写进整块布局）。
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// f32 按位写入（内核侧 as_type<float> 读回）。
    pub fn f(&mut self, v: f32) -> &mut Self {
        self.words.push(v.to_bits());
        self
    }
}

/// 一条 dispatch：内核名 + 参数块偏移 + 网格（后端把它编码成命令）。
#[derive(Clone)]
pub struct Dispatch {
    /// 内核名（KernelSet 的键）。
    pub kernel: &'static str,
    /// 节点名（调试/对拍用；entry/exit 等合成步是 `<...>` 形式）。
    pub node: String,
    /// 节点序号（合成步 = usize::MAX）。
    pub node_idx: usize,
    /// 输出区域偏移（word）。
    pub out_off: u32,
    /// 输出元素数（逻辑 numel，调试/对拍用）。
    pub out_n: u32,
    /// 参数块偏移（word）。
    pub p_off: u32,
    /// dispatch 网格（工作组数；local_size = 256 由后端统一）。
    pub groups: [u32; 3],
    /// 输入明细 (名, 偏移, 元素数)——步进对拍用。
    pub ins: Vec<(String, u32, u32)>,
    /// 输入形状（剖析打印用）。
    pub in_shapes: Vec<Vec<i64>>,
    /// 输出形状。
    pub out_shape: Vec<i64>,
}

/// 一个输入形状的整图计划（纯数据：布局 + dispatch + 上传清单）。
pub struct GraphPlan {
    /// arena 总 word 数（= 字节数 / 4）。
    pub total_words: u32,
    /// real_n 单元的 word 偏移（构造上恒 0；显式存字段防布局演算漂移）。
    pub rn_word: u32,
    /// 图输入的 f32 NCHW 区偏移（host 每次 memcpy 的目标——**不是**
    /// 节点引用的 NHWC 区，见 build 内注释）。
    pub in_f32_off: u32,
    /// dispatch 序列（整图执行序）。
    pub dispatches: Vec<Dispatch>,
    /// 每次装载写一次的静态数据（rn 初值 + 参数块）。
    pub uploads: Vec<(u32, Vec<u32>)>,
    /// init 直传权重（重排后）：复用同图 arena 时整段跳过（内容与形状
    /// 无关）。
    pub upload_init: Vec<(u32, Arc<Vec<u32>>)>,
    /// 图输出：(名字, 偏移, 元素数)（按 graph.outputs 顺序；argmax 出
    /// 口的元素数 = 行数×2）。
    pub out_offs: Vec<(String, u32, usize)>,
    /// 图输出形状（argmax 出口已是 [N,T,2]）。
    pub out_shapes: Vec<Vec<i64>>,
}

/// 计划构建器：会话侧共享状态的只读视图。
///
/// `repack_cache`（权重重排缓存）与 `producer_idx`（生产者索引表）由
/// 会话持有、跨形状复用——重排算术与形状无关（~9ms/83 层 conv 是冷启
/// 动大头），回溯表图会话内不变。
pub struct PlanBuilder<'a> {
    /// 已优化的图（CPU 侧装载期优化后的形态）。
    pub graph: &'a Graph,
    /// 权重（loader 的 take 语义产物）。
    pub initializers: &'a HashMap<String, Tensor>,
    /// 会话级权重重排缓存（按名键控）。
    pub repack_cache: &'a Mutex<HashMap<String, Arc<Vec<u32>>>>,
    /// 输出名 → 节点索引（linear_by_producer 的回溯表）。
    pub producer_idx: &'a HashMap<String, usize>,
    /// rec 会话的 CTC argmax 出口（rank-3 输出改走 n_exit3_argmax）。
    pub argmax_exit: bool,
}

impl PlanBuilder<'_> {
    /// 对给定输入形状构建整图计划。
    ///
    /// 不支持的算子/形状在此报 [`Error::Graph`] 并点名——后端把它透传
    /// 给 create_session 的分级/探针逻辑，不静默降级。
    pub fn build(&self, input_name: &str, in_shape: &[i64]) -> Result<GraphPlan> {
        let t_build = std::time::Instant::now();
        let table = planner::infer_shapes(
            self.graph,
            self.initializers,
            &[(input_name.to_string(), in_shape.to_vec(), DType::F32)],
        )?;

        let dbg_env = std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some();
        let refs = consumer_counts(self.graph);
        let mut live: HashMap<String, u32> = refs.clone();
        let is_output = |nm: &str| self.graph.is_graph_output(nm);

        let mut layout = Layout::new();
        let mut offs: HashMap<String, u32> = HashMap::new();
        // 权重占区（F32 且被引用；i64 常量链由 planner 折叠，不上传）。
        // 权重**永不释放**：dispatch 每次重放都读它，区域复用=数据被覆盖。
        let mut upload: Vec<(u32, Vec<u32>)> = Vec::new();
        let mut upload_init: Vec<(u32, Arc<Vec<u32>>)> = Vec::new();
        // real_n 单元（arena 第 0 词）：本次 run 的真实批维行数，run() 每
        // 次提交前主机直写；n_ 内核读 params[0] 做批维早退。必须**最先**
        // 分配：内核按 params[0] 绝对寻址，静态区只增不复用，word 0 不能
        // 让给权重/激活。初值 1 = 全通过（等 run() 覆写）。
        let rn_word = layout.alloc_static(1);
        debug_assert_eq!(rn_word, 0);
        upload.push((rn_word, vec![1u32]));
        // conv 族权重（重排目标）名单：预环跳过，节点处理时惰性重排。
        let mut conv_w_names: std::collections::HashSet<String> = Default::default();
        for n in &self.graph.nodes {
            if n.op_type == "Conv" || n.op_type == "ConvTranspose" {
                for w in n.inputs.iter().skip(1).take(2) {
                    if !w.is_empty() {
                        conv_w_names.insert(w.clone());
                    }
                }
            }
        }
        // **确定序**（按名排序）：HashMap 迭代序逐进程随机——区域布局会
        // 跟着漂移，任何依赖布局的越界/重叠 bug 都会以「两次运行分歧层
        // 不同」的形态出现（实测发生过）。排序后布局恒定，分歧可稳定
        // 复现定位。
        let mut init_sorted: Vec<(&String, &Tensor)> = self.initializers.iter().collect();
        init_sorted.sort_by(|a, b| a.0.cmp(b.0));
        for (name, t) in init_sorted {
            let Some(v) = table.values.get(name) else {
                continue;
            };
            if v.dtype != DType::F32 || refs.get(name).copied().unwrap_or(0) == 0 {
                continue;
            }
            if conv_w_names.contains(name) {
                continue; // conv 权重/bias 惰性重排（map_node_n）
            }
            // 转换进会话级缓存（与 conv 权重重排同款）：非 conv 权重可
            // 巨大（small rec 的分类器矩阵 4.5M 元素）——曾每计划重算
            // to_bits/to_nhwc，是精确宽形状税的主部。
            let words: Arc<Vec<u32>> = std::sync::Arc::clone(
                self.repack_cache
                    .lock()
                    .map_err(|_| Error::Device("重排缓存锁中毒".into()))?
                    .entry(format!("init:{name}"))
                    .or_insert_with(|| std::sync::Arc::new(nhwc::init_to_nhwc(t))),
            );
            let off = layout.alloc_static(words.len() as u32);
            offs.insert(name.clone(), off);
            upload_init.push((off, words));
        }

        struct Rec {
            kernel: &'static str,
            node: String,
            node_idx: usize,
            out_off: u32,
            out_n: u32,
            p_off: u32,
            groups: [u32; 3],
            ins: Vec<(String, u32, u32)>,
            in_shapes: Vec<Vec<i64>>,
            out_shape: Vec<i64>,
        }
        let mut recs: Vec<Rec> = Vec::new();
        let mut params: Vec<(u32, Vec<u32>)> = Vec::new();

        // 图输入占区：f32 区每 run 重写（host memcpy）；节点一律引用
        // entry 转换出的 NHWC 区。两区都不释放。
        let in_words = in_shape.iter().product::<i64>() as u32;
        let in_f32_off = layout.alloc(in_words);
        let in_f16_off = layout.alloc(nhwc::nhwc_words(in_shape));
        offs.insert(input_name.to_string(), in_f16_off);
        // entry 转换 dispatch（整图第一条）
        {
            let (nb, c, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
            let mut pb = ParamBlock::new();
            pb.u(in_f32_off)
                .u(in_f16_off)
                .u(nb as u32)
                .u(c as u32)
                .u(h as u32)
                .u(w as u32)
                .u(nhwc::cpad4(c));
            let p_off = layout.alloc_static(pb.len_words());
            params.push((p_off, pb.words().to_vec()));
            recs.push(Rec {
                kernel: "n_entry",
                node: "<entry>".into(),
                node_idx: usize::MAX,
                out_off: in_f16_off,
                out_n: in_words,
                p_off,
                groups: [((nb * h * w) as u32).div_ceil(256), 1, 1],
                ins: vec![(input_name.to_string(), in_f32_off, in_words)],
                in_shapes: vec![in_shape.to_vec()],
                out_shape: in_shape.to_vec(),
            });
        }

        // SE 融合预扫：HardSigmoid/Sigmoid(gate[1,C,1,1]) 的唯一消费者是
        // Mul(feature, gate_out) 时，记入 skip 集（其 Mul 用 n_channel 的
        // fused op 读前激活门）。QPPOCR_GPU_SE_FUSION 显式开、
        // QPPOCR_GPU_NO_SE 关（默认关——n_channel 的非融合 op0/1 已是
        // 单 dispatch，融合是历史 A/B 项）。
        let mut fused: std::collections::HashSet<String> = Default::default();
        let se_env = std::env::var_os("QPPOCR_GPU_SE_FUSION").is_some()
            && std::env::var_os("QPPOCR_GPU_NO_SE").is_none();
        if se_env {
            for n in &self.graph.nodes {
                if n.op_type != "HardSigmoid" && n.op_type != "Sigmoid" {
                    continue;
                }
                let in_v = table.values.get(&n.inputs[0]);
                if in_v.map(|v| v.shape.len() != 4 || v.shape[2] != 1 || v.shape[3] != 1)
                    != Some(false)
                {
                    continue; // 非 [1,C,1,1] 门
                }
                // 找唯一消费者是 Mul 的
                let out_name = &n.outputs[0];
                let consumers: Vec<&Node> = self
                    .graph
                    .nodes
                    .iter()
                    .filter(|m| m.inputs.iter().any(|i| i == out_name))
                    .collect();
                if consumers.len() != 1 || consumers[0].op_type != "Mul" {
                    continue;
                }
                let mul = consumers[0];
                // Mul 的另一个输入是 4D 特征图（非门）
                let other = if &mul.inputs[0] == out_name {
                    &mul.inputs[1]
                } else {
                    &mul.inputs[0]
                };
                let other_v = table.values.get(other);
                if other_v.map(|v| v.shape.len() != 4 || v.shape[2] == 1) != Some(false) {
                    continue;
                }
                fused.insert(n.name.clone());
                eprintln!("[plan] SE 融合: {} → {} (fused)", n.name, mul.name);
            }
        }
        // conv 族权重的惰性重排区（名 → 偏移，防共享权重重排两份）
        let mut w_offs: HashMap<String, u32> = HashMap::new();
        // SE 归约的两阶段共享 scratch（f32 words，按最大块需求开一份）
        let reduce_scratch: u32 = {
            let mut need = 0u32;
            for n in &self.graph.nodes {
                if n.op_type == "GlobalAveragePool"
                    || (n.op_type == "ReduceMean"
                        && planner::axes_from(n, &table.values)
                            .map(|a| {
                                let r = shape_of(&table, &n.inputs[0]).len() as i64;
                                a.iter()
                                    .map(|&x| if x < 0 { x + r } else { x })
                                    .collect::<Vec<_>>()
                            })
                            .is_ok_and(|norm| {
                                norm.len() == 2 && norm.contains(&2) && norm.contains(&3)
                            }))
                {
                    let xs = shape_of(&table, &n.inputs[0]);
                    if xs.len() == 4 {
                        let m_blocks = ((xs[2] * xs[3]) as u32).div_ceil(1024);
                        let nb = xs[0] as u32;
                        need = need.max(nb * m_blocks * nhwc::cpad4(xs[1]));
                    }
                }
            }
            if need > 0 {
                layout.alloc_static(need)
            } else {
                0
            }
        };
        // exit 转换列表 (f16 源名, 图输出名, act)——节点环后统一发
        let mut pending_exits: Vec<(String, String, u32)> = Vec::new();
        // argmax 出口的图输出名（读回长度/形状按 [N,T,2] 覆写）
        let mut argmax_outs: std::collections::HashSet<String> = Default::default();
        // 每张量的存储形状 (rows, cols)——cols 是连续维。
        // rank-4 conv 族 = (N*H*W, Cpad)；MatMul 尾部 = (M, Npad)；
        // Squeeze/Unsqueeze/Transpose(0,2,1) 是存储恒等（别名）。
        // 形状不可从逻辑 shape 单向推导（同形 rank-3 有两种来源），
        // 必须按生产者记账。
        let mut rc: HashMap<String, (u32, u32)> = HashMap::new();
        {
            let (nr, cr) = (
                in_shape[0] * in_shape[2] * in_shape[3],
                nhwc::cpad4(in_shape[1]),
            );
            rc.insert(input_name.to_string(), (nr as u32, cr));
        }

        for (node_idx, n) in self.graph.nodes.iter().enumerate() {
            let out = n.outputs[0].clone();
            // ---- 存储恒等节点（Squeeze/Unsqueeze/Transpose(0,2,1)/
            // Slice(dim0)）：零分配零 dispatch，输出别名输入（Slice 带偏移）。
            // 输入不释放（存活期 = 别名的最后消费者——记 immortal，泄漏量
            // ~25KB/节点可忽略）。
            {
                // dim0 连续切片（注意力的 Q/K/V 拆分）：输出 = 输入的
                // start 段——纯偏移别名。starts/ends/axes/steps 是
                // initializer 常量（opset 13+）。
                if n.op_type == "Slice" {
                    let geti = |k: usize| -> Option<i64> {
                        let nm = n.inputs.get(k + 1)?;
                        self.initializers
                            .get(nm)
                            .and_then(|t| t.i64.first().copied())
                            .or_else(|| {
                                // 折叠链的值在 planner 表里
                                table.values.get(nm).and_then(|v| {
                                    v.konst.as_ref().and_then(|kv| kv.as_i64().first().copied())
                                })
                            })
                    };
                    let (start, end, axis, step) = (geti(0), geti(1), geti(2), geti(3));
                    let inn = table.values.get(&n.inputs[0]);
                    let outn = table.values.get(&out);
                    let ok = matches!(
                        (start, end, axis, step),
                        (Some(s), Some(_e), Some(0), Some(1) | None) if s >= 0
                    );
                    if let (Some(inn), Some(outn)) = (inn, outn) {
                        if ok {
                            let dim0 = inn.shape.first().copied().unwrap_or(1).max(1);
                            let row = inn.shape.iter().product::<i64>() / dim0;
                            if let Some(o) = offs.get(&n.inputs[0]).copied() {
                                let s = start.unwrap();
                                offs.insert(out.clone(), o + (s * row) as u32);
                                // rc 按输出形状重算（列=末维不变、行数去掉
                                // dim0 ——照搬输入会把行数多算 3 倍）
                                if let Some(&last) = outn.shape.last() {
                                    if last > 0 {
                                        let rows = outn.shape.iter().product::<i64>() / last;
                                        rc.insert(out.clone(), (rows as u32, last as u32));
                                    }
                                }
                                continue;
                            }
                        }
                    }
                    // 非 dim0/非连续：落到 map_node 报不支持（Slice 无臂）
                }
                let is_alias = match n.op_type.as_str() {
                    "Reshape" => {
                        // 存储恒等条件：元素数不变（纯形状重排）。
                        let inn = table.values.get(&n.inputs[0]);
                        let outn = table.values.get(&out);
                        inn.map(|v| v.shape.iter().product::<i64>())
                            == outn.map(|v| v.shape.iter().product::<i64>())
                    }
                    "Squeeze" | "Unsqueeze" => true,
                    "Transpose" => {
                        let perm = n.attr("perm").map(|a| a.ints.clone()).unwrap_or_default();
                        // [0,2,1] 的存储恒等**条件**：输入存储 rc 与输出
                        // 逻辑 rc 一致（rows=行数、cols=末维）。盲别名曾把
                        // 词表头 A 读成错误定向（输出全 NaN 的根源）。不满足
                        // 即走通用转置内核。
                        if perm == [0, 2, 1] {
                            let outn = table.values.get(&out);
                            if let (Some(&r), Some(o)) = (rc.get(&n.inputs[0]), outn) {
                                let last = o.shape.last().copied().unwrap_or(1).max(1);
                                let rows = o.shape.iter().product::<i64>() / last;
                                r == (rows as u32, last as u32)
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }
                    _ => false,
                };
                if is_alias {
                    let src = &n.inputs[0];
                    if let Some(o) = offs.get(src).copied() {
                        offs.insert(out.clone(), o);
                        if let Some(r) = rc.get(src).copied() {
                            rc.insert(out.clone(), r);
                        }
                        continue;
                    }
                }
            }
            // 被融合的节点：跳过（其消费者 Mul 用 n_channel 的 fused op
            // 替代）。不清输入——Mul 的 fused 内核直接读 HardSigmoid 的
            // **输入**（前激活门值），该张量仍需存活。
            if fused.contains(&n.name) {
                continue;
            }
            let out_v = table.values.get(&out).ok_or_else(|| {
                Error::Graph(format!("planner 缺少 {} 的输出 {}", n.op_type, out))
            })?;
            // i64 常量链节点：planner 已折叠出值，无区域无 dispatch
            if out_v.dtype == DType::I64 {
                release_inputs(
                    n,
                    &mut live,
                    &mut offs,
                    &mut layout,
                    &table,
                    &is_output,
                    self.initializers,
                    input_name,
                    false,
                    &rc,
                    dbg_env,
                );
                continue;
            }
            // 图输出 Sigmoid 在 exit 融合（省一整趟读写）。其输入区域不
            // 释放（无消费者递减）——正确，常驻到计划结束。
            if n.op_type == "Sigmoid" && is_output(&out) && offs.contains_key(&n.inputs[0]) {
                pending_exits.push((n.inputs[0].clone(), out.clone(), 4));
                continue;
            }
            // 输出区域先占（参数要引用它）。按 NHWC-f32 word 数（含
            // Cpad4 padding）。
            let out_n = numel(&table, &out);
            let out_rc = node_out_rc(n, &table, &rc)
                .map_err(|e| Error::Graph(format!("{}: {e}", n.name)))?;
            let out_words = if out_v.shape.len() == 4 {
                nhwc::nhwc_words(&out_v.shape)
            } else {
                let (r, c) = out_rc;
                r * c
            };
            let out_off = layout.alloc(out_words);
            if dbg_env {
                eprintln!(
                    "[plan][live] {out} @{} 分配于 {}（len {out_words}）",
                    out_off, n.name
                );
            }
            offs.insert(out.clone(), out_off);
            rc.insert(out.clone(), out_rc);

            // 节点 → 内核路由：可能展开多条 dispatch（reduce 两阶段）。
            let routes = self.map_node_n(
                n,
                &table,
                &offs,
                &fused,
                &mut layout,
                &mut upload,
                &mut w_offs,
                reduce_scratch,
                &mut rc,
            )?;
            for (kernel, pb, groups) in routes {
                // 参数块与数据同块（u32 视图）；静态内容，装载期写一次
                let p_off = layout.alloc_static(pb.len_words());
                params.push((p_off, pb.words().to_vec()));
                let ins: Vec<(String, u32, u32)> = n
                    .inputs
                    .iter()
                    .filter(|i| !i.is_empty())
                    .map(|i| {
                        (
                            i.clone(),
                            offs.get(i).copied().unwrap_or(0),
                            numel(&table, i),
                        )
                    })
                    .collect();
                let in_shapes: Vec<Vec<i64>> = n
                    .inputs
                    .iter()
                    .filter(|i| !i.is_empty())
                    .map(|i| shape_of(&table, i))
                    .collect();
                recs.push(Rec {
                    kernel,
                    node: n.name.clone(),
                    node_idx,
                    out_off,
                    out_n,
                    p_off,
                    groups,
                    ins,
                    in_shapes,
                    out_shape: out_v.shape.clone(),
                });
            } // routes 循环尾（release 按节点一次，不按 dispatch）

            release_inputs(
                n,
                &mut live,
                &mut offs,
                &mut layout,
                &table,
                &is_output,
                self.initializers,
                input_name,
                false,
                &rc,
                dbg_env,
            );
        }

        // exit 转换（每个图输出一条；sigmoid 融合的在节点环里挂单）。
        // 输出区按图输出顺序分配 f32 NCHW word 数——out_offs 指它（host
        // 读回）。
        {
            let mut exit_order: Vec<String> = Vec::new();
            for o in self.graph.outputs.iter() {
                if !pending_exits.iter().any(|(_, name, _)| name == o) {
                    exit_order.push(o.clone());
                }
            }
            for (_, name, _) in &pending_exits {
                exit_order.push(name.clone());
            }
            for name in exit_order {
                let (src_name, act) = pending_exits
                    .iter()
                    .find(|(_, n, _)| *n == name)
                    .map(|(s, _, a)| (s.clone(), *a))
                    .unwrap_or((name.clone(), 0));
                let Some(src_off) = offs.get(&src_name).copied() else {
                    return Err(Error::Graph(format!("exit：{src_name} 无区域")));
                };
                let Some(v) = table.values.get(&name) else {
                    return Err(Error::Graph(format!("exit：图输出 {name} 无形状")));
                };
                let out_words = numel(&table, &name);
                if v.shape.len() != 4 {
                    // rank-3 输出（rec 的 [B,T,V]）：存储 = NCHW 行主序
                    // 的恒等拷贝（cols=V 连续 = NCHW 最后轴），f32→f32。
                    // pad 列（cpad 超出 V）不写出。argmax_exit 时改走
                    // n_exit3_argmax：每行写 (val, idx) 2 word，出区同缩。
                    let (rows, cpad) = rc.get(&src_name).copied().ok_or_else(|| {
                        Error::Graph(format!("exit：rank-{} 输出 {name} 无 rc", v.shape.len()))
                    })?;
                    let real_cols = *v.shape.last().unwrap() as u32;
                    // 每批行数（rows = N_pad×T，批是外维）：空行早退用。
                    let rpb = (rows / v.shape[0].max(1) as u32).max(1);
                    let argmax = self.argmax_exit;
                    let out_n_words = if argmax { rows * 2 } else { out_words };
                    let out_off = layout.alloc(out_n_words);
                    let mut pb = ParamBlock::new();
                    pb.u(src_off).u(out_off).u(rows).u(real_cols).u(cpad).u(rpb);
                    let p_off = layout.alloc_static(pb.len_words());
                    params.push((p_off, pb.words().to_vec()));
                    recs.push(Rec {
                        kernel: if argmax { "n_exit3_argmax" } else { "n_exit3" },
                        node: format!("<exit3:{name}>"),
                        node_idx: usize::MAX,
                        out_off,
                        out_n: out_n_words,
                        p_off,
                        groups: if argmax {
                            [rows, 1, 1]
                        } else {
                            [(rows * real_cols).div_ceil(256), 1, 1]
                        },
                        ins: vec![(src_name.clone(), src_off, out_n_words)],
                        in_shapes: vec![v.shape.clone()],
                        out_shape: v.shape.clone(),
                    });
                    if argmax {
                        argmax_outs.insert(name.clone());
                    }
                    offs.insert(name.clone(), out_off);
                    continue;
                }
                let out_off = layout.alloc(out_words);
                let (nb, c, h, w) = (v.shape[0], v.shape[1], v.shape[2], v.shape[3]);
                let mut pb = ParamBlock::new();
                pb.u(src_off)
                    .u(out_off)
                    .u(nb as u32)
                    .u(c as u32)
                    .u(h as u32)
                    .u(w as u32)
                    .u(nhwc::cpad4(c))
                    .u(act);
                let p_off = layout.alloc_static(pb.len_words());
                params.push((p_off, pb.words().to_vec()));
                recs.push(Rec {
                    kernel: "n_exit",
                    node: format!("<exit:{name}>"),
                    node_idx: usize::MAX,
                    out_off,
                    out_n: out_words,
                    p_off,
                    groups: [((nb * h * w) as u32).div_ceil(256), 1, 1],
                    ins: vec![(src_name.clone(), src_off, out_words)],
                    in_shapes: vec![v.shape.clone()],
                    out_shape: v.shape.clone(),
                });
                // out_offs 指向 f32 区（复用下方统一构造：先记到 offs 供其读取）
                offs.insert(name.clone(), out_off);
            }
        }

        let out_offs: Vec<(String, u32, usize)> = self
            .graph
            .outputs
            .iter()
            .map(|o| {
                let off = offs.get(o).copied().unwrap_or(0);
                // argmax 出口：元素数 = 行数×2（(val,idx) 对），非逻辑 numel
                let n = if argmax_outs.contains(o) {
                    numel(&table, o) as usize
                        / table
                            .values
                            .get(o)
                            .map(|v| *v.shape.last().unwrap_or(&1) as usize)
                            .unwrap_or(1)
                        * 2
                } else {
                    numel(&table, o) as usize
                };
                (o.clone(), off, n)
            })
            .collect();
        let out_shapes: Vec<Vec<i64>> = self
            .graph
            .outputs
            .iter()
            .map(|o| {
                table
                    .values
                    .get(o)
                    .map(|v| {
                        if argmax_outs.contains(o) {
                            // [N,T,V] → [N,T,2]
                            let mut s = v.shape.clone();
                            *s.last_mut().unwrap() = 2;
                            s
                        } else {
                            v.shape.clone()
                        }
                    })
                    .unwrap_or_default()
            })
            .collect();

        if std::env::var_os("QPPOCR_GPU_BUILD_TIME").is_some() {
            eprintln!(
                "[plan] 计划就绪：输入 {in_shape:?}，{} 个 dispatch，整块 {:.1} MB（建 {:.1} ms）",
                recs.len(),
                layout.total as f64 * 4.0 / 1e6,
                t_build.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(GraphPlan {
            total_words: layout.total,
            rn_word,
            in_f32_off,
            dispatches: recs
                .into_iter()
                .map(|r| Dispatch {
                    kernel: r.kernel,
                    node: r.node,
                    node_idx: r.node_idx,
                    out_off: r.out_off,
                    out_n: r.out_n,
                    p_off: r.p_off,
                    groups: r.groups,
                    ins: r.ins,
                    in_shapes: r.in_shapes,
                    out_shape: r.out_shape,
                })
                .collect(),
            uploads: {
                let mut up = upload;
                up.extend(params);
                up
            },
            upload_init,
            out_offs,
            out_shapes,
        })
    }

    /// 节点 → (内核名, 参数块, dispatch 网格)。可能展开多条（reduce
    /// 两阶段、定向中转前置、逐元素门拆分）。
    ///
    /// 全图激活按 NHWC f32 语义路由；偏移一律 u32 word 单位。conv 族
    /// 权重/偏置经 [`n_weight_off`] 惰性重排（k-major / tap-major）。
    #[allow(clippy::too_many_arguments)]
    fn map_node_n(
        &self,
        n: &Node,
        table: &planner::ShapeTable,
        offs: &HashMap<String, u32>,
        fused: &std::collections::HashSet<String>,
        layout: &mut Layout,
        uploads: &mut Vec<(u32, Vec<u32>)>,
        w_offs: &mut HashMap<String, u32>,
        reduce_scratch: u32,
        rc: &mut HashMap<String, (u32, u32)>,
    ) -> Result<Vec<(&'static str, ParamBlock, [u32; 3])>> {
        use nhwc::{cpad4, repack_conv_w, repack_convt_w, repack_dw_w};
        let shape_of = |name: &str| -> Vec<i64> {
            table
                .values
                .get(name)
                .map(|v| v.shape.clone())
                .unwrap_or_default()
        };
        let off_of = |name: &str| -> Result<u32> {
            offs.get(name).copied().ok_or_else(|| {
                Error::Graph(format!(
                    "{}: 输入 {} 无区域（非权重/前驱输出）",
                    n.op_type, name
                ))
            })
        };
        let div256 = |t: u32| t.div_ceil(256);
        let out = &n.outputs[0];
        let out_shape = shape_of(out);
        // act 参数（conv 族：0=无 1=gelu 2=relu，镜像 Activation）
        let act = planner::get_i(n.attr("act"), 0) as u32;
        let c1 = planner::get_f(n.attr("act_c1"), std::f32::consts::SQRT_2);
        let c2 = planner::get_f(n.attr("act_c2"), 1.0);
        let c3 = planner::get_f(n.attr("act_c3"), 0.5);

        match n.op_type.as_str() {
            "Conv" => {
                let xs = shape_of(&n.inputs[0]);
                let ws = shape_of(&n.inputs[1]); // [Co, Ci, kh, kw]
                let (sh, sw) = strides(n);
                let pads = pads4(n);
                let (kh, kw) = (ws[2] as usize, ws[3] as usize);
                let nb = xs[0] as u32;
                let (oh, ow) = (out_shape[2] as u32, out_shape[3] as u32);
                let m_dim = oh * ow;
                let group = planner::get_i(n.attr("group"), 1);
                if group == 1 {
                    let (co, ci) = (ws[0] as usize, ws[1] as usize);
                    let w_off = n_weight_off(
                        self.initializers,
                        &n.inputs[1],
                        layout,
                        uploads,
                        w_offs,
                        self.repack_cache,
                        |t| repack_conv_w(&t.f32, co, ci, kh, kw),
                    )?;
                    let b_off = self.n_bias_off(n, ws[0], layout, uploads, w_offs)?;
                    // 残差（fuse_conv_residual 折进 conv 的 inputs[3]；
                    // NHWC 同布局直加，act 之后——镜像 CPU conv2d_res）
                    let r_off = if n.inputs.len() > 3 && !n.inputs[3].is_empty() {
                        off_of(&n.inputs[3])?
                    } else {
                        OFF_NONE
                    };
                    let mut pb = ParamBlock::new();
                    pb.u(off_of(&n.inputs[0])?)
                        .u(w_off)
                        .u(b_off)
                        .u(off_of(out)?)
                        .u(m_dim)
                        .u(cpad4(ws[0]))
                        .u(ow)
                        .u(xs[3] as u32)
                        .u(xs[2] as u32)
                        .u(cpad4(ws[1]) / 4)
                        .u((kh * kw) as u32)
                        .u(kw as u32)
                        .u(sh as u32)
                        .u(sw as u32)
                        .u(pads.0 as u32)
                        .u(pads.1 as u32)
                        .u(act)
                        .f(c1)
                        .f(c2)
                        .f(c3)
                        .u(r_off);
                    // 寄存器分块：Co%8==0 走 m4×n8（两列共享输入 gather，
                    // 输入流量减半）；奇数列回 m4×n4 基线。
                    let nv = cpad4(ws[0]) / 4;
                    if nv % 2 == 0 {
                        Ok(vec![(
                            "n_conv8",
                            pb,
                            [div256(m_dim.div_ceil(4) * (nv / 2)), nb, 1],
                        )])
                    } else {
                        Ok(vec![(
                            "n_conv",
                            pb,
                            [div256(m_dim.div_ceil(4) * nv), nb, 1],
                        )])
                    }
                } else if group == xs[1] && ws[0] == xs[1] {
                    // depthwise：ws = [C, 1, kh, kw]
                    let cch = ws[0] as usize;
                    let w_off = n_weight_off(
                        self.initializers,
                        &n.inputs[1],
                        layout,
                        uploads,
                        w_offs,
                        self.repack_cache,
                        |t| repack_dw_w(&t.f32, cch, kh, kw),
                    )?;
                    let b_off = self.n_bias_off(n, ws[0], layout, uploads, w_offs)?;
                    let r_off = if n.inputs.len() > 3 && !n.inputs[3].is_empty() {
                        off_of(&n.inputs[3])?
                    } else {
                        OFF_NONE
                    };
                    let cp = cpad4(ws[0]);
                    let mut pb = ParamBlock::new();
                    pb.u(off_of(&n.inputs[0])?)
                        .u(w_off)
                        .u(b_off)
                        .u(off_of(out)?)
                        .u(m_dim)
                        .u(cp)
                        .u(ow)
                        .u(xs[3] as u32)
                        .u(xs[2] as u32)
                        .u((kh * kw) as u32)
                        .u(kw as u32)
                        .u(sh as u32)
                        .u(sw as u32)
                        .u(pads.0 as u32)
                        .u(pads.1 as u32)
                        .u(act)
                        .f(c1)
                        .f(c2)
                        .f(c3)
                        .u(r_off);
                    Ok(vec![("n_conv_dw", pb, [div256(m_dim * cp / 4), nb, 1])])
                } else {
                    Err(Error::Graph(format!(
                        "n_ 路径：Conv group={group} 不支持（仅 group=1 或 depthwise），节点 {}",
                        n.name
                    )))
                }
            }
            "ConvTranspose" => {
                let xs = shape_of(&n.inputs[0]);
                let ws = shape_of(&n.inputs[1]); // [Ci, Co, kh, kw]
                let (sh, sw) = strides(n);
                let (kh, kw) = (ws[2] as usize, ws[3] as usize);
                if kh > sh as usize || kw > sw as usize {
                    return Err(Error::Graph(format!(
                        "n_ 路径：ConvTranspose k{kh}x{kw} > s{sh}x{sw}（多 tap 输出未支持），节点 {}",
                        n.name
                    )));
                }
                let (ci, co) = (ws[0] as usize, ws[1] as usize);
                let w_off = n_weight_off(
                    self.initializers,
                    &n.inputs[1],
                    layout,
                    uploads,
                    w_offs,
                    self.repack_cache,
                    |t| repack_convt_w(&t.f32, ci, co, kh, kw),
                )?;
                let b_off = self.n_bias_off(n, ws[1], layout, uploads, w_offs)?;
                let (oh, ow) = (out_shape[2] as u32, out_shape[3] as u32);
                let m_dim = oh * ow;
                let nv = cpad4(ws[1]) / 4;
                let cip = cpad4(ws[0]);
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(w_off)
                    .u(b_off)
                    .u(off_of(out)?)
                    .u(m_dim)
                    .u(cpad4(ws[1]))
                    .u(ow)
                    .u(xs[3] as u32)
                    .u(xs[2] as u32)
                    .u(cip)
                    .u(cip / 4)
                    .u(sh as u32)
                    .u(sw as u32)
                    .u(kw as u32)
                    .u(act)
                    .f(c1)
                    .f(c2)
                    .f(c3);
                Ok(vec![("n_convt", pb, [div256(m_dim * nv), xs[0] as u32, 1])])
            }
            "MaxPool" | "AveragePool" => {
                let xs = shape_of(&n.inputs[0]);
                let ks = n
                    .attr("kernel_shape")
                    .filter(|a| a.ints.len() == 2)
                    .ok_or_else(|| Error::Graph("kernel_shape required".into()))?;
                let (sh, sw) = strides(n);
                let pads = pads4(n);
                let (oh, ow) = (out_shape[2] as u32, out_shape[3] as u32);
                let m_dim = oh * ow;
                let cp = cpad4(xs[1]);
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(off_of(out)?)
                    .u(m_dim)
                    .u(cp)
                    .u(ow)
                    .u(xs[3] as u32)
                    .u(xs[2] as u32)
                    .u(ks.ints[0] as u32)
                    .u(ks.ints[1] as u32)
                    .u(sh as u32)
                    .u(sw as u32)
                    .u(pads.0 as u32)
                    .u(pads.1 as u32)
                    .u(u32::from(n.op_type == "MaxPool"));
                Ok(vec![(
                    "n_pool",
                    pb,
                    [div256(m_dim * cp / 4), xs[0] as u32, 1],
                )])
            }
            "GlobalAveragePool" | "ReduceMean" => {
                if n.op_type == "ReduceMean" {
                    let axes = planner::axes_from(n, &table.values)
                        .map_err(|e| Error::Graph(format!("ReduceMean: {e}")))?;
                    let xs = shape_of(&n.inputs[0]);
                    let r = xs.len() as i64;
                    let norm: Vec<i64> = axes
                        .iter()
                        .map(|&a| if a < 0 { a + r } else { a })
                        .collect();
                    if norm.len() == 1 && norm[0] == r - 1 {
                        // 末轴均值（LayerNorm 分解链）：一行 WG 归约，出
                        // [rows,4]（lane0=均值）。rpb 供空行早退。
                        let rows: u32 = xs[..xs.len() - 1].iter().product::<i64>() as u32;
                        let cols = *xs.last().unwrap() as u32;
                        let (in_rows, cpad) = rc
                            .get(&n.inputs[0])
                            .copied()
                            .ok_or_else(|| Error::Graph("末轴均值输入无 rc".into()))?;
                        if in_rows != rows {
                            return Err(Error::Graph(format!(
                                "末轴均值行数不符：rc={in_rows} 形状={rows}"
                            )));
                        }
                        let rpb = (rows / xs[0].max(1) as u32).max(1);
                        let mut pb = ParamBlock::new();
                        pb.u(off_of(&n.inputs[0])?)
                            .u(off_of(out)?)
                            .u(rows)
                            .u(cols)
                            .u(cpad)
                            .u(rpb);
                        return Ok(vec![("n_reduce_last", pb, [rows, 1, 1])]);
                    }
                    if !(norm.len() == 2 && norm.contains(&2) && norm.contains(&3)) {
                        return Err(Error::Graph(format!(
                            "n_ 路径 ReduceMean 只支持 axes={{2,3}} 或末轴，实得 {axes:?}"
                        )));
                    }
                }
                let xs = shape_of(&n.inputs[0]);
                let cp = cpad4(xs[1]);
                if cp > 1024 {
                    return Err(Error::Graph(format!(
                        "n_ 路径 SE 归约通道 {cp} > 1024（两阶段内核上限：\
                         cp4 ≤ 256 线程平铺；超出会静默算零，必须显式拒绝）"
                    )));
                }
                let nb = xs[0] as u32;
                let m_dim = (xs[2] * xs[3]) as u32;
                let m_blocks = m_dim.div_ceil(1024);
                // 阶段 1：分块部分和 → scratch（每批一段）
                let mut pb1 = ParamBlock::new();
                pb1.u(off_of(&n.inputs[0])?)
                    .u(reduce_scratch)
                    .u(m_dim)
                    .u(cp);
                let r1 = ("n_reduce_hw", pb1, [m_blocks, nb, 1]);
                // 阶段 2：scratch → gate
                let mut pb2 = ParamBlock::new();
                pb2.u(reduce_scratch)
                    .u(off_of(out)?)
                    .u(m_blocks)
                    .u(cp)
                    .u(m_dim);
                let r2 = ("n_reduce_fin", pb2, [nb, 1, 1]);
                Ok(vec![r1, r2])
            }
            "Resize" => {
                let mode = n
                    .attr("mode")
                    .filter(|a| a.has_s)
                    .map(|a| a.s.clone())
                    .unwrap_or_else(|| "nearest".into());
                if mode != "nearest" {
                    return Err(Error::Graph(format!(
                        "n_ 路径 Resize 只支持 nearest，实得 {mode}"
                    )));
                }
                let xs = shape_of(&n.inputs[0]);
                let (oh, ow) = (out_shape[2] as u32, out_shape[3] as u32);
                let m_dim = oh * ow;
                let cp = cpad4(xs[1]);
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(off_of(out)?)
                    .u(m_dim)
                    .u(cp)
                    .u(ow)
                    .u(xs[3] as u32)
                    .u(xs[2] as u32)
                    // 主机端 f64 预计算的比例系数：内核乘法版索引
                    //（GPU 浮点除法可能有倒数截断缺陷，见 n_resize）。
                    .f((xs[2] as f64 / oh as f64) as f32)
                    .f((xs[3] as f64 / ow as f64) as f32);
                Ok(vec![(
                    "n_resize",
                    pb,
                    [div256(m_dim * cp / 4), xs[0] as u32, 1],
                )])
            }
            "Concat" => {
                let axis = planner::get_i(n.attr("axis"), 0);
                if axis != 1 {
                    return Err(Error::Graph(format!(
                        "n_ 路径 Concat 只支持 axis=1，实得 {axis}"
                    )));
                }
                let xs = shape_of(&n.inputs[0]);
                // vec4 整段拷贝要求所有输入 C%4==0（否则段内 pad 会错位）
                for inn in &n.inputs {
                    if inn.is_empty() {
                        continue;
                    }
                    let s = shape_of(inn);
                    if s[1] % 4 != 0 {
                        return Err(Error::Graph(format!(
                            "n_ 路径 Concat 输入 {inn} 通道 {} 非 %4（段内 pad 会错位）",
                            s[1]
                        )));
                    }
                }
                let hw = (xs[0] * xs[2] * xs[3]) as u32;
                let out_c4 = cpad4(out_shape[1]) / 4;
                let mut pb = ParamBlock::new();
                pb.u(off_of(out)?).u(hw).u(n.inputs.len() as u32).u(out_c4);
                for inn in &n.inputs {
                    if inn.is_empty() {
                        continue;
                    }
                    let s = shape_of(inn);
                    pb.u(off_of(inn)?).u(cpad4(s[1]) / 4);
                }
                Ok(vec![("n_concat_c", pb, [div256(hw * out_c4), 1, 1])])
            }
            "MatMul" => {
                // [.., M, K] × [K, N] = 1×1 conv 语义复用 n_conv8：
                // A 存储天然 [M 行, K 列]（NHWC C=K）；B 转置成 [N,K] 后
                // 按 [Co=N, Ci=K, 1,1] k-major 重排。
                let a = shape_of(&n.inputs[0]);
                let b = shape_of(&n.inputs[1]);
                // **动态 B**（注意力 QK^T / scores×V）：B 是运行期张量
                // [..batch, K, N]——批量瘦 GEMM，K=15 或 T，朴素点积。
                // A/B/出均精确连续存储（rc 行=末维，无 pad）。
                if !self.initializers.contains_key(&n.inputs[1]) {
                    if a.len() < 2 || b.len() < 2 {
                        return Err(Error::Graph(format!(
                            "动态 MatMul 形状不支持：A={a:?} B={b:?}"
                        )));
                    }
                    let (m, k, nn) = (
                        a[a.len() - 2] as u32,
                        *a.last().unwrap() as u32,
                        *b.last().unwrap() as u32,
                    );
                    if *a.last().unwrap() != b[b.len() - 2] {
                        return Err(Error::Graph(format!("动态 MatMul K 不符：A={a:?} B={b:?}")));
                    }
                    let batch: i64 = a[..a.len() - 2].iter().product();
                    let heads = ((batch / a[0].max(1)) as u32).max(1);
                    let mut pb = ParamBlock::new();
                    pb.u(off_of(&n.inputs[0])?)
                        .u(off_of(&n.inputs[1])?)
                        .u(off_of(out)?)
                        .u(m)
                        .u(k)
                        .u(nn)
                        .u(batch as u32)
                        .u(heads);
                    return Ok(vec![("n_attn_mm", pb, [1, (batch as u32 * m).max(1), 1])]);
                }
                if b.len() != 2 {
                    return Err(Error::Graph(format!("MatMul B 需 2D [K,N]，实得 {b:?}")));
                }
                let m_dim = a[..a.len() - 1].iter().product::<i64>() as u32;
                let (k, nn) = (b[0] as usize, b[1] as usize);
                let bt = self.initializers.get(&n.inputs[1]).ok_or_else(|| {
                    Error::Graph(format!(
                        "MatMul B {} 不是 initializer（仅支持常量）",
                        n.inputs[1]
                    ))
                })?;
                let mut w_t = vec![0f32; nn * k];
                for i in 0..k {
                    for j in 0..nn {
                        w_t[j * k + i] = bt.f32[i * nn + j];
                    }
                }
                let w_off = n_weight_off(
                    self.initializers,
                    &n.inputs[1],
                    layout,
                    uploads,
                    w_offs,
                    self.repack_cache,
                    |_| nhwc::repack_conv_w(&w_t, nn, k, 1, 1),
                )?;
                let (_, cols) = rc
                    .get(out)
                    .copied()
                    .ok_or_else(|| Error::Graph("MatMul 输出无 rc".into()))?;
                let nv = cols / 4;
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(w_off)
                    .u(OFF_NONE)
                    .u(off_of(out)?)
                    .u(m_dim)
                    .u(cols)
                    .u(m_dim) // outW
                    .u(m_dim) // inW：边界检查用（iw 最大 = m-1）
                    .u(1) // inH
                    .u(nhwc::cpad4(k as i64) / 4)
                    .u(1)
                    .u(1)
                    .u(1)
                    .u(1)
                    .u(0)
                    .u(0)
                    .u(0)
                    .f(std::f32::consts::SQRT_2)
                    .f(1.0)
                    .f(0.5)
                    .u(OFF_NONE);
                if nv % 2 == 0 {
                    Ok(vec![(
                        "n_conv8",
                        pb,
                        [div256(m_dim.div_ceil(4) * (nv / 2)), 1, 1],
                    )])
                } else {
                    Ok(vec![("n_conv", pb, [div256(m_dim.div_ceil(4) * nv), 1, 1])])
                }
            }
            "Softmax" => {
                let axis = planner::get_i(n.attr("axis"), -1);
                let osh = shape_of(out);
                let r = if axis < 0 {
                    osh.len() as i64 + axis
                } else {
                    axis
                };
                if r != osh.len() as i64 - 1 {
                    return Err(Error::Graph(format!(
                        "Softmax 只支持最后轴，实得 axis={axis}"
                    )));
                }
                let (rows, cpad) = rc
                    .get(&n.inputs[0])
                    .copied()
                    .ok_or_else(|| Error::Graph("Softmax 输入无 rc".into()))?;
                let cols = osh.last().copied().unwrap_or(0) as u32;
                // 每批行数（rows = N_pad×T，批是外维）：n_softmax 的空行
                // 早退要用 real_n×rpb 定位真实行段。
                let rpb = (rows / osh[0].max(1) as u32).max(1);
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(off_of(out)?)
                    .u(rows)
                    .u(cols)
                    .u(cpad)
                    .u(rpb);
                Ok(vec![("n_softmax", pb, [rows, 1, 1])])
            }
            "BatchNormalization" => {
                let xs = shape_of(&n.inputs[0]);
                if xs.len() != 3 && xs.len() != 4 {
                    return Err(Error::Graph(format!("BN 只支持 rank-3/4，实得 {xs:?}")));
                }
                let c = xs[1] as usize;
                let eps = planner::get_f(n.attr("epsilon"), 1e-5);
                let get = |nm: &str| -> Result<Vec<f32>> {
                    self.initializers
                        .get(nm)
                        .map(|t| t.f32.to_vec())
                        .ok_or_else(|| Error::Graph(format!("BN 参数 {nm} 不是 initializer")))
                };
                let scale = get(&n.inputs[1])?;
                let bv = get(&n.inputs[2])?;
                let mean = get(&n.inputs[3])?;
                let var = get(&n.inputs[4])?;
                let mut g = vec![0f32; c];
                let mut bb = vec![0f32; c];
                for i in 0..c {
                    let sc = scale.get(i).copied().unwrap_or(1.0)
                        / (var.get(i).copied().unwrap_or(0.0) + eps).sqrt();
                    g[i] = sc;
                    bb[i] = bv.get(i).copied().unwrap_or(0.0)
                        - mean.get(i).copied().unwrap_or(0.0) * sc;
                }
                let pack_c = |vals: &[f32]| -> Vec<u32> {
                    let cp = nhwc::cpad4(c as i64) as usize;
                    let mut v = vec![0f32; cp];
                    v[..c].copy_from_slice(vals);
                    v.iter().map(|f| f.to_bits()).collect()
                };
                let g_off = {
                    let data = pack_c(&g);
                    let off = layout.alloc_static(data.len() as u32);
                    uploads.push((off, data));
                    off
                };
                let b_off2 = {
                    let data = pack_c(&bb);
                    let off = layout.alloc_static(data.len() as u32);
                    uploads.push((off, data));
                    off
                };
                let (rows, cpad) = rc
                    .get(&n.inputs[0])
                    .copied()
                    .ok_or_else(|| Error::Graph("BN 输入无 rc".into()))?;
                let mut pb = ParamBlock::new();
                pb.u(5u32) // op=bn：a*g + b 双通道广播
                    .u(off_of(&n.inputs[0])?)
                    .u(g_off)
                    .u(b_off2)
                    .u(off_of(out)?)
                    .u(rows * cpad)
                    .u(cpad)
                    .f(0.0)
                    .f(0.0)
                    .u(rows); // 门单行：全部特征行映到第 0 行
                Ok(vec![("n_channel", pb, [div256(rows * cpad / 4), 1, 1])])
            }
            "Sigmoid" | "Relu" | "HardSigmoid" | "Clip" | "Sqrt" => {
                let (op, p1, p2) = match n.op_type.as_str() {
                    "Sigmoid" => (1u32, 0f32, 0f32),
                    "Relu" => (0, 0.0, 0.0),
                    "HardSigmoid" => (
                        2,
                        planner::get_f(n.attr("alpha"), 0.2),
                        planner::get_f(n.attr("beta"), 0.5),
                    ),
                    "Sqrt" => (10, 0.0, 0.0),
                    _ => (
                        3,
                        planner::get_f(n.attr("min"), -3.4e38),
                        planner::get_f(n.attr("max"), 3.4e38),
                    ),
                };
                let n_words = if shape_of(&n.inputs[0]).len() == 4 {
                    nhwc::nhwc_words(&shape_of(&n.inputs[0]))
                } else {
                    let (r, c) = rc.get(&n.inputs[0]).copied().unwrap_or((0, 0));
                    r * c
                };
                let mut pb = ParamBlock::new();
                pb.u(op)
                    .u(off_of(&n.inputs[0])?)
                    .u(OFF_NONE)
                    .u(off_of(out)?)
                    .u(n_words)
                    .f(p1)
                    .f(p2);
                Ok(vec![("n_elem", pb, [div256(n_words / 4), 1, 1])])
            }
            "Add" | "Mul" | "Sub" | "Div" => {
                let a = shape_of(&n.inputs[0]);
                let b = shape_of(&n.inputs[1]);
                let b_n = numel(table, &n.inputs[1]);
                let a_words = if a.len() == 4 {
                    nhwc::nhwc_words(&a)
                } else {
                    let (r, c) = rc.get(&n.inputs[0]).copied().unwrap_or((0, 0));
                    r * c
                };
                if a == b {
                    let op = match n.op_type.as_str() {
                        "Add" => 4u32,
                        "Mul" => 5,
                        "Sub" => 12,
                        _ => 13,
                    };
                    // 定向一致性（同 MulAddScale 门分支）：按生产者定向中转。
                    let mut pre: Vec<(&'static str, ParamBlock, [u32; 3])> = Vec::new();
                    let mut b_off = off_of(&n.inputs[1])?;
                    if let Some(t) = rc.get(&n.inputs[0]).copied() {
                        let s = rc.get(&n.inputs[1]).copied().unwrap_or(t);
                        let lin = linear_by_producer(
                            self.graph,
                            self.producer_idx,
                            table,
                            rc,
                            self.initializers,
                            &n.inputs[1],
                        );
                        let lin0 = linear_by_producer(
                            self.graph,
                            self.producer_idx,
                            table,
                            rc,
                            self.initializers,
                            &n.inputs[0],
                        );
                        b_off = orient4(layout, &mut pre, b_off, &b, s, t, lin, lin0)?;
                    }
                    let mut pb = ParamBlock::new();
                    pb.u(op)
                        .u(off_of(&n.inputs[0])?)
                        .u(b_off)
                        .u(off_of(out)?)
                        .u(a_words)
                        .f(0.0)
                        .f(0.0);
                    let mut ds = pre;
                    ds.push(("n_elem", pb, [div256(a_words / 4), 1, 1]));
                    Ok(ds)
                } else if a.len() == 3 && b.len() == 3 && b[2] == 1 && b[0] == a[0] && b[1] == a[1]
                {
                    // rank-3 行向量广播（LN 链的 x-mean、x/std）：b 是
                    // [B,T,1] → 存储 [rows,4]（lane0 有效），行内 4 通道
                    // 同减/除。p1 携带每行 vec4 数 c4 的位型。
                    let (_, cpad) = rc
                        .get(&n.inputs[0])
                        .copied()
                        .ok_or_else(|| Error::Graph("行向量广播输入无 rc".into()))?;
                    let op = if n.op_type == "Sub" { 8u32 } else { 11 };
                    let mut pb = ParamBlock::new();
                    pb.u(op)
                        .u(off_of(&n.inputs[0])?)
                        .u(off_of(&n.inputs[1])?)
                        .u(off_of(out)?)
                        .u(a_words)
                        .f(f32::from_bits(cpad / 4))
                        .f(0.0);
                    Ok(vec![("n_elem", pb, [div256(a_words / 4), 1, 1])])
                } else if b.len() == 4
                    && b[2] == 1
                    && b[3] == 1
                    && b[0] == a[0]
                    && b[1] == a[1]
                    && matches!(n.op_type.as_str(), "Add" | "Mul")
                {
                    // [N,C,1,1] 通道广播（rec 批的 SE 门；n==b 的批维）
                    // [1,C,1,1] 通道广播；SE 融合门 → fused（读前激活值）
                    //（Sub/Div 无通道广播形态——不进此分支）
                    let gate_producer = self
                        .graph
                        .nodes
                        .iter()
                        .find(|m| m.outputs.first() == Some(&n.inputs[1]));
                    let is_fused_gate = gate_producer.is_some_and(|m| {
                        fused.contains(&m.name)
                            && (m.op_type == "HardSigmoid" || m.op_type == "Sigmoid")
                    });
                    let (op, gate_off, p1, p2) = if is_fused_gate {
                        let gp = gate_producer.unwrap();
                        let g_off = off_of(&gp.inputs[0])?; // 前激活门
                        if gp.op_type == "HardSigmoid" {
                            (
                                3u32,
                                g_off,
                                planner::get_f(gp.attr("alpha"), 0.2),
                                planner::get_f(gp.attr("beta"), 0.5),
                            )
                        } else {
                            (4, g_off, 0.0, 0.0)
                        }
                    } else if n.op_type == "Mul" {
                        (0, off_of(&n.inputs[1])?, 0.0, 0.0)
                    } else {
                        (1, off_of(&n.inputs[1])?, 0.0, 0.0)
                    };
                    let cp = cpad4(a[1]);
                    let hw_gate = (a[2] * a[3]) as u32; // 每门行的特征行数
                    let mut pb = ParamBlock::new();
                    pb.u(op)
                        .u(off_of(&n.inputs[0])?)
                        .u(gate_off)
                        .u(OFF_NONE) // r_off（muladd_scale 专用）
                        .u(off_of(out)?)
                        .u(a_words)
                        .u(cp)
                        .f(p1)
                        .f(p2)
                        .u(hw_gate);
                    Ok(vec![("n_channel", pb, [div256(a_words / 4), 1, 1])])
                // [V] 尾轴广播——b_n==1 是标量，须走后面的标量分支：不设此
                // 守卫时 cls-on-GPU 全图错位（cls_dw_precise 断言与
                // cls_session_end_to_end 双双失败；引擎默认 cls 走 CPU，
                // 100 图分数测不出，勿以引擎 A/B 判此分支）。
                } else if b.len() == 1
                    && b_n > 1
                    && a.last() == Some(&b[0])
                    && a.len() >= 2
                    && matches!(n.op_type.as_str(), "Add" | "Mul")
                {
                    // [V] 尾轴广播（rank-2/3 的 a：cls 的 Add([B,2],[2])、
                    // rec 的 MatMul bias）——存储列 = 尾轴 → n_channel
                    let (rows, cpad) = rc
                        .get(&n.inputs[0])
                        .copied()
                        .ok_or_else(|| Error::Graph("[1,V] 广播输入无 rc".into()))?;
                    let mut pb = ParamBlock::new();
                    pb.u(if n.op_type == "Add" { 1u32 } else { 0u32 })
                        .u(off_of(&n.inputs[0])?)
                        .u(off_of(&n.inputs[1])?)
                        .u(OFF_NONE)
                        .u(off_of(out)?)
                        .u(rows * cpad)
                        .u(cpad)
                        .f(0.0)
                        .f(0.0)
                        .u(rows); // 门单行：全部特征行映到第 0 行
                    Ok(vec![("n_channel", pb, [div256(rows * cpad / 4), 1, 1])])
                } else if (b.is_empty() || (b.len() == 1 && b_n == 1))
                    && matches!(n.op_type.as_str(), "Add" | "Mul")
                {
                    // 标量广播：[V] 退化成单值（cls 的 gate×标量）或 rank-0
                    //（LN 的 var+eps——paddle2onnx 导出 helper.constant 无维）
                    let mut pb = ParamBlock::new();
                    pb.u(if n.op_type == "Add" { 6u32 } else { 7u32 })
                        .u(off_of(&n.inputs[0])?)
                        .u(off_of(&n.inputs[1])?)
                        .u(off_of(out)?)
                        .u(a_words)
                        .f(0.0)
                        .f(0.0);
                    Ok(vec![("n_elem", pb, [div256(a_words / 4), 1, 1])])
                } else {
                    Err(Error::Graph(format!(
                        "n_ 路径 {}：只支持同形或 [1,C,1,1]/[V]/行向量 广播，实得 {a:?} vs {b:?}",
                        n.op_type
                    )))
                }
            }
            "Pow" => {
                // 常量指数 2（LN 方差链）→ square；其它指数不支持。
                let e2 = self
                    .initializers
                    .get(&n.inputs[1])
                    .and_then(|t| t.f32.first().copied());
                if e2 != Some(2.0) {
                    return Err(Error::Graph(format!(
                        "n_ 路径 Pow 只支持常量指数 2（LN 方差），实得 {e2:?}"
                    )));
                }
                let a_words = if shape_of(&n.inputs[0]).len() == 4 {
                    nhwc::nhwc_words(&shape_of(&n.inputs[0]))
                } else {
                    let (r, c) = rc.get(&n.inputs[0]).copied().unwrap_or((0, 0));
                    r * c
                };
                let mut pb = ParamBlock::new();
                pb.u(9u32) // square
                    .u(off_of(&n.inputs[0])?)
                    .u(OFF_NONE)
                    .u(off_of(out)?)
                    .u(a_words)
                    .f(0.0)
                    .f(0.0);
                Ok(vec![("n_elem", pb, [div256(a_words / 4), 1, 1])])
            }
            "MulAddScale" => {
                let fs = shape_of(&n.inputs[0]);
                if fs.len() != 4 {
                    // rank-3 门控（small rec 的 LN 尾巴 ×scale+shift）：门与
                    // 残差都是 [C] 常量广播——正是 op5（bn 的 a*g+b 双行
                    // 广播）的语义，单 dispatch（hw=rows 全部行映到门 0 行）。
                    // 镜像 CPU 慢路径（Mul 广播 + Add 广播）。
                    let (rows, cpad) = rc
                        .get(&n.inputs[0])
                        .copied()
                        .ok_or_else(|| Error::Graph("rank-3 MulAddScale 输入无 rc".into()))?;
                    let mut pb = ParamBlock::new();
                    pb.u(5u32)
                        .u(off_of(&n.inputs[0])?)
                        .u(off_of(&n.inputs[1])?)
                        .u(off_of(&n.inputs[2])?)
                        .u(off_of(out)?)
                        .u(rows * cpad)
                        .u(cpad)
                        .f(0.0)
                        .f(0.0)
                        .u(rows); // 门单行
                    return Ok(vec![("n_channel", pb, [div256(rows * cpad / 4), 1, 1])]);
                }
                let a_words = nhwc::nhwc_words(&fs);
                let cp = cpad4(fs[1]);
                // **逐元素门检测**（swish：Add(Mul(x, Sigmoid(x)), res)——
                // 门 shape == 特征 shape，op2 的 `g[gateRow*c4+rowC4]` 只读
                // 每批的 w=0 列，对它会整个读错。拆回 mul + add 两个
                // n_elem（镜像 CPU 慢路径），中间量用临时区。
                let gate_shape = shape_of(&n.inputs[1]);
                if gate_shape == fs {
                    let tmp_off = layout.alloc(a_words);
                    // 定向一致性：mul/add 三操作数的字节序必须一致；按生产者
                    // 定向中转成 inputs[0]（NHWC 基准）的定向。
                    let base_rc = rc.get(&n.inputs[0]).copied();
                    let mut pre: Vec<(&'static str, ParamBlock, [u32; 3])> = Vec::new();
                    let mut g_off = off_of(&n.inputs[1])?;
                    let mut r_off = off_of(&n.inputs[2])?;
                    if let Some(t) = base_rc {
                        let lin0 = linear_by_producer(
                            self.graph,
                            self.producer_idx,
                            table,
                            rc,
                            self.initializers,
                            &n.inputs[0],
                        );
                        {
                            let s = rc.get(&n.inputs[1]).copied().unwrap_or(t);
                            let lin = linear_by_producer(
                                self.graph,
                                self.producer_idx,
                                table,
                                rc,
                                self.initializers,
                                &n.inputs[1],
                            );
                            g_off = orient4(layout, &mut pre, g_off, &gate_shape, s, t, lin, lin0)?;
                        }
                        let r_shape = shape_of(&n.inputs[2]);
                        {
                            let s = rc.get(&n.inputs[2]).copied().unwrap_or(t);
                            let lin = linear_by_producer(
                                self.graph,
                                self.producer_idx,
                                table,
                                rc,
                                self.initializers,
                                &n.inputs[2],
                            );
                            r_off = orient4(layout, &mut pre, r_off, &r_shape, s, t, lin, lin0)?;
                        }
                    }
                    let mk_pb = |op: u32, a: u32, b: u32, o: u32| -> ParamBlock {
                        let mut p = ParamBlock::new();
                        p.u(op).u(a).u(b).u(o).u(a_words).f(0.0).f(0.0);
                        p
                    };
                    let mut ds = pre;
                    ds.push((
                        "n_elem",
                        mk_pb(5, off_of(&n.inputs[0])?, g_off, tmp_off),
                        [div256(a_words / 4), 1, 1],
                    ));
                    ds.push((
                        "n_elem",
                        mk_pb(4, tmp_off, r_off, off_of(out)?),
                        [div256(a_words / 4), 1, 1],
                    ));
                    return Ok(ds);
                }
                let mut pb = ParamBlock::new();
                pb.u(2u32) // muladd_scale
                    .u(off_of(&n.inputs[0])?)
                    .u(off_of(&n.inputs[1])?)
                    .u(off_of(&n.inputs[2])?)
                    .u(off_of(out)?)
                    .u(a_words)
                    .u(cp)
                    .f(0.0)
                    .f(0.0)
                    .u((fs[2] * fs[3]) as u32); // 每门行的特征行数
                Ok(vec![("n_channel", pb, [div256(a_words / 4), 1, 1])])
            }
            "Transpose" => {
                // 通用真转置（别名之外的 perm——注意力的 [2,0,3,1,4] 等）。
                // rank ≤ 5、精确存储；存储定向恒等的 [0,2,1] 已在上游别名
                // 段短路，到达这里的都是要搬数据的。
                let ish = shape_of(&n.inputs[0]);
                let perm = n.attr("perm").map(|a| a.ints.clone()).unwrap_or_default();
                if ish.len() > 5 || perm.len() != ish.len() {
                    return Err(Error::Graph(format!(
                        "Transpose 仅支持 rank ≤ 5 且带 perm，实得 rank={} perm={perm:?}",
                        ish.len()
                    )));
                }
                let total: i64 = ish.iter().product();
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(off_of(out)?)
                    .u(total as u32)
                    .u(ish.len() as u32);
                for &d in &ish {
                    pb.u(d as u32);
                }
                for _ in ish.len()..5 {
                    pb.u(1); // rank 补齐（内核按 rank 截断）
                }
                for &p in &perm {
                    pb.u(p as u32);
                }
                Ok(vec![("n_transpose_nd", pb, [div256(total as u32), 1, 1])])
            }
            other => Err(Error::Graph(format!(
                "n_ 路径不支持算子 {other}（节点 {}）",
                n.name
            ))),
        }
    }

    /// conv 族 bias 的惰性 f16 重排区。
    #[allow(clippy::too_many_arguments)]
    fn n_bias_off(
        &self,
        n: &Node,
        co: i64,
        layout: &mut Layout,
        uploads: &mut Vec<(u32, Vec<u32>)>,
        w_offs: &mut HashMap<String, u32>,
    ) -> Result<u32> {
        if n.inputs.len() <= 2 || n.inputs[2].is_empty() {
            return Ok(OFF_NONE);
        }
        n_weight_off(
            self.initializers,
            &n.inputs[2],
            layout,
            uploads,
            w_offs,
            self.repack_cache,
            |t| nhwc::conv_bias(&t.f32, co),
        )
    }
}

/// 区域布局：活性复用的 bump + first-fit 空闲表（全部相对同一整块）。
pub(crate) struct Layout {
    pub(crate) free: Vec<(u32, u32)>,
    pub(crate) cursor: u32,
    /// 已触及的最大偏移（word）——arena 的总字数。
    pub(crate) total: u32,
}

impl Layout {
    pub(crate) fn new() -> Self {
        Self {
            free: Vec::new(),
            cursor: 0,
            total: 0,
        }
    }

    pub(crate) fn alloc(&mut self, n: u32) -> u32 {
        let n = n.div_ceil(4) * 4; // 16 B 对齐
        for i in 0..self.free.len() {
            if self.free[i].1 >= n {
                let off = self.free[i].0;
                let rest = self.free[i].1 - n;
                if rest == 0 {
                    self.free.swap_remove(i);
                } else {
                    // 剩余区段后移到 [off+n, ...)——留原地就是与本次
                    // 分配重叠（曾致 in-place 别名：输入输出同址、深度
                    // 卷积读写互相踩）。
                    self.free[i].0 = off + n;
                    self.free[i].1 = rest;
                }
                return off;
            }
        }
        let off = self.cursor;
        self.cursor += n;
        self.total = self.cursor;
        off
    }

    pub(crate) fn free(&mut self, off: u32, n: u32) {
        let n = n.div_ceil(4) * 4;
        if n > 0 {
            self.free.push((off, n));
        }
    }

    /// 静态区（权重/参数）专用：**只 bump、永不复用**。
    ///
    /// 激活区域可以 free-复用——生产者 dispatch 先写、消费者后读，
    /// 单次重放内自洽。但静态内容只在装载期写一次，而命令缓冲每帧重放
    /// 时**复用区的原生产者 dispatch 仍会写那片内存**——参数落进复用区
    /// = 运行时被冲掉、消费方读到垃圾参数（实测形态：conv 参数损坏 →
    /// 巨循环 → GPU 挂死设备丢失）。
    pub(crate) fn alloc_static(&mut self, n: u32) -> u32 {
        let off = self.cursor;
        self.cursor += n.div_ceil(4) * 4;
        self.total = self.cursor;
        off
    }
}

pub(crate) fn consumer_counts(graph: &Graph) -> HashMap<String, u32> {
    let mut refs: HashMap<String, u32> = HashMap::new();
    for n in &graph.nodes {
        for inn in &n.inputs {
            if !inn.is_empty() {
                *refs.entry(inn.clone()).or_default() += 1;
            }
        }
    }
    refs
}

pub(crate) fn numel(table: &planner::ShapeTable, name: &str) -> u32 {
    table
        .values
        .get(name)
        .map(|v| v.shape.iter().product::<i64>() as u32)
        .unwrap_or(0)
}

/// planner 形状表里取形状（缺项给空——调用方容错）。
pub(crate) fn shape_of(table: &planner::ShapeTable, name: &str) -> Vec<i64> {
    table
        .values
        .get(name)
        .map(|v| v.shape.clone())
        .unwrap_or_default()
}

/// 节点输出的存储形状 (rows, cols)——cols 为连续维。
///
/// rank-4 conv 族 = (N·H·W, Cpad)；MatMul = (批展开 M, Npad)；
/// rank-3 的逐元素/softmax/BN 继承输入 rc（同形或广播不变行列）。
fn node_out_rc(
    n: &Node,
    table: &planner::ShapeTable,
    rc: &HashMap<String, (u32, u32)>,
) -> std::result::Result<(u32, u32), String> {
    let sh = |name: &str| -> Vec<i64> {
        table
            .values
            .get(name)
            .map(|v| v.shape.clone())
            .unwrap_or_default()
    };
    if n.op_type == "MatMul" {
        let a = sh(&n.inputs[0]);
        let b = sh(&n.inputs[1]);
        if a.len() < 2 || b.len() < 2 {
            return Err(format!("MatMul 形状不支持：A={a:?} B={b:?}"));
        }
        let m: i64 = a[..a.len() - 1].iter().product();
        let k = a[a.len() - 1];
        let nn = *b.last().unwrap();
        if k != b[b.len() - 2] {
            return Err(format!("MatMul K 不符：A={a:?} B={b:?}"));
        }
        // 动态 B（注意力 QK^T / scores×V，b 为 rank-3/4 批量矩阵）：
        // 精确存储不加 pad——下游 softmax/转置都以末维为步长。
        if b.len() != 2 {
            return Ok((m as u32, nn as u32));
        }
        if k % 4 != 0 {
            return Err(format!("MatMul K={k} 非 %4（A={a:?}）"));
        }
        return Ok((m as u32, nhwc::cpad4(nn)));
    }
    // 真转置（非 [0,2,1] 别名——注意力头的 [2,0,3,1,4] 等）与 dim0
    // 切片：精确连续存储 (numel/末维, 末维)，无通道 pad。
    if n.op_type == "Transpose" || n.op_type == "Slice" {
        let o = sh(&n.outputs[0]);
        if !o.is_empty() && *o.last().unwrap() > 0 {
            let numel: i64 = o.iter().product();
            return Ok(((numel / o[o.len() - 1]) as u32, *o.last().unwrap() as u32));
        }
    }
    // 末轴均值（LN 分解链）：输出 [.., 1] → rc = (行数, 4)——不能继承
    // 输入的 rc（列数塌成 1，cpad4(1)=4）。
    if n.op_type == "ReduceMean" {
        if let Ok(axes) = planner::axes_from(n, &table.values) {
            let ish = sh(&n.inputs[0]);
            let r = ish.len() as i64;
            let norm: Vec<i64> = axes
                .iter()
                .map(|&a| if a < 0 { a + r } else { a })
                .collect();
            if norm.len() == 1 && norm[0] == r - 1 {
                let m: i64 = ish[..(r as usize - 1)].iter().product();
                return Ok((m as u32, 4));
            }
        }
    }
    // 其余：rank-4 按形状；rank-3/2 继承输入
    let o = sh(&n.outputs[0]);
    if o.len() == 4 {
        Ok(((o[0] * o[2] * o[3]) as u32, nhwc::cpad4(o[1])))
    } else if let Some(r) = rc.get(&n.inputs[0]).copied() {
        Ok(r)
    } else {
        Err(format!(
            "{} 输出 rank-{} 无 rc 来源（输入 {} 未记账）",
            n.op_type,
            o.len(),
            n.inputs[0]
        ))
    }
}

/// conv 族权重的惰性重排：首次引用时重排 + 静态占区 + 进上传列表。
/// 名字键控（共享权重只重排一份）。
fn n_weight_off(
    initializers: &HashMap<String, Tensor>,
    name: &str,
    layout: &mut Layout,
    uploads: &mut Vec<(u32, Vec<u32>)>,
    w_offs: &mut HashMap<String, u32>,
    repack_cache: &Mutex<HashMap<String, Arc<Vec<u32>>>>,
    repack: impl FnOnce(&Tensor) -> Vec<u32>,
) -> Result<u32> {
    if let Some(&o) = w_offs.get(name) {
        return Ok(o);
    }
    let t = initializers.get(name).ok_or_else(|| {
        Error::Graph(format!(
            "n_ 路径：权重 {name} 不是 initializer（仅支持常量权重）"
        ))
    })?;
    // 重排与形状无关：会话级按名缓存（首次算，重建 memcpy）
    let data = {
        let mut cache = repack_cache
            .lock()
            .map_err(|_| Error::Device("重排缓存锁中毒".into()))?;
        std::sync::Arc::clone(
            cache
                .entry(name.to_string())
                .or_insert_with(|| std::sync::Arc::new(repack(t))),
        )
    };
    let off = layout.alloc_static(data.len() as u32);
    uploads.push((off, (*data).to_vec()));
    w_offs.insert(name.to_string(), off);
    Ok(off)
}

/// 操作数字节定向推断：沿存储恒等别名链（Reshape/Squeeze/Unsqueeze 与
/// 满足别名条件的 Transpose[0,2,1]）回溯到物质化生产者。真转置/动态 B
/// MatMul/dim0 切片按**真线性**（逻辑形行主序）写字节；其余（conv/元素
/// 族/reduce/softmax）按 NHWC 族（rc 记账）。Some(true)=真线性，
/// Some(false)=NHWC 族，None=无生产者（图输入/权重）或不可判。
/// 背景：C==W 时真线性 rc 与 NHWC rc **数值相同**（[N,C,1,W] 互为方阵
/// 转置≠恒等），rc 对照无法消歧——必须按生产者判（T=120=C 时
/// 曾静默搅拌整图）。
fn linear_by_producer(
    graph: &Graph,
    node_idx: &HashMap<String, usize>,
    table: &planner::ShapeTable,
    rc: &HashMap<String, (u32, u32)>,
    initializers: &HashMap<String, Tensor>,
    name: &str,
) -> Option<bool> {
    let mut cur = name.to_string();
    let mut hops = 0usize;
    loop {
        let n = node_idx.get(&cur).map(|&i| &graph.nodes[i])?;
        hops += 1;
        if hops > graph.nodes.len() {
            return None; // 别名链成环（理论不可能）——按不可判处理
        }
        match n.op_type.as_str() {
            "Reshape" | "Squeeze" | "Unsqueeze" => {
                cur = n.inputs[0].clone();
            }
            "Transpose" => {
                let perm = n.attr("perm").map(|a| a.ints.clone()).unwrap_or_default();
                if perm != [0, 2, 1] {
                    return Some(true); // 真转置：真线性
                }
                // [0,2,1]：输入 rc 与输出逻辑一致时是别名——继续回溯
                let outn = table.values.get(&n.outputs[0]);
                if let (Some(&r), Some(o)) = (rc.get(&n.inputs[0]), outn) {
                    let last = o.shape.last().copied().unwrap_or(1).max(1);
                    let rows = o.shape.iter().product::<i64>() / last;
                    if r == (rows as u32, last as u32) {
                        cur = n.inputs[0].clone();
                        continue;
                    }
                }
                return Some(true);
            }
            "MatMul" => {
                // 动态 B（注意力）= 真线性；静态 [K,N] = (M, cpad4(N)) 族
                let b_const = n
                    .inputs
                    .get(1)
                    .map(|b| initializers.contains_key(b))
                    .unwrap_or(false);
                return Some(!b_const);
            }
            "Slice" => return Some(true), // dim0 切片别名（偏移不转向）
            _ => return Some(false),
        }
    }
}

/// 逐元素操作数的**存储定向中转**：操作数字节序与基准不一致时（长跳跃
/// 相遇：NHWC 卷积链张量 × 真转置产物，分类器全烂根因），把真线性一方
/// 用 n_transpose_nd（perm [0,2,3,1]）转成基准的 NHWC 族定向。判定按
/// 生产者定向（`linear_by_producer`）+ 形状序比较，**不看 rc 数值等价**
/// （C==W 时两序 rc 相同但字节互为方阵转置≠恒等，曾静默搅拌整图）。
/// 不可中转的组合显式报错（不静默搅拌）。仅支持 C%4==0：pad 通道的
/// 零填充中转不覆盖，宁可拒绝。
#[allow(clippy::too_many_arguments)]
fn orient4(
    layout: &mut Layout,
    pre: &mut Vec<(&'static str, ParamBlock, [u32; 3])>,
    off: u32,
    shape: &[i64],
    src_rc: (u32, u32),
    tgt_rc: (u32, u32),
    src_linear: Option<bool>,
    tgt_linear: Option<bool>,
) -> Result<u32> {
    let prod = shape.iter().product::<i64>();
    if shape.len() == 3 {
        // rank-3 的 rc 即字节序（无 NHWC/线性二义）：rc 对照足够。
        return if src_rc == tgt_rc {
            Ok(off)
        } else {
            Err(Error::Graph(format!(
                "rank-3 操作数 rc 不一致：{src_rc:?} vs {tgt_rc:?}（{shape:?}）"
            )))
        };
    }
    if shape.len() != 4 {
        return Err(Error::Graph(format!(
            "定向中转仅支持 rank-3/4，实得 {shape:?}"
        )));
    }
    let (_nn, cc, _hh, ww) = (shape[0], shape[1], shape[2], shape[3]);
    // 字节序比较（H==1 的 [N,C,1,W]）：真线性 (n,c,w) vs NHWC (n,w,c)
    // ——C==1 或 W==1 时两序坍缩相同，否则互为方阵转置、必须中转。
    let orders_equal = cc == 1 || ww == 1;
    match src_linear {
        None => {
            // 无生产者（initializer/图输入）：字节序按 rc 对照（旧语义）
            if src_rc == tgt_rc {
                Ok(off)
            } else {
                Err(Error::Graph(format!(
                    "逐元素操作数（常量）定向不可判：rc={src_rc:?} vs {tgt_rc:?}（{shape:?}）"
                )))
            }
        }
        Some(true) => {
            if orders_equal || tgt_linear == Some(true) {
                // 序本相同，或基准也是真线性（两方一致）
                Ok(off)
            } else if cc % 4 == 0 && tgt_linear == Some(false) {
                let total = (prod as u32).max(1);
                let tmp = layout.alloc(total);
                let mut pb = ParamBlock::new();
                pb.u(off).u(tmp).u(total).u(shape.len() as u32);
                for &d in shape {
                    pb.u(d as u32);
                }
                for _ in shape.len()..5 {
                    pb.u(1);
                }
                for &p in &[0i64, 2, 3, 1] {
                    pb.u(p as u32);
                }
                pre.push(("n_transpose_nd", pb, [total.div_ceil(256), 1, 1]));
                Ok(tmp)
            } else {
                Err(Error::Graph(format!(
                    "定向基准不可判（tgt={tgt_rc:?}，lin={tgt_linear:?}，shape={shape:?}）"
                )))
            }
        }
        Some(false) => {
            // NHWC 族操作数：与 NHWC 基准一致；基准若是真线性且序不同，
            // 反向中转未实现——显式拒绝。
            if tgt_linear == Some(false) || orders_equal || src_rc == tgt_rc {
                Ok(off)
            } else {
                Err(Error::Graph(format!(
                    "NHWC 操作数遇真线性基准（{src_rc:?} → {tgt_rc:?}）——反向中转未支持"
                )))
            }
        }
    }
}

/// 消费释放：输入若再无消费者，归还区域给布局复用。
/// **豁免**：图输出（要读回）、权重（dispatch 每次重放都读）、图输入
/// （每次 run 重写）——这三类区域一旦被复用就是数据 corrupt。
/// 区域大小按模式取：f32 元素数 / NHWC word 数（与分配一致，否则空闲
/// 表尺寸错位）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn release_inputs(
    n: &Node,
    live: &mut HashMap<String, u32>,
    offs: &mut HashMap<String, u32>,
    layout: &mut Layout,
    table: &planner::ShapeTable,
    is_output: &dyn Fn(&str) -> bool,
    initializers: &HashMap<String, Tensor>,
    input_name: &str,
    f32_mode: bool,
    rc_free: &HashMap<String, (u32, u32)>,
    dbg: bool,
) {
    for inn in &n.inputs {
        if inn.is_empty() {
            continue;
        }
        if is_output(inn) || initializers.contains_key(inn) || inn == input_name {
            continue;
        }
        if let Some(c) = live.get_mut(inn) {
            *c = c.saturating_sub(1);
            if dbg {
                eprintln!("[plan][live] {inn} 消费于 {}（剩 {}）", n.name, *c);
            }
            if *c == 0 && table.values.get(inn).map(|v| v.dtype) == Some(DType::F32) {
                if let Some(o) = offs.remove(inn) {
                    if dbg {
                        eprintln!("[plan][live] {inn} @{} 释放于 {}", o, n.name);
                    }
                    let size = match table.values.get(inn) {
                        Some(v) if !f32_mode && v.shape.len() == 4 => nhwc::nhwc_words(&v.shape),
                        Some(_) if !f32_mode => {
                            // rank-3 等：按记账的 (rows, cols)（MatMul 尾链）
                            rc_free.get(inn).copied().map(|(r, c)| r * c).unwrap_or(0)
                        }
                        _ => numel(table, inn),
                    };
                    if std::env::var_os("QPPOCR_GPU_NO_FREE").is_none() {
                        layout.free(o, size);
                    }
                }
            }
        }
    }
}

fn strides(n: &Node) -> (i64, i64) {
    let s = n
        .attr("strides")
        .map(|a| a.ints.clone())
        .unwrap_or_default();
    if s.len() == 2 { (s[0], s[1]) } else { (1, 1) }
}

fn pads4(n: &Node) -> (i64, i64, i64, i64) {
    let p = n.attr("pads").map(|a| a.ints.clone()).unwrap_or_default();
    if p.len() >= 4 {
        (p[0], p[1], p[2], p[3])
    } else {
        (0, 0, 0, 0)
    }
}
