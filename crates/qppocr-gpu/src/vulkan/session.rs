//! VulkanSession：图计划生成 + 会话执行（Phase 1-G2）。
//!
//! 装载期（create_session）：图过 planner 静态形状推理。输入形状是
//! **运行期**才定的——首次 run 按输入形状现建计划并缓存：op→内核
//! 映射 → 活性复用的区域布局（**单块**：KernelSet 绑单缓冲）→ 权重
//! 与参数块上传 → 整图 dispatch 录进一条可复用命令缓冲。
//!
//! 运行期（run）：按输入形状取计划 → memcpy 写入 → submit_wait_cb →
//! 读回输出。零逐帧重录、零描述符操作——每帧成本就是一次提交。
//!
//! 已知边界（诚实记录）：
//! - 计划缓存未封顶（形状多样性高时 arena 无界增长——LRU 在性能轮加）；
//! - SE 融合内核（fused_*）未接模式匹配，sigmoid/hardsigmoid 走独立
//!   dispatch + mul_c——正确性优先，融合在性能轮按 profile 上；
//! - 逐节点 dump 不支持（QPPOCR_DUMP_DIR 设置时 stderr 声明，不静默）。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ash::vk;
use qppocr_core::device::{DeviceKind, DeviceSession};
use qppocr_core::error::{Error, Result};
use qppocr_core::onnx::model::{Graph, Node};
use qppocr_core::tensor::{DType, Tensor};
use qppocr_kernels::buf::F32Buf;

use super::memory::Arena;
use super::pipeline::{
    KernelSet, OFF_NONE, ParamBlock, PcBinary, PcChannel, PcMulAddScale, PcUnary, PcUnaryF,
    record_dispatch,
};
use crate::planner;

/// 计划缓存的条目明细（调试/步进对拍用）：(内核, 节点名, 节点序号,
/// 输出偏移, 输出元素数, PC 字节, dispatch 网格, 输入明细, 输入形状,
/// 输出形状)。
type PlanRec = (
    String,
    String,
    usize,
    u32,
    u32,
    Vec<u8>,
    [u32; 3],
    Vec<(String, u32, u32)>,
    Vec<Vec<i64>>,
    Vec<i64>,
);

/// 一个输入形状对应的整图执行计划。
struct Plan {
    /// 整块基址（持久映射；存活期归 `_arena`）。
    base: *mut u8,
    /// 输入名 → 偏移（float 元素，相对 base）。
    in_offs: HashMap<String, u32>,
    /// 图输出：(名字, 偏移, 元素数)（按 graph.outputs 顺序）。
    out_offs: Vec<(String, u32, usize)>,
    out_shapes: Vec<Vec<i64>>,
    /// 录好的整图命令缓冲（可复用）。
    cb: vk::CommandBuffer,
    /// 逐 dispatch 明细（步进对拍调试用；正常路径不读）。
    #[allow(dead_code)]
    recs: Vec<PlanRec>,
    /// 剖析查询池（QPPOCR_GPU_PROF 时存在；索引 0 = 首个 dispatch 前，
    /// i = 第 i-1 个 dispatch 后）。录制在 CB 里，每次重放自复位。
    qpool: Option<vk::QueryPool>,
    /// 整块字节数（计划缓存的预算记账）。
    block_bytes: u64,
    /// 计划存活期归本计划；逐出时 take 归还会话 arena 池（见
    /// [`VulkanSession::arena_pool`]——免掉大块页提交）。
    arena: Option<Arena>,
    _ks: KernelSet,
    /// 查询池销毁用的设备句柄（drop 序：先于 ctx 释放）。
    _dev: ash::Device,
}

impl Drop for Plan {
    fn drop(&mut self) {
        if let Some(p) = self.qpool.take() {
            // SAFETY: 池由本计划独占；drop 时设备上无未完成工作（run
            // 等完信号才返回）；_dev/ctx 仍存活（本体在字段 drop 前执行）。
            unsafe { self._dev.destroy_query_pool(p, None) };
        }
    }
}

// SAFETY: base 是 _arena 持久映射的裸指针，存活期与 Plan 相同且无并发
// 写（run 全程持锁）；句柄均为 Vulkan 整数。跨线程只是把 Plan 放进
// Mutex 保护的缓存。
unsafe impl Send for Plan {}
// SAFETY: 同上——&Plan 的共享只发生在持锁的 run 里。
unsafe impl Sync for Plan {}

/// 区域布局：活性复用的 bump + first-fit 空闲表（全部相对同一整块）。
struct Layout {
    free: Vec<(u32, u32)>,
    cursor: u32,
    total: u32,
}

impl Layout {
    fn new() -> Self {
        Self {
            free: Vec::new(),
            cursor: 0,
            total: 0,
        }
    }

    fn alloc(&mut self, n: u32) -> u32 {
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

    fn free(&mut self, off: u32, n: u32) {
        let n = n.div_ceil(4) * 4;
        if n > 0 {
            self.free.push((off, n));
        }
    }

    /// 静态区（权重/参数）专用：**只 bump、永不复用**。
    ///
    /// 激活区域可以 free-复用——生产者 dispatch 先写、消费者后读，
    /// 单次重放内自洽。但静态内容只在装载期写一次，而 CB 每帧重放
    /// 时**复用区的原生产者 dispatch 仍会写那片内存**——参数落进
    /// 复用区 = 运行时被冲掉、消费方读到垃圾参数（本 bug 的实测形态：
    /// conv 参数损坏 → 巨循环 → GPU 挂死设备丢失）。
    fn alloc_static(&mut self, n: u32) -> u32 {
        let off = self.cursor;
        self.cursor += n.div_ceil(4) * 4;
        self.total = self.cursor;
        off
    }
}

fn consumer_counts(graph: &Graph) -> HashMap<String, u32> {
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

fn numel(table: &planner::ShapeTable, name: &str) -> u32 {
    table
        .values
        .get(name)
        .map(|v| v.shape.iter().product::<i64>() as u32)
        .unwrap_or(0)
}

/// planner 形状表里取形状（缺项给空——调用方容错）。
fn shape_of(table: &planner::ShapeTable, name: &str) -> Vec<i64> {
    table
        .values
        .get(name)
        .map(|v| v.shape.clone())
        .unwrap_or_default()
}

/// GPU 会话：设备共享句柄 + 形状键计划缓存。
pub(crate) struct VulkanSession {
    graph: Graph,
    initializers: HashMap<String, Tensor>,
    input_name: String,
    plans: Mutex<Vec<(Vec<i64>, Plan)>>,
    /// 会话级管线缓存：按形状重建计划时驱动侧复用编译产物
    ///（形状多样性 × LRU 驱逐会让每帧都重建，38 条管线重建 ~5ms）。
    pipeline_cache: vk::PipelineCache,
    /// 会话级权重重排缓存：重排算术（83 层 conv 的 k-major 循环 ~9ms）
    /// 与形状无关——按名缓存，重建只付 memcpy（~1ms）。
    repack_cache: Mutex<HashMap<String, std::sync::Arc<Vec<u32>>>>,
    /// 会话级 arena 池：逐出计划归还整块，新形状 reset 复用（容量够时）。
    /// 137MB 新块页提交 ~7ms vs 复用 ~0——形状多样性 × LRU 驱逐下每帧
    /// 重建曾把它当固定税。池封顶 2 块（按容量优先留大）。
    arena_pool: Mutex<Vec<Arena>>,
    /// 持有整个上下文。**必须声明在最后**：字段按声明序 drop——plans
    /// 里的管线/缓冲销毁要用设备，而设备销毁发生在 ctx（Arc<Inner>）
    /// 的最后一个引用释放时。ctx 在前 = 先毁设备再毁管线 = 对已销毁
    /// 设备调用 destroy（实测 CLI 路径 100% 段错误；测试路径因 ctx
    /// 局部变量活得比会话久而侥幸通过）。
    ctx: Arc<super::Inner>,
}

impl VulkanSession {
    pub(crate) fn new(
        ctx: Arc<super::Inner>,
        graph: Graph,
        initializers: HashMap<String, Tensor>,
    ) -> Result<Self> {
        if std::env::var_os("QPPOCR_DUMP_DIR").is_some() {
            eprintln!(
                "[gpu] QPPOCR_DUMP_DIR 已设置：GPU 会话不落盘逐节点 dump（融合调度下\
                 无逐节点中间量）；节点级诊断请用 --device cpu"
            );
        }
        let input_name = graph
            .inputs
            .iter()
            .find(|s| !s.is_empty())
            .cloned()
            .unwrap_or_default();
        // SAFETY: 空初始数据；缓存归本会话（Drop 销毁），创建先于任何管线。
        let pipeline_cache = unsafe {
            ctx.device
                .raw()
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
        }
        .map_err(|e| Error::Device(format!("建管线缓存失败: {e}")))?;
        Ok(Self {
            graph,
            initializers,
            input_name,
            plans: Mutex::new(Vec::new()),
            pipeline_cache,
            repack_cache: Mutex::new(HashMap::new()),
            arena_pool: Mutex::new(Vec::new()),
            ctx,
        })
    }

    /// 调试：给定形状返回 (整块基址, [(内核, 节点, 输出偏移, 输出元素数)])。
    /// 测试读中间量统计用（n_ 模式的区域是 f16 NHWC）。
    #[allow(clippy::type_complexity)]
    pub(crate) fn debug_recs(
        &self,
        in_shape: &[i64],
    ) -> Option<(
        *mut u8,
        Vec<(String, String, usize, u32, u32, Vec<u8>, Vec<i64>)>,
    )> {
        let plans = self.plans.lock().ok()?;
        let p = plans.iter().find(|(s, _)| s == in_shape).map(|(_, p)| p)?;
        Some((
            p.base,
            p.recs
                .iter()
                .map(|(k, n, idx, o, on, pc, _, _, _, osh)| {
                    (k.clone(), n.clone(), *idx, *o, *on, pc.clone(), osh.clone())
                })
                .collect(),
        ))
    }

    /// 调试：重放 recs[from..=to]（之前可先 mutate 整块基址），
    /// 返回 (基址, 末 rec 的输出偏移/元素数)。现场取证用。
    /// （旧单 rec 形态等价 from==to。）
    pub(crate) fn debug_replay(
        &self,
        in_shape: &[i64],
        rec_idx: usize,
        mutate: Option<&dyn Fn(*mut u8)>,
    ) -> Result<(*mut u8, u32, u32)> {
        self.debug_replay_range(in_shape, rec_idx, rec_idx, mutate)
    }

    pub(crate) fn debug_replay_range(
        &self,
        in_shape: &[i64],
        from: usize,
        to: usize,
        mutate: Option<&dyn Fn(*mut u8)>,
    ) -> Result<(*mut u8, u32, u32)> {
        let mut plans = self
            .plans
            .lock()
            .map_err(|_| Error::Device("计划缓存锁中毒".into()))?;
        let plan = plans
            .iter_mut()
            .find(|(s, _)| s == in_shape)
            .map(|(_, p)| p)
            .ok_or_else(|| Error::Device("debug_replay：无该形状计划".into()))?;
        if let Some(f) = mutate {
            f(plan.base);
        }
        // 录制区间 → 一条 CB → 提交等完（锁内：与 run 同一串行化域）
        let mut recs_out: Vec<(String, Vec<u8>, [u32; 3])> = Vec::new();
        for r in &plan.recs[from..=to] {
            recs_out.push((r.0.clone(), r.5.clone(), r.6));
        }
        let (last_off, last_n) = (plan.recs[to].3, plan.recs[to].4);
        let base = plan.base;
        let ks = &plan._ks;
        let mut idx = from;
        self.ctx.device.submit_one_shot(|d, cb| {
            for (kernel, pc, groups) in &recs_out {
                // SAFETY: cb 录制态；PC 与该 rec 构建期参数一致。
                unsafe {
                    super::pipeline::record_dispatch(d, cb, ks, kernel, pc, *groups);
                }
                idx += 1;
                let _ = idx;
            }
        })?;
        Ok((base, last_off, last_n))
    }

    /// 调试：图输入名。
    pub(crate) fn input_name_for_test(&self) -> String {
        self.input_name.clone()
    }

    /// 步进执行 + 逐节点对拍（输入已写入；QPPOCR_GPU_DUMP_CMP 给出
    /// CPU dump 目录）。首个分歧节点带两侧首元素值报错。
    fn stepped_run_and_cmp(&self, plan: &Plan) -> Result<()> {
        let dump_dir = std::env::var_os("QPPOCR_GPU_DUMP_CMP");
        for (i, (kernel, node, idx, out_off, out_n, pc, groups, ins, _shapes, _osh)) in
            plan.recs.iter().enumerate()
        {
            // 输入复查：每个输入区域当前值 vs 它生产时的 dump——
            // 不匹配 = 该区域被复用写花（liveness bug 的直接证据）。
            if let Some(dir) = &dump_dir {
                for (inm, ioff, in_n) in ins {
                    let Some(prod_idx) = node_index_of(&self.graph, inm) else {
                        continue;
                    };
                    let path = std::path::Path::new(dir).join(format!("s0_{prod_idx:06}.f32"));
                    let Ok(bytes) = std::fs::read(&path) else {
                        continue;
                    };
                    let rank = rank_of(&bytes);
                    let want: Vec<f32> = bytes[4 + 8 * rank..]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    // SAFETY: base 持久映射；ioff+in_n 在 total 内。
                    let got: Vec<f32> = unsafe {
                        std::slice::from_raw_parts(
                            plan.base.add(*ioff as usize * 4) as *const f32,
                            *in_n as usize,
                        )
                    }
                    .to_vec();
                    if got.len() == want.len() {
                        let e = got
                            .iter()
                            .zip(&want)
                            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
                            .fold(0.0f32, f32::max);
                        if e > 1e-3 {
                            return Err(Error::Graph(format!(
                                "[step] 节点 {node} 的输入 {inm} 在 dispatch 前已被改写：                                 rel|diff|={e:.3e}（区域被复用/覆盖——liveness 或参数错位）"
                            )));
                        }
                    }
                }
            }
            let ks = &plan._ks;
            let pc = pc.clone();
            let kernel = kernel.clone();
            let groups = *groups;
            self.ctx.device.submit_one_shot(|d, cb| {
                // SAFETY: cb 录制态；PC 与内核参数块对应（build 期生成）。
                unsafe {
                    record_dispatch(d, cb, ks, &kernel, &pc, groups);
                }
            })?;
            // 逐层 f16 统计：找数值死亡层（QPPOCR_GPU_STEP_STATS 时）
            if std::env::var_os("QPPOCR_GPU_STEP_STATS").is_some() && *out_n > 0 {
                let nf = *out_n as usize; // 全量扫描（死亡层定位）
                let stats = if kernel == "n_exit" {
                    // SAFETY: base 持久映射；out_off+n ≤ total。
                    let v: Vec<f32> = unsafe {
                        std::slice::from_raw_parts(
                            plan.base.add(*out_off as usize * 4) as *const f32,
                            nf,
                        )
                    }
                    .to_vec();
                    (
                        v.iter().map(|x| x.abs()).fold(0.0f32, f32::max),
                        v.iter().sum::<f32>() / v.len() as f32,
                        v.iter().filter(|x| x.is_nan()).count(),
                    )
                } else {
                    // SAFETY: 同上。
                    let w: Vec<u32> = unsafe {
                        std::slice::from_raw_parts(
                            plan.base.add(*out_off as usize * 4) as *const u32,
                            nf.div_ceil(2),
                        )
                    }
                    .to_vec();
                    let v = super::fp16::f16_words_to_f32(&w);
                    (
                        v.iter().map(|x| x.abs()).fold(0.0f32, f32::max),
                        v.iter().sum::<f32>() / v.len() as f32,
                        v.iter().filter(|x| x.is_nan()).count(),
                    )
                };
                eprintln!(
                    "[gpu][层] #{i:<3} {kernel:<12} {node:<26} absmax={:.4} mean={:.6} nan={}",
                    stats.0, stats.1, stats.2
                );
            }
            eprintln!("[gpu][step] {i}/{} {} {} ok", plan.recs.len(), kernel, node);
            if let Some(dir) = &dump_dir {
                let path = std::path::Path::new(dir).join(format!("s0_{idx:06}.f32"));
                if let Ok(bytes) = std::fs::read(&path) {
                    let rank = rank_of(&bytes);
                    let want: Vec<f32> = bytes[4 + 8 * rank..]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    // SAFETY: base 持久映射；out_off+out_n 在 total 内。
                    let got: Vec<f32> = unsafe {
                        std::slice::from_raw_parts(
                            plan.base.add(*out_off as usize * 4) as *const f32,
                            *out_n as usize,
                        )
                    }
                    .to_vec();
                    if got.len() == want.len() {
                        let e = got
                            .iter()
                            .zip(&want)
                            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
                            .fold(0.0f32, f32::max);
                        if e > 0.3 {
                            let bad: Vec<String> = got
                                .iter()
                                .zip(&want)
                                .enumerate()
                                .filter(|(_, (a, b))| (**a - **b).abs() / (1.0 + (*b).abs()) > 0.3)
                                .take(3)
                                .map(|(i, (a, b))| format!("[{i}] gpu={a:.6} cpu={b:.6}"))
                                .collect();
                            let n = got.len();
                            return Err(Error::Graph(format!(
                                "[step] 首个分歧节点 {node}（{kernel}，idx={idx}，{n} 元素）：                                 rel|diff|={e:.3e}，{}",
                                bad.join(" ")
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn build_plan(&self, in_shape: &[i64]) -> Result<Plan> {
        let t_build = std::time::Instant::now();
        let t_shape0 = std::time::Instant::now();
        let table = planner::infer_shapes(
            &self.graph,
            &self.initializers,
            &[(self.input_name.clone(), in_shape.to_vec(), DType::F32)],
        )?;

        let dbg2 = std::env::var_os("QPPOCR_GPU_BUILD_TIME").is_some();
        if dbg2 {
            eprintln!(
                "[gpu][dbg] infer_shapes {:.1} ms",
                t_shape0.elapsed().as_secs_f64() * 1000.0
            );
        }
        let t_node0 = std::time::Instant::now();
        let mut t_map_acc = 0.0f64;
        let refs = consumer_counts(&self.graph);
        if std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some() {
            if let Some(n0) = self.graph.nodes.first() {
                let o0 = &n0.outputs[0];
                eprintln!("[gpu][live] {} 的消费计数 = {:?}", o0, refs.get(o0));
            }
        }
        let mut live: HashMap<String, u32> = refs.clone();

        // 路径选择：默认 = n_ 内核族（NHWC f32 + k-major 权重；真实图与
        // 旧路径逐位一致 2e-8、GPU 总时间 24.7ms vs 旧路径 77ms）。
        // QPPOCR_GPU_F32=1 回退旧 NCHW f32 路径（A/B 与兜底）。
        let f32_mode = std::env::var_os("QPPOCR_GPU_F32").is_some();

        let mut layout = Layout::new();
        let mut ledger: Vec<(String, u32, u32)> = Vec::new(); // (名, off, len)
        let mut offs: HashMap<String, u32> = HashMap::new();
        // 权重占区（F32 且被引用；i64 常量链由 planner 折叠，不上传）。
        // 权重**永不释放**：命令缓冲每次重放都读它，区域复用=数据被覆盖。
        // 统一 word 上传：f32 数据按位转 u32。
        let mut upload: Vec<(u32, Vec<u32>)> = Vec::new();
        // conv 族权重（重排目标）名单：预环跳过，节点处理时惰性重排。
        let mut conv_w_names: std::collections::HashSet<String> = std::collections::HashSet::new();
        if !f32_mode {
            for n in &self.graph.nodes {
                if n.op_type == "Conv" || n.op_type == "ConvTranspose" {
                    for w in n.inputs.iter().skip(1).take(2) {
                        if !w.is_empty() {
                            conv_w_names.insert(w.clone());
                        }
                    }
                }
            }
        }
        // **确定序**（按名排序）：HashMap 迭代序逐进程随机——区域布局会跟着
        // 漂移，任何依赖布局的越界/重叠 bug 都会以「两次运行分歧层不同」
        // 的形态出现（实测发生过）。排序后布局恒定，分歧可稳定复现定位。
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
            let words: Vec<u32> = if f32_mode {
                t.f32.iter().map(|f| f.to_bits()).collect()
            } else {
                super::nhwc::init_to_nhwc(t)
            };
            let off = layout.alloc_static(words.len() as u32);
            ledger.push((format!("w:{name}"), off, words.len() as u32));
            offs.insert(name.clone(), off);
            upload.push((off, words));
        }

        struct Rec {
            kernel: &'static str,
            node: String,
            node_idx: usize,
            out_off: u32,
            out_n: u32,
            pc: Vec<u8>,
            groups: [u32; 3],
            ins: Vec<(String, u32, u32)>, // (输入名, off, numel)
            in_shapes: Vec<Vec<i64>>,
            out_shape: Vec<i64>,
        }
        let mut recs: Vec<Rec> = Vec::new();
        let mut params: Vec<(u32, Vec<u32>)> = Vec::new();
        let is_output = |nm: &str| self.graph.is_graph_output(nm);

        // 图输入占区：f32 区每 run 重写（host memcpy）；n_ 模式另有
        // entry 转换出的 NHWC-f16 区，节点一律引用后者。两区都不释放。
        let in_words = in_shape.iter().product::<i64>() as u32;
        let in_f32_off = layout.alloc(in_words);
        if f32_mode {
            offs.insert(self.input_name.clone(), in_f32_off);
        } else {
            let in_f16_off = layout.alloc(super::nhwc::nhwc_words(in_shape));
            offs.insert(self.input_name.clone(), in_f16_off);
            // entry 转换 dispatch（整图第一条）
            let (nb, c, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
            let mut pb = ParamBlock::new();
            pb.u(in_f32_off)
                .u(in_f16_off)
                .u(nb as u32)
                .u(c as u32)
                .u(h as u32)
                .u(w as u32)
                .u(super::nhwc::cpad4(c));
            let p_off = layout.alloc_static(pb.len_words());
            ledger.push(("p:<entry>".into(), p_off, pb.len_words()));
            params.push((p_off, pb.words().to_vec()));
            recs.push(Rec {
                kernel: "n_entry",
                node: "<entry>".into(),
                node_idx: usize::MAX,
                out_off: in_f16_off,
                out_n: in_words,
                pc: super::pipeline::PcParams { p_off }.bytes().to_vec(),
                groups: [((nb * h * w) as u32).div_ceil(256), 1, 1],
                ins: vec![(self.input_name.clone(), in_f32_off, in_words)],
                in_shapes: vec![in_shape.to_vec()],
                out_shape: in_shape.to_vec(),
            });
        }

        // SE 融合预扫：HardSigmoid/Sigmoid(gate[1,C,1,1]) 的唯一消费者是
        // Mul(feature, gate_out) 时，合为 fused_*_mul（省一整趟 NCHW 读写
        // + 独立门 dispatch）。融合掉的节点记入 skip 集。
        // n_ 模式默认开（n_channel 内核原生支持）；QPPOCR_GPU_NO_SE 关。
        let mut fused = std::collections::HashSet::new(); // 被融合掉的节点名
        let se_fusion_on = if f32_mode {
            std::env::var_os("QPPOCR_GPU_SE_FUSION").is_some()
        } else {
            std::env::var_os("QPPOCR_GPU_NO_SE").is_none()
        };
        // n_ 模式：conv 族权重的惰性重排区（名 → 偏移，防共享权重重排两份）
        let mut w_offs: HashMap<String, u32> = HashMap::new();
        // n_ 模式：SE 归约的两阶段共享 scratch（f32 words，按最大块需求开一份）
        let reduce_scratch: u32 = if f32_mode {
            0
        } else {
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
                    if xs.len() == 4 && xs[0] == 1 {
                        let m_blocks = (xs[2] * xs[3]) as u32;
                        let m_blocks = m_blocks.div_ceil(1024);
                        need = need.max(m_blocks * super::nhwc::cpad4(xs[1]));
                    }
                }
            }
            if need > 0 {
                let off = layout.alloc_static(need);
                ledger.push(("w:<reduce-scratch>".into(), off, need));
                off
            } else {
                0
            }
        };
        // n_ 模式：exit 转换列表 (f16 源名, 图输出名, act)——节点环后统一发
        let mut pending_exits: Vec<(String, String, u32)> = Vec::new();
        for i in 0..self.graph.nodes.len() {
            let n = &self.graph.nodes[i];
            if n.op_type != "HardSigmoid" && n.op_type != "Sigmoid" {
                continue;
            }
            let in_v = table.values.get(&n.inputs[0]);
            if in_v.map(|v| v.shape.len() != 4 || v.shape[2] != 1 || v.shape[3] != 1) != Some(false)
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
            // 融合：HardSigmoid/Sigmoid 节点标记为跳过
            if se_fusion_on {
                fused.insert(n.name.clone());
                eprintln!("[gpu] SE 融合: {} → {} (fused)", n.name, mul.name);
            }
        }

        for (node_idx, n) in self.graph.nodes.iter().enumerate() {
            let out = n.outputs[0].clone();
            // 被融合的节点：跳过（其消费者 Mul 用 fused_* 内核替代）。
            // 不清输入——Mul 的 fused 内核直接读 HardSigmoid 的**输入**
            //（前激活门值），该张量仍需存活。
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
                    &self.initializers,
                    &self.input_name,
                    f32_mode,
                );
                continue;
            }
            // n_ 模式：图输出 Sigmoid 在 exit 融合（省一整趟 f16 读写）。
            // 其输入区域不释放（无消费者递减）——正确，常驻到计划结束。
            if !f32_mode
                && n.op_type == "Sigmoid"
                && is_output(&out)
                && offs.contains_key(&n.inputs[0])
            {
                pending_exits.push((n.inputs[0].clone(), out.clone(), 4));
                continue;
            }
            // 输出区域先占（map_node 的参数要引用它）。
            // n_ 模式按 NHWC-f16 word 数（含 Cpad4 padding）。
            let out_n = numel(&table, &out);
            let out_words = if f32_mode {
                out_n
            } else {
                super::nhwc::nhwc_words(&out_v.shape)
            };
            let out_off = layout.alloc(out_words);
            if std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some() {
                eprintln!(
                    "[gpu][live] {out} @{} 分配于 {}（len {out_words}）",
                    out_off, n.name
                );
            }
            ledger.push((format!("n:{}", n.name), out_off, out_words));
            offs.insert(out.clone(), out_off);

            // 节点 → 内核路由：n_ 路径可能展开多条 rec（reduce 两阶段），
            // f32 路径恒单条。
            let t_map = std::time::Instant::now();
            #[allow(clippy::type_complexity)]
            let routes: Vec<(&'static str, Option<ParamBlock>, Vec<u8>, [u32; 3])> = if f32_mode {
                let (k, pb, pc_inline, groups) = self.map_node(n, &table, &offs, &fused)?;
                vec![(k, pb, pc_inline, groups)]
            } else {
                self.map_node_n(
                    n,
                    &table,
                    &offs,
                    &fused,
                    &mut layout,
                    &mut upload,
                    &mut w_offs,
                    reduce_scratch,
                )?
                .into_iter()
                .map(|(k, pb, g)| (k, Some(pb), Vec::new(), g))
                .collect()
            };
            t_map_acc += t_map.elapsed().as_secs_f64();
            for (kernel, pb, pc_inline, groups) in routes {
                if std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some() && n.name == "Conv.5" {
                    eprintln!(
                        "[gpu][dbg] {} 输入形状: {:?}",
                        n.name,
                        n.inputs
                            .iter()
                            .map(|i| table.values.get(i).map(|v| v.shape.clone()))
                            .collect::<Vec<_>>()
                    );
                    eprintln!(
                        "[gpu][dbg] 输出形状: {:?} 偏移={:?}",
                        out_v.shape,
                        offs.get(&out)
                    );
                    if let Some(pb) = &pb {
                        eprintln!("[gpu][dbg] 参数块 words = {:?}", pb.words());
                    }
                    eprintln!(
                        "[gpu][dbg] 各输入偏移: {:?}",
                        n.inputs
                            .iter()
                            .map(|i| offs.get(i).copied())
                            .collect::<Vec<_>>()
                    );
                    eprintln!("[gpu][dbg] layout.total = {}", layout.total);
                }
                let pc: Vec<u8> = if let Some(pb) = pb {
                    // 参数块与数据同块（u32 视图）；静态内容，装载期写一次
                    let p_off = layout.alloc_static(pb.len_words());
                    ledger.push((format!("p:{}", n.name), p_off, pb.len_words()));
                    params.push((p_off, pb.words().to_vec()));
                    super::pipeline::PcParams { p_off }.bytes().to_vec()
                } else {
                    pc_inline
                };
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
                    pc,
                    groups,
                    ins,
                    in_shapes,
                    out_shape: out_v.shape.clone(),
                });
            } // routes 循环尾（release 按节点一次，不按 rec）

            release_inputs(
                n,
                &mut live,
                &mut offs,
                &mut layout,
                &table,
                &is_output,
                &self.initializers,
                &self.input_name,
                f32_mode,
            );
        }

        if dbg2 {
            eprintln!(
                "[gpu][dbg] 节点环 {:.1} ms（其中 map_node_n {:.1}）",
                t_node0.elapsed().as_secs_f64() * 1000.0,
                t_map_acc * 1000.0
            );
        }
        // n_ 模式：exit 转换（每个图输出一条；sigmoid 融合的在节点环里挂单）。
        // 输出区按图输出顺序分配 f32 NCHW word 数——out_offs 指它（host 读回）。
        if !f32_mode {
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
                let (nb, c, h, w) = (v.shape[0], v.shape[1], v.shape[2], v.shape[3]);
                let out_words = numel(&table, &name);
                let out_off = layout.alloc(out_words);
                let mut pb = ParamBlock::new();
                pb.u(src_off)
                    .u(out_off)
                    .u(nb as u32)
                    .u(c as u32)
                    .u(h as u32)
                    .u(w as u32)
                    .u(super::nhwc::cpad4(c))
                    .u(act);
                let p_off = layout.alloc_static(pb.len_words());
                ledger.push((format!("p:<exit:{name}>"), p_off, pb.len_words()));
                params.push((p_off, pb.words().to_vec()));
                recs.push(Rec {
                    kernel: "n_exit",
                    node: format!("<exit:{name}>"),
                    node_idx: usize::MAX,
                    out_off,
                    out_n: out_words,
                    pc: super::pipeline::PcParams { p_off }.bytes().to_vec(),
                    groups: [((nb * h * w) as u32).div_ceil(256), 1, 1],
                    ins: vec![(src_name.clone(), src_off, out_words)],
                    in_shapes: vec![v.shape.clone()],
                    out_shape: v.shape.clone(),
                });
                // out_offs 指向 f32 区（复用下方统一构造：先记到 offs 供其读取）
                offs.insert(name.clone(), out_off);
            }
        }

        // 对账（调试模式）：静态区（w:/p:）两两不重叠、不与图输出重叠。
        // 激活区允许复用（合法性由「生产者先写消费者后读」保证），故
        // 只查静态区。alloc_static 的 bump-only 构造本身就排除了静态区
        // 落入复用区——这段是防御性验证。
        if std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some() {
            let mut statics: Vec<(String, u32, u32)> = ledger
                .iter()
                .filter(|(nm, _, _)| nm.starts_with("w:") || nm.starts_with("p:"))
                .cloned()
                .collect();
            for o in self.graph.outputs.iter() {
                if let Some(off) = offs.get(o) {
                    statics.push((format!("OUT:{o}"), *off, numel(&table, o)));
                }
            }
            statics.sort_by_key(|(_, o, _)| *o);
            for w in statics.windows(2) {
                let (n1, o1, l1) = &w[0];
                let (n2, o2, _) = &w[1];
                assert!(
                    o1 + l1 <= *o2,
                    "静态区重叠: {n1} [{o1},{}) vs {n2} @{o2}",
                    o1 + l1
                );
            }
        }
        // === 实分配：整块一次（KernelSet 绑单缓冲）===
        let t_alloc = std::time::Instant::now();
        let dbg_time = std::env::var_os("QPPOCR_GPU_BUILD_TIME").is_some();
        let need = layout.total as vk::DeviceSize * 4;
        // 池里找容量够的块 reset 复用；没有再新建
        let mut arena = {
            let mut pool = self
                .arena_pool
                .lock()
                .map_err(|_| Error::Device("arena 池锁中毒".into()))?;
            let mut hit = None;
            for i in 0..pool.len() {
                if pool[i].reset_if_fits(need) {
                    hit = Some(i);
                    break;
                }
            }
            match hit {
                Some(i) => pool.swap_remove(i),
                None => Arena::new(
                    self.ctx.device.raw().clone(),
                    self.ctx.mem_types.staging,
                    true,
                ),
            }
        };
        let whole = arena.alloc(need).map_err(|e| {
            Error::Device(format!("计划整块分配失败（{} floats）: {e}", layout.total))
        })?;
        let (buf, buf_size) = arena
            .chunk_range()
            .ok_or_else(|| Error::Device("arena 空".into()))?;
        let ks =
            KernelSet::new_with_cache(self.ctx.device.raw(), buf, buf_size, self.pipeline_cache)?;

        if dbg_time {
            eprintln!(
                "[gpu][dbg] arena+KernelSet {:.1} ms",
                t_alloc.elapsed().as_secs_f64() * 1000.0
            );
        }
        let t_up = std::time::Instant::now();
        // 权重 + 参数块上传（coherent 映射直写；统一 u32 word 视图）
        // 越界自查：主机侧写越界映射内存 = 打死设备（本机实测形态）。
        for (off, data) in upload.iter().chain(params.iter()) {
            assert!(
                off + data.len() as u32 <= layout.total,
                "静态上传越界: {off}+{} > {}",
                data.len(),
                layout.total
            );
        }
        // SAFETY: whole 是整块持久映射；off+len ≤ total。
        unsafe {
            for (off, data) in &upload {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    whole.ptr.add(*off as usize * 4) as *mut u32,
                    data.len(),
                );
            }
            for (p_off, words) in &params {
                std::ptr::copy_nonoverlapping(
                    words.as_ptr(),
                    whole.ptr.add(*p_off as usize * 4) as *mut u32,
                    words.len(),
                );
            }
        }

        if dbg_time {
            eprintln!(
                "[gpu][dbg] 上传 {:.1} ms",
                t_up.elapsed().as_secs_f64() * 1000.0
            );
        }
        let t_rec = std::time::Instant::now();
        // 录制整图 dispatch 序列
        let dev = self.ctx.device.raw().clone();
        let cb = self.ctx.device.alloc_reusable_cb()?;
        // 剖析模式（QPPOCR_GPU_PROF=1|full）：每 dispatch 后一枚时间戳。
        let (ts_bits, _) = self.ctx.device.timestamps();
        let qpool = if std::env::var_os("QPPOCR_GPU_PROF").is_some() && ts_bits > 0 {
            let ci = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(recs.len() as u32 + 1);
            // SAFETY: ci 合法；池归本计划（Drop 销毁）。
            Some(
                unsafe { dev.create_query_pool(&ci, None) }
                    .map_err(|e| Error::Device(format!("建时间戳查询池失败: {e}")))?,
            )
        } else {
            None
        };
        // SAFETY: cb 处于录制态；查询池刚创建且未被并发使用（每次重放
        // 先复位——查询在重放里被重写）。
        unsafe {
            if let Some(p) = qpool {
                dev.cmd_reset_query_pool(cb, p, 0, recs.len() as u32 + 1);
                dev.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, p, 0);
            }
        }
        for (i, r) in recs.iter().enumerate() {
            // SAFETY: cb 处于录制态；PC 与内核参数块逐字段对应（map_node 保证）。
            unsafe {
                record_dispatch(&dev, cb, &ks, r.kernel, &r.pc, r.groups);
            }
            if let Some(p) = qpool {
                // SAFETY: 同上；BOTTOM_OF_PIPE = 前序工作全部完成之时。
                unsafe {
                    dev.cmd_write_timestamp(
                        cb,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        p,
                        i as u32 + 1,
                    );
                }
            }
        }
        self.ctx.device.end_reusable_cb(cb)?;

        if dbg_time {
            eprintln!(
                "[gpu][dbg] CB 录制 {:.1} ms",
                t_rec.elapsed().as_secs_f64() * 1000.0
            );
        }
        let out_offs: Vec<(String, u32, usize)> = self
            .graph
            .outputs
            .iter()
            .map(|o| {
                let off = offs.get(o).copied().unwrap_or(0);
                (o.clone(), off, numel(&table, o) as usize)
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
                    .map(|v| v.shape.clone())
                    .unwrap_or_default()
            })
            .collect();

        eprintln!(
            "[gpu] 计划就绪：输入 {in_shape:?}，{} 个 dispatch，整块 {:.1} MB（建 {:.1} ms）",
            recs.len(),
            layout.total as f64 * 4.0 / 1e6,
            t_build.elapsed().as_secs_f64() * 1000.0
        );
        let dbg_recs: Vec<PlanRec> = recs
            .iter()
            .map(|r| {
                (
                    r.kernel.to_string(),
                    r.node.clone(),
                    r.node_idx,
                    r.out_off,
                    r.out_n,
                    r.pc.clone(),
                    r.groups,
                    r.ins.clone(),
                    r.in_shapes.clone(),
                    r.out_shape.clone(),
                )
            })
            .collect();
        Ok(Plan {
            base: whole.ptr,
            block_bytes: layout.total as u64 * 4,
            arena: Some(arena),
            recs: dbg_recs,
            // in_offs 指向 **f32 区**（host 每次 memcpy 的目标）。n_ 模式下
            // offs[input] 是 entry 转换出的 f16 区——拿它当 memcpy 目标会把
            // 原始 f32 字节灌进 f16 激活/参数区（曾碾碎参数块打死设备）。
            in_offs: [(self.input_name.clone(), in_f32_off)]
                .into_iter()
                .collect(),
            out_offs,
            out_shapes,
            cb,
            qpool,
            _ks: ks,
            _dev: dev.clone(),
        })
    }

    /// 逐 dispatch GPU 时间剖析（QPPOCR_GPU_PROF=1 | full）。
    ///
    /// 时间戳读回后按 dispatch 取相邻差：第 i 段 = 第 i 个 dispatch 的
    /// 执行 + 其前的屏障/排队开销。`=1` 打印按内核聚合 + top-20；
    /// `=full` 额外逐 dispatch 列表。附 GPU 总时间 vs 提交往返的墙钟
    /// （差值 = 主机等待/提交开销）。
    fn prof_print(&self, plan: &Plan, pool: vk::QueryPool, wall_ms: f64) {
        let dev = self.ctx.device.raw();
        let n = plan.recs.len();
        let mut stamps = vec![0u64; n + 1];
        // SAFETY: pool 归本计划且提交已等完信号；data 切片长度即查询数。
        let ok = unsafe {
            dev.get_query_pool_results(pool, 0, &mut stamps, vk::QueryResultFlags::TYPE_64)
        }
        .is_ok();
        if !ok {
            eprintln!("[gpu][prof] 时间戳查询不可用（查询未完成？）");
            return;
        }
        let (_, period) = self.ctx.device.timestamps();
        let ms = |d: u64| d as f64 * period as f64 / 1e6;
        let deltas: Vec<f64> = (0..n)
            .map(|i| ms(stamps[i + 1].saturating_sub(stamps[i])))
            .collect();
        let total: f64 = deltas.iter().sum();

        // 按内核聚合
        let mut by_kernel: HashMap<&str, (usize, f64)> = HashMap::new();
        for (r, d) in plan.recs.iter().zip(&deltas) {
            let e = by_kernel.entry(r.0.as_str()).or_default();
            e.0 += 1;
            e.1 += *d;
        }
        let mut kers: Vec<(&str, usize, f64)> =
            by_kernel.into_iter().map(|(k, (c, t))| (k, c, t)).collect();
        kers.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        eprintln!(
            "[gpu][prof] GPU 总 {total:.2} ms / 提交往返墙钟 {wall_ms:.2} ms / \
             {n} dispatch（均 {:.3} ms）",
            total / n as f64
        );
        for (k, c, t) in &kers {
            eprintln!(
                "[gpu][prof]   {k:<22} ×{c:<4} {t:8.3} ms（均 {:.3}）",
                t / *c as f64
            );
        }
        // top-20 热点（节点名）
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| deltas[b].partial_cmp(&deltas[a]).unwrap());
        let full = std::env::var("QPPOCR_GPU_PROF").unwrap_or_default() == "full";
        let top = if full { n } else { idx.len().min(20) };
        for &i in &idx[..top] {
            let (kernel, node, _, _out_off, _out_n, _, groups, _, shapes, _osh) = &plan.recs[i];
            let sh: Vec<String> = shapes
                .iter()
                .take(2)
                .map(|s| {
                    if s.len() == 4 {
                        format!("[{},{}, {},{}]", s[0], s[1], s[2], s[3])
                    } else {
                        format!("{s:?}")
                    }
                })
                .collect();
            eprintln!(
                "[gpu][prof]   #{i:<4} {kernel:<20} {node:<28} {:.3} ms  grid {groups:?} in {} n={_out_n}",
                deltas[i],
                sh.join(" × "),
            );
        }
    }

    /// n_ 模式的节点 → (内核名, 参数块, dispatch 网格)。
    ///
    /// 全图激活按 NHWC f16 语义路由；偏移一律 u32 word 单位。conv 族
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
    ) -> Result<Vec<(&'static str, ParamBlock, [u32; 3])>> {
        use super::nhwc::{cpad4, repack_conv_w, repack_convt_w, repack_dw_w};
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
                        &self.initializers,
                        &n.inputs[1],
                        layout,
                        uploads,
                        w_offs,
                        &self.repack_cache,
                        |t| repack_conv_w(&t.f32, co, ci, kh, kw),
                    )?;
                    let b_off =
                        self.n_bias_off(n, ws[0], layout, uploads, w_offs, &self.repack_cache)?;
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
                    let nv = cpad4(ws[0]) / 4;
                    Ok(vec![(
                        "n_conv",
                        pb,
                        [div256(m_dim.div_ceil(4) * nv), nb, 1],
                    )])
                } else if group == xs[1] && ws[0] == xs[1] {
                    // depthwise：ws = [C, 1, kh, kw]
                    let cch = ws[0] as usize;
                    let w_off = n_weight_off(
                        &self.initializers,
                        &n.inputs[1],
                        layout,
                        uploads,
                        w_offs,
                        &self.repack_cache,
                        |t| repack_dw_w(&t.f32, cch, kh, kw),
                    )?;
                    let b_off =
                        self.n_bias_off(n, ws[0], layout, uploads, w_offs, &self.repack_cache)?;
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
                    &self.initializers,
                    &n.inputs[1],
                    layout,
                    uploads,
                    w_offs,
                    &self.repack_cache,
                    |t| repack_convt_w(&t.f32, ci, co, kh, kw),
                )?;
                let b_off =
                    self.n_bias_off(n, ws[1], layout, uploads, w_offs, &self.repack_cache)?;
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
                    let r = shape_of(&n.inputs[0]).len() as i64;
                    let norm: Vec<i64> = axes
                        .iter()
                        .map(|&a| if a < 0 { a + r } else { a })
                        .collect();
                    if !(norm.len() == 2 && norm.contains(&2) && norm.contains(&3)) {
                        return Err(Error::Graph(format!(
                            "n_ 路径 ReduceMean 只支持 axes={{2,3}}，实得 {axes:?}"
                        )));
                    }
                }
                let xs = shape_of(&n.inputs[0]);
                if xs[0] != 1 {
                    return Err(Error::Graph("n_ 路径 SE 归约只支持 N=1".into()));
                }
                let cp = cpad4(xs[1]);
                if cp > 256 {
                    return Err(Error::Graph(format!(
                        "n_ 路径 SE 归约通道 {cp} > 256（两阶段内核上限）"
                    )));
                }
                let m_dim = (xs[2] * xs[3]) as u32;
                let m_blocks = m_dim.div_ceil(1024);
                // 阶段 1：分块部分和 → scratch
                let mut pb1 = ParamBlock::new();
                pb1.u(off_of(&n.inputs[0])?)
                    .u(reduce_scratch)
                    .u(m_dim)
                    .u(cp);
                let r1 = ("n_reduce_hw", pb1, [m_blocks, 1, 1]);
                // 阶段 2：scratch → f16 gate
                let mut pb2 = ParamBlock::new();
                pb2.u(reduce_scratch)
                    .u(off_of(out)?)
                    .u(m_blocks)
                    .u(cp)
                    .u(m_dim);
                let r2 = ("n_reduce_fin", pb2, [1, 1, 1]);
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
                    .u(xs[2] as u32);
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
            "Sigmoid" | "Relu" | "HardSigmoid" | "Clip" => {
                let (op, p1, p2) = match n.op_type.as_str() {
                    "Sigmoid" => (1u32, 0f32, 0f32),
                    "Relu" => (0, 0.0, 0.0),
                    "HardSigmoid" => (
                        2,
                        planner::get_f(n.attr("alpha"), 0.2),
                        planner::get_f(n.attr("beta"), 0.5),
                    ),
                    _ => (
                        3,
                        planner::get_f(n.attr("min"), -3.4e38),
                        planner::get_f(n.attr("max"), 3.4e38),
                    ),
                };
                let n_words = super::nhwc::nhwc_words(&shape_of(&n.inputs[0]));
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
            "Add" | "Mul" => {
                let a = shape_of(&n.inputs[0]);
                let b = shape_of(&n.inputs[1]);
                let b_n = numel(table, &n.inputs[1]);
                let a_words = super::nhwc::nhwc_words(&a);
                if a == b {
                    let mut pb = ParamBlock::new();
                    pb.u(if n.op_type == "Add" { 4 } else { 5 })
                        .u(off_of(&n.inputs[0])?)
                        .u(off_of(&n.inputs[1])?)
                        .u(off_of(out)?)
                        .u(a_words)
                        .f(0.0)
                        .f(0.0);
                    Ok(vec![("n_elem", pb, [div256(a_words / 4), 1, 1])])
                } else if b.len() == 4 && b_n == b[1] as u32 {
                    // [1,C,1,1] 通道广播；SE 融合门 → fused（读前激活值）
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
                    let mut pb = ParamBlock::new();
                    pb.u(op)
                        .u(off_of(&n.inputs[0])?)
                        .u(gate_off)
                        .u(OFF_NONE) // r_off（muladd_scale 专用）
                        .u(off_of(out)?)
                        .u(a_words)
                        .u(cp)
                        .f(p1)
                        .f(p2);
                    Ok(vec![("n_channel", pb, [div256(a_words / 4), 1, 1])])
                } else {
                    Err(Error::Graph(format!(
                        "n_ 路径 {}：只支持同形或 [1,C,1,1] 广播，实得 {a:?} vs {b:?}",
                        n.op_type
                    )))
                }
            }
            "MulAddScale" => {
                let fs = shape_of(&n.inputs[0]);
                let a_words = super::nhwc::nhwc_words(&fs);
                let cp = cpad4(fs[1]);
                let mut pb = ParamBlock::new();
                pb.u(2u32) // muladd_scale
                    .u(off_of(&n.inputs[0])?)
                    .u(off_of(&n.inputs[1])?)
                    .u(off_of(&n.inputs[2])?)
                    .u(off_of(out)?)
                    .u(a_words)
                    .u(cp)
                    .f(0.0)
                    .f(0.0);
                Ok(vec![("n_channel", pb, [div256(a_words / 4), 1, 1])])
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
        _repack_cache: &Mutex<HashMap<String, std::sync::Arc<Vec<u32>>>>,
    ) -> Result<u32> {
        if n.inputs.len() <= 2 || n.inputs[2].is_empty() {
            return Ok(OFF_NONE);
        }
        n_weight_off(
            &self.initializers,
            &n.inputs[2],
            layout,
            uploads,
            w_offs,
            &self.repack_cache,
            |t| super::nhwc::conv_bias(&t.f32, co),
        )
    }

    /// 节点 → (内核名, 参数块 | 内联 PC, dispatch 网格)。
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    fn map_node(
        &self,
        n: &Node,
        table: &planner::ShapeTable,
        offs: &HashMap<String, u32>,
        fused: &std::collections::HashSet<String>,
    ) -> Result<(&'static str, Option<ParamBlock>, Vec<u8>, [u32; 3])> {
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
        let out = &n.outputs[0];
        let out_shape = shape_of(out);
        let out_n = numel(table, out);

        match n.op_type.as_str() {
            "Conv" => {
                let xs = shape_of(&n.inputs[0]);
                let ws = shape_of(&n.inputs[1]);
                let (sh, sw) = strides(n);
                let pads = pads4(n);
                // 1x1 s1 p0 g1 N=1：tiled GEMM 路径（NCHW 输入天然
                // [Ci,HW]，零重排）。det 图 61/83 conv 走这里。
                if ws[2] == 1
                    && ws[3] == 1
                    && sh == 1
                    && sw == 1
                    && pads == (0, 0, 0, 0)
                    && std::env::var_os("QPPOCR_GPU_NO_GEMM").is_none()
                    && planner::get_i(n.attr("group"), 1) == 1
                    && xs[0] == 1
                {
                    let m = ws[0] as u32;
                    // GEMM 的 N 维 = 空间 H×W（不含 M）——out_n 是 M×H×W，
                    // 拿它当 HW 会超界（实测设备挂死的根因）
                    let hw = (out_shape[2] * out_shape[3]) as u32;
                    let mut pb = ParamBlock::new();
                    pb.u(off_of(&n.inputs[0])?)
                        .u(off_of(&n.inputs[1])?)
                        .u(if n.inputs.len() > 2 && !n.inputs[2].is_empty() {
                            off_of(&n.inputs[2])?
                        } else {
                            OFF_NONE
                        })
                        .u(if n.inputs.len() > 3 && !n.inputs[3].is_empty() {
                            off_of(&n.inputs[3])?
                        } else {
                            OFF_NONE
                        })
                        .u(off_of(out)?)
                        .u(1) // n（占位）
                        .u(ws[1] as u32) // ci
                        .u(1)
                        .u(1) // h, w（占位）
                        .u(m) // m
                        .u(1)
                        .u(1) // kh, kw
                        .u(1)
                        .u(1) // sh, sw
                        .u(0)
                        .u(0) // ph, pw
                        .u(1) // group
                        .u(planner::get_i(n.attr("act"), 0) as u32)
                        .f(planner::get_f(n.attr("act_c1"), std::f32::consts::SQRT_2))
                        .f(planner::get_f(n.attr("act_c2"), 1.0))
                        .f(planner::get_f(n.attr("act_c3"), 0.5))
                        .u(hw) // ohw（= N 维）
                        .u(0); // ow（GEMM 不用）
                    return Ok((
                        "conv_gemm",
                        Some(pb),
                        Vec::new(),
                        [hw / 16 + 1, m / 16 + 1, 1],
                    ));
                }
                let b_off = if n.inputs.len() > 2 && !n.inputs[2].is_empty() {
                    off_of(&n.inputs[2])?
                } else {
                    OFF_NONE
                };
                let r_off = if n.inputs.len() > 3 && !n.inputs[3].is_empty() {
                    off_of(&n.inputs[3])?
                } else {
                    OFF_NONE
                };
                let oh = out_shape[2] as u32;
                let ow = out_shape[3] as u32;
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(off_of(&n.inputs[1])?)
                    .u(b_off)
                    .u(r_off)
                    .u(off_of(out)?)
                    .u(xs[0] as u32)
                    .u(xs[1] as u32)
                    .u(xs[2] as u32)
                    .u(xs[3] as u32)
                    .u(ws[0] as u32)
                    .u(ws[2] as u32)
                    .u(ws[3] as u32)
                    .u(sh as u32)
                    .u(sw as u32)
                    .u(pads.0 as u32)
                    .u(pads.1 as u32)
                    .u(planner::get_i(n.attr("group"), 1) as u32)
                    .u(planner::get_i(n.attr("act"), 0) as u32)
                    .f(planner::get_f(n.attr("act_c1"), std::f32::consts::SQRT_2))
                    .f(planner::get_f(n.attr("act_c2"), 1.0))
                    .f(planner::get_f(n.attr("act_c3"), 0.5))
                    .u(oh * ow)
                    .u(ow);
                Ok((
                    "conv",
                    Some(pb),
                    Vec::new(),
                    [
                        ow / 8 + 1,
                        oh / 8 + 1,
                        out_shape[0] as u32 * out_shape[1] as u32,
                    ],
                ))
            }
            "ConvTranspose" => {
                let xs = shape_of(&n.inputs[0]);
                let ws = shape_of(&n.inputs[1]);
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(off_of(&n.inputs[1])?)
                    .u(off_of(out)?)
                    .u(xs[0] as u32)
                    .u(xs[1] as u32)
                    .u(xs[2] as u32)
                    .u(xs[3] as u32)
                    .u(ws[1] as u32);
                Ok((
                    "convtranspose",
                    Some(pb),
                    Vec::new(),
                    [
                        (2 * xs[3]) as u32 / 8 + 1,
                        (2 * xs[2]) as u32 / 8 + 1,
                        out_shape[0] as u32 * out_shape[1] as u32,
                    ],
                ))
            }
            "MaxPool" | "AveragePool" => {
                let xs = shape_of(&n.inputs[0]);
                let ks = n
                    .attr("kernel_shape")
                    .filter(|a| a.ints.len() == 2)
                    .ok_or_else(|| Error::Graph("kernel_shape required".into()))?;
                let (sh, sw) = strides(n);
                let pads = pads4(n);
                let oh = out_shape[2] as u32;
                let ow = out_shape[3] as u32;
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(off_of(out)?)
                    .u(xs[0] as u32)
                    .u(xs[1] as u32)
                    .u(xs[2] as u32)
                    .u(xs[3] as u32)
                    .u(ks.ints[0] as u32)
                    .u(ks.ints[1] as u32)
                    .u(sh as u32)
                    .u(sw as u32)
                    .u(pads.0 as u32)
                    .u(pads.1 as u32)
                    .u(u32::from(n.op_type == "MaxPool"))
                    .u(oh)
                    .u(ow);
                Ok((
                    "pool",
                    Some(pb),
                    Vec::new(),
                    [
                        ow / 8 + 1,
                        oh / 8 + 1,
                        out_shape[0] as u32 * out_shape[1] as u32,
                    ],
                ))
            }
            "GlobalAveragePool" | "ReduceMean" => {
                if n.op_type == "ReduceMean" {
                    let axes = planner::axes_from(n, &table.values)
                        .map_err(|e| Error::Graph(format!("ReduceMean: {e}")))?;
                    let r = shape_of(&n.inputs[0]).len() as i64;
                    let norm: Vec<i64> = axes
                        .iter()
                        .map(|&a| if a < 0 { a + r } else { a })
                        .collect();
                    if !(norm.len() == 2 && norm.contains(&2) && norm.contains(&3)) {
                        return Err(Error::Graph(format!(
                            "ReduceMean 只支持 axes={{2,3}}（SE 归约），实得 {axes:?}"
                        )));
                    }
                }
                let xs = shape_of(&n.inputs[0]);
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(off_of(out)?)
                    .u(xs[0] as u32)
                    .u(xs[1] as u32)
                    .u(xs[2] as u32)
                    .u(xs[3] as u32);
                Ok((
                    "reduce_hw",
                    Some(pb),
                    Vec::new(),
                    // 一个 WG 归约一个 (n,c) 通道平面（内核内线程跨步 +
                    // subgroup 归约）——曾误用 (n*c)/256+1，64 通道只给
                    // 1 个 WG，每线程串行扫 5 万元素，单个 GAP 2.9 ms。
                    [(xs[0] * xs[1]) as u32, 1, 1],
                ))
            }
            "Resize" => {
                let mode = n
                    .attr("mode")
                    .filter(|a| a.has_s)
                    .map(|a| a.s.clone())
                    .unwrap_or_else(|| "nearest".into());
                if mode != "nearest" {
                    return Err(Error::Graph(format!(
                        "Resize 只支持 nearest（det 全是它），实得 {mode}"
                    )));
                }
                let xs = shape_of(&n.inputs[0]);
                let oh = out_shape[2] as u32;
                let ow = out_shape[3] as u32;
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(off_of(out)?)
                    .u(xs[0] as u32)
                    .u(xs[1] as u32)
                    .u(xs[2] as u32)
                    .u(xs[3] as u32)
                    .u(oh)
                    .u(ow);
                Ok((
                    "resize_nearest",
                    Some(pb),
                    Vec::new(),
                    [
                        ow / 8 + 1,
                        oh / 8 + 1,
                        out_shape[0] as u32 * out_shape[1] as u32,
                    ],
                ))
            }
            "Concat" => {
                let axis = planner::get_i(n.attr("axis"), 0);
                if axis != 1 {
                    return Err(Error::Graph(format!("Concat 只支持 axis=1，实得 {axis}")));
                }
                let xs = shape_of(&n.inputs[0]);
                let mut pb = ParamBlock::new();
                pb.u(off_of(out)?)
                    .u(xs[0] as u32)
                    .u(xs[2] as u32)
                    .u(xs[3] as u32)
                    .u(n.inputs.len() as u32)
                    .u(out_shape[1] as u32)
                    .u(0)
                    .u(0);
                for inn in &n.inputs {
                    if inn.is_empty() {
                        continue;
                    }
                    pb.u(off_of(inn)?).u(shape_of(inn)[1] as u32);
                }
                Ok((
                    "concat_c",
                    Some(pb),
                    Vec::new(),
                    [
                        xs[3] as u32 / 8 + 1,
                        xs[2] as u32 / 8 + 1,
                        out_shape[0] as u32 * out_shape[1] as u32,
                    ],
                ))
            }
            "MulAddScale" => {
                let fs = shape_of(&n.inputs[0]);
                let (c, hw) = (fs[1] as u32, (fs[2] * fs[3]) as u32);
                let pc = PcMulAddScale {
                    f_off: off_of(&n.inputs[0])?,
                    gate_off: off_of(&n.inputs[1])?,
                    r_off: off_of(&n.inputs[2])?,
                    out_off: off_of(out)?,
                    hw,
                    c,
                };
                Ok((
                    "muladd_scale",
                    None,
                    pc.bytes().to_vec(),
                    [hw / 256 + 1, c, fs[0] as u32],
                ))
            }
            "Sigmoid" | "Relu" | "HardSigmoid" | "Clip" => {
                let pc = if n.op_type == "Clip" {
                    PcUnaryF {
                        in_off: off_of(&n.inputs[0])?,
                        out_off: off_of(out)?,
                        n: out_n,
                        p1: planner::get_f(n.attr("min"), -3.4e38),
                        p2: planner::get_f(n.attr("max"), 3.4e38),
                    }
                    .bytes()
                    .to_vec()
                } else if n.op_type == "HardSigmoid" {
                    PcUnaryF {
                        in_off: off_of(&n.inputs[0])?,
                        out_off: off_of(out)?,
                        n: out_n,
                        p1: planner::get_f(n.attr("alpha"), 0.2),
                        p2: planner::get_f(n.attr("beta"), 0.5),
                    }
                    .bytes()
                    .to_vec()
                } else {
                    PcUnary {
                        in_off: off_of(&n.inputs[0])?,
                        out_off: off_of(out)?,
                        n: out_n,
                    }
                    .bytes()
                    .to_vec()
                };
                let kernel = match n.op_type.as_str() {
                    "Sigmoid" => "sigmoid",
                    "Relu" => "relu",
                    "HardSigmoid" => "hardsigmoid",
                    _ => "clip",
                };
                Ok((kernel, None, pc, [out_n / 256 + 1, 1, 1]))
            }
            "Add" | "Mul" => {
                let a = shape_of(&n.inputs[0]);
                let b = shape_of(&n.inputs[1]);
                let b_n = numel(table, &n.inputs[1]);
                if a == b {
                    let pc = PcBinary {
                        a_off: off_of(&n.inputs[0])?,
                        b_off: off_of(&n.inputs[1])?,
                        out_off: off_of(out)?,
                        n: out_n,
                    };
                    Ok((
                        if n.op_type == "Add" { "add" } else { "mul" },
                        None,
                        pc.bytes().to_vec(),
                        [out_n / 256 + 1, 1, 1],
                    ))
                } else if b.len() == 4 && b_n == b[1] as u32 {
                    // [1,C,1,1] 通道广播
                    // SE 融合：门的激活算子被跳过时，用 fused_*_mul
                    //（一趋 NCHW 读写替代独立门 dispatch + mul_c 两趋）
                    let gate_producer = self
                        .graph
                        .nodes
                        .iter()
                        .find(|m| m.outputs.first() == Some(&n.inputs[1]));
                    let is_fused_gate = gate_producer.is_some_and(|m| {
                        fused.contains(&m.name)
                            && (m.op_type == "HardSigmoid" || m.op_type == "Sigmoid")
                    });
                    if is_fused_gate && n.op_type == "Mul" {
                        let gp = gate_producer.unwrap();
                        let kernel = if gp.op_type == "HardSigmoid" {
                            "fused_hardsigmoid_mul"
                        } else {
                            "fused_sigmoid_mul"
                        };
                        // gate 偏移指到 HardSigmoid/Sigmoid 的**输入**（前激活）
                        let pc = PcChannel {
                            a_off: off_of(&n.inputs[0])?,
                            b_off: off_of(&gp.inputs[0])?,
                            out_off: off_of(out)?,
                            hw: (a[2] * a[3]) as u32,
                            c: a[1] as u32,
                        };
                        return Ok((
                            kernel,
                            None,
                            pc.bytes().to_vec(),
                            [(a[2] * a[3]) as u32 / 256 + 1, a[1] as u32, a[0] as u32],
                        ));
                    }
                    let pc = PcChannel {
                        a_off: off_of(&n.inputs[0])?,
                        b_off: off_of(&n.inputs[1])?,
                        out_off: off_of(out)?,
                        hw: (a[2] * a[3]) as u32,
                        c: a[1] as u32,
                    };
                    Ok((
                        if n.op_type == "Mul" { "mul_c" } else { "add_c" },
                        None,
                        pc.bytes().to_vec(),
                        [(a[2] * a[3]) as u32 / 256 + 1, a[1] as u32, a[0] as u32],
                    ))
                } else {
                    Err(Error::Graph(format!(
                        "{}: 只支持同形或 [1,C,1,1] 通道广播，实得 {a:?} vs {b:?}",
                        n.op_type
                    )))
                }
            }
            other => Err(Error::Graph(format!(
                "GPU 计划不支持算子 {other}（节点 {}）——当前内核集覆盖 det",
                n.name
            ))),
        }
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
    repack_cache: &Mutex<HashMap<String, std::sync::Arc<Vec<u32>>>>,
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

/// 输出名 → 产出它的节点序号（线性扫，调试路径，图 ≤ 数百节点）。
fn node_index_of(graph: &Graph, out: &str) -> Option<usize> {
    graph
        .nodes
        .iter()
        .position(|n| n.outputs.first() == Some(&out.to_string()))
}

/// dump 文件头的 rank（i32 LE）。
fn rank_of(bytes: &[u8]) -> usize {
    if bytes.len() < 4 {
        return 0;
    }
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize
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

impl Drop for VulkanSession {
    fn drop(&mut self) {
        // SAFETY: 缓存句柄由本会话独占；此刻 plans（含管线）已先行
        // drop（字段声明序），ctx 的设备仍存活（ctx 是最后字段）。
        unsafe {
            self.ctx
                .device
                .raw()
                .destroy_pipeline_cache(self.pipeline_cache, None)
        };
    }
}

impl DeviceSession for VulkanSession {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Vulkan
    }

    fn prefers_host_parallelism(&self) -> bool {
        false // GPU 内部已并行：pipeline 走串行批（Phase 0 接缝）
    }

    fn run(&self, inputs: Vec<(String, Tensor)>) -> Result<Vec<Tensor>> {
        let mut plans = self
            .plans
            .lock()
            .map_err(|_| Error::Device("计划缓存锁中毒".into()))?;
        let in_shape: Vec<i64> = inputs
            .iter()
            .find(|(nm, _)| nm == &self.input_name)
            .map(|(_, t)| t.shape.clone())
            .ok_or_else(|| Error::Graph(format!("缺少输入 {}", self.input_name)))?;
        if !plans.iter().any(|(s, _)| *s == in_shape) {
            let plan = self.build_plan(&in_shape)?;
            plans.push((in_shape.clone(), plan));
            // 字节预算封顶：每个计划 ≈ 整块 arena（100-140 MB @ 960 输入），
            // 100 图语料的形状多样性会无界增长。按块大小淘汰最旧的
            // （Vec 头部）直到总额 ≤ 预算；命中的形状下次重建
            // （管线缓存已把重建压到 ~2ms）。
            const DEFAULT_BUDGET_MB: u64 = 768;
            let budget: u64 = std::env::var("QPPOCR_GPU_PLAN_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_BUDGET_MB);
            let mut total: u64 = plans.iter().map(|(_, p)| p.block_bytes).sum();
            let mut evicted_arenas: Vec<Arena> = Vec::new();
            while total > budget * (1 << 20) && plans.len() > 1 {
                let (_, mut evicted) = plans.remove(0);
                total -= evicted.block_bytes;
                if let Some(a) = evicted.arena.take() {
                    evicted_arenas.push(a);
                }
            }
            // 归还的块进池（封顶 2，按容量留大——下一形状 reset 复用）
            if !evicted_arenas.is_empty() {
                if let Ok(mut pool) = self.arena_pool.lock() {
                    for a in evicted_arenas {
                        pool.push(a);
                        if pool.len() > 2 {
                            // 淘汰容量最小的
                            let mut min_i = 0;
                            for i in 1..pool.len() {
                                if pool[i].chunk_bytes() < pool[min_i].chunk_bytes() {
                                    min_i = i;
                                }
                            }
                            pool.swap_remove(min_i);
                        }
                    }
                }
            }
        }
        let plan = plans
            .iter()
            .find(|(s, _)| *s == in_shape)
            .map(|(_, p)| p)
            .unwrap();

        for (nm, t) in &inputs {
            if let Some(off) = plan.in_offs.get(nm) {
                // SAFETY: base 是整块持久映射；off+len 在 total 内（按输入形状分配）。
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        t.f32.as_ptr(),
                        plan.base.add(*off as usize * 4) as *mut f32,
                        t.f32.len(),
                    );
                }
            }
        }
        if std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some() {
            self.stepped_run_and_cmp(plan)?;
        } else {
            let t0 = std::time::Instant::now();
            self.ctx.device.submit_wait_cb(plan.cb)?;
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if let Some(p) = plan.qpool {
                self.prof_print(plan, p, wall_ms);
            }
        }
        let mut outs = Vec::with_capacity(plan.out_offs.len());
        for (i, (name, off, n)) in plan.out_offs.iter().enumerate() {
            let shape = plan.out_shapes.get(i).cloned().unwrap_or_default();
            let mut buf = F32Buf::with_zeroed(*n);
            // SAFETY: 已等信号；off+n ≤ total。
            unsafe {
                std::ptr::copy_nonoverlapping(
                    plan.base.add(*off as usize * 4) as *const f32,
                    buf.as_mut_slice().as_mut_ptr(),
                    *n,
                );
            }
            outs.push(Tensor {
                name: name.clone(),
                shape,
                dtype: DType::F32,
                f32: buf,
                i64: Vec::new(),
            });
        }
        Ok(outs)
    }
}

/// 消费释放：输入若再无消费者，归还区域给布局复用。
/// **豁免**：图输出（要读回）、权重（CB 每次重放都读）、图输入（每次
/// run 重写）——这三类区域一旦被复用就是数据 corrupt。
/// 区域大小按模式取：f32 元素数 / NHWC-f16 word 数（与分配一致，否则
/// 空闲表尺寸错位）。
#[allow(clippy::too_many_arguments)]
fn release_inputs(
    n: &Node,
    live: &mut HashMap<String, u32>,
    offs: &mut HashMap<String, u32>,
    layout: &mut Layout,
    table: &planner::ShapeTable,
    is_output: &dyn Fn(&str) -> bool,
    initializers: &HashMap<String, Tensor>,
    input_name: &str,
    f32_mode: bool,
) {
    let dbg = std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some();
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
                eprintln!("[gpu][live] {inn} 消费于 {}（剩 {}）", n.name, *c);
            }
            if *c == 0 && table.values.get(inn).map(|v| v.dtype) == Some(DType::F32) {
                if let Some(o) = offs.remove(inn) {
                    if dbg {
                        eprintln!("[gpu][live] {inn} @{} 释放于 {}", o, n.name);
                    }
                    let size = match table.values.get(inn) {
                        Some(v) if !f32_mode => super::nhwc::nhwc_words(&v.shape),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::VulkanContext;
    use qppocr_core::executor::Session;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    /// 端到端：真实 tiny det，GPU 整图计划 vs CPU executor，同一输入
    /// 的概率图容差对拍；同形二次 run 走缓存、异形 run 建新计划。
    #[test]
    fn det_session_end_to_end() {
        let p = std::path::Path::new("../../models/tiny/det.onnx");
        if !p.is_file() {
            eprintln!("[gpu] 无模型，跳过");
            return;
        }
        let bytes = std::fs::read(p).unwrap();
        let Some(ctx) = (match VulkanContext::open(None) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("[gpu] 跳过（无满足基线的设备）: {e}");
                None
            }
        }) else {
            return;
        };

        let (h, w) = (960i64, 864i64);
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mk_input = |name: &str, seed: &mut u64| -> Tensor {
            let mut buf = F32Buf::with_zeroed((3 * h * w) as usize);
            for v in buf.as_mut_slice().iter_mut() {
                *v = lcg(seed);
            }
            Tensor {
                name: name.into(),
                shape: vec![1, 3, h, w],
                dtype: DType::F32,
                f32: buf,
                i64: Vec::new(),
            }
        };
        // ★ 同一份输入克隆给 CPU/GPU——mk_input 每次调用是新随机数据，
        //   曾因两边各生成一份导致「全图分歧」的假阳性。
        let input_once = mk_input("", &mut seed);
        let cl = move |nm: &str| {
            let mut t = input_once.clone();
            t.name = nm.to_string();
            t
        };

        let _ = &cl; // 占位（下方用 cl() 统一取同一份输入）
        // CPU 参考（带 dump：GPU 步进模式的逐节点对拍判据）
        let dump_dir = std::env::temp_dir().join("qppocr-g2-dbg");
        let _ = std::fs::remove_dir_all(&dump_dir);
        std::fs::create_dir_all(&dump_dir).unwrap();
        // SAFETY: 测试独占该环境变量（仅 Session::run 读）。
        unsafe { std::env::set_var("QPPOCR_DUMP_DIR", &dump_dir) };
        let cpu = Session::from_memory(&bytes, "tiny.det").unwrap();
        let in_name = cpu
            .graph
            .inputs
            .iter()
            .find(|s| !s.is_empty())
            .cloned()
            .unwrap();
        let t0 = std::time::Instant::now();
        let out_cpu = cpu.run(vec![(in_name.clone(), cl(&in_name))]).unwrap();
        // SAFETY: 同上。
        unsafe { std::env::remove_var("QPPOCR_DUMP_DIR") };
        eprintln!(
            "[gpu] CPU 前向 {:.1} ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );

        if std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some() {
            // SAFETY: 同上（本测试进程内）。
            unsafe { std::env::set_var("QPPOCR_GPU_DUMP_CMP", &dump_dir) };
        }
        // GPU 会话（同图同权重）
        let (graph, init) = {
            let s = Session::from_memory(&bytes, "tiny.det2").unwrap();
            s.into_parts()
        };
        let gpu = VulkanSession::new(ctx.inner.clone(), graph, init).unwrap();
        let t1 = std::time::Instant::now();
        let out_gpu = gpu.run(vec![(in_name.clone(), cl(&in_name))]).unwrap();
        eprintln!(
            "[gpu] GPU 首跑（含建计划）{:.1} ms",
            t1.elapsed().as_secs_f64() * 1000.0
        );
        // 同形二次（缓存命中 + CB 重放）
        let t2 = std::time::Instant::now();
        let _ = gpu.run(vec![(in_name.clone(), cl(&in_name))]).unwrap();
        eprintln!(
            "[gpu] GPU 二跑（缓存）{:.1} ms",
            t2.elapsed().as_secs_f64() * 1000.0
        );

        assert_eq!(out_cpu.len(), out_gpu.len());
        let a = &out_gpu[0].f32;
        let b = &out_cpu[0].f32;
        assert_eq!(a.len(), b.len(), "输出元素数不一致");
        assert_eq!(out_gpu[0].shape, out_cpu[0].shape, "输出形状不一致");
        let max_abs = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let mean_abs = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>()
            / a.len() as f32;
        eprintln!("[gpu] det 概率图：max|diff| = {max_abs:.3e}，mean|diff| = {mean_abs:.3e}");
        // 值域检查：概率图应在 [0,1]，有效文本区域的值应接近 1
        let gpu_min = a.iter().cloned().fold(1f32, f32::min);
        let gpu_max = a.iter().cloned().fold(0f32, f32::max);
        let gpu_mean = a.iter().sum::<f32>() / a.len() as f32;
        let cpu_min = b.iter().cloned().fold(1f32, f32::min);
        let cpu_max = b.iter().cloned().fold(0f32, f32::max);
        let cpu_mean = b.iter().sum::<f32>() / b.len() as f32;
        let gpu_hi = a.iter().filter(|v| **v > 0.2).count();
        let cpu_hi = b.iter().filter(|v| **v > 0.2).count();
        eprintln!("[gpu] GPU 值域 [{gpu_min:.4},{gpu_max:.4}] mean={gpu_mean:.4} >0.2: {gpu_hi}");
        eprintln!("[gpu] CPU 值域 [{cpu_min:.4},{cpu_max:.4}] mean={cpu_mean:.4} >0.2: {cpu_hi}");
        // 验收口径：mean 反映整体数值健康（逐层 fma/累加序差异的传播），
        // max 允许 sigmoid 斜坡处的极值放大（det_thresh=0.2 附近的框边界
        // 翻转由 verify.py 的字符级对拍判定，不在这里卡死）。
        assert!(mean_abs < 1e-3, "概率图均值偏差超容差: {mean_abs}");
        assert!(max_abs < 5e-2, "概率图极值偏差超容差: {max_abs}");
    }
}
