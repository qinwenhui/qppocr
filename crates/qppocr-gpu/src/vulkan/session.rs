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
    /// real_n 单元的 word 偏移（构造上恒 0；显式存字段防布局演算漂移）。
    rn_word: u32,
    /// 本计划是否有一条在飞的 deferred 提交（同形状两条在飞 = 输入/
    /// 输出区同址互踩；stage 置位、收账复位，debug 断言兜底）。
    inflight: std::sync::atomic::AtomicBool,
    /// 计划存活期归本计划；**Drop 时**（最后引用释放 = 无在飞 CB）与
    /// `ks` 成对归还会话池（见 [`VulkanSession::arena_pool`]——免掉大块
    /// 页提交与内核套件重建；Arc 共享池句柄给本计划）。
    arena: Option<Arena>,
    /// 计划录制/执行要用；Drop 时与 arena 成对归池。None 仅出现在归池后
    /// 的 Drop 中途或裸测试构造。
    ks: Option<KernelSet>,
    /// 会话池句柄（Drop 归还用）。
    pool: std::sync::Arc<Mutex<Vec<(Arena, KernelSet)>>>,
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
        // 归还 (arena, KernelSet) 对：此刻无任何 CB 引用（最后引用释放，
        // deferred 已收账/同步 run 已等完），reset 后可安全复用。锁中毒
        // （会话级灾难）也照常归还——into_inner 取数据继续。
        if let (Some(a), Some(ks)) = (self.arena.take(), self.ks.take()) {
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            pool.push((a, ks));
            if pool.len() > 2 {
                // 淘汰容量最小的（封顶 2 对：引擎 deferred 流水把计划
                // Arc 存活拉到下一图之后——池深 1 在 img k+1 建计划时
                // img k 的对还没归池，结构性 miss；深 2 吃住流水）
                let mut min_i = 0;
                for i in 1..pool.len() {
                    if pool[i].0.chunk_bytes() < pool[min_i].0.chunk_bytes() {
                        min_i = i;
                    }
                }
                pool.swap_remove(min_i);
            }
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
        return Ok((m as u32, super::nhwc::cpad4(nn)));
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
            let norm: Vec<i64> = axes.iter().map(|&a| if a < 0 { a + r } else { a }).collect();
            if norm.len() == 1 && norm[0] == r - 1 {
                let m: i64 = ish[..(r as usize - 1)].iter().product();
                return Ok((m as u32, 4));
            }
        }
    }
    // 其余：rank-4 按形状；rank-3/2 继承输入
    let o = sh(&n.outputs[0]);
    if o.len() == 4 {
        Ok(((o[0] * o[2] * o[3]) as u32, super::nhwc::cpad4(o[1])))
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

/// planner 形状表里取形状（缺项给空——调用方容错）。
fn shape_of(table: &planner::ShapeTable, name: &str) -> Vec<i64> {
    table
        .values
        .get(name)
        .map(|v| v.shape.clone())
        .unwrap_or_default()
}

/// 一个形状的计划槽：primary + 可选影子（同形状双批在飞时分流）。
struct PlanSlot {
    shape: Vec<i64>,
    primary: std::sync::Arc<Plan>,
    shadow: Option<std::sync::Arc<Plan>>,
}

impl PlanSlot {
    fn bytes(&self) -> u64 {
        self.primary.block_bytes
            + self.shadow.as_ref().map_or(0, |s| s.block_bytes)
    }
}

/// GPU 会话：设备共享句柄 + 形状键计划缓存。
pub(crate) struct VulkanSession {
    graph: Graph,
    initializers: HashMap<String, Tensor>,
    input_name: String,
    /// 形状键计划缓存（含**影子计划**槽）：primary 服务串行重放；同形状
    /// 第二条在飞时（引擎同桶批 k+1 提交而 k 未收账）自动分流 shadow——
    /// 独立 arena/CB/rn_word，与 primary 零共享，防写穿不需要「先收账再
    /// 提交」的串行化（rec_infer 35ms 的主因）。惰性建：首次同桶并发才
    /// 付一份构建（池化后 ~2-3ms，一次性）。
    plans: Mutex<Vec<PlanSlot>>,
    /// 会话级管线缓存：按形状重建计划时驱动侧复用编译产物
    ///（形状多样性 × LRU 驱逐会让每帧都重建，38 条管线重建 ~5ms）。
    pipeline_cache: vk::PipelineCache,
    /// 会话级权重重排缓存：重排算术（83 层 conv 的 k-major 循环 ~9ms）
    /// 与形状无关——按名缓存，重建只付 memcpy（~1ms）。
    repack_cache: Mutex<HashMap<String, std::sync::Arc<Vec<u32>>>>,
    /// 会话级 **(arena, KernelSet) 成对池**：计划 Drop（最后引用释放 =
    /// 无在飞 CB）时归还两件套，新形状 reset 复用（容量够时）。137MB 新
    /// 块页提交 ~7ms vs 复用 ~0；KernelSet 绑定 arena 的块缓冲（reset 不
    /// 改缓冲句柄/尺寸，绑定恒有效），成对复用把重建从「arena 免页提交 +
    /// KernelSet 全套重建（DSL/描述符池/管线 2-10ms）」压到 ~0。
    /// Arc 共享给 Plan：**归池在 Drop 而非驱逐时**——deferred 流水下驱逐
    /// 刻旧计划 Arc 常在飞（get_mut 失败），驱逐路径归池曾大量失效
    /// （实测每图仍 17ms 新建）。池封顶 2 对（按容量留大——deferred 流水深度 2，见 Drop 归池注释）。
    arena_pool: std::sync::Arc<Mutex<Vec<(Arena, KernelSet)>>>,
    /// 批维补齐粒度（rec 会话=8，其余 1）：run() 把输入批维向上取整到
    /// 该值建计划，空行由内核 real_n 早退守卫零算力——(bsz,W) 形状塌缩
    /// 成 (grain,W桶)，计划全命中。与 `DeviceSession::batch_grain` 同源
    /// （mod.rs 按模型角色设置；引擎按 trait 值分批）。
    pub(crate) batch_grain: i32,
    /// 宽度精确模式（无注意力 mask 的 rec——SVTR 注意力头）：桶宽右零
    /// 填充经注意力混入真步概率，**模型本身对填充不 invariant**（CPU
    /// 实证 small 932/1036 掉 46 行；small_rec_padding_invariance 守卫）
    /// ——true 时 bucket_grain=1（精确宽，同宽行仍合批）。纯卷积 rec
    /// （tiny）实证填充不变，维持 64 桶。
    pub(crate) exact_width: bool,
    /// rec 会话的 CTC argmax 出口：rank-3 概率输出不走 n_exit3 全量
    /// 拷贝，改 n_exit3_argmax 每时间步写 (val, idx) 对——读回从
    /// T×V×4 B/行（~3 MB）降到 8 B/行。本机 clflush 读回 ~1 ms/MB，
    /// 全量读回实测 ~18 ms/图，是 rec 上 GPU 的大头。引擎侧用
    /// `DeviceSession::rec_argmax_pairs` 探测并走 ctc_decode_pairs。
    pub(crate) argmax_exit: bool,
    /// 计划缓存的字节预算（MB，每会话独立；QPPOCR_GPU_PLAN_MB 覆写）。
    /// **预算 per-session**——det 与 rec 会话各一份，create_session 按
    /// 角色设定：det 形状逐图一次性、暖缓存重建仅 ~2.5-3ms → 小预算；
    /// rec 桶形状每图复用、重建 ~13ms → 大预算。旧的一刀切 2048/会话
    /// 曾把单进程推到 5.5GB（对决实测 6× 竞品，见 COMPARISON.md（九）
    /// 节）。直接构造（测试）默认 2048 不变。
    pub(crate) plan_budget_mb: u64,
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
            arena_pool: std::sync::Arc::new(Mutex::new(Vec::new())),
            batch_grain: 1,
            exact_width: false,
            argmax_exit: false,
            plan_budget_mb: 2048,
            ctx,
        })
    }

    /// 覆盖面探针：小形状试建一次计划（不出执行）。装载期分级用——
    /// 不支持的算子与形状无关，探针必现（见 mod.rs）。
    pub(crate) fn probe(&self, in_shape: &[i64]) -> Result<()> {
        self.build_plan(in_shape).map(|_| ())
    }

    /// 探针失败后的退路：把吃进去的 (graph, initializers) 原样吐还，
    /// 供 CPU 会话组装。Vulkan 资源（pipeline_cache）随 Drop 释放；
    /// build_plan 失败不入缓存 → plans 恒空、无 arena 泄漏。
    pub(crate) fn into_cpu_parts(self) -> qppocr_core::executor::Session {
        let mut me = self;
        let graph = std::mem::take(&mut me.graph);
        let initializers = std::mem::take(&mut me.initializers);
        drop(me);
        qppocr_core::executor::Session::from_parts(graph, initializers)
    }

    /// 执行的前半段（run / run_deferred 共用）：计划建/查 → real_n 写入
    /// → 输入 memcpy → **提交不等**。返回 (计划 Arc, 信号值, 真实行数,
    /// 补齐行数)；调用方 wait_signal(signal) 后 readback_outputs。
    ///
    /// 同形状两条在飞自动分流影子计划（独立 arena/CB/rn_word，零共享）
    /// ——引擎同桶批 k+1 可在 k 收账前提交。第三条在飞报错（流水深度
    /// 应 ≤2）。
    fn stage(
        &self,
        inputs: Vec<(String, Tensor)>,
    ) -> Result<(std::sync::Arc<Plan>, u64, i64, i64)> {
        let mut plans = self
            .plans
            .lock()
            .map_err(|_| Error::Device("计划缓存锁中毒".into()))?;
        let in_shape: Vec<i64> = inputs
            .iter()
            .find(|(nm, _)| nm == &self.input_name)
            .map(|(_, t)| t.shape.clone())
            .ok_or_else(|| Error::Graph(format!("缺少输入 {}", self.input_name)))?;
        // 批维补齐（batch_grain>1 的 rec 会话）：计划按补齐形状建/查，
        // 真实行数写 real_n 单元，内核空行工作组早退——(bsz,W) 形状
        // 塌缩成 (grain,W桶)。空行的输入/输出区是陈旧数据，但被守卫
        // 恒不读，无需清零。
        let n_real = *in_shape.first().unwrap_or(&1);
        let grain = self.batch_grain.max(1) as i64;
        let n_pad = (n_real + grain - 1) / grain * grain;
        let plan_shape = if n_pad != n_real {
            let mut s = in_shape.clone();
            s[0] = n_pad;
            s
        } else {
            in_shape.clone()
        };
        // 选计划：primary 空闲用 primary；同形状在飞（引擎同桶批 k+1 而
        // k 未收账）分流影子——惰性建，独立 arena/CB/rn_word 零共享。
        let mut just_built = false;
        let plan = if let Some(slot) = plans.iter_mut().find(|s| s.shape == plan_shape) {
            let claim = |p: &Plan| {
                p.inflight.store(true, std::sync::atomic::Ordering::Release);
            };
            if !slot
                .primary
                .inflight
                .load(std::sync::atomic::Ordering::Acquire)
            {
                claim(&slot.primary);
                slot.primary.clone()
            } else if slot.shadow.as_ref().is_some_and(|s| {
                !s.inflight.load(std::sync::atomic::Ordering::Acquire)
            }) {
                let s = slot.shadow.clone().unwrap();
                claim(&s);
                s
            } else if slot.shadow.is_some() {
                return Err(Error::Device(
                    "同形状三条在飞：引擎流水深度应 ≤2（primary+shadow 均忙）".into(),
                ));
            } else {
                let shadow = std::sync::Arc::new(self.build_plan(&plan_shape)?);
                claim(&shadow);
                slot.shadow = Some(shadow.clone());
                just_built = true;
                shadow
            }
        } else {
            let plan = std::sync::Arc::new(self.build_plan(&plan_shape)?);
            plan.inflight.store(true, std::sync::atomic::Ordering::Release);
            plans.push(PlanSlot {
                shape: plan_shape.clone(),
                primary: plan.clone(),
                shadow: None,
            });
            just_built = true;
            plan
        };
        if just_built {
            // 字节预算封顶（per-session，角色分配见字段文档；env 覆写）：
            // det ~137MB/计划、逐图一次性（~41 个形状装不下任何预算，靠
            // 三级缓存把重建压到 ~2.5-3ms）；rec 桶 ~24 个 × 60MB 均值，
            // 预算 ≥1.4GB 时全命中。按块大小淘汰最旧的（Vec 头部）；
            // 影子一并计入/驱逐。
            let budget: u64 = std::env::var("QPPOCR_GPU_PLAN_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(self.plan_budget_mb);
            let mut total: u64 = plans.iter().map(|s| s.bytes()).sum();
            while total > budget * (1 << 20) && plans.len() > 1 {
                let evicted = plans.remove(0);
                total -= evicted.bytes();
                // 归池在 Plan::drop（最后引用释放时）：deferred 在飞持有
                // Arc 的时刻此处不强取——drop 时自然归还池。
            }
        }
        let reset_inflight = |p: &Plan| {
            p.inflight
                .store(false, std::sync::atomic::Ordering::Release);
        };
        // real_n 必须先于提交写入（内核读 params[0] 做早退；步进对拍路径
        // 同样经此后重放 CB）。**写后过发布屏障**（SFENCE+clflush 写回
        // 逐出）：4 字节小写无容量压力、不会自我逐出，与 CB 头部屏障
        // 配对（两道都做过缺一复现的 A/B，见 memory.rs publish_clean）。
        // SAFETY: base 是整块持久映射；rn_word 是 build 期静态分配。
        unsafe {
            let p = plan.base.add(plan.rn_word as usize * 4) as *mut u32;
            std::ptr::write_volatile(p, n_real as u32);
            super::memory::publish_clean(p as *const u8, 4);
        }

        for (nm, t) in &inputs {
            if let Some(off) = plan.in_offs.get(nm) {
                if std::env::var_os("QPPOCR_GPU_BUILD_TIME").is_some() {
                    eprintln!(
                        "[gpu][io] 输入 {nm} → @{off}（tensor.name={:?} len={} 头={:?})",
                        t.name,
                        t.f32.len(),
                        &t.f32.as_slice()[..4.min(t.f32.len())]
                    );
                }
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
        let t_submit = std::time::Instant::now();
        let submitted = self.ctx.device.submit_cb(plan.cb);
        if std::env::var_os("QPPOCR_GPU_BUILD_TIME").is_some() {
            eprintln!(
                "[gpu][dbg] submit 主机 {:.2} ms（{} dispatch）",
                t_submit.elapsed().as_secs_f64() * 1000.0,
                plan.recs.len()
            );
        }
        match submitted {
            Ok(signal) => Ok((plan, signal, n_real, n_pad)),
            Err(e) => {
                reset_inflight(&plan);
                Err(e)
            }
        }
    }

    /// 调试：给定形状返回 (整块基址, [(内核, 节点, 偏移, 元素数, PC, 形状, 输入明细)])。
    #[allow(clippy::type_complexity)]
    pub(crate) fn debug_recs(
        &self,
        in_shape: &[i64],
    ) -> Option<(
        *mut u8,
        Vec<(
            String,
            String,
            usize,
            u32,
            u32,
            Vec<u8>,
            Vec<i64>,
            Vec<(String, u32, u32)>,
        )>,
    )> {
        let plans = self.plans.lock().ok()?;
        let p = plans
            .iter()
            .find(|s| s.shape == in_shape)
            .map(|s| &s.primary)?;
        Some((
            p.base,
            p.recs
                .iter()
                .map(|(k, n, idx, o, on, pc, _, ins, _, osh)| {
                    (
                        k.clone(),
                        n.clone(),
                        *idx,
                        *o,
                        *on,
                        pc.clone(),
                        osh.clone(),
                        ins.clone(),
                    )
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
            .find(|s| s.shape == in_shape)
            .map(|s| &mut s.primary)
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
        let ks = plan.ks.as_ref().expect("计划无 KernelSet（已归池？）");
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
            let ks = plan.ks.as_ref().expect("计划无 KernelSet（已归池？）");
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
        // 环内 env 查询一次算好（Windows 上 var_os 是系统调用级开销，
        // 曾以每节点 3 次的频率吃掉节点环 2/3 的时间——rec 计划实测 4.4ms）
        let dbg_env = std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some();

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
        // real_n 单元（arena 第 0 词）：本次 run 的真实批维行数，run() 每
        // 次提交前主机直写；n_ 内核读 params[0] 做批维早退——批维补齐
        // （batch_grain）后空行工作组零算力。必须**最先**分配：内核按
        // params[0] 绝对寻址，静态区只增不复用，word 0 不能让给权重/激活。
        // 初值 1 = 全通过（等 run() 覆写）。
        let rn_word = layout.alloc_static(1);
        debug_assert_eq!(rn_word, 0);
        upload.push((rn_word, vec![1u32]));
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
                    if xs.len() == 4 {
                        let m_blocks = ((xs[2] * xs[3]) as u32).div_ceil(1024);
                        let nb = xs[0] as u32;
                        need = need.max(nb * m_blocks * super::nhwc::cpad4(xs[1]));
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
        // argmax 出口的图输出名（读回长度/形状按 [N,T,2] 覆写，见 out_offs）
        let mut argmax_outs: std::collections::HashSet<String> = std::collections::HashSet::new();
        // n_ 模式：每张量的存储形状 (rows, cols)——cols 是连续维。
        // rank-4 conv 族 = (N*H*W, Cpad)；MatMul 尾部 = (M, Npad)；
        // Squeeze/Unsqueeze/Transpose(0,2,1) 是存储恒等（别名）。
        // 形状不可从逻辑 shape 单向推导（同形 rank-3 有两种来源），
        // 必须按生产者记账。
        let mut rc: HashMap<String, (u32, u32)> = HashMap::new();
        if !f32_mode {
            let ish: Vec<i64> = in_shape.to_vec();
            let (nr, cr) = (ish[0] * ish[2] * ish[3], super::nhwc::cpad4(ish[1]));
            rc.insert(self.input_name.clone(), (nr as u32, cr));
        }
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
            // ---- 存储恒等节点（Squeeze/Unsqueeze/Transpose(0,2,1)/
            // Slice(dim0)）：零分配零 dispatch，输出别名输入（Slice 带偏移）。
            // 输入不释放（存活期 = 别名的最后消费者——记 immortal，泄漏量
            // ~25KB/节点可忽略）。
            if !f32_mode {
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
                    let (start, end, axis, step) =
                        (geti(0), geti(1), geti(2), geti(3));
                    let inn = table.values.get(&n.inputs[0]);
                    let outn = table.values.get(&out);
                    let ok = matches!(
                        (start, end, axis, step),
                        (Some(s), Some(_e), Some(0), Some(1) | None) if s >= 0
                    );
                    if ok && inn.is_some() && outn.is_some() {
                        let inn = inn.unwrap();
                        let dim0 = inn.shape.first().copied().unwrap_or(1).max(1);
                        let row = inn.shape.iter().product::<i64>() / dim0;
                        let off0 = offs.get(&n.inputs[0]).copied();
                        if let Some(o) = off0 {
                            let s = start.unwrap();
                            offs.insert(out.clone(), o + (s * row) as u32);
                            // rc 按输出形状重算（列=末维不变、行数去掉
                            // dim0 ——照搬输入会把行数多算 3 倍）
                            let outn = outn.unwrap();
                            if let Some(&last) = outn.shape.last() {
                                if last > 0 {
                                    let rows =
                                        outn.shape.iter().product::<i64>() / last;
                                    rc.insert(out.clone(), (rows as u32, last as u32));
                                }
                            }
                            continue;
                        }
                    }
                    // 非 dim0/非连续：落到 map_node 报不支持（Slice 无臂）
                }
                let is_alias = match n.op_type.as_str() {
                    "Reshape" => {
                        // 存储恒等条件：元素数不变（纯形状重排）。
                        // cls 尾部 Reshape([B,2]→[B,2]) 走这里；带 0/-1
                        // 推断的动态 reshape 若改 numel 会在下方显式报错。
                        let inn = table.values.get(&n.inputs[0]);
                        let outn = table.values.get(&out);
                        inn.map(|v| v.shape.iter().product::<i64>())
                            == outn.map(|v| v.shape.iter().product::<i64>())
                    }
                    "Squeeze" | "Unsqueeze" => true,
                    "Transpose" => {
                        let perm = n.attr("perm").map(|a| a.ints.clone()).unwrap_or_default();
                        // [0,2,1] 的存储恒等**条件**：输入存储 rc 与输出
                        // 逻辑 rc 一致（rows=行数、cols=末维）。small rec 的
                        // Transpose.8 输入是真转置产物（rc=[B*120,T]），
                        // 输出逻辑要 [B*T,120]——盲别名曾把词表头 A 读成
                        // 错误定向（输出全 NaN 的根源）。不满足即走通用
                        // 转置内核。
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
                    &rc,
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
            // 输出存储形状（n_ 模式）：rank-4 按形状；MatMul 按 [M,N]；
            // rank-3 逐元素/softmax/BN 继承输入的 rc（生产者记账）。
            let out_rc = if f32_mode {
                None
            } else {
                Some(
                    node_out_rc(n, &table, &rc)
                        .map_err(|e| Error::Graph(format!("{}: {e}", n.name)))?,
                )
            };
            let out_words = if f32_mode {
                out_n
            } else if out_v.shape.len() == 4 {
                super::nhwc::nhwc_words(&out_v.shape)
            } else {
                let (r, c) = out_rc.unwrap_or((out_n, super::nhwc::cpad4(1)));
                r * c
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
            if let Some(r) = out_rc {
                rc.insert(out.clone(), r);
            }

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
                    &mut rc,
                )?
                .into_iter()
                .map(|(k, pb, g)| (k, Some(pb), Vec::new(), g))
                .collect()
            };
            t_map_acc += t_map.elapsed().as_secs_f64();
            for (kernel, pb, pc_inline, groups) in routes {
                if dbg_env && n.name == "Conv.5" {
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
                &rc,
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
                    ledger.push((format!("p:<exit3:{name}>"), p_off, pb.len_words()));
                    params.push((p_off, pb.words().to_vec()));
                    recs.push(Rec {
                        kernel: if argmax { "n_exit3_argmax" } else { "n_exit3" },
                        node: format!("<exit3:{name}>"),
                        node_idx: usize::MAX,
                        out_off,
                        out_n: out_n_words,
                        pc: super::pipeline::PcParams { p_off }.bytes().to_vec(),
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
        // 池里找容量够的 (arena, KernelSet) 对 reset 复用；没有再新建。
        // 复用对的 KernelSet 绑定即 arena 块缓冲（reset 不换缓冲），免整套
        // DSL/描述符池/管线重建（实测 2-10ms → ~0）。
        let (arena, ks, whole) = {
            let mut pool = self
                .arena_pool
                .lock()
                .map_err(|_| Error::Device("arena 池锁中毒".into()))?;
            let alloc_whole = |a: &mut Arena| -> Result<super::memory::Region> {
                a.alloc(need).map_err(|e| {
                    Error::Device(format!("计划整块分配失败（{} floats）: {e}", layout.total))
                })
            };
            let mut hit = None;
            for i in 0..pool.len() {
                if pool[i].0.reset_if_fits(need) {
                    hit = Some(i);
                    break;
                }
            }
            match hit {
                Some(i) => {
                    let (mut a, ks) = pool.swap_remove(i);
                    let whole = alloc_whole(&mut a)?;
                    (a, ks, whole)
                }
                None => {
                    let mut a = Arena::new(
                        self.ctx.device.raw().clone(),
                        self.ctx.mem_types.staging,
                        true,
                    );
                    let whole = alloc_whole(&mut a)?;
                    let (buf, buf_size) = a
                        .chunk_range()
                        .ok_or_else(|| Error::Device("arena 空".into()))?;
                    let ks = KernelSet::new_with_cache(
                        self.ctx.device.raw(),
                        buf,
                        buf_size,
                        self.pipeline_cache,
                    )?;
                    (a, ks, whole)
                }
            }
        };

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
            // 头部全量内存屏障（每次重放都执行）：本机驱动的第三个
            // 一致性缺口——GPU 侧缓存会保留旧地址行，主机写穿到 DRAM
            // 也被它遮住（img-003 间歇整行空文本的现场：批内稀疏行、
            // 主机侧 sfence+clflush 输入发布无效）。屏障强制设备缓存
            // 对本次重放前的全部写入（含主机写）失效。代价每次重放
            // 一道屏障（µs 级）。
            let bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            dev.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[bar],
                &[],
                &[],
            );
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
            rn_word,
            inflight: std::sync::atomic::AtomicBool::new(false),
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
            ks: Some(ks),
            pool: std::sync::Arc::clone(&self.arena_pool),
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
        prof_print_with(&self.ctx, plan, pool, wall_ms);
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
        rc: &mut HashMap<String, (u32, u32)>,
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
                    // 寄存器分块：Co%8==0 走 m4×n8（两列共享输入 gather，
                    // 输入流量减半）；奇数列回 m4×n4 基线。
                    // n16（Co=16 全列进一线程）实测全面更慢：m4 版寄存器
                    // 崩（Conv.82 4×慢）、m2 版权重 L1 流量翻倍成新瓶颈
                    // （5×慢）——n8 是本机（Arc Pro，无 coopmat）的均衡点。
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
                    let xs = shape_of(&n.inputs[0]);
                    let r = xs.len() as i64;
                    let norm: Vec<i64> = axes
                        .iter()
                        .map(|&a| if a < 0 { a + r } else { a })
                        .collect();
                    if norm.len() == 1 && norm[0] == r - 1 {
                        // 末轴均值（small rec 的 LayerNorm 分解链）：一行 WG
                        // 归约，出 [rows,4]（lane0=均值）。rpb 供空行早退。
                        let rows: u32 =
                            xs[..xs.len() - 1].iter().product::<i64>() as u32;
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
                    //（GPU 浮点除法在本机有倒数截断缺陷，见 n_resize.comp）。
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
                        return Err(Error::Graph(format!(
                            "动态 MatMul K 不符：A={a:?} B={b:?}"
                        )));
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
                    return Ok(vec![(
                        "n_attn_mm",
                        pb,
                        [1, (batch as u32 * m).max(1), 1],
                    )]);
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
                    &self.initializers,
                    &n.inputs[1],
                    layout,
                    uploads,
                    w_offs,
                    &self.repack_cache,
                    |_| super::nhwc::repack_conv_w(&w_t, nn, k, 1, 1),
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
                    .u(super::nhwc::cpad4(k as i64) / 4)
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
                    let cp = super::nhwc::cpad4(c as i64) as usize;
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
                    super::nhwc::nhwc_words(&shape_of(&n.inputs[0]))
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
                    super::nhwc::nhwc_words(&a)
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
                            &self.graph,
                            table,
                            rc,
                            &self.initializers,
                            &n.inputs[1],
                        );
                        let lin0 = linear_by_producer(
                            &self.graph,
                            table,
                            rc,
                            &self.initializers,
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
                } else if a.len() == 3
                    && b.len() == 3
                    && b[2] == 1
                    && b[0] == a[0]
                    && b[1] == a[1]
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
                    super::nhwc::nhwc_words(&shape_of(&n.inputs[0]))
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
                    return Ok(vec![(
                        "n_channel",
                        pb,
                        [div256(rows * cpad / 4), 1, 1],
                    )]);
                }
                let a_words = super::nhwc::nhwc_words(&fs);
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
                            &self.graph,
                            table,
                            rc,
                            &self.initializers,
                            &n.inputs[0],
                        );
                        {
                            let s = rc.get(&n.inputs[1]).copied().unwrap_or(t);
                            let lin = linear_by_producer(
                                &self.graph,
                                table,
                                rc,
                                &self.initializers,
                                &n.inputs[1],
                            );
                            g_off =
                                orient4(layout, &mut pre, g_off, &gate_shape, s, t, lin, lin0)?;
                        }
                        let r_shape = shape_of(&n.inputs[2]);
                        {
                            let s = rc.get(&n.inputs[2]).copied().unwrap_or(t);
                            let lin = linear_by_producer(
                                &self.graph,
                                table,
                                rc,
                                &self.initializers,
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
                let _b_n = numel(table, &n.inputs[1]);
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
                } else if b.len() == 4 && b[2] == 1 && b[3] == 1 && b[0] == a[0] && b[1] == a[1] {
                    // [N,C,1,1] 通道广播（rec 批的 SE 门；n==b 的批维）
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
/// 操作数字节定向推断：沿存储恒等别名链（Reshape/Squeeze/Unsqueeze 与
/// 满足别名条件的 Transpose[0,2,1]）回溯到物质化生产者。真转置/动态 B
/// MatMul/dim0 切片按**真线性**（逻辑形行主序）写字节；其余（conv/元素
/// 族/reduce/softmax）按 NHWC 族（rc 记账）。Some(true)=真线性，
/// Some(false)=NHWC 族，None=无生产者（图输入/权重）或不可判。
/// 背景：C==W 时真线性 rc 与 NHWC rc **数值相同**（[N,C,1,W] 互为方阵
/// 转置≠恒等），rc 对照无法消歧——必须按生产者判（small T=120=C 时
/// 曾静默搅拌整图）。
fn linear_by_producer(
    graph: &Graph,
    table: &planner::ShapeTable,
    rc: &HashMap<String, (u32, u32)>,
    initializers: &HashMap<String, Tensor>,
    name: &str,
) -> Option<bool> {
    let mut cur = name.to_string();
    for _ in 0..graph.nodes.len() {
        let n = graph
            .nodes
            .iter()
            .find(|m| m.outputs.first() == Some(&cur))?;
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
    None
}

/// 逐元素操作数的**存储定向中转**：操作数字节序与基准不一致时（长跳跃
/// 相遇：NHWC 卷积链张量 × 真转置产物，small rec Add.203 的分类器全烂
/// 根因），把真线性一方用 n_transpose_nd（perm [0,2,3,1]）转成基准的
/// NHWC 族定向。判定按生产者定向（`linear_by_producer`）+ 形状序比较，
/// **不看 rc 数值等价**（C==W 时两序 rc 相同但字节互为方阵转置≠恒等，
/// small T=120=C 曾静默搅拌整图）。不可中转的组合显式报错（不静默
/// 搅拌）。仅支持 C%4==0：pad 通道的零填充中转不覆盖，宁可拒绝。
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

    fn bucket_grain(&self) -> i32 {
        // rec 每行一个裸宽 → 每形状一次 ~10ms 计划重建（曾把全 GPU 的
        // rec 拖到 2× CPU）。64 桶把语料宽度塌缩到 ~20 个形状，重建
        // 只付一次；右侧零填充的额外计算远小于重建税。
        // 例外：无注意力 mask 的模型（exact_width）填充会改真步输出
        // （非「额外计算」而是「错值」）——退精确宽，重建税交给计划缓存。
        if self.exact_width {
            1
        } else {
            64
        }
    }

    fn batch_grain(&self) -> i32 {
        self.batch_grain
    }

    fn rec_argmax_pairs(&self) -> bool {
        self.argmax_exit
    }

    fn run(&self, inputs: Vec<(String, Tensor)>) -> Result<Vec<Tensor>> {
        let staged = self.stage(inputs)?;
        if std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some() {
            self.stepped_run_and_cmp(&staged.0)?;
        }
        let t0 = std::time::Instant::now();
        self.ctx.device.wait_signal(staged.1)?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if let Some(p) = staged.0.qpool {
            self.prof_print(&staged.0, p, wall_ms);
        }
        staged
            .0
            .inflight
            .store(false, std::sync::atomic::Ordering::Release);
        Ok(readback_outputs(&staged.0, staged.2, staged.3))
    }

    fn run_deferred(
        &self,
        inputs: Vec<(String, Tensor)>,
    ) -> Result<Box<dyn qppocr_core::device::DeferredRun + Send>> {
        let (plan, signal, n_real, n_pad) = self.stage(inputs)?;
        Ok(Box::new(VulkanDeferred {
            ctx: self.ctx.clone(),
            plan,
            signal,
            n_real,
            n_pad,
        }))
    }
}

/// prof_print 的自由函数形态（run 与 VulkanDeferred::complete 共用）。
fn prof_print_with(ctx: &Arc<super::Inner>, plan: &Plan, pool: vk::QueryPool, wall_ms: f64) {
    let dev = ctx.device.raw();
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
    let (_, period) = ctx.device.timestamps();
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


/// 执行的后半段（run / deferred::complete 共用）：等信号后按真实行数
/// 拷出图输出（含 clflush 读回屏障与补齐缩行）。
fn readback_outputs(plan: &Plan, n_real: i64, n_pad: i64) -> Vec<Tensor> {
    let mut outs = Vec::with_capacity(plan.out_offs.len());
    for (i, (name, off, n)) in plan.out_offs.iter().enumerate() {
        let mut shape = plan.out_shapes.get(i).cloned().unwrap_or_default();
        // 补齐计划的真实行读回：只取前 n_real 行（批是外维、行连续，
        // 空行区是陈旧数据）。n 按 n_pad 建的——按比例缩回。
        let n_take: usize = if n_pad != n_real {
            n / n_pad.max(1) as usize * n_real as usize
        } else {
            *n
        };
        if !shape.is_empty() {
            shape[0] = n_real;
        }
        let mut buf = F32Buf::with_zeroed(n_take);
        // 读回屏障：本机驱动对 coherent 映射的设备写不做主机缓存一致
        //（vkInvalidate 无效），读前 clflush 逐出陈旧行（见 memory.rs）。
        // SAFETY: base 是整块持久映射；off+n_take ≤ total（计划保证）。
        super::memory::readback_clean(
            unsafe { plan.base.add(*off as usize * 4) } as *const u8,
            n_take * 4,
        );
        // SAFETY: 已等信号；off+n_take ≤ total。
        unsafe {
            std::ptr::copy_nonoverlapping(
                plan.base.add(*off as usize * 4) as *const f32,
                buf.as_mut_slice().as_mut_ptr(),
                n_take,
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
    outs
}

/// 批间流水的收账句柄：持有计划 Arc（防预算逐出悬空）与提交信号，
/// complete = 等信号 + 读回 + 解除在飞标记。
struct VulkanDeferred {
    ctx: Arc<super::Inner>,
    plan: std::sync::Arc<Plan>,
    signal: u64,
    n_real: i64,
    n_pad: i64,
}

impl qppocr_core::device::DeferredRun for VulkanDeferred {
    fn complete(self: Box<Self>) -> Result<Vec<Tensor>> {
        let t0 = std::time::Instant::now();
        self.ctx.device.wait_signal(self.signal)?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if let Some(p) = self.plan.qpool {
            prof_print_with(&self.ctx, &self.plan, p, wall_ms);
        }
        self.plan
            .inflight
            .store(false, std::sync::atomic::Ordering::Release);
        let t_rb = std::time::Instant::now();
        let out = readback_outputs(&self.plan, self.n_real, self.n_pad);
        if std::env::var_os("QPPOCR_GPU_BUILD_TIME").is_some() {
            eprintln!(
                "[gpu][dbg] complete：wait {wall_ms:.2} ms + readback {:.2} ms",
                t_rb.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(out)
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
    rc_free: &HashMap<String, (u32, u32)>,
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
                        Some(v) if !f32_mode && v.shape.len() == 4 => {
                            super::nhwc::nhwc_words(&v.shape)
                        }
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
        let mut cpu = Session::from_memory(&bytes, "tiny.det").unwrap();
        cpu.set_dump_dir(dump_dir.to_str().unwrap());
        let in_name = cpu
            .graph
            .inputs
            .iter()
            .find(|s| !s.is_empty())
            .cloned()
            .unwrap();
        let t0 = std::time::Instant::now();
        let out_cpu = cpu.run(vec![(in_name.clone(), cl(&in_name))]).unwrap();
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
