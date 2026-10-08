//! MetalSession：共享计划模型（`qppocr_gpu::plan`）的 Metal 执行侧。
//!
//! 与 VulkanSession 的对应关系：
//! - 计划构建**完全共用**（PlanBuilder → dispatch 列表 + 布局 + 上传
//!   清单）——两后端的算子覆盖、参数编码、区域布局逐位一致；
//! - 执行模型差异：Vulkan「装载期录 CB、每帧重放」；Metal 命令缓冲
//!   一次性——**每 run 重编码**（~200 dispatch ≈ 1ms 级编码开销，P4
//!   的 ICB/合并再优化）。dispatch 间的写读依赖经
//!   `memory_barrier_with_resources` 显式化（Vulkan 的隐式 dispatch
//!   顺序 Metal 不承诺）；
//! - 同步模型：`commit + wait_until_completed`（P2 串行执行；P3 批间
//!   流水再上 add_completed_handler，对应 Vulkan 的 deferred 对）；
//! - 统一内存（Apple Silicon 全系 UMA）：`StorageModeShared` 缓冲
//!   CPU/GPU 同址，输入写入与输出读回**零 staging 零 clflush**
//!   （Vulkan 侧的三类驱动一致性缺口在 Metal 上结构性不存在）。

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use qppocr_core::device::{DeviceKind, DeviceSession};
use qppocr_core::error::{Error, Result};
use qppocr_core::onnx::model::Graph;
use qppocr_core::tensor::{DType, Tensor};
use qppocr_gpu::plan::{self, PlanBuilder};
use qppocr_kernels::buf::F32Buf;

use super::device::submit_compute_timed;
use super::pipeline::KernelSet;

/// `run` 的调用序号（跨会话全局递增）。切块诊断用它区分"同进程第几次
/// 执行"——挂死现象与"第二次 in-process run"高度相关，序号是判据的一部分。
static RUN_SEQ: AtomicU64 = AtomicU64::new(0);

/// 一个输入形状的整图计划（Metal 实体化）。
///
/// `pub(crate)` + 字段可见：`prof` 需要直接读 dispatch 列表与 arena 做
/// 设备端计时；后端模块外部仍不可见。
pub(crate) struct MetalPlan {
    /// 整块 Shared arena（统一内存：CPU/GPU 同址）。
    pub(crate) buf: metal::Buffer,
    /// `buf.contents()`——裸指针缓存（写输入/读输出用）。
    base: *mut u8,
    /// 本计划所属的 arena（= `buf` 的所有者）。**判「我那片静态字节还在不在」
    /// 的账本挂在这里**，而不是挂会话：arena 增长时旧计划留在旧 arena 里，
    /// 与新的那批互不干扰（`ensure_statics` 的依据）。
    arena: Arc<Mutex<Arena>>,
    /// 静态区的**写入清单**（顺序 = build 期 uploads → upload_init）。
    /// 缓存命中时要靠它把自己那片字节重新落地——见 `ensure_statics`。
    /// 权重项是 `repack_cache` 里同一个 `Arc`（不额外占内存）。
    statics: Vec<(u32, Arc<Vec<u32>>)>,
    /// `statics` 的标识视图（按偏移排序，(偏移, 字数, `Arc` 指针)），
    /// 与 `Arena::blobs` 同构——两者相等即「此刻 arena 里那片字节就是本
    /// 计划的」。排序后可直接 `==` 比较。
    recipe: Vec<(u32, u32, usize)>,
    /// 图输入的 f32 NCHW 区偏移（host 每次 memcpy 的目标）。
    in_f32_off: u32,
    /// real_n 单元的 word 偏移（构造上恒 0）。
    rn_word: u32,
    /// 图输出：(名字, 偏移, 元素数)。
    out_offs: Vec<(String, u32, usize)>,
    out_shapes: Vec<Vec<i64>>,
    /// dispatch 序列（每 run 重编码执行）。
    pub(crate) dispatches: Vec<plan::Dispatch>,
    /// 整块字节数（计划缓存预算记账）。
    block_bytes: u64,
}

// SAFETY: base 是 buf（Shared arena）的持久映射指针，存活期与 MetalPlan
// 相同且无并发写（run 全程持会话执行锁）；metal owned 类型 Send+Sync。
unsafe impl Send for MetalPlan {}
// SAFETY: 同上——&MetalPlan 的共享只发生在持锁的 run 里。
unsafe impl Sync for MetalPlan {}

/// 会话级常驻 arena：同图不同形状**复用同一块设备内存**。
///
/// 静态段（权重 + 参数块）与形状无关——rec 每来一个行宽就是一个新形状，
/// 今天每建一次计划就 `new_buffer(146 MB)` + 重传 60.7 MB 权重：实测
/// 「建 7.4 ms」里 ~4 ms 是那 39.7 MB 的 init 权重重传、~1.4 ms 是重复分配。
/// 一图 N 行就是 N 次这样的浪费。
///
/// 复用判据（保守：任一不满足就退化成「新分配 + 全量重传」= 今天的行为）：
/// - 容量够（`cap_words >= total_words`）；
/// - `static_words` 逐位相同（同图必然成立）；
/// - 上一份计划的静态 blob **分区**（排序后的 (偏移, 字数) 序列）与之完全相同
///   ——分区相同才保证「同一偏移处的字节」语义可比。
///
/// 分区相同后逐个 blob 比**内容标识**（`Arc` 指针，出自重排缓存）：
/// 标识相同 ⇒ 那片字节就是本次要写的内容，跳过。参数块/RN 单元每次新建
/// （标识必不同）⇒ 照写。
///
/// ⚠️ **`blobs` 的语义是「最后触碰本 arena 的那个计划的静态清单」**，不是
/// 「上一份计划的」（`plan.rs:1800-1806` 的「静态区只在装载期写一次」只在
/// **一个计划独占一块 arena** 时成立，而这里会话级共享——Vulkan 侧每个
/// `Plan` 私有 arena，所以那边没有这个问题）。
///
/// 跨 build 时静态区**从偏移 0 重新分配**（`Layout::alloc_static` 与 `alloc`
/// 共用 `cursor`，每次 `Layout::new()` 归零），于是 build j 的**激活区**可以
/// 压在 build i 的**参数块**上——实测 `[1,3,48,384]` 的 6 个参数块有
/// 1536/1536 字节落在 `4x3x48x256` / `3x3x48x512` / `1x3x48x1024` 的激活区内。
/// 参数块是**形状相关**的（编码 stride/shape），没有任何布局技巧能让两个
/// 计划共用同一偏移。所以跨 build 的完整性只能靠**每次执行前重放**，见
/// `MetalSession::ensure_statics`：build 与 run **都**更新本清单（run 的激活
/// 写也会压掉别人的静态区，所以 run 同样是一次「触碰」）。
struct Arena {
    buf: metal::Buffer,
    base: *mut u8,
    cap_words: u32,
    static_words: u32,
    /// 最后触碰本 arena 的计划写下的静态 blob：(偏移, 字数, 内容标识)
    blobs: Vec<(u32, u32, usize)>,
}

// SAFETY: 同 MetalPlan——base 是 buf 的持久映射指针，存活期不短于 Arena
// 且只在会话执行锁下访问；metal owned 类型 Send+Sync。
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

/// Metal GPU 会话：形状键计划缓存（P2 det 范围——串行执行；rec/cls 的
/// 批维补齐/合批/argmax 出口/影子计划在 P3 接入）。
pub(crate) struct MetalSession {
    ctx: Arc<super::Inner>,
    graph: Graph,
    initializers: HashMap<String, Tensor>,
    input_name: String,
    /// 会话级内核套件：Metal 的 set_buffer 在编码期绑 arena（与 Vulkan
    /// 的 KernelSet 绑死 arena 块不同）——一套管线服务全部计划。
    ks: KernelSet,
    repack_cache: Mutex<HashMap<String, Arc<Vec<u32>>>>,
    producer_idx: std::sync::OnceLock<HashMap<String, usize>>,
    plans: Mutex<Vec<(Vec<i64>, Arc<MetalPlan>)>>,
    /// 常驻 arena（见 `Arena`）：整会话一块，形状间复用。`Arc` 是因为每个
    /// 计划要**持有自己那块 arena 的账本引用**（判静态区是否还在位），
    /// 而 arena 增长换代时旧计划留在旧 arena 里、互不干扰。
    arena: Mutex<Option<Arc<Mutex<Arena>>>>,
    /// 计划缓存的字节预算（MB，每会话独立；QPPOCR_GPU_PLAN_MB 覆写）
    /// ——create_session 按角色设定（det 小 / rec 大），镜像 Vulkan 侧。
    pub(crate) plan_budget_mb: u64,
    /// 批维补齐粒度（rec 会话=8，其余 1）：run() 把输入批维向上取整到
    /// 该值建计划，空行由内核 real_n 早退守卫零算力——(bsz,W) 形状塌缩
    /// 成 (grain,W桶)，计划全命中。与 `DeviceSession::batch_grain` 同源
    /// （mod.rs 按模型角色设置；引擎按 trait 值分批）。
    pub(crate) batch_grain: i32,
    /// 宽度精确模式（无注意力 mask 的 rec）：桶宽右零填充经注意力混入
    /// 真步概率（模型对填充不 invariant，CPU 实证掉行）——true 时
    /// bucket_grain=1（精确宽，同宽行仍合批）。纯卷积 rec（tiny）填充
    /// 不变，维持 64 桶。
    pub(crate) exact_width: bool,
    /// rec 会话的 CTC argmax 出口：rank-3 概率输出改 n_exit3_argmax 每
    /// 时间步写 (val, idx) 对——读回从 T×V×4 B/行降到 8 B/行。引擎按
    /// `rec_argmax_pairs` 探测并走 ctc_decode_pairs。
    pub(crate) argmax_exit: bool,
    /// 执行串行锁：每 run 重编码 + 同步等待不支持并发在飞（deferred
    /// 流水是 P4 的性能项；trait 默认的同步 run_deferred 语义等价）。
    run_lock: Mutex<()>,
    /// 逐内核剖析的剩余次数（`QPPOCR_GPU_KERNEL_PROF`，0 = 关闭）。
    prof_left: std::sync::atomic::AtomicUsize,
}

/// 网格上限默认值（单位：线程组数，每组 256 线程）。
///
/// **默认 0 = 不夹**，即回到「每线程恰好吃一个 tile」的行为。
///
/// 夹网格这套机制保留着（内核里写了 stride 循环，计划侧有 `grid_cap`），
/// 但**实测在真实流水线上没有收益、反而略亏**。det 前向 5 次取最小：
///
/// | cap | 总线程 | det 墙钟 |
/// |---|---|---|
/// | 0   | 不限      | **1474.7 ms** |
/// | 64  | 16384     | 1506.1 ms |
/// | 256 | 65536     | 1559.4 ms |
/// | 32  | 8192      | 1596.2 ms |
/// | 16  | 4096      | 1875.3 ms |
///
/// 只有 cap=16 明显变差，其余都在噪声里，而 cap=0 名义最好。
///
/// 为什么会这样：下面那张「线程越少越快」的表是**长跑**（MB~GB 级流量）
/// 的性质，而真实内核是**短突发**——det 154 条 dispatch / 1475 ms ≈
/// 9.6 ms 一条，吃的是每条 dispatch 的固定开销与访存停顿，不是稳态带宽。
/// 所以按稳态带宽曲线去夹网格并不对症。（另外 det 自己的字节流量只有约
/// 1.2 GB，按 24 GB/s 算也才 ~50 ms，本来就不是带宽瓶颈。）
///
/// 那条长跑曲线本身是可靠的——256 MB 合并访存 + 手工展开 + **逐字节校验
/// 拷贝确实完成**的内核，在本机实测：
///
/// | 总线程数（组数） | 256 MB 拷贝 |
/// |---|---|
/// | 8K (32 组)      | **24 GB/s** |
/// | 32K (128 组)    | 13 GB/s |
/// | 262K (1024 组)  | 8.8 GB/s |
/// | 2M (8192 组)    | 3.9 GB/s |
///
/// 同机 CPU memcpy 同一 256 MB 是 **136 GB/s**。
///
/// 顺带更正一个曾被引用的数：旧记录里「空转内核 0.7~2.7 µs/组、与内核
/// 算什么无关」站不住——用 v1 探针重测（网格取自 det 真实取值），空转
/// dispatch 是 **0.15~3.4 ms/条且与组数不同向**（9720 组 0.15 ms、
/// 288 组 2.7 ms），说明斜率法在这个粒度上的噪声地板就有 ~1 ms/条，
/// 比要测的信号还大。**逐内核的绝对值不可信，只有总量可信。**
const GROUP_CAP_DEFAULT: u32 = 0;

/// 网格上限：`QPPOCR_METAL_GROUP_CAP` 显式给值（`0` = 关，回到每线程一个
/// tile 的旧行为）时用它，否则用 [`GROUP_CAP_DEFAULT`]（默认 0 = 不夹）。
///
/// 留这个开关是为了**扫参**：夹网格是对计划形状的改动，得有一条「关掉就
/// 逐位回到改前」的路，否则量到差异时说不清是哪一步带来的。夹与不夹的
/// 输出**逐位相同**（stride 只改变哪个线程算哪个 tile，不改变任何加法次序）。
fn group_cap() -> u32 {
    match std::env::var("QPPOCR_METAL_GROUP_CAP") {
        Ok(v) => v.trim().parse().unwrap_or(GROUP_CAP_DEFAULT),
        Err(_) => GROUP_CAP_DEFAULT,
    }
}

impl MetalSession {
    pub(crate) fn new(
        ctx: Arc<super::Inner>,
        graph: Graph,
        initializers: HashMap<String, Tensor>,
    ) -> Result<Self> {
        let ks = KernelSet::new(&ctx.device)?;
        let input_name = graph
            .inputs
            .iter()
            .find(|s| !s.is_empty())
            .cloned()
            .unwrap_or_default();
        if std::env::var_os("QPPOCR_DUMP_DIR").is_some() {
            eprintln!(
                "[metal] QPPOCR_DUMP_DIR 已设置：GPU 会话不落盘逐节点 dump；\
                 节点级诊断请用 --device cpu"
            );
        }
        Ok(Self {
            ctx,
            graph,
            initializers,
            input_name,
            ks,
            repack_cache: Mutex::new(HashMap::new()),
            producer_idx: std::sync::OnceLock::new(),
            plans: Mutex::new(Vec::new()),
            arena: Mutex::new(None),
            // det 形状逐图一次性、共享计划重建便宜（无 arena 池化缺口，
            // 权重重排有缓存）→ 小预算；env 覆写。
            plan_budget_mb: 384,
            batch_grain: 1,
            exact_width: false,
            argmax_exit: false,
            run_lock: Mutex::new(()),
            prof_left: std::sync::atomic::AtomicUsize::new(
                std::env::var("QPPOCR_GPU_KERNEL_PROF")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
            ),
        })
    }

    /// 覆盖面探针：小形状试建一次计划（不出执行）——算子支持与形状
    /// 无关，不支持的算子在此点名（create_session 分级用）。
    pub(crate) fn probe(&self, in_shape: &[i64]) -> Result<()> {
        self.build_plan(in_shape).map(|_| ())
    }

    /// 探针失败后的退路：把吃进去的 (graph, initializers) 原样吐还，
    /// 供 CPU 会话组装（Metal 资源随 Drop 释放；probe 计划不入缓存）。
    pub(crate) fn into_cpu_parts(self) -> qppocr_core::executor::Session {
        let mut me = self;
        let graph = std::mem::take(&mut me.graph);
        let initializers = std::mem::take(&mut me.initializers);
        drop(me);
        qppocr_core::executor::Session::from_parts(graph, initializers)
    }

    fn producer_idx(&self) -> &HashMap<String, usize> {
        self.producer_idx.get_or_init(|| {
            self.graph
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(i, n)| n.outputs.first().map(|o| (o.clone(), i)))
                .collect()
        })
    }

    /// 诊断（**零 GPU 风险**）：按形状序列连续建计划，但**不派发任何 dispatch**。
    ///
    /// 走 `plan_for`，所以计划进缓存、持有 arena 引用，与正式路径的行为一致。
    /// 用途：复现与核对「同一块 arena 上的第 N 次 build」——正式路径要跑完整
    /// 流水线才走得到那里，而那条路一旦出错，整台机器的 GPU 都要重启才能恢复。
    /// 配合 `QPPOCR_METAL_ARENA_DUMP` / `QPPOCR_METAL_ARENA_FILL` 使用。
    ///
    /// **只由 `kernel_tests::arena_reuse_replay_det` 调用**，所以整条链（连同它
    /// 依赖的 `statics_in_place`）都挂在 `cfg(test)` 下——否则非测试构建里它们
    /// 是死代码，`cargo check --all-targets` 会一直报 unused。
    #[cfg(test)]
    pub(crate) fn debug_replay_builds(&self, shapes: &[Vec<i64>]) -> Result<()> {
        let _guard = self
            .run_lock
            .lock()
            .map_err(|_| Error::Device("执行锁中毒".into()))?;
        let mut built: Vec<(Vec<i64>, Arc<MetalPlan>)> = Vec::new();
        for s in shapes {
            let p = self.plan_for(s)?;
            built.push((s.clone(), p));
        }
        // ① 全表核对：这一串计划建完之后，每份计划的静态区还在不在位。
        // 不在位 = 这份计划若此刻**命中缓存**直接派发，消费方会读到别人的数据。
        // 这一步只读内存，不派发；判据完全独立（按计划自己的偏移读回 arena）。
        let mut clobbered = 0usize;
        for (s, p) in &built {
            let (b, n, y, yt) = self.statics_in_place(p)?;
            if b != n {
                clobbered += 1;
                eprintln!(
                    "[verify] ✖ {s:?} 静态区已被冲掉：{b}/{n} 块在位、\
                     {:.2}/{:.2} MB —— 该计划命中的话会读到垃圾参数",
                    y as f64 * 4.0 / 1e6,
                    yt as f64 * 4.0 / 1e6
                );
            }
        }
        eprintln!(
            "[verify] 建完 {} 份计划：{clobbered} 份静态区被冲掉（未重放）",
            built.len()
        );
        // ② 逐份走 `ensure_statics`（正式路径在每次派发前都走这一步），并在
        // **紧接着它之后**核对——这才是运行期的语义：任何一个计划被派发时，
        // 它自己那片字节必须齐全；不同计划的静态区本来就可能互相覆盖（参数块
        // 是形状相关的，同一偏移不可能两用），所以「同时都在位」不是要求。
        let mut restored = 0usize;
        for (s, p) in &built {
            self.ensure_statics(p)?;
            let (b, n, y, yt) = self.statics_in_place(p)?;
            if b == n && y == yt {
                restored += 1;
            } else {
                eprintln!("[verify] ✖ 重放未恢复 {s:?}：{b}/{n} 块在位");
            }
        }
        if restored != built.len() {
            return Err(Error::Device(format!(
                "静态区重放未恢复：{restored}/{} 份——缓存命中的计划仍会读到陈旧字节",
                built.len()
            )));
        }
        eprintln!(
            "[verify] 逐份 `ensure_statics` 之后：{restored}/{} 份静态区全部在位",
            built.len()
        );
        Ok(())
    }

    fn build_plan(&self, in_shape: &[i64]) -> Result<MetalPlan> {
        let t0 = std::time::Instant::now();
        let built = PlanBuilder {
            graph: &self.graph,
            initializers: &self.initializers,
            repack_cache: &self.repack_cache,
            producer_idx: self.producer_idx(),
            argmax_exit: self.argmax_exit,
            // 网格上限：默认 [`GROUP_CAP_DEFAULT`] = 0（不夹）。夹网格在真实
            // 流水线上实测无收益、略亏，理由与实测表见该常量的文档。
            grid_cap: group_cap(),
        }
        .build(&self.input_name, in_shape)?;

        // 本次静态区的**写入清单**（顺序 = build 期写序：`uploads` → `upload_init`）。
        // 缓存命中时要靠它重放自己那片字节（`ensure_statics`），所以计划里必须
        // 留一份——权重项就是重排缓存里的同一个 `Arc`，`clone` 只加引用计数，
        // 不复制数据；参数块很小（KB 级）。
        let statics: Vec<(u32, Arc<Vec<u32>>)> = built
            .uploads
            .iter()
            .chain(built.upload_init.iter())
            .map(|(off, d)| (*off, Arc::clone(d)))
            .collect();
        // 静态 blob 的标识清单（按偏移排序）——与 `Arena::blobs` 同构。
        //
        // **含参数块**（旧版只列 `upload_init`）：参数块的标识每次 build 新建，
        // 进清单只会让「跨 build 命中跳过」失效（本来也不该跨 build 信任参数
        // 块），但换来一条精确等式——`Arena::blobs == plan.recipe` 即「此刻
        // arena 里那片字节就是本计划的」，缓存命中的重传判据靠它。
        let mut cur: Vec<(u32, u32, usize)> = statics
            .iter()
            .map(|(off, v)| (*off, v.len() as u32, Arc::as_ptr(v) as usize))
            .collect();
        cur.sort_unstable();

        // 分配/复用 arena：容量够就沿用（旧 arena 若仍被既有计划持有则
        // 自然存活——MetalPlan 各持一份引用）。
        let bytes = built.total_words as u64 * 4;
        // 诊断开关：停用 arena 复用（每次新建 buffer、全量上传）——用来
        // 把「复用的代价」与「复用引入的错」分开定位。
        let share = std::env::var_os("QPPOCR_METAL_NO_ARENA").is_none();
        let (arena, buf, base, prev) = {
            let mut guard = self
                .arena
                .lock()
                .map_err(|_| Error::Device("arena 锁中毒".into()))?;
            // 复用前提：现行 arena 的容量够装下本计划。容量读进来就要放锁——
            // 下面写字节的阶段不能再持会话的 arena 锁（那会把「重放静态区」
            // 也串进来，而重放走的是计划自己那份 `Arc`）。
            let cur_cap = match guard.as_ref() {
                Some(a) => Some(
                    a.lock()
                        .map_err(|_| Error::Device("arena 锁中毒".into()))?
                        .cap_words,
                ),
                None => None,
            };
            if !share || !cur_cap.is_some_and(|c| c >= built.total_words) {
                // 新分配：留 25% 余量，且**至少翻倍上一版容量**——rec 的
                // 行宽逐个变，计划尺寸跟着跳（探针 63 MB → 一行 138 MB）；
                // 只留 25% 会一路「刚分配又不够」，每次都全量重传。翻倍
                // 让一个会话最多长几次就稳定（实测 63→79→317，此后全命中）。
                //
                // ⚠️ 翻倍只在**复用**模式下成立。非复用模式（NO_ARENA）每份
                // 计划都新建 arena 并把它自己写回 guard，`old` 于是变成上一份
                // 计划自己——容量逐计划翻倍、无界增长，最终 new_buffer 申请不
                // 到存储。本仓实证：崩在 build_plan 的 _platform_memmove，
                // KERN_INVALID_ADDRESS at 0x0，整批 worker 一起死。
                let old = if share { cur_cap.unwrap_or(0) as u64 } else { 0 };
                let cap = (built.total_words as u64 + built.total_words as u64 / 4 + 1024)
                    .max(old * 2);
                // cap_words 是 u32；翻倍链一旦越界，这里必须报错而不是静默截断
                // （截断后记录的容量小于实际分配，后续计划的越界自查会失准）。
                if cap > u32::MAX as u64 {
                    return Err(Error::Device(format!(
                        "arena 容量越界：{:.1} MB > u32 上限",
                        cap as f64 * 4.0 / (1024.0 * 1024.0)
                    )));
                }
                let nb = self
                    .ctx
                    .device
                    .new_buffer(cap * 4, metal::MTLResourceOptions::StorageModeShared);
                let b = nb.contents() as *mut u8;
                // Metal 申请不到存储时 new_buffer 仍返回一个对象，但 contents()
                // 是**空指针**；直接 memmove 进去就是段错误。宁可返回错误也不要
                // 让整个批的 worker 被信号带走。
                if b.is_null() {
                    return Err(Error::Device(format!(
                        "arena 分配失败：{:.1} MB（Metal 未给出可寻址存储，通常是显存/内存不足）",
                        cap as f64 * 4.0 / (1024.0 * 1024.0)
                    )));
                }
                *guard = Some(Arc::new(Mutex::new(Arena {
                    buf: nb,
                    base: b,
                    cap_words: cap as u32,
                    static_words: built.static_words,
                    blobs: Vec::new(),
                })));
            }
            // 取走 arena 的共享句柄：本计划会持有它（账本的判据挂在 arena 上），
            // 之后所有对本 arena 的写都走这份 `Arc`，不再碰会话那层锁。
            let a = guard.as_ref().expect("arena 必已就绪").clone();
            let (buf, base, blobs) = {
                let g = a.lock().map_err(|_| Error::Device("arena 锁中毒".into()))?;
                (g.buf.clone(), g.base, g.blobs.clone())
            };
            (a, buf, base, blobs)
        };
        // 诊断开关：把 arena 整块填成统一字节。
        //
        // 用法是与另一次「填**不同**字节」的运行对拍：两次 dump 里**不一致**的
        // 字节，就是本次 build 从开始到写完**从未写过**的字节。如果这些字节被
        // shader 读到，这个计划就依赖了「新分配的 buffer 是干净的」—— 而 Metal
        // **不保证** new_buffer 清零。新 buffer 上它侥幸成立；**复用 arena 时
        // 那些字节是上一张图的残留**，于是同样一张图第一次跑正常、第二次读到
        // 垃圾 → 巨循环 → 挂死（本仓实测形态，见 plan.rs:1804）。
        //
        // 填充会作废跳过集声称的内容，所以一并清空 prev（强制全量上传）。
        let prev = match std::env::var("QPPOCR_METAL_ARENA_FILL").ok().and_then(|v| {
            u8::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()
        }) {
            Some(fb) => {
                // SAFETY: base 指向 cap_words 个 word 的映射，而 bytes = total_words*4
                // 且复用/新建都保证 cap_words >= total_words（上方判断）。
                unsafe { std::ptr::write_bytes(base, fb, bytes as usize) };
                Vec::new()
            }
            None => prev,
        };
        // 越界自查（与 Vulkan 侧同款纪律：写越界 = 设备级故障）
        for (off, data) in &built.uploads {
            assert!(
                off + data.len() as u32 <= built.total_words,
                "静态上传越界: {off}+{} > {}",
                data.len(),
                built.total_words
            );
        }
        for (off, data) in &built.upload_init {
            assert!(
                off + data.len() as u32 <= built.total_words,
                "静态上传越界: {off}+{} > {}",
                data.len(),
                built.total_words
            );
        }
        // 跳过集 = **最后触碰本 arena 的计划**的 (偏移, 字数, 标识)。三项全等
        // 即「那片字节在位」：它写过同样内容在同一区间，且在其后没有任何写落
        // 在那上面（它自己的 run 的激活区不与自己的静态区重叠——同 build 内
        // `alloc_static` 与 `alloc` 不重叠，见 plan.rs 的 `alloc_static`）。
        //
        // ⚠️ 旧注释写的是「上一份计划……两次计划之间没有任何写落在它上面」，
        // 那是**假前提**：共享 arena 下别的计划既会写自己的静态区（`Layout`
        // 每次从偏移 0 重新分配），它 run 时的激活区还会压在别人的参数块上。
        // 清单本身没错，错的是「跨计划的命中可以无条件信任」这一步——参数块
        // 标识每次 build 新建，天然被排除在跨 build 跳过之外；权重则由
        // `ensure_statics` 在每次执行前重新对账。
        let mut skip: std::collections::HashSet<(u32, u32, usize)> =
            cur.iter().filter(|b| prev.contains(b)).copied().collect();
        if std::env::var_os("QPPOCR_METAL_STATIC_DEBUG").is_some() {
            let mut by_off = 0usize;
            let mut by_id = 0usize;
            for b in &cur {
                if let Some(p) = prev.iter().find(|p| p.0 == b.0 && p.1 == b.1) {
                    by_off += 1;
                    if p.2 == b.2 {
                        by_id += 1;
                    }
                }
            }
            eprintln!(
                "[static] {in_shape:?}: cur={} prev={} 命中={}（同偏移 {} / 同标识 {}）",
                cur.len(),
                prev.len(),
                skip.len(),
                by_off,
                by_id
            );
        }
        if std::env::var_os("QPPOCR_METAL_STATIC_DEBUG").is_some() {
            // 上传体量分布：`uploads` 每形状照写，`upload_init` 可跳。两清单
            // 谁在扛 20+ MB、是「少数巨块」还是「173 个小块堆出来的」，直接
            // 决定还有没有优化空间——离散量，不受机器负载窗影响。
            let mut up: Vec<(u32, u32)> =
                built.uploads.iter().map(|(o, d)| (*o, d.len() as u32)).collect();
            up.sort_by_key(|b| std::cmp::Reverse(b.1));
            let up_bytes: u64 = built.uploads.iter().map(|(_, d)| d.len() as u64).sum();
            let init_bytes: u64 = built.upload_init.iter().map(|(_, d)| d.len() as u64).sum();
            eprintln!(
                "[static] {in_shape:?}: uploads {} 块/{:.2} MB，upload_init {} 块/{:.2} MB；\
                 uploads 最大 6 块: {:?}",
                built.uploads.len(),
                up_bytes as f64 * 4.0 / 1e6,
                built.upload_init.len(),
                init_bytes as f64 * 4.0 / 1e6,
                up.iter().take(6).map(|(o, n)| (*o, *n)).collect::<Vec<_>>()
            );
        }
        // SAFETY: base 是整块映射；off+len ≤ total（上方断言）。
        let (mut wrote_words, mut skipped_words) = (0u64, 0u64);
        // SAFETY: base 是整块映射；off+len ≤ total（上方断言）。
        unsafe {
            for (off, data) in &statics {
                let id = Arc::as_ptr(data) as usize;
                if skip.remove(&(*off, data.len() as u32, id)) {
                    skipped_words += data.len() as u64;
                    continue;
                }
                wrote_words += data.len() as u64;
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    base.add(*off as usize * 4) as *mut u32,
                    data.len(),
                );
            }
        }
        // 诊断开关：把 build **写完之后**的 arena 落盘（只 dump 计划真正使用的
        // [0, total_words)）。配合 QPPOCR_METAL_ARENA_FILL 做两次运行对拍：
        // 不一致的字节 = 本次 build 从未写过。文件名带序号与形状，方便按序对齐。
        if let Some(prefix) = std::env::var_os("QPPOCR_METAL_ARENA_DUMP") {
            static SEQ: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let idx = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let shape = in_shape
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("x");
            let path = format!("{}.{idx:03}.{shape}.bin", prefix.to_string_lossy());
            // SAFETY: base 有效且 [0, total_words) 在映射内（同上断言）。
            let slice = unsafe { std::slice::from_raw_parts(base, bytes as usize) };
            match std::fs::write(&path, slice) {
                Ok(()) => eprintln!(
                    "[arena-dump] #{idx:03} {in_shape:?} total={} word static={} \
                     写 {:.1}/跳 {:.1} MB → {path}",
                    built.total_words,
                    built.static_words,
                    wrote_words as f64 * 4.0 / 1e6,
                    skipped_words as f64 * 4.0 / 1e6
                ),
                Err(e) => eprintln!("[arena-dump] #{idx:03} 写 {path} 失败：{e}"),
            }
        }
        // 记录本次落地后的 blob 清单（供下一形状/下一次执行比对）。**必须在写
        // 完之后再记**：它描述的是「此刻 arena 里那片字节的内容」，同时是
        // `ensure_statics` 判定「本计划的静态区还在位」的依据。
        {
            let mut g = arena
                .lock()
                .map_err(|_| Error::Device("arena 锁中毒".into()))?;
            g.static_words = built.static_words;
            g.blobs = cur.clone();
        }
        // 静态段占比：`total` 里权重/参数占多少。形状无关的权重如果按形状
        // 各存一份，N 个桶形状就是 N 份——这是「整块 120 MB 而输入才 1 MB」
        // 的头号嫌疑。QPPOCR_METAL_PLAN_DUMP 打全量 dispatch 表。
        let static_mb = built.static_words as f64 * 4.0 / 1e6;
        eprintln!(
            "[metal] 计划就绪：输入 {in_shape:?}，{} 个 dispatch，整块 {:.1} MB\
             （静态 {static_mb:.1} MB，建 {:.1} ms，静态上传 写 {:.1}/跳 {:.1} MB）",
            built.dispatches.len(),
            bytes as f64 / 1e6,
            t0.elapsed().as_secs_f64() * 1000.0,
            wrote_words as f64 * 4.0 / 1e6,
            skipped_words as f64 * 4.0 / 1e6
        );
        if std::env::var_os("QPPOCR_METAL_STATIC_DEBUG").is_some() {
            // 复用 arena 的前提：**同图不同形状的静态布局逐位相同**。
            // 布局哈希 = 各静态 blob 的 (偏移, 长度) 序列——这是判据。
            use std::hash::{Hash, Hasher};
            let mut h_init = std::collections::hash_map::DefaultHasher::new();
            let mut w_init = 0u64;
            for (off, v) in &built.upload_init {
                (off, v.len()).hash(&mut h_init);
                w_init += v.len() as u64;
            }
            let mut h_up = std::collections::hash_map::DefaultHasher::new();
            let mut w_up = 0u64;
            for (off, v) in &built.uploads {
                (off, v.len()).hash(&mut h_up);
                w_up += v.len() as u64;
            }
            eprintln!(
                "[static] 形状 {in_shape:?} static_words={} \
                 upload_init={} blob/{:.2} MB hash={:x} · uploads={} blob/{:.2} MB hash={:x}",
                built.static_words,
                built.upload_init.len(),
                w_init as f64 * 4.0 / 1e6,
                h_init.finish(),
                built.uploads.len(),
                w_up as f64 * 4.0 / 1e6,
                h_up.finish(),
            );
        }
        if std::env::var_os("QPPOCR_METAL_PLAN_DUMP").is_some() {
            // 卷积步额外打 `taps / cin4 / co4 / m`。
            //
            // 为什么非加不可：只看 `grid` 与 `out` **分不出**一个 conv 是 1×1
            // 还是 3×3，而两者的最优内核结构恰好相反 —— 窗口序（`p_window`）
            // 在 1×1 上是 2.45× 的赢，在同分辨率的 3×3 上却是 3.1× 的输。
            // 没有 taps 就会照错的对象下手（det 83 个 conv 里 61 个是 1×1，
            // 光看 channel 数猜不出来）。
            //
            // 参数块直接从宿主副本 `uploads` 按偏移取 —— 纯数据，不碰 GPU 缓冲。
            // 各步布局见 shader 头部；`n_conv8`/`n_conv` 同构，`n_conv_dw` 不同。
            let pubs: std::collections::HashMap<u32, &Vec<u32>> =
                built.uploads.iter().map(|(o, v)| (*o, &**v)).collect();
            for (i, d) in built.dispatches.iter().enumerate() {
                let extra = pubs
                    .get(&d.p_off)
                    // 认一下这确实是我这一步的参数块：params[P+3] 是 out_off。
                    .filter(|p| p.get(3).copied() == Some(d.out_off))
                    .map_or(String::new(), |p| {
                        let g = |k: usize| p.get(k).copied().unwrap_or(0);
                        match d.kernel {
                            "n_conv8" | "n_conv" => format!(
                                "  cin={:<4} co={:<4} taps={} ({}x{})  m={}",
                                g(9) * 4,
                                g(5),
                                g(10),
                                g(10) / g(11).max(1),
                                g(11),
                                g(4)
                            ),
                            "n_conv_dw" => format!(
                                "  c={:<4} taps={} ({}x{})  m={}",
                                g(5),
                                g(9),
                                g(9) / g(10).max(1),
                                g(10),
                                g(4)
                            ),
                            _ => String::new(),
                        }
                    });
                // `nid` = 图节点序号，**与 `node` 名后缀无关**（后者是 ONNX 名字，
                // 形如 `Conv.20`，编号来自算子计数器不是图序）。逐内核剖析器打的
                // `node#N` 用的就是 `d.node_idx`，不带 `nid` 就**无法**把剖析出的
                // 热点对回具体一行——这个坑已经踩过两轮了，别再删。
                eprintln!(
                    "[plan] #{i:<4} nid={nid:<5} {:<16} grid={:<12} out={:?} node={}{extra}",
                    d.kernel,
                    format!("{:?}", d.groups),
                    d.out_shape,
                    d.node,
                    nid = d.node_idx
                );
            }
        }
        Ok(MetalPlan {
            base,
            arena,
            statics,
            recipe: cur,
            in_f32_off: built.in_f32_off,
            rn_word: built.rn_word,
            out_offs: built.out_offs,
            out_shapes: built.out_shapes,
            dispatches: built.dispatches,
            block_bytes: bytes,
            buf,
        })
    }

    /// 确保 `plan` 的静态区此刻**逐位在位**（缓存命中路径的安全门）。
    ///
    /// 为什么必须有这道门：会话级 arena 被所有计划共用，而静态区在**每次
    /// build 都从偏移 0 起重新分配**（`Layout::alloc_static` 与 `alloc` 共用一个
    /// `cursor`，`Layout::new()` 每次归零），于是别的计划一 build（乃至它的 run
    /// ——激活区就压在别人的参数块上，实测 `[1,3,48,384]` 的参数块
    /// 1536/1536 字节落在 `4x3x48x256`/`3x3x48x512`/`1x3x48x1024` 的激活区内），
    /// 本计划缓存里那些偏移就指向别人的数据。而 `plan_for` 命中缓存时
    /// **直接返回、一个字节都不重传**，于是重放时消费方读到垃圾参数。
    ///
    /// 参数块是**形状相关**的（编码 stride/shape），没有布局技巧能让两个计划
    /// 共用同一偏移——所以只有一条出路：**执行前把自己的那片字节重放回去**。
    ///
    /// 判据：`plan.arena.blobs == plan.recipe`。`blobs` 记的是「最后触碰本
    /// arena 的计划」的清单：那个计划自己的静态区必然完好（同 build 内
    /// `alloc_static` 与 `alloc` 不重叠），而任何别的计划 build/run 都会改写
    /// 这份账本 → 等号不成立 → 重放。等号成立时零开销（同一形状连跑、同宽
    /// 合批的稳态就是这么走的）。
    ///
    /// 重放的代价 = 该计划的静态字节（det 约 1.7 MB；rec 更小）。相对它换回的
    /// 是「不必重建整个计划」——计划缓存的本意（省掉布局 + 编码的毫秒级开销）
    /// 完全保留，多出来的只是一次几百 µs 的 memcpy。
    fn ensure_statics(&self, plan: &MetalPlan) -> Result<()> {
        let mut g = plan
            .arena
            .lock()
            .map_err(|_| Error::Device("arena 锁中毒".into()))?;
        if g.blobs == plan.recipe {
            return Ok(()); // 那片字节此刻就是本计划的
        }
        // 逐 blob 判等仍然有效：`blobs` 是最后触碰者的清单，它写过的区间在
        // 其后没有任何写（同上）。
        let mut skip: std::collections::HashSet<(u32, u32, usize)> = g
            .blobs
            .iter()
            .filter(|b| plan.recipe.contains(b))
            .copied()
            .collect();
        let (mut wrote, mut skipped) = (0u64, 0u64);
        // SAFETY: `a.base` 指向 `cap_words` 个 word 的映射，而计划的所有偏移
        // 都 < 它自己的 total_words ≤ 本 arena 建它时的 cap_words（arena 只增
        // 不减、计划只落在自己那块的范围内）。
        unsafe {
            let base = g.base;
            for (off, data) in &plan.statics {
                let id = Arc::as_ptr(data) as usize;
                if skip.remove(&(*off, data.len() as u32, id)) {
                    skipped += data.len() as u64;
                    continue;
                }
                wrote += data.len() as u64;
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    base.add(*off as usize * 4) as *mut u32,
                    data.len(),
                );
            }
        }
        g.blobs = plan.recipe.clone();
        drop(g);
        if wrote > 0 || std::env::var_os("QPPOCR_METAL_STATIC_DEBUG").is_some() {
            eprintln!(
                "[static] 缓存命中重放静态区：写 {:.2}/跳 {:.2} MB（{} 块，plan recipe {} 块）",
                wrote as f64 * 4.0 / 1e6,
                skipped as f64 * 4.0 / 1e6,
                plan.statics.len(),
                plan.recipe.len()
            );
        }
        Ok(())
    }

    /// 诊断（**零 GPU 风险**）：本计划的静态字节此刻是否逐位在位。
    ///
    /// 返回 `(在位块数, 总块数, 在位字节数, 总字节数)`——直接按 `plan.statics`
    /// 的偏移从 arena 读出字节与清单比对，**不假设**任何写入路径，所以既能证明
    /// 漏洞（命中前大量不在位），也能证明修复（重放后全部在位）。
    #[cfg(test)]
    fn statics_in_place(&self, plan: &MetalPlan) -> Result<(usize, usize, u64, u64)> {
        let g = plan
            .arena
            .lock()
            .map_err(|_| Error::Device("arena 锁中毒".into()))?;
        let (mut ok_blk, mut ok_by, mut all_by) = (0usize, 0u64, 0u64);
        for (off, data) in &plan.statics {
            all_by += data.len() as u64;
            // SAFETY: 同 `ensure_statics`——偏移在映射内。
            let live = unsafe {
                std::slice::from_raw_parts(g.base.add(*off as usize * 4) as *const u32, data.len())
            };
            if live == data.as_slice() {
                ok_blk += 1;
                ok_by += data.len() as u64;
            }
        }
        Ok((ok_blk, plan.statics.len(), ok_by, all_by))
    }

    /// 找/建形状计划（含字节预算驱逐）。
    ///
    /// ⚠️ 返回值**不保证**静态区在位——命中缓存时直接返回、一个字节都不重传。
    /// 派发前必须先过 [`Self::ensure_statics`]（`run` 里已接；这也是唯一的
    /// 派发路径）。判据与理由见该方法。
    fn plan_for(&self, shape: &[i64]) -> Result<Arc<MetalPlan>> {
        let mut plans = self
            .plans
            .lock()
            .map_err(|_| Error::Device("计划缓存锁中毒".into()))?;
        if let Some((_, p)) = plans.iter().find(|(s, _)| s == shape) {
            return Ok(p.clone());
        }
        let p = Arc::new(self.build_plan(shape)?);
        plans.push((shape.to_vec(), p.clone()));
        let budget: u64 = std::env::var("QPPOCR_GPU_PLAN_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(self.plan_budget_mb);
        let mut total: u64 = plans.iter().map(|(_, p)| p.block_bytes).sum();
        while total > budget * (1 << 20) && plans.len() > 1 {
            let (_, evicted) = plans.remove(0);
            total -= evicted.block_bytes;
        }
        Ok(p)
    }

    /// 逐内核设备端计时（诊断）。**前缀差分法**：在同一命令缓冲里依次
    /// 提交前 i 条 dispatch（i = 1..n），相邻前缀的墙钟之差即第 i 条内核
    /// 的边际成本。
    ///
    /// 为什么不用「每条一个命令缓冲」：Apple GPU 的单次提交有 ~0.27 ms
    /// 的启动延迟（实测：104 条逐条提交合计 32 ms，而同一计划的单缓冲
    /// 真实墙钟只有 4.5 ms）——那样量到的是启动延迟不是算力。前缀差分把
    /// 这个常数消掉，留在差值里的是真正的 GPU 执行时间。
    ///
    /// 代价 O(n²) 的重复执行（i=104 / 4.5 ms 的计划约 0.25 s）。由
    /// `QPPOCR_GPU_KERNEL_PROF` 触发（值 = 剖析的 run 次数）。
    fn profile_kernels(&self, plan: &MetalPlan) -> Result<()> {
        let t0 = std::time::Instant::now();
        let rows = super::prof::sample_kernels(&self.ctx.queue, &self.ks, plan)?;
        // 按内核名汇总（总量 + 条数 + 单条最大）。
        let mut agg: HashMap<&'static str, (f64, usize, f64)> = HashMap::new();
        for r in &rows {
            let e = agg.entry(r.kernel).or_insert((0.0, 0, 0.0));
            e.0 += r.ms;
            e.1 += 1;
            e.2 = e.2.max(r.ms);
        }
        let total: f64 = rows.iter().map(|r| r.ms).sum();
        let mut v: Vec<_> = agg.into_iter().collect();
        v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
        eprintln!(
            "[prof] 逐内核设备端合计 {total:.2} ms（{} 条，计数器采样）",
            rows.len()
        );
        for (k, (sum, n, mx)) in &v {
            eprintln!("[prof]   {k:<18} {sum:>8.2} ms  {n:>4} 条  单条最大 {mx:>7.2} ms");
        }
        // 最贵的 12 条明细（节点序号 + 网格）。
        let mut top = rows.clone();
        top.sort_by(|a, b| b.ms.partial_cmp(&a.ms).unwrap());
        for r in top.iter().take(20) {
            eprintln!(
                "[prof]   · {:<16} {:>7.3} ms  node#{:<4} grid={:?} reps={}",
                r.kernel, r.ms, r.node_idx, r.groups, r.reps
            );
        }
        eprintln!(
            "[prof] 剖析本身耗时 {:.1} ms（含采样屏障串行化）",
            t0.elapsed().as_secs_f64() * 1000.0
        );
        Ok(())
    }

}

impl DeviceSession for MetalSession {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Metal
    }

    fn run(&self, inputs: Vec<(String, Tensor)>) -> Result<Vec<Tensor>> {
        // P2：同会话串行执行（重编码 + 同步等待不支持并发在飞）。
        // det 的现实调用形态就是逐图串行；P3 的 deferred 流水再拆。
        let _guard = self
            .run_lock
            .lock()
            .map_err(|_| Error::Device("执行锁中毒".into()))?;
        let in_shape: Vec<i64> = inputs
            .iter()
            .find(|(nm, _)| nm == &self.input_name)
            .map(|(_, t)| t.shape.clone())
            .ok_or_else(|| Error::Graph(format!("缺少输入 {}", self.input_name)))?;
        // 批维补齐（batch_grain>1 的 rec 会话）：计划按补齐形状建/查，
        // 真实行数写 real_n 单元，内核空行工作组早退。空行的输入/输出
        // 区是陈旧数据，但被守卫恒不读，无需清零。
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
        let plan = self.plan_for(&plan_shape)?;
        // 静态区安全门：命中缓存的计划**不会**重新上传（`plan_for` 直接返回），
        // 而共享 arena 上别的计划随时会压掉它那片字节 —— 派发前重放一次。
        // 见 `ensure_statics`。这是唯一的派发路径，`run_deferred` 走默认实现
        // 转投 `run`，所以这道门覆盖全部执行。
        self.ensure_statics(&plan)?;

        // real_n 先于提交写入（内核读 params[0] 做早退）。统一内存 CPU
        // 写即设备可见，无 publish 屏障。
        // SAFETY: base 是整块映射；rn_word 是 build 期静态分配。
        unsafe {
            let p = plan.base.add(plan.rn_word as usize * 4) as *mut u32;
            std::ptr::write_volatile(p, n_real as u32);
        }
        // 输入 memcpy（f32 NCHW 区）
        for (nm, t) in &inputs {
            if nm == &self.input_name {
                // SAFETY: base 持久映射；in_f32_off+len ≤ total（按输入形状分配）。
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        t.f32.as_ptr(),
                        plan.base.add(plan.in_f32_off as usize * 4) as *mut f32,
                        t.f32.len(),
                    );
                }
                break;
            }
        }

        // 每 run 重编码（Metal 命令缓冲一次性）：dispatch 间用资源屏障
        // 显式化写读依赖（Vulkan 的隐式 dispatch 顺序 Metal 不承诺）。
        // 诊断开关：跳过硬件的逐条资源屏障（`QPPOCR_METAL_NO_BARRIER`）。
        // 生产语义**必须**有屏障（Metal 不承诺相邻 dispatch 的写读可见
        // 顺序）；这个开关只用来量「154 个串行点」本身值多少毫秒——数值
        // 会错（后一条读到前一条未完成的结果），纯粹是计时实验。
        let no_barrier = std::env::var_os("QPPOCR_METAL_NO_BARRIER").is_some();
        // 诊断开关：对 **rec 计划**（`argmax_exit`）跳过指定内核的派发，
        // 逗号分隔（`QPPOCR_METAL_STUB_REC=n_conv8,n_conv`；`@名字` 按
        // 节点名精确匹配，例 `@MatMul.12`——用来把「同一内核、两个映射」
        // 的两档（≥1 MB 走窗口序 / <1 MB 走列主序）分开量）。
        //
        // 为什么这个消融**有效**，而「把 n_conv8 整体置空」无效：rec 在 det
        // 下游，切出多少行完全由 det 的输出决定，rec 的输出只是文本、不回馈
        // 给任何调度决策——所以跳过后 rec 仍然跑同样的次数，量到的就是内核
        // 本身的成本。整体置空会让 det 输出为空 ⇒ 一行都切不出来 ⇒ rec 一次
        // 也不跑，量到的是 0 而不是成本（真发生过：读数 83.5 ms/img 看着像
        // -45%，其实整个 rec 阶段没跑）。
        //
        // 数值必然错（下游读到上一轮的陈旧数据），纯粹是计时实验。屏障照旧
        // 发射：这样量到的是内核的算力/访存成本，屏障本身另由 NO_BARRIER 量。
        let stub_rec: Vec<String> = std::env::var("QPPOCR_METAL_STUB_REC")
            .ok()
            .map(|v| {
                v.split(',')
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let skip_rec = |k: &str, node: &str| {
            self.argmax_exit
                && stub_rec.iter().any(|s| match s.strip_prefix('@') {
                    Some(n) => n == node,
                    None => s == k,
                })
        };
        // 诊断开关：把这一整条命令缓冲区按 N 条 dispatch 切块，每块单独
        // commit+wait，并把每块墙钟**立刻**刷到 stderr（`QPPOCR_METAL_CHUNK`）。
        //
        // 为什么需要它：整块提交时，全部 dispatch 先在 CPU 侧编完、才进唯一
        // 那次 wait，所以日志只能告诉你"计划建好了"，无法告诉你卡在哪条。
        // 切块之后，**等待时间突然跳档的那一块**就直接指向把 GPU 访存通路
        // 打瘸的那条 dispatch；配上 `QPPOCR_METAL_PLAN_DUMP` 就知道它是什么。
        //
        // 代价：切块会多出 (块数-1) 次 commit，块间没有硬件屏障语义差别
        // （每块内部仍按 no_barrier 决定是否插资源屏障），所以**结果正确性
        // 与整块提交一致**，数值只受额外提交开销影响。默认不设此变量时走的
        // 是原来的整块路径，逐位不变。
        let chunk = std::env::var("QPPOCR_METAL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0);
        // 配套刹车：任一块的等待超过这个毫秒数就认定"访存通路刚被这一块打瘸"，
        // 立刻退出进程，不再提交后续 dispatch。默认不开（不设变量时行为不变）。
        // 开着的两个好处：① 一分钟内拿到凶手区间，不用等 5 分钟挂死；
        // ② 少给已瘸的 GPU 灌工作量——瘸了之后每次提交都在拖 WindowServer。
        let bail_ms = std::env::var("QPPOCR_METAL_CHUNK_BAIL")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|&v| v > 0.0);
        let run_seq = RUN_SEQ.fetch_add(1, Ordering::Relaxed);
        // 并发（--workers 0）时多个线程的日志会交错，带线程号才能归属。
        let tid = std::thread::current().id();
        // 消融模式下数字一定是错的，每次 run 都喊一声，免得忘了自己在量什么。
        if self.argmax_exit && !stub_rec.is_empty() {
            eprintln!(
                "[metal] ⚠ 消融模式：rec 计划（run#{run_seq}）跳过 {stub_rec:?} 的派发，\
                 **本次读数只作计时用、数值无效**"
            );
        }
        let t0 = std::time::Instant::now();
        let (encode_ms, wait_ms) = if let Some(n) = chunk {
            let ds = &plan.dispatches;
            let (mut enc_acc, mut wait_acc) = (0.0f64, 0.0f64);
            eprintln!(
                "[chunk] run#{run_seq} {tid:?} 输入 {in_shape:?}（真实行 {n_real}）→ {} 条 \
                 dispatch，切块 {n}，整块 {} word",
                ds.len(),
                plan.buf.length() / 4
            );
            for (i, blk) in ds.chunks(n).enumerate() {
                let lo = i * n;
                let (e, w) = submit_compute_timed(&self.ctx.queue, |enc| {
                    for d in blk {
                        if !skip_rec(d.kernel, &d.node) {
                            self.ks
                                .dispatch(enc, d.kernel, &plan.buf, d.p_off, d.groups)
                                .expect("dispatch（内核名来自计划，必在套件内）");
                        }
                        if !no_barrier {
                            let resources: [&metal::ResourceRef; 1] = [&plan.buf];
                            enc.memory_barrier_with_resources(&resources);
                        }
                    }
                })?;
                enc_acc += e;
                wait_acc += w;
                // stderr 无缓冲：这一行在下一块提交前一定已经落地，所以
                // 进程被卡住时最后看到的那一行就是凶手区间。
                eprintln!(
                    "[chunk] run#{run_seq} {tid:?} {lo:>4}..{:<4} 等待 {w:8.2} ms  [{:?} → {:?}]",
                    lo + blk.len(),
                    blk.first().map_or("?", |d| d.kernel),
                    blk.last().map_or("?", |d| d.kernel),
                );
                if let Some(bail) = bail_ms {
                    if w > bail {
                        eprintln!(
                            "[chunk] ✖ run#{run_seq} {tid:?} dispatch {lo}..{} 等待 {w:.1} ms > \
                             阈值 {bail:.0} ms —— 访存通路很可能就被这一块打瘸了。\
                             立即退出（不再提交后续 dispatch）。凶手内核区间：{:?} → {:?}",
                            lo + blk.len(),
                            blk.first().map_or("?", |d| d.kernel),
                            blk.last().map_or("?", |d| d.kernel),
                        );
                        std::process::exit(97);
                    }
                }
            }
            (enc_acc, wait_acc)
        } else {
            submit_compute_timed(&self.ctx.queue, |enc| {
                for d in &plan.dispatches {
                    if !skip_rec(d.kernel, &d.node) {
                        self.ks
                            .dispatch(enc, d.kernel, &plan.buf, d.p_off, d.groups)
                            .expect("dispatch（内核名来自计划，必在套件内）");
                    }
                    if !no_barrier {
                        let resources: [&metal::ResourceRef; 1] = [&plan.buf];
                        enc.memory_barrier_with_resources(&resources);
                    }
                }
            })?
        };
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if std::env::var_os("QPPOCR_GPU_PROF").is_some() {
            let n = plan.dispatches.len();
            eprintln!(
                "[metal] run#{run_seq} 输入 {in_shape:?}（真实行 {n_real}）→ {n} 个 \
                 dispatch，墙钟 {wall_ms:.2} ms = 编码 {encode_ms:.2}（{:.1} µs/条）+ \
                 等待 {wait_ms:.2}",
                encode_ms / n.max(1) as f64 * 1000.0
            );
        }
        // 逐内核剖析（一次性，预算由 QPPOCR_GPU_KERNEL_PROF 给）——放在
        // 正常执行之后，量的是「刚跑过同一份数据」的设备端耗时。
        if self
            .prof_left
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |v| v.checked_sub(1),
            )
            .is_ok()
        {
            self.profile_kernels(&plan)?;
        }

        // 读回（统一内存：已 wait_until_completed，直接读）。补齐计划的
        // 真实行读回：只取前 n_real 行（批是外维、行连续；空行区是陈旧
        // 数据）。n 按 n_pad 建的——按比例缩回。
        let mut outs = Vec::with_capacity(plan.out_offs.len());
        for (i, (name, off, n)) in plan.out_offs.iter().enumerate() {
            let mut shape = plan.out_shapes.get(i).cloned().unwrap_or_default();
            if !shape.is_empty() {
                shape[0] = n_real;
            }
            let n_take: usize = if n_pad != n_real {
                n / n_pad.max(1) as usize * n_real as usize
            } else {
                *n
            };
            let mut buf = F32Buf::with_zeroed(n_take);
            // SAFETY: base 持久映射；off+n_take ≤ total（计划保证）。
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
        Ok(outs)
    }

    fn prefers_host_parallelism(&self) -> bool {
        false // GPU 内部已并行
    }

    fn bucket_grain(&self) -> i32 {
        // rec 每行一个裸宽 → 每形状一次计划重建。64 桶把语料宽度塌缩
        // 到少数形状（右侧零填充的额外计算远小于重建税）；例外：
        // exact_width 模型填充会改真步输出（非「额外计算」而是「错值」）
        // ——退 8px 对齐（注意力混入窗口极小，GT 实测与逐像素精确无差）。
        if self.exact_width { 8 } else { 64 }
    }

    fn batch_grain(&self) -> i32 {
        self.batch_grain
    }

    /// 合批上限恒 8：与补齐粒度（`batch_grain`，默认 1）解耦。合的是
    /// 真实行，只省每轮 173 条 dispatch 的编码/提交开销，不增算力——
    /// 宽行网格 bsz=1 已吃满 EU，但同桶多行仍该一次提交。
    fn rec_chunk(&self) -> usize {
        8
    }

    fn rec_argmax_pairs(&self) -> bool {
        self.argmax_exit
    }
}
