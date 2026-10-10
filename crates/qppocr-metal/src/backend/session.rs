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
    /// 计划缓存记账用的**独占**字节 = 动态（激活）区，即整块 − 静态段。
    ///
    /// 为什么不是整块：会话级 arena 全会话只有**一块**（见 `Arena`），静态段
    /// （rec 60.7 MB）也全会话只有一份。按整块记账等于把这份共享内存重复计
    /// N 次，而它远比计划本身大——实测 rec 一份计划「整块 68.3 MB、静态
    /// 60.7 MB」，于是 1536 MB 预算只留得住 22 份，可一图就有 12 个桶形状、
    /// 16 图共 147 个不同形状 ⇒ 淘汰-重建成为常态（实测 147 建 / 53 命中，
    /// 其中 **45 建纯属淘汰浪费**）。独占口径下 1536 MB 能留 ~120 份。
    block_bytes: u64,
    /// 静态段字节数。同会话所有计划共用一份（最大值即上界），
    /// `plan_for` 里只计一次——见 [`Self::block_bytes`]。
    static_bytes: u64,
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
/// 标识相同 ⇒ 那片字节就是本次要写的内容，跳过。参数块/RN 单元每次新建，
/// 标识自然不同于**存活**的旧分配 ⇒ 照写。
///
/// ⚠️ 但「新建的标识必不同于旧账本里的标识」这句话**只在旧账本里的地址还被
/// 人占着时成立**。指针是可回收的整数：写下账本的计划一被丢掉（`probe()` 探完
/// 即弃、缓存淘汰），它那些 `Arc` 地址就会被后续分配接住，于是新参数块会撞上
/// 旧账本里的同一条 ⇒ 假命中 ⇒ **那一块根本没写**。`Arena::pins` 就是为堵这个
/// 而存在的（钉住账本里的地址），它的文档里有 2026-10-11 那次丢行的完整形态。
///
/// ⚠️ **`blobs` 的语义是「最后触碰本 arena 的那个计划的静态清单」**，不是
/// 「上一份计划的」（`plan.rs:1800-1806` 的「静态区只在装载期写一次」只在
/// **一个计划独占一块 arena** 时成立，而这里会话级共享——Vulkan 侧每个
/// `Plan` 私有 arena，所以那边没有这个问题）。
///
/// 跨 build 时每个计划的静态区**各自从偏移 0 重新分配**（`Layout::alloc_static`
/// 只 bump `static_cursor`，每次 `Layout::new()` 归零），于是**同一偏移**在不同
/// 计划里装着**不同的东西**——实测 rec 各形状的静态分区逐位相同（都是 152 块、
/// `static_words=1106412`），差的全在参数块的**值**上。参数块是**形状相关**的
/// （编码 stride/shape），没有任何布局技巧能让两个计划共用同一偏移。所以跨 build
/// 的完整性只能靠**每次执行前重放**，见 `MetalSession::ensure_statics`。
///
/// （旧注释在这里写「build j 的激活区可以压在 build i 的参数块上」，那是
/// `static_reserve` 引入**之前**的实测：那时两段共用一个 bump。现在激活段从
/// `reserve`（本会话静态段上界）起，激活写碰不到任何计划的静态区——但这只解决
/// 了「别的东西写回来」，「同一偏移处是谁的字节」仍要靠每次重放。）
struct Arena {
    buf: metal::Buffer,
    base: *mut u8,
    cap_words: u32,
    static_words: u32,
    /// 最后触碰本 arena 的计划写下的静态 blob：(偏移, 字数, 内容标识)
    blobs: Vec<(u32, u32, usize)>,
    /// **`blobs` 里每个「内容标识」的持有权**——判据成立的前提，缺了它就是
    /// 2026-10-11 抓到的静默丢行。
    ///
    /// `blobs` 拿 `Arc::as_ptr` 当内容标识，可**指针只是个整数**：它唯一标识
    /// 一份 `Vec`，只在「那份 `Vec` 还活着」时才成立。而写下这套账本的计划随时
    /// 会被丢掉——`probe()`（`build_plan(shape, 0).map(|_| ())`，探完即弃）、
    /// `plan_for` 的 FIFO 淘汰、`PlanBuilder::build` 第一趟自查的临时计划，都会
    /// 释放自己的 blob `Arc`。释放之后**同一地址会被下一次 `Arc::new` 重新分配**
    /// （同尺寸块，分配器 LIFO），于是新计划的块撞上账本里的旧地址 ⇒ 被判「已在
    /// 位」⇒ 跳过写入 ⇒ arena 里留的是**别人**的字节。
    ///
    /// 实测形态：rec 预热计划（探针建完即弃）的 103 个参数块地址被真实计划接住，
    /// `n_reduce_fin / ReduceMean.0` 的参数 `m_dim` 停在预热的 256（本形状要 384），
    /// `[1,3,48,128]` 那一桶里的一行就静默没了（~2.7% 的图）。**签名是账本命中数
    /// 多出 1**（`命中=85` 而非 84），多出来的那 1 条就是假命中。
    ///
    /// 拿着 `Arc` 就等于把地址钉住：只要三元组还在账本里，它指向的 `Vec` 就活着，
    /// 同地址不可能属于别人 ⇒「标识相同」重新成为**内容相同**的充分条件。
    /// 代价只是引用计数 +1（权重本就由 `repack_cache` 长存，参数块 KB 级）。
    pins: Vec<Arc<Vec<u32>>>,
}

impl Arena {
    /// 重设账本。`blobs` 与 `pins` **必须同源**（都由同一份 `statics` 派生，
    /// `recipe` = `statics` 的 `(偏移, 字数, 指针)` 视图）——分开设就是漏钉，
    /// 见 `pins` 的文档。所有写账本的地方都走这里，别再手写 `g.blobs = …`。
    fn put_ledger(&mut self, statics: &[(u32, Arc<Vec<u32>>)], recipe: Vec<(u32, u32, usize)>) {
        debug_assert_eq!(
            statics.len(),
            recipe.len(),
            "账本的样子与钉住的 Arc 必须同源（同一份 statics）"
        );
        // 回归开关（`QPPOCR_METAL_LEDGER_UNPIN=1`，**只在同时开着 `STATIC_VERIFY`
        // 时才生效**）：不钉地址，把 2026-10-11 那类假命中放回来。它的唯一用途是
        // 证明蹲坑脚本还抓得到——没有它，`flake_param.sh` 报「300 轮干净」就无法
        // 区分「修好了」和「判据瞎了」。
        //
        // 互锁的理由：只有 `STATIC_VERIFY` 会在派发前**拒发**坏计划；单开本开关
        // 会让错的参数真的进命令缓冲（本仓有「读垃圾参数 → 巨循环 → GPU 打瘸」的先例）。
        let unpin = std::env::var_os("QPPOCR_METAL_LEDGER_UNPIN").is_some()
            && std::env::var_os("QPPOCR_METAL_STATIC_VERIFY").is_some();
        self.pins = if unpin {
            Vec::new()
        } else {
            statics.iter().map(|(_, v)| Arc::clone(v)).collect()
        };
        self.blobs = recipe;
    }
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
    /// 权重按 fp16 存（`QPPOCR_METAL_WF16`，默认 **false = f32**）。
    /// 会话级常量——同一会话里同一份权重只重排一次，不存在中间态。
    /// 见 [`weight_f16`]。
    ///
    /// `pub(crate)` 是为了测试能直连两条路对拍（同 `batch_grain` 的用法）；
    /// 计划是**惰性构建**的，所以构造后、首次 run 前改它仍然生效。
    pub(crate) weight_f16: bool,
    /// 1×1 稠密 conv 走 simdgroup_matrix MMA 内核（`n_conv8_mma`）。
    ///
    /// **默认 true**：全程 f32，没有 `weight_f16` 那种精度档位可谈；累加顺序
    /// 与标量路不同（MMA 的 K 分片顺序），故**不是逐位相同**——但误差就是
    /// float 舍入本身（`conv8_mma_matches_cpu` 对拍 4 形状最大 1.5e-8）。
    /// 这是「换个加法顺序」，不是「降精度」，所以不需要用户决策。
    /// `QPPOCR_METAL_CONV1X1_MMA=0` 可关闭（A/B 对照用）。
    ///
    /// 见 [`conv1x1_mma`] 与 `plan.rs::PlanBuilder::conv1x1_mma`。
    pub(crate) conv1x1_mma: bool,
    /// 稠密 k×k conv（k>1）走 `n_conv8_mma3` + `n_conv8_edge`。
    ///
    /// 见 [`conv_kxk_mma`] 与 `plan.rs::PlanBuilder::conv_kxk_mma` 的启用判据。
    pub(crate) conv_kxk_mma: bool,
    /// ConvTranspose 走空间分块的 `n_convt4w`（`QPPOCR_METAL_CONVT_TILE`，
    /// 默认 **true**）。
    ///
    /// 见 [`conv_t_tile`] 与 `plan.rs::PlanBuilder::conv_t_tile` 的启用判据。
    pub(crate) conv_t_tile: bool,
    /// 执行串行锁：每 run 重编码 + 同步等待不支持并发在飞（deferred
    /// 流水是 P4 的性能项；trait 默认的同步 run_deferred 语义等价）。
    run_lock: Mutex<()>,
    /// 逐内核剖析的剩余次数（`QPPOCR_GPU_KERNEL_PROF`，0 = 关闭）。
    prof_left: std::sync::atomic::AtomicUsize,
    /// 本会话静态段的上界（word）。**单调不减**——见 `plan_for`：静态段
    /// 与激活段分离后，静态偏移只由这一个数决定，而它一变，所有计划的
    /// 静态偏移一起变（已建计划则要靠重放找补，正是要消除的那笔开销）。
    static_reserve: std::sync::atomic::AtomicU32,
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

/// 权重的 fp16 存储：`QPPOCR_METAL_WF16` 显式给值（`1`/`0`）时用它，
/// 否则 **f32**（精度是默认立场——用户侧的既定要求「默认就是 32 精度」）。
///
/// 量过的账——**探针先算、落地后复测**，两栏都在：
///
/// | | 热点 1×1 | 3×3 | CTC 头 | 精度代价（CPU 全模型 f16 = 上界） |
/// |---|---|---|---|---|
/// | 探针 `p_whalf` 预测 | 1.7~2.1× | 1.26~1.45× | 2.59× | CER 0.072~0.199% |
///
/// 落地后的**端到端实测**（2026-10-09，`bench/gpuhealth/ab.sh` 交错四轮
/// 取配对中位数；精度用 `wf16_ab.sh` 走同一份 `models/`）：
///
/// | tier | 总耗时（40 图/轮） | 配对中位 Δ | 加速 | 逐图行数变化 | 行文本不同 | CER |
/// |---|---|---|---|---|---|---|
/// | tiny | 53.6 → 45.6 ms | −7.50 ms | 1.17× | 1/100 张 | 6/1096 | 0.072% |
/// | small | 143 → 98 ms | −43.05 ms | 1.45× | 0 张 | 0/1041 | **0.000%** |
/// | medium | 866 → 485 ms | −380.65 ms | 1.79× | 0 张 | 5/1034 | 0.017% |
///
/// 三档全部**低于**上表那个上界（0.199/0.072/0.165 分别对应 tiny/small/
/// medium），与「只覆盖部分层」的预期一致。
///
/// 归因是**单变量**的（medium det 逐内核剖析，`QPPOCR_GPU_KERNEL_PROF=1`）：
/// `n_conv8` 191.01 ms → `n_conv8h` 99.16 ms（**1.93×**），而 `n_conv_dw`
/// 23.5→21.9、`n_convt` 11.1→9.0、`n_elem` 3.9→4.1、`n_channel` 2.4→2.4、
/// `n_reduce_*` 0.97→0.92——**只有被改的那条内核动了**，其余是干净的对照列。
/// 静态段同步减半（medium det 62.0→31.4 MB；rec 148.1→110.1 MB，未全减
/// 是因为 `n_conv`/`n_conv_dw`/`n_convt` 与非偶 `nv` 的 `n_conv8` 仍是 f32）。
///
/// 几点如实记下、别被 1.79× 冲昏头：① 收益随模型变大而**变大**
/// （tiny 1.17× < small 1.45× < medium 1.79×），小模型权重占比低时几乎
/// 白给；② tiny 有 **1 张图的行数变了**（检测级事件，不是识别级）；
/// ③ 行文本变的那几处里有 `Paddle0CR`→`PaddleOCR`、`PaddIeOCR`→
/// `PaddleOCR` 这种**改对了**的，也有反方向的——数量级只有 0.5%，说明
/// 这些行本来就在判定边界上，两种精度都可能倒向任一边，不是「fp16 更准」。
///
/// 留这个开关（而不是直接改默认）的理由有两条：① 精度是产品决策，默认
/// 不能悄悄变；② 它是「一次可以逐位回到改前」的对照——否则量到端到端
/// 差异时说不清是哪一步带来的。关掉时内核选择、重排、plan 布局**逐位**
/// 回到改前（`n_conv8` + `repack_conv_w`），不是「差不多」。
fn weight_f16() -> bool {
    matches!(
        std::env::var("QPPOCR_METAL_WF16").as_deref(),
        Ok("1") | Ok("on") | Ok("true")
    )
}

/// 1×1 conv 走 `n_conv8_mma`（`QPPOCR_METAL_CONV1X1_MMA`，默认 **true**）。
///
/// 与 [`weight_f16`] 不同，这个开关**不动精度**（MMA 与标量路都是 f32，
/// 对拍最大误差 1e-8 = float 舍入），所以默认开；置 `0`/`off` 关掉，
/// 供 A/B 对照与排障（`bench/gpuhealth/env_ab.sh` 直接吃它）。
fn conv1x1_mma() -> bool {
    !matches!(
        std::env::var("QPPOCR_METAL_CONV1X1_MMA").as_deref(),
        Ok("0") | Ok("off") | Ok("false")
    )
}

/// 稠密 k×k conv（k>1）走 `n_conv8_mma3` + `n_conv8_edge`
/// （`QPPOCR_METAL_CONV_KXK_MMA`，默认 **true**）。
///
/// 同 [`conv1x1_mma`]：不动精度（f32 进 f32 出，只有 K 分片的加法顺序不同，
/// `conv8_mma3_matches_cpu` 对拍 5 形状最大 2.0e-7），默认开；置 `0`/`off`
/// 关掉，供 A/B 对照。
///
/// **与 1×1 那条不同的一点**：这条会改变**派发条数**（一条 conv → 两条
/// dispatch），所以关掉它时计划层逐位回到改前，端到端 A/B 的对照列必须是
/// 它而不是别的开关。
fn conv_kxk_mma() -> bool {
    !matches!(
        std::env::var("QPPOCR_METAL_CONV_KXK_MMA").as_deref(),
        Ok("0") | Ok("off") | Ok("false")
    )
}

/// ConvTranspose 走空间分块的 `n_convt4w`
/// （`QPPOCR_METAL_CONVT_TILE`，默认 **true**）。
///
/// 同 [`conv1x1_mma`]：不动精度（f32 进 f32 出，累加次序与 `n_convt` 的
/// ci4 快路逐 ci 相同，探针对拍差 0.000e0），所以默认开；置 `0`/`off`
/// 关掉，供 A/B 对照与排障（对照列会**逐位回到** `n_convt` 的派发）。
fn conv_t_tile() -> bool {
    !matches!(
        std::env::var("QPPOCR_METAL_CONVT_TILE").as_deref(),
        Ok("0") | Ok("off") | Ok("false")
    )
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
            weight_f16: {
                let on = weight_f16();
                if on {
                    // 非默认状态一律**声明**（同 QPPOCR_GPU_STAGES 的纪律）：
                    // 这是个会动精度的开关，静默生效会让人对着数字猜。
                    eprintln!(
                        "[metal] QPPOCR_METAL_WF16：权重按 fp16 存储\
                         （走 n_conv8 的层），精度代价见 weight_f16() 文档"
                    );
                }
                on
            },
            conv1x1_mma: conv1x1_mma(),
            conv_kxk_mma: conv_kxk_mma(),
            conv_t_tile: conv_t_tile(),
            run_lock: Mutex::new(()),
            // ⚠ 这个预算是**每会话一份**，而 det / rec / cls 是三个会话
            //（各自 new 一次）。所以 `QPPOCR_GPU_KERNEL_PROF=3` 不是
            //「剖前 3 次运行」，而是「**每个会话各剖前 3 份计划**」——
            // rec 有 10+ 份形状计划时，给 3 只会剖到前 3 份，其余静默缺失。
            // 曾照此得出「rec 设备时间只有 38 ms、其余是主机开销」的假结论
            //（真值 127 ms / 94% 设备绑定）。要全剖就给够，如 `=60`。
            prof_left: std::sync::atomic::AtomicUsize::new(
                std::env::var("QPPOCR_GPU_KERNEL_PROF")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
            ),
            static_reserve: std::sync::atomic::AtomicU32::new(0),
        })
    }

    /// 覆盖面探针：小形状试建一次计划（不出执行）——算子支持与形状
    /// 无关，不支持的算子在此点名（create_session 分级用）。
    pub(crate) fn probe(&self, in_shape: &[i64]) -> Result<()> {
        self.build_plan(in_shape, 0).map(|_| ()) // 探针：保留量未知，build 自查重建
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

    /// `static_reserve` = 本会话静态段的上界（word）——由 `plan_for` 从
    /// 已有计划里取最大值传入。理由见 `plan::Layout` 与 `PlanBuilder::build`：
    /// 静态段与激活段各用各的 bump 指针，静态偏移就不再随形状漂移，
    /// `ensure_statics` 的跨形状重拷（rec 实测 76.4 MB / 1.7 ms 一次）随之归零。
    fn build_plan(&self, in_shape: &[i64], static_reserve: u32) -> Result<MetalPlan> {
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
            // 权重的 fp16 存储（默认关）。见 [`weight_f16`]。
            weight_f16: self.weight_f16,
            // 1×1 conv 走 simdgroup_matrix MMA（默认开）。见 [`conv1x1_mma`]。
            conv1x1_mma: self.conv1x1_mma,
            // k×k（k>1）同款（默认开）。见 [`conv_kxk_mma`]。
            conv_kxk_mma: self.conv_kxk_mma,
            // ConvTranspose 的空间分块版（默认开）。见 [`conv_t_tile`]。
            conv_t_tile: self.conv_t_tile,
        }
        .build(&self.input_name, in_shape, static_reserve)?;

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
        // 缓存记账口径：独占 = 动态区。静态段会话共享，`plan_for` 里单独只计
        // 一次最大值（理由见 `MetalPlan::block_bytes` 的文档）。
        let cache_bytes = (built.total_words - built.static_words) as u64 * 4;
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
                    pins: Vec::new(),
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
        // 即「那片字节在位」：它写过同样内容在同一区间，且在其后没有任何写落在
        // 那上面。
        //
        // 「其后没有任何写」这一半现在由构造保证，不再靠约定：激活段的 bump 从
        // `reserve` 起（`plan.rs` 的 `Layout::with_reserve`，`alloc` 里还有
        // `debug_assert!(off >= self.reserve)`），而 `reserve` 是**本会话所有形状
        // 静态段的上界**（`plan_for` 传 `static_reserve`，`build` 自查不满足就重建）
        // ⇒ 任何计划的激活写都落在 [reserve, …)，碰不到任何计划的静态区
        // [0, static_words) 。（旧注释说「激活区会压在别人的参数块上」——那是
        // `static_reserve` 引入**之前**的实测，现已不成立；见 §每形状重置静态段。）
        //
        // 剩下的风险全在「标识」这一半：它是个可回收的指针，账本条目所指的
        // 计划一旦被丢掉，地址就会被后来的分配接住（`probe()` 探完即弃是主要
        // 来源）⇒ 新参数块撞上旧条目 ⇒ 假命中 ⇒ 那一块**根本没写**。
        // `Arena::pins` 把账本里的地址钉住，才使「标识全等 ⇒ 内容相同」成立。
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
            // 账本与钉住的 Arc 同源落地：`cur` 就是 `statics` 的 (偏移, 字数, 指针)
            // 视图，钉住 `statics` 即可保证账本里每个地址在条目有效期内都活着
            // （见 `Arena::pins`——不钉，指针就会被下一次分配复用，假命中）。
            g.put_ledger(&statics, cur.clone());
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
            // 参数块**逐字**转储（`QPPOCR_METAL_PLAN_DUMP_FULL=1`）：上面那行是
            // 我挑着打的字段，而「两个派发长得一样却差 1.47×」这种指控，只要
            // 有一个我没打的字段不同就会被误判成「参数相同」——比如 P+6/7/8
            // （outW/inW/inH，内核按它算行地址）。全打出来才能否证。
            let full = std::env::var_os("QPPOCR_METAL_PLAN_DUMP_FULL").is_some();
            for (i, d) in built.dispatches.iter().enumerate() {
                let extra = pubs
                    .get(&d.p_off)
                    // 认一下这确实是我这一步的参数块：params[P+3] 是 out_off。
                    .filter(|p| p.get(3).copied() == Some(d.out_off))
                    .map_or(String::new(), |p| {
                        let g = |k: usize| p.get(k).copied().unwrap_or(0);
                        // MMA 系内核的 P+4/P+5/P+9/P+10/P+11 与 `n_conv8` **逐字段
                        // 同义**（见各 shader 头部：mDim/cols/cin4v/taps/kW），
                        // 所以同一段格式化直接复用。少了 cin 就没法算 GFLOPS，
                        // 也就没法把「谁跑得远低于天花板」排出来。
                        match d.kernel {
                            "n_conv8" | "n_conv" | "n_conv8_mma" | "n_conv8_mma3"
                            | "n_conv8_mma3_n32" | "n_conv8_edge" => format!(
                                // 尾声三项（`P+16` act / `P+2` bias / `P+20` 残差，
                                // `OFF_NONE=0xffffffff`）：形状/网格完全相同的两个
                                // 派发差 1.47×（`Conv.55` vs `Conv.58`）时，这三项
                                // 是**头号嫌疑**，所以先打出来否证它——事后证明确实
                                // 不是它（三项逐字相同），根因是下面那个 `w%64B`。
                                "  cin={:<4} co={:<4} taps={} ({}x{})  m={}  尾: act={} \
                                 bias={} res={}  偏: in={} w={} r={} out={}（w%64B={}）（块={}）",
                                g(9) * 4,
                                g(5),
                                g(10),
                                g(10) / g(11).max(1),
                                g(11),
                                g(4),
                                g(16),
                                g(2) != u32::MAX,
                                g(20) != u32::MAX,
                                g(0),
                                g(1),
                                g(20),
                                g(3),
                                // 权重块**起点字节偏移 mod 64**：这个相位把同
                                // 形状同网格的 `n_conv8_mma` 派发分成四档（1.25
                                // / 1.45 / 1.75 / 1.90 ms，≡0 最快，幅度 1.51×）
                                // ——`Conv.55` vs `Conv.58` 差 1.47× 就是它。
                                // 曾经这里打的是 `% 16`，恒为 0（任何 float4
                                // 偏移都是 16 的倍数），白打了一格。
                                (g(1) * 4) % 64,
                                d.p_off
                            ),
                            "n_conv_dw" => format!(
                                "  c={:<4} taps={} ({}x{})  m={}",
                                g(5),
                                g(9),
                                g(9) / g(10).max(1),
                                g(10),
                                g(4)
                            ),
                            // 步长卷积：P+5=co、P+9=cin、P+13=kW、P+11/P+12=步长。
                            // 打的是**输出位置数**（m）而 FLOPs 按每个输出位置
                            // 累 kW² 个输入算 —— 步长 2 时实际贡献只有 1/4，
                            // 所以这一项是**上界**，别拿它当准数。
                            "n_convt" => format!(
                                "  cin={:<4} co={:<4} taps={} ({}x{})  m={}  步长 {}x{}",
                                g(9),
                                g(5),
                                g(13) * g(13),
                                g(13),
                                g(13),
                                g(4),
                                g(11),
                                g(12)
                            ),
                            _ => String::new(),
                        }
                    });
                let extra = if full {
                    match pubs.get(&d.p_off) {
                        Some(p) => format!("{extra}\n         P={p:x?}"),
                        None => extra,
                    }
                } else {
                    extra
                };
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
            block_bytes: cache_bytes,
            static_bytes: built.static_words as u64 * 4,
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
        // 诊断开关：**跳过静态区重放**（数值必然错，只作计时）。用它是为了把
        // 「重放这 ~21 MB」的成本从 `rec_infer` 里单独拎出来——`rec_infer` 包住
        // pack→提交→收账（等待+读回+CTC 解码）整段，光看它会把这笔主机侧拷贝
        // 算成 GPU 开销（消融掉全部 rec 内核后它仍有 ~37 ms/图）。
        if std::env::var_os("QPPOCR_METAL_STATIC_NOREPLAY").is_some() {
            return Ok(());
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
        let t_replay = std::time::Instant::now();
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
        // 重放完，账本换成本计划的（`pins` 同源钉住，见 `Arena::pins`）。
        g.put_ledger(&plan.statics, plan.recipe.clone());
        drop(g);
        if wrote > 0 || std::env::var_os("QPPOCR_METAL_STATIC_DEBUG").is_some() {
            eprintln!(
                "[static] 缓存命中重放静态区：写 {:.2}/跳 {:.2} MB（{} 块，plan recipe {} 块），\
                 耗时 {:.2} ms",
                wrote as f64 * 4.0 / 1e6,
                skipped as f64 * 4.0 / 1e6,
                plan.statics.len(),
                plan.recipe.len(),
                t_replay.elapsed().as_secs_f64() * 1000.0
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
        // 命中/新建的每一笔都记账（`QPPOCR_GPU_BUILD_TIME`）：计划重建是
        // 纯 CPU、与 GPU 等待串行的 33 ms/图（占单图 20%），而**同一形状
        // 被建两次**是当时最大的疑点——命中率是「能不能再省一半」的全部依据。
        let dbg = std::env::var_os("QPPOCR_GPU_BUILD_TIME").is_some();
        if let Some((_, p)) = plans.iter().find(|(s, _)| s == shape) {
            if dbg {
                eprintln!("[cache] 命中 {shape:?}（缓存 {} 份）", plans.len());
            }
            return Ok(p.clone());
        }
        // 静态段保留量（word）——**会话内单调不减**：计划缓存会淘汰，从
        // 缓存里现取会让保留量回退，新计划随之落到更小的偏移上，反而把
        // 已建计划的偏移稳定性打掉。首个形状传 0，`build` 量出真实大小后
        // 自己重建一遍，之后每次只走一遍。
        let reserve = self.static_reserve.load(Ordering::Relaxed);
        let p = Arc::new(self.build_plan(shape, reserve)?);
        self.static_reserve
            .fetch_max((p.static_bytes / 4) as u32, Ordering::Relaxed);
        if dbg {
            eprintln!("[cache] 新建 {shape:?}（缓存 {} 份）", plans.len() + 1);
        }
        plans.push((shape.to_vec(), p.clone()));
        let budget: u64 = std::env::var("QPPOCR_GPU_PLAN_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(self.plan_budget_mb);
        // 独占区累加 + 静态段按**最大的一份**只计一次（会话共享一块 arena，
        // 见 `MetalPlan::block_bytes`）。旧口径把两份都按整块逐计划累加，
        // 实测让 rec 的计划命中率掉到 26%（147 建 / 53 命中），其中 45 次
        // 新建纯粹是被自己的共享静态段挤掉后重建的。
        //
        // 每剔一份就整体重算：静态段那一项是**集合上的最大值**，剔掉的正是
        // 队首，增量维护会让它偏大（把已经不在集合里的份额继续算着）。n 是
        // 几十~几百，O(n²) 在这里完全不是瓶颈。
        let weigh = |ps: &[(Vec<i64>, Arc<MetalPlan>)]| -> u64 {
            ps.iter().map(|(_, p)| p.block_bytes).sum::<u64>()
                + ps.iter().map(|(_, p)| p.static_bytes).max().unwrap_or(0)
        };
        while weigh(&plans) > budget * (1 << 20) && plans.len() > 1 {
            plans.remove(0);
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
        // 最贵的 20 条明细（节点名 + 序号 + 网格）。
        //
        // `QPPOCR_GPU_KERNEL_PROF_ALL=1` 放开到全部：排掉 20 名之后的条目在看
        // 「慢是这条独有的还是这一族都慢」时是必需的——只留 top-20 时，
        // 快的那一半整段不入镜，任何「谁比谁慢」的结论都缺对照臂。
        let mut top = rows.clone();
        top.sort_by(|a, b| b.ms.partial_cmp(&a.ms).unwrap());
        let keep = if std::env::var_os("QPPOCR_GPU_KERNEL_PROF_ALL").is_some() {
            top.len()
        } else {
            20
        };
        for r in top.iter().take(keep) {
            eprintln!(
                "[prof]   · {:<16} {:>7.3} ms  node#{:<4} {:<24} grid={:?} reps={}",
                r.kernel, r.ms, r.node_idx, r.node, r.groups, r.reps
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

        // 本 run 的序号（诊断用；提前到这里只为让 memcpy 前后的两次落盘同名配对）。
        let run_seq = RUN_SEQ.fetch_add(1, Ordering::Relaxed);
        // real_n 先于提交写入（内核读 params[0] 做早退）。统一内存 CPU
        // 写即设备可见，无 publish 屏障。
        // SAFETY: base 是整块映射；rn_word 是 build 期静态分配。
        unsafe {
            let p = plan.base.add(plan.rn_word as usize * 4) as *mut u32;
            std::ptr::write_volatile(p, n_real as u32);
        }
        // 诊断开关：把本计划的**动态区** [静态段末, 计划末) 在喂输入之前填成
        // 给定 u32（`QPPOCR_METAL_FILL_DYN=<十六进制字>`，如 `0` / `3f800000`）。
        //
        // 与 `ARENA_FILL` 的区别：那个填**整块**（含静态段、且清空跳过集 ⇒ 强制
        // 全量静态上传）；这个只填动态区、**一个静态字节都不碰**，也不动跳过集。
        // 于是它能一刀切开两种「逐轮不同」：若填 0 后症状消失、填 1.0f 后症状
        // 改变 ⇒ 本计划**读到了自己没写过的动态字节**（那处本该是 0/某个定值）；
        // 若两个填值都对症状无影响 ⇒ 与字节内容无关，是**派发之间的竞态/时序**。
        if let Some(fv) = std::env::var("QPPOCR_METAL_FILL_DYN")
            .ok()
            .and_then(|v| u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok())
        {
            let start = (plan.static_bytes / 4) as usize;
            let n = (plan.block_bytes / 4) as usize;
            // SAFETY: [start, start+n) = [静态段末, 计划末) 在映射内（同上断言）。
            unsafe {
                let p = plan.base.add(start * 4) as *mut u32;
                for i in 0..n {
                    std::ptr::write_volatile(p.add(i), fv);
                }
            }
        }
        // 输入 memcpy（f32 NCHW 区）
        let mut in_words = 0usize;
        for (nm, t) in &inputs {
            if nm == &self.input_name {
                in_words = t.f32.len();
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
        // 诊断（`QPPOCR_METAL_IO_DUMP`）：memcpy 之后、dispatch **之前**再落一份输入。
        // 与 run 之后那份配对比较，把「host 喂进来的就是错的」和「被某个内核覆盖了」
        // 一刀切开——这是判据，不是猜测。
        if self.argmax_exit {
            if let Some(prefix) = std::env::var_os("QPPOCR_METAL_IO_DUMP") {
                let shape = in_shape
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join("x");
                // SAFETY: in_f32_off + in_words ≤ total_words（memcpy 按输入形状分配）。
                let pre = unsafe {
                    std::slice::from_raw_parts(
                        plan.base.add(plan.in_f32_off as usize * 4),
                        in_words * 4,
                    )
                };
                let path = format!(
                    "{}.run{run_seq:03}.{shape}.in_pre",
                    prefix.to_string_lossy()
                );
                if let Err(e) = std::fs::write(&path, pre) {
                    eprintln!("[io-dump] 写 {path} 失败：{e}");
                }
            }
        }

        // 诊断（`QPPOCR_METAL_STATIC_VERIFY=1`，**纯读、零 GPU 风险**）：派发前把本
        // 计划的**静态清单逐字与 arena 实际内容比对**。清单里既有权重（`repack_cache`
        // 的 `Arc`）也有**参数块**（编码 stride/shape/`out_off`/`rows`）；任何一块
        // 不在位，内核读到的就是**别的计划的参数**——出口会把 `(val,idx)` 写到别人的
        // 偏移上、或按更小的 `rows` 提前返回 ⇒ 输出区保持上一张张量的残渣 ⇒
        // 「整行静默消失」（坏轮证据：相对激活区 79484..79515 那 32 字全程等于环境值）。
        //
        // 为什么必须看**内容**：`ensure_statics` 的早退判据是 `g.blobs == plan.recipe`、
        // 跳过集也比 `(偏移, 字数, Arc 指针)`，**全是指针值**。指针可复用（`Vec` 的
        // `Arc` 一旦 drop，同一地址会分给别的 `Vec`），所以这些判据有假阳性的余地，
        // 而唯一能证伪的证据是内容。唯一预期的例外是 `rn_word`（表头 word 0）——
        // host 每 run 按真实行数覆写，内容本就与建成时不同。
        //
        // **发现不在位就拒绝派发**（不是只打印）：这份参数块的用途是给内核编址，
        // 错的参数轻则丢一行、重则让内核按别人的偏移/形状跑成巨循环把 GPU 打瘸
        // （本仓有先例，见 `bench/gpuhealth/health.sh` 的判据）。既然已经知道它
        // 是错的，就绝不能再送进命令缓冲。要观察「错的参数怎么走坏下游」的人显式
        // 加 `QPPOCR_METAL_STATIC_VERIFY_WARN=1` 退回只警告。
        if std::env::var_os("QPPOCR_METAL_STATIC_VERIFY").is_some() {
            use std::fmt::Write as _;
            let g = plan
                .arena
                .lock()
                .map_err(|_| Error::Device("arena 锁中毒".into()))?;
            let (mut nbad, mut first) = (0usize, String::new());
            for (off, data) in &plan.statics {
                let lo = *off as usize;
                let hi = lo + data.len();
                let rn = plan.rn_word as usize;
                if lo <= rn && rn < hi {
                    continue; // host 每 run 覆写，内容本就与清单不同
                }
                // SAFETY: 偏移在本 arena 映射内（同 `ensure_statics`）。
                let live = unsafe {
                    std::slice::from_raw_parts(g.base.add(lo * 4) as *const u32, data.len())
                };
                if live == data.as_slice() {
                    continue;
                }
                nbad += 1;
                if nbad <= 3 {
                    let (mut hv, mut wv) = (String::new(), String::new());
                    for (i, (a, b)) in live.iter().zip(data.iter()).enumerate() {
                        if a != b {
                            let _ = write!(hv, "@{i}:{a:08x} ");
                            let _ = write!(wv, "@{i}:{b:08x} ");
                        }
                    }
                    // 判据的核心三问：这块的 `Arc` 指针、账本里同 (偏移,字数) 记的
                    // 指针、以及**三元组是否真的命中账本**。命中 ⇒ 跳过判据踩到了
                    // 「同一个 `Arc` 地址先后属于两个 `Vec`」（有 free 才有这种复用）；
                    // 不命中而内容又不对 ⇒ 写完之后被压掉（另一个计划的激活区）。
                    let id = Arc::as_ptr(data) as usize;
                    let led = g
                        .blobs
                        .iter()
                        .find(|(o, l, _)| o == off && *l == data.len() as u32);
                    let own = plan.dispatches.iter().find(|d| d.p_off == *off);
                    let _ = write!(
                        first,
                        "\n       off={off} len={} 本块ptr={id:x} 账本ptr={} 命中账本={} \
                         归属={} 实[{hv}] 应[{wv}]",
                        data.len(),
                        led.map_or_else(|| "-".to_string(), |(_, _, p)| format!("{p:x}")),
                        led.is_some_and(|(_, _, p)| *p == id),
                        own.map_or_else(
                            || "（不是任何 dispatch 的参数块）".to_string(),
                            |d| format!("{} / {} 网格={:?}", d.kernel, d.node, d.groups)
                        ),
                    );
                }
            }
            eprintln!(
                "[verify] run#{run_seq} {in_shape:?}：静态 {}/{} 块不在位{}",
                nbad,
                plan.statics.len(),
                if first.is_empty() {
                    String::new()
                } else {
                    format!(" ——{first}")
                }
            );
            if nbad > 0 && std::env::var_os("QPPOCR_METAL_STATIC_VERIFY_WARN").is_none() {
                return Err(Error::Device(format!(
                    "{in_shape:?} 静态区 {} 块不在位——账本把别的计划的字节当成了本计划的，\
                     已拒绝派发（参数错会让内核按别人的偏移/形状跑；只警告不拦：\
                     设 QPPOCR_METAL_STATIC_VERIFY_WARN=1）",
                    nbad
                )));
            }
        }
        // 诊断（`QPPOCR_METAL_EXIT_PARAM=1`，纯读）：把 rec 出口节点
        // （`n_exit3_argmax`）的参数块、网格、输出偏移在**派发前**逐字打出来。
        // 坏轮若在这块显示的是**别的计划**的 `out_off`/`rows`（如预热计划
        // `[1,3,64,64]` 的 `rows=8`），那就是「出口把 16 字写去了别处」的直证。
        if self.argmax_exit && std::env::var_os("QPPOCR_METAL_EXIT_PARAM").is_some() {
            use std::fmt::Write as _;
            if let Some(d) = plan.dispatches.iter().find(|d| d.kernel == "n_exit3_argmax") {
                let blob = plan.statics.iter().find(|(off, _)| *off == d.p_off);
                let n = blob.map(|(_, v)| v.len()).unwrap_or(6).max(6);
                let (mut have, mut want) = (String::new(), String::new());
                // SAFETY: `p_off` 是本计划的静态分配，其后的 6 个字在映射内。
                unsafe {
                    let p = plan.base.add(d.p_off as usize * 4) as *const u32;
                    for i in 0..n {
                        let _ = write!(have, "{:08x} ", std::ptr::read_unaligned(p.add(i)));
                    }
                }
                if let Some((_, v)) = blob {
                    for w in v.iter() {
                        let _ = write!(want, "{w:08x} ");
                    }
                }
                // SAFETY: `rn_word` 是 build 期静态分配（构造上恒 0）。
                let rn = unsafe {
                    std::ptr::read_unaligned(plan.base.add(plan.rn_word as usize * 4) as *const u32)
                };
                eprintln!(
                    "[exit-param] run#{run_seq} {in_shape:?} p_off={} out_off={} groups={:?} \
                     应有[{want}]实有[{have}]rn={rn}（rn 是 host 每 run 覆写的真实行数；\
                     出口守门 = 行号 >= rn*rpb 才早退）",
                    d.p_off, d.out_off, d.groups
                );
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
        // 诊断开关：把 rec 计划里指定内核的派发**连续重复 N 次**
        // （`QPPOCR_METAL_REC_REPEAT=n_conv8,n_conv` + `QPPOCR_METAL_REC_REPEAT_N=2`）。
        //
        // 与 STUB_REC 不同，**这个开关不改变数值**：内核是输入的纯函数，
        // 连续跑两遍第二遍写的字节与第一遍逐位相同，下游读到的仍是对的。
        // 所以它是唯一能「在真实流水线里、带着真实 arena 与真实上游数据」
        // 量「第 k 遍的边际成本」的手段。
        //
        // 问的是：生产里 rec 的 conv8 是**冷启动/冷 L2**主导，还是稳态算力
        // 主导？微基准 `conv8_grid_scaling_real_shapes` 用 reps 循环量同一个
        // 内核、同样的形状，得到 750–1100 GFLOPS（权重每 rep 都被重读 ⇒
        // L2 热）；而端到端消融反推只有 ~240 GFLOPS。**第二遍若便宜很多，
        // 差价就是「权重没留在 L2 / 每条 dispatch 都要重新起转」——那是可以
        // 靠跨行复用权重拿回来的（11 行独立，却各读一遍全部 20.8 MB）。**
        let rep_rec: Vec<String> = std::env::var("QPPOCR_METAL_REC_REPEAT")
            .ok()
            .map(|v| {
                v.split(',')
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let rep_rec_n: u32 = std::env::var("QPPOCR_METAL_REC_REPEAT_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(1);
        // 匹配规则与上面的 `skip_rec` 一致：裸名比内核名，`@名` 比节点名。
        //
        // `@` 这一支是**补的**：原实现形参写作 `_node` 直接丢掉，于是
        // `QPPOCR_METAL_REC_REPEAT=@Conv.42` 恒不命中、delta 恒为 0——2026-10-09
        // 那次「@Node 探针测出 0」不是「dispatch 免费」，是**仪器没接线**。
        // 逐节点量「同一个 conv 在生产里 vs 孤立跑」的差价全靠它（§13.3 留的
        // 209 vs 422 GB/s 那个开口），所以必须真的能选中单个节点。
        let rec_reps = |k: &str, node: &str| -> u32 {
            let hit = rep_rec.iter().any(|s| match s.strip_prefix('@') {
                Some(n) => n == node,
                None => s == k,
            });
            if self.argmax_exit && hit {
                rep_rec_n
            } else {
                1
            }
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
        // 并发（--workers 0）时多个线程的日志会交错，带线程号才能归属。
        let tid = std::thread::current().id();
        // 消融模式下数字一定是错的，每次 run 都喊一声，免得忘了自己在量什么。
        if self.argmax_exit && !stub_rec.is_empty() {
            eprintln!(
                "[metal] ⚠ 消融模式：rec 计划（run#{run_seq}）跳过 {stub_rec:?} 的派发，\
                 **本次读数只作计时用、数值无效**"
            );
        }
        // 诊断开关（`QPPOCR_METAL_WATCH=<前缀>` + `QPPOCR_METAL_WATCH_WIN=<off:len>[,...]`，
        // 偏移是**激活区相对**的字下标，与 `IO_DUMP` 同一坐标系）：每提交完一块
        // （配 `CHUNK=1` 时即每条派发）就把这些窗口读出来追加到 `<前缀>.trace`，
        // 并在本轮开跑前先落一份 `<前缀>.pre`。
        //
        // 用来回答「某个字区间在坏轮里到底有没有被写过」：好轮/坏轮各一条 trace，
        // **第一条窗口内容不同的派发**就是凶手；若坏轮里某窗口自始至终都等于
        // `.pre`（环境内容）⇒ 那几十字节**压根没被本计划的任何派发写过**，
        // 问题就落在「谁本该写它」而不是「谁写坏了它」。
        let watch = std::env::var_os("QPPOCR_METAL_WATCH").map(|p| {
            let wins: Vec<(usize, usize)> = std::env::var("QPPOCR_METAL_WATCH_WIN")
                .unwrap_or_default()
                .split(',')
                .filter_map(|s| {
                    let (a, b) = s.split_once(':')?;
                    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
                })
                .collect();
            let shape = in_shape
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("x");
            (format!("{}.run{run_seq:03}.{shape}", p.to_string_lossy()), wins)
        });
        /// 把若干「激活区相对字偏移」窗口读成一行十六进制（诊断用，纯读）。
        unsafe fn watch_line(base: *const u8, sw: usize, wins: &[(usize, usize)]) -> String {
            use std::fmt::Write as _;
            let mut s = String::with_capacity(wins.iter().map(|&(_, l)| l * 9 + 1).sum());
            for &(off, len) in wins {
                for k in 0..len {
                    // SAFETY: 调用方保证窗口在映射内。
                    let w = unsafe {
                        std::ptr::read_unaligned(base.add((sw + off + k) * 4) as *const u32)
                    };
                    let _ = write!(s, "{w:08x} ");
                }
                s.push('|');
            }
            s
        }
        if let (true, Some((prefix, wins))) = (self.argmax_exit, watch.as_ref()) {
            if !wins.is_empty() {
                let sw = (plan.static_bytes / 4) as usize;
                // SAFETY: 窗口都在 [static_words, total_words) 内（Layout 两段分离），
                // 且此刻还未提交任何 dispatch，无并发写。
                let line = unsafe { watch_line(plan.base, sw, wins) };
                let path = format!("{prefix}.pre");
                let _ = std::fs::write(&path, format!("{line}\n"));
            }
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
                            for _ in 0..rec_reps(d.kernel, &d.node) {
                                self.ks
                                    .dispatch(enc, d.kernel, &plan.buf, d.p_off, d.groups)
                                    .expect("dispatch（内核名来自计划，必在套件内）");
                            }
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
                // 逐块窗口采样（见上面 `QPPOCR_METAL_WATCH` 的说明）。
                if let (true, Some((prefix, wins))) = (self.argmax_exit, watch.as_ref()) {
                    if !wins.is_empty() {
                        let sw = (plan.static_bytes / 4) as usize;
                        // SAFETY: 同上（窗口在映射内）；本块已 wait 返回，GPU 不再写。
                        let line = unsafe { watch_line(plan.base, sw, wins) };
                        let path = format!("{prefix}.trace");
                        use std::io::Write as _;
                        if let Ok(mut f) = std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(&path)
                        {
                            let _ = writeln!(
                                f,
                                "{lo:>4}..{:<4} {:>22} {line}",
                                lo + blk.len(),
                                blk.first().map_or("?", |d| d.kernel),
                            );
                        }
                    }
                }
            }
            (enc_acc, wait_acc)
        } else {
            submit_compute_timed(&self.ctx.queue, |enc| {
                for d in &plan.dispatches {
                    if !skip_rec(d.kernel, &d.node) {
                        for _ in 0..rec_reps(d.kernel, &d.node) {
                            self.ks
                                .dispatch(enc, d.kernel, &plan.buf, d.p_off, d.groups)
                                .expect("dispatch（内核名来自计划，必在套件内）");
                        }
                    }
                    if !no_barrier {
                        let resources: [&metal::ResourceRef; 1] = [&plan.buf];
                        enc.memory_barrier_with_resources(&resources);
                    }
                }
            })?
        };
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        // 诊断开关：把本计划**跑完之后**的激活区落盘（`QPPOCR_METAL_IO_DUMP=<前缀>`）。
        //
        // 与 `QPPOCR_METAL_ARENA_DUMP` 互补：那个只在 build 后、覆盖 [0,total)，
        // 看不到任何运行期中间量。这个覆盖 [静态段末, 计划末) = 输入 + 全部中间
        // 激活，用来在「同输入、同程序、却逐轮结果不同」时定位**第一处分歧字节**。
        // 文件名带 `run_seq` 与形状，多次运行之间可直接 `cmp` 对齐。
        // 纯读、不碰任何状态；不设变量时零开销。
        // 只落 **rec 计划**（`argmax_exit`）：det 的激活区大一个量级，全落盘会撑爆。
        let io_dump = std::env::var_os("QPPOCR_METAL_IO_DUMP");
        if let (true, Some(prefix)) = (self.argmax_exit, io_dump) {
            let start = (plan.static_bytes / 4) as usize;
            let n = (plan.block_bytes / 4) as usize;
            let shape = in_shape
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("x");
            let path = format!("{}.run{run_seq:03}.{shape}.bin", prefix.to_string_lossy());
            // SAFETY: [static_words, total_words) 是激活段（`Layout` 两段分离），
            // 在映射内；run 已同步结束且全程持执行锁，无并发写。
            let slice = unsafe { std::slice::from_raw_parts(plan.base.add(start * 4), n * 4) };
            if let Err(e) = std::fs::write(&path, slice) {
                eprintln!("[io-dump] 写 {path} 失败：{e}");
            }
            // 单独落**输入区**（`in_f32_off`，长度 = 本次 host 张量长度）——
            // 判别「分歧发生在 rec 内部」还是「host 喂进来的输入本就逐轮不同」。
            // SAFETY: in_f32_off + in_words ≤ total_words（memcpy 按输入形状分配）。
            let ins = unsafe {
                std::slice::from_raw_parts(
                    plan.base.add(plan.in_f32_off as usize * 4),
                    in_words * 4,
                )
            };
            let ipath = format!("{path}.in_post");
            if let Err(e) = std::fs::write(&ipath, ins) {
                eprintln!("[io-dump] 写 {ipath} 失败：{e}");
            }
            eprintln!(
                "[io-dump] run#{run_seq:03} {in_shape:?} 落盘：static_words={} \
                 in_f32_off={} in_words={in_words} 激活区 words={} → {}",
                plan.static_bytes / 4,
                plan.in_f32_off,
                plan.block_bytes / 4,
                path
            );
        }
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
