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
use std::sync::{Arc, Mutex};

use qppocr_core::device::{DeviceKind, DeviceSession};
use qppocr_core::error::{Error, Result};
use qppocr_core::onnx::model::Graph;
use qppocr_core::tensor::{DType, Tensor};
use qppocr_gpu::plan::{self, PlanBuilder};
use qppocr_kernels::buf::F32Buf;

use super::device::submit_compute;
use super::pipeline::KernelSet;

/// 一个输入形状的整图计划（Metal 实体化）。
struct MetalPlan {
    /// 整块 Shared arena（统一内存：CPU/GPU 同址）。
    buf: metal::Buffer,
    /// `buf.contents()`——裸指针缓存（写输入/读输出用）。
    base: *mut u8,
    /// 图输入的 f32 NCHW 区偏移（host 每次 memcpy 的目标）。
    in_f32_off: u32,
    /// real_n 单元的 word 偏移（构造上恒 0）。
    rn_word: u32,
    /// 图输出：(名字, 偏移, 元素数)。
    out_offs: Vec<(String, u32, usize)>,
    out_shapes: Vec<Vec<i64>>,
    /// dispatch 序列（每 run 重编码执行）。
    dispatches: Vec<plan::Dispatch>,
    /// 整块字节数（计划缓存预算记账）。
    block_bytes: u64,
}

// SAFETY: base 是 buf（Shared arena）的持久映射指针，存活期与 MetalPlan
// 相同且无并发写（run 全程持会话执行锁）；metal owned 类型 Send+Sync。
unsafe impl Send for MetalPlan {}
// SAFETY: 同上——&MetalPlan 的共享只发生在持锁的 run 里。
unsafe impl Sync for MetalPlan {}

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
            // det 形状逐图一次性、共享计划重建便宜（无 arena 池化缺口，
            // 权重重排有缓存）→ 小预算；env 覆写。
            plan_budget_mb: 384,
            batch_grain: 1,
            exact_width: false,
            argmax_exit: false,
            run_lock: Mutex::new(()),
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

    fn build_plan(&self, in_shape: &[i64]) -> Result<MetalPlan> {
        let t0 = std::time::Instant::now();
        let built = PlanBuilder {
            graph: &self.graph,
            initializers: &self.initializers,
            repack_cache: &self.repack_cache,
            producer_idx: self.producer_idx(),
            argmax_exit: self.argmax_exit,
        }
        .build(&self.input_name, in_shape)?;

        // Shared arena 一次分配（统一内存，无 staging）
        let bytes = built.total_words as u64 * 4;
        let buf = self
            .ctx
            .device
            .new_buffer(bytes, metal::MTLResourceOptions::StorageModeShared);
        // contents() 是安全接口（返回持久映射指针；统一内存）
        let base = buf.contents() as *mut u8;
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
        // SAFETY: base 是整块映射；off+len ≤ total（上方断言）。
        unsafe {
            for (off, data) in &built.uploads {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    base.add(*off as usize * 4) as *mut u32,
                    data.len(),
                );
            }
            for (off, data) in &built.upload_init {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    base.add(*off as usize * 4) as *mut u32,
                    data.len(),
                );
            }
        }
        eprintln!(
            "[metal] 计划就绪：输入 {in_shape:?}，{} 个 dispatch，整块 {:.1} MB（建 {:.1} ms）",
            built.dispatches.len(),
            bytes as f64 / 1e6,
            t0.elapsed().as_secs_f64() * 1000.0
        );
        Ok(MetalPlan {
            base,
            in_f32_off: built.in_f32_off,
            rn_word: built.rn_word,
            out_offs: built.out_offs,
            out_shapes: built.out_shapes,
            dispatches: built.dispatches,
            block_bytes: bytes,
            buf,
        })
    }

    /// 找/建形状计划（含字节预算驱逐）。
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
        let t0 = std::time::Instant::now();
        submit_compute(&self.ctx.queue, |enc| {
            for d in &plan.dispatches {
                self.ks
                    .dispatch(enc, d.kernel, &plan.buf, d.p_off, d.groups)
                    .expect("dispatch（内核名来自计划，必在套件内）");
                let resources: [&metal::ResourceRef; 1] = [&plan.buf];
                enc.memory_barrier_with_resources(&resources);
            }
        })?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if std::env::var_os("QPPOCR_GPU_PROF").is_some() {
            eprintln!(
                "[metal] {} 个 dispatch，墙钟 {wall_ms:.2} ms（含编码 + GPU + 等待）",
                plan.dispatches.len()
            );
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

    fn rec_argmax_pairs(&self) -> bool {
        self.argmax_exit
    }
}
