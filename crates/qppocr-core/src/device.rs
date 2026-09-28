//! 设备接缝：多后端（CPU / Vulkan / CUDA）在引擎层的抽象。
//!
//! 与 `qppocr-kernels::Backend` 的分工（后者文档同款约定）：CPU 内核
//! 后端（AVX2/NEON/标量）是**同一内存模型下**的微内核选择，归 kernels
//! 的 `arch` 分发层；设备级后端的内存模型与调度粒度都不同，属于这里
//! ——kernels 层不感知设备，本层不关心微内核。
//!
//! 接缝刻意最小：[`DeviceContext`] 装载期每模型建一个会话（权重一次性
//! 驻留设备），[`DeviceSession::run`] 是唯一执行入口——与
//! [`crate::executor::Session::run`] 同签名，pipeline 对 det/cls/rec 的
//! 全部编排经它穿过，预处理/后处理留在设备边界外。虚调用只在 run
//! 边界（一次推理一次，10–100 ms 粒度）；逐算子路径绝不进 trait。
//!
//! 诊断（`QPPOCR_DUMP_DIR` / `QPPOCR_PROF`）由各实现内部环境变量驱动，
//! 不进 trait 面——CPU 实现里的环境检查原地不动。唯一的共享状态是
//! dump 会话 id 计数器（[`next_dump_session_id`]）：det-GPU + rec-CPU
//! 的混合部署下，两边各持一个计数器会让对拍目录互相覆盖。

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;
use crate::onnx::model::Graph;
use crate::tensor::Tensor;

/// 设备类别。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    /// 本机 CPU（[`crate::executor::Session`]）。
    Cpu,
    /// Vulkan 计算设备。
    Vulkan,
    /// CUDA 计算设备。
    Cuda,
}

/// 设备的人读描述（日志 / bench 报告用）。
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    /// 设备类别。
    pub kind: DeviceKind,
    /// 设备名，如 `"Intel(R) Arc Graphics"`；CPU 是 `"host (x86_64)"`。
    pub name: String,
    /// API 与版本串，如 `"vulkan 1.4"`、`"cpu avx2"`。
    pub api: String,
}

/// 会话构造选项。
///
/// 空壳起步（`#[non_exhaustive]`）：precision（fp16）等旋钮后续版本加，
/// 调用方一律经 [`Default`] 构造。
#[non_exhaustive]
#[derive(Debug, Default)]
pub struct SessionOptions {}

/// 设备上下文：装载期工厂，一个设备一份。
pub trait DeviceContext: Send + Sync {
    /// 设备类别。
    fn kind(&self) -> DeviceKind;
    /// 人读描述。
    fn info(&self) -> DeviceInfo;
    /// 从（已优化的）图与权重组装会话。
    ///
    /// 契约：`graph` 已过 CPU 侧装载期图优化（融合算子语义见
    /// `executor.rs` 的算子 match）；`graph.initializers` 恒为空——权重
    /// 全在第二参（loader 的 take 语义），按值 move（克隆即双份常驻）。
    /// 实现把权重一次性驻留设备；不支持的算子在此报
    /// [`crate::error::Error::Device`] 并点名，不静默降级。
    fn create_session(
        &self,
        graph: Graph,
        initializers: HashMap<String, Tensor>,
        opts: &SessionOptions,
    ) -> Result<Arc<dyn DeviceSession>>;
}

/// 设备会话：一次装载、多次 run。
///
/// `Send + Sync` 且 `&self` 并发安全——rec 分片 / cls 批的现实调用形态
/// 就是多线程并发 `run`（权重只读）。
pub trait DeviceSession: Send + Sync {
    /// 设备类别。
    fn kind(&self) -> DeviceKind;
    /// 跑一遍图。`inputs` 是 (名字, 张量) 列表；返回按图输出顺序排列的
    /// 输出。
    fn run(&self, inputs: Vec<(String, Tensor)>) -> Result<Vec<Tensor>>;
    /// 是否受益于**宿主侧**多线程扇出：CPU = true；GPU 内部已并行，
    /// false 时 pipeline 的 cls / rec 批退化为串行提交（省线程 spawn
    /// 与提交争抢）。
    fn prefers_host_parallelism(&self) -> bool;
}

impl DeviceSession for crate::executor::Session {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn run(&self, inputs: Vec<(String, Tensor)>) -> Result<Vec<Tensor>> {
        crate::executor::Session::run(self, inputs)
    }

    fn prefers_host_parallelism(&self) -> bool {
        true
    }
}

/// CPU 设备上下文：默认工厂，包装 [`crate::executor::Session`]。
pub struct CpuContext;

impl DeviceContext for CpuContext {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn info(&self) -> DeviceInfo {
        DeviceInfo {
            kind: DeviceKind::Cpu,
            name: format!("host ({})", std::env::consts::ARCH),
            api: format!("cpu {:?}", qppocr_kernels::detect_backend()),
        }
    }

    fn create_session(
        &self,
        graph: Graph,
        initializers: HashMap<String, Tensor>,
        _opts: &SessionOptions,
    ) -> Result<Arc<dyn DeviceSession>> {
        Ok(Arc::new(crate::executor::Session::from_parts(
            graph,
            initializers,
        )))
    }
}

/// 默认（CPU）设备上下文。
pub fn cpu_context() -> Arc<dyn DeviceContext> {
    Arc::new(CpuContext)
}

/// `QPPOCR_DUMP_DIR` 的会话 id（全局递增）。
///
/// det/cls/rec 三个会话的节点索引都从 0 起，落盘文件名靠 `s{id}_` 前缀
/// 区分归属。计数器放这里而不是各实现内部：混合部署（如 det 走 GPU、
/// rec 走 CPU）下两边各持一个计数器，id 会撞、对拍目录互相覆盖。
static DUMP_SESSION_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// 取下一个 dump 会话 id。
pub fn next_dump_session_id() -> usize {
    DUMP_SESSION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
