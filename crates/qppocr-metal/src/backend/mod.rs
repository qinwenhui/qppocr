//! Metal 后端：设备枚举、设备上下文与命令队列。
//!
//! 里程碑（对齐 Vulkan 后端的 Phase 划分）：
//! - **P1**：枚举 + 设备打开（本模块）+ 命令提交（[`device`]）+ MSL
//!   运行时编译与内核分发（[`pipeline`]）+ `n_entry`/`n_elem` 位级冒烟
//!   （`kernel_tests`）；
//! - **P2（当前）**：det 全算子整图会话（[`session`]——计划构建与
//!   Vulkan 共用 `qppocr_gpu::plan`，det 端到端与 CPU 对拍）；
//! - **P3**：rec/cls + 分级部署；**P4**：Apple GPU 调优（TBDR /
//!   simdgroup / 统一内存零拷贝上传）。
//!
//! 与 Vulkan 侧的对应关系：`MTLDevice` ≈ 物理+逻辑设备合一（Apple
//! 侧无两阶段），`MTLCommandQueue` ≈ 计算队列，`MTLCommandBuffer` +
//! `wait_until_completed` ≈ 提交 + timeline 等值（P3 批间流水再上
//! `add_completed_handler` 异步收尾）。

pub(crate) mod device;
#[cfg(test)]
mod kernel_tests;
pub(crate) mod pipeline;
pub(crate) mod session;

use metal::MTLGPUFamily;
use qppocr_core::device::{
    DeviceContext, DeviceInfo, DeviceKind, DeviceSession, ModelRole, SessionOptions,
};
use qppocr_core::error::{Error, Result};
use qppocr_core::onnx::model::Graph;
use qppocr_core::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

/// 枚举本机全部 Metal 设备（`MTLCopyAllDevices`）。无 Metal = 空列表。
pub(crate) fn enumerate() -> Vec<DeviceInfo> {
    metal::Device::all()
        .iter()
        .map(|d| DeviceInfo {
            kind: DeviceKind::Metal,
            name: d.name().into(),
            api: family_api(d),
        })
        .collect()
}

/// 设备能力描述：报最高支持的 GPU 家族。
///
/// 只报探测事实（`supportsFamily`），不做芯片代际映射——家族号本身
/// （apple7/8/9、metal4……）就是 Apple 官方的能力分层，够诊断用。
fn family_api(d: &metal::DeviceRef) -> String {
    let families = [
        (MTLGPUFamily::Metal4, "metal4"),
        (MTLGPUFamily::Apple9, "apple9"),
        (MTLGPUFamily::Apple8, "apple8"),
        (MTLGPUFamily::Apple7, "apple7"),
        (MTLGPUFamily::Apple6, "apple6"),
        (MTLGPUFamily::Apple5, "apple5"),
        (MTLGPUFamily::Apple4, "apple4"),
        (MTLGPUFamily::Mac2, "mac2"),
        (MTLGPUFamily::Mac1, "mac1"),
    ];
    for (fam, name) in families {
        if d.supports_family(fam) {
            return format!("metal {name}");
        }
    }
    "metal".into()
}

/// 上下文的实体（Arc 内共享）。
pub(crate) struct Inner {
    name: String,
    api: String,
    /// Metal 设备（物理/逻辑合一）。metal-rs 的 owned 类型均为
    /// `unsafe impl Sync + Send`（objc 引用计数对象），可跨线程共享。
    pub(crate) device: metal::Device,
    /// 命令队列：会话的每次前向经它出命令缓冲。
    pub(crate) queue: metal::CommandQueue,
}

/// 一个 Metal 设备的上下文。
///
/// **Clone = Arc 共享**：会话持有整个上下文（与 `VulkanContext` 同款
/// 纪律——上下文的生命期必须盖住它派生的全部会话资源）。
#[derive(Clone)]
pub struct MetalContext {
    pub(crate) inner: Arc<Inner>,
}

impl MetalContext {
    /// 打开第 `index` 个 Metal 设备（`None` = 第一个）。
    ///
    /// 不满足条件时明确报错并**列出看到的一切**，不静默降级——与
    /// `VulkanContext::open` 同一纪律。
    pub fn open(index: Option<u32>) -> Result<Self> {
        let all = metal::Device::all();
        if all.is_empty() {
            return Err(Error::Device(
                "没有任何 Metal 设备——macOS 10.11+ 的 Mac 均具备；\
                 虚拟机需透传 GPU（Apple Silicon VM 默认有）"
                    .into(),
            ));
        }
        let seen = || {
            all.iter()
                .map(|d| format!("{} ({})", d.name(), family_api(d)))
                .collect::<Vec<_>>()
                .join("; ")
        };
        let pick = match index {
            Some(i) => all.get(i as usize).ok_or_else(|| {
                Error::Device(format!(
                    "--device metal:{i} 超出范围（共 {} 个 Metal 设备）：{}",
                    all.len(),
                    seen()
                ))
            })?,
            None => &all[0],
        };
        Ok(Self {
            inner: Arc::new(Inner {
                name: pick.name().into(),
                api: family_api(pick),
                device: pick.clone(),
                queue: pick.new_command_queue(),
            }),
        })
    }

    /// 整图执行是否就绪（det/rec/cls 全接入）。
    ///
    /// 门面的 `GpuApi::Auto` 在 macOS 据此优先 Metal（原生直连）；
    /// 未就绪的开发阶段让位 Vulkan/MoltenVK，用户无感。
    pub fn sessions_ready(&self) -> bool {
        true
    }
}

impl DeviceContext for MetalContext {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Metal
    }

    fn info(&self) -> DeviceInfo {
        DeviceInfo {
            kind: DeviceKind::Metal,
            name: self.inner.name.clone(),
            api: self.inner.api.clone(),
        }
    }

    fn create_session(
        &self,
        graph: Graph,
        initializers: HashMap<String, Tensor>,
        opts: &SessionOptions,
    ) -> Result<Arc<dyn DeviceSession>> {
        // 部署分级（QPPOCR_GPU_STAGES，默认 **all**，与 Vulkan 侧同款
        // 语义）：det+rec+cls 全上 GPU；=detrec 可退回 cls-CPU、=det 退回
        // 仅-det（rec 走 CPU——一次性单图冷进程避付 rec 计划构建税）。
        // 委托 CPU 时 **stderr 声明**，不是静默降级。
        let stages = std::env::var("QPPOCR_GPU_STAGES").unwrap_or_else(|_| "all".into());
        let gpu_ok = match opts.model {
            ModelRole::Det => true,
            ModelRole::Rec => stages == "detrec" || stages == "all",
            ModelRole::Cls => stages == "all",
            _ => false,
        };
        if !gpu_ok && opts.model != ModelRole::Det {
            eprintln!(
                "[metal] QPPOCR_GPU_STAGES={stages}：{:?} 模型走 CPU 会话（分级部署）",
                opts.model
            );
            return Ok(Arc::new(qppocr_core::executor::Session::from_parts(
                graph,
                initializers,
            )));
        }
        // rec：动态 B MatMul = 注意力头——无注意力 mask 的模型对桶宽右
        // 零填充**不 invariant**（填充步经注意力混入真步概率；CPU 实证
        // 掉行）——退精确宽。纯卷积 rec（tiny）实证填充不变，维持 64 桶。
        let exact_width = opts.model == ModelRole::Rec
            && graph.nodes.iter().any(|n| {
                n.op_type == "MatMul"
                    && n.inputs
                        .get(1)
                        .is_some_and(|b| !initializers.contains_key(b))
            });
        let mut session = session::MetalSession::new(self.inner.clone(), graph, initializers)?;
        // 覆盖面探针：算子支持与形状无关，试建一次小计划——不支持时
        // stderr 声明后退回 CPU 会话（非静默降级）。探针留在缓存（真
        // 形状另建，可忽略）。
        if let Err(e) = session.probe(&[1, 3, 48, 64]) {
            eprintln!(
                "[metal] {:?} 会话探测失败（{e}）：走 CPU 会话（分级部署，\
                 补齐 n_ 覆盖面后自动上 GPU）",
                opts.model
            );
            return Ok(Arc::new(session.into_cpu_parts()));
        }
        // 计划预算按角色分配（与 Vulkan 侧同款）：det 形状逐图一次性、
        // 暖缓存重建便宜 → 384MB；rec 桶形状每图复用 → 1536MB（全进、
        // 零驱逐）。QPPOCR_GPU_PLAN_MB 仍可覆写。
        session.plan_budget_mb = match opts.model {
            ModelRole::Det => 384,
            _ => 1536,
        };
        // rec 会话批维补齐粒度 8：引擎按此合批（宽填充到批内最大桶宽），
        // 会话补齐空行 + 内核 real_n 早退。
        if opts.model == ModelRole::Rec {
            session.batch_grain = 8;
            // CTC 头只吃每时间步 argmax：GPU 端归约成 (val, idx) 对再回读
            //（全量 T×V 概率回读纯属带宽浪费）。
            session.argmax_exit = true;
            session.exact_width = exact_width;
        }
        Ok(Arc::new(session))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 枚举在 Metal 机器上应列出（CI 的 macos runner 预期一项）；无
    /// Metal 的环境（CI 非 mac job）自动通过。
    #[test]
    fn enumerate_lists_or_skips() {
        for d in enumerate() {
            assert_eq!(d.kind, DeviceKind::Metal);
            eprintln!("[metal] {} ({})", d.name, d.api);
        }
    }

    /// 有 Metal 的机器上 open(None) 必须成功；无设备环境跳过（CI）。
    #[test]
    fn open_first_or_skip() {
        match MetalContext::open(None) {
            Ok(ctx) => {
                assert_eq!(ctx.kind(), DeviceKind::Metal);
                eprintln!(
                    "[metal] opened: {} ({}) | unified={:?} | max_tg={:?}",
                    ctx.inner.name,
                    ctx.inner.api,
                    ctx.inner.device.has_unified_memory(),
                    ctx.inner.device.max_threads_per_threadgroup()
                );
            }
            Err(e) => eprintln!("[metal] open 跳过（无 Metal 设备）: {e}"),
        }
    }
}
