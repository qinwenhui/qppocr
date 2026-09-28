//! qppocr-gpu：qppocr 的 GPU 设备后端。
//!
//! 实现 `qppocr-core` 的设备接缝（[`DeviceContext`]）：装载期经
//! [`DeviceContext::create_session`] 把（已优化的）图与权重一次性移入
//! 设备，执行入口与 CPU `Session::run` 同签名——pipeline 对设备无感，
//! 预处理 / DB 后处理 / 裁剪 / CTC 留在设备边界外的 CPU 上。
//!
//! - **Vulkan 计算后端**（feature `vulkan`，默认开）：Vulkan 1.4 基线
//!   （timeline semaphore / sync2 已核心化）。实例按 1.1 请求（最大化
//!   可枚举面），可用性按物理设备各自的 `apiVersion` 判定——低于 1.4
//!   的设备会出现在枚举列表里并标注，不静默消失。
//! - **CUDA**（feature `cuda`）：枚举级预留（dlopen 驱动，无编译期 SDK
//!   依赖），计算内核后续版本接入。
//!
//! 无 loader / 无 ICD 的机器上枚举返回空列表——只有用户**显式**要求 GPU
//! 时，空列表才变成明确报错（列出手头看到的一切），不静默回退 CPU。
//!
//! 本 crate 是 GPU 相关 unsafe 的汇聚地：ash 的 raw 指针面全部封在
//! `vulkan` 模块内部，不外泄；unsafe 块逐个带 Safety 注释。

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod vulkan;

use qppocr_core::device::DeviceInfo;

#[cfg(feature = "cuda")]
pub use cuda::CudaContext;
pub use vulkan::VulkanContext;

/// 枚举本机全部 GPU（跨已编译的后端，Vulkan 在前、CUDA 其后）。
///
/// 失败（无 loader / 无 ICD / 驱动异常）= 空列表，绝不当错误——
/// 「有没有 GPU」是环境事实，只有显式要求 GPU 时才构成错误。
pub fn list_devices() -> Vec<DeviceInfo> {
    let mut out = Vec::new();
    #[cfg(feature = "vulkan")]
    out.extend(vulkan::enumerate());
    #[cfg(feature = "cuda")]
    out.extend(cuda::enumerate());
    out
}
