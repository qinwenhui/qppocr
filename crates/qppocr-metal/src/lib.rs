//! qppocr-metal：qppocr 的 Metal 设备后端（Apple GPU 原生直连）。
//!
//! macOS 上 Metal 是唯一的原生 GPU API（Vulkan 需经 MoltenVK 翻译层，
//! 内核调优假设经翻译后不再保真）；Apple 平台的硬件面又高度收敛——
//! 统一内存、TBDR 架构、simdgroup 恒 32 lane，一份内核可以针对整个
//! M 系列家族调优。本 crate 就是给这个环境做原生直连的。
//!
//! 实现与 Vulkan 后端（`qppocr-gpu`）镜像同构：装载期把图与权重一次
//! 性移入设备、单一大 arena + 参数块间接寻址、内核经偏移访问数据——
//! GLSL compute 与 MSL 的绑定模型一一对应（同一缓冲的 f32/u32/float4
//! 三视图绑在 buffer 0/1/2，push constant 由 `set_bytes` 承担）。内核
//! 源码（`shaders/*.metal`）与 `qppocr-gpu/shaders/*.comp` 逐行对应，
//! 改语义必须两边同步。
//!
//! - **MSL 源码内嵌、运行时编译**（`newLibraryWithSource`）：无需离线
//!   shader 编译器，装载期一次编译、全程复用。
//! - **统一内存零拷贝**：`StorageModeShared` 的缓冲 CPU/GPU 同址可见
//!   （Apple Silicon 全系 UMA），无 staging 往返。
//!
//! 范围（P1 地基）：设备枚举 / 打开 + `n_entry`/`n_elem` 内核位级冒烟；
//! **整图执行尚未接入**——`create_session` 按 [`qppocr_core::device::DeviceContext`]
//! 契约明确报错并列出图中算子，不静默降级。det 全算子在 P2 接入。
//!
//! 非 macOS 平台（或 feature `metal` 关闭）编译为空枚举 + 明确报错——
//! 与 Vulkan 侧「无 loader 枚举为空、显式要求才报错」同一纪律。
//!
//! 本 crate 是 Metal 相关 unsafe 的汇聚地：metal 绑定的 objc 消息面
//! 封在 `backend` 模块内部，不外泄；unsafe 块逐个带 Safety 注释。

/// macOS 实装（feature `metal` 开启时）；否则编译 [`fallback`] 的占位。
#[cfg(all(feature = "metal", target_os = "macos"))]
mod backend;
/// 非 macOS / 未编 Metal 时的占位实现——类型面完整，行为全是明确报错。
#[cfg(not(all(feature = "metal", target_os = "macos")))]
mod fallback;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use backend::MetalContext;
#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub use fallback::MetalContext;

use qppocr_core::device::DeviceInfo;

/// 枚举本机全部 Metal 设备。
///
/// 无 Metal（非 macOS / feature 未编 / 系统过旧）= 空列表，绝不当错误
/// ——「有没有 GPU」是环境事实，只有显式要求时才构成错误（与
/// `qppocr_gpu::list_devices` 同一约定）。
pub fn list_devices() -> Vec<DeviceInfo> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    return backend::enumerate();
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    Vec::new()
}
