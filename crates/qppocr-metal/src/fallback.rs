//! 占位实现：本平台不是 macOS（或 feature `metal` 未编）。
//!
//! 类型面与真后端一致（`MetalContext` 可被门面无 cfg 地引用）；行为
//! 全是带原因的明确报错——绝不让「以为在跑 Metal」的调用静默变味。

use qppocr_core::device::{DeviceContext, DeviceInfo, DeviceKind, DeviceSession, SessionOptions};
use qppocr_core::error::{Error, Result};
use qppocr_core::onnx::model::Graph;
use qppocr_core::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

const WHY: &str = if cfg!(target_os = "macos") {
    "Metal 后端未编译（feature \"metal\" 关闭）"
} else {
    "Metal 仅在 macOS 上可用"
};

/// Metal 设备上下文的占位：`open` 恒报错（本平台无 Metal）。
pub struct MetalContext;

impl MetalContext {
    /// 恒报错——非 macOS 平台没有 Metal；错误信息给出替代路径。
    pub fn open(index: Option<u32>) -> Result<Self> {
        let _ = index;
        Err(Error::Device(format!(
            "{WHY}；用 --device vulkan 或 --device cpu"
        )))
    }

    /// 整图执行是否就绪（见真后端的同名方法）。占位恒 false——本平台
    /// `open` 恒失败，此值实际不可达（门面的 Auto 也只在 macOS 查询）。
    pub fn sessions_ready(&self) -> bool {
        false
    }
}

impl DeviceContext for MetalContext {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Metal
    }

    fn info(&self) -> DeviceInfo {
        DeviceInfo {
            kind: DeviceKind::Metal,
            name: "metal (本平台不可用)".into(),
            api: WHY.into(),
        }
    }

    fn create_session(
        &self,
        graph: Graph,
        initializers: HashMap<String, Tensor>,
        _opts: &SessionOptions,
    ) -> Result<Arc<dyn DeviceSession>> {
        // 占位：到不了这里（open 已拦），实现只为完整闭合 trait 面。
        drop(graph);
        drop(initializers);
        Err(Error::Device(format!(
            "{WHY}；用 --device vulkan 或 --device cpu"
        )))
    }
}
