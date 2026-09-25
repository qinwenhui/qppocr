//! qppocr 引擎主体：ONNX 解析、图优化、执行器、det/cls/rec 流水线与后处理。
//!
//! 本 crate `forbid(unsafe_code)` —— 所有 SIMD 内核都在 `qppocr-kernels` 里，
//! 这里只经安全 API 调用。

#![forbid(unsafe_code)]
//  镜像代码：字面值/结构照抄（位级一致），不做惯用化改写
#![allow(
    clippy::excessive_precision,
    clippy::field_reassign_with_default,
    clippy::approx_constant,
    clippy::needless_range_loop,
    clippy::redundant_guards
)]

pub mod error;
pub use error::Error;

/// 引擎线程池的总线程数（含主线程）。批量多进程部署按它折算每进程
/// 线程数（`thread_count / 进程数`）。
pub fn thread_count() -> usize {
    #[cfg(feature = "parallel")]
    {
        qppocr_kernels::par::pool_thread_count()
    }
    #[cfg(not(feature = "parallel"))]
    {
        qppocr_kernels::par::threads()
    }
}

pub mod executor;
pub mod graph;
pub mod onnx;
pub mod pipeline;
pub mod tensor;
