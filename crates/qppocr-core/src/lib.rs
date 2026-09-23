//! qppocr 引擎主体：ONNX 解析、图优化、执行器、det/cls/rec 流水线与后处理。
//!
//! 本 crate `forbid(unsafe_code)` —— 所有 SIMD 内核都在 `qppocr-kernels` 里，
//! 这里只经安全 API 调用（DESIGN.md §2 铁律 3）。

#![forbid(unsafe_code)]
