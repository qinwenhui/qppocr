//! `qppocr` —— 纯 Rust、手写内核的 PP-OCRv6 推理引擎。
//!
//! 不依赖 ONNX Runtime、不依赖 tract、不依赖任何 C/C++ 库：ONNX 解析、
//! 图优化、算子、调度、检测/方向/识别流水线、后处理，全部自己实现。
//!
//! 引擎与 UI 无关：核心 API 是「一张图进、一个结果出」。批量、切片、
//! 多进程、GUI 都是上层的事。
//!
//! ```text
//! let engine = Engine::new(Tier::Small, "models/")?;
//! let out = engine.run_image_file("receipt.png")?;
//! for line in &out.lines {
//!     println!("{:.2}  {}", line.confidence, line.text);
//! }
//! ```
//!
//! （API 在阶段 4 落地；上面的例子届时会变成可运行的 doctest。）
//!
//! **本仓库不含模型**：PP-OCRv6 权重来自 PaddlePaddle（Apache-2.0），
//! 请自行获取，见仓库 [`NOTICE`] 与设计文档附录 D 的清单及 SHA-256。
//!
//! [`NOTICE`]: https://github.com/qinwenhui/qppocr/blob/main/NOTICE

#![forbid(unsafe_code)]
