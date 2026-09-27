//! ONNX 解析：手写 protobuf 读取器 + 内存图表示。

pub mod model;
pub mod parse;

pub use model::{Attribute, Graph, Node};
pub use parse::{load_onnx, load_onnx_memory};
