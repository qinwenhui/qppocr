//! 错误类型。库不该强制调用方的错误类型——`anyhow`
//! 留给应用，这里是一个手写的枚举。

use std::fmt;

/// qppocr-core 的错误。
#[derive(Debug)]
pub enum Error {
    /// ONNX 解析失败（带字节偏移）。
    Parse(String),
    /// 模型文件读不了。
    Io(String),
    /// 图里有执行不了的东西（不支持的算子、缺输入……）。
    /// **不静默降级**：报错点名，不给"尽力而为"。
    Graph(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse(m) => write!(f, "onnx parse: {m}"),
            Error::Io(m) => write!(f, "io: {m}"),
            Error::Graph(m) => write!(f, "graph: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// 便捷结果别名。
pub type Result<T> = std::result::Result<T, Error>;
