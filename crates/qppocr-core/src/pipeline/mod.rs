//! det → cls → rec 流水线（设计文档 ）。
//!
//! 模块划分对应
//! - [`image`]：Image、缩放、旋转、自动色阶
//! - [`geometry`]：凸包/最小面积矩形/连通域/DB 后处理/框合并/杂波判定
//! - [`crop`]：透视裁剪、pack_crop、cls_view
//! - [`rec`]：CTC 解码、逐列墨迹、像素空格
//! - [`engine`]：OcrEngine 主流程（含区域重试）
//! - [`config`]：调参基准的 36 参数（数值照搬）

pub mod config;
pub mod crop;
pub mod engine;
pub mod geometry;
pub mod image;
pub mod rec;

pub use config::PipelineConfig;
pub use engine::{Dictionary, Engine, OcrResult, TextLine, Timings};
