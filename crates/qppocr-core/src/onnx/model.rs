//! 内存中的 ONNX 表示（`onnx_model.hpp` 的对应物）。
//!
//! 只留推理需要的：节点、属性、常量张量、图输入输出。字段号注释与
//! onnx.proto 对应，解析器（[`crate::onnx::parse``]）逐字段镜像
//! `onnx_parser.cpp`。

use std::collections::HashMap;

use crate::tensor::Tensor;

/// 节点属性（AttributeProto 的子集）。
#[derive(Clone, Debug, Default)]
pub struct Attribute {
    /// 属性名。
    pub name: String,
    /// 浮点值（field 2）。
    pub f: f32,
    /// 整数值（field 3）。
    pub i: i64,
    /// 字符串值（field 4）。
    pub s: String,
    /// 浮点列表（field 7）。
    pub floats: Vec<f32>,
    /// 整数列表（field 8）。
    pub ints: Vec<i64>,
    /// 张量值（field 5，Constant 的 value）。
    pub tensor: Option<Tensor>,
    /// 是否带 `f`。
    pub has_f: bool,
    /// 是否带 `i`。
    pub has_i: bool,
    /// 是否带 `s`。
    pub has_s: bool,
}

/// 计算节点（NodeProto 的子集）。
#[derive(Clone, Debug, Default)]
pub struct Node {
    /// 算子类型（如 `"Conv"`）。`FusedGelu` 是图优化产生的中间态。
    pub op_type: String,
    /// 节点名。
    pub name: String,
    /// 输入名列表（field 1）。
    pub inputs: Vec<String>,
    /// 输出名列表（field 2）。
    pub outputs: Vec<String>,
    /// 属性列表（field 5）。
    pub attrs: Vec<Attribute>,
}

impl Node {
    /// 按名取属性。
    pub fn attr(&self, n: &str) -> Option<&Attribute> {
        self.attrs.iter().find(|a| a.name == n)
    }
}

/// 图（GraphProto + ModelProto 的身份字段）。
#[derive(Clone, Debug, Default)]
pub struct Graph {
    /// 可执行节点（Constant 折叠、Identity 消除之后）。
    pub nodes: Vec<Node>,
    /// 常量权重（含 Constant 折叠进来的）。
    pub initializers: HashMap<String, Tensor>,
    /// 图输入名。
    pub inputs: Vec<String>,
    /// 图输出名。
    pub outputs: Vec<String>,
    /// ModelProto.metadata_props——rec 模型把字典放在这里。
    pub metadata: HashMap<String, String>,

    // ---- 模型身份，报告「实际加载了什么」用 ----
    /// 模型路径（或显示名）。
    pub model_path: String,
    /// GraphProto.name。
    pub graph_name: String,
    /// 生产者名。
    pub producer_name: String,
    /// 生产者版本。
    pub producer_version: String,
    /// 文档串（首行）。
    pub doc_string: String,
    /// IR 版本。
    pub ir_version: i64,
    /// opset 版本（ai.onnx 域）。
    pub opset_version: i64,
    /// 文件字节数。
    pub file_bytes: u64,
    /// 权重总元素数。
    pub param_count: i64,
    /// 优化后的节点数。
    pub node_count: i64,
    /// fuse_gelu 折掉的 GELU 子图数。
    pub fused_gelu: i64,
    /// fold_conv_bias 吸收的独立 bias Add 数。
    pub folded_bias: i64,
    /// drop_identity_pairs 删除的相互抵消 Transpose 数。
    pub dropped_identity: i64,
    /// fuse_conv_activation 融合的 conv+act 对数。
    pub fused_conv_act: i64,
}

impl Graph {
    /// 名字是否是图输出。
    pub fn is_graph_output(&self, nm: &str) -> bool {
        self.outputs.iter().any(|o| o == nm)
    }
    /// 「det」/「rec」/「cls」，从文件名猜（诊断用）。
    pub fn kind(&self) -> &'static str {
        let p = &self.model_path;
        if p.contains("_det_") || p.contains("det_infer") || p.contains("det.onnx") {
            "det"
        } else if p.contains("_rec_") || p.contains("rec_infer") || p.contains("rec.onnx") {
            "rec"
        } else if p.contains("_cls_") || p.contains("cls_infer") || p.contains("cls.onnx") {
            "cls"
        } else {
            "model"
        }
    }
}
