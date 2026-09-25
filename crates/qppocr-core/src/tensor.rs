//! 张量：连续、NCHW、行主序。
//!
//! 不做步长/视图系统——OCR 的图用不到，而步长系统会让每个内核都要处理
//! 非连续输入，得不偿失。有效载荷按 dtype 二选一，
//! 张量结构与图协议一一对应。

/// 数据类型。解析器在入口处归一化：I32/BOOL → [`DType::I64`]，
/// F64 → [`DType::F32`]（同款——内核只吃这两种）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DType {
    #[default]
    /// 32 位浮点（默认）。
    F32,
    /// 64 位整数（含 I32/BOOL）。
    I64,
}

/// 连续行主序张量。
///
/// F32 载荷走池化缓冲 [`F32Buf`]：图节点输出「分配→写满→读→释放」
/// 每节点一次，池省掉的是每次 fresh 块的 soft page fault（不是 memset）。
#[derive(Clone, Debug, Default)]
pub struct Tensor {
    /// 名字（图里的 tensor 名）。
    pub name: String,
    /// 形状。
    pub shape: Vec<i64>,
    /// 数据类型。
    pub dtype: DType,
    /// F32 载荷（`dtype == F32` 时有效；池化）。
    pub f32: qppocr_kernels::buf::F32Buf,
    /// I64 载荷（`dtype == I64` 时有效；shape 类小张量，不值得池化）。
    pub i64: Vec<i64>,
}

impl Tensor {
    /// 元素总数。
    pub fn numel(&self) -> i64 {
        self.shape.iter().product()
    }
    /// 维数。
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    /// 负索引安全的取维。
    pub fn shape_at(&self, i: isize) -> i64 {
        let r = self.shape.len() as isize;
        let i = if i < 0 { i + r } else { i };
        assert!((0..r).contains(&i), "shape index out of range");
        self.shape[i as usize]
    }
    /// f32 视图；dtype 不是 F32 时 panic（内核入口的显式契约）。
    pub fn as_f32(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32, "tensor {} is not F32", self.name);
        self.f32.as_slice()
    }
}
