//! qppocr 的算子内核。
//!
//! 这是整个 workspace 里**唯一允许 `unsafe` 的 crate**（DESIGN.md §2 铁律 3）：
//! SIMD 内核靠裸指针 + 精确的寄存器分配拿性能，写成惯用的 slice 下标会引入
//! bounds check。边界全部集中在 `x86` 等后端模块里，上层（`qppocr-core`）
//! 以 `#![forbid(unsafe_code)]` 强制走安全 API。
//!
//! 每个 SIMD 内核都有对应的标量版本（[`scalar`]），它不是备用方案，是判据：
//! 单元测试逐位（或按明确容差）比对两版输出。

#![deny(unsafe_op_in_unsafe_fn)]

pub mod scalar;

/// 后端选择。运行时按 CPU 特性分派（`is_x86_feature_detected!`），
/// 编译期保证标量实现永远可用作兜底与判据。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// 参考实现：正确性判据，任何平台可用。
    Scalar,
    /// x86-64 AVX2 + FMA。
    Avx2,
}

/// 探测当前机器可用的最快后端。
pub fn detect_backend() -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return Backend::Avx2;
        }
    }
    Backend::Scalar
}
