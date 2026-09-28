//! qppocr 的算子内核。
//!
//! 这是整个 workspace 里**唯一允许 `unsafe` 的 crate**：
//! SIMD 内核靠裸指针 + 精确的寄存器分配拿性能，写成惯用的 slice 下标会引入
//! bounds check。且并行切分写的是不相交但无法用 `split_at_mut` 表达的区间
//! （同一些行的不同列段），只能用裸指针。边界全部集中在内核分发层，
//! 上层（`qppocr-core`）以 `#![forbid(unsafe_code)]` 强制走安全 API。
//!
//! 每个内核文件的结构（循环顺序、分块、寄存器分配意图、位级语义）
//! 有明确约定；注释里的性能结论来自本项目的参考环境，仅供参考。
//!
//! `scalar`（本 crate 的标量参考实现）不是备用方案，是判据：SIMD 版与它
//! 逐位比对。
//!
//! 架构接线只出现在 [`arch`]（分发层）与各后端文件（[`x86`]、
//! [`aarch64`]）；内核文件本身与架构无关。设备级执行（GPU）不属於
//! 本层——它的内存模型与调度粒度都不同，将来在引擎层接入。

#![deny(unsafe_op_in_unsafe_fn)]
// 内核保持下标循环与手写 clamp 的结构——f32::clamp 的 NaN 行为不同，
// 惯用改写会改变位级语义。
#![allow(
    clippy::needless_range_loop,
    clippy::manual_clamp,
    clippy::manual_div_ceil
)]

pub mod activation;
pub mod arch;
pub mod buf;
pub mod conv;
pub mod elementwise;
pub mod gemm;
pub mod par;
#[cfg(feature = "parallel")]
mod pool;
pub mod pool2d;
pub mod resize;
pub mod scalar;
pub mod shape;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

/// CPU 内核后端。运行时按 CPU 特性分派（x86 上 `is_x86_feature_detected!`，
/// aarch64 上 NEON 是基线指令集无需探测），编译期保证标量实现永远可用作
/// 兜底与判据。
///
/// 设备级后端（GPU）不在此枚举里——那是引擎层的概念，见 [`arch`] 的
/// 模块说明。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// 参考实现：正确性判据，任何平台可用。
    Scalar,
    /// x86-64 AVX2 + FMA。
    Avx2,
    /// aarch64 NEON + FMA（FMLA）。
    Neon,
}

/// 探测当前机器可用的最快 CPU 内核后端。
pub fn detect_backend() -> Backend {
    #[cfg(target_arch = "aarch64")]
    {
        // f32 NEON 是 aarch64 基线指令集（ARMv8-A 起强制），无需运行时探测。
        Backend::Neon
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return Backend::Avx2;
        }
        Backend::Scalar
    }
}

/// 强制后端（`None` = 自动探测）。
///
/// 供逐位对拍测试使用（scalar vs 本架构的 SIMD 后端）；生产代码不应调用
/// ——引擎构造时自动探测一次。全局、非线程局部：测试里串行设置。
static FORCED_BACKEND: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// 强制后续内核调用走指定后端；`None` 恢复自动探测。
pub fn force_backend(b: Option<Backend>) {
    let v = match b {
        None => 0,
        Some(Backend::Scalar) => 1,
        Some(Backend::Avx2) => 2,
        Some(Backend::Neon) => 3,
    };
    FORCED_BACKEND.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// 当前生效后端（含强制覆盖）。[`arch`] 分发层的唯一选择器。
#[inline]
pub(crate) fn current_backend() -> Backend {
    match FORCED_BACKEND.load(std::sync::atomic::Ordering::Relaxed) {
        1 => Backend::Scalar,
        2 => Backend::Avx2,
        3 => Backend::Neon,
        _ => detect_backend(),
    }
}
