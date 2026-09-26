//! qppocr 的算子内核。
//!
//! 这是整个 workspace 里**唯一允许 `unsafe` 的 crate**：
//! SIMD 内核靠裸指针 + 精确的寄存器分配拿性能，写成惯用的 slice 下标会引入
//! bounds check。且并行切分写的是不相交但无法用 `split_at_mut` 表达的区间
//! （同一些行的不同列段），只能用裸指针。边界全部集中在内核分发层，
//! 上层（`qppocr-core`）以 `#![forbid(unsafe_code)]` 强制走安全 API。
//!
//! 每个内核文件的结构（循环顺序、分块、寄存器分配意图、位级语义）
//! 有明确约定，注释里的实测结论一并携带。
//!
//! `scalar`（本 crate 的标量参考实现）不是备用方案，是判据：SIMD 版与它
//! 逐位比对。

#![deny(unsafe_op_in_unsafe_fn)]
// 内核保持下标循环与手写 clamp 的结构——f32::clamp 的 NaN 行为不同，
// 惯用改写会改变位级语义。见
// Rust 的写法」。
#![allow(
    clippy::needless_range_loop,
    clippy::manual_clamp,
    clippy::manual_div_ceil
)]

pub mod activation;
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

/// 强制后端（`None` = 自动探测）。
///
/// 供逐位对拍测试使用（scalar vs avx2）；生产代码不应调用——引擎构造时
/// 自动探测一次。全局、非线程局部：测试里串行设置。
static FORCED_BACKEND: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// 强制后续内核调用走指定后端；`None` 恢复自动探测。
pub fn force_backend(b: Option<Backend>) {
    let v = match b {
        None => 0,
        Some(Backend::Scalar) => 1,
        Some(Backend::Avx2) => 2,
    };
    FORCED_BACKEND.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// 当前生效后端（含强制覆盖）。
#[allow(dead_code)] // 诊断用；非 x86_64 目标上无调用方
pub(crate) fn current_backend() -> Backend {
    match FORCED_BACKEND.load(std::sync::atomic::Ordering::Relaxed) {
        1 => Backend::Scalar,
        2 => Backend::Avx2,
        _ => detect_backend(),
    }
}

/// AVX2 是否可用（含强制覆盖）。非 x86_64 恒 false（编译期折叠）。
#[cfg(target_arch = "x86_64")]
pub(crate) fn use_avx2() -> bool {
    current_backend() == Backend::Avx2
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)] // 分发宏的标量臂不调用它（编译期即知无 SIMD）
pub(crate) fn use_avx2() -> bool {
    false
}

/// SIMD/标量双臂分发：x86_64 上运行时探测，其他架构编译期取标量臂。
///
/// 曾经的写法 `#[cfg(target_arch = "x86_64")] if use_avx2() {vec} else
/// {scalar}` 把 **else 臂一起 cfg 掉了**——非 x86 目标上内核什么都不算
/// （能编译、输出错，CI 的 aarch64 -D warnings 才暴露出来）。这个宏保证
/// 标量臂在所有架构都编译且执行。
macro_rules! arch_dispatch {
    ($avx2_arm:expr, $scalar_arm:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if crate::use_avx2() {
                $avx2_arm
            } else {
                $scalar_arm
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            $scalar_arm
        }
    }};
}
pub(crate) use arch_dispatch;
