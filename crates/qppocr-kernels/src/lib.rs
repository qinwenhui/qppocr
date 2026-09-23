//! qppocr 的算子内核。
//!
//! 这是整个 workspace 里**唯一允许 `unsafe` 的 crate**（DESIGN.md §2 铁律 3）：
//! SIMD 内核靠裸指针 + 精确的寄存器分配拿性能，写成惯用的 slice 下标会引入
//! bounds check。且并行切分写的是不相交但无法用 `split_at_mut` 表达的区间
//! （同一些行的不同列段），只能用裸指针。边界全部集中在内核分发层，
//! 上层（`qppocr-core`）以 `#![forbid(unsafe_code)]` 强制走安全 API。
//!
//! 每个内核文件对应 C++ 参考实现 `ops.cpp` 的一部分，**逐句照抄其结构**
//! （循环顺序、分块、寄存器分配意图、位级语义），注释里的实测结论一并携带。
//!
//! `scalar`（本 crate 的标量参考实现）不是备用方案，是判据：SIMD 版与它
//! 逐位比对（DESIGN.md §3.2、§8.3）。

#![deny(unsafe_op_in_unsafe_fn)]
// 内核逐句镜像 C++ ops.cpp 的循环结构（下标循环、手写 clamp），惯用 Rust
// 改写会改变结构甚至位级语义（f32::clamp 的 NaN 行为与 C++ 不同）——见
// DESIGN.md §3.2「不要顺手改成更 Rust 的写法」。
#![allow(
    clippy::needless_range_loop,
    clippy::manual_clamp,
    clippy::manual_div_ceil
)]

pub mod activation;
pub mod conv;
pub mod elementwise;
pub mod gemm;
pub mod par;
pub mod pool2d;
pub mod resize;
pub mod scalar;
pub mod shape;

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
