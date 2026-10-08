//! 命令提交：命令缓冲的录制、提交与等待。
//!
//! P1 的同步骨架——对应 Vulkan 侧的 `submit_one_shot`：录命令、提交、
//! 主机等待完成。P3 批间流水再上 `add_completed_handler`（异步收尾，
//! 对应 Vulkan 侧「只提交不等」的 submit_cb/wait_signal 对）。

use metal::MTLCommandBufferStatus;
use qppocr_core::error::{Error, Result};

/// 一次性提交：建命令缓冲 → 录 compute → 提交 → 等待完成。
///
/// `wait_until_completed` 返回后再读 Shared 缓冲即设备写毕的数据
/// （统一内存同址可见，无 sync/staging）。
pub(crate) fn submit_compute(
    queue: &metal::CommandQueueRef,
    record: impl FnOnce(&metal::ComputeCommandEncoderRef),
) -> Result<()> {
    submit_compute_timed(queue, record).map(|_| ())
}

/// 同 [`submit_compute`]，但另报 `(编码 ms, 等待 ms)`。
///
/// 主机侧摊薄（编码/屏障条数）与设备侧占用（算力/带宽/launch 间隔）
/// 是两条完全不同的优化路线，而「墙钟」把它们混成一个数。这两个相位
/// 在同一负载窗内同涨同落，但**比例**是刻度无关的读数：编码 ≫ 等待
/// ⇒ 该减 dispatch 条数；等待 ≫ 编码 ⇒ 该减设备侧工作量。
///
/// 每 forward 两次 `Instant::now()`，对 173 条 dispatch 的计划可忽略。
pub(crate) fn submit_compute_timed(
    queue: &metal::CommandQueueRef,
    record: impl FnOnce(&metal::ComputeCommandEncoderRef),
) -> Result<(f64, f64)> {
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let t0 = std::time::Instant::now();
    record(enc);
    enc.end_encoding();
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    cb.commit();
    let t1 = std::time::Instant::now();
    cb.wait_until_completed();
    let wait_ms = t1.elapsed().as_secs_f64() * 1000.0;
    match cb.status() {
        MTLCommandBufferStatus::Completed => Ok((encode_ms, wait_ms)),
        st => Err(Error::Device(format!(
            "Metal 命令缓冲未正常完成（状态 {st:?}）——设备丢失或被移除？"
        ))),
    }
}

