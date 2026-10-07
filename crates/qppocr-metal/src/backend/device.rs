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
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    record(enc);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    match cb.status() {
        MTLCommandBufferStatus::Completed => Ok(()),
        st => Err(Error::Device(format!(
            "Metal 命令缓冲未正常完成（状态 {st:?}）——设备丢失或被移除？"
        ))),
    }
}
