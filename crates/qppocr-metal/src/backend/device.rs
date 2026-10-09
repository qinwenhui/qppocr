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
    // 派发类型（诊断开关，`QPPOCR_METAL_CONCURRENT=1`）——**已实测证伪，留着
    // 是为了不再试第二遍**。
    //
    // 猜想：「`Serial` 下每条 dispatch 之后都完整排空+重启，而 rec 一图
    // 676 条 conv8（grid 只有 9~17 个线程组、单条几十 µs）会把排空开销摊到
    // 每条上，改用 `Concurrent` 放开重叠即可吃掉这笔」。§9.9 那条「NO_BARRIER
    // 只值 −0.20 ms」不能用来反驳它——serial 派发类型下屏障本就冗余，删掉当然
    // 没变化，真正没被否证的变量就是这里的派发类型，所以必须单独量。
    //
    // 2026-10-09 实测（`bench/gpuhealth/conc_ab.sh`，20 图 × 3 轮，奇偶轮换序，
    // 同一二进制只切环境变量）：中位配对差 **Δtotal = +0.90 ms（并发略慢）**，
    // 单向为正 8/10、负 2/10，`det` 列几乎不动（Δ +0.00）⇒ 效应真实且只落在
    // rec 上，不是噪声。数值先验：两版 OCR 输出 218 行逐字节相同，所以这不是
    // 「并发不安全」，而是**根本没得重叠**——相邻 dispatch 本来就互为读写依赖，
    // `Serial` 只是把依赖图已经强加的顺序写明了。
    //
    // 结论：派发类型**不是** rec conv8 慢于孤立微基准的自变量，别再往这条路上
    // 找。（往前数被否掉的还有：epilogue 融合、屏障、权重带宽、跨图 batching。）
    let enc = if std::env::var_os("QPPOCR_METAL_CONCURRENT").is_some() {
        cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent)
    } else {
        cb.new_compute_command_encoder()
    };
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

