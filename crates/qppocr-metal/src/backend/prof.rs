//! 逐内核 GPU 计时（**重复外推法**）。
//!
//! ## 为什么不是别的
//!
//! - **`MTLCounterSampleBuffer`**：官方口径，但本机（Apple M4，AGXG16G）
//!   实测只支持 `AtStageBoundary` 一个采样点——计算编码器与 blit 编码器上
//!   的 `sampleCountersInBuffer:atSampleIndex:withBarrier:` 都直接触发
//!   `not supported on this device` 断言（`supportsCounterSampling` 对
//!   `AtDraw`/`AtDispatch` 均返回 false）。此路不通。
//! - **每条 dispatch 一个命令缓冲**：量到的是命令缓冲之间的调度空隙，
//!   不是算力（实测 104 条逐条提交合计 32 ms，而同一计划单缓冲真实墙钟
//!   4.5 ms）。
//! - **前缀差分**（同一缓冲里跑前 i 条，相邻前缀墙钟相减）：差值被两次
//!   独立提交之间的调度抖动淹没（同一条内核两次剖析给出 3.20 ms / 0.08 ms）。
//!
//! ## 这一版
//!
//! 对第 i 条 dispatch：在**一条命令缓冲内**把它重复 `R` 次（条间插屏障），
//! 量总墙钟 `T(R)`。同一条内核在两个不同的 `R` 上各量一次，斜率
//! `(T(R₂) − T(R₁)) / (R₂ − R₁)` 就是单次的真实执行时间——命令缓冲的固定
//! 开销（创建、提交、等待、首尾空隙）对整个差值项**同时出现**，被斜率
//! 消掉。
//!
//! `R` 自适应：从小 `R` 起步，若差值太小（信号被抖动淹没）就翻倍重测，
//! 直到 `T(R) − T(1)` 超过噪声地板或撞上上限。这样小内核多跑几轮、大内核
//! 少跑几轮，总剖析成本可控。
//!
//! 代价：重复执行让输入留在 L2（真实执行里上一级内核刚写过、大概率也在
//! L2），**偏乐观但同量级**。用它排名次、定比例是对的，别拿它当端到端墙钟。

use std::time::Instant;

use qppocr_core::error::{Error, Result};

use super::device::submit_compute;
use super::pipeline::KernelSet;
use super::session::MetalPlan;

/// 一条 dispatch 的设备端耗时（毫秒）与身份。
#[derive(Clone)]
pub(crate) struct KernelTime {
    /// 内核名（`n_conv` / `n_elem` / …）。
    pub kernel: &'static str,
    /// 图里的节点序号（定位是哪一层）。
    pub node_idx: usize,
    /// 图里的节点名（`Conv.78` 之类）。**与 ONNX 直接对得上**——只有序号时
    /// 无法把「哪条 dispatch 贵」与模型的算子、形状对上，`node#160` 这种
    /// 编号要另开一次建图调试才查得到，等于查不动。
    pub node: String,
    /// 网格（组数）：[x, y, z]。
    pub groups: [u32; 3],
    /// 单次执行时间（毫秒，斜率口径）。
    pub ms: f64,
    /// 为了得到 `ms` 实际用的重复次数（诊断用：越大说明信号越弱）。
    pub reps: u32,
}

/// 噪声地板：`T(R) − T(1)` 至少要有这么多毫秒，斜率才可信。
const NOISE_FLOOR_MS: f64 = 1.0;
/// 重复次数的起点与上限（每轮翻倍）。
const REP_START: u32 = 4;
const REP_MAX: u32 = 256;

/// 重复 `reps` 次执行第 `idx` 条 dispatch，返回墙钟（毫秒）。
fn replay_one(
    queue: &metal::CommandQueueRef,
    ks: &KernelSet,
    plan: &MetalPlan,
    idx: usize,
    reps: u32,
) -> Result<f64> {
    let d = &plan.dispatches[idx];
    let t = Instant::now();
    submit_compute(queue, |enc| {
        for _ in 0..reps {
            ks.dispatch(enc, d.kernel, &plan.buf, d.p_off, d.groups)
                .expect("dispatch（内核名来自计划，必在套件内）");
            let res: [&metal::ResourceRef; 1] = [&plan.buf];
            enc.memory_barrier_with_resources(&res);
        }
    })?;
    Ok(t.elapsed().as_secs_f64() * 1000.0)
}

/// 测第 `idx` 条 dispatch 的单次执行时间：先跑一次作基准，再逐级加倍
/// 重复次数，直到差值高出噪声地板。
///
/// 返回 `(单次毫秒, 最终重复次数, 基准毫秒)`。
fn time_one(
    queue: &metal::CommandQueueRef,
    ks: &KernelSet,
    plan: &MetalPlan,
    idx: usize,
) -> Result<(f64, u32, f64)> {
    let t1 = replay_one(queue, ks, plan, idx, 1)?;
    let mut r = REP_START;
    loop {
        let t_r = replay_one(queue, ks, plan, idx, r)?;
        let delta = t_r - t1;
        if delta >= NOISE_FLOOR_MS || r >= REP_MAX {
            // 斜率：delta / (r − 1)。r = 1 的基准里也已含一次真实执行。
            let per = delta / (r as f64 - 1.0);
            return Ok((per, r, t1));
        }
        r = (r * 2).min(REP_MAX);
    }
}

/// 逐条剖析整个计划。
///
/// `err` 只在设备级失败时返回；单条内核算不出可信斜率时按 0 记并标记
/// `reps`（打印时会体现），不中断整轮剖析。
pub(crate) fn sample_kernels(
    queue: &metal::CommandQueueRef,
    ks: &KernelSet,
    plan: &MetalPlan,
) -> Result<Vec<KernelTime>> {
    let n = plan.dispatches.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if plan.buf.length() == 0 {
        return Err(Error::Device("计划 arena 为空，无法剖析".into()));
    }
    let mut out = Vec::with_capacity(n);
    for idx in 0..n {
        let d = &plan.dispatches[idx];
        let (ms, reps, base) = time_one(queue, ks, plan, idx)?;
        let ms = if ms.is_finite() && ms > 0.0 { ms } else { 0.0 };
        let _ = base;
        out.push(KernelTime {
            kernel: d.kernel,
            node_idx: d.node_idx,
            node: d.node.clone(),
            groups: d.groups,
            ms,
            reps,
        });
    }
    Ok(out)
}
