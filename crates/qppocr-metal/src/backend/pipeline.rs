//! compute 内核：MSL 源码内嵌 + 运行时编译 + 统一分发。
//!
//! **绑定模型（与 Vulkan 侧 pipeline.rs 一一对应）**：
//! - 整个会话**一个** `MTLBuffer`（arena）绑在 buffer 0/1/2——内核按
//!   指针类型取别名视图（f32 数据 / u32 参数块 / float4 向量化数据），
//!   Metal 天然允许同一缓冲多绑定，GLSL 侧的「三 binding 别名」无需
//!   描述符就免费得到；
//! - 参数块经 buffer 1（u32 视图）+ `p_off` 偏移间接寻址（`set_bytes`
//!   注入，push constant 的对应物）；
//! - MSL 源码运行时编译（`newLibraryWithSource`）：装载期一次编译、
//!   全程复用，无需离线 shader 编译器与 SPIR-V 产物链。
//!
//! **dispatch 网格语义**：计划模型给的是工作组数（GLSL 的
//! local_size_x=256 对应 Metal 的 256 线程/线程组）；网格 =
//! (groups.x*256, groups.y, groups.z)，内核按 `thread_position_in_grid`
//! /`threadgroup_position_in_grid` 取与 GLSL 全局 invocation 逐位一致
//! 的坐标。内核自带越界早退，非整组余量安全。
//!
//! 精确数学：编译选项显式关 fast-math——保住与 CPU/Vulkan 侧逐位
//! 可比的运算语义（IEEE 基本运算不被近似/重排；GLSL 的 `precise` 关键
//! 字由本开关承担）。

use metal::{BufferRef, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use qppocr_core::error::{Error, Result};
use std::collections::HashMap;

/// 全部内核：`(名, MSL 源码)`。include_str! 编进二进制。清单与
/// qppocr-gpu 的 n_ 内核族一一对应（计划模型的 kernel 名在此全部可查）。
const SHADERS: &[(&str, &str)] = &[
    ("n_entry", include_str!("../../shaders/n_entry.metal")),
    ("n_elem", include_str!("../../shaders/n_elem.metal")),
    ("n_conv", include_str!("../../shaders/n_conv.metal")),
    ("n_conv8", include_str!("../../shaders/n_conv8.metal")),
    ("n_conv_dw", include_str!("../../shaders/n_conv_dw.metal")),
    ("n_convt", include_str!("../../shaders/n_convt.metal")),
    ("n_pool", include_str!("../../shaders/n_pool.metal")),
    ("n_resize", include_str!("../../shaders/n_resize.metal")),
    ("n_concat_c", include_str!("../../shaders/n_concat_c.metal")),
    ("n_channel", include_str!("../../shaders/n_channel.metal")),
    (
        "n_reduce_hw",
        include_str!("../../shaders/n_reduce_hw.metal"),
    ),
    (
        "n_reduce_fin",
        include_str!("../../shaders/n_reduce_fin.metal"),
    ),
    (
        "n_reduce_last",
        include_str!("../../shaders/n_reduce_last.metal"),
    ),
    ("n_exit", include_str!("../../shaders/n_exit.metal")),
    ("n_exit3", include_str!("../../shaders/n_exit3.metal")),
    (
        "n_exit3_argmax",
        include_str!("../../shaders/n_exit3_argmax.metal"),
    ),
    ("n_softmax", include_str!("../../shaders/n_softmax.metal")),
    ("n_attn_mm", include_str!("../../shaders/n_attn_mm.metal")),
    (
        "n_transpose_nd",
        include_str!("../../shaders/n_transpose_nd.metal"),
    ),
];

/// 内核分发的线程组大小——与 GLSL 侧 `local_size_x = 256` 对齐。
/// Apple GPU 每线程组上限 ≥ 1024，256 恒安全。
const THREADS_PER_GROUP: u64 = 256;

/// 一套内核：名 → compute pipeline state（装载期编译一次，会话级
/// 共享——set_buffer 编码期绑 arena，与计划无关）。
pub(crate) struct KernelSet {
    pipes: HashMap<&'static str, ComputePipelineState>,
}

impl KernelSet {
    /// 编译全部 MSL 源码并建管线。
    ///
    /// 每个源文件独立成库（`newLibraryWithSource` 单源单函数）——
    /// 编译失败时错误信息带内核名，可直接定位。
    pub(crate) fn new(device: &metal::DeviceRef) -> Result<Self> {
        let opts = metal::CompileOptions::new();
        // 精确数学：MTLCompileOptions 的 fastMathEnabled 默认即关，显式
        // 设置是给读代码的人看的——这条开关决定逐位对拍是否成立。
        opts.set_fast_math_enabled(false);
        let mut pipes = HashMap::new();
        for (name, src) in SHADERS {
            let lib = device
                .new_library_with_source(src, &opts)
                .map_err(|e| Error::Device(format!("MSL 编译失败（{name}）: {e}")))?;
            let func = lib
                .get_function(name, None)
                .map_err(|e| Error::Device(format!("取内核函数 {name} 失败: {e}")))?;
            let pipe = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| Error::Device(format!("建 compute 管线（{name}）失败: {e}")))?;
            pipes.insert(*name, pipe);
        }
        Ok(Self { pipes })
    }

    fn get(&self, name: &str) -> Result<&ComputePipelineState> {
        self.pipes
            .get(name)
            .ok_or_else(|| Error::Device(format!("内核 {name} 不在 KernelSet 里")))
    }

    /// 录一条内核分发：管线 + arena 三视图绑定 + `p_off` 注入 + dispatch。
    ///
    /// `groups` 是**工作组数**（计划模型的网格）；调用方随后的
    /// `memory_barrier_with_resources` 建立与下一 dispatch 的写读依赖。
    pub(crate) fn dispatch(
        &self,
        enc: &ComputeCommandEncoderRef,
        name: &str,
        arena: &BufferRef,
        p_off: u32,
        groups: [u32; 3],
    ) -> Result<()> {
        enc.set_compute_pipeline_state(self.get(name)?);
        // 同一 arena 绑 0/1/2：f32 / u32 / float4 三视图（各内核按需
        // 声明，编号沿用 GLSL 侧 binding 编号）。
        enc.set_buffer(0, Some(arena), 0);
        enc.set_buffer(1, Some(arena), 0);
        enc.set_buffer(2, Some(arena), 0);
        // push constant 的对应物：4 字节参数块偏移（set_bytes 在本调用
        // 期间同步拷贝，无生命周期尾巴）
        enc.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &p_off as *const u32 as *const std::ffi::c_void,
        );
        enc.dispatch_threads(
            MTLSize::new(
                groups[0] as u64 * THREADS_PER_GROUP,
                groups[1] as u64,
                groups[2] as u64,
            ),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
        Ok(())
    }
}
