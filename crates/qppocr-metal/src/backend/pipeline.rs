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
    // 同一份源码三个入口：`n_conv8`（f32 权重）、`n_conv8h`（fp16 权重）、
    // `n_conv8_edge`（边框补算，与 `n_conv8_mma3` 配套）。内核体只有一份
    // 模板，见该 .metal 的头注释。
    //
    // ⚠️ 三个入口共用一份源码 ⇒ **一个模板编译不过，三支一起消失**，整个
    // Metal 后端全灭。所以改 `conv8_body` 的模板签名后必须先跑
    // `cargo test -p qppocr-metal`（`KernelSet::new` 才会真的编译 MSL），
    // `cargo build` 是不看 .metal 的。
    ("n_conv8", include_str!("../../shaders/n_conv8.metal")),
    ("n_conv8h", include_str!("../../shaders/n_conv8.metal")),
    ("n_conv8_edge", include_str!("../../shaders/n_conv8.metal")),
    // 同一份源码两个入口：`n_conv8_mma`（64 通道块）与 `n_conv8_mma_n32`
    // （32 通道块，收 Co 是 32 倍数但不是 64 倍数的那批 1×1 conv）。与
    // `n_conv8_mma3` 的双实例化是同一套做法，只有 NCOLS/NSG 模板参数不同
    // ——见该 .metal 头注释「两档」一节。同样：一份源码两个入口，编译不过
    // 一起消失。
    ("n_conv8_mma", include_str!("../../shaders/n_conv8_mma.metal")),
    (
        "n_conv8_mma_n32",
        include_str!("../../shaders/n_conv8_mma.metal"),
    ),
    // 第三档 `<64,4>`：与 `n_conv8_mma` 同走 64 通道块，但线程组 128、
    // 每 simdgroup 瓦片 32×32（`acc[4][4]`，每 MMA 装载数 0.75 → 0.50）。
    // ⚠️ **已否证（0.33×，8 形状全负），计划层永不派发。** 注册在这里只是
    // 让 `microbench::conv8_mma_nsg4_scan` 有对象可量——**删掉它那条扫描
    // 与对拍会一起失效**。见该 .metal 头注释「第三档」一节。
    (
        "n_conv8_mma_sg4",
        include_str!("../../shaders/n_conv8_mma.metal"),
    ),
    // 同一份源码两个入口：`n_conv8_mma3`（64 通道块）与 `n_conv8_mma3_n32`
    // （32 通道块，收 Co 是 32 倍数但不是 64 倍数的那批 conv）。网格公式、
    // 参数块、覆盖矩形两条完全一致，只有 NCOLS 模板参数不同——见该 .metal
    // 头注释「32 通道块」一节。同样：一份源码两个入口，编译不过一起消失。
    ("n_conv8_mma3", include_str!("../../shaders/n_conv8_mma3.metal")),
    (
        "n_conv8_mma3_n32",
        include_str!("../../shaders/n_conv8_mma3.metal"),
    ),
    ("n_conv_dw", include_str!("../../shaders/n_conv_dw.metal")),
    // depthwise 的线程组瓦片暂存版（Metal 独有，探针 1.76~1.86×，只在
    // 9×9 一类的大 tap 上派发；3×3 实测更慢所以判据里排掉）。见该文件
    // 头注释与 `PlanBuilder::dw_tile`。
    (
        "n_conv_dw_tile",
        include_str!("../../shaders/n_conv_dw_tile.metal"),
    ),
    ("n_convt", include_str!("../../shaders/n_convt.metal")),
    // 2×2/s2 的空间分块版（探针 3.45×，见该文件头注释）。只在
    // `PlanBuilder::conv_t_tile` 为真且几何/并行度满足时派发。
    ("n_convt4w", include_str!("../../shaders/n_convt4w.metal")),
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
    // rec 头的融合出口（Metal 独有，Vulkan 侧仍走 n_softmax +
    // n_exit3_argmax）：一条内核对 logits 做完 softmax 与 argmax，逐位
    // 等于原来两条 dispatch 的组合。见该 .metal 头注释。
    (
        "n_head_argmax",
        include_str!("../../shaders/n_head_argmax.metal"),
    ),
    ("n_attn_mm", include_str!("../../shaders/n_attn_mm.metal")),
    (
        "n_transpose_nd",
        include_str!("../../shaders/n_transpose_nd.metal"),
    ),
];

/// 内核分发的线程组大小——与 GLSL 侧 `local_size_x = 256` 对齐。
/// Apple GPU 每线程组上限 ≥ 1024，256 恒安全。
const THREADS_PER_GROUP: u64 = 256;

/// 线程组尺寸的**按内核覆盖表**——只有下面列出的内核例外，其余恒
/// [`THREADS_PER_GROUP`]。
///
/// `n_conv8_mma3_n32` 用 **128 线程 / 4 个 simdgroup**：同一块 64×32 瓦片下，
/// 每 simdgroup 的瓦片从 16×16 抬到 32×16，每 MMA 的装载数 1.00 → 0.75。
/// 实测 1.26×（`microbench::conv8_mma3_load_attribution`），网格组数/覆盖矩形/
/// 尾块掩码/参数/暂存区尺寸一字不改。
///
/// ⚠️ **这张表必须与内核里的 `NSG` 模板实参一致**（`n_conv8_mma3.metal` 的
/// `mma3_body<NCOLS, NSG>`）：内核按 `tid>>5` 认 simdgroup，多给了线程就会有
/// 线程认领到不存在的 sg、去写 `ost` 的界外行——**静默算错**，不会报错。
/// 兜底是 `conv8_mma3_matches_cpu` 那条 CPU 对拍（它自带独立硬编码的覆盖几何）。
fn threads_per_group(name: &str) -> u64 {
    match name {
        // NCOLS=32 的两族都是 4 个 simdgroup = 128 线程（NCOLS=64 那族是
        // 默认的 256）；`n_conv8_mma_sg4` 是 **NCOLS=64 但 NSG=4** 的第三档，
        // 同为 128 线程（该档已被否证、计划层不派发，但扫描测试要按正确的
        // 线程数跑——写错了量的就是另一个程序）。**少一条这里就是静默算错**
        // ——见 `n_conv8_mma.metal` 模板上那条 NSG 耦合警告。
        "n_conv8_mma3_n32" | "n_conv8_mma_n32" | "n_conv8_mma_sg4" => 128,
        _ => THREADS_PER_GROUP,
    }
}

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

    /// 取某内核的管线（诊断/微基准要拿单条管线自己录分发时用）。
    pub(crate) fn get(&self, name: &str) -> Result<&ComputePipelineState> {
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
        dispatch_with_tg(
            enc,
            self.get(name)?,
            arena,
            p_off,
            groups,
            threads_per_group(name),
        )
    }
}

/// 录一条分发（管线已定）。**与 `KernelSet` 解耦**——微基准拿自己的
/// 变体管线复用同一套绑定/网格语义，量到的差异才只归因于内核本身。
pub(crate) fn dispatch_with(
    enc: &ComputeCommandEncoderRef,
    pipe: &ComputePipelineState,
    arena: &BufferRef,
    p_off: u32,
    groups: [u32; 3],
) -> Result<()> {
    dispatch_with_tg(enc, pipe, arena, p_off, groups, THREADS_PER_GROUP)
}

/// 同上，但线程组尺寸可指定。
///
/// 存在的唯一理由是**诊断**：生产路径恒用 [`THREADS_PER_GROUP`]，而
/// 「线程组尺寸是否影响达成吞吐」本身是 Mac 侧一个待证的性能假设
/// （见 `microbench::threads_per_group_sweep`）。把它做成参数而不是在
/// 探针里另抄一份编码，量到的差异才只归因于尺寸这一项。
pub(crate) fn dispatch_with_tg(
    enc: &ComputeCommandEncoderRef,
    pipe: &ComputePipelineState,
    arena: &BufferRef,
    p_off: u32,
    groups: [u32; 3],
    tg: u64,
) -> Result<()> {
    enc.set_compute_pipeline_state(pipe);
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
            groups[0] as u64 * tg,
            groups[1] as u64,
            groups[2] as u64,
        ),
        MTLSize::new(tg, 1, 1),
    );
    Ok(())
}
