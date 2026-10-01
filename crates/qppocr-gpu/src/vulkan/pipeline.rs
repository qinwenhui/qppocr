//! compute 管线：单 SSBO + push constant 的绑定模型。
//!
//! **架构决策（对着「每帧一次提交」打）**：
//! - 整个会话**一个** storage buffer（arena 的块）+ **一个**描述符集——
//!   内核经 push constant 里的 float 元素偏移寻址，零描述符churn；
//! - 管线按（内核名 × 常量形状）静态装载；命令缓冲**录制一次、反复
//!   提交**（SIMULTANEOUS_USE）——每帧的 CPU 成本 = 写输入 + submit
//!   + 等信号，没有逐帧重录。
//!
//! SPIR-V 签入（`shaders/spirv/`，`tools/shader-build` 编译，改源后
//! 必须重跑并提交产物）。

use ash::Device;
use ash::vk;
use qppocr_core::error::{Error, Result};
use std::collections::HashMap;

/// 全部内核：`(名, SPIR-V 字节)`。include_bytes! 编进二进制。
const SHADERS: &[(&str, &[u8])] = &[
    ("add", include_bytes!("../../shaders/spirv/add.spv")),
    ("add_c", include_bytes!("../../shaders/spirv/add_c.spv")),
    ("clip", include_bytes!("../../shaders/spirv/clip.spv")),
    (
        "concat_c",
        include_bytes!("../../shaders/spirv/concat_c.spv"),
    ),
    ("conv", include_bytes!("../../shaders/spirv/conv.spv")),
    (
        "conv_gemm_nhwc",
        include_bytes!("../../shaders/spirv/conv_gemm_nhwc.spv"),
    ),
    (
        "conv_gemm_f16",
        include_bytes!("../../shaders/spirv/conv_gemm_f16.spv"),
    ),
    (
        "from_f16",
        include_bytes!("../../shaders/spirv/from_f16.spv"),
    ),
    ("conv_dw", include_bytes!("../../shaders/spirv/conv_dw.spv")),
    (
        "conv_gemm",
        include_bytes!("../../shaders/spirv/conv_gemm.spv"),
    ),
    (
        "conv_k_gemm",
        include_bytes!("../../shaders/spirv/conv_k_gemm.spv"),
    ),
    (
        "convtranspose",
        include_bytes!("../../shaders/spirv/convtranspose.spv"),
    ),
    (
        "fused_hardsigmoid_mul",
        include_bytes!("../../shaders/spirv/fused_hardsigmoid_mul.spv"),
    ),
    (
        "fused_sigmoid_mul",
        include_bytes!("../../shaders/spirv/fused_sigmoid_mul.spv"),
    ),
    (
        "hardsigmoid",
        include_bytes!("../../shaders/spirv/hardsigmoid.spv"),
    ),
    ("mul", include_bytes!("../../shaders/spirv/mul.spv")),
    ("mul_c", include_bytes!("../../shaders/spirv/mul_c.spv")),
    (
        "muladd_scale",
        include_bytes!("../../shaders/spirv/muladd_scale.spv"),
    ),
    ("pool", include_bytes!("../../shaders/spirv/pool.spv")),
    (
        "reduce_hw",
        include_bytes!("../../shaders/spirv/reduce_hw.spv"),
    ),
    ("relu", include_bytes!("../../shaders/spirv/relu.spv")),
    (
        "resize_nearest",
        include_bytes!("../../shaders/spirv/resize_nearest.spv"),
    ),
    ("to_f16", include_bytes!("../../shaders/spirv/to_f16.spv")),
    (
        "nchw_to_nhwc",
        include_bytes!("../../shaders/spirv/nchw_to_nhwc.spv"),
    ),
    (
        "nhwc_to_nchw",
        include_bytes!("../../shaders/spirv/nhwc_to_nchw.spv"),
    ),
    ("sigmoid", include_bytes!("../../shaders/spirv/sigmoid.spv")),
    // ---- NHWC-f16 内核族（n_ 前缀；详见各 .comp 头注释）----
    (
        "n_channel",
        include_bytes!("../../shaders/spirv/n_channel.spv"),
    ),
    (
        "n_concat_c",
        include_bytes!("../../shaders/spirv/n_concat_c.spv"),
    ),
    ("n_conv", include_bytes!("../../shaders/spirv/n_conv.spv")),
    ("n_conv8", include_bytes!("../../shaders/spirv/n_conv8.spv")),
    (
        "n_conv_dw",
        include_bytes!("../../shaders/spirv/n_conv_dw.spv"),
    ),
    ("n_convt", include_bytes!("../../shaders/spirv/n_convt.spv")),
    ("n_elem", include_bytes!("../../shaders/spirv/n_elem.spv")),
    ("n_entry", include_bytes!("../../shaders/spirv/n_entry.spv")),
    ("n_exit", include_bytes!("../../shaders/spirv/n_exit.spv")),
    ("n_pool", include_bytes!("../../shaders/spirv/n_pool.spv")),
    (
        "n_reduce_hw",
        include_bytes!("../../shaders/spirv/n_reduce_hw.spv"),
    ),
    (
        "n_reduce_fin",
        include_bytes!("../../shaders/spirv/n_reduce_fin.spv"),
    ),
    (
        "n_resize",
        include_bytes!("../../shaders/spirv/n_resize.spv"),
    ),
    (
        "n_transpose",
        include_bytes!("../../shaders/spirv/n_transpose.spv"),
    ),
    (
        "n_softmax",
        include_bytes!("../../shaders/spirv/n_softmax.spv"),
    ),
    ("n_exit3", include_bytes!("../../shaders/spirv/n_exit3.spv")),
    (
        "n_exit3_argmax",
        include_bytes!("../../shaders/spirv/n_exit3_argmax.spv"),
    ),
    (
        "n_reduce_last",
        include_bytes!("../../shaders/spirv/n_reduce_last.spv"),
    ),
    (
        "n_transpose_nd",
        include_bytes!("../../shaders/spirv/n_transpose_nd.spv"),
    ),
    (
        "n_attn_mm",
        include_bytes!("../../shaders/spirv/n_attn_mm.spv"),
    ),
];

/// push constant 块（与各 .comp 的 PC 布局一一对应；float 元素偏移）。
#[repr(C)]
pub(crate) struct PcUnary {
    pub in_off: u32,
    pub out_off: u32,
    pub n: u32,
}

/// hardsigmoid（alpha/beta）/ clip（lo/hi）。
#[repr(C)]
pub(crate) struct PcUnaryF {
    pub in_off: u32,
    pub out_off: u32,
    pub n: u32,
    pub p1: f32,
    pub p2: f32,
}

/// 同形二元。
#[repr(C)]
pub(crate) struct PcBinary {
    pub a_off: u32,
    pub b_off: u32,
    pub out_off: u32,
    pub n: u32,
}

/// 逐通道族（mul_c / fused_*）：A[N,C,HW]，b[C]。
#[repr(C)]
pub(crate) struct PcChannel {
    pub a_off: u32,
    pub b_off: u32,
    pub out_off: u32,
    pub hw: u32,
    pub c: u32,
}

/// muladd_scale。
#[repr(C)]
pub(crate) struct PcMulAddScale {
    pub f_off: u32,
    pub gate_off: u32,
    pub r_off: u32,
    pub out_off: u32,
    pub hw: u32,
    pub c: u32,
}

/// 布局转换内核的 PC：直接装参数（不经参数块间接层）。
#[repr(C)]
pub(crate) struct PcLayout {
    pub in_off: u32,
    pub out_off: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
}

/// 参数块内核（conv/pool/resize/convT/reduce/concat）：全部形状参数
/// 放 arena（128 B 的 push constant 装不下），PC 只带参数块偏移。
#[repr(C)]
pub(crate) struct PcParams {
    /// 参数块在 SSBO 里的 **u32 下标**（binding 1 的 uint 视图同源）。
    pub p_off: u32,
}

/// fused_hardsigmoid_mul。
#[repr(C)]
pub(crate) struct PcChannelF {
    pub a_off: u32,
    pub g_off: u32,
    pub out_off: u32,
    pub hw: u32,
    pub c: u32,
    pub alpha: f32,
    pub beta: f32,
}

/// PC 结构的裸字节视图（与 .comp 的 push constant 块逐字段对应）。
macro_rules! pc_bytes {
    ($($t:ty),* $(,)?) => {$(
        impl $t {
            pub(crate) fn bytes(&self) -> &[u8] {
                // SAFETY: repr(C) 定长 POD 的连续字节读取，无内部指针。
                unsafe {
                    std::slice::from_raw_parts(self as *const Self as *const u8, size_of::<Self>())
                }
            }
        }
    )*};
}

pc_bytes!(
    PcLayout,
    PcUnary,
    PcUnaryF,
    PcBinary,
    PcChannel,
    PcMulAddScale,
    PcChannelF,
    PcParams
);

/// 参数块写入器：形状参数太多（conv 20+ 项）装不进 128 B 的 push
/// constant，写进 arena 一小块，内核经 binding 1（同一缓冲的 u32
/// 视图）读取。装载期写一次、随 CB 复用——零每帧成本。
pub(crate) struct ParamBlock {
    words: Vec<u32>,
}

/// 「无此输入」的哨兵偏移（conv 的 bias/residual 等）。
pub(crate) const OFF_NONE: u32 = u32::MAX;

impl ParamBlock {
    pub(crate) fn new() -> Self {
        Self { words: Vec::new() }
    }

    pub(crate) fn u(&mut self, v: u32) -> &mut Self {
        self.words.push(v);
        self
    }

    /// 字数（布局分配用）。
    pub(crate) fn len_words(&self) -> u32 {
        self.words.len() as u32
    }

    /// 只读视图（session 直接写进整块布局）。
    pub(crate) fn words(&self) -> &[u32] {
        &self.words
    }

    /// f32 按位写入（内核侧 uintBitsToFloat 读回）。
    pub(crate) fn f(&mut self, v: f32) -> &mut Self {
        self.words.push(v.to_bits());
        self
    }

    /// 写入 arena 区域（u32 视图），返回 push constant。
    /// `region` 必须按 words.len()*4 分配（调用方保证）。
    pub(crate) fn finish(self, region: &super::memory::Region) -> PcParams {
        // SAFETY: 持久映射可写内存；写入长度 ≤ 分配长度，无越界。
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.words.as_ptr(),
                region.ptr as *mut u32,
                self.words.len(),
            );
        }
        PcParams {
            p_off: region.offset as u32 / 4,
        }
    }
}

/// 一套内核：共享的管线布局/描述符集 + 名字 → 管线。
pub(crate) struct KernelSet {
    layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    pipes: HashMap<&'static str, vk::Pipeline>,
    device: Device,
}

impl KernelSet {
    /// 无缓存版（一次性测试用）；会话内请走 [`Self::new_with_cache`]。
    pub(crate) fn new(device: &Device, buffer: vk::Buffer, size: vk::DeviceSize) -> Result<Self> {
        Self::new_with_cache(device, buffer, size, vk::PipelineCache::null())
    }

    /// 建：单 SSBO 绑定（绑定 arena 的块缓冲）+ push constant 区
    /// （compute，0..128）+ 全部管线。
    ///
    /// `cache`：会话级管线缓存——同会话按形状重建计划时驱动侧复用
    /// 编译产物（重建从 ~5-9ms 掉到 ~1-2ms；形状多样性 × LRU 驱逐会让
    /// 每帧都重建，实测曾吃掉 det 的全部 GPU 收益）。
    pub(crate) fn new_with_cache(
        device: &Device,
        buffer: vk::Buffer,
        size: vk::DeviceSize,
        cache: vk::PipelineCache,
    ) -> Result<Self> {
        // 三视图绑定：binding 0 = f32[]（计算数据）、binding 1 = u32[]
        //（参数块）、binding 2 = f16vec4[]（NHWC-f16 内核族 + coopmat 装载）
        //——同一 VkBuffer 的别名视图，GLSL 侧按需声明。
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: dsl_ci 合法；布局只被本 KernelSet 使用。
        let dsl = unsafe { device.create_descriptor_set_layout(&dsl_ci, None) }
            .map_err(|e| Error::Device(format!("建 descriptor layout 失败: {e}")))?;
        let pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(128);
        let set_layouts = [dsl];
        let pc_ranges = [pc_range];
        let pl_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&pc_ranges);
        // SAFETY: pl_ci 引用的 dsl 刚创建且存活。
        let layout = unsafe { device.create_pipeline_layout(&pl_ci, None) }
            .map_err(|e| Error::Device(format!("建 pipeline layout 失败: {e}")))?;

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3 * SHADERS.len() as u32)];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(SHADERS.len() as u32);
        // SAFETY: pool_ci 合法。
        let pool = unsafe { device.create_descriptor_pool(&pool_ci, None) }
            .map_err(|e| Error::Device(format!("建 descriptor pool 失败: {e}")))?;
        let set_layouts = [dsl];
        let ds_ci = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&set_layouts);
        // SAFETY: pool/layout 存活。
        let set = unsafe { device.allocate_descriptor_sets(&ds_ci) }
            .map_err(|e| Error::Device(format!("分配 descriptor set 失败: {e}")))?[0];
        let buf_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(size);
        // 三个绑定指向同一缓冲（f32 / u32 / f16vec4 视图）
        let buf_infos = [buf_info, buf_info, buf_info];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buf_infos[..1]),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buf_infos[1..2]),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buf_infos[2..]),
        ];
        // SAFETY: set/buffer 存活；写入结构在调用期间有效。
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        // 全部管线
        let entry = c"main";
        let mut pipes = HashMap::with_capacity(SHADERS.len());
        for &(name, spv) in SHADERS {
            let module = {
                let code: Vec<u32> = spv
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let sm_ci = vk::ShaderModuleCreateInfo::default().code(&code);
                // SAFETY: SPIR-V 来自 naga 产物（4 字节对齐的 LE u32）。
                unsafe { device.create_shader_module(&sm_ci, None) }
                    .map_err(|e| Error::Device(format!("{name}: 建 shader module 失败: {e}")))?
            };
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry);
            let cp_ci = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            // SAFETY: stage 引用的 module 存活；layout 存活。
            // （ash 0.38 返回 Result<Vec, (Vec, Result)>——部分失败也带产物，我们只要整体失败码。）
            let made = unsafe { device.create_compute_pipelines(cache, &[cp_ci], None) }
                .map_err(|(_, e)| Error::Device(format!("{name}: 建管线失败: {e}")))?;
            let pipe = made[0];
            // SAFETY: module 可在管线创建后释放（管线持有其引用）。
            unsafe { device.destroy_shader_module(module, None) };
            pipes.insert(name, pipe);
        }
        // SAFETY: dsl 的职责在 layout/set 建完后结束。
        unsafe { device.destroy_descriptor_set_layout(dsl, None) };
        Ok(Self {
            layout,
            pool,
            set,
            pipes,
            device: device.clone(),
        })
    }

    pub(crate) fn pipeline(&self, name: &str) -> vk::Pipeline {
        *self
            .pipes
            .get(name)
            .unwrap_or_else(|| panic!("内核不存在: {name}"))
    }

    /// 描述符集（单 SSBO 双视图）——录制时必须显式绑定。
    pub(crate) fn set(&self) -> vk::DescriptorSet {
        self.set
    }

    pub(crate) fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl Drop for KernelSet {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: 句柄由本结构独占；销毁时设备上无未完成工作
            //（提交侧已等信号）；无自定义分配器。
            for p in self.pipes.values() {
                self.device.destroy_pipeline(*p, None);
            }
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// 录制一次 dispatch（bind + push constants + dispatch）。
///
/// # Safety
/// `cb` 必须处于录制态（begin 之后、end 之前）；`pc` 的布局必须与
/// `name` 内核的 push constant 块一致（由调用方保证——Phase G 的
/// 计划生成器按内核名选 PC 结构，不经过自由组合）。
pub(crate) unsafe fn record_dispatch(
    device: &Device,
    cb: vk::CommandBuffer,
    ks: &KernelSet,
    name: &str,
    pc: &[u8],
    groups: [u32; 3],
) {
    unsafe {
        // SAFETY: 前置条件见本函数的 # Safety 文档。
        // 先录内存/执行屏障：同一 CB 里前一个 dispatch 的 SSBO 写对
        // 本 dispatch 可见（规范要求；驱动短序列可能容忍，长图必炸——
        // 实测 140-dispatch 无屏障时结果错乱、有屏障后逐位恢复）。
        let barriers = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &barriers,
            &[],
            &[],
        );
        let sets = [ks.set()];
        device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            ks.layout(),
            0,
            &sets,
            &[],
        );
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, ks.pipeline(name));
        device.cmd_push_constants(cb, ks.layout(), vk::ShaderStageFlags::COMPUTE, 0, pc);
        device.cmd_dispatch(cb, groups[0], groups[1], groups[2]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::VulkanContext;
    use crate::vulkan::memory::{Arena, Region};
    use qppocr_kernels::activation::{hardsigmoid, sigmoid_tensor};

    /// 确定性 LCG（不引 rand）：输出 [-8, 8)。
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 16.0 - 8.0
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// 十个内核一条命令缓冲、一次提交；逐内核与 CPU 判据容差对拍
    ///（容差口径：sigmoid 走 GPU exp vs CPU 多项式 exp ~1e-6 级；
    /// muladd_scale 的 CPU 刻意两次舍入而 GPU 可能 fma，≤1 ulp；
    /// 纯比较/算术内核应逐位一致）。另测已录制 CB 的重复提交开销——
    /// 这是 a≈5ms 目标的载体。
    #[test]
    fn elementwise_vs_cpu() {
        let ctx = match VulkanContext::open(None) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[gpu] 跳过（无满足基线的设备）: {e}");
                return;
            }
        };
        let dev = ctx.inner.device.raw().clone();
        let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);

        // 区域总量必须落在一块内（KernelSet 绑单缓冲——Phase F 的
        // 已知边界；Phase G 的图计划按总量一次开块）：14 区域 × 1MB。
        const N: usize = 1 << 18; // 256K 元素
        const C: usize = 256; // 通道族：N = C*HW
        const HW: usize = N / C;
        let a = arena.alloc((N * 4) as vk::DeviceSize).unwrap();
        let b = arena.alloc((N * 4) as vk::DeviceSize).unwrap();
        let gate = arena.alloc((C * 4) as vk::DeviceSize).unwrap();
        let r = arena.alloc((N * 4) as vk::DeviceSize).unwrap();
        let mut outs: Vec<Region> = Vec::new();
        for _ in 0..10 {
            outs.push(arena.alloc((N * 4) as vk::DeviceSize).unwrap());
        }
        let (buf, buf_size) = arena.chunk_range().unwrap();
        // 单块断言：多块时下面所有 offset 都指向没绑定的缓冲（曾因此
        // 全量读垃圾——max|diff|≈1.0）。
        for reg in [&a, &b, &r, &gate] {
            assert_eq!(reg.buffer, buf, "区域跨块：KernelSet 单缓冲绑定被突破");
        }
        for reg in &outs {
            assert_eq!(reg.buffer, buf);
        }

        let mut seed = 0x853c49e6748fea9b_u64;
        // SAFETY: 持久映射可写内存；区域互不相交且都在块内；字节数是
        // 元素数的 4 倍，f32 视图不越界。
        unsafe {
            for reg in [&a, &b, &r] {
                let s = std::slice::from_raw_parts_mut(reg.ptr as *mut f32, N);
                for v in s.iter_mut() {
                    *v = lcg(&mut seed);
                }
            }
            let g = std::slice::from_raw_parts_mut(gate.ptr as *mut f32, C);
            for v in g.iter_mut() {
                *v = lcg(&mut seed) * 0.25;
            }
        }
        let ks = KernelSet::new(&dev, buf, buf_size).unwrap();

        let el = |reg: &Region| reg.offset as u32 / 4;
        let cb = ctx.inner.device.alloc_reusable_cb().unwrap();
        let g1 = [(N as u32).div_ceil(256), 1, 1];
        let gc = [(HW as u32).div_ceil(256), C as u32, 1];
        // SAFETY: cb 处于录制态；PC 结构与各内核的块逐字段对应。
        unsafe {
            record_dispatch(
                &dev,
                cb,
                &ks,
                "sigmoid",
                PcUnary {
                    in_off: el(&a),
                    out_off: el(&outs[0]),
                    n: N as u32,
                }
                .bytes(),
                g1,
            );
            record_dispatch(
                &dev,
                cb,
                &ks,
                "hardsigmoid",
                PcUnaryF {
                    in_off: el(&a),
                    out_off: el(&outs[1]),
                    n: N as u32,
                    p1: 0.2,
                    p2: 0.5,
                }
                .bytes(),
                g1,
            );
            record_dispatch(
                &dev,
                cb,
                &ks,
                "relu",
                PcUnary {
                    in_off: el(&a),
                    out_off: el(&outs[2]),
                    n: N as u32,
                }
                .bytes(),
                g1,
            );
            record_dispatch(
                &dev,
                cb,
                &ks,
                "clip",
                PcUnaryF {
                    in_off: el(&a),
                    out_off: el(&outs[3]),
                    n: N as u32,
                    p1: -1.0,
                    p2: 1.0,
                }
                .bytes(),
                g1,
            );
            record_dispatch(
                &dev,
                cb,
                &ks,
                "add",
                PcBinary {
                    a_off: el(&a),
                    b_off: el(&b),
                    out_off: el(&outs[4]),
                    n: N as u32,
                }
                .bytes(),
                g1,
            );
            record_dispatch(
                &dev,
                cb,
                &ks,
                "mul",
                PcBinary {
                    a_off: el(&a),
                    b_off: el(&b),
                    out_off: el(&outs[5]),
                    n: N as u32,
                }
                .bytes(),
                g1,
            );
            let pc_ch = PcChannel {
                a_off: el(&a),
                b_off: el(&gate),
                out_off: el(&outs[6]),
                hw: HW as u32,
                c: C as u32,
            };
            record_dispatch(&dev, cb, &ks, "mul_c", pc_ch.bytes(), gc);
            record_dispatch(
                &dev,
                cb,
                &ks,
                "muladd_scale",
                PcMulAddScale {
                    f_off: el(&a),
                    gate_off: el(&gate),
                    r_off: el(&r),
                    out_off: el(&outs[7]),
                    hw: HW as u32,
                    c: C as u32,
                }
                .bytes(),
                gc,
            );
            let pc_fs = PcChannel {
                a_off: el(&a),
                b_off: el(&gate),
                out_off: el(&outs[8]),
                hw: HW as u32,
                c: C as u32,
            };
            record_dispatch(&dev, cb, &ks, "fused_sigmoid_mul", pc_fs.bytes(), gc);
            record_dispatch(
                &dev,
                cb,
                &ks,
                "fused_hardsigmoid_mul",
                PcChannelF {
                    a_off: el(&a),
                    g_off: el(&gate),
                    out_off: el(&outs[9]),
                    hw: HW as u32,
                    c: C as u32,
                    alpha: 0.2,
                    beta: 0.5,
                }
                .bytes(),
                gc,
            );
        }
        ctx.inner.device.end_reusable_cb(cb).unwrap();
        ctx.inner.device.submit_wait_cb(cb).unwrap();

        // SAFETY: 已等到信号；coherent 映射直接可读。
        let read = |reg: &Region| unsafe { std::slice::from_raw_parts(reg.ptr as *const f32, N) };
        let av = read(&a);
        let bv = read(&b);
        let rv = read(&r);
        // SAFETY: 同上；gate 区域 C 个元素。
        let gv = unsafe { std::slice::from_raw_parts(gate.ptr as *const f32, C) };

        let mut ref_out = vec![0f32; N];
        sigmoid_tensor(av, &mut ref_out);
        let d = max_abs_diff(read(&outs[0]), &ref_out);
        eprintln!("[gpu] sigmoid        max|diff| = {d:.3e}");
        assert!(d < 1e-5, "sigmoid 超容差: {d}");

        hardsigmoid(av, 0.2, 0.5, &mut ref_out);
        let d = max_abs_diff(read(&outs[1]), &ref_out);
        eprintln!("[gpu] hardsigmoid    max|diff| = {d:.3e}");
        assert!(d < 1e-5, "hardsigmoid 超容差: {d}");

        ref_out.copy_from_slice(av);
        for v in ref_out.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let d = max_abs_diff(read(&outs[2]), &ref_out);
        eprintln!("[gpu] relu           max|diff| = {d:.3e}");
        assert!(d == 0.0, "relu 应逐位一致: {d}");

        ref_out.copy_from_slice(av);
        for v in ref_out.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        let d = max_abs_diff(read(&outs[3]), &ref_out);
        eprintln!("[gpu] clip           max|diff| = {d:.3e}");
        assert!(d == 0.0, "clip 应逐位一致: {d}");

        for (i, v) in ref_out.iter_mut().enumerate() {
            *v = av[i] + bv[i];
        }
        let d = max_abs_diff(read(&outs[4]), &ref_out);
        eprintln!("[gpu] add            max|diff| = {d:.3e}");
        assert!(d == 0.0, "add 应逐位一致: {d}");

        for (i, v) in ref_out.iter_mut().enumerate() {
            *v = av[i] * bv[i];
        }
        let d = max_abs_diff(read(&outs[5]), &ref_out);
        eprintln!("[gpu] mul            max|diff| = {d:.3e}");
        assert!(d == 0.0, "mul 应逐位一致: {d}");

        for (i, v) in ref_out.iter_mut().enumerate() {
            *v = av[i] * gv[i / HW];
        }
        let d = max_abs_diff(read(&outs[6]), &ref_out);
        eprintln!("[gpu] mul_c          max|diff| = {d:.3e}");
        assert!(d == 0.0, "mul_c 应逐位一致: {d}");

        for (i, v) in ref_out.iter_mut().enumerate() {
            // CPU 侧刻意两次舍入（镜像 mul_add_scale_alloc 的不做 fma）
            *v = av[i] * gv[i / HW] + rv[i];
        }
        let d = max_abs_diff(read(&outs[7]), &ref_out);
        eprintln!("[gpu] muladd_scale   max|diff| = {d:.3e}");
        assert!(d < 1e-5, "muladd_scale 超容差: {d}");

        let mut gate_sig = vec![0f32; C];
        sigmoid_tensor(gv, &mut gate_sig);
        for (i, v) in ref_out.iter_mut().enumerate() {
            *v = av[i] * gate_sig[i / HW];
        }
        let d = max_abs_diff(read(&outs[8]), &ref_out);
        eprintln!("[gpu] fused_sig_mul  max|diff| = {d:.3e}");
        assert!(d < 1e-5, "fused_sigmoid_mul 超容差: {d}");

        for (i, v) in ref_out.iter_mut().enumerate() {
            let t = (gv[i / HW] * 0.2 + 0.5).clamp(0.0, 1.0);
            *v = av[i] * t;
        }
        let d = max_abs_diff(read(&outs[9]), &ref_out);
        eprintln!("[gpu] fused_hs_mul   max|diff| = {d:.3e}");
        assert!(d < 1e-5, "fused_hardsigmoid_mul 超容差: {d}");

        // 已录制 CB 的重复提交开销（整图 = 一次提交的载体）
        let mut times: Vec<f64> = Vec::new();
        for _ in 0..10 {
            let t0 = std::time::Instant::now();
            ctx.inner.device.submit_wait_cb(cb).unwrap();
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        eprintln!(
            "[gpu] 10-dispatch CB 重提交：中位 {:.3} ms（min {:.3}）",
            times[5], times[0]
        );
    }
}
