//! Vulkan 后端：实例、物理设备枚举与设备上下文。
//!
//! 里程碑进度（Phase 1）：
//! - **A**：枚举 + 设备打开（本模块）；
//! - **D**：逻辑设备 / 计算队列 / timeline semaphore / 一次性提交
//!   （[`device`]）+ memory type 选择与 bump arena（[`memory`]）；
//! - **F**：compute 管线与内核、权重装载、会话执行。
//!
//! 版本策略：实例按 **1.1** 请求（最大化可枚举面——按 1.4 请求会让
//! 只有 1.3 ICD 的机器连枚举都失败），可用性按每个物理设备自己的
//! `apiVersion` 判定：低于 1.4 基线的设备在枚举里**可见并标注**，
//! 打开时给出列明一切的明确报错。不静默。

// Phase F（计算管线与会话执行）起由 create_session 消费；当前只有
// roundtrip 测试使用——豁免 dead_code 到接线完成。
#[allow(dead_code)]
pub(crate) mod device;
pub(crate) mod fp16;
#[allow(dead_code)]
pub(crate) mod memory;
pub(crate) mod nhwc;
#[allow(dead_code)]
pub(crate) mod pipeline;
pub(crate) mod session;

#[cfg(test)]
mod kernel_tests;
#[cfg(test)]
mod n_kernel_tests;

use ash::vk;
use ash::{Entry, Instance};
use qppocr_core::device::{DeviceContext, DeviceInfo, DeviceKind, DeviceSession, SessionOptions};
use qppocr_core::error::{Error, Result};
use qppocr_core::onnx::model::Graph;
use qppocr_core::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

/// Vulkan 1.4 基线：timeline semaphore / sync2 已核心化，无需扩展探测。
/// 打包后的 apiVersion 是单调的，直接整数比较。
const REQUIRED_API: u32 = vk::make_api_version(0, 1, 4, 0);

/// `PhysicalDeviceProperties::device_name`（`[i8; 256]`，C 字符串）→ String。
fn device_name(p: &vk::PhysicalDeviceProperties) -> String {
    let bytes: Vec<u8> = p
        .device_name
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn api_str(v: u32) -> String {
    format!(
        "vulkan {}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}

/// 按指定 apiVersion 建实例。
///
/// 枚举走 **1.1**（门槛最低——按 1.4 请求会让只有 1.3 ICD 的机器连
/// 枚举都失败）；`open` 走 **1.4**——loader 只给实例版本以内的核心函数
/// 派发指针（vkWaitSemaphores 是 1.2 核心函数，1.1 实例下是 null，
/// 调用即 panic），timeline 语义必须实例 ≥ 1.2。
/// loader 缺失 / ICD 异常 = `None`（「有没有 GPU」是环境事实，不当错误）。
fn instance_at(api: u32) -> Option<(Entry, Instance)> {
    // SAFETY: Entry::load 只按平台规则定位并装载 Vulkan loader
    // （Windows vulkan-1.dll / Linux libvulkan.so.1），不触碰其内部状态。
    let entry = unsafe { Entry::load() }.ok()?;
    let app = vk::ApplicationInfo::default().api_version(api);
    let ci = vk::InstanceCreateInfo::default().application_info(&app);
    // SAFETY: app/ci 是栈上值且在本调用期间存活；无扩展、无分配器回调。
    let instance = unsafe { entry.create_instance(&ci, None) }.ok()?;
    Some((entry, instance))
}

/// 枚举本机全部 Vulkan 物理设备（不筛版本——可用性在 [`VulkanContext::open`]
/// 判定）。无 loader / 无设备 = 空列表。
pub(crate) fn enumerate() -> Vec<DeviceInfo> {
    let Some((_entry, instance)) = instance_at(vk::make_api_version(0, 1, 1, 0)) else {
        return Vec::new();
    };
    // SAFETY: instance 存活；失败（驱动异常）按「无设备」处理。
    let pds = unsafe { instance.enumerate_physical_devices() }.unwrap_or_default();
    let mut out = Vec::with_capacity(pds.len());
    for pd in pds {
        // SAFETY: pd 来自同实例的枚举结果。
        let p = unsafe { instance.get_physical_device_properties(pd) };
        out.push(DeviceInfo {
            kind: DeviceKind::Vulkan,
            name: device_name(&p),
            api: api_str(p.api_version),
        });
    }
    // SAFETY: instance 由本函数独占，此后不再使用；无自定义分配器。
    // （ash 0.38 的 Instance 没有 Drop——不显式销毁就是泄漏。）
    unsafe { instance.destroy_instance(None) };
    out
}

/// loader 库的「永不卸载」守卫。
///
/// 退出时 FreeLibrary(vulkan-1.dll) 会与驱动内部线程赛跑——实测：
/// stdout 重定向到文件时 100% 段错误、管道时 0%（纯时序）。Vulkan
/// 应用的通行做法就是**不卸载 loader**：进程结束由 OS 回收。泄漏量
/// = 每进程一个库句柄，可忽略。
#[allow(dead_code)] // 字段 0 故意不读不释放（见上文档）
struct NeverUnload(std::mem::ManuallyDrop<Entry>);

/// 实例的销毁守卫：ash 0.38 的 `Instance` 没有 Drop，不显式销毁即泄漏。
///
/// 包成独立结构是为了让 drop **排序**可表达：字段按声明序释放，
/// [`VulkanContext`] 里 device → instance → entry 的顺序即销毁序——
/// spec 要求实例销毁前其派生的逻辑设备必须全部销毁（先毁实例是 UB，
/// 本机实测为间歇性 ACCESS_VIOLATION）。
struct InstanceGuard(Instance);

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        // SAFETY: 实例由本守卫独占；此刻逻辑设备（声明在前的字段）
        // 已释放完毕；无自定义分配器。
        unsafe { self.0.destroy_instance(None) };
    }
}

/// 上下文的实体（Arc 内共享；字段序即销毁序 device → instance → entry）。
pub(crate) struct Inner {
    name: String,
    api: String,
    /// 逻辑设备与提交基建。
    pub(crate) device: std::sync::Arc<device::VulkanDevice>,
    /// 选定的 memory types 与 UMA 判定。
    pub(crate) mem_types: memory::MemoryTypes,
    /// 协作矩阵（XMX/DNA）形状：fp16 输入、f32 累加、Subgroup 域。
    pub(crate) coopmat: Option<CoopMat>,
    /// 计算队列族的 subgroup 大小（coopmat 内核的排布前提）。
    pub(crate) subgroup_size: u32,
    /// 实例句柄：`Drop::drop` 里显式 destroy（ash 0.38 无 Drop）。
    /// 字段从不读取——存在即销毁序职责（见 [`InstanceGuard`]）。
    #[allow(dead_code)]
    instance: InstanceGuard,
    /// loader 库句柄，声明在最后；经 [`NeverUnload`] 故意不卸载（见其文档）。
    #[allow(dead_code)]
    entry: NeverUnload,
}

/// 探测到的协作矩阵形状（Intel XMX 8×16×16 / AMD 16×16×16 …）。
/// 字段仅作诊断回报（Debug 打印）；本机无 coopmat，路径未启用。
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct CoopMat {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

/// 一个 Vulkan 设备的上下文：实例 + 选定的物理设备 + 执行基建。
///
/// **Clone = Arc 共享**：会话持有整个上下文（实例的存活期必须盖住
/// 它派生的全部逻辑设备——门面把 ctx 当局部变量用完就扔，会话若只
/// 持设备不持实例，实例销毁后提交即 UB，实测 SEGV）。
#[derive(Clone)]
pub struct VulkanContext {
    pub(crate) inner: std::sync::Arc<Inner>,
}

impl VulkanContext {
    /// 打开第 `index` 个满足 1.4 基线的设备（`None` = 第一个）。
    ///
    /// 不满足条件时明确报错并**列出看到的一切**——包括低于基线的设备
    /// （标注其版本），不静默降级也不报空泛的「不可用」。
    pub fn open(index: Option<u32>) -> Result<Self> {
        // 实例按 1.4 建理由见 [`instance_at`]：核心函数派发按实例版本走。
        // 失败含两种：无 loader（枚举也为空）与 loader < 1.4——都构成
        // 「显式要 GPU 但不可用」的明确错误。
        let (entry, instance) = instance_at(REQUIRED_API).ok_or_else(|| {
            Error::Device(
                "无法创建 Vulkan 1.4 实例——loader 缺失（装显卡驱动即随带；\
                 Linux 另需 vulkan-loader 包）或版本低于 1.4"
                    .into(),
            )
        })?;
        // SAFETY: instance 存活。
        let pds = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| Error::Device(format!("vkEnumeratePhysicalDevices 失败: {e}")))?;
        if pds.is_empty() {
            return Err(Error::Device(
                "有 Vulkan loader 但没有任何物理设备（ICD 未装或被禁用）".into(),
            ));
        }
        let all: Vec<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)> = pds
            .iter()
            .map(|&pd| {
                // SAFETY: pd 来自同实例的枚举结果。
                (pd, unsafe { instance.get_physical_device_properties(pd) })
            })
            .collect();
        let usable: Vec<usize> = all
            .iter()
            .enumerate()
            .filter(|(_, (_, p))| p.api_version >= REQUIRED_API)
            .map(|(i, _)| i)
            .collect();
        let seen: String = all
            .iter()
            .map(|(_, p)| format!("{} ({})", device_name(p), api_str(p.api_version)))
            .collect::<Vec<_>>()
            .join("; ");
        if usable.is_empty() {
            return Err(Error::Device(format!(
                "检测到 Vulkan 设备但不满足 1.4 基线：{seen}。GPU 路径需要 \
                 Vulkan 1.4+ 驱动；或用 --device cpu"
            )));
        }
        let pick = match index {
            Some(i) => *usable.get(i as usize).ok_or_else(|| {
                Error::Device(format!(
                    "--device vulkan:{i} 超出范围（满足基线的设备 {} 个）：{seen}",
                    usable.len()
                ))
            })?,
            None => usable[0],
        };
        let (physical, props) = all[pick];
        let name = device_name(&props);
        // 协作矩阵探测：VK_KHR_cooperative_matrix + fp16/f32/Subgroup 形状。
        // 没有就 None——内核族走标量 f16 路径（仍远快于 f32 标量）。
        let coopmat = probe_coopmat(&entry, &instance, physical);
        // 计算队列族的 subgroup 大小（Intel=16；coopmat 内核按 16 排布）。
        let mut sg = vk::PhysicalDeviceSubgroupProperties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg);
        // SAFETY: physical 来自同实例枚举；p2 的 pNext 链在调用期间存活。
        unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
        let subgroup_size = sg.subgroup_size;
        // 计算队列族：优先**专用**计算队列（无图形位），核显上通常是 0 号
        // 图形队列之外的独立族，提交不被图形负载排队。
        // SAFETY: physical 来自同实例的枚举结果。
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let queue_family = families
            .iter()
            .position(|f| {
                f.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .or_else(|| {
                families
                    .iter()
                    .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            })
            .ok_or_else(|| Error::Device(format!("设备 {name} 没有计算队列")))?
            as u32;
        let device = std::sync::Arc::new(device::VulkanDevice::new(
            &instance,
            physical,
            queue_family,
            coopmat.is_some(),
        )?);
        let mem_types = memory::MemoryTypes::pick(&instance, physical)?;
        Ok(Self {
            inner: std::sync::Arc::new(Inner {
                name: name.clone(),
                api: api_str(props.api_version),
                device,
                mem_types,
                coopmat,
                subgroup_size,
                instance: InstanceGuard(instance),
                entry: NeverUnload(std::mem::ManuallyDrop::new(entry)),
            }),
        })
    }
}

/// 探测 fp16×f32(Subgroup) 的协作矩阵形状。
fn probe_coopmat(
    entry: &Entry,
    instance: &Instance,
    physical: vk::PhysicalDevice,
) -> Option<CoopMat> {
    if std::env::var_os("QPPOCR_GPU_NO_CM").is_some() {
        return None; // 手动关：对标量路径做 A/B
    }
    // SAFETY: physical 来自同实例枚举；失败按「无扩展」处理。
    let exts = unsafe { instance.enumerate_device_extension_properties(physical) }
        .ok()
        .unwrap_or_default();
    let cm_name = ash::khr::cooperative_matrix::NAME;
    let has = exts.iter().any(|e| {
        // SAFETY: extension_name 是以 NUL 结尾的 C 字符数组。
        let n = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
        n == cm_name
    });
    if !has {
        return None;
    }
    let cm = ash::khr::cooperative_matrix::Instance::new(entry, instance);
    // SAFETY: 扩展存在（函数指针刚由 loader 加载）。
    let props = unsafe { cm.get_physical_device_cooperative_matrix_properties(physical) }.ok()?;
    // A/B=fp16、C/Result=f32、Subgroup 域——XMX/DNA 的标准形态。
    props
        .iter()
        .find(|p| {
            p.a_type == vk::ComponentTypeKHR::FLOAT16
                && p.b_type == vk::ComponentTypeKHR::FLOAT16
                && p.c_type == vk::ComponentTypeKHR::FLOAT32
                && p.result_type == vk::ComponentTypeKHR::FLOAT32
                && p.scope == vk::ScopeKHR::SUBGROUP
        })
        .map(|p| CoopMat {
            m: p.m_size,
            n: p.n_size,
            k: p.k_size,
        })
}

impl DeviceContext for VulkanContext {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Vulkan
    }

    fn info(&self) -> DeviceInfo {
        DeviceInfo {
            kind: DeviceKind::Vulkan,
            name: self.inner.name.clone(),
            api: self.inner.api.clone(),
        }
    }

    fn create_session(
        &self,
        graph: Graph,
        initializers: HashMap<String, Tensor>,
        opts: &SessionOptions,
    ) -> Result<Arc<dyn DeviceSession>> {
        // 部署分级（QPPOCR_GPU_STAGES，默认 **detrec**）：det+rec 上 GPU、
        // cls 留 CPU。=det 可退回仅-det（rec 走 CPU——一次性单图冷进程
        // 避付 rec 计划构建税 ~13ms×桶数，冷机单图实测 85 vs 158ms）。
        // 转正依据（bench/COMPARISON.md 2026-09-30 各节）：持续/批量
        // detrec 全链反超纯 CPU 26% 且 CPU 线程全释放、机器热态下
        // det-only 的 CPU rec 劣化 37→60ms 而 detrec 恒稳；冷机语料
        // 1.028 平手；精度逐位同（GT 860/1036）。
        // cls 恒 CPU（=all 才上 GPU）：GPU cls argmax 真实图 ~14% 行边界
        // 翻转分歧（翻转错=毁整行）且更慢（5.3 vs 4.2ms）。委托 CPU 时
        // **stderr 声明**，不是静默降级。
        let stages =
            std::env::var("QPPOCR_GPU_STAGES").unwrap_or_else(|_| "detrec".into());
        let gpu_ok = match opts.model {
            qppocr_core::device::ModelRole::Det => true,
            qppocr_core::device::ModelRole::Rec => stages == "detrec" || stages == "all",
            qppocr_core::device::ModelRole::Cls => stages == "all",
            _ => false,
        };
        if !gpu_ok && opts.model != qppocr_core::device::ModelRole::Det {
            eprintln!(
                "[gpu] QPPOCR_GPU_STAGES={stages}：{:?} 模型走 CPU 会话（分级部署）",
                opts.model
            );
            return Ok(Arc::new(qppocr_core::executor::Session::from_parts(
                graph,
                initializers,
            )));
        }
        // rec：**覆盖面探针**——一次性探针会话小形状试建计划（克隆图/
        // 权重，探完即弃）。不支持的算子（如 small 的 rank-5 注意力
        // Transpose 族）与形状无关，探针必现；失败则 stderr 声明后原图
        // 退回 CPU 会话（非静默降级；cls 同款分级纪律）——补齐内核后
        // 探针自动放行。
        if opts.model == qppocr_core::device::ModelRole::Rec {
            let probe_err = session::VulkanSession::new(
                self.inner.clone(),
                graph.clone(),
                initializers.clone(),
            )?
            .probe(&[1, 3, 48, 64])
            .err();
            if let Some(e) = probe_err {
                eprintln!(
                    "[gpu] rec 会话探测失败（{e}）：rec 走 CPU 会话（分级部署，\
                     补齐 n_ 覆盖面后自动上 GPU）"
                );
                return Ok(Arc::new(qppocr_core::executor::Session::from_parts(
                    graph,
                    initializers,
                )));
            }
        }
        let mut session = session::VulkanSession::new(self.inner.clone(), graph, initializers)?;
        // rec 会话批维补齐粒度 8：引擎按此合批（宽填充到批内最大桶），
        // 会话补齐空行 + 内核 real_n 早退——实测每行提交往返 ~0.85ms、
        // 11 行图 11 次提交吃 9.3ms；合批后 (8,W桶) 形状全命中。det/cls
        // 批维恒 1（det 单图；cls 留 CPU）。
        if opts.model == qppocr_core::device::ModelRole::Rec {
            session.batch_grain = 8;
            // CTC 头只吃每时间步 argmax：GPU 端归约成 (val, idx) 对再回读
            //（全量 T×V 概率回读实测 ~18 ms/图，clflush 读回带宽是大头）。
            session.argmax_exit = true;
        }
        Ok(Arc::new(session))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 枚举在有设备的机器上应列出（本机开发环境：Intel Arc 核显，
    /// 期望一项 "vulkan 1.4.x"）；无 loader/ICD 的机器（CI）自动通过。
    #[test]
    fn enumerate_lists_or_skips() {
        let ds = enumerate();
        for d in &ds {
            assert_eq!(d.kind, DeviceKind::Vulkan);
            eprintln!("[vulkan] {} ({})", d.name, d.api);
        }
    }

    /// 满足基线的机器上 open(None) 必须成功；无设备环境跳过（CI）。
    #[test]
    fn open_first_or_skip() {
        match VulkanContext::open(None) {
            Ok(ctx) => {
                assert_eq!(ctx.kind(), DeviceKind::Vulkan);
                eprintln!(
                    "[vulkan] opened: {} ({}) | UMA={} | coopmat={:?} | subgroup={}",
                    ctx.inner.name,
                    ctx.inner.api,
                    ctx.inner.mem_types.uma,
                    ctx.inner.coopmat,
                    ctx.inner.subgroup_size
                );
            }
            Err(e) => eprintln!("[vulkan] open 跳过（无满足基线的设备）: {e}"),
        }
    }

    /// 端到端冒烟：arena 分配两段 → 主机写入（UMA/coherent 映射）→
    /// GPU 拷贝 → 主机读回验证。串起 Phase D 的全部基建：
    /// 逻辑设备、队列、timeline 提交与等待、memory type、映射。
    #[test]
    fn copy_roundtrip_or_skip() {
        let ctx = match VulkanContext::open(None) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[vulkan] roundtrip 跳过（无满足基线的设备）: {e}");
                return;
            }
        };
        const N: usize = 4096;
        let mut arena = memory::Arena::new(
            ctx.inner.device.raw().clone(),
            ctx.inner.mem_types.staging,
            true,
        );
        let src = arena.alloc(N as vk::DeviceSize).expect("arena src");
        let dst = arena.alloc(N as vk::DeviceSize).expect("arena dst");
        assert_eq!(src.buffer, dst.buffer);
        assert!(dst.offset >= src.offset + src.size);
        // 主机写入（HOST_COHERENT：无需 flush 即对设备可见）
        // SAFETY: ptr 来自持久映射的可写内存，范围在块内且互不重叠。
        unsafe {
            let s = std::slice::from_raw_parts_mut(src.ptr, N);
            for (i, b) in s.iter_mut().enumerate() {
                *b = (i % 251) as u8;
            }
            std::ptr::write_bytes(dst.ptr, 0, N);
        }
        ctx.inner
            .device
            .submit_one_shot(|d, cb| {
                // SAFETY: cb 处于录制态；两段同一缓冲、区域不相交——
                // BufferCopy 的偏移必须是**区域**偏移（漏了就是拷回自己）。
                unsafe {
                    d.cmd_copy_buffer(
                        cb,
                        src.buffer,
                        dst.buffer,
                        &[vk::BufferCopy::default()
                            .src_offset(src.offset)
                            .dst_offset(dst.offset)
                            .size(N as vk::DeviceSize)],
                    )
                }
            })
            .expect("submit");
        // SAFETY: 已等到 timeline 信号，拷贝完成；coherent 映射直接可读。
        let got = unsafe { std::slice::from_raw_parts(dst.ptr, N) };
        let want: Vec<u8> = (0..N).map(|i| (i % 251) as u8).collect();
        assert_eq!(got, want, "GPU 拷贝回读不符");
        eprintln!(
            "[vulkan] copy roundtrip OK (UMA={})",
            ctx.inner.mem_types.uma
        );
    }
}
