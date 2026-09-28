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
#[allow(dead_code)]
pub(crate) mod memory;
#[allow(dead_code)]
pub(crate) mod pipeline;

#[cfg(test)]
mod kernel_tests;

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

/// 一个 Vulkan 设备的上下文：实例 + 选定的物理设备 + 执行基建。
///
/// `device`（逻辑设备/队列/提交）与 `mem_types`（含 UMA 判定）在
/// [`VulkanContext::open`] 时一并就绪；`create_session` 待 Phase F 的
/// planner 与内核接入。
pub struct VulkanContext {
    name: String,
    api: String,
    /// 逻辑设备与提交基建（Phase F 起由会话消费；当前仅测试使用）。
    #[allow(dead_code)]
    pub(crate) device: device::VulkanDevice,
    /// 选定的 memory types 与 UMA 判定（同上）。
    #[allow(dead_code)]
    pub(crate) mem_types: memory::MemoryTypes,
    /// 字段序即销毁序（device → instance → entry）：见 [`InstanceGuard`]。
    /// 字段从不读取——存在即销毁序职责，豁免 dead_code。
    #[allow(dead_code)]
    instance: InstanceGuard,
    /// loader 库句柄，**必须声明在最后**：它的卸载要晚于前面所有字段
    /// Drop 里对 loader/驱动函数表（驻留在库内）的最后一次调用。
    /// 字段本身从不读取，存在即职责，豁免 dead_code。
    #[allow(dead_code)]
    entry: Entry,
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
        let device = device::VulkanDevice::new(&instance, physical, queue_family)?;
        let mem_types = memory::MemoryTypes::pick(&instance, physical)?;
        Ok(Self {
            name: name.clone(),
            api: api_str(props.api_version),
            device,
            mem_types,
            instance: InstanceGuard(instance),
            entry,
        })
    }
}

impl DeviceContext for VulkanContext {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Vulkan
    }

    fn info(&self) -> DeviceInfo {
        DeviceInfo {
            kind: DeviceKind::Vulkan,
            name: self.name.clone(),
            api: self.api.clone(),
        }
    }

    fn create_session(
        &self,
        graph: Graph,
        initializers: HashMap<String, Tensor>,
        _opts: &SessionOptions,
    ) -> Result<Arc<dyn DeviceSession>> {
        // Phase 1-D 已到：设备/队列/timeline/内存基建就绪。下一步是
        // planner（静态形状推理）+ 计算管线与内核（Phase E/F）。
        // 图与权重在此显式释放。
        drop(graph);
        drop(initializers);
        Err(Error::Device(
            "Vulkan 计算内核尚未实现（Phase 1 进行中：设备与内存基建已就绪，\
             planner 与内核随后接入）；当前请用 --device cpu"
                .into(),
        ))
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
                    "[vulkan] opened: {} ({}) | UMA={}",
                    ctx.name, ctx.api, ctx.mem_types.uma
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
        let mut arena = memory::Arena::new(ctx.device.raw().clone(), ctx.mem_types.staging, true);
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
        ctx.device
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
        eprintln!("[vulkan] copy roundtrip OK (UMA={})", ctx.mem_types.uma);
    }
}
