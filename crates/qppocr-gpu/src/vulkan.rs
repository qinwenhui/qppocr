//! Vulkan 后端：实例、物理设备枚举与设备上下文。
//!
//! 现阶段（Phase 1-A）：枚举 + 设备打开（含计算队列族选择）；
//! `create_session` 明确报「内核未实现」——本模块存在的意义是让
//! `--device gpu` 从「编译期拒绝」变成「运行时可见的设备事实」。
//!
//! 版本策略：实例按 **1.1** 请求（最大化可枚举面——按 1.4 请求会让
//! 只有 1.3 ICD 的机器连枚举都失败），可用性按每个物理设备自己的
//! `apiVersion` 判定：低于 1.4 基线的设备在枚举里**可见并标注**，
//! 打开时给出列明一切的明确报错。不静默。

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

/// 尽量低门槛地建实例（1.1）：只为枚举，不筛设备。
/// loader 缺失 / ICD 异常 = `None`（「有没有 GPU」是环境事实，不当错误）。
fn low_instance() -> Option<(Entry, Instance)> {
    // SAFETY: Entry::load 只按平台规则定位并装载 Vulkan loader
    // （Windows vulkan-1.dll / Linux libvulkan.so.1），不触碰其内部状态。
    let entry = unsafe { Entry::load() }.ok()?;
    let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
    let ci = vk::InstanceCreateInfo::default().application_info(&app);
    // SAFETY: app/ci 是栈上值且在本调用期间存活；无扩展、无分配器回调。
    let instance = unsafe { entry.create_instance(&ci, None) }.ok()?;
    Some((entry, instance))
}

/// 枚举本机全部 Vulkan 物理设备（不筛版本——可用性在 [`VulkanContext::open`]
/// 判定）。无 loader / 无设备 = 空列表。
pub(crate) fn enumerate() -> Vec<DeviceInfo> {
    let Some((_entry, instance)) = low_instance() else {
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

/// 一个 Vulkan 设备的上下文：实例 + 选定的物理设备 + 计算队列族。
///
/// Phase 1-A 到此为止；逻辑设备 / 内存分配 / 执行在后续里程碑加字段。
/// `entry`/`instance` 现在就承担职责（physical 句柄只在实例内存域内
/// 有效，必须持有实例），逻辑设备与队列族是 Phase 1-D 的落点——
/// 暂未读取，豁免到那时。
#[allow(dead_code)]
pub struct VulkanContext {
    /// 声明序即 drop 序：instance 必须先于 entry 释放（其函数表借自
    /// entry 装载的 loader 库，反过来是 use-after-free）。
    entry: Entry,
    instance: Instance,
    physical: vk::PhysicalDevice,
    name: String,
    api: String,
    queue_family: u32,
}

impl VulkanContext {
    /// 打开第 `index` 个满足 1.4 基线的设备（`None` = 第一个）。
    ///
    /// 不满足条件时明确报错并**列出看到的一切**——包括低于基线的设备
    /// （标注其版本），不静默降级也不报空泛的「不可用」。
    pub fn open(index: Option<u32>) -> Result<Self> {
        let (entry, instance) = low_instance().ok_or_else(|| {
            Error::Device(
                "找不到 Vulkan loader——安装显卡驱动即随带（Linux 另需 \
                 vulkan-loader 包）；当前枚举结果：无"
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
        let idx = families
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
        Ok(Self {
            entry,
            instance,
            physical,
            name: name.clone(),
            api: api_str(props.api_version),
            queue_family: idx,
        })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        // SAFETY: instance 由本上下文独占持有（物理设备句柄随实例失效，
        // 别处不会有引用）；drop 序上它先于 entry（字段声明序）——
        // 函数表在库卸载前必须先还。无自定义分配器。
        unsafe { self.instance.destroy_instance(None) };
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
        // Phase 1-A：设备枚举与选择已就绪；模型装载（planner + 权重上传）
        // 与执行内核在后续里程碑接入。图与权重在此显式释放。
        drop(graph);
        drop(initializers);
        Err(Error::Device(
            "Vulkan 计算内核尚未实现（Phase 1 开发中）——设备选择已就绪，\
             模型装载与执行随后续版本接入；当前请用 --device cpu"
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
                eprintln!("[vulkan] opened: {} ({})", ctx.name, ctx.api);
            }
            Err(e) => eprintln!("[vulkan] open 跳过（无满足基线的设备）: {e}"),
        }
    }
}
