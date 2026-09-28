//! 逻辑设备：队列、timeline semaphore、一次性命令提交。
//!
//! Phase 1-D 的执行基建：一个计算队列、一枚全上下文共享的 timeline
//! semaphore（信号值全局单调递增）、一个可重置的命令池。会话的每帧
//! 提交都经 [`VulkanDevice::submit_one_shot`]——录命令、带信号值提交、
//! 主机等值，Phase 2 再把「等」从热路径上拆掉（双缓冲 staging 与
//! 异步提交）。

use ash::vk;
use ash::{Device, Instance};
use qppocr_core::error::{Error, Result};
use std::sync::atomic::{AtomicU64, Ordering};

/// 一次性等待的超时（10 s）：正常的一次 copy/compute 前向远低于此；
/// 超时大概率是设备丢失或死锁，宁可报错不要挂死。
const WAIT_TIMEOUT_NS: u64 = 10_000_000_000;

/// 逻辑设备与执行基建。
pub(crate) struct VulkanDevice {
    device: Device,
    queue: vk::Queue,
    /// 共享 timeline 的下一个信号值（0 留给初始态）。
    next_signal: AtomicU64,
    command_pool: vk::CommandPool,
    timeline: vk::Semaphore,
}

impl VulkanDevice {
    /// 建逻辑设备（api 1.4 + timeline semaphore）并取出计算队列。
    ///
    /// timeline semaphore 在 1.2 起核心化——1.4 物理设备必然支持，
    /// 但仍显式查询特性并启用（feature 不开即使核心也不生效）。
    pub(crate) fn new(
        instance: &Instance,
        physical: vk::PhysicalDevice,
        queue_family: u32,
    ) -> Result<Self> {
        // 特性先行查询：开不了就带着明确原因失败，不靠驱动兜底。
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut info2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut v12);
        // SAFETY: physical 来自同一实例的枚举；info2 的 pNext 链在调用期间存活。
        unsafe { instance.get_physical_device_features2(physical, &mut info2) };
        if v12.timeline_semaphore == 0 {
            return Err(Error::Device(
                "设备不支持 timeline semaphore（1.4 物理设备不应如此——\
                 驱动异常？）"
                    .into(),
            ));
        }
        let priorities = [1.0f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities);
        let queue_cis = [queue_ci];
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
        let ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_cis)
            .push_next(&mut v12);
        // SAFETY: ci 与其 pNext 链在调用期间存活；无自定义分配器。
        let device = unsafe { instance.create_device(physical, &ci, None) }
            .map_err(|e| Error::Device(format!("vkCreateDevice 失败: {e}")))?;
        // SAFETY: family/索引来自设备创建时的队列描述。
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: pool_ci 合法；池只被本上下文使用。
        let command_pool = unsafe { device.create_command_pool(&pool_ci, None) }
            .map_err(|e| Error::Device(format!("创建命令池失败: {e}")))?;
        // timeline semaphore：类型通过 pNext 指定，初始值 0。
        let mut sem_type =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let sem_ci = vk::SemaphoreCreateInfo::default().push_next(&mut sem_type);
        // SAFETY: sem_ci 与其 pNext 链在调用期间存活。
        let timeline = unsafe { device.create_semaphore(&sem_ci, None) }
            .map_err(|e| Error::Device(format!("创建 timeline semaphore 失败: {e}")))?;
        Ok(Self {
            device,
            queue,
            next_signal: AtomicU64::new(1),
            command_pool,
            timeline,
        })
    }

    /// 取下一个信号值（提交前调用；提交与等待共用这个值）。
    fn take_signal(&self) -> u64 {
        self.next_signal.fetch_add(1, Ordering::Relaxed)
    }

    /// 一次性提交：分配命令缓冲 → 录制 → 带 timeline 信号提交 → 主机等值。
    ///
    /// Phase 1 的同步骨架；命令缓冲用完即还（池可重置）。
    pub(crate) fn submit_one_shot(
        &self,
        record: impl FnOnce(&Device, vk::CommandBuffer),
    ) -> Result<()> {
        let ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: 池存活且属于本设备的队列族；每线程串行使用（Phase 1
        // 的提交都在锁内——pipeline 的串行批保证）。
        let cb = unsafe { self.device.allocate_command_buffers(&ai) }
            .map_err(|e| Error::Device(format!("分配命令缓冲失败: {e}")))?[0];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: cb 来自本池，处于初始态。
        unsafe { self.device.begin_command_buffer(cb, &begin) }
            .map_err(|e| Error::Device(format!("begin 命令缓冲失败: {e}")))?;
        record(&self.device, cb);
        // SAFETY: 录制完毕。
        unsafe { self.device.end_command_buffer(cb) }
            .map_err(|e| Error::Device(format!("end 命令缓冲失败: {e}")))?;
        let signal = self.take_signal();
        // timeline 信号值经 pNext（VkTimelineSemaphoreSubmitInfo）——
        // ash 0.38 的 SubmitInfo 没有内联 builder 字段。切片先落局部
        //（builder 借用要求活过结构体本身）。
        let cbs = [cb];
        let signal_values = [signal];
        let sems = [self.timeline];
        let mut tl_submit =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
        let si = vk::SubmitInfo::default()
            .command_buffers(&cbs)
            .signal_semaphores(&sems)
            .push_next(&mut tl_submit);
        // SAFETY: 队列来自本设备的计算族；si 及其 pNext 引用的
        // cb/semaphore 均存活且有效。
        unsafe {
            self.device
                .queue_submit(self.queue, &[si], vk::Fence::null())
        }
        .map_err(|e| Error::Device(format!("vkQueueSubmit 失败: {e}")))?;
        let sems = [self.timeline];
        let values = [signal];
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(&sems)
            .values(&values);
        // SAFETY: timeline 为有效句柄；等待值是刚提交的信号值。
        unsafe { self.device.wait_semaphores(&wait, WAIT_TIMEOUT_NS) }
            .map_err(|e| Error::Device(format!("等待 timeline 信号失败: {e}")))?;
        // SAFETY: 提交已等完，命令缓冲可安全归还池。
        unsafe { self.device.free_command_buffers(self.command_pool, &[cb]) };
        Ok(())
    }

    /// 暴露给内存/内核模块的设备句柄（只读用途）。
    pub(crate) fn raw(&self) -> &Device {
        &self.device
    }

    /// 分配一个**可复用**命令缓冲：录制一次、反复提交（不释放不重置）。
    ///
    /// 这是「每帧一次提交」的载体——整图的 dispatch 序列在装载期录好，
    /// 每帧只 submit。池没开 RESET_COMMAND_BUFFER，靠不重置保证语义。
    pub(crate) fn alloc_reusable_cb(&self) -> Result<vk::CommandBuffer> {
        let ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: 池存活；缓冲从不重置/释放（SIMULTANEOUS_USE 语义靠
        // 「录制一次、多帧提交」保证——每帧提交前上一帧已等完信号）。
        let cb = unsafe { self.device.allocate_command_buffers(&ai) }
            .map_err(|e| Error::Device(format!("分配命令缓冲失败: {e}")))?[0];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
        // SAFETY: cb 刚分配，处于初始态。
        unsafe { self.device.begin_command_buffer(cb, &begin) }
            .map_err(|e| Error::Device(format!("begin 命令缓冲失败: {e}")))?;
        Ok(cb)
    }

    /// 结束可复用命令缓冲的录制（录制完才能提交）。
    pub(crate) fn end_reusable_cb(&self, cb: vk::CommandBuffer) -> Result<()> {
        // SAFETY: cb 处于录制态。
        unsafe { self.device.end_command_buffer(cb) }
            .map_err(|e| Error::Device(format!("end 命令缓冲失败: {e}")))
    }

    /// 提交**已录制**的命令缓冲并等它完成（timeline 信号）。
    ///
    /// 与 `submit_one_shot` 的区别：不录不释放——热路径的每帧成本就
    /// 是这里：一次 submit + 一次主机等值。
    pub(crate) fn submit_wait_cb(&self, cb: vk::CommandBuffer) -> Result<()> {
        let signal = self.take_signal();
        let cbs = [cb];
        let signal_values = [signal];
        let sems = [self.timeline];
        let mut tl_submit =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
        let si = vk::SubmitInfo::default()
            .command_buffers(&cbs)
            .signal_semaphores(&sems)
            .push_next(&mut tl_submit);
        // SAFETY: cb 已结束录制且未被并发提交（调用方串行）；信号量有效。
        unsafe {
            self.device
                .queue_submit(self.queue, &[si], vk::Fence::null())
        }
        .map_err(|e| Error::Device(format!("vkQueueSubmit 失败: {e}")))?;
        let values = [signal];
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(&sems)
            .values(&values);
        // SAFETY: 等待值是刚提交的信号值。
        unsafe { self.device.wait_semaphores(&wait, WAIT_TIMEOUT_NS) }
            .map_err(|e| Error::Device(format!("等待 timeline 信号失败: {e}")))?;
        Ok(())
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // SAFETY: 句柄由本结构独占，drop 时设备上不再有未完成工作
        //（submit_one_shot 每次都等到信号）；无自定义分配器。
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_semaphore(self.timeline, None);
            self.device.destroy_device(None);
        }
    }
}
