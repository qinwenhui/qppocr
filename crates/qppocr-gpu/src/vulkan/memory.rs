//! 内存：memory type 选择与 bump arena。
//!
//! 分配策略（对着推理引擎的使用模式设计，不做通用 malloc）：
//! - **权重**：装载期一次、会话终生——arena 只进不出；
//! - **激活**：按 shape 键的计划复用（Phase F 接入），后续版本换
//!   per-run 复位式 arena + 双缓冲。
//!
//! UMA（核显）检测：`HOST_VISIBLE|HOST_COHERENT` 的类型同时带
//! `DEVICE_LOCAL` 时，staging 与设备内存同一池——上传零拷贝路径，
//! 是本项目的家用核显主场景。离散卡走 staging → device 拷贝（Phase F）。

use ash::vk;
use ash::{Device, Instance};
use qppocr_core::error::{Error, Result};

/// 选定的两个 memory type 与 UMA 判定。
pub(crate) struct MemoryTypes {
    /// HOST_VISIBLE | HOST_COHERENT（可映射，上传/回读用）。
    pub staging: u32,
    /// DEVICE_LOCAL（设备侧常驻；UMA 上与 staging 同一类型）。
    pub device: u32,
    /// staging 类型本身就 DEVICE_LOCAL（核显共享内存）。
    pub uma: bool,
}

impl MemoryTypes {
    pub(crate) fn pick(instance: &Instance, physical: vk::PhysicalDevice) -> Result<Self> {
        // SAFETY: physical 来自同实例枚举。
        let props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let required = |flags: vk::MemoryPropertyFlags, i: u32| {
            props.memory_types[i as usize].property_flags & flags == flags
        };
        let staging = (0..props.memory_type_count)
            .find(|&i| {
                required(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    i,
                )
            })
            .ok_or_else(|| {
                Error::Device("没有 HOST_VISIBLE|HOST_COHERENT 的 memory type".into())
            })?;
        let staging_dev_local = props.memory_types[staging as usize]
            .property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
        let device = match (0..props.memory_type_count).find(|&i| {
            required(vk::MemoryPropertyFlags::DEVICE_LOCAL, i)
                && !props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        }) {
            // 离散 VRAM 存在 → 用它；只有核显 → DEVICE_LOCAL 即 staging 同款。
            Some(i) => i,
            None => staging,
        };
        Ok(Self {
            staging,
            device,
            uma: staging_dev_local,
        })
    }
}

/// 一段已分配的区域（句柄是 copy 的；内存存活期归 [`Arena`]）。
pub(crate) struct Region {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
    /// 可映射类型的持久映射地址；不可映射类型为 null。
    pub ptr: *mut u8,
}

// SAFETY: Region 只携带 Vulkan 句柄（整数）、偏移与映射地址，全部
// 来自按内存类型独占的 arena 块；跨线程传递仅用于命令录制侧的只读
// 引用，无内部可变状态——与 ash 句柄本身的 Send+Sync 语义一致。
unsafe impl Send for Region {}
// SAFETY: 同上：&Region 的跨线程共享是只读的（写入只经 &mut 分配方）。
unsafe impl Sync for Region {}

/// bump arena：链式大块，只进不出（权重/常驻数据的形状）。
///
/// v1 的诚实边界：没有 free——推理会话的权重与按 shape 计划的缓冲都
/// 是「分配一次、活到会话结束」的形态，碎片化要等 Phase F 的 per-run
/// 复位 arena 才谈得上。
pub(crate) struct Arena {
    device: Device,
    memory_type: u32,
    mappable: bool,
    chunks: Vec<Chunk>,
    cursor: vk::DeviceSize,
}

struct Chunk {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    mapped: *mut u8,
}

const FIRST_CHUNK: vk::DeviceSize = 16 << 20;

impl Arena {
    pub(crate) fn new(device: Device, memory_type: u32, mappable: bool) -> Self {
        Self {
            device,
            memory_type,
            mappable,
            chunks: Vec::new(),
            cursor: 0,
        }
    }

    /// 分配 `size` 字节（向上对齐到 16——常见 SSBO 对齐下限）。
    pub(crate) fn alloc(&mut self, size: vk::DeviceSize) -> Result<Region> {
        let align: vk::DeviceSize = 16;
        let size = size.div_ceil(align) * align;
        if self.chunks.is_empty() || self.cursor + size > self.chunks.last().unwrap().size {
            // 新块：至少装下本次请求，并给后续留量（封顶 256 MB——
            // 超大单请求仍按其自身大小给足）。
            let want = size.clamp(FIRST_CHUNK, 256 << 20).max(size);
            self.push_chunk(want)?;
        }
        let chunk = self.chunks.last_mut().unwrap();
        let offset = self.cursor;
        self.cursor += size;
        let ptr = if self.mappable {
            // SAFETY: 可映射类型、整块已在分配时映射；offset+size 在块内。
            unsafe { chunk.mapped.add(offset as usize) }
        } else {
            std::ptr::null_mut()
        };
        Ok(Region {
            buffer: chunk.buffer,
            offset,
            size,
            ptr,
        })
    }

    /// 当前（最后一块）的缓冲与整块大小——KernelSet 绑定 SSBO 用。
    ///
    /// 已知边界（Phase F）：多块时只有最后一块可绑定。Phase G 的图
    /// 计划保证全部区域落在一块里（按计划总量一次开块）。
    pub(crate) fn chunk_range(&self) -> Option<(vk::Buffer, vk::DeviceSize)> {
        self.chunks.last().map(|c| (c.buffer, c.size))
    }

    fn push_chunk(&mut self, size: vk::DeviceSize) -> Result<()> {
        let device = &self.device;
        let bci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: bci 合法；缓冲只绑定本块内存。
        let buffer = unsafe { device.create_buffer(&bci, None) }
            .map_err(|e| Error::Device(format!("vkCreateBuffer({size} B) 失败: {e}")))?;
        // SAFETY: 查询的是刚创建的缓冲的需求。
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mai = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(self.memory_type);
        // SAFETY: 大小来自缓冲需求；类型经调用方选定。
        let memory = unsafe { device.allocate_memory(&mai, None) }
            .map_err(|e| Error::Device(format!("vkAllocateMemory({} B) 失败: {e}", req.size)))?;
        // SAFETY: 偏移 0 满足 req.alignment；绑定后本块独占该内存。
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| Error::Device(format!("绑定缓冲内存失败: {e}")))?;
        let mapped = if self.mappable {
            // SAFETY: 类型可映射；范围覆盖整块；映射与块同生命周期。
            // （ash 0.38 的 map_memory 直接返回指针。）
            unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
                .map_err(|e| Error::Device(format!("vkMapMemory 失败: {e}")))?
                as *mut u8
        } else {
            std::ptr::null_mut()
        };
        self.chunks.push(Chunk {
            buffer,
            memory,
            size: req.size,
            mapped,
        });
        self.cursor = 0;
        Ok(())
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: 块句柄由 arena 独占；先还映射再毁内存，最后毁缓冲。
        unsafe {
            for c in &self.chunks {
                if !c.mapped.is_null() {
                    self.device.unmap_memory(c.memory);
                }
                self.device.destroy_buffer(c.buffer, None);
                self.device.free_memory(c.memory, None);
            }
        }
    }
}
