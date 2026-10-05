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

/// 读回屏障：主机读设备写入的映射内存前，逐出 `[ptr, ptr+bytes)` 的
/// CPU 缓存行（x86_64 用 clflush；其它架构 no-op，依赖驱动一致性）。
///
/// 实证（2026-09-29，Intel Arc Pro + Windows 驱动）：HOST_COHERENT 映射
/// 在 timeline 信号量主机等待后仍间歇读到设备写之前的旧值——dw 批式
/// 内核测试 ~30% 概率「随机批子集部分行全零」，睡 200ms 后自愈（缓存
/// 行自然逐出），vkInvalidateMappedMemoryRanges 与 device_wait_idle 均
/// 无效（驱动 no-op）。clflush 逐出后 10/10 全清。生产读回路径（run 的
/// 输出拷出）必须先过这道屏障，否则结果间歇含零。
#[allow(dead_code)]
pub(crate) fn readback_clean(_ptr: *const u8, _bytes: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::_mm_clflush;
        let mut off = 0usize;
        while off < _bytes {
            // SAFETY: ptr 来自持久映射，ptr+off 在映射区内；clflush 对
            // 任意有效地址安全（含未驻留行）。
            unsafe { _mm_clflush(_ptr.add(off)) };
            off += 64;
        }
    }
}

/// 发布屏障：主机写完映射内存后，把 `[ptr, ptr+bytes)` 的脏行**写回并
/// 逐出**，使设备立即可见——readback_clean 的镜像方向。
///
/// 与 CB 头部内存屏障（session.rs build_plan）**配对使用、缺一不可**
/// （间歇性整行空文本的 A/B 实证，2026-09-30）：主机侧小写
/// （rn=real_n 单元，4 字节）无容量压力、永不自我逐出，需 SFENCE +
/// clflush 写回逐出；GPU 侧缓存的旧行另由 CB 屏障失效。
///
/// ★ 必须先 SFENCE 再 clflush：CLFLUSH 与前置 store **弱有序**
/// （Intel SDM：CLFLUSH 不与 stores 排序，除非插入 SFENCE）——漏了
/// fence 时逐出的是旧行、store 随后才落缓存，实测竞态照旧。
pub(crate) fn publish_clean(_ptr: *const u8, _bytes: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::{_mm_clflush, _mm_sfence};
        // SAFETY: 无内存操作；序列化本线程的 store buffer。
        unsafe { _mm_sfence() };
        let mut off = 0usize;
        while off < _bytes {
            // SAFETY: 同 readback_clean。
            unsafe { _mm_clflush(_ptr.add(off)) };
            off += 64;
        }
    }
}

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
        // staging 优先 HOST_CACHED：写合并（非缓存）型写入快但**主机读回
        // 慢 ~100×**（输出回读实测 3.3MB 吃 ~26ms）。缓存型读写都走 L3。
        let hv_hc = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let staging = (0..props.memory_type_count)
            .find(|&i| required(hv_hc | vk::MemoryPropertyFlags::HOST_CACHED, i))
            .or_else(|| {
                (0..props.memory_type_count).find(|&i| {
                    required(hv_hc, i)
                        && !props.memory_types[i as usize]
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::HOST_CACHED)
                })
            })
            .or_else(|| (0..props.memory_type_count).find(|&i| required(hv_hc, i)))
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

// SAFETY: Arena 携带 Vulkan 句柄（整数）与持久映射裸指针，跨线程仅经
// 会话的 Mutex 串行访问（与 Plan 的 unsafe Send 同一纪律）；无内部
// 可变状态逃逸。
unsafe impl Send for Arena {}

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

    /// 首块容量（池的淘汰比较用）。
    pub(crate) fn chunk_bytes(&self) -> vk::DeviceSize {
        self.chunks.first().map(|c| c.size).unwrap_or(0)
    }

    /// 游标归零复用：单块且容量 ≥ need 时清空复用（免掉 vkAllocateMemory
    /// 的页提交——137MB 新块实测 ~7ms，复用 ~0）。多块 arena 不复用。
    pub(crate) fn reset_if_fits(&mut self, need: vk::DeviceSize) -> bool {
        if self.chunks.len() == 1 && self.chunks[0].size >= need {
            self.cursor = 0;
            true
        } else {
            false
        }
    }

    /// 当前（最后一块）的缓冲与整块大小——KernelSet 绑定 SSBO 用。
    ///
    /// 已知边界（Phase F）：多块时只有最后一块可绑定。Phase G 的图
    /// 计划保证全部区域落在一块里（按计划总量一次开块）。
    pub(crate) fn chunk_range(&self) -> Option<(vk::Buffer, vk::DeviceSize)> {
        self.chunks.last().map(|c| (c.buffer, c.size))
    }

    /// 显式失效所有已映射块的主机视图（vkInvalidateMappedMemoryRanges）。
    ///
    /// 规范上 HOST_COHERENT 内存无需 invalidate；本机（Arc Pro +
    /// Windows 驱动）实测无效（见 [`readback_clean`]——真正的修复）。
    /// 保留给非 coherent 类型的未来路径，当前无调用方。
    #[allow(dead_code)]
    pub(crate) fn invalidate_mapped(&self) -> Result<()> {
        for c in &self.chunks {
            if c.mapped.is_null() {
                continue;
            }
            let range = vk::MappedMemoryRange::default()
                .memory(c.memory)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            // SAFETY: memory 属本设备且处于映射态；range 覆盖整块映射。
            unsafe { self.device.invalidate_mapped_memory_ranges(&[range]) }
                .map_err(|e| Error::Device(format!("invalidate_mapped_memory_ranges: {e}")))?;
        }
        Ok(())
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
