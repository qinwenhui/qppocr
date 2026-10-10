//! 卷积内核的**变体二分**（仅测试）。
//!
//! ## 为什么要有这一层
//!
//! `conv8_throughput` 量出 3×3 conv 只有 ~80–190 GFLOPS，而 M4 的 fp32
//! 峰值约 4 TFLOPS——差 20 倍。光看数字猜不出是访存没合并、寄存器压力、
//! 还是装载/算数指令比失衡。这一层把 `n_conv8` 的热循环抄成几个**只差
//! 一个变量**的变体，让量到的差异能干净地归因到那一个变量。
//!
//! ## 变体表
//!
//! | 名字 | 与 `n_conv8` 的差 | 结果正确？ | 要回答的问题 |
//! |---|---|---|---|
//! | `p_base` | 无（逐行抄） | 是 | 量法本身可信吗 |
//! | `p_r8` | 行 4→8，仍 n8 | 是 | 加大行分块、摊薄权重装载值多少 |
//! | `p_nocoal` | 输入地址 → 常量 0 | **否**（诊断） | 输入 gather 的分散访存占多少 |
//! | `p_fmaonly` | 访存全砍，只留 FMA | **否**（诊断） | 卡在「算不动」还是「喂不上」 |
//! | `p_wconst` | 权重装载 → 常量 | **否**（诊断） | 权重流量占多少（与上一条夹逼） |
//! | `p_m8n4` | 行 4→8 且 通道块 2→1 | 是 | 同寄存器预算下把权重复用翻倍值多少 |
//! | `p_wsmall` | 权重地址 → 1 KB 窗口 | **否**（诊断） | 卡在权重**流量**还是权重**装载指令** |
//! | `p_ismall` | **激活读**地址 → 1 KB 窗口 | **否**（诊断） | 激活**流量**值不值（fp16 激活该不该做）——`p_wsmall` 的激活侧对照 |
//! | `p_t4n2`…`p_t16n2` | 模板化的 ROWS×NBLK（网格不变） | 是 | 瓦片空间里最优点在哪（仅 1×1） |
//! | `p_k1fast` | k1 同分辨率时把 gather 地址提出热循环 | 是 | 输入侧那 3 倍是**流量**还是**装载指令+地址算术** |
//! | `p_window` | 只改 gid 映射：同片权重的行组铺到一起 | 是 | 复用距离（9 MB → ~123 KB）值多少 |
//! | `p_colmaj` | 只改 gid 映射：warp 内相邻 lane 通道连续 | 是 | 一条权重装载横跨几条 cache line 值多少 |
//! | `p_colmajw` | 窗口序 + 窗内列主序 | **否**（诊断） | 两个映射能否叠成一条路、免掉按尺寸切阈值 |
//! | `p_colmajsmall` | 列主序 + 权重地址掩进 1 KB | **否**（诊断） | **`p_wsmall` 的 1.56× 余量被列主序吃掉了吗**（决定任务 #12 做不做） |
//! | `p_wstage` | 列主序 + 权重切片协作搬进线程组内存 | **是**（应位级等于 `p_colmaj`） | 任务 #12 的可落地处方到底值多少 |
//! | `p_whalf` | 权重存 fp16（其余一字不改） | 否（fp16 舍入） | **权重足迹砍半值多少**——决定 fp16 权重这条路做不做 |
//!
//! `p_wsmall` 是这组里最要紧的一条：`p_base` 与 `p_wconst` 的差（83%）既
//! 可能是「权重流量」（→ 线程组暂存能救）也可能是「权重装载指令条数」
//! （→ 只能减少条数，暂存白搭）。它保留条数与地址算术、只把地址掩进 1 KB
//! 窗口，于是两个假设被劈开。
//!
//! `p_nocoal` 与 `p_fmaonly` 故意算错，**只看时间**：前者把每线程的输入
//! 地址钉死到第 0 个位置（同组 lane 读同一地址 → 广播，1 个事务），后者
//! 把输入与权重全换成循环变量算出的常量，只留算数。
//!
//! ## 为什么是「逐行抄」而不是模板
//!
//! 先写过一版带占位符的模板（`acc[ROWS][2]` 数组 + 双层展开循环）。
//! 结果 `p_base` 在同一形状上比 `n_conv8` **慢 1.6 倍**（9.84 ms vs
//! 5.98 ms）——数组形式没被完全标量化，落到了线程私有内存。控制组都
//! 对不上，变体差异就没法归因。所以这里是显式的标量累加器，和
//! `n_conv8` 逐字一致。
//!
//! ## 纪律
//!
//! 这些内核**不在** `KernelSet::SHADERS` 清单里，只在本模块编译、
//! 只被测试调用，不会进任何生产计划。

use std::collections::HashMap;

use metal::ComputePipelineState;
use qppocr_core::error::{Error, Result};

/// 变体：内核名 + 网格用的 (行数, 每线程通道块数)——`groups_for` 要按它
/// 算覆盖。
///
/// `rows`/`nblk` 描述的是**网格**（线程覆盖），不是内核内部怎么分块。
/// 对 `p_t*` 这几个瓦片探针，两者是**故意分开的**：网格一律用计划给的
/// m4n8 公式（见 `grid_of`），内核内部换瓦片——这样量到的差异才是
/// 「生产里换个瓦片」的收益。
#[derive(Clone, Copy)]
pub(crate) struct Variant {
    pub name: &'static str,
    pub rows: u32,
    /// 每线程持有的输出通道块数（每块 4 个通道）。m4×n8 是 2，m8×n4 是 1。
    pub nblk: u32,
    /// 内核内部的寄存器瓦片（`行×列`），仅用于打印归因。
    pub tile: &'static str,
    /// 只对 1×1 卷积有意义（3×3 上实测更慢）——扫描时按形状**跳过**
    /// 而不是照跑（照跑会早退成 0.002 ms，看着像三千倍加速）。
    pub k1_only: bool,
    /// `n_conv8`（控制组）才要求网格与计划逐位一致。
    pub is_control: bool,
}

/// 全部变体（顺序即打印顺序）。
pub(crate) const VARIANTS: &[Variant] = &[
    Variant { name: "p_base", rows: 4, nblk: 2, tile: "m4×n2", k1_only: false, is_control: true },
    Variant { name: "p_r8", rows: 8, nblk: 2, tile: "m8×n2", k1_only: false, is_control: false },
    Variant { name: "p_nocoal", rows: 4, nblk: 2, tile: "m4×n2", k1_only: false, is_control: false },
    Variant { name: "p_fmaonly", rows: 4, nblk: 2, tile: "m4×n2", k1_only: false, is_control: false },
    Variant { name: "p_wconst", rows: 4, nblk: 2, tile: "m4×n2", k1_only: false, is_control: false },
    Variant { name: "p_m8n4", rows: 8, nblk: 1, tile: "m8×n1", k1_only: false, is_control: false },
    // 诊断：权重地址掩进 1 KB 窗口（条数不变、流量≈0）——把「带宽」与
    // 「装载指令」两个假设分开，见 probe_p_wsmall.metal 的头注释。
    Variant { name: "p_wsmall", rows: 4, nblk: 2, tile: "m4×n2", k1_only: false, is_control: false },
    // 诊断：**激活**地址掩进 1 KB 窗口（条数不变、流量≈0）——`p_wsmall`
    // 只管权重那半边，这条补上激活那半边，见 probe_p_ismall.metal 头注释。
    // 要回答的是「把激活也改成 fp16（足迹砍半）值不值」。
    Variant { name: "p_ismall", rows: 4, nblk: 2, tile: "m4×n2", k1_only: false, is_control: false },
    // 窗口序：只改 gid 映射，把「同一片权重的行组」铺到一起，复用距离
    // 9 MB → ~123 KB。热循环、网格、覆盖全与 p_base 一致。
    Variant { name: "p_window", rows: 4, nblk: 2, tile: "m4×n2", k1_only: false, is_control: false },
    Variant { name: "p_window8", rows: 8, nblk: 1, tile: "m8×n1", k1_only: false, is_control: false },
    // 窗口序 + 窗内列主序（见 probe_p_colmajw.metal 头注释）：窗口序管权重
    // 复用距离、列主序管 lane 排布。若它 ≥ max(p_window, p_colmaj)，
    // 生产就不必按权重尺寸切两套映射。只对 1×1 有意义（窗口序的门）。
    Variant { name: "p_colmajw", rows: 4, nblk: 2, tile: "m4×n2", k1_only: true, is_control: false },
    // 列主序 warp 排布：相邻 lane 的通道连续（一条装载 = 4 条 line 填满，
    // 而不是 stride-2 的 8 条 half-used）。指令条数、网格、覆盖全同
    // `p_base`，见 probe_p_colmaj.metal 的头注释。
    Variant { name: "p_colmaj", rows: 4, nblk: 2, tile: "m4×n2", k1_only: false, is_control: false },
    // 诊断：列主序 + 权重地址掩进 1 KB（见 probe_p_colmajsmall.metal）。
    // 要回答的是「p_wsmall 那 1.56× 的余量被列主序吃掉了吗」——决定任务
    // #12（线程组暂存权重切片）到底做不做。与 p_colmaj 同形，便于直接读差。
    Variant { name: "p_colmajsmall", rows: 4, nblk: 2, tile: "m4×n2", k1_only: false, is_control: false },
    // 线程组暂存权重切片（见 probe_p_wstage.metal）。只对 1×1 开：3×3 要走
    // 9 遍 taps、每遍一对屏障，不是这条路的靶子。**位级应等于 p_colmaj**。
    Variant { name: "p_wstage", rows: 4, nblk: 2, tile: "m4×n2", k1_only: true, is_control: false },
    // 诊断：同上但**删掉屏障**（故意算错）。把「屏障的代价」与「组内存占用率
    // 的代价」劈开——见 probe_p_wstage_nb.metal 的头注释。
    Variant { name: "p_wstage_nb", rows: 4, nblk: 2, tile: "m4×n2", k1_only: true, is_control: false },
    // k1 同分辨率快路：gather 地址整体提出热循环（见 probe_p_k1fast.metal）。
    // 与 p_base 的**唯一**差别是那个 `if`——累加顺序逐字相同，结果位级一致。
    Variant { name: "p_k1fast", rows: 4, nblk: 2, tile: "m4×n2", k1_only: true, is_control: false },
    // 权重存 fp16（见 probe_p_whalf.metal 头注释）：与 p_base 的唯一语义差
    // 是权重基址 `>>2` → `>>1`，装载条数/网格/热循环全同。要回答的是
    // 「§4 认定的 line 数/局部性」这根杠杆上，fp16 值多少——**在动生产
    // 代码之前**把这条路的价值量出来。
    Variant { name: "p_whalf", rows: 4, nblk: 2, tile: "m4×n2", k1_only: false, is_control: false },
    // 瓦片空间（网格一律 = 计划的 m4n8；`p_t4n2` 是模板自身的控制组，
    // 它必须与 `p_base` 同时间——对不上就是模板没把累加器标量化）。
    Variant { name: "p_t4n2", rows: 4, nblk: 2, tile: "m4×n2", k1_only: true, is_control: false },
    Variant { name: "p_t8n1", rows: 4, nblk: 2, tile: "m8×n1", k1_only: true, is_control: false },
    Variant { name: "p_t8n2", rows: 4, nblk: 2, tile: "m8×n2", k1_only: true, is_control: false },
    Variant { name: "p_t16n1", rows: 4, nblk: 2, tile: "m16×n1", k1_only: true, is_control: false },
    Variant { name: "p_t16n2", rows: 4, nblk: 2, tile: "m16×n2", k1_only: true, is_control: false },
];

/// 变体源码。公共头部（参数解析 + 索引推导 + act 语义）与
/// `n_conv8.metal` 一致；差异只在 gather 与累加那几行，逐处标了注释。
///
/// `p_peak4`/`p_peak16` 不参与变体扫描——它们不是卷积，是**算力标尺**
/// （见 `fma_peak_reference`），参数块语义也不同（只有 `params[p_off]`
/// = 外层迭代数）。
const PROBE_SHADERS: &[(&str, &str)] = &[
    ("p_base", include_str!("probe_p_base.metal")),
    ("p_r8", include_str!("probe_p_r8.metal")),
    ("p_nocoal", include_str!("probe_p_nocoal.metal")),
    ("p_fmaonly", include_str!("probe_p_fmaonly.metal")),
    ("p_wconst", include_str!("probe_p_wconst.metal")),
    ("p_m8n4", include_str!("probe_p_m8n4.metal")),
    ("p_wsmall", include_str!("probe_p_wsmall.metal")),
    ("p_ismall", include_str!("probe_p_ismall.metal")),
    ("p_window", include_str!("probe_p_window.metal")),
    ("p_window8", include_str!("probe_p_window8.metal")),
    ("p_colmajw", include_str!("probe_p_colmajw.metal")),
    ("p_colmaj", include_str!("probe_p_colmaj.metal")),
    ("p_colmajsmall", include_str!("probe_p_colmajsmall.metal")),
    ("p_wstage", include_str!("probe_p_wstage.metal")),
    ("p_wstage_nb", include_str!("probe_p_wstage_nb.metal")),
    ("p_k1fast", include_str!("probe_p_k1fast.metal")),
    ("p_whalf", include_str!("probe_p_whalf.metal")),
    // 同一份源码被列多次：每个 (名字, 源码) 是独立一次编译、只取该名字
    // 对应的入口——模板化的瓦片探针就是这样一次源码、五个内核。
    ("p_t4n2", include_str!("probe_p_tile.metal")),
    ("p_t8n1", include_str!("probe_p_tile.metal")),
    ("p_t8n2", include_str!("probe_p_tile.metal")),
    ("p_t16n1", include_str!("probe_p_tile.metal")),
    ("p_t16n2", include_str!("probe_p_tile.metal")),
    ("p_peak4", include_str!("probe_p_peak.metal")),
    ("p_peak16", include_str!("probe_p_peak.metal")),
    // `n_conv8_mma` 的四臂诊断（同一份源码、四个入口）——前三个劈开热循环
    // 那 25% 的缺口是「装载指令」还是「流量/延迟」，第四个给后半一条**能
    // 落地**的处方（权重离线重排成分片连续）。见该文件头注释。
    ("p_mma_base", include_str!("probe_mma_split.metal")),
    ("p_mma_small", include_str!("probe_mma_split.metal")),
    ("p_mma_hoist", include_str!("probe_mma_split.metal")),
    ("p_mma_wfrag", include_str!("probe_mma_split.metal")),
    ("p_mma_asmall", include_str!("probe_mma_split.metal")),
    ("p_mma_bsmall", include_str!("probe_mma_split.metal")),
    ("p_mma_ost16", include_str!("probe_mma_split.metal")),
    ("p_mma_ost2", include_str!("probe_mma_split.metal")),
];

/// 变体内核的管线集（一次编译，反复用）。
pub(crate) struct ProbeSet {
    pipes: HashMap<&'static str, ComputePipelineState>,
}

/// depthwise 探针（`probe_dw.metal`，见该文件的头注释）。与 conv8 那组
/// 分开是因为参数块语义不同（`n_conv_dw` 的 `params[P+9]` 是 taps 不是
/// cin4v），混进一张表会让 `groups_for` 的网格公式对错对象。
pub(crate) const DW_PROBE_SHADERS: &[(&str, &str)] = &[
    ("p_dw_base", include_str!("probe_dw.metal")),
    ("p_dw_nodiv", include_str!("probe_dw.metal")),
    ("p_dw_nobc", include_str!("probe_dw.metal")),
    ("p_dw_now", include_str!("probe_dw.metal")),
    ("p_dw_fma", include_str!("probe_dw.metal")),
    // 4 个通道块 / 线程（16 通道）。**网格与其他变体不同**，见
    // `dw_groups_for_wide`；调用方必须按变体选网格。
    ("p_dw_wide4", include_str!("probe_dw.metal")),
    ("p_dw_noin", include_str!("probe_dw.metal")),
    ("p_dw_win", include_str!("probe_dw.metal")),
    // 全展开特化（编译期常量 KH/KW/pad）。**结果正确**，可过 CPU 对拍；
    // 各只在匹配的形状上有意义（`dw_spec_ok` 判）。
    ("p_dw_u9", include_str!("probe_dw.metal")),
    ("p_dw_u3", include_str!("probe_dw.metal")),
];

/// simdgroup_matrix 探针（`probe_mm_sg.metal`：纯 GEMM，竞品结构；`probe_mm_sg_conv8.metal`：适配 conv8 形状）。
///
/// **单开一张表**：参数块语义（`p[0..3] = M,N,K`）与网格（64×64 瓦片）
/// 都和 conv8 那组无关，混进去 `groups_for` 会给出错的网格。
pub(crate) const MM_PROBE_SHADERS: &[(&str, &str)] =
    &[("p_mm_sg", include_str!("probe_mm_sg.metal")),
      ("p_mm_sg_conv8", include_str!("probe_mm_sg_conv8.metal"))];

/// 64×64 瓦片的网格：`[ceil(M/64), ceil(N/64), 1]`，线程组固定 256。
pub(crate) fn mm_sg_groups(m_pad: u32, n_pad: u32) -> [u32; 3] {
    [m_pad.div_ceil(64), n_pad.div_ceil(64), 1]
}

/// `p_dw_u9`/`p_dw_u3` 只对「参数与其模板常量逐位相符」的形状成立 ——
/// 拿 u9 去跑 3×3 会读到常量算出的错地址，输出全错。
pub(crate) fn dw_spec_ok(name: &str, kh: u32, kw: u32, pad: u32, stride: u32) -> bool {
    match name {
        "p_dw_u9" => (kh, kw, pad, stride) == (9, 9, 4, 1),
        "p_dw_u3" => (kh, kw, pad, stride) == (3, 3, 1, 1),
        _ => true,
    }
}

impl ProbeSet {
    /// 逐个变体独立编译成一个库——编译失败时错误信息里带得动内核名，
    /// 和 `KernelSet` 同一纪律。
    pub(crate) fn new(device: &metal::DeviceRef) -> Result<Self> {
        Self::from_list(device, PROBE_SHADERS)
    }

    /// 同 `new`，但编译指定的表（depthwise 那组用 `DW_PROBE_SHADERS`）。
    pub(crate) fn from_list(
        device: &metal::DeviceRef,
        shaders: &[(&'static str, &'static str)],
    ) -> Result<Self> {
        let opts = metal::CompileOptions::new();
        // 与生产侧同开关：关 fast-math，保住运算语义可比。
        opts.set_fast_math_enabled(false);
        let mut pipes = HashMap::new();
        for (name, src) in shaders {
            let lib = device
                .new_library_with_source(src, &opts)
                .map_err(|e| Error::Device(format!("探针 MSL 编译失败（{name}）: {e}")))?;
            let func = lib
                .get_function(name, None)
                .map_err(|e| Error::Device(format!("取探针函数 {name} 失败: {e}")))?;
            let pipe = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| Error::Device(format!("建探针管线（{name}）失败: {e}")))?;
            pipes.insert(*name, pipe);
        }
        Ok(Self { pipes })
    }

    pub(crate) fn pipe(&self, name: &str) -> Result<&ComputePipelineState> {
        self.pipes
            .get(name)
            .ok_or_else(|| Error::Device(format!("探针 {name} 不在 ProbeSet 里")))
    }
}

/// 变体在给定形状下的网格（工作组数）。公式必须与对应内核里 `tiles` 的
/// 算法一致，否则线程覆盖不全，量出来的时间就是假的。
///
/// `nblk` = 每线程吃掉几个通道块，故一行 tile 数 = `nv / nblk`。
pub(crate) fn groups_for(m_dim: u32, co4: u32, rows: u32, nblk: u32) -> [u32; 3] {
    let nv = co4.div_ceil(4);
    let tiles = m_dim.div_ceil(rows) * (nv / nblk);
    [tiles.div_ceil(256), 1, 1]
}

/// `n_conv_dw` 的网格（工作组数）：一线程一个 (m, c4)，故线程数
/// = `m_dim * (cDim/4)`。必须与 `plan.rs` 里 `div256(m_dim * cp / 4)`
/// 逐字一致（`cp` = `cpad4(Co)`，已含在 `c_dim` 里）。
pub(crate) fn dw_groups_for(m_dim: u32, c_dim: u32) -> [u32; 3] {
    [(m_dim * (c_dim / 4)).div_ceil(256), 1, 1]
}

/// `p_dw_wide4` 的网格：一线程 4 个通道块，故线程数 = `mDim * cv4 / 4`。
pub(crate) fn dw_groups_for_wide(m_dim: u32, c_dim: u32) -> [u32; 3] {
    [(m_dim * (c_dim / 16)).div_ceil(256), 1, 1]
}
