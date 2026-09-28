//! `qppocr` —— 纯 Rust、手写内核的 PP-OCRv6 推理引擎。
//!
//! 不依赖 ONNX Runtime、不依赖 tract、不依赖任何 C/ 库：ONNX 解析、
//! 图优化、算子、调度、检测/方向/识别流水线、后处理，全部自己实现。
//!
//! # 上手（三行）
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use qppocr::{Engine, Tier};
//!
//! let engine = Engine::new(Tier::Small, "models/")?;
//! let out = engine.run_image_file("receipt.png")?;
//! for line in &out.lines {
//!     println!("{:.2}\t{}", line.confidence, line.text);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # 定制
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use qppocr::{Engine, Preset, Tier};
//!
//! let engine = Engine::builder()
//!     .tier(Tier::Tiny)
//!     .preset(Preset::Speed)
//!     .threads(4)
//!     .build("models/")?;
//! # Ok(())
//! # }
//! ```
//!
//! # 模型
//!
//! 本仓库不含模型：PP-OCRv6 权重来自 PaddlePaddle（Apache-2.0），自行
//! 获取后按 [`ModelSource::Dir`] 的布局摆放；SHA-256 默认与官方原件
//! 核对。
//!
//! # 默认值与调参
//!
//! 默认值在我们的参考环境（x86 桌面机、商品图语料）上标定，开箱即用；
//! 但 CPU 代际、核数、图源（票据 / 扫描件 / 街景）都会移动最优点。
//! 常规项走 [`Config`]，更细的旋钮走 [`Advanced`]，字段注释写明各自
//! 影响什么——建议拿自己的机器和语料量一量再定。

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod image;
mod models;
mod sha256;

pub use qppocr_core::Error as CoreError;
pub use qppocr_core::pipeline::{Dictionary, OcrResult, TextLine, Timings};
/// CPU 内核后端探测（AVX2 / NEON / 标量），诊断与基准报告用。
pub use qppocr_core::{Backend, detect_backend};

pub use image::{Image, rgb_from_bytes};
// 解码 trio 只在 image-decode feature 下存在——不门控这行时，
// default-features=false 的下游整个 crate 编不过（外部用户实测报告）
#[cfg(feature = "image-decode")]
pub use image::{decode_bytes, decode_file, probe_dimensions};

/// 引擎线程池的总线程数（含主线程；`parallel` feature 关闭时为 1）。
/// 多进程批量部署按它折算每进程线程数（`thread_count / 进程数`）。
pub fn thread_count() -> usize {
    qppocr_core::thread_count()
}

pub use models::{ModelSource, Tier};

use qppocr_core::pipeline::Engine as CoreEngine;
/// 生效配置的完整快照（`Engine::config` 的返回类型；30 项全字段 pub，
/// 设置页回显/预设导出用；`serde` feature 开启时可序列化）。
pub use qppocr_core::pipeline::PipelineConfig;

/// 错误。库不该强制调用方的错误类型——不用 `anyhow`。
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// 模型装载失败（读不到、SHA 不符、字典缺失……错误信息带修复建议）。
    Model(String),
    /// 图像解码或尺寸不符。
    Image(String),
    /// 推理失败（不支持的算子、图不可解……）。
    Inference(CoreError),
    /// 设备不可用 / 不支持（显式要求 GPU 而编译期未启用或运行时无设备）。
    /// **不静默回退**到 CPU。
    Device(String),
    /// IO。
    Io(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Model(m) => write!(f, "model: {m}"),
            Error::Image(m) => write!(f, "image: {m}"),
            Error::Inference(e) => write!(f, "inference: {e}"),
            Error::Device(m) => write!(f, "device: {m}"),
            Error::Io(m) => write!(f, "io: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// 计算设备选择（[`EngineBuilder::device`]）。默认 CPU。
///
/// 默认值是起点不是结论：弱核显上 GPU 可能输给 AVX2 CPU，量产部署前
/// 拿目标机器量一量再定。
#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DeviceChoice {
    /// 本机 CPU。
    #[default]
    Cpu,
    /// GPU。`index` 选设备（`None` = 第一个可用）。
    Gpu {
        /// GPU API。
        api: GpuApi,
        /// 设备序号。
        index: Option<u32>,
    },
}

/// GPU API 选择。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GpuApi {
    /// 有什么用什么（当前即 Vulkan）。
    #[default]
    Auto,
    /// Vulkan 计算后端。
    Vulkan,
    /// CUDA 计算后端。
    Cuda,
}

impl From<CoreError> for Error {
    fn from(e: CoreError) -> Self {
        Error::Inference(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

/// 预设：三组常用取值，作为起点；在预设之上还可以用 [`Config`] /
/// [`Advanced`] 继续按环境调整。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum Preset {
    /// 速度优先：`rec_height` 40，关区域重试。
    Speed,
    /// 默认：全部字段取默认值。
    #[default]
    Balanced,
    /// 精度优先：`rec_height` 48，开区域重试与边距判定。
    Accuracy,
}

/// 公开配置。字段全部是 `Option`：`None` = 跟预设走，
/// 设了就**只覆盖这一项**，其余仍跟预设。
///
/// 这里放的是不看基准也能说明白取舍的项；更底层、需要结合自己负载
/// 测量的旋钮在 [`Advanced`]。
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[non_exhaustive]
pub struct Config {
    /// 识别画布高度（像素，32 的倍数）。低更快、高更准。
    pub rec_height: Option<u32>,
    /// 检测输入长边上限。
    pub det_max_side: Option<u32>,
    /// 原图长边预缩上限。0 = 不缩。
    pub max_side_len: Option<u32>,
    /// 0 = 自动（按可用核数）。
    pub threads: usize,
    /// 是否跑 0/180 方向分类（默认 **true**——倒置文本翻正是 OCR 引擎
    /// 的预期默认行为，对齐 PaddleOCR 与本 crate 的 CLI；无 cls.onnx
    /// 时该阶段自动跳过，开了也无害）。关掉即完全跳过该阶段（不加载、
    /// 不推理、`rotation` 恒 0）。
    pub detect_orientation: bool,
    /// 极端宽高比时上下补边（PP-OCR 的 `use_vertical_padding`）。
    pub vertical_padding: Option<bool>,
}

/// 引擎的底层调参项。默认值是我们的参考环境上取的折中，不是普适
/// 最优——不同 CPU、不同图源很可能有一组更好的值。字段注释写明各自
/// 影响什么、往哪边调会发生什么，按自己的环境调即可。
///
/// 经 [`EngineBuilder::advanced`] 进入：闭包收到的初始值是预设生效后
/// 的值，只改点名的项，其余保留。
///
/// 字段集随版本演进（`#[non_exhaustive]`），不做跨小版本的稳定性承诺。
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[non_exhaustive]
pub struct Advanced {
    /// DB 二值化阈值。上游默认 0.5；本引擎默认 0.2——低阈值多保小字与
    /// 断笔、多进杂块，按图源取舍。
    pub det_thresh: f32,
    /// 框内概率均值下限。
    pub box_thresh: f32,
    /// unclip 扩张比例。
    pub unclip_ratio: f32,
    /// unclip 垂直分量。对行高影响最直接——框整体偏高或偏低先调它。
    pub unclip_perp: f32,
    /// 边距杂波判定阈值。0 = 关。
    pub unclip_margin_thresh: f32,
    /// 区域重试触发阈值。0 = 关。
    pub retry_conf: f64,
    /// 同行框合并间隙（行高的倍数）。0 = 关。
    pub merge_line_gap: f64,
    /// 批内最宽/最窄行长宽比上限。
    pub rec_batch_ratio: f64,
    /// 像素空格阈值。
    pub rec_space_gap: f64,
    /// 方向分类翻转阈值。
    pub cls_thresh: f32,

    // ---- 以下与 `PipelineConfig` 同名同义 ----
    /// 检测输入长边上限。双向的：低于工作图长边时会把检测输入缩小
    /// （精度换速度）。
    pub det_max_side: i32,
    /// 检测输入像素上限（原图的倍数，只在补边时生效）。
    pub det_pixel_budget: f64,
    /// 检测短边目标（"min" limit type）。
    pub det_limit_side_len: i32,
    /// 原图预缩上限。
    pub max_side_len: i32,
    /// 连通域上限。
    pub max_candidates: usize,
    /// 连通域前是否膨胀 DB 掩码。
    pub use_dilation: bool,
    /// 极端宽高比时上下补黑边。
    pub vertical_padding: bool,
    /// 补边判定的宽高比阈值。
    pub width_height_ratio: f64,
    /// 补边判定的最小高度。
    pub min_height: i32,
    /// 识别画布高度（32 的倍数；换值即换 rec 输入分布）。
    pub rec_height: i32,
    /// 批宽下限（px @ rec_height）。
    pub rec_min_width: i32,
    /// 批宽对齐粒度，0 = 精确宽。
    pub rec_width_grain: i32,
    /// 补白代替放大的高度比例下限，0 = 总是放大。
    pub rec_pad_min_h: f64,
    /// 每批行数。
    pub rec_batch: usize,
    /// 分类器画布高。
    pub cls_height: i32,
    /// 分类器画布宽。
    pub cls_width: i32,
    /// 超宽行取居中窗口做方向分类（默认关：常规行被窗口截掉特征后，
    /// 方向判定会变差）。
    pub cls_window: bool,
    /// 一次方向分类前向塞几行（纯性能旋钮，输出不变）。
    pub cls_batch: usize,
    /// rec 外层分片数（两级并行）。`usize::MAX` = 按档位自动，
    /// `0` = 强制不分片（纯性能旋钮，输出不变）。
    pub rec_shards: usize,
    /// 自动色阶预处理，会改变检测器看到的输入（默认关）。
    pub enhance_contrast: bool,
    /// 检测放大倍数（成本 ~N²；裁剪仍取原图，默认 1）。
    pub upscale: i32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            rec_height: None,
            det_max_side: None,
            max_side_len: None,
            threads: 0,
            // 倒置翻正是预期默认（PaddleOCR 同款）；详见字段注释。
            detect_orientation: true,
            vertical_padding: None,
        }
    }
}

impl Default for Advanced {
    fn default() -> Self {
        // = PipelineConfig::default() 的对应字段
        Advanced {
            det_thresh: 0.2,
            box_thresh: 0.5,
            unclip_ratio: 1.6,
            unclip_perp: 1.0,
            unclip_margin_thresh: 0.0,
            // 区域重试默认关：它是应用层的质量增强（低置信区域重跑一遍
            // det），代价是触发图 det 翻倍（语料上 9/100 张）。要开用
            // Accuracy 预设或 Advanced::retry_conf。
            retry_conf: 0.0,
            merge_line_gap: 0.5,
            rec_batch_ratio: 1.1,
            rec_space_gap: 0.3,
            cls_thresh: 0.9,
            det_max_side: 960,
            det_pixel_budget: 6.0,
            det_limit_side_len: 736,
            max_side_len: 960,
            max_candidates: 1000,
            use_dilation: true,
            vertical_padding: true,
            width_height_ratio: 8.0,
            min_height: 30,
            rec_height: 48,
            rec_min_width: 16,
            rec_width_grain: 0,
            rec_pad_min_h: 0.0,
            // ⚠ 必须与 PipelineConfig::default() 一致：`.advanced(
            // Advanced::default())` 会**整份覆盖**，这里写别的值会让
            // 显式传默认 Advanced 的人悄悄换一套行为。
            rec_batch: 1,
            cls_height: 48,
            cls_width: 192,
            cls_window: false,
            cls_batch: 1,
            rec_shards: usize::MAX,
            enhance_contrast: false,
            upscale: 1,
        }
    }
}

/// 分片数的自动规则：识别网络**每层的算子规模**决定两级并行划不划算。
///
/// 实测（100 图语料、逐图交错）：小档每层 ~10M MAC，分片后每片只有
/// 2 个参与者，fork 的协调成本盖过收益——慢 6%；大档每层 ~100M MAC，
/// 分片快 10%。判据用档位而不是量出来的 MAC（构造期拿不到逐层形状），
/// 两者在本项目的三个档位上是一致的。
fn auto_rec_shards(tier: Tier, want: usize) -> usize {
    // 显式值原样透传（`usize::MAX` = 自动）。
    //
    // ⚠ 这里踩过两次坑，都记下来：
    //   1. `par::threads()` 会顺手按**默认布局**把池建起来，所以「先取线程数、
    //      再请求布局」会让布局请求永远失效（engine 侧已改用 planned_threads）。
    //   2. 这个函数一度被写成无条件 `return 0`，于是显式 `--rec-shards N`
    //      也被吞掉——连续几轮扫描全程都在测「不分片」，得出的「无差别」
    //      是假象。**改默认值时必须确认覆盖路径还通**（fork 调试打印是最快的
    //      验证：分片生效时能看到 slot=1..N、p.threads=m）。
    if want != usize::MAX {
        return want;
    }
    match tier {
        // 50 张图逐图交错的实测（中位，行阶段 = cls + rec）：
        //   tiny  0 片 51.2 / 8 片 44.2 / 10 片 41.4 / **12 片 40.6** / 14 片 40.7
        //   small 0 片 268.7 / 2 片 225.9 / 3 片 216.5 / **4 片 201.6** / 6 片 233.7
        // 即 tiny 1.26x、small 1.33x。两档的最优点不同：小档识别网络的算子
        // 小（~10M MAC），线程切细更划算；大档算子大（~100M MAC），每片给
        // 4 个线程才吃得饱。
        Tier::Tiny => 12,
        Tier::Small | Tier::Medium => 4,
    }
}

/// OCR 引擎。一次构造、多次运行；`&self` 并发安全（权重只读）。
pub struct Engine {
    core: CoreEngine,
    cfg: PipelineConfig,
    device: DeviceChoice,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // 摘要式：权重不属于 Debug 输出；config 是全部行为参数。
        f.debug_struct("Engine")
            .field("config", &self.cfg)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Engine {
    /// 最简构造：默认档（`Tier::Small` + `Preset::Balanced`）。
    ///
    /// `models_dir` 的布局见 [`ModelSource::Dir`]。
    pub fn new(tier: Tier, models_dir: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        Engine::builder().tier(tier).build(models_dir)
    }

    /// 定制构造的入口。
    pub fn builder() -> EngineBuilder {
        EngineBuilder::default()
    }

    /// 完整流水线：det → cls → rec；置信度弱时区域重试。
    pub fn run(&self, image: &Image) -> Result<OcrResult, Error> {
        self.core.run(image).map_err(Error::from)
    }

    /// 便利：解码文件 + [`Engine::run`]（feature `image-decode`）。
    #[cfg(feature = "image-decode")]
    pub fn run_image_file(&self, path: impl AsRef<std::path::Path>) -> Result<OcrResult, Error> {
        self.run(&decode_file(path)?)
    }

    /// 便利：解码字节 + [`Engine::run`]（feature `image-decode`）。
    #[cfg(feature = "image-decode")]
    pub fn run_image_bytes(&self, bytes: &[u8]) -> Result<OcrResult, Error> {
        self.run(&decode_bytes(bytes)?)
    }

    /// 只检测不识别（框带分数，无文本）。
    pub fn run_det_only(&self, image: &Image) -> Result<OcrResult, Error> {
        self.core.run_det_only(image).map_err(Error::from)
    }

    /// 生效配置的快照（预设 + 覆盖后的最终值）。
    pub fn config(&self) -> &PipelineConfig {
        &self.cfg
    }

    /// 实际运行设备（bench / 诊断报告用；`Auto` API 的解析结果在此回报）。
    pub fn device(&self) -> &DeviceChoice {
        &self.device
    }
}

/// Advanced 覆盖闭包的装箱类型（builder 内部）。
type AdvancedFn = Box<dyn FnOnce(&mut Advanced)>;

/// [`Engine`] 的 builder。
#[derive(Default)]
pub struct EngineBuilder {
    tier: Option<Tier>,
    preset: Option<Preset>,
    cfg: Config,
    advanced_fn: Option<AdvancedFn>,
    verify: Option<bool>,
    device: Option<DeviceChoice>,
}

impl EngineBuilder {
    /// 模型档位（默认 `Small`）。
    pub fn tier(mut self, tier: Tier) -> Self {
        self.tier = Some(tier);
        self
    }

    /// 预设（默认 `Balanced`）。
    pub fn preset(mut self, preset: Preset) -> Self {
        self.preset = Some(preset);
        self
    }

    /// 公开配置项（覆盖预设的对应项）。
    pub fn config(mut self, cfg: Config) -> Self {
        self.cfg = cfg;
        self
    }

    /// 修改 [`Advanced`] 常数。闭包收到的初始值是
    /// 预设生效后的值——只改你点名的项，其余保留。
    pub fn advanced(mut self, f: impl FnOnce(&mut Advanced) + 'static) -> Self {
        self.advanced_fn = Some(Box::new(f));
        self
    }

    /// SHA-256 校验开关（默认开）。自备重导出模型时关掉。
    pub fn verify_sha256(mut self, verify: bool) -> Self {
        self.verify = Some(verify);
        self
    }

    /// 快捷：线程数（0 = 自动）。等价 `config(Config { threads: n, .. })`。
    pub fn threads(mut self, n: usize) -> Self {
        self.cfg.threads = n;
        self
    }

    /// 快捷：是否跑方向分类。
    pub fn detect_orientation(mut self, on: bool) -> Self {
        self.cfg.detect_orientation = on;
        self
    }

    /// 计算设备（默认 CPU）。GPU 需要编译期启用 `gpu` feature；显式要
    /// GPU 而不可用时构造报错，**不静默回退** CPU。
    pub fn device(mut self, d: DeviceChoice) -> Self {
        self.device = Some(d);
        self
    }

    /// 构造引擎。
    pub fn build(self, models_dir: impl AsRef<std::path::Path>) -> Result<Engine, Error> {
        let source = ModelSource::Dir(models_dir.as_ref().to_path_buf());
        self.build_with(source)
    }

    /// 用显式 [`ModelSource`] 构造（WASM / 移动端从字节加载）。
    pub fn build_with(self, source: ModelSource) -> Result<Engine, Error> {
        let tier = self.tier.unwrap_or_default();
        let preset = self.preset.unwrap_or_default();
        // 设备先解析：GPU 不可用要在读模型字节之前报错（快速失败），
        // 不让「以为在跑 GPU」的人先等完权重加载才发现。
        let device = self.device.unwrap_or_default();
        let ctx = resolve_device(&device)?;
        let loaded = source.load(tier, self.verify.unwrap_or(true))?;
        let mut cfg = resolve_config(preset, &self.cfg, self.advanced_fn);
        if matches!(device, DeviceChoice::Gpu { .. }) {
            // GPU 会话不受益于宿主侧行扇出（提交在设备队列上串行化），
            // 分片只会增加线程 spawn 与争抢。pipeline 侧还有
            // `prefers_host_parallelism()` 运行时门控——这里是构造期
            // 第一道，cfg 快照也如实反映。
            cfg.rec_shards = 0;
        } else {
            cfg.rec_shards = auto_rec_shards(tier, cfg.rec_shards);
        }
        // 关方向分类 = 真不装配 cls（不加载、不推理）。曾经只把 cls_thresh
        // 设成 INFINITY：cls 照跑、时间照花、模型照占内存。
        let cls_bytes = if self.cfg.detect_orientation {
            loaded.cls.as_deref()
        } else {
            None
        };
        let core = CoreEngine::open_bytes_with(
            ctx.as_ref(),
            &loaded.det,
            &loaded.rec,
            cls_bytes,
            tier.label(),
            loaded.dict,
            cfg.clone(),
        )
        .map_err(Error::from)?;
        Ok(Engine { core, cfg, device })
    }
}

/// [`DeviceChoice`] → 设备上下文。
///
/// GPU 路径由 `gpu` feature（`qppocr-gpu` crate）提供，落地后在此接入；
/// 未编译进来时明确报错——静默回退 CPU 会让「我以为在跑 GPU」的基准
/// 数字全部作废。
fn resolve_device(
    choice: &DeviceChoice,
) -> Result<std::sync::Arc<dyn qppocr_core::device::DeviceContext>, Error> {
    match choice {
        DeviceChoice::Cpu => Ok(qppocr_core::device::cpu_context()),
        DeviceChoice::Gpu { .. } => Err(Error::Device(
            "GPU support not compiled in (enable feature \"gpu\")".into(),
        )),
    }
}

impl Tier {
    fn label(self) -> &'static str {
        match self {
            Tier::Tiny => "tiny",
            Tier::Small => "small",
            Tier::Medium => "medium",
        }
    }
}

/// 预设 → 公开覆盖 → Advanced 覆盖，得到最终 PipelineConfig。
fn resolve_config(preset: Preset, cfg: &Config, advanced_fn: Option<AdvancedFn>) -> PipelineConfig {
    // 1) 默认值
    let mut pc = PipelineConfig::default();
    // 2) 预设补丁
    match preset {
        Preset::Speed => {
            pc.rec_height = 40;
            pc.retry_conf = 0.0;
        }
        Preset::Balanced => {
            pc.rec_height = 48;
            // retry 继承默认（关）。曾经这里留 0.85，等于「默认模式背着
            // 一半用户开慢路径」——基准测试里更是白送 9/100 张的双倍 det。
        }
        Preset::Accuracy => {
            pc.rec_height = 48;
            pc.retry_conf = 0.85;
            // 边距杂波判定：实测难图 0.48 vs 语料 0.00–0.06，
            // 0.15 在两者之间留了余量（该值未做 sweep，依据见注释）
            pc.unclip_margin_thresh = 0.15;
        }
    }
    // 3) 公开覆盖
    if let Some(h) = cfg.rec_height {
        pc.rec_height = h as i32;
    }
    if let Some(s) = cfg.det_max_side {
        pc.det_max_side = s as i32;
    }
    if let Some(s) = cfg.max_side_len {
        pc.max_side_len = s as i32;
    }
    if cfg.threads > 0 {
        pc.threads = cfg.threads;
    }
    if !cfg.detect_orientation {
        // 双保险：cls 不装配（见 build_with），阈值同时设无穷大——
        // 直接用 core 的调用方只关阈值也能得到「永不翻转」。
        pc.cls_thresh = f32::INFINITY; // 永不翻转
    }
    if let Some(vp) = cfg.vertical_padding {
        pc.vertical_padding = vp;
    }
    // 4) Advanced 覆盖（初始值 = 预设后）
    let mut adv = Advanced {
        det_thresh: pc.det_thresh,
        box_thresh: pc.box_thresh,
        unclip_ratio: pc.unclip_ratio,
        unclip_perp: pc.unclip_perp,
        unclip_margin_thresh: pc.unclip_margin_thresh,
        retry_conf: pc.retry_conf,
        merge_line_gap: pc.merge_line_gap,
        rec_batch_ratio: pc.rec_batch_ratio,
        rec_space_gap: pc.rec_space_gap,
        cls_thresh: pc.cls_thresh,
        det_max_side: pc.det_max_side,
        det_pixel_budget: pc.det_pixel_budget,
        det_limit_side_len: pc.det_limit_side_len,
        max_side_len: pc.max_side_len,
        max_candidates: pc.max_candidates,
        use_dilation: pc.use_dilation,
        vertical_padding: pc.vertical_padding,
        width_height_ratio: pc.width_height_ratio,
        min_height: pc.min_height,
        rec_height: pc.rec_height,
        rec_min_width: pc.rec_min_width,
        rec_width_grain: pc.rec_width_grain,
        rec_pad_min_h: pc.rec_pad_min_h,
        rec_batch: pc.rec_batch,
        cls_height: pc.cls_height,
        cls_width: pc.cls_width,
        cls_window: pc.cls_window,
        cls_batch: pc.cls_batch,
        rec_shards: pc.rec_shards,
        enhance_contrast: pc.enhance_contrast,
        upscale: pc.upscale,
    };
    if let Some(f) = advanced_fn {
        f(&mut adv);
    }
    pc.det_thresh = adv.det_thresh;
    pc.box_thresh = adv.box_thresh;
    pc.unclip_ratio = adv.unclip_ratio;
    pc.unclip_perp = adv.unclip_perp;
    pc.unclip_margin_thresh = adv.unclip_margin_thresh;
    pc.retry_conf = adv.retry_conf;
    pc.merge_line_gap = adv.merge_line_gap;
    pc.rec_batch_ratio = adv.rec_batch_ratio;
    pc.rec_space_gap = adv.rec_space_gap;
    pc.cls_thresh = adv.cls_thresh;
    pc.det_max_side = adv.det_max_side;
    pc.det_pixel_budget = adv.det_pixel_budget;
    pc.det_limit_side_len = adv.det_limit_side_len;
    pc.max_side_len = adv.max_side_len;
    pc.max_candidates = adv.max_candidates;
    pc.use_dilation = adv.use_dilation;
    pc.vertical_padding = adv.vertical_padding;
    pc.width_height_ratio = adv.width_height_ratio;
    pc.min_height = adv.min_height;
    pc.rec_height = adv.rec_height;
    pc.rec_min_width = adv.rec_min_width;
    pc.rec_width_grain = adv.rec_width_grain;
    pc.rec_pad_min_h = adv.rec_pad_min_h;
    pc.rec_batch = adv.rec_batch;
    pc.cls_height = adv.cls_height;
    pc.cls_width = adv.cls_width;
    pc.cls_window = adv.cls_window;
    pc.cls_batch = adv.cls_batch;
    pc.rec_shards = adv.rec_shards;
    pc.enhance_contrast = adv.enhance_contrast;
    pc.upscale = adv.upscale;
    pc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 预设的实际解析结果必须与文档声明一致。区域重试曾在这里脱节：
    /// 0.2.0 宣称「默认关闭」，但 core 默认值是 0.85、Balanced 预设又
    /// 不清零——发布版实际默认开启，两个版本后才被发现。声明过的
    /// 默认行为一律进这条测试（「声明 = 断言」）。
    #[test]
    fn preset_contract_matches_docs() {
        let cases = [
            (Preset::Speed, 0.0f64, 40i32),
            (Preset::Balanced, 0.0, 48),
            (Preset::Accuracy, 0.85, 48),
        ];
        for (preset, retry, rec_h) in cases {
            let pc = resolve_config(preset, &Config::default(), None);
            assert_eq!(
                pc.retry_conf, retry,
                "{preset:?} 的区域重试默认与文档不符"
            );
            assert_eq!(pc.rec_height, rec_h, "{preset:?} 的 rec_height 与文档不符");
        }
        // Advanced::default() 的自注契约：「必须与 PipelineConfig::default()
        // 一致」——整份覆盖时不得悄悄换行为。
        assert_eq!(Advanced::default().retry_conf, 0.0);
    }
}
