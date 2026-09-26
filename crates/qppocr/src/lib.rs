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
//! # 为什么「默认值就是最优值」
//!
//! 上游默认 `det_thresh=0.5`，我们测出 `0.2` 更好（exact 91.31→91.99%）；
//! `rec_height=48` 是 sweep 的尖峰。这类结论沉淀在默认值里——
//! `cargo add qppocr` 后不配任何东西，拿到的就是最好的一档。

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod image;
mod models;
mod sha256;

pub use qppocr_core::Error as CoreError;
pub use qppocr_core::pipeline::{Dictionary, OcrResult, TextLine, Timings};

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
    /// IO。
    Io(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Model(m) => write!(f, "model: {m}"),
            Error::Image(m) => write!(f, "image: {m}"),
            Error::Inference(e) => write!(f, "inference: {e}"),
            Error::Io(m) => write!(f, "io: {m}"),
        }
    }
}

impl std::error::Error for Error {}

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

/// 预设：三个经过整段基准验证的档位。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum Preset {
    /// 速度优先：`rec_height` 40，关区域重试。
    Speed,
    /// 默认：全部基准值（36 参数的测量结果，）。
    #[default]
    Balanced,
    /// 精度优先：`rec_height` 48，开区域重试与边距判定。
    Accuracy,
}

/// 公开配置。字段全部是 `Option`：`None` = 跟预设走，
/// 设了就**只覆盖这一项**，其余仍跟预设。
///
/// 判据：用户不需要跑基准就能讲清楚该往哪边调吗？
/// 能 → 进这里；不能（要量具）→ [`Advanced`]。
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[non_exhaustive]
pub struct Config {
    /// 识别画布高度（像素）。40 更快、48 更准（sweep）。
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

/// 基准测出来的常数。**不要改**——每一项都有实测依据
/// 。
///
/// 保留它是为了让我们自己能继续调参，以及极少数有自己量具的用户一条路。
/// 这里的字段**不提供稳定性承诺**，任何一个小版本都可能变。
/// 只能经 [`EngineBuilder::advanced`] 进入。
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[non_exhaustive]
pub struct Advanced {
    /// DB 二值化阈值。上游 0.5；0.2 是帕累托改进。
    pub det_thresh: f32,
    /// 框内概率均值下限。
    pub box_thresh: f32,
    /// unclip 扩张比例。
    pub unclip_ratio: f32,
    /// unclip 垂直分量。⚠ 最敏感的一个。
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

    // ---- 以下与 `PipelineConfig` 同名同义（tuning.hpp 基准值）----
    /// 检测输入长边上限。⚠ 双向帽：低于工作图长边会把检测输入**缩小**。
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
    /// 超宽行取居中窗口分类。⚠ 对上游 PP-LCNet 有害（默认关）。
    pub cls_window: bool,
    /// 自动色阶预处理。⚠ 改变检测器所见（默认关）。
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
        // = PipelineConfig::default() 的对应字段（tuning.hpp 基准值）
        Advanced {
            det_thresh: 0.2,
            box_thresh: 0.5,
            unclip_ratio: 1.6,
            unclip_perp: 1.0,
            unclip_margin_thresh: 0.0,
            retry_conf: 0.85,
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
            rec_batch: 6,
            cls_height: 48,
            cls_width: 192,
            cls_window: false,
            enhance_contrast: false,
            upscale: 1,
        }
    }
}

/// OCR 引擎。一次构造、多次运行；`&self` 并发安全（权重只读）。
pub struct Engine {
    core: CoreEngine,
    cfg: PipelineConfig,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // 摘要式：权重不属于 Debug 输出；config 是全部行为参数。
        f.debug_struct("Engine")
            .field("config", &self.cfg)
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

    /// 构造引擎。
    pub fn build(self, models_dir: impl AsRef<std::path::Path>) -> Result<Engine, Error> {
        let source = ModelSource::Dir(models_dir.as_ref().to_path_buf());
        self.build_with(source)
    }

    /// 用显式 [`ModelSource`] 构造（WASM / 移动端从字节加载）。
    pub fn build_with(self, source: ModelSource) -> Result<Engine, Error> {
        let tier = self.tier.unwrap_or_default();
        let preset = self.preset.unwrap_or_default();
        let loaded = source.load(tier, self.verify.unwrap_or(true))?;
        let cfg = resolve_config(preset, &self.cfg, self.advanced_fn);
        // 关方向分类 = 真不装配 cls（不加载、不推理）。曾经只把 cls_thresh
        // 设成 INFINITY：cls 照跑、时间照花、模型照占内存。
        let cls_bytes = if self.cfg.detect_orientation {
            loaded.cls.as_deref()
        } else {
            None
        };
        let core = CoreEngine::open_bytes(
            &loaded.det,
            &loaded.rec,
            cls_bytes,
            tier.label(),
            loaded.dict,
            cfg.clone(),
        )
        .map_err(Error::from)?;
        Ok(Engine { core, cfg })
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
    // 1) 基准值（tuning.hpp，）
    let mut pc = PipelineConfig::default();
    // 2) 预设补丁
    match preset {
        Preset::Speed => {
            pc.rec_height = 40;
            pc.retry_conf = 0.0;
        }
        Preset::Balanced => {
            pc.rec_height = 48;
        }
        Preset::Accuracy => {
            pc.rec_height = 48;
            pc.retry_conf = 0.85;
            // 边距杂波判定：实测难图 0.48 vs 语料 0.00–0.06，
            // 0.15 在两者之间留了余量（值无基准 sweep，注释在案）
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
    pc.enhance_contrast = adv.enhance_contrast;
    pc.upscale = adv.upscale;
    pc
}
