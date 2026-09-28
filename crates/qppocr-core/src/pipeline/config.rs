//! 流水线配置：det / rec / cls 的全部行为参数与默认值。

/// 流水线配置：det / rec / cls 的全部行为参数。
///
/// 默认值在本项目的参考环境（x86 桌面机、商品图语料）上标定，作为
/// 出厂起点；字段注释说明各自影响与取舍，按目标环境调整即可。
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PipelineConfig {
    // ---- 检测 ----
    /// 检测输入长边上限，0 = 不限。
    pub det_max_side: i32,
    /// 检测输入像素不超过原图多少倍（只在补边发生时生效）。
    pub det_pixel_budget: f64,
    /// 短边规则目标（"min" limit type）。
    pub det_limit_side_len: i32,
    /// 原图预缩上限，0 = 不缩。
    pub max_side_len: i32,
    /// DB 二值化阈值。上游默认 0.5；本引擎默认 0.2——低阈值多保小字与
    /// 断笔、多进杂块，按图源取舍。
    pub det_thresh: f32,
    /// 框内概率均值下限。
    pub box_thresh: f32,
    /// unclip 扩张比例。
    pub unclip_ratio: f32,
    /// unclip 垂直分量比例，对行高影响最直接的一项。
    pub unclip_perp: f32,
    /// 边距杂波判定阈值，0 = 关。
    pub unclip_margin_thresh: f32,
    /// 连通域上限。
    pub max_candidates: usize,
    /// 连通域前是否膨胀 DB 掩码。
    pub use_dilation: bool,
    /// 极端宽高比时上下补黑边（PP-OCR 的 use_vertical_padding）。
    pub vertical_padding: bool,
    /// 宽高比超过它才补边。
    pub width_height_ratio: f64,
    /// 最小高度。
    pub min_height: i32,

    // ---- 识别 ----
    /// 识别画布高度。
    pub rec_height: i32,
    /// 批宽下限（px @ rec_height）。
    pub rec_min_width: i32,
    /// 批宽对齐粒度，0 = 精确宽。
    pub rec_width_grain: i32,
    /// 补白到画布高度的比例下限，0 = 总是放大。
    pub rec_pad_min_h: f64,
    /// 识别**一批最多几行**。扇出是「一批一个线程、批内串行」，
    /// 所以这个值只在**行数远多于线程数**时才有意义（`per_batch =
    /// min(rec_batch, ceil(行数/线程数))`）——行数不超线程数时恒为
    /// 「一行一批」，它和 `rec_batch_ratio` 都不生效。
    ///
    /// 默认 1（纯逐行）：批宽取批内最宽行，行高不齐的语料批起来既有
    /// padding 浪费又损失并行度。行数远超线程数的批量场景可以调大试试。
    pub rec_batch: usize,
    /// 批内最宽/最窄行比例上限（0 = 不限）。收紧减少 padding，过紧会把
    /// 行拆成更多小批。
    pub rec_batch_ratio: f64,
    /// 像素空格阈值。默认 0.3，按图源的字距密集程度调整。
    pub rec_space_gap: f64,

    // ---- 方向分类 ----
    /// 分类器画布高。
    pub cls_height: i32,
    /// 分类器画布宽。
    pub cls_width: i32,
    /// 翻转阈值。
    pub cls_thresh: f32,
    /// 超宽行是否取居中窗口再分类（默认关）。针对 30:1 的窄长图：整行压
    /// 到 192 px 宽时压缩严重、方向判不准；开窗后常规行又可能被截掉特征，
    /// 倒置行的判定变差——按图源里窄长图的占比权衡。
    pub cls_window: bool,
    /// 一次方向分类前向塞几行（批数 = `ceil(行数 / cls_batch)`）。
    /// cls 画布固定 48×192、单行前向很小，批起来省的固定开销通常盖不过
    /// 批间并行度的损失——在我们的参考机上 1 最快。行数远超线程数的
    /// 场景可以自己量一量再定。
    pub cls_batch: usize,

    // ---- 框合并 / 重试 ----
    /// 同行相邻框合并间隙（行高的倍数），0 = 关。凸包拟合（§5.6）。
    pub merge_line_gap: f64,
    /// 区域重试触发阈值（最弱行置信度），0 = 关。
    pub retry_conf: f64,

    // ---- 预处理 ----
    /// 自动色阶（默认关——它改变检测器看到的东西）。
    pub enhance_contrast: bool,
    /// 检测放大倍数（成本 ~N²，默认 1）。
    pub upscale: i32,

    /// rec 阶段的**分片数**（两级并行的外层宽度）。
    ///
    /// 分片 = 外层若干线程各自负责若干行，每行一个**独立小池**，其算子
    /// 再在该池内 fork。单池做不到这一点：`fork_mu` 会把并发 fork 串起来，
    /// 外层线程互相排队。池数 = 分片数，池大小 = 总线程数 / 分片数。
    ///
    /// - `usize::MAX`（默认）= **按档位自动**：识别网络每层的算子规模够大
    ///   分片才划算（小档算子小、fork 协调成本占比高；大档反之）。
    ///   自动规则见 facade 的 `auto_rec_shards`，也可显式指定后实测。
    /// - `0` = 强制不分片。`n` = 强制 n 片。
    pub rec_shards: usize,

    /// 线程数，0 = 自动。
    pub threads: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            det_max_side: 960,
            det_pixel_budget: 6.0,
            det_limit_side_len: 736,
            max_side_len: 960,
            det_thresh: 0.2,
            box_thresh: 0.5,
            unclip_ratio: 1.6,
            unclip_perp: 1.0,
            unclip_margin_thresh: 0.0,
            max_candidates: 1000,
            use_dilation: true,
            vertical_padding: true,
            width_height_ratio: 8.0,
            min_height: 30,
            rec_height: 48,
            rec_min_width: 16,
            rec_width_grain: 0,
            rec_pad_min_h: 0.0,
            rec_batch: 1,
            rec_batch_ratio: 1.1,
            rec_space_gap: 0.3,
            cls_height: 48,
            cls_width: 192,
            cls_thresh: 0.9,
            cls_window: false,
            cls_batch: 1,
            merge_line_gap: 0.5,
            retry_conf: 0.85,
            enhance_contrast: false,
            upscale: 1,
            rec_shards: usize::MAX,
            threads: 0,
        }
    }
}
