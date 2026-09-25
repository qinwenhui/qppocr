//! 流水线配置：调参基准 的 36 个行为参数，**数值一个不改**
//!。

/// 流水线配置（默认值 = 调参基准）。
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
    /// DB 二值化阈值。上游默认 0.5；0.2 是帕累托改进（exact 91.31→91.99%）。
    pub det_thresh: f32,
    /// 框内概率均值下限。
    pub box_thresh: f32,
    /// unclip 扩张比例。
    pub unclip_ratio: f32,
    /// unclip 垂直分量比例。⚠ 最敏感的一个。
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
    /// 批宽下限（px @ rec_height）。曾设 320，UI 截图上浪费 3.4 倍。
    pub rec_min_width: i32,
    /// 批宽对齐粒度，0 = 精确宽。
    pub rec_width_grain: i32,
    /// 补白到画布高度的比例下限，0 = 总是放大。
    pub rec_pad_min_h: f64,
    /// 每批行数（每批固定开销 ~8.6 ms）。
    pub rec_batch: usize,
    /// 批内最宽/最窄行比例上限（0 = 不限）。⚠ 不是越小越好（§6.2）。
    pub rec_batch_ratio: f64,
    /// 像素空格阈值（§6.4）。0.3 是唯一四份语料全不回退的档。
    pub rec_space_gap: f64,

    // ---- 方向分类 ----
    /// 分类器画布高。
    pub cls_height: i32,
    /// 分类器画布宽。
    pub cls_width: i32,
    /// 翻转阈值。
    pub cls_thresh: f32,
    /// 超宽行是否取居中窗口再分类（ 给转换版 cls 打的补丁：30:1 的
    /// 小长图压缩 7.6x 后分类器答错）。**对上游 PP-LCNet 有害**——倒置
    /// 英文行被窗口截断后特征不足会误判（实测 win=true 0.72 说正立、
    /// win=false 0.9998 说倒置）。默认关。
    pub cls_window: bool,

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
            rec_batch: 6,
            rec_batch_ratio: 1.1,
            rec_space_gap: 0.3,
            cls_height: 48,
            cls_width: 192,
            cls_thresh: 0.9,
            cls_window: false,
            merge_line_gap: 0.5,
            retry_conf: 0.85,
            enhance_contrast: false,
            upscale: 1,
            threads: 0,
        }
    }
}
