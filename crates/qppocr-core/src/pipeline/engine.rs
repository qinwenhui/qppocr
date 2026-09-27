//! OCR 引擎（OCR 引擎主流程）：det → cls → rec 主流程。
//!
//! ★ 字典是一等输入：上游 rec 模型不带内嵌字典
//!（medium 一个元数据都没有），所以 [`Engine::open`] 接受
//! 「从模型 metadata 取」或「显式给一份」两种来源，取不到时报错并
//! 指出该档位字典的行数——不静默降级。

use crate::error::{Error, Result};
use crate::executor::Session;
use crate::tensor::{DType, Tensor};

use super::config::PipelineConfig;
use super::crop::{batch_ratio, cls_view, crop_pads, crop_text_box, pack_crop};
use super::geometry::{box_margin_clutter, db_postprocess, merge_same_line, sort_reading_order};
use super::image::{Image, resize_bilinear_img, rotate_image};
use super::rec::{add_gap_spaces, column_ink, ctc_decode};
use qppocr_kernels::buf::F32Buf;

/// 单个字符的坐标（[`TextLine::chars`] 的元素）。
///
/// `text` 恰好一个字符（含像素空格判定的空格——它的框是裁剪图上
/// **真实空白 run** 的区间，不是借用相邻字符的边界）。
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CharSpan {
    /// 字符本身（单 char）。
    pub text: String,
    /// 字符的四角点 TL/TR/BR/BL，**原图坐标**（与 [`TextLine::pts`] 同一
    /// 约定、同一坐标系）。
    pub pts: [[f32; 2]; 4],
}

/// 识别结果的一行。
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextLine {
    /// 解码文本（可能为空 = 检测到但读不出；**保留**而不是丢弃——
    /// 丢掉会让「检测器打散了这行」伪装成完整结果）。
    pub text: String,
    /// 逐字坐标（与 `text` 逐字符对齐；空文本为空 vec）。来自 CTC 时间步
    /// → 裁剪图 x → 四边形角点插值的映射（PaddleOCR `return_word_box`
    /// 同款近似：角点精确、内部沿边线性）。已穿过全部坐标变换链：
    /// 长边帽缩放/补边/det 输入缩放/透视裁剪/竖条转正/cls 180° 翻转/
    /// 区域重试偏移——**坐标永远在原图空间**。
    pub chars: Vec<CharSpan>,
    /// CTC 置信度。
    pub confidence: f32,
    /// 0 或 180：分类器判定倒置并翻正过就是 180。
    pub rotation: i32,
    /// 本行文本来自**区域重试**的第二遍（首遍弱行被整体替换，见
    /// `retry_conf`）；false = 首遍直接读出。下游用它做逐行「重读过」
    /// 标识——`num_det_retried` 只计整图次数，分不清是哪几行。
    pub retried: bool,
    /// 四角点 TL/TR/BR/BL，**原图坐标**（不是检测器的工作副本）。
    pub pts: [[f32; 2]; 4],
}

/// 一次 run 的结果。
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OcrResult {
    /// 文本行（阅读序）。
    pub lines: Vec<TextLine>,
    /// 检测器工作副本宽（长边帽之后）。
    pub work_w: i32,
    /// 检测器工作副本高。
    pub work_h: i32,
    /// 检测网络实际输入宽。
    pub det_input_w: i32,
    /// 检测网络实际输入高。
    pub det_input_h: i32,
    /// 检测框数（合并前）。
    pub num_boxes: usize,
    /// 合并进行数。
    pub num_merged: usize,
    /// 杂框收紧数。
    pub num_decluttered: usize,
    /// 区域重试次数。
    pub num_det_retried: usize,
    /// 翻正行数。
    pub num_flipped: usize,
    /// 空文本行数。
    pub num_unread: usize,
    /// 分阶段毫秒。
    pub timings: Timings,
}

/// 分阶段计时（字段名即 JSON 导出约定）。
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Timings {
    /// det 预处理。
    pub det_pre_ms: f64,
    /// det 前向。
    pub det_infer_ms: f64,
    /// det 后处理。
    pub det_post_ms: f64,
    /// 裁剪。
    pub crop_ms: f64,
    /// 方向分类。
    pub cls_ms: f64,
    /// rec 预处理。
    pub rec_pre_ms: f64,
    /// rec 前向。
    pub rec_infer_ms: f64,
    /// rec 后处理。
    pub rec_post_ms: f64,
    /// 总计。
    pub total_ms: f64,
}

/// 计时起点（`start.elapsed_ms()` 得毫秒）。字段命名的 `*_ms` 与基准对齐。
trait ElapsedMs {
    fn elapsed_ms(&self) -> f64;
}
impl ElapsedMs for std::time::Instant {
    fn elapsed_ms(&self) -> f64 {
        self.elapsed().as_secs_f64() * 1000.0
    }
}
fn now() -> std::time::Instant {
    std::time::Instant::now()
}

/// 字典来源。
#[derive(Clone, Debug)]
pub enum Dictionary {
    /// 从识别模型 metadata 的 `character` 取（tiny/small 转换版走这条）。
    Embedded,
    /// 显式给一份（上游模型走这条；每行一个字符，UTF-8）。
    Text(String),
}

/// OCR 引擎：一次构造、多次 run；`run(&self)`（权重只读）。
pub struct Engine {
    /// 检测会话。
    det: Session,
    /// 识别会话。
    rec: Session,
    /// 方向分类会话（可选）。
    cls: Option<Session>,
    /// 流水线配置。
    cfg: PipelineConfig,
    /// 字符表（0 = blank，末尾 = 空格）。
    charset: Vec<String>,
    /// det 图输入名。
    det_in: String,
    /// rec 图输入名。
    rec_in: String,
    /// cls 图输入名。
    cls_in: String,
}

impl Engine {
    /// 打开引擎。`dict` 给 [`Dictionary::Text`] 时用外部字典（上游模型）；
    /// [`Dictionary::Embedded`] 从 rec 模型 metadata 取，取不到报错并
    /// 给出修复建议（§4.4 的硬要求）。
    pub fn open(
        det_path: &std::path::Path,
        rec_path: &std::path::Path,
        cls_path: Option<&std::path::Path>,
        dict: Dictionary,
        cfg: PipelineConfig,
    ) -> Result<Self> {
        let det = Session::open(det_path)?;
        let rec = Session::open(rec_path)?;
        let cls = match cls_path {
            Some(p) => Some(Session::open(p)?),
            None => None,
        };
        Self::from_sessions(det, rec, cls, dict, cfg)
    }

    /// 从内存字节构造（WASM / 移动端；display_name 只用于报告）。
    pub fn open_bytes(
        det: &[u8],
        rec: &[u8],
        cls: Option<&[u8]>,
        display_name: &str,
        dict: Dictionary,
        cfg: PipelineConfig,
    ) -> Result<Self> {
        let det = Session::from_memory(det, &format!("{display_name}.det.onnx"))?;
        let rec = Session::from_memory(rec, &format!("{display_name}.rec.onnx"))?;
        let cls = match cls {
            Some(b) => Some(Session::from_memory(
                b,
                &format!("{display_name}.cls.onnx"),
            )?),
            None => None,
        };
        Self::from_sessions(det, rec, cls, dict, cfg)
    }

    fn from_sessions(
        det: Session,
        rec: Session,
        cls: Option<Session>,
        dict: Dictionary,
        cfg: PipelineConfig,
    ) -> Result<Self> {
        // 线程数要在首次并行算子前请求（池首用时定容；先建的引擎赢，
        // 之后请求 no-op）。cfg.threads = 0 = 自动。
        if cfg.threads > 0 {
            qppocr_kernels::par::set_threads(cfg.threads);
        }
        // 字典：blank 在 0、字典项、空格在末尾（ppocr 的约定）
        let raw: String = match dict {
            Dictionary::Embedded => {
                rec.graph
                    .metadata
                    .get("character")
                    .cloned()
                    .ok_or_else(|| {
                        Error::Graph(
                            "rec 模型不带 'character' 字典元数据（上游原件如此）——\
                             请用 Dictionary::Text 显式提供：tiny 档 6904 行、\
                             small/medium 档 18708 行"
                                .into(),
                        )
                    })?
            }
            Dictionary::Text(s) => s,
        };
        let mut charset = vec!["blank".to_string()];
        let mut cur = String::new();
        for ch in raw.chars() {
            if ch == '\n' {
                if cur.ends_with('\r') {
                    cur.pop();
                }
                charset.push(std::mem::take(&mut cur));
            } else {
                cur.push(ch);
            }
        }
        if !cur.is_empty() {
            charset.push(cur);
        }
        charset.push(" ".to_string());

        // ★ 字典与识别模型匹配性校验：输出类别数必须等于 blank + 字典 + 空格。
        // 拿错字典硬报错并给出修复建议（§4.4：字典与档位不匹配是最高频错误）。
        // rec 输出形状 [B, T, C] 在运行时才定 C——从末节点输出张量的形状读不到，
        // 用首次 run 校验（见 run_once 里 c_len 断言）。这里先校验行数档位。
        if !matches!(charset.len(), 6906 | 18710) {
            // 允许自定义字典，但不是常见档位时提示一下（不阻止——用户可能
            // 真的换了字典）。真正的硬校验在 run 时的 C == charset.len()。
        }

        let det_in = det
            .graph
            .inputs
            .iter()
            .find(|s| !s.is_empty())
            .cloned()
            .unwrap_or_default();
        let rec_in = rec
            .graph
            .inputs
            .iter()
            .find(|s| !s.is_empty())
            .cloned()
            .unwrap_or_default();
        let cls_in = cls
            .as_ref()
            .and_then(|c| c.graph.inputs.iter().find(|s| !s.is_empty()).cloned())
            .unwrap_or_default();

        Ok(Self {
            det,
            rec,
            cls,
            cfg,
            charset,
            det_in,
            rec_in,
            cls_in,
        })
    }

    /// 配置引用。
    pub fn config(&self) -> &PipelineConfig {
        &self.cfg
    }

    /// 只检测不识别（框带分数，无文本）。
    pub fn run_det_only(&self, img: &Image) -> Result<OcrResult> {
        self.run_once(img, true)
    }

    /// 完整流水线：det → cls → rec；置信度弱时区域重试（retry_conf）。
    pub fn run(&self, img: &Image) -> Result<OcrResult> {
        let mut first = self.run_once(img, false)?;
        if self.cfg.retry_conf <= 0.0 {
            return Ok(first);
        }

        // 识别器不确定哪些行
        let weak: Vec<usize> = first
            .lines
            .iter()
            .enumerate()
            .filter(|(_, l)| !l.text.is_empty() && (l.confidence as f64) < self.cfg.retry_conf)
            .map(|(i, _)| i)
            .collect();
        if weak.is_empty() {
            return Ok(first);
        }

        // ★ 在弱行占据的**区域**上重试，不是整图重跑：检测器的短边规则
        // 会把给它的东西放大到短边 736——裁出可疑部分等于按字符给检测器
        // 高得多的分辨率，代价只有那个矩形。难图.png 实测：整图重试
        // 1920×576（1.1 MP）读 0.875；484×115 的区域 960×224（0.21 MP，
        // 5 倍便宜）读 0.912。
        let (mut x0, mut y0, mut x1, mut y1) = (1e18f64, 1e18f64, -1e18f64, -1e18f64);
        for &i in &weak {
            for p in &first.lines[i].pts {
                x0 = x0.min(p[0] as f64);
                y0 = y0.min(p[1] as f64);
                x1 = x1.max(p[0] as f64);
                y1 = y1.max(p[1] as f64);
            }
        }
        let mx = 8.0f64.max((x1 - x0) * 0.10);
        let my = 8.0f64.max((y1 - y0) * 0.25);
        let rx0 = (x0 - mx).max(0.0) as i32;
        let ry0 = (y0 - my).max(0.0) as i32;
        let rx1 = ((x1 + mx).min(img.w as f64)) as i32;
        let ry1 = ((y1 + my).min(img.h as f64)) as i32;
        if rx1 - rx0 < 8 || ry1 - ry0 < 8 {
            return Ok(first);
        }

        let mut sub = Image {
            w: rx1 - rx0,
            h: ry1 - ry0,
            c: 3,
            orig_w: rx1 - rx0,
            orig_h: ry1 - ry0,
            data: vec![0u8; ((rx1 - rx0) as usize) * ((ry1 - ry0) as usize) * 3],
        };
        for y in 0..sub.h {
            let src_row = img.row(ry0 + y);
            // 行内偏移是 rx0*3（: img.row(ry0+y) + rx0*3），不是全图平铺
            let off = (rx0 as usize) * 3;
            let len = (sub.w as usize) * 3;
            let dst_row = &mut sub.data[(y as usize) * len..(y as usize + 1) * len];
            dst_row.copy_from_slice(&src_row[off..off + len]);
        }

        let second = self.run_once(&sub, false)?;
        if second.lines.is_empty() {
            return Ok(first);
        }

        // 只在区域的结果更好时替换弱行
        let best_new = second
            .lines
            .iter()
            .map(|l| l.confidence as f64)
            .fold(0.0, f64::max);
        let worst_old = weak
            .iter()
            .map(|&i| first.lines[i].confidence as f64)
            .fold(1.0, f64::min);
        if best_new <= worst_old {
            return Ok(first);
        }

        let mut kept: Vec<TextLine> = Vec::new();
        for (i, l) in first.lines.iter().enumerate() {
            if !weak.contains(&i) {
                kept.push(l.clone());
            }
        }
        for mut l in second.lines {
            l.retried = true; // 来自重试遍的替换行
            for p in &mut l.pts {
                p[0] += rx0 as f32;
                p[1] += ry0 as f32;
            }
            // 每字坐标同偏移：重试跑在子图上，line 与 chars 必须同链映射
            for cs in &mut l.chars {
                for p in &mut cs.pts {
                    p[0] += rx0 as f32;
                    p[1] += ry0 as f32;
                }
            }
            kept.push(l);
        }
        kept.sort_by(|a, b| {
            a.pts[0][1]
                .partial_cmp(&b.pts[0][1])
                .unwrap()
                .then(a.pts[0][0].partial_cmp(&b.pts[0][0]).unwrap())
        });
        first.lines = kept;
        first.num_unread = first.lines.iter().filter(|l| l.text.is_empty()).count();
        first.num_det_retried += 1;
        first.timings.det_infer_ms += second.timings.det_infer_ms;
        first.timings.rec_infer_ms += second.timings.rec_infer_ms;
        first.timings.total_ms += second.timings.total_ms;
        Ok(first)
    }

    /// 每字四边形：最终文本（含插入空格）逐字符 → 裁剪图 x 区间 →
    /// （cls 翻转还原）→ 沿行四边形上下边插值 → `to_image` 回原图。
    ///
    /// 竖条（`crop_text_box` 顺时针转正）的角点对应：裁剪图 TL/TR/BR/BL ↔
    /// `pts[3]/[0]/[1]/[2]`。空格用 `add_gap_spaces` 记下的空白 run 区间。
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn build_char_spans(
        text: &str,
        marks: &[crate::pipeline::rec::DecodeMark],
        cols: &[i32],
        spaces: &[crate::pipeline::rec::SpaceSpan],
        cr_w: i32,
        flipped: bool,
        quad: &[[f32; 2]; 4],
        to_image: &dyn Fn(&[[f32; 2]; 4], &mut [[f32; 2]; 4]),
    ) -> Vec<CharSpan> {
        if text.is_empty() || cr_w <= 0 {
            return Vec::new();
        }
        let n = marks.len();
        // 第 k 个真实字符的左缘（crop 坐标）；cols 缺失时按字数均分。
        let left = |k: usize| -> f32 {
            if k < cols.len() {
                cols[k] as f32
            } else if n > 0 {
                cr_w as f32 * k as f32 / n as f32
            } else {
                0.0
            }
        };
        // 竖条判定与 crop_text_box 同款（ch/cw >= 1.5 顺时针转正）
        let w1 = (quad[0][0] - quad[1][0]).hypot(quad[0][1] - quad[1][1]);
        let w2 = (quad[2][0] - quad[3][0]).hypot(quad[2][1] - quad[3][1]);
        let h1 = (quad[0][0] - quad[3][0]).hypot(quad[0][1] - quad[3][1]);
        let h2 = (quad[1][0] - quad[2][0]).hypot(quad[1][1] - quad[2][1]);
        let rotated = h1.max(h2) / w1.max(w2).max(1.0) >= 1.5;
        let corners_src: [[f32; 2]; 4] = if rotated {
            [quad[3], quad[0], quad[1], quad[2]]
        } else {
            [quad[0], quad[1], quad[2], quad[3]]
        };
        let mut q = [[0f32; 2]; 4];
        to_image(&corners_src, &mut q);
        let (tl, tr, br, bl) = (&q[0], &q[1], &q[2], &q[3]);
        let lerp = |a: &[f32; 2], b: &[f32; 2], u: f32| {
            [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u]
        };

        let mut out = Vec::with_capacity(text.chars().count());
        let mut mi = 0usize; // 已消费的真实字符数
        let mut off = 0usize; // 最终文本字节游标
        for ch in text.chars() {
            let sz = ch.len_utf8();
            // 该字符在（未翻转）裁剪图上的 x 区间
            let (mut l, mut r) = if let Some(sp) = spaces.iter().find(|s| s.off == off) {
                (sp.x0 as f32, sp.x1 as f32)
            } else {
                let a = left(mi);
                let b = if mi + 1 < n {
                    left(mi + 1)
                } else {
                    cr_w as f32
                };
                mi += 1;
                (a, b)
            };
            if r < l {
                std::mem::swap(&mut l, &mut r);
            }
            if flipped {
                // rec 看到的是 180° 翻转后的裁剪图；还原到裁剪时的方向
                let (l2, r2) = (cr_w as f32 - r, cr_w as f32 - l);
                l = l2;
                r = r2;
            }
            let ul = (l / cr_w as f32).clamp(0.0, 1.0);
            let ur = (r / cr_w as f32).clamp(0.0, 1.0);
            out.push(CharSpan {
                text: ch.to_string(),
                pts: [
                    lerp(tl, tr, ul),
                    lerp(tl, tr, ur),
                    lerp(bl, br, ur),
                    lerp(bl, br, ul),
                ],
            });
            off += sz;
        }
        out
    }

    /// 一遍 det+crop+rec。`run` 包这层以在首遍不确定时换尺度重试。
    fn run_once(&self, img: &Image, det_only: bool) -> Result<OcrResult> {
        let mut res = OcrResult::default();
        let t_start = now();

        // ---- 预处理：长边帽（ppocr reduce_max_side）----
        // ★ 内存：超帽路径直接把 resize 结果当 work，不做「先整图克隆、
        // 再被替换」的过渡——那是原图之外又一整份的瞬时峰值（3000×2000
        // 的 big.png 上 18 MB）。未超帽才需要克隆（后面补边/增强要写）。
        let mut work: Image;
        let (mut ratio_h, mut ratio_w) = (1.0f64, 1.0f64);
        {
            let (h, w) = (img.h, img.w);
            // 只在长边真的超帽时碰像素。无条件对齐 32 的旧做法会悄悄
            // 重采样每一张图——900x520 变 896x512，白白丢细节。
            let mut resized: Option<Image> = None;
            if h.max(w) > self.cfg.max_side_len {
                let ratio = self.cfg.max_side_len as f64 / (h.max(w)) as f64;
                let rh = ((h as f64 * ratio / 32.0).round_ties_even() as i32) * 32;
                let rw = ((w as f64 * ratio / 32.0).round_ties_even() as i32) * 32;
                if rh > 0 && rw > 0 && (rh != h || rw != w) {
                    resized = Some(resize_bilinear_img(img, rw, rh));
                    ratio_h = h as f64 / rh as f64;
                    ratio_w = w as f64 / rw as f64;
                }
            }
            work = resized.unwrap_or_else(|| img.clone());
        }

        if self.cfg.enhance_contrast {
            super::image::auto_levels(&mut work);
        }
        // ★ 识别器从原图裁（增强后），不从缩过的 work 裁。这里用
        // 借用（基准是 `const Image* crop_src = &img`）——默认路径零拷贝，
        // 只有开增强才物化整图副本。
        let mut enhanced_full: Image;
        let crop_src: &Image = if self.cfg.enhance_contrast {
            enhanced_full = img.clone();
            super::image::auto_levels(&mut enhanced_full);
            &enhanced_full
        } else {
            img
        };

        // ---- 竖长/极矮图补边（PP-OCR 的 use_vertical_padding）----
        let src_px = work.w as f64 * work.h as f64;
        let mut pad_top: i32 = 0;
        if self.cfg.vertical_padding
            && work.h > 0
            && work.w > 0
            && (work.h <= self.cfg.min_height
                || work.w as f64 / work.h as f64 > self.cfg.width_height_ratio)
        {
            let want =
                (work.w as f64 / self.cfg.width_height_ratio).max(self.cfg.min_height as f64) * 2.0;
            let pad = ((want - work.h as f64).abs() / 2.0) as i32;
            if pad > 0 {
                let mut padded = Image {
                    w: work.w,
                    h: work.h + pad * 2,
                    c: 3,
                    orig_w: work.orig_w,
                    orig_h: work.orig_h,
                    data: vec![0u8; (work.w as usize) * ((work.h + pad * 2) as usize) * 3],
                };
                for y in 0..work.h {
                    let src_row = work.row(y);
                    let dst_off = ((y + pad) as usize) * (work.w as usize) * 3;
                    padded.data[dst_off..dst_off + (work.w as usize) * 3].copy_from_slice(src_row);
                }
                work = padded;
                pad_top = pad;
            }
        }

        // ---- 检测 ----
        let _t_det0 = now();
        let (dh, dw) = (work.h, work.w);
        let mut dratio = 1.0f64;
        if dh.min(dw) < self.cfg.det_limit_side_len {
            dratio = self.cfg.det_limit_side_len as f64 / (dh.min(dw)) as f64;
        }
        if self.cfg.upscale > 1 {
            dratio *= self.cfg.upscale as f64;
        }
        // 但不越过长边帽。普通照片不生效（短边规则本来就低于帽）；
        // 补过边的细条上这是 2944x736 与检测器同样满意的尺寸之差。
        if self.cfg.det_max_side > 0 {
            let cap = self.cfg.det_max_side as f64 / (dh.max(dw)) as f64;
            if cap < dratio {
                dratio = cap;
            }
        }
        // ★ 检测成本上限，按**源图像素**的倍数计（只在补边发生时生效——
        // 测过，对每张图都用是错的：tiny.png 600x200 想要 3.68x 倍放大，
        // 压到 2x 会改它的文本）。补边时 work_px 全是文件里从来没有的
        // 黑条，放大它们是白付钱。小长图.png：194→51 ms，文本相同。
        if self.cfg.det_pixel_budget > 0.0 && src_px > 0.0 && pad_top > 0 {
            let work_px = work.w as f64 * work.h as f64;
            let dmax = (self.cfg.det_pixel_budget * src_px / work_px).sqrt();
            if dratio > dmax {
                dratio = dmax;
            }
        }
        let mut nh = (dh as f64 * dratio) as i32;
        let mut nw = (dw as f64 * dratio) as i32;
        nh = ((nh as f64 / 32.0).round_ties_even() as i32) * 32;
        nw = ((nw as f64 / 32.0).round_ties_even() as i32) * 32;
        nh = nh.max(32);
        nw = nw.max(32);
        // ★ 恒等 resize 跳过：目标尺寸 == work 尺寸时双线性插值是
        // 逐字节拷贝（scale=1、fx=ly=0，位级等价），白付一遍全图插值
        //（864×960 上 ~2ms，落在「未归类」桶里）。短边规则不放大、
        // 长边帽刚卡住的图经常命中。 无此守卫。
        let det_img = if nw == work.w && nh == work.h {
            work.clone()
        } else {
            resize_bilinear_img(&work, nw, nh)
        };
        res.work_w = work.w;
        res.work_h = work.h;
        res.det_input_w = nw;
        res.det_input_h = nh;

        let t_pre0 = now();
        // core 是 forbid(unsafe)——det 输入的 uninit 优化下沉不了，
        // 清零（8.5 MB warm ~0.7 ms）可接受；若 profile 显示它重要，
        // 把「RGB→归一化 NCHW」整体下沉为 kernels 的安全 API。
        let mut in_f32 =
            qppocr_kernels::buf::F32Buf::with_zeroed(3 * (nh as usize) * (nw as usize));
        let (scale, mean, istd) = (1.0f32 / 255.0, 0.5f32, 1.0f32 / 0.5f32);
        for y in 0..nh {
            let r = det_img.row(y);
            for x in 0..nw {
                for c in 0..3 {
                    let v = r[(x as usize) * 3 + c] as f32 * scale;
                    in_f32[c * (nh as usize) * (nw as usize)
                        + (y as usize) * (nw as usize)
                        + x as usize] = (v - mean) * istd;
                }
            }
        }
        res.timings.det_pre_ms = t_pre0.elapsed_ms();

        let t_inf0 = now();
        let det_out = self.det.run(vec![(
            self.det_in.clone(),
            Tensor {
                name: String::new(),
                shape: vec![1, 3, nh as i64, nw as i64],
                dtype: DType::F32,
                f32: in_f32,
                i64: Vec::new(),
            },
        )])?;
        res.timings.det_infer_ms = t_inf0.elapsed_ms();
        let pred = &det_out[0];

        let t_post0 = now();
        let mut boxes = db_postprocess(
            &pred.f32,
            nh,
            nw,
            work.w,
            work.h,
            self.cfg.det_thresh,
            self.cfg.box_thresh,
            self.cfg.unclip_ratio,
            self.cfg.unclip_perp,
            self.cfg.max_candidates,
            self.cfg.use_dilation,
        );
        res.num_boxes = boxes.len();
        // ★ work（未超帽时是原图的整份克隆）此后不再使用——裁剪从原图
        // 借用走（crop_src）。在这里释放，裁剪/识别阶段不再背着它。
        drop(work);

        sort_reading_order(&mut boxes);
        res.num_merged = merge_same_line(&mut boxes, self.cfg.merge_line_gap as f32);
        res.timings.det_post_ms = t_post0.elapsed_ms();

        // 检测器坐标 → 图像坐标：work 是检测器的私有副本（缩过/补过边），
        // 两个比率映射回来；补边量减掉。调用方看到的坐标永远在原图上。
        let to_image = |inp: &[[f32; 2]; 4], out: &mut [[f32; 2]; 4]| {
            for k in 0..4 {
                out[k][0] = (inp[k][0] as f64 * ratio_w) as f32;
                out[k][1] = ((inp[k][1] as f64 - pad_top as f64) * ratio_h) as f32;
            }
        };

        if det_only || boxes.is_empty() {
            for b in &boxes {
                let mut l = TextLine {
                    text: String::new(),
                    chars: Vec::new(),
                    confidence: b.score,
                    rotation: 0,
                    retried: false,
                    pts: [[0.0; 2]; 4],
                };
                to_image(&b.pts, &mut l.pts);
                res.lines.push(l);
            }
            res.timings.total_ms = t_start.elapsed_ms();
            return Ok(res);
        }

        // ---- 裁剪 ----
        let t_crop0 = now();
        // ★ 从解码原图裁，不从缩过的 work 裁：检测跑在缩图上，框回来是
        // work 坐标；crop 也在 work 上做就把分辨率在识别器看到之前扔掉
        // ——scan_small（4000x3000 帽到 2000 宽）裁出 20-34 px 高的条再
        // 放大到 48，文本回来是烂的。经同一组比率映射回原图不花钱，
        // 12 MP 照片上给识别器 4 倍线性分辨率。
        let mut crops: Vec<Image> = Vec::with_capacity(boxes.len());
        let mut wh_ratio: Vec<f32> = Vec::with_capacity(boxes.len());
        for b in boxes.iter_mut() {
            let mut pts = [[0f32; 2]; 4];
            to_image(&b.pts, &mut pts);
            // 边距杂波判定：unclip 加的框是背景还是杂波？难图.png 的
            // 分离信号（0.48 vs 语料 0.00-0.06）。命中则同中心同方向收窄。
            if self.cfg.unclip_margin_thresh > 0.0 && b.tight_h > 0.0 && !pred.f32.is_empty() {
                let margin = box_margin_clutter(b, &pred.f32, nh, nw, crop_src);
                if std::env::var("QPPOCR_DEBUG_MARGIN").is_ok() {
                    eprintln!(
                        "[margin] thresh={} tight_h={:.1} margin={:.3} nh={} nw={} img={}x{} box=({:.0},{:.0})-({:.0},{:.0})",
                        self.cfg.unclip_margin_thresh,
                        b.tight_h,
                        margin,
                        nh,
                        nw,
                        crop_src.w,
                        crop_src.h,
                        b.pts[0][0],
                        b.pts[0][1],
                        b.pts[2][0],
                        b.pts[2][1]
                    );
                }
                if margin > self.cfg.unclip_margin_thresh {
                    let c = [
                        (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) * 0.25,
                        (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) * 0.25,
                    ];
                    let mut ex2 = [pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]];
                    let mut ey2 = [pts[2][0] - pts[1][0], pts[2][1] - pts[1][1]];
                    let lw = (ex2[0] * ex2[0] + ex2[1] * ex2[1]).sqrt();
                    let lh = (ey2[0] * ey2[0] + ey2[1] * ey2[1]).sqrt();
                    if lw > 1e-3 && lh > 1e-3 {
                        ex2[0] /= lw;
                        ex2[1] /= lw;
                        ey2[0] /= lh;
                        ey2[1] /= lh;
                        let want = b.tight_h * (1.0 + 0.5 * self.cfg.unclip_ratio);
                        let hw = lw * 0.5;
                        let hh = lh.min(want) * 0.5;
                        let sg = [[-1.0f32, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
                        for k in 0..4 {
                            pts[k][0] = c[0] + sg[k][0] * hw * ex2[0] + sg[k][1] * hh * ey2[0];
                            pts[k][1] = c[1] + sg[k][0] * hw * ex2[1] + sg[k][1] * hh * ey2[1];
                        }
                        // 收紧框写回 boxes[i]：不只是显示——区域重试的
                        // 裁剪区域从这些 pts 计算，留肥框会把刚剔除的杂波
                        // 又包回重试区域（难图实测：不写回时收紧只救一半）。
                        b.pts = pts;
                        res.num_decluttered += 1;
                    }
                }
            }
            let crop = crop_text_box(crop_src, &pts);
            wh_ratio.push(batch_ratio(
                &crop,
                self.cfg.rec_height,
                self.cfg.rec_pad_min_h,
            ));
            crops.push(crop);
        }
        // 裁剪落盘（诊断，QPPOCR_SAVE_CROPS=<dir>；PPM，cls 翻转前——
        // 与参考实现的落盘点一致）
        if let Ok(dir) = std::env::var("QPPOCR_SAVE_CROPS") {
            for (i, c) in crops.iter().enumerate() {
                let path = std::path::Path::new(&dir).join(format!("crop_{i:02}.ppm"));
                if let Ok(mut f) = std::fs::File::create(&path) {
                    use std::io::Write as _;
                    let _ = write!(
                        f,
                        "P6
{} {}
255
",
                        c.w, c.h
                    );
                    let _ = f.write_all(&c.data);
                }
            }
        }
        res.timings.crop_ms = t_crop0.elapsed_ms();

        // ---- 方向分类：把读作倒置的裁剪翻 180° ----
        let mut flipped = vec![false; crops.len()];
        if let Some(cls) = &self.cls {
            let tc0 = now();
            let (ch, cw) = (self.cfg.cls_height, self.cfg.cls_width);
            let mut beg = 0usize;
            while beg < crops.len() {
                let end = (beg + self.cfg.rec_batch).min(crops.len());
                let bsz = end - beg;
                let mut batch_f32 = F32Buf::with_zeroed(bsz * 3 * (ch as usize) * (cw as usize));
                for i in beg..end {
                    let view = if self.cfg.cls_window {
                        cls_view(&crops[i], cw, ch)
                    } else {
                        crops[i].clone()
                    };
                    pack_crop(
                        &view,
                        ch,
                        cw,
                        &mut batch_f32[(i - beg) * 3 * (ch as usize) * (cw as usize)..],
                        false,
                    );
                }
                if let Ok(dir) = std::env::var("QPPOCR_SAVE_CROPS") {
                    let path =
                        std::path::Path::new(&dir).join(format!("clsin_{:03}_{:03}.f32", beg, end));
                    let bytes: Vec<u8> = batch_f32
                        .as_slice()
                        .iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect();
                    let _ = std::fs::write(path, bytes);
                }
                let out = cls.run(vec![(
                    self.cls_in.clone(),
                    Tensor {
                        name: String::new(),
                        shape: vec![bsz as i64, 3, ch as i64, cw as i64],
                        dtype: DType::F32,
                        f32: batch_f32,
                        i64: Vec::new(),
                    },
                )])?;
                let o = &out[0]; // [B, 2]
                for i in beg..end {
                    let p = &o.f32[(i - beg) * 2..];
                    if p[1] > p[0] && p[1] >= self.cfg.cls_thresh {
                        crops[i] = rotate_image(&crops[i], 180);
                        flipped[i] = true;
                        res.num_flipped += 1;
                    }
                }
                beg = end;
            }
            res.timings.cls_ms = tc0.elapsed_ms();
        }

        // ---- 按长宽比排序（ppocr 减少批内 padding 浪费）----
        let mut order: Vec<usize> = (0..crops.len()).collect();
        order.sort_by(|&a, &b| wh_ratio[a].partial_cmp(&wh_ratio[b]).unwrap());

        let mut lines: Vec<TextLine> = (0..boxes.len())
            .map(|_| TextLine {
                text: String::new(),
                chars: Vec::new(),
                confidence: 0.0,
                rotation: 0,
                retried: false,
                pts: [[0.0; 2]; 4],
            })
            .collect();
        let img_h = self.cfg.rec_height;
        // 批按宽度相近切，不只按数量。一批一个张量，批内行都 pad 到最宽
        // 行的宽——行被撑到自身 3 倍宽时识别器读它的方式会变（CJK 与数字
        // 间的空格不再输出，值 4.7 个点）。
        let mut beg = 0usize;
        while beg < order.len() {
            let mut end = beg + 1;
            if self.cfg.rec_batch_ratio > 0.0 {
                let base = wh_ratio[order[beg]].max(1e-6);
                while end < order.len()
                    && end - beg < self.cfg.rec_batch
                    && wh_ratio[order[end]] as f64 <= base as f64 * self.cfg.rec_batch_ratio
                {
                    end += 1;
                }
            } else {
                end = (beg + self.cfg.rec_batch).min(order.len());
            }
            let mut max_wh_ratio = self.cfg.rec_min_width as f32 / img_h as f32;
            for &i in &order[beg..end] {
                max_wh_ratio = max_wh_ratio.max(wh_ratio[i]);
            }
            let mut img_w = (img_h as f32 * max_wh_ratio) as i32;
            if self.cfg.rec_width_grain > 1 {
                img_w = (img_w + self.cfg.rec_width_grain - 1) / self.cfg.rec_width_grain
                    * self.cfg.rec_width_grain;
            }

            let t_batch0 = now();
            let bsz = end - beg;
            let mut batch_f32 = F32Buf::with_zeroed(bsz * 3 * (img_h as usize) * (img_w as usize));
            for (k, &i) in order[beg..end].iter().enumerate() {
                pack_crop(
                    &crops[i],
                    img_h,
                    img_w,
                    &mut batch_f32[k * 3 * (img_h as usize) * (img_w as usize)..],
                    crop_pads(&crops[i], img_h, self.cfg.rec_pad_min_h),
                );
            }
            res.timings.rec_pre_ms += t_batch0.elapsed_ms();

            let t_ri = now();
            let out = self.rec.run(vec![(
                self.rec_in.clone(),
                Tensor {
                    name: String::new(),
                    shape: vec![bsz as i64, 3, img_h as i64, img_w as i64],
                    dtype: DType::F32,
                    f32: batch_f32,
                    i64: Vec::new(),
                },
            )])?;
            res.timings.rec_infer_ms += t_ri.elapsed_ms();

            let t_rp = now();
            let o = &out[0]; // [B, T, C]
            let t_len = o.shape[1] as usize;
            let c_len = o.shape[2] as usize;
            // ★ 字典/模型配对硬校验。换错字典时 C 与字符表长度错位，
            // ctc_decode 的越界索引是静默跳过——整表错位且不报错（曾用
            // dict_small_medium.txt 配 small：少一个前导换行，18710→18709，
            // 输出全部错一个字符）。from_sessions 的注释承诺过这里要硬校验，
            // 一直没实现，这里补上。
            if c_len != self.charset.len() {
                return Err(crate::error::Error::Graph(format!(
                    "识别输出类别数 {c_len} 与字符表 {} 不符：字典与 rec 模型不配对
  期望行数：tiny 档 6904、small/medium 档 18708
  检查 {}/dict.txt 的来源与行数",
                    self.charset.len(),
                    "rec"
                )));
            }
            let want_gaps = self.cfg.rec_space_gap > 0.0;
            for (k, &i) in order[beg..end].iter().enumerate() {
                let (mut text, conf, marks) =
                    ctc_decode(&o.f32[k * t_len * c_len..], t_len, c_len, &self.charset);
                // 时间步 → 裁剪图 x 的左缘（像素空格与每字坐标共用这套映射）。
                // 「识别器看的是 imgW 宽的张量，这行占 [0, content_w)，其余是
                // 批 padding」——除以 content_w 再乘回裁剪宽。
                let cr = &crops[i];
                let cols: Vec<i32> = if cr.w > 0 && cr.h > 0 {
                    let pad_only = crop_pads(cr, img_h, self.cfg.rec_pad_min_h);
                    let content_w = if pad_only {
                        cr.w.min(img_w)
                    } else {
                        ((img_h as f64 * cr.w as f64 / cr.h as f64).ceil()) as i32
                    };
                    let content_w = content_w.min(img_w);
                    if content_w > 0 {
                        marks
                            .iter()
                            .map(|m| {
                                let p = (m.step as f64 / t_len as f64 * img_w as f64
                                    / content_w as f64)
                                    .clamp(0.0, 1.0);
                                (p * cr.w as f64) as i32
                            })
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };
                let mut spaces = Vec::new();
                if want_gaps && marks.len() >= 2 && cols.len() == marks.len() {
                    let ink = column_ink(cr);
                    let (t2, sp) = add_gap_spaces(
                        &text,
                        &marks,
                        &cols,
                        &ink,
                        cr.w,
                        (self.cfg.rec_space_gap * cr.h as f64) as f32,
                    );
                    text = t2;
                    spaces = sp;
                }
                // ★ 每字坐标（原图空间；含全部逆变换，见 TextLine::chars 文档）
                let chars = Self::build_char_spans(
                    &text,
                    &marks,
                    &cols,
                    &spaces,
                    cr.w,
                    flipped[i],
                    &boxes[i].pts,
                    &to_image,
                );
                lines[i].chars = chars;
                lines[i].text = text;
                lines[i].confidence = conf;
                lines[i].rotation = if flipped[i] { 180 } else { 0 };
                to_image(&boxes[i].pts, &mut lines[i].pts);
            }
            res.timings.rec_post_ms += t_rp.elapsed_ms();
            beg = end;
        }
        res.timings.total_ms = t_start.elapsed_ms();

        // 读不出的框保留不丢（见 TextLine::text 文档）
        res.num_unread = lines.iter().filter(|l| l.text.is_empty()).count();
        res.lines = lines;
        Ok(res)
    }
}
