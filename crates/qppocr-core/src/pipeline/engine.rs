//! OCR 引擎（OCR 引擎主流程）：det → cls → rec 主流程。
//!
//! 字典是一等输入：上游 rec 模型不带内嵌字典
//!（medium 一个元数据都没有），所以 [`Engine::open`] 接受
//! 「从模型 metadata 取」或「显式给一份」两种来源，取不到时报错并
//! 指出该档位字典的行数——不静默降级。

use std::sync::Arc;

use crate::device::{DeviceContext, DeviceSession, SessionOptions, cpu_context};
use crate::error::{Error, Result};
use crate::executor::Session;
use crate::onnx::model::Graph;
use crate::tensor::{DType, Tensor};

use super::config::PipelineConfig;
use super::crop::{batch_ratio, cls_view, crop_pads, crop_text_box, pack_crop};
use super::geometry::{box_margin_clutter, db_postprocess, merge_same_line, sort_reading_order};
use super::image::{Image, resize_bilinear_img, rotate_image};
use super::rec::{add_gap_spaces, column_ink, ctc_decode, ctc_decode_pairs};
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

/// rec 一批的产出（并行段只产出、串行段统一合并——core 是
/// `forbid(unsafe_code)`，不能靠裸指针把 `lines` 切给多线程写）。
struct RecLineOut {
    /// 该行在 `lines`/`boxes` 里的下标（`order` 里存的就是它）。
    idx: usize,
    /// 解码文本（可能为空 = 检测到但读不出）。
    text: String,
    /// 逐字坐标（原图空间）。
    chars: Vec<CharSpan>,
    /// CTC 置信度。
    confidence: f32,
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
    /// 走「方向第二意见」重读的行数（cls 存疑 且 首读低置信）。
    pub num_cls_retry: usize,
    /// 其中**翻转版更自信因而改用了翻转版**的行数。
    pub num_cls_retry_flipped: usize,
    /// 空文本行数。
    pub num_unread: usize,
    /// 分阶段毫秒。
    pub timings: Timings,
}

/// 这一行要不要取「方向第二意见」——两个条件**同时**成立：
///
/// 1. `cls_conf_max < cls_thresh`：分类器**存疑**（两类最大概率没到门）。
///    注意这不是「判错了」，是「倾向某一边但不敢拍板」。
/// 2. `first_conf < retry_conf`：该行**首次解码置信度低**，读出来就不像话。
///
/// 条件 1 单独用是不够的：同一段概率区间里混着该翻和不该翻的行，`p1`
/// 完全交错，**任何门都分不开**（gen100/small 的 `[0.65,0.90)` 里是 6 对 5）。
/// 条件 2 才是真正的判据——rec 自己的置信度把两边切在 0.53~0.72 与
/// 0.98~1.00 之间，实测 13/13 无重叠；顺带它把代价锁死在「已经读坏了的行」。
///
/// 推导、两臂实测与被否证的备选（单纯降 `cls_thresh`）：
/// `bench/gpuhealth/logs/cls_retry/SUMMARY.md`。
#[inline]
fn wants_cls_retry(cls_conf_max: f32, cls_thresh: f32, first_conf: f32, retry_conf: f32) -> bool {
    retry_conf > 0.0 && cls_conf_max < cls_thresh && first_conf < retry_conf
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

/// 计时起点（`start.elapsed_ms()` 得毫秒）。字段一律以 `*_ms` 结尾。
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

/// 图的第一个非空输入名（三个会话的输入张量都按它命名传入）。
///
/// 在 `create_session` 消费图之前调用——设备会话不保证保留 [`Graph`]。
fn first_input(g: &Graph) -> String {
    g.inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap_or_default()
}

/// OCR 引擎：一次构造、多次 run；`run(&self)`（权重只读）。
pub struct Engine {
    /// 检测会话。
    det: Arc<dyn DeviceSession>,
    /// 识别会话。
    rec: Arc<dyn DeviceSession>,
    /// 方向分类会话（可选）。
    cls: Option<Arc<dyn DeviceSession>>,
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
    /// 打开引擎（默认 CPU 设备）。`dict` 给 [`Dictionary::Text`] 时用外部
    /// 字典（上游模型）；[`Dictionary::Embedded`] 从 rec 模型 metadata 取，
    /// 取不到报错并给出修复建议（§4.4 的硬要求）。
    pub fn open(
        det_path: &std::path::Path,
        rec_path: &std::path::Path,
        cls_path: Option<&std::path::Path>,
        dict: Dictionary,
        cfg: PipelineConfig,
    ) -> Result<Self> {
        let ctx = cpu_context();
        Self::open_with(ctx.as_ref(), det_path, rec_path, cls_path, dict, cfg)
    }

    /// 打开引擎（指定设备）：det/rec/cls 三个模型都经 `ctx` 构造会话
    /// （装载期一次性上传权重）。
    pub fn open_with(
        ctx: &dyn DeviceContext,
        det_path: &std::path::Path,
        rec_path: &std::path::Path,
        cls_path: Option<&std::path::Path>,
        dict: Dictionary,
        cfg: PipelineConfig,
    ) -> Result<Self> {
        let mut opts = SessionOptions::default();
        opts.model = crate::device::ModelRole::Det;
        let (det_graph, det_init) = Session::parts_from_path(det_path)?;
        let det_in = first_input(&det_graph);
        let det = ctx.create_session(det_graph, det_init, &opts)?;

        // ★ 字典与输入名在 create_session 消费图之前取：设备会话不保证
        //   保留 Graph（权重上传后拓扑只活在它的执行计划里）。
        let (rec_graph, rec_init) = Session::parts_from_path(rec_path)?;
        let raw: String = match dict {
            Dictionary::Embedded => {
                rec_graph
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
        let rec_in = first_input(&rec_graph);
        let rec = ctx.create_session(rec_graph, rec_init, &opts)?;

        let (cls, cls_in) = match cls_path {
            Some(p) => {
                let (g, init) = Session::parts_from_path(p)?;
                let name = first_input(&g);
                let mut opts_cls = SessionOptions::default();
                opts_cls.model = crate::device::ModelRole::Cls;
                (Some(ctx.create_session(g, init, &opts_cls)?), name)
            }
            None => (None, String::new()),
        };

        Self::from_device_sessions(det, rec, cls, raw, det_in, rec_in, cls_in, cfg)
    }

    /// 从内存字节构造（默认 CPU 设备；WASM / 移动端，display_name 只用于
    /// 报告）。
    pub fn open_bytes(
        det: &[u8],
        rec: &[u8],
        cls: Option<&[u8]>,
        display_name: &str,
        dict: Dictionary,
        cfg: PipelineConfig,
    ) -> Result<Self> {
        let ctx = cpu_context();
        Self::open_bytes_with(ctx.as_ref(), det, rec, cls, display_name, dict, cfg)
    }

    /// 从内存字节构造（指定设备）。
    pub fn open_bytes_with(
        ctx: &dyn DeviceContext,
        det: &[u8],
        rec: &[u8],
        cls: Option<&[u8]>,
        display_name: &str,
        dict: Dictionary,
        cfg: PipelineConfig,
    ) -> Result<Self> {
        let mut opts = SessionOptions::default();
        opts.model = crate::device::ModelRole::Det;
        let (det_graph, det_init) =
            Session::parts_from_memory(det, &format!("{display_name}.det.onnx"))?;
        let det_in = first_input(&det_graph);
        let det = ctx.create_session(det_graph, det_init, &opts)?;

        let mut opts = SessionOptions::default();
        opts.model = crate::device::ModelRole::Rec;
        let (rec_graph, rec_init) =
            Session::parts_from_memory(rec, &format!("{display_name}.rec.onnx"))?;
        // ★ 字典与输入名在 create_session 消费图之前取（同 open_with）。
        let raw: String = match dict {
            Dictionary::Embedded => {
                rec_graph
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
        let rec_in = first_input(&rec_graph);
        let rec = ctx.create_session(rec_graph, rec_init, &opts)?;

        let (cls, cls_in) = match cls {
            Some(b) => {
                let (g, init) = Session::parts_from_memory(b, &format!("{display_name}.cls.onnx"))?;
                let name = first_input(&g);
                // ★ 此前漏设：opts 残留 Rec——cls 会话被按 Rec 路由
                //（QPPOCR_GPU_STAGES=detrec 曾因此把 cls 误上 GPU，
                //  翻转边界分歧毁掉 ~14% 行的文本才暴露）。
                opts.model = crate::device::ModelRole::Cls;
                (Some(ctx.create_session(g, init, &opts)?), name)
            }
            None => (None, String::new()),
        };

        Self::from_device_sessions(det, rec, cls, raw, det_in, rec_in, cls_in, cfg)
    }

    /// 由已构造的设备会话组装引擎（字典原文与三个输入名已在上游提取）。
    #[allow(clippy::too_many_arguments)]
    fn from_device_sessions(
        det: Arc<dyn DeviceSession>,
        rec: Arc<dyn DeviceSession>,
        cls: Option<Arc<dyn DeviceSession>>,
        raw: String,
        det_in: String,
        rec_in: String,
        cls_in: String,
        cfg: PipelineConfig,
    ) -> Result<Self> {
        // 线程数要在首次并行算子前请求（池首用时定容；先建的引擎赢，
        // 之后请求 no-op）。cfg.threads = 0 = 自动。
        if cfg.threads > 0 {
            qppocr_kernels::par::set_threads(cfg.threads);
        }
        // ★ 两级并行（仅受益于宿主侧扇出的会话，即 CPU）：rec 的外层分片
        //   各占一个**独立小池**，其算子在该池内 fork，互不排队。单池做不
        //   到这一点——`fork_mu` 会把并发 fork 串起来，外层线程互相等，
        //   等于没分片。
        //   布局 = [全尺寸默认池, 分片池 × k]，k*分片大小 ≈ 总线程数。
        //   GPU 会话（prefers_host_parallelism() = false）不扇出，分片池
        //   永远空转——跳过布局请求；池 0 仍按全尺寸创建，裁剪 / db 后
        //   处理这些设备边界外的 CPU 并行不受影响。
        let shards = cfg.rec_shards;
        if shards > 1 && rec.prefers_host_parallelism() {
            // ★ 必须用 planned_threads()：threads() 会顺手把池按**默认布局**
            //   建起来，之后 request_pools 就成了 no-op——分片线程会全部挤在
            //   0 号池上被 fork_mu 串行化（实测逐算子耗时翻倍）。
            let total = qppocr_kernels::par::planned_threads();
            let m = (total / shards).max(2);
            let mut sizes = vec![total];
            sizes.extend(std::iter::repeat_n(m, shards));
            qppocr_kernels::par::request_pools(&sizes);
        }
        // 字典：blank 在 0、字典项、空格在末尾（ppocr 的约定）
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

    /// 把 `batches`（`(begin, end)` 下标对）分给 N 个线程并行执行，
    /// **每个线程内部串行**（`par::set_serial_execution(true)`）。
    ///
    /// 为什么不是每批一个 `parallel_for`：行内的算子级 fork 会争池子的
    /// `fork_mu` 互相排队，扩展性受损。改成「每行一个线程、行内串行」后，
    /// 墙钟 ≈ 最慢单行的耗时。
    ///
    /// 串行执行**不改切分决策**（`par::threads()` 仍报真实池大小），
    /// 因此各内核的分块与累加序与并行路径完全一致——逐位可验证。
    fn run_batches<T, F>(batches: &[(usize, usize)], f: F) -> Result<Vec<T>>
    where
        T: Send,
        F: Fn(usize, usize) -> Result<T> + Sync,
    {
        let n = batches.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let cap = std::env::var("QPPOCR_ROW_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(usize::MAX);
        let nthreads = qppocr_kernels::par::threads().min(n).min(cap).max(1);
        let row_dbg = std::env::var_os("QPPOCR_ROW_DEBUG").is_some();
        if row_dbg {
            eprintln!("[row] batches={n} nthreads={nthreads}");
        }
        if nthreads == 1 {
            // 单线程：直接顺序跑，不走线程创建
            let mut out = Vec::with_capacity(n);
            for &(b, e) in batches {
                out.push(f(b, e)?);
            }
            return Ok(out);
        }
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let next = AtomicUsize::new(0);
        let slots: Vec<Mutex<Option<Result<T>>>> = (0..n).map(|_| Mutex::new(None)).collect();
        let t_scope = std::time::Instant::now();
        std::thread::scope(|s| {
            for _ in 0..nthreads {
                s.spawn(|| {
                    qppocr_kernels::par::set_serial_execution(true);
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        let (b, e) = batches[i];
                        let t1 = std::time::Instant::now();
                        let r = f(b, e);
                        if row_dbg {
                            eprintln!(
                                "[row]  batch {i} [{b},{e}) {:.2} ms",
                                t1.elapsed().as_secs_f64() * 1000.0
                            );
                        }
                        *slots[i].lock().unwrap() = Some(r);
                    }
                });
            }
        });
        if row_dbg {
            eprintln!(
                "[row]  fan-out wall {:.2} ms",
                t_scope.elapsed().as_secs_f64() * 1000.0
            );
        }
        let mut out = Vec::with_capacity(n);
        for slot in slots {
            match slot.into_inner().unwrap() {
                Some(r) => out.push(r?),
                None => return Err(Error::Graph("run_batches: 批次槽未回填（内部错误）".into())),
            }
        }
        Ok(out)
    }

    /// 与 [`Self::run_batches`] 同样按批扇出，但每个外层线程用**自己的
    /// 分片池**（`pool_slot` 1..=k），于是批内的算子还能在本池内 fork——
    /// 单池下并发的 fork 会被 `fork_mu` 串起来，外层线程互相排队。
    ///
    /// 每行的算子数 ~80 且顺序依赖，外层线程不能太多（分片池会太小、
    /// 每池的 fork 收益盖不过协调成本）；k 由 `rec_shards` 定。
    fn run_batches_sharded<T, F>(batches: &[(usize, usize)], f: F, shards: usize) -> Result<Vec<T>>
    where
        T: Send,
        F: Fn(usize, usize) -> Result<T> + Sync,
    {
        let n = batches.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let k = shards.min(n).max(1);
        if k < 2 {
            return Self::run_batches(batches, f);
        }
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let next = AtomicUsize::new(0);
        let slots: Vec<Mutex<Option<Result<T>>>> = (0..n).map(|_| Mutex::new(None)).collect();
        std::thread::scope(|s| {
            let (fr, nr, sr) = (&f, &next, &slots);
            for j in 0..k {
                let jj = j;
                s.spawn(move || {
                    // 本线程的 fork 走分片池 jj+1（0 号是默认池）。
                    qppocr_kernels::par::set_pool_slot(1 + jj);
                    loop {
                        let i = nr.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        let (b, e) = batches[i];
                        let r = fr(b, e);
                        *sr[i].lock().unwrap() = Some(r);
                    }
                });
            }
        });
        let mut out = Vec::with_capacity(n);
        for slot in slots {
            match slot.into_inner().unwrap() {
                Some(r) => out.push(r?),
                None => {
                    return Err(Error::Graph(
                        "run_batches_sharded: 批次槽未回填（内部错误）".into(),
                    ));
                }
            }
        }
        Ok(out)
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
        // 借用而不是克隆——默认路径零拷贝，
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
        // ★ 通道在外层：三个平面各自连续写。写成「行外层、通道内层」时
        //   每个像素要在相隔 nh·nw 的三个地址间跳一次，等于同时开 3 条
        //   远隔的写流——960×832 上实测 2.3 ms，比算术该有的慢一个量级。
        //   逐元素算术**一字不改**（同样的 `*scale`、`-mean`、`*istd` 顺序），
        //   所以逐位不变。
        let plane = (nh as usize) * (nw as usize);
        for c in 0..3 {
            let dst = &mut in_f32[c * plane..(c + 1) * plane];
            for y in 0..nh as usize {
                let r = det_img.row(y as i32);
                let drow = &mut dst[y * nw as usize..(y + 1) * nw as usize];
                for (x, d) in drow.iter_mut().enumerate() {
                    let v = r[x * 3 + c] as f32 * scale;
                    *d = (v - mean) * istd;
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
        // ★ 每个框的裁剪彼此独立，并行做。`crop_text_box` 的透视反采样
        //   每个输出像素要一次除法（8×8 单应求逆后按需算），9 行串行
        //   实测 4.2 ms。并行段只产出结果，写回 boxes（收紧框）与计数
        //   在下面的串行合并里做。
        let cropped = super::geometry::par_map(&boxes, |b| {
            let mut pts = [[0f32; 2]; 4];
            to_image(&b.pts, &mut pts);
            // 边距杂波判定：unclip 加的框是背景还是杂波？难图.png 的
            // 分离信号（0.48 vs 语料 0.00-0.06）。命中则同中心同方向收窄。
            let mut tightened = false;
            if self.cfg.unclip_margin_thresh > 0.0 && b.tight_h > 0.0 && !pred.f32.is_empty() {
                let margin = box_margin_clutter(b, &pred.f32, nh, nw, crop_src);
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
                        tightened = true;
                    }
                }
            }
            let crop = crop_text_box(crop_src, &pts);
            let wr = batch_ratio(&crop, self.cfg.rec_height, self.cfg.rec_pad_min_h);
            (crop, wr, tightened, pts)
        });
        let mut crops: Vec<Image> = Vec::with_capacity(boxes.len());
        let mut wh_ratio: Vec<f32> = Vec::with_capacity(boxes.len());
        for (i, (crop, wr, tightened, pts)) in cropped.into_iter().enumerate() {
            if tightened {
                // 收紧框写回 boxes[i]：不只是显示——区域重试的裁剪区域
                // 从这些 pts 计算，留肥框会把刚剔除的杂波又包回重试区域。
                boxes[i].pts = pts;
                res.num_decluttered += 1;
            }
            crops.push(crop);
            wh_ratio.push(wr);
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

        // ---- 方向分类：并入下方逐行扇出（每行先分类后识别，见 run_batches）
        // 旧实现是独立的一串串行 cls 批（~8.5 ms 纯串行），现在每行在自己
        // 的线程里先做 cls 再做 rec——分类结论直接用于该行的裁剪与旋转，
        // 不需要跨线程回写 `crops`（core 是 forbid(unsafe_code)）。
        let mut flipped = vec![false; crops.len()];
        // 方向分类器对每行的把握（两类最大概率）。没有 cls 时恒 1.0
        //（= 不存在方向疑问，第二意见不会触发）。见 `cls_retry_conf`。
        let mut cls_conf = vec![1.0f32; crops.len()];
        // `QPPOCR_CLS_THRESH` 覆写。**img-028 的形态**：三条 180° 行里
        // 一条 p1=0.9956 翻对了，另两条 p1=0.7075 / 0.6876 —— 分类器
        // **倾向是对的**（p1 > p0），只是没跨过 0.9 的门 ⇒ 原样喂进 rec
        // 吐出 `'n ono nnd dern n n ne n2'`。这个门到底是保护还是误杀，
        // 只能扫出来看：降门会不会把 GT-0 的行翻错。见 acc_cls.py。
        let thresh = match std::env::var("QPPOCR_CLS_THRESH") {
            Ok(v) => v.trim().parse::<f32>().unwrap_or(self.cfg.cls_thresh),
            Err(_) => self.cfg.cls_thresh,
        };

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
        //
        // ★ 批次划分：行数少时切细到「每行一批」，逐行 fan-out 吃满并行
        //   ——行与行完全独立，每行在自己线程里**串行**执行的墙钟 ≈ 单行
        //   耗时（~14 ms），而「批串行 + 批内并行」在 16 线程上只拿到
        //   3.2x（fork 协调开销盖过了收益）。行数多时仍按 rec_batch 分组
        //   （批内存与 padding 效率）。
        let rec_batch_cfg = std::env::var("QPPOCR_REC_BATCH")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(self.cfg.rec_batch);
        let per_batch = if !self.rec.prefers_host_parallelism() {
            // GPU 会话：同桶行合批至 rec_chunk 上限（下方桶宽分组分支）——
            // 合的是**真实**行，省的是每轮编码/提交开销，不浪费算力。
            // **不再**顺带补齐批维：补齐只对读 `params[0]` 早退的内核免费，
            //而批维折进 x 网格的算子（分类头）按补齐量白算。历史遗留的
            // 「补齐到 grain 换形状塌缩」在 Metal 上实测付 8 倍算力（见
            // MetalSession::batch_grain），计划体积也大 2.5–8 倍、逼出
            // 预算淘汰；退真实行数后计划变小反而全命中。
            std::env::var("QPPOCR_REC_CHUNK")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&v| v > 0)
                .unwrap_or(self.rec.rec_chunk())
        } else if std::env::var_os("QPPOCR_REC_BATCH_NOCLAMP").is_some() {
            rec_batch_cfg.max(1)
        } else {
            rec_batch_cfg
                .min(order.len().div_ceil(qppocr_kernels::par::threads().max(1)))
                .max(1)
        };
        let batch_ratio = std::env::var("QPPOCR_REC_BATCH_RATIO")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .unwrap_or(self.cfg.rec_batch_ratio);
        let mut batches: Vec<(usize, usize)> = Vec::new();
        // 分桶粒度（与 rec_fn 同一条优先级链）：env 显式值 > 配置 > 设备
        // 上报（GPU 会话 64——每新形状付 ~10ms 计划重建，见 DeviceSession::
        // bucket_grain）。**1 是合法值**：显式给 1 = 关掉宽度取整，喂给
        // 模型的宽与 CPU 逐像素相同（诊断/精度对拍用——宽度取整会经无
        // mask 的注意力混入真步，是 CPU/GPU 置信度差异的一个来源）。
        let rec_grain = std::env::var("QPPOCR_REC_GRAIN")
            .ok()
            .and_then(|v| v.trim().parse::<i32>().ok())
            .filter(|&v| v > 0)
            .or_else(|| {
                let g = self.cfg.rec_width_grain;
                (g > 1).then_some(g)
            })
            .unwrap_or_else(|| self.rec.bucket_grain());
        // 行的**原始**宽（未取整）——分组判据用原始宽，img_w 取整只影响
        // 计划形状。
        let raw_w = |r: f32| -> i32 { (img_h as f32 * r) as i32 };
        let bucket_w = |r: f32| -> i32 {
            let w = raw_w(r);
            if rec_grain > 1 {
                (w + rec_grain - 1) / rec_grain * rec_grain
            } else {
                w
            }
        };
        // ── 分组判据：绝对桶（旧）vs 相对跨度（新，默认）────────────────
        //
        // 旧的 `bucket_w(a) == bucket_w(b)` 要求两行落进**同一个 32 px
        // 桶**，而桶边界是任意的：224 和 256 是相邻两桶 ⇒ 永远合不了，
        // 尽管只差 32 px。实测（`QPPOCR_GPU_PROF=1` 的 `[metal] run#`）
        // 一行一 run 的图里 10 行要跑 **9 次**，每个 run 付一次 ~2.3 ms
        // 的设备端地板（173 条 dispatch 的启动延迟，见
        // `bench/gpuhealth/logs/rec_grain_20261011.md`）。
        //
        // 新判据改成**相对跨度**：组内 `max(w) - min(w) <= grain` 就继续
        // 并进来，批宽 = 组内最大宽。行的宽是有序的（order 按比例排序），
        // 所以跨度就是 `w[end] - w[beg]`。
        //
        // **填充上界与旧判据完全相同**（都是 ≤ grain px，因为组宽取组内
        // 最大值、组内最小者至少差 grain 内），但能多合：上面那张图的 9
        // 个桶变成 {224,256,256}{288}{384,416}{448}{896}{992,1024} 六组。
        //
        // `QPPOCR_REC_BUCKET=1` 退回旧判据（A/B 用；也是唯一需要它的场合
        // ——两者填充上界同量级，新判据只是不再被桶边界挡住）。
        let legacy_bucket = std::env::var_os("QPPOCR_REC_BUCKET").is_some();
        let mut beg = 0usize;
        while beg < order.len() {
            let mut end = beg + 1;
            if rec_grain > 1 && legacy_bucket {
                // 旧：同桶行合批。order 按比例排序，同桶行连续。
                let wb = bucket_w(wh_ratio[order[beg]]);
                while end < order.len()
                    && end - beg < per_batch
                    && bucket_w(wh_ratio[order[end]]) == wb
                {
                    end += 1;
                }
            } else if rec_grain > 1 {
                // 新：组内跨度 ≤ grain 就并（批宽 = 组内最大宽）。
                let w0 = raw_w(wh_ratio[order[beg]]);
                while end < order.len()
                    && end - beg < per_batch
                    && raw_w(wh_ratio[order[end]]) - w0 <= rec_grain
                {
                    end += 1;
                }
            } else if batch_ratio > 0.0 {
                let base = wh_ratio[order[beg]].max(1e-6);
                while end < order.len()
                    && end - beg < per_batch
                    && wh_ratio[order[end]] as f64 <= base as f64 * batch_ratio
                {
                    end += 1;
                }
            } else {
                end = (beg + per_batch).min(order.len());
            }
            batches.push((beg, end));
            beg = end;
        }

        let t_ri = now();
        // 每行的处理裁剪（cls 判定倒置时是旋转过的副本）
        let mut line_crop: Vec<std::borrow::Cow<'_, Image>> =
            crops.iter().map(std::borrow::Cow::Borrowed).collect();
        if let Some(cls) = &self.cls {
            let (ch, cw) = (self.cfg.cls_height, self.cfg.cls_width);
            // cls 也用同一套扇出：按 `cls_batch` 行一组切批，批间扇出、批内
            // 串行。cls 画布固定 48×192，批起来只是把 N 填进批次维。
            // GPU：批内并行由设备吃掉，下限 32（cls_batch=1 是 CPU 的取舍，
            // 批化在 CPU 上 4.12→11.95ms 被回退，GPU 反向——逐行提交纯亏）。
            //
            // **下限 32（2026-10-11 由 8 抬到 32）**：每多切一批就多付一个
            // 设备端地板（cls 计划 104 条 dispatch，1 次 run 约 2.4~3.2 ms，
            // 而 run 内从 2 行加到 8 行只贵 ~0.4 ms ⇒ 地板是大头）。
            // 语料统计（corpus100 small，每图 6~17 行）：per=8 → 1.73 run/图、
            // per=16 → 1.02、**per=32 → 1.00（饱和）**。直测（一图一进程，
            // 交错 3 轮 × 5 图）：cls 段 **3.17 → 2.44 ms/图**，3/3 同号。
            // `bench` 端到端 A/B 同号但落在噪声里（−1.40/−0.40/+1.00）。
            //
            // **逐位不变**：批维不参与任何归约，且**不补齐**（只把真实行填进
            // 批次维，没有空行白算）⇒ 与 per=8 的 322 行输出逐位一致
            // （含置信度 4 位小数，`grain_acc.py` 一图一进程对拍）。
            // 顺带：per 越大**计划形状越少**（10 行图：per=8 要 (8)+(2) 两个
            // 形状，per=32 只要 (10)）。`QPPOCR_CLS_BATCH` 覆写整条表达式。
            let per = if !cls.prefers_host_parallelism() {
                std::env::var("QPPOCR_CLS_BATCH")
                    .ok()
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .filter(|&v| v > 0)
                    .unwrap_or_else(|| self.cfg.cls_batch.max(32))
            } else {
                self.cfg.cls_batch.max(1)
            };
            let cls_batches: Vec<(usize, usize)> = (0..crops.len())
                .step_by(per)
                .map(|b| (b, (b + per).min(crops.len())))
                .collect();
            // 返回 `(是否翻转, 两类最大概率)`。第二项是给下面的「第二意见」
            // 用的：`max(p0,p1) < cls_thresh` 就是「分类器倾向某一边但没到门」。
            let cls_fn = |b: usize, e: usize| -> Result<Vec<(bool, f32)>> {
                let n = e - b;
                let stride = 3 * (ch as usize) * (cw as usize);
                let mut buf = F32Buf::with_zeroed(n * stride);
                for (k, i) in (b..e).enumerate() {
                    let view = if self.cfg.cls_window {
                        cls_view(&crops[i], cw, ch)
                    } else {
                        crops[i].clone()
                    };
                    pack_crop(&view, ch, cw, &mut buf[k * stride..], false);
                }
                let out = cls.run(vec![(
                    self.cls_in.clone(),
                    Tensor {
                        name: String::new(),
                        shape: vec![n as i64, 3, ch as i64, cw as i64],
                        dtype: DType::F32,
                        f32: buf,
                        i64: Vec::new(),
                    },
                )])?;
                let o = &out[0]; // [n, 2]
                // `QPPOCR_CLS_DEBUG=1`：逐行打印分类器读数。**这就是当初
                // 找到根因的工具**——`img-028` 的两条 180° 行给的是
                // `p1=0.7075 / 0.6876`，「倾向对但没到门」，于是倒置裁切
                // 原样进了 rec。整套 `bench/gpuhealth/cls_sweep.py` /
                // `cls_thresh_ab.py` 都靠这一行输出。
                if std::env::var("QPPOCR_CLS_DEBUG").is_ok() {
                    for k in 0..n {
                        let c = &crops[b + k];
                        let (p0, p1) = (o.f32[k * 2], o.f32[k * 2 + 1]);
                        let ratio = c.w as f32 / c.h.max(1) as f32;
                        eprintln!(
                            "[cls] 行{} 裁{}x{} ({:.2}:1) p0={:.4} p1={:.4} 判定={} 存疑={}",
                            b + k,
                            c.w,
                            c.h,
                            ratio,
                            p0,
                            p1,
                            p1 > p0 && p1 >= thresh,
                            p0.max(p1) < thresh
                        );
                    }
                }
                Ok((0..n)
                    .map(|k| {
                        let (p0, p1) = (o.f32[k * 2], o.f32[k * 2 + 1]);
                        (p1 > p0 && p1 >= thresh, p0.max(p1))
                    })
                    .collect())
            };
            // 设备会话不受益于宿主侧扇出（GPU 内部已并行，N 个小批的
            // 线程 spawn + 并发提交是纯开销）——退化为串行循环，与
            // `run_batches` 的单线程分支同构。
            let decisions = if cls.prefers_host_parallelism() {
                Self::run_batches(&cls_batches, cls_fn)?
            } else {
                let mut out = Vec::with_capacity(cls_batches.len());
                for &(b, e) in &cls_batches {
                    out.push(cls_fn(b, e)?);
                }
                out
            };
            let decisions: Vec<(bool, f32)> = decisions.into_iter().flatten().collect();
            // 分类器对该行的把握（两类最大概率）。`< cls_thresh` ⇒ 存疑，
            // 是下面「第二意见」的准入条件之一。
            cls_conf = decisions.iter().map(|&(_, c)| c).collect();
            for (i, &(flip, _)) in decisions.iter().enumerate() {
                if flip {
                    flipped[i] = true;
                    res.num_flipped += 1;
                    line_crop[i] = std::borrow::Cow::Owned(rotate_image(&crops[i], 180));
                }
            }
            res.timings.cls_ms = t_ri.elapsed_ms();
        }
        let t_ri = now(); // ★ 重新起表：cls 的墙钟不能算进 rec
        // 打包+提交（异步）：GPU 批间流水的前半——pack(k+1) 与 GPU(k)
        // 执行重叠；CPU 会话的 run_deferred 默认实现 = 同步 run（语义等价）。
        // `ord` 是要跑的行号（主路径 = `order` 的一段，第二意见 = 那一小撮
        // 重试行）；`src` 是**实际喂给 rec 的裁切**（第二意见会把它换成
        // 翻转版），`flp` 与之配套记录该行此刻是否被翻转（决定 `rotation`
        // 与每字坐标的逆变换）。
        let rec_stage = |ord: &[usize],
                         src: &[std::borrow::Cow<'_, Image>]|
         -> Result<(Box<dyn crate::device::DeferredRun + Send>, i32)> {
            let mut max_wh_ratio = self.cfg.rec_min_width as f32 / img_h as f32;
            for &i in ord {
                max_wh_ratio = max_wh_ratio.max(wh_ratio[i]);
            }
            let mut img_w = (img_h as f32 * max_wh_ratio) as i32;
            if rec_grain > 1 {
                img_w = (img_w + rec_grain - 1) / rec_grain * rec_grain;
            }

            let bsz = ord.len();
            // 批维 = 真实行数（bsz）：**不补齐**。补齐只在所有内核都读
            // real_n 早退时才免费，而折进网格的算子按补齐量白算——Metal
            // 实测付 8 倍。形状多样性靠桶宽分组（rec_grain）压，不靠补齐。
            // CPU 会话批维即行数，同理。
            let mut batch_f32 = F32Buf::with_zeroed(bsz * 3 * (img_h as usize) * (img_w as usize));
            let t_pack = std::time::Instant::now();
            for (k, &i) in ord.iter().enumerate() {
                let c = &src[i];
                pack_crop(
                    c,
                    img_h,
                    img_w,
                    &mut batch_f32[k * 3 * (img_h as usize) * (img_w as usize)..],
                    crop_pads(c, img_h, self.cfg.rec_pad_min_h),
                );
            }
            if std::env::var_os("QPPOCR_GPU_BUILD_TIME").is_some() {
                eprintln!(
                    "[pack] {bsz} 行 → {img_w} 宽：{} ms",
                    t_pack.elapsed().as_secs_f64() * 1000.0
                );
            }

            let deferred = self.rec.run_deferred(vec![(
                self.rec_in.clone(),
                Tensor {
                    name: String::new(),
                    shape: vec![bsz as i64, 3, img_h as i64, img_w as i64],
                    dtype: DType::F32,
                    f32: batch_f32,
                    i64: Vec::new(),
                },
            )])?;
            Ok((deferred, img_w))
        };
        // 解码（纯 CPU，与设备无关）：argmax 对或概率矩阵 → 文本/置信度/
        // 每字坐标。img_w 是该批的打包宽（时间步→列映射要用）。
        let rec_decode = |out: Vec<Tensor>,
                          ord: &[usize],
                          src: &[std::borrow::Cow<'_, Image>],
                          flp: &[bool],
                          img_w: i32|
         -> Result<Vec<RecLineOut>> {
            let mut batch_lines = Vec::with_capacity(ord.len());
            let o = &out[0]; // [B, T, C]（或 GPU argmax 出口的 [B, T, 2] 对）
            let t_len = o.shape[1] as usize;
            // GPU argmax 出口：每时间步 (val, idx) 对，解码语义与概率
            // 路径逐字相同（ctc_decode_pairs）；CPU 会话恒概率矩阵。
            let pairs = self.rec.rec_argmax_pairs();
            let c_len = if pairs { 2 } else { o.shape[2] as usize };
            if !pairs {
                // ★ 字典/模型配对硬校验。换错字典时 C 与字符表长度错位，
                // ctc_decode 的越界索引是静默跳过——整表错位且不报错。
                let c_real = o.shape[2] as usize;
                if c_real != self.charset.len() {
                    return Err(crate::error::Error::Graph(format!(
                        "识别输出类别数 {c_real} 与字符表 {} 不符：字典与 rec 模型不配对\n  期望行数：tiny 档 6904、small/medium 档 18708",
                        self.charset.len()
                    )));
                }
            }
            let want_gaps = self.cfg.rec_space_gap > 0.0;
            for (k, &i) in ord.iter().enumerate() {
                let (mut text, conf, marks) = if pairs {
                    ctc_decode_pairs(&o.f32[k * t_len * 2..], t_len, &self.charset)
                } else {
                    ctc_decode(&o.f32[k * t_len * c_len..], t_len, c_len, &self.charset)
                };
                // 时间步 → 裁剪图 x 的左缘（像素空格与每字坐标共用这套映射）。
                let cr = &src[i];
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
                    flp[i],
                    &boxes[i].pts,
                    &to_image,
                );
                batch_lines.push(RecLineOut {
                    idx: i,
                    text,
                    chars,
                    confidence: conf,
                });
            }
            Ok(batch_lines)
        };
        // 收账+解码（两路共用）：GPU 的 complete = 等信号+读回。
        let rec_collect = |staged: (Box<dyn crate::device::DeferredRun + Send>, i32),
                           ord: &[usize],
                           src: &[std::borrow::Cow<'_, Image>],
                           flp: &[bool]|
         -> Result<Vec<RecLineOut>> {
            let (deferred, img_w) = staged;
            let out = deferred.complete()?;
            rec_decode(out, ord, src, flp, img_w)
        };
        // 串行兼容路：CPU 会话 / 分片并行路径用（deferred 默认=同步）。
        let rec_fn = |beg: usize, end: usize| -> Result<Vec<RecLineOut>> {
            let (ord, src, flp) = (&order[beg..end], &line_crop[..], &flipped[..]);
            let staged = rec_stage(ord, src)?;
            rec_collect(staged, ord, src, flp)
        };
        // ★ 两种并行的切法只能二选一（池不支持嵌套 fork）：**行间并行**
        //   （一批一个线程、批内串行，即 `run_batches`）还是**行内并行**
        //   （批次顺序执行、算子内部 fork 铺满池）。100 图逐图交错实测，
        //   行间并行在两档都赢，行内并行最强的一档也慢 1.06x（small）到
        //   1.43x（tiny）——rec 的算子虽大，但每行 ~80 个算子顺序依赖，
        //   每次 fork ~97 us 的协调开销乘上去不划算。换方案前先看这条。
        let shards = self.cfg.rec_shards;
        let outs = if !self.rec.prefers_host_parallelism() {
            // 设备会话：批间流水——pack(k+1)+提交(k+1) 与 GPU(k) 执行
            // 重叠，收账(k-1)（等信号+读回+解码）也与 GPU(k) 重叠。
            // GPU 空隙从「pack+提交+等待+读回+解码」缩到 pack+提交
            //（实测批均 ~1ms 对 ~4ms GPU）。同桶连续批 = 同一计划
            //（输入/输出区同址）：先收账在飞批再提交，防写穿。
            let mut outs: Vec<Vec<RecLineOut>> = Vec::with_capacity(batches.len());
            /// 在飞批：(staged 句柄, order 区间 lo, hi)
            struct Pending(
                Option<(Box<dyn crate::device::DeferredRun + Send>, i32)>,
                usize,
                usize,
            );
            let mut pending: Option<Pending> = None;
            for &(b, e) in &batches {
                // 同桶连续批不再「先收账再提交」：GPU 会话对同形状双批在飞
                // 自动分流影子计划（独立 arena/CB，零共享）——批 k+1 提交
                // 与 GPU(k) 重叠，收账 k-1 与 GPU(k) 重叠（真正的批间流水；
                // 旧的防写穿串行化曾把 rec_infer 拖成批串行和）。
                let staged = rec_stage(&order[b..e], &line_crop)?;
                if let Some(p) = pending.take() {
                    outs.push(rec_collect(
                        p.0.unwrap(),
                        &order[p.1..p.2],
                        &line_crop,
                        &flipped,
                    )?);
                }
                pending = Some(Pending(Some(staged), b, e));
            }
            if let Some(p) = pending.take() {
                outs.push(rec_collect(
                    p.0.unwrap(),
                    &order[p.1..p.2],
                    &line_crop,
                    &flipped,
                )?);
            }
            outs
        } else if shards > 1 {
            Self::run_batches_sharded(&batches, rec_fn, shards)?
        } else {
            Self::run_batches(&batches, rec_fn)?
        };
        res.timings.rec_infer_ms += t_ri.elapsed_ms();

        // ---- 合并（串行）：并行段只产出结果，这里统一写回 ----
        for batch_out in outs {
            for lo in batch_out {
                let i = lo.idx;
                lines[i].chars = lo.chars;
                lines[i].text = lo.text;
                lines[i].confidence = lo.confidence;
                lines[i].rotation = if flipped[i] { 180 } else { 0 };
                to_image(&boxes[i].pts, &mut lines[i].pts);
            }
        }

        // ---- 方向存疑的行取「第二意见」----
        // 先决条件两条，缺一不可：
        //   ① `cls` 对该行**存疑**——两类最大概率没到 `cls_thresh`。这不是
        //      「判错了」，是「倾向某一边但不敢拍板」。GT-180 的行从没低过
        //      p1=0.65，可硬门 0.9 还是漏翻了 6 行（gen100/small 实测）。
        //   ② 该行**首次解码置信度低**——读出来就不像话。
        // 判据为什么不能只靠 ①：同一段概率区间里混着 6 行该翻（GT-180）和
        // 5 行不该翻（GT-0/±90），p1 完全交错，**任何门都分不开**。而 rec
        // 自己的置信度把它们切得很干净：13/13 无例外，读错的方向 0.53~0.72、
        // 读对的方向 0.98~1.00。所以判据从「分类器有多自信」换成「识别器
        // 读出来像不像话」。
        // 条件 ② 顺带把代价锁死：只对**已经读坏了**的行多跑一次 rec。
        // 工具与实测：`bench/gpuhealth/cls_sweep.py` / `cls_thresh_ab.py`。
        let retry_conf = match std::env::var("QPPOCR_CLS_RETRY") {
            Ok(v) => v.trim().parse::<f32>().unwrap_or(self.cfg.cls_retry_conf),
            Err(_) => self.cfg.cls_retry_conf,
        };
        if retry_conf > 0.0 {
            let retry: Vec<usize> = (0..lines.len())
                .filter(|&i| wants_cls_retry(cls_conf[i], thresh, lines[i].confidence, retry_conf))
                .collect();
            if !retry.is_empty() {
                // 翻转版裁切：本来翻了就取原图，本来没翻就转 180°。
                // 另起 `src`/`flp` 而不改 `line_crop`/`flipped`——主路径的
                // 结果（`lines[i].pts` 等）已经写好了，这里只是做取舍。
                //
                // 注：**今天 `flipped[i]` 在这里恒为 false**——`cls_conf[i]`
                // 是两类最大概率，`< cls_thresh` 蕴含 `p1 < cls_thresh`，
                // 而翻判定要求 `p1 >= cls_thresh`，两者互斥（对任意
                // `cls_thresh` 都成立，不只是 0.9）。也就是说第二意见
                // **只在「分类器说不翻」的行上触发**，正是 8 条实测全中的
                // 那一批。仍按两向写，是为了将来决策规则变了不至于静默错。
                let mut src: Vec<std::borrow::Cow<'_, Image>> =
                    crops.iter().map(std::borrow::Cow::Borrowed).collect();
                for &i in &retry {
                    src[i] = if flipped[i] {
                        std::borrow::Cow::Borrowed(&crops[i])
                    } else {
                        std::borrow::Cow::Owned(rotate_image(&crops[i], 180))
                    };
                }
                let mut flp = flipped.clone();
                for &i in &retry {
                    flp[i] = !flp[i];
                }
                res.num_cls_retry += retry.len();
                let staged = rec_stage(&retry, &src)?;
                let outs2 = rec_collect(staged, &retry, &src, &flp)?;
                for lo in outs2 {
                    let i = lo.idx;
                    // **只在更自信时才采纳**——这一条就是安全性来源：
                    // 翻转版若不比原版像话，一切照旧，绝不会把读对的行改坏。
                    if lo.confidence > lines[i].confidence {
                        lines[i].chars = lo.chars;
                        lines[i].text = lo.text;
                        lines[i].confidence = lo.confidence;
                        lines[i].rotation = if flp[i] { 180 } else { 0 };
                        if flp[i] {
                            res.num_flipped += 1;
                        } else {
                            res.num_flipped -= 1;
                        }
                        flipped[i] = flp[i];
                        res.num_cls_retry_flipped += 1;
                    }
                }
            }
        }
        res.timings.total_ms = t_start.elapsed_ms();

        // 读不出的框保留不丢（见 TextLine::text 文档）
        res.num_unread = lines.iter().filter(|l| l.text.is_empty()).count();
        res.lines = lines;
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::wants_cls_retry;

    // 三个输入都取自 gen100/small 的实测（`cls_sweep.py` 的表 +
    // `cls_thresh_ab.py` 的两臂置信度），门取默认 0.9 / 0.85。

    #[test]
    fn 存疑且首读低置信才取第二意见() {
        // img-028 行3：cls 存疑（p1=0.7075 没到 0.9），首读 garbage 0.7195
        assert!(wants_cls_retry(0.7075, 0.9, 0.7195, 0.85));
        // img-052 行11：存疑 0.8339，首读 0.6067
        assert!(wants_cls_retry(0.8339, 0.9, 0.6067, 0.85));
    }

    #[test]
    fn 首读读得好就不动它() {
        // img-054 行1：cls 也存疑（0.7828），但首读 0.9897 —— **这是最关键
        // 的一条**：不带条件 2 的话这行会被翻倒，把读对的改成垃圾。
        assert!(!wants_cls_retry(0.7828, 0.9, 0.9897, 0.85));
        // img-096 行3：存疑 0.7811、首读 0.9929
        assert!(!wants_cls_retry(0.7811, 0.9, 0.9929, 0.85));
    }

    #[test]
    fn 分类器自信时不取第二意见() {
        // 置信度 ≥ 门 ⇒ 分类器已经拍了板，不管 rec 读出什么。
        assert!(!wants_cls_retry(0.9956, 0.9, 0.1, 0.85));
        assert!(!wants_cls_retry(0.98, 0.9, 0.1, 0.85));
    }

    #[test]
    fn 关掉开关就一行都不碰() {
        let off = 0.0;
        assert!(!wants_cls_retry(0.5, 0.9, 0.0, off));
        assert!(!wants_cls_retry(0.5, 0.9, 0.84, off));
    }

    #[test]
    fn 边界都是严格小于() {
        // 恰好等于门 ⇒ 不算存疑（与 cls 判定里的 `>= thresh` 互斥，不重叠）
        assert!(!wants_cls_retry(0.9, 0.9, 0.5, 0.85));
        // 首读恰好等于 retry_conf ⇒ 不重试
        assert!(!wants_cls_retry(0.5, 0.9, 0.85, 0.85));
    }
}
