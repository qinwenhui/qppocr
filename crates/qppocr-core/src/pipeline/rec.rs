//! CTC 贪心解码与像素空格（设计文档 对应部分）。

use super::image::Image;

/// ctc_decode 输出的每个字符：它在哪个时间步发射、在返回串里的字节起点。
/// 时间步能把「两个字之间有没有空格」映射回裁剪图的列，从而问图像那列
/// 是不是空的——见 rec_space_gap。
#[derive(Clone, Copy, Debug)]
pub struct DecodeMark {
    /// 时间步（[0, T)）。
    pub step: usize,
    /// 字符在解码串里的字节偏移。
    pub off: usize,
}

/// `add_gap_spaces` 插入的一个空格：在最终文本里的字节偏移 + 裁剪图上
/// 空白 run 的 x 区间（`TextLine::chars` 给空格真实框用）。
pub struct SpaceSpan {
    /// 最终文本字节偏移（指向插入的空格）。
    pub off: usize,
    /// 空白 run 左缘（裁剪图像素）。
    pub x0: i32,
    /// 空白 run 右缘。
    pub x1: i32,
}

/// 贪心 CTC：`best != 0 && best != prev` 时追加字符。
/// ★ 字典末尾的空格项是**能输出的**——不要在解码阶段剥掉它。
///
/// argmax 用两趟标量（与 基准的 AVX2 路径语义一致：首个严格最大者胜，
/// 平局取最小下标）。注意  在 C≥8 时走向量 max、再找第一个相等位——
/// f32 max 的 NaN 语义在合法 logits 下无差异，标量逐元素 `>` 的首胜
/// 与之等价。
pub fn ctc_decode(
    logits: &[f32],
    t_len: usize,
    c_len: usize,
    charset: &[String],
) -> (String, f32, Vec<DecodeMark>) {
    let mut text = String::new();
    let mut conf_sum = 0f64;
    let mut conf_n = 0usize;
    let mut prev: i64 = -1;
    let mut marks = Vec::new();
    for t in 0..t_len {
        let row = &logits[t * c_len..(t + 1) * c_len];
        let mut best = 0usize;
        let mut bestv = row[0];
        for (c, &v) in row.iter().enumerate().skip(1) {
            if v > bestv {
                bestv = v;
                best = c;
            }
        }
        let b = best as i64;
        if b != 0 && b != prev {
            if best < charset.len() {
                marks.push(DecodeMark {
                    step: t,
                    off: text.len(),
                });
                text.push_str(&charset[best]);
            }
            conf_sum += bestv as f64;
            conf_n += 1;
        }
        prev = b;
    }
    let conf = if conf_n > 0 {
        (conf_sum / conf_n as f64) as f32
    } else {
        0.0
    };
    (text, conf, marks)
}

// ---------------------------------------------------------------- 像素空格

/// 每列墨迹：该列最极端像素离背景多远（0..1）。**极性无关**——语料里
/// 深底白字与浅底黑字都有，背景是「这行像素里最多的值」，直方图中位数
/// 找它。
pub fn column_ink(c: &Image) -> Vec<f32> {
    let mut ink = vec![0f32; c.w.max(1) as usize];
    if c.w <= 0 || c.h <= 0 || c.data.is_empty() {
        return ink;
    }
    let mut hist = [0i64; 256];
    for y in 0..c.h {
        let row = c.row(y);
        for x in 0..c.w {
            let p = (x as usize) * 3;
            hist[((row[p] as i32 + row[p + 1] as i32 + row[p + 2] as i32) / 3) as usize] += 1;
        }
    }
    let total = (c.w as i64) * (c.h as i64);
    let mut bg = 128usize;
    let mut acc: i64 = 0;
    for (v, &h) in hist.iter().enumerate() {
        acc += h;
        if acc * 2 >= total {
            bg = v;
            break;
        }
    }
    for x in 0..c.w {
        let mut mx = 0i32;
        for y in 0..c.h {
            let row = c.row(y);
            let p = (x as usize) * 3;
            let g = (row[p] as i32 + row[p + 1] as i32 + row[p + 2] as i32) / 3;
            let d = if g > bg as i32 {
                g - bg as i32
            } else {
                bg as i32 - g
            };
            if d > mx {
                mx = d;
            }
        }
        ink[x as usize] = mx as f32 / 255.0;
    }
    ink
}

/// 离背景这么远的列算空。抗锯齿让字形附近总有少量墨，不能是 0；
/// 30/255 低于最淡的真实笔画、高于这些语料压缩噪声的地板。
const EMPTY_INK: f32 = 0.12;

/// UTF-8 字节串的首个码点（空/截断返回 0）。
fn utf8_first(s: &[u8]) -> u32 {
    if s.is_empty() {
        return 0;
    }
    let c = s[0];
    if c < 0x80 {
        return c as u32;
    }
    let extra = if c >= 0xF0 {
        3
    } else if c >= 0xE0 {
        2
    } else if c >= 0xC0 {
        1
    } else {
        0
    };
    if extra as usize + 1 > s.len() {
        return 0;
    }
    let mut cp = (c as u32) & (0xFFu32 >> (extra + 1));
    for &b in &s[1..1 + extra as usize] {
        cp = (cp << 6) | (b as u32 & 0x3F);
    }
    cp
}

/// 全角 CJK 标点。这些字形 1 em 宽但只画自己 em 框的一部分——逗号只
/// 墨左下角，其余空白读起来跟空格一样，没有绝对阈值能分开。测过：每个
/// 帮了合成语料的阈值都把空格插进 `第1行，包含` 的逗号后，把扫描语料
/// 从 92.3% 打到 17.3%。判间隙时字符已知，用它消歧而不是更宽的阈值。
fn is_cjk_punct(cp: u32) -> bool {
    (0x3000..=0x303F).contains(&cp) // 、。〈〉《》「」『』【】〔〕…
        || (0xFF01..=0xFF0F).contains(&cp) // ！＂＃＄％＆＇（）＊＋，－．／
        || (0xFF1A..=0xFF20).contains(&cp) // ：；＜＝＞？＠
        || (0xFF3B..=0xFF40).contains(&cp) // ［＼］＾＿｀
        || (0xFF5B..=0xFF65).contains(&cp) // ｛｜｝～｟｠｡｢｣
        || cp == 0x2018
        || cp == 0x2019
        || cp == 0x201C
        || cp == 0x201D
        || cp == 0x2014
        || cp == 0x2026
        || cp == 0x00B7
}

/// 两个解码字符之间的裁剪像素空了超过 `min_gap_px` 就插一个空格。
/// `cols[k]` 是字符 k 在裁剪像素里的起点（由 marks[k].step 与张量几何推出）。
pub fn add_gap_spaces(
    text: &str,
    marks: &[DecodeMark],
    cols: &[i32],
    ink: &[f32],
    crop_w: i32,
    min_gap_px: f32,
) -> (String, Vec<SpaceSpan>) {
    if marks.len() < 2 || cols.len() != marks.len() || text.is_empty() {
        return (text.to_string(), Vec::new());
    }
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len() + marks.len());
    let mut spaces = Vec::new();
    let mut prev_space = false;
    let mut prev_cp = 0u32;
    for (k, &m) in marks.iter().enumerate() {
        let end = if k + 1 < marks.len() {
            marks[k + 1].off
        } else {
            text.len()
        };
        let seg = &bytes[m.off..end];
        let mut cur_space = !seg.is_empty();
        for &q in seg {
            if q != b' ' && q != b'\t' {
                cur_space = false;
                break;
            }
        }
        let cur_cp = utf8_first(seg);

        // ★ 已输出的空格旁边不许再插：空格本身是解码字符、渲染为空白，
        // 它两侧的空白段各被当成一次间隙——没有这个守卫 native dependency
        // 会变 native  dependency。标点旁边同理（见 is_cjk_punct）。
        if k > 0 && !cur_space && !prev_space && !is_cjk_punct(cur_cp) && !is_cjk_punct(prev_cp) {
            let (b, a) = (cols[k - 1], cols[k]);
            if a > b {
                let (mut run, mut best) = (0i32, 0i32);
                let (mut best_x0, mut best_x1) = (b, b);
                for x in b..a {
                    if x < 0 || x >= crop_w || ink[x as usize] < EMPTY_INK {
                        run += 1;
                        if run > best {
                            best = run;
                            best_x0 = x + 1 - run;
                            best_x1 = x + 1;
                        }
                    } else {
                        run = 0;
                    }
                }
                if best as f32 >= min_gap_px {
                    // 记录插入位置与空白 run 的区间——每字坐标（TextLine::chars）
                    // 用它给空格一个真实的框，而不是借用相邻字符的边界。
                    spaces.push(SpaceSpan {
                        off: out.len(),
                        x0: best_x0,
                        x1: best_x1,
                    });
                    out.push(' ');
                }
            }
        }
        // Rust 的 String::push_str 需要 &str——seg 是合法 UTF-8 子串
        // （marks 的 off 都落在字符边界上：它们来自 push_str 的边界）。
        out.push_str(std::str::from_utf8(seg).unwrap_or(""));
        prev_space = cur_space;
        prev_cp = cur_cp;
    }
    (out, spaces)
}
