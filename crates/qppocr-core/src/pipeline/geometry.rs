//! 几何：凸包、最小面积外接矩形、四角排序、连通域、box_score、
//! DB 后处理与框合并（设计文档 对应部分）。

use super::image::Image;

/// 二维点。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pt {
    /// x。
    pub x: f32,
    /// y。
    pub y: f32,
}

/// 检测框：四角点（TL/TR/BR/BL，图像坐标）+ 分数 + 未扩张高度。
#[derive(Clone, Copy, Debug, Default)]
pub struct TextBox {
    /// 四角点。
    pub pts: [[f32; 2]; 4],
    /// DB 区域内概率均值。
    pub score: f32,
    /// unclip 扩张**前**的 DB 区域高度。边距杂波判定（unclip_margin_thresh）
    /// 需要它在框被证明是杂框时重建更紧的框。
    pub tight_h: f32,
}

fn cross(o: &Pt, a: &Pt, b: &Pt) -> f32 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// Andrew 单调链凸包。
pub fn convex_hull(mut p: Vec<Pt>) -> Vec<Pt> {
    let n = p.len();
    if n < 3 {
        return p;
    }
    p.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap()
            .then(a.y.partial_cmp(&b.y).unwrap())
    });
    let mut h = vec![Pt::default(); 2 * n];
    let mut k = 0usize;
    for &pt in &p {
        while k >= 2 && cross(&h[k - 2], &h[k - 1], &pt) <= 0.0 {
            k -= 1;
        }
        h[k] = pt;
        k += 1;
    }
    let t = k + 1;
    let mut i = n as isize - 2;
    while i >= 0 {
        let pt = p[i as usize];
        while k >= t && cross(&h[k - 2], &h[k - 1], &pt) <= 0.0 {
            k -= 1;
        }
        h[k] = pt;
        k += 1;
        i -= 1;
    }
    h.truncate(k - 1);
    h
}

/// 旋转卡壳最小面积外接矩形：返回 4 角（无序）与最短边。
pub fn min_area_rect(hull: &[Pt]) -> (Option<[Pt; 4]>, f32) {
    if hull.is_empty() {
        return (None, 0.0);
    }
    if hull.len() == 1 {
        return (Some([hull[0]; 4]), 0.0);
    }
    if hull.len() == 2 {
        return (Some([hull[0], hull[1], hull[1], hull[0]]), 0.0);
    }
    let mut best_area = 3.4e38f32;
    let mut best_min = 0.0f32;
    let mut best = [Pt::default(); 4];
    let n = hull.len();
    for i in 0..n {
        let a = &hull[i];
        let b = &hull[(i + 1) % n];
        let mut ex = b.x - a.x;
        let mut ey = b.y - a.y;
        let len = (ex * ex + ey * ey).sqrt();
        if len < 1e-12 {
            continue;
        }
        ex /= len;
        ey /= len;
        // 所有点投影到 (ex,ey) 与其法向
        let (mut min_u, mut max_u, mut min_v, mut max_v) =
            (3.4e38f32, -3.4e38f32, 3.4e38f32, -3.4e38f32);
        for hj in hull {
            let u = hj.x * ex + hj.y * ey;
            let v = -hj.x * ey + hj.y * ex;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let w = max_u - min_u;
        let h = max_v - min_v;
        let area = w * h;
        if area < best_area {
            best_area = area;
            best_min = w.min(h);
            let us = [min_u, max_u, max_u, min_u];
            let vs = [min_v, min_v, max_v, max_v];
            for (k, ((u, v), bk)) in us.iter().zip(vs.iter()).zip(best.iter_mut()).enumerate() {
                let _ = k;
                bk.x = u * ex - v * ey;
                bk.y = u * ey + v * ex;
            }
        }
    }
    (Some(best), best_min)
}

/// 四角排序为 TL, TR, BR, BL（ppocr get_mini_boxes 的次序）。
pub fn order_box_tl_tr_br_bl(p: &mut [Pt; 4]) {
    // 按 x 排序（ std::sort 不稳定——四个点 x 相等是退化情形，忽略）
    p.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
    let i1 = if p[1].y > p[0].y { 0 } else { 1 };
    let i4 = if p[1].y > p[0].y { 1 } else { 0 };
    let i2 = if p[3].y > p[2].y { 2 } else { 3 };
    let i3 = if p[3].y > p[2].y { 3 } else { 2 };
    let r = [p[i1], p[i2], p[i3], p[i4]];
    p.copy_from_slice(&r);
}

/// 对切片并行 map（连通域彼此独立）。core 是 `forbid(unsafe_code)`，
/// 用「结果槽 + 原子领票」而不是裸指针切分。
pub(crate) fn par_map<T: Sync, U: Send, F>(items: &[T], f: F) -> Vec<U>
where
    F: Fn(&T) -> U + Sync,
{
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let n = items.len();
    let nthreads = qppocr_kernels::par::threads().min(n).max(1);
    if nthreads <= 1 {
        return items.iter().map(&f).collect();
    }
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<U>>> = (0..n).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..nthreads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    *slots[i].lock().unwrap() = Some(f(&items[i]));
                }
            });
        }
    });
    slots
        .into_iter()
        .filter_map(|m| m.into_inner().unwrap())
        .collect()
}

/// 8 连通域标记；每个连通域返回一个点表（带尺寸下限过滤）。
fn connected_components(mask: &[u8], hh: i32, ww: i32, min_pixels: usize) -> Vec<Vec<Pt>> {
    let mut label = vec![-1i32; (hh as usize) * (ww as usize)];
    let mut out: Vec<Vec<Pt>> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut cur: Vec<Pt> = Vec::new();
    for y0 in 0..hh {
        for x0 in 0..ww {
            let idx = (y0 as usize) * (ww as usize) + x0 as usize;
            if mask[idx] == 0 || label[idx] >= 0 {
                continue;
            }
            let id = out.len() as i32;
            cur.clear();
            stack.clear();
            stack.push(idx);
            label[idx] = id;
            while let Some(p) = stack.pop() {
                let (py, px) = ((p / ww as usize) as i32, (p % ww as usize) as i32);
                cur.push(Pt {
                    x: px as f32,
                    y: py as f32,
                });
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let ny = py + dy;
                        let nx = px + dx;
                        if ny < 0 || ny >= hh || nx < 0 || nx >= ww {
                            continue;
                        }
                        let ni = (ny as usize) * (ww as usize) + nx as usize;
                        if mask[ni] != 0 && label[ni] < 0 {
                            label[ni] = id;
                            stack.push(ni);
                        }
                    }
                }
            }
            if cur.len() >= min_pixels {
                out.push(std::mem::take(&mut cur));
            }
        }
    }
    out
}

/// 多边形内部的 pred 均值（ppocr box_score_fast）。扫描线奇偶填充，
/// 覆盖与 cv2.fillPoly 一致；累加用 f64。
fn box_score_fast(pred: &[f32], hh: i32, ww: i32, bpts: &[Pt; 4]) -> f32 {
    let mut xmin = bpts.iter().map(|p| p.x).fold(f32::MAX, f32::min).floor() as i32;
    let mut xmax = bpts.iter().map(|p| p.x).fold(f32::MIN, f32::max).ceil() as i32;
    let mut ymin = bpts.iter().map(|p| p.y).fold(f32::MAX, f32::min).floor() as i32;
    let mut ymax = bpts.iter().map(|p| p.y).fold(f32::MIN, f32::max).ceil() as i32;
    xmin = xmin.clamp(0, ww - 1);
    xmax = xmax.clamp(0, ww - 1);
    ymin = ymin.clamp(0, hh - 1);
    ymax = ymax.clamp(0, hh - 1);

    let mut sum = 0f64;
    let mut count = 0i64;
    for y in ymin..=ymax {
        let sy = y as f32 + 0.5;
        let mut xsect = [0f32; 8];
        let mut nx = 0;
        for i in 0..4 {
            let a = &bpts[i];
            let b = &bpts[(i + 1) % 4];
            if (a.y <= sy && b.y > sy) || (b.y <= sy && a.y > sy) {
                let t = (sy - a.y) / (b.y - a.y);
                xsect[nx] = a.x + t * (b.x - a.x);
                nx += 1;
            }
        }
        let xs = &mut xsect[..nx];
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut i = 0;
        while i + 1 < nx {
            let xa = (xs[i] - 0.5).ceil() as i32;
            let xb = (xs[i + 1] - 0.5).floor() as i32;
            for x in xa..=xb {
                if x < xmin || x > xmax {
                    continue;
                }
                sum += pred[(y as usize) * (ww as usize) + x as usize] as f64;
                count += 1;
            }
            i += 2;
        }
    }
    if count == 0 {
        return 0.0;
    }
    (sum / count as f64) as f32
}

/// DB 后处理：二值化 → 膨胀 → 连通域 → 每域最小面积矩形 → 打分过滤 →
/// unclip 扩张。`(nh, nw)` 是概率图尺寸；`(dest_h, dest_w)` 是输出坐标
/// 所在的工作图尺寸。
#[allow(clippy::too_many_arguments)]
pub fn db_postprocess(
    pred: &[f32],
    nh: i32,
    nw: i32,
    dest_w: i32,
    dest_h: i32,
    thresh: f32,
    box_thresh: f32,
    unclip_ratio: f32,
    unclip_perp: f32,
    max_candidates: usize,
    use_dilation: bool,
) -> Vec<TextBox> {
    let dbg = std::env::var("QPPOCR_DEBUG_DB").is_ok();
    let t0 = std::time::Instant::now();
    let n = (nh as usize) * (nw as usize);
    let mut mask = vec![0u8; n];
    for (m, &p) in mask.iter_mut().zip(pred.iter()) {
        *m = if p > thresh { 1 } else { 0 };
    }
    let t_bin = t0.elapsed().as_secs_f64() * 1000.0;

    if use_dilation {
        // cv2.dilate 2x2 核：anchor=(1,1)，窗口 [y-1..y]×[x-1..x]，
        // 越界样本忽略（morphology 边界语义）。走 kernels 的 AVX2 逐行
        // 并行版（max 可交换 ⇒ 与标量**逐位相同**）；标量版 832×832
        // 掩码实测 3.8 ms，是 DB 后处理最大的一项。
        let mut dil = vec![0u8; n];
        qppocr_kernels::resize::dilate2x2_max(&mask, &mut dil, nh as usize, nw as usize);
        mask = dil;
    }

    let t_dil = t0.elapsed().as_secs_f64() * 1000.0;
    let mut comps = connected_components(&mask, nh, nw, 1);
    let t_cc = t0.elapsed().as_secs_f64() * 1000.0;
    comps.truncate(max_candidates);
    if dbg {
        eprintln!(
            "[db] binarize={t_bin:.2} dilate={:.2} cc={:.2} comps={}",
            t_dil - t_bin,
            t_cc - t_dil,
            comps.len()
        );
    }

    let min_size = 3.0f32;
    let mut boxes = Vec::new();
    // ★ 凸包并行预算：每个连通域的凸包要对它全部像素排序（一行 5000+ 点），
    //   而连通域彼此独立——实测这是 per-box 的 92%（2.17/2.36 ms）。
    let hulls: Vec<Vec<Pt>> = par_map(&comps, |c| convex_hull(c.clone()));
    for (ci, _comp) in comps.iter().enumerate() {
        let hull = &hulls[ci];
        let (Some(mut rect), sside) = min_area_rect(hull) else {
            continue;
        };
        if sside < min_size {
            continue;
        }
        order_box_tl_tr_br_bl(&mut rect);

        let score = box_score_fast(pred, nh, nw, &rect);
        if score < box_thresh {
            continue;
        }

        // unclip：输入总是矩形，圆角偏移 d 就是四边各扩 d（最小面积矩形不变）
        let e01 = (rect[1].x - rect[0].x).hypot(rect[1].y - rect[0].y);
        let e12 = (rect[2].x - rect[1].x).hypot(rect[2].y - rect[1].y);
        let (w, h) = (e01, e12);
        let area = w * h;
        let perim = 2.0 * (w + h);
        let dist = if perim > 1e-6 {
            area * unclip_ratio / perim
        } else {
            0.0
        };
        let cx = (rect[0].x + rect[1].x + rect[2].x + rect[3].x) * 0.25;
        let cy = (rect[0].y + rect[1].y + rect[2].y + rect[3].y) * 0.25;
        let (mut ex, mut ey) = (Pt { x: 0.0, y: 0.0 }, Pt { x: 0.0, y: 0.0 });
        if w > 1e-6 {
            ex.x = (rect[1].x - rect[0].x) / w;
            ex.y = (rect[1].y - rect[0].y) / w;
        }
        if h > 1e-6 {
            ey.x = (rect[2].x - rect[1].x) / h;
            ey.y = (rect[2].y - rect[1].y) / h;
        }
        let tight_h = h;
        let hw = w * 0.5 + dist;
        let hh = h * 0.5 + dist * unclip_perp;
        let sgn = [[-1.0f32, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let mut grown = [Pt::default(); 4];
        for (i, g) in grown.iter_mut().enumerate() {
            g.x = cx + sgn[i][0] * hw * ex.x + sgn[i][1] * hh * ey.x;
            g.y = cy + sgn[i][0] * hw * ex.y + sgn[i][1] * hh * ey.y;
        }
        let (Some(mut rect2), sside2) = min_area_rect(&convex_hull(grown.to_vec())) else {
            continue;
        };
        if sside2 < min_size + 2.0 {
            continue;
        }
        order_box_tl_tr_br_bl(&mut rect2);

        let mut tb = TextBox::default();
        tb.score = score;
        // 带回调用方坐标——裁剪决策在那里
        tb.tight_h = tight_h / nh as f32 * dest_h as f32;
        for (i, tb_pt) in tb.pts.iter_mut().enumerate() {
            let bx = (rect2[i].x / nw as f32 * dest_w as f32).round_ties_even();
            let by = (rect2[i].y / nh as f32 * dest_h as f32).round_ties_even();
            tb_pt[0] = bx.clamp(0.0, dest_w as f32);
            tb_pt[1] = by.clamp(0.0, dest_h as f32);
        }
        boxes.push(tb);
    }
    boxes
}

// ================================================================ 框合并

/// 轴对齐外接框。
struct BoxBounds {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

#[allow(dead_code)]
fn box_bounds(pts: &[[f32; 2]; 4]) -> BoxBounds {
    let mut b = BoxBounds {
        x0: 3.4e38,
        y0: 3.4e38,
        x1: -3.4e38,
        y1: -3.4e38,
    };
    for p in pts {
        b.x0 = b.x0.min(p[0]);
        b.x1 = b.x1.max(p[0]);
        b.y0 = b.y0.min(p[1]);
        b.y1 = b.y1.max(p[1]);
    }
    b
}

/// 框自身的坐标系：中心、行方向单位向量、法向、半长、半高。
#[derive(Clone, Copy)]
struct BoxFrame {
    cx: f32,
    cy: f32,
    dx: f32,
    dy: f32,
    nx: f32,
    ny: f32,
    hl: f32,
    hh: f32,
}

fn box_frame(p: &[[f32; 2]; 4]) -> BoxFrame {
    let lx = (p[0][0] + p[3][0]) * 0.5;
    let ly = (p[0][1] + p[3][1]) * 0.5;
    let rx = (p[1][0] + p[2][0]) * 0.5;
    let ry = (p[1][1] + p[2][1]) * 0.5;
    let mut dx = rx - lx;
    let mut dy = ry - ly;
    let len = (dx * dx + dy * dy).sqrt();
    if len > 1e-6 {
        dx /= len;
        dy /= len;
    } else {
        dx = 1.0;
        dy = 0.0;
    }
    let l0 = ((p[3][0] - p[0][0]) * (p[3][0] - p[0][0])
        + (p[3][1] - p[0][1]) * (p[3][1] - p[0][1]))
        .sqrt();
    let l1 = ((p[2][0] - p[1][0]) * (p[2][0] - p[1][0])
        + (p[2][1] - p[1][1]) * (p[2][1] - p[1][1]))
        .sqrt();
    BoxFrame {
        cx: (lx + rx) * 0.5,
        cy: (ly + ry) * 0.5,
        dx,
        dy,
        nx: -dy,
        ny: dx,
        hl: len * 0.5,
        hh: (l0 + l1) * 0.25,
    }
}

/// 同行判定，在**框自身坐标系**里量。
///
/// 曾经比较轴对齐外接框的 y 范围——在倾斜文本上恰好是错的：300 px 行
/// 倾斜 10° 的 AABB 有 91 px 高（凸包在两个轴上都为倾角付账），相邻两行
/// 在 y 上重叠过半而被并成一个框。倾斜语料（93% 行倾斜）实测损失
/// 4.44 个 exact 行。沿行自身法向量分隔、沿行方向量间隙，水平文本上
/// 退化为旧行为。
fn mergeable(a: &TextBox, b: &TextBox, gap_ratio: f32) -> bool {
    let (fa, fb) = (box_frame(&a.pts), box_frame(&b.pts));
    let (hmin, hmax) = (fa.hh.min(fb.hh), fa.hh.max(fb.hh));
    if hmin <= 0.0 || fa.hl <= 0.0 || fb.hl <= 0.0 {
        return false;
    }
    // 只合并横排：高过长的框不是行碎片
    if fa.hl < fa.hh || fb.hl < fb.hh {
        return false;
    }
    if hmax > 1.6 * hmin {
        return false;
    }
    let (px, py) = (fb.cx - fa.cx, fb.cy - fa.cy);
    let perp = (px * fa.nx + py * fa.ny).abs();
    if perp > 0.5 * hmin {
        return false;
    }
    let along = px * fa.dx + py * fa.dy;
    let gap = (along.abs() - (fa.hl + fb.hl)).max(0.0);
    gap <= gap_ratio * hmin
}

/// 把 `b` 并进 `a`：八个角点的凸包拟合旋转矩形。
///
/// 旧版只移动 a 的右上/右下两个角到 b 的——只对两碎片严格共基线时成立。
/// img-097.jpg 上它把两个 ~200x37 碎片并成 9x65 的细条，整行文字丢失。
/// 凸包拟合不可能那样塌：结果总包含两个输入。
fn merge_boxes(a: &mut TextBox, b: &TextBox) {
    let mut pts: Vec<Pt> = Vec::with_capacity(8);
    for k in 0..4 {
        pts.push(Pt {
            x: a.pts[k][0],
            y: a.pts[k][1],
        });
        pts.push(Pt {
            x: b.pts[k][0],
            y: b.pts[k][1],
        });
    }
    let hull = convex_hull(pts);
    let (Some(mut rect), _) = min_area_rect(&hull) else {
        return;
    };
    order_box_tl_tr_br_bl(&mut rect);
    for k in 0..4 {
        a.pts[k][0] = rect[k].x;
        a.pts[k][1] = rect[k].y;
    }
    // 合并行的 DB 分数取较弱者：DB 不确定的碎片不会因为有了邻居而变确定
    a.score = a.score.min(b.score);
}

/// 把 `boxes[i]` 的同行邻居全部吸收进它。要求 `boxes` 已按阅读序排列；
/// 返回吸收数（被吸收者删除，调用方下一轮重读 boxes[i]）。
fn absorb_line_neighbours(boxes: &mut Vec<TextBox>, i: usize, gap_ratio: f32) -> usize {
    let mut merged = 0;
    while i + 1 < boxes.len() && mergeable(&boxes[i], &boxes[i + 1], gap_ratio) {
        let b = boxes.remove(i + 1);
        merge_boxes(&mut boxes[i], &b);
        merged += 1;
    }
    merged
}

/// 阅读序排序（ppocr sorted_boxes）：先按 y 排，再在 ±10px 内按 x 冒泡。
pub fn sort_reading_order(boxes: &mut [TextBox]) {
    boxes.sort_by(|a, b| {
        a.pts[0][1]
            .partial_cmp(&b.pts[0][1])
            .unwrap()
            .then(a.pts[0][0].partial_cmp(&b.pts[0][0]).unwrap())
    });
    let n = boxes.len();
    for i in 0..n.saturating_sub(1) {
        let mut j = i + 1;
        while j > 0 {
            j -= 1;
            if (boxes[j + 1].pts[0][1] - boxes[j].pts[0][1]).abs() < 10.0
                && boxes[j + 1].pts[0][0] < boxes[j].pts[0][0]
            {
                boxes.swap(j, j + 1);
            } else {
                break;
            }
        }
    }
}

/// 对整个框列表做同行合并（merge_line_gap > 0 时）。
pub fn merge_same_line(boxes: &mut Vec<TextBox>, gap_ratio: f32) -> usize {
    let mut total = 0;
    if gap_ratio > 0.0 && boxes.len() > 1 {
        let mut i = 0;
        while i < boxes.len() {
            total += absorb_line_neighbours(boxes, i, gap_ratio);
            i += 1;
        }
    }
    total
}

/// 边距杂波判定（box_margin_clutter）：unclip 给这行加的边框里，
/// 文字不占的行的边缘能量 ÷ 文字占的行的边缘能量。~0 = 扩张只买到
/// 背景；大 = 买到杂波——难图.png（护栏上的白字时间戳）上它把正确检测
/// 变成 `2026-06-07-19129:18`。
///
/// dilate 是承重的：DB 区域是**收缩后**的文字（~0.6x 字高），直接用它
/// 会把字的上下沿当「边距」——那版对语料也报 0.4-0.6，看起来信号失效，
/// 其实是测量坏了。
pub fn box_margin_clutter(b: &TextBox, prob: &[f32], nh: i32, nw: i32, img: &Image) -> f32 {
    if nw <= 0 || nh <= 0 || img.w < 4 || img.h < 4 {
        return 0.0;
    }
    let sx = nw as f32 / img.w as f32;
    let sy = nh as f32 / img.h as f32;
    let mut xs = [0f32; 4];
    let mut ys = [0f32; 4];
    for k in 0..4 {
        xs[k] = b.pts[k][0] * sx;
        ys[k] = b.pts[k][1] * sy;
    }
    let px0 = (xs.iter().copied().fold(f32::MAX, f32::min).floor() as i32).max(0);
    let px1 = (xs.iter().copied().fold(f32::MIN, f32::max).ceil() as i32).min(nw);
    let py0 = (ys.iter().copied().fold(f32::MAX, f32::min).floor() as i32).max(0);
    let py1 = (ys.iter().copied().fold(f32::MIN, f32::max).ceil() as i32).min(nh);
    if px1 - px0 < 4 || py1 - py0 < 4 {
        return 0.0;
    }

    // 每行：概率 >0.3 的列占比 >5% 记为文字行
    let mut text = vec![0u8; (py1 - py0) as usize];
    for y in py0..py1 {
        let mut hit = 0;
        for x in px0..px1 {
            if prob[(y as usize) * (nw as usize) + x as usize] > 0.3 {
                hit += 1;
            }
        }
        text[(y - py0) as usize] = if hit as f32 / (px1 - px0) as f32 > 0.05 {
            1
        } else {
            0
        };
    }
    let dil = (3i32.max((py1 - py0) / 8)) as isize;
    let mut is_text = vec![0u8; text.len()];
    for (i, &t) in text.iter().enumerate() {
        if t != 0 {
            for k in -dil..=dil {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < is_text.len() {
                    is_text[j as usize] = 1;
                }
            }
        }
    }

    let ix0 = (1f32.max(px0 as f32 / sx)) as i32;
    let ix1 = ((img.w - 1) as f32).min(px1 as f32 / sx) as i32;
    let iy0 = (1f32.max(py0 as f32 / sy)) as i32;
    let iy1 = ((img.h - 1) as f32).min(py1 as f32 / sy) as i32;
    if ix1 - ix0 < 4 || iy1 - iy0 < 4 {
        return 0.0;
    }
    let mut in_sum = 0f64;
    let mut out_sum = 0f64;
    let mut in_n: i64 = 0;
    let mut out_n: i64 = 0;
    for y in iy0..iy1 {
        let r = img.row(y);
        let rd = img.row(y + 1);
        let mut py = (y as f32 * sy) as i32 - py0;
        py = py.clamp(0, is_text.len() as i32 - 1);
        let t = is_text[py as usize] != 0;
        for x in ix0..ix1 {
            let g = (r[((x + 1) as usize) * (img.c as usize)] as i32
                - r[(x as usize) * (img.c as usize)] as i32)
                .abs()
                + (rd[(x as usize) * (img.c as usize)] as i32
                    - r[(x as usize) * (img.c as usize)] as i32)
                    .abs();
            if t {
                in_sum += g as f64;
                in_n += 1;
            } else {
                out_sum += g as f64;
                out_n += 1;
            }
        }
    }
    if in_n < 8 || out_n < 8 {
        return 0.0;
    }
    let tin = in_sum / in_n as f64;
    if tin > 1e-6 {
        ((out_sum / out_n as f64) / tin) as f32
    } else {
        0.0
    }
}
