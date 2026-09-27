//! 透视裁剪与 rec/cls 预处理（透视裁剪与预处理）。

use super::image::Image;

/// 四点透视裁剪：解 8×8 线性方程组建 dst→src 单应，反 sampled 双线性，
/// BORDER_REPLICATE（warpPerspective 在这里的语义）。高宽比 ≥1.5 的
/// 竖条顺时针转正（ppocr 行为）。
pub fn crop_text_box(src: &Image, pts_in: &[[f32; 2]; 4]) -> Image {
    let pts = pts_in;
    let w1 = (pts[0][0] - pts[1][0]).hypot(pts[0][1] - pts[1][1]);
    let w2 = (pts[2][0] - pts[3][0]).hypot(pts[2][1] - pts[3][1]);
    let h1 = (pts[0][0] - pts[3][0]).hypot(pts[0][1] - pts[3][1]);
    let h2 = (pts[1][0] - pts[2][0]).hypot(pts[1][1] - pts[2][1]);
    let cw = 1.max(w1.max(w2) as i32);
    let ch = 1.max(h1.max(h2) as i32);

    // dst 四角 (0,0),(cw,0),(cw,ch),(0,ch) → src pts：直接解 8×8
    let mut a = [[0f64; 9]; 8];
    for i in 0..4 {
        let dx = if i == 1 || i == 2 { cw as f64 } else { 0.0 };
        let dy = if i >= 2 { ch as f64 } else { 0.0 };
        let (sx, sy) = (pts[i][0] as f64, pts[i][1] as f64);
        a[i * 2] = [dx, dy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dy * sx, sx];
        a[i * 2 + 1] = [0.0, 0.0, 0.0, dx, dy, 1.0, -dx * sy, -dy * sy, sy];
    }
    // 高斯消元（f64，与基准一致——这步的精度决定裁剪的亚像素位置）
    for col in 0..8 {
        let mut piv = col;
        for r in col + 1..8 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if piv != col {
            a.swap(col, piv);
        }
        let d = a[col][col];
        if d.abs() < 1e-12 {
            continue;
        }
        for c in col..9 {
            a[col][c] /= d;
        }
        for r in 0..8 {
            if r == col {
                continue;
            }
            let f = a[r][col];
            if f == 0.0 {
                continue;
            }
            for c in col..9 {
                a[r][c] -= f * a[col][c];
            }
        }
    }
    let mut hm = [0f64; 9];
    for (i, h) in hm.iter_mut().take(8).enumerate() {
        *h = a[i][8];
    }
    hm[8] = 1.0;

    let c = src.c as usize;
    let mut dst = Image {
        w: cw,
        h: ch,
        c: src.c,
        orig_w: src.orig_w,
        orig_h: src.orig_h,
        data: vec![0u8; (cw as usize) * (ch as usize) * c],
    };
    for y in 0..ch {
        for x in 0..cw {
            let (xf, yf) = (x as f64, y as f64);
            let den = hm[6] * xf + hm[7] * yf + hm[8];
            let sx = (hm[0] * xf + hm[1] * yf + hm[2]) / den;
            let sy = (hm[3] * xf + hm[4] * yf + hm[5]) / den;
            let mut ix = sx.floor() as i32;
            let mut iy = sy.floor() as i32;
            let fx = (sx - ix as f64) as f32;
            let fy = (sy - iy as f64) as f32;
            ix = ix.clamp(0, src.w - 1);
            iy = iy.clamp(0, src.h - 1);
            let ix1 = (ix + 1).min(src.w - 1);
            let iy1 = (iy + 1).min(src.h - 1);
            let r0 = src.row(iy);
            let r1 = src.row(iy1);
            let o = &mut dst.data[((y as usize) * (cw as usize) + x as usize) * c..][..c];
            for (ch_i, oc) in o.iter_mut().enumerate() {
                let v00 = r0[(ix as usize) * c + ch_i] as f32;
                let v01 = r0[(ix1 as usize) * c + ch_i] as f32;
                let v10 = r1[(ix as usize) * c + ch_i] as f32;
                let v11 = r1[(ix1 as usize) * c + ch_i] as f32;
                // 位级镜像：首项两乘，其余 fma
                let t = v00 * (1.0 - fx) * (1.0 - fy);
                let t = (v01 * fx).mul_add(1.0 - fy, t);
                let t = (v10 * (1.0 - fx)).mul_add(fy, t);
                let v = (v11 * fx).mul_add(fy, t);
                *oc = (v + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
    // ppocr 把竖条转正
    if ch as f32 / cw as f32 >= 1.5 {
        let mut rot = Image {
            w: ch,
            h: cw,
            c: src.c,
            orig_w: src.orig_w,
            orig_h: src.orig_h,
            data: vec![0u8; (ch as usize) * (cw as usize) * c],
        };
        for y in 0..ch {
            for x in 0..cw {
                for ch_i in 0..c {
                    let d = ((x as usize) * (rot.w as usize) + (ch as usize - 1 - y as usize)) * c
                        + ch_i;
                    let s = ((y as usize) * (cw as usize) + x as usize) * c + ch_i;
                    rot.data[d] = dst.data[s];
                }
            }
        }
        return rot;
    }
    dst
}

/// 这个高度的裁剪是「补白」上画布而不是放大（pack_crop 的 pad_only）。
pub fn crop_pads(c: &Image, img_h: i32, min_h: f64) -> bool {
    min_h > 0.0 && c.h > 0 && c.h <= img_h && c.h as f64 >= min_h * img_h as f64
}

/// 批宽计算用的长宽比。补白裁剪是 w/imgH 而不是 w/h：补白不把行变宽，
/// 画布只需容纳 `w` 列——喂 w/h 进来会预留行永远不会画的宽度，而成本
/// 由张量宽度决定，提速会悄悄蒸发。
pub fn batch_ratio(c: &Image, img_h: i32, pad_min_h: f64) -> f32 {
    if c.h <= 0 {
        return 1.0;
    }
    if crop_pads(c, img_h, pad_min_h) {
        c.w as f32 / img_h as f32
    } else {
        c.w as f32 / c.h as f32
    }
}

/// 把一个裁剪打包成 [1,3,H,W] 张量体：缩放（或 pad_only 时原分辨率居中）
/// + 归一化 (v-0.5)/0.5 + CHW。
///
/// `pad_only`：裁剪已接近 imgH 时放大只是发明像素——40 px 高的行的信息
/// 就在 40 px 里，插值到 48 只会让条更宽。原分辨率居中。加的行是白的，
/// 绝不会带进邻行的墨——这就是它比加肥检测框（unclip_ratio）安全的原因。
pub fn pack_crop(crop: &Image, img_h: i32, img_w: i32, dst: &mut [f32], pad_only: bool) {
    let c = 3usize;
    if pad_only {
        let y0 = (img_h - crop.h) / 2;
        let cols = crop.w.min(img_w);
        for y in 0..crop.h {
            let row = crop.row(y);
            let base = ((y + y0) as usize) * (img_w as usize);
            for x in 0..cols {
                for ch in 0..c {
                    let v = row[(x as usize) * 3 + ch] as f32 / 255.0;
                    dst[ch * (img_h as usize) * (img_w as usize) + base + x as usize] =
                        (v - 0.5) / 0.5;
                }
            }
        }
        return;
    }
    let ratio = crop.w as f32 / crop.h as f32;
    let mut resized_w = (img_h as f32 * ratio).ceil() as i32;
    resized_w = resized_w.min(img_w).max(1);
    let r = super::image::resize_bilinear_img(crop, resized_w, img_h);
    for y in 0..img_h {
        let row = r.row(y);
        for x in 0..resized_w {
            for ch in 0..c {
                let v = row[(x as usize) * 3 + ch] as f32 / 255.0;
                dst[(ch * (img_h as usize) + y as usize) * (img_w as usize) + x as usize] =
                    (v - 0.5) / 0.5;
            }
        }
    }
}

/// 方向分类器的输入是 cls_width×cls_height（192×48，4:1），pack_crop 会把
/// 任何东西压进去。极端行被压毁字形：小长图.png 裁出 1642×54（30:1），
/// 水平压缩 7.6 倍后分类器高置信度地答「没旋转」，倒置文本静默走错。
///
/// 方向是整行的属性，居中窗口回答同一个问题。只有比网络能表示的更宽的
/// 裁剪才开窗；以下原样返回，常规路径分毫不动。
pub fn cls_view(crop: &Image, cw: i32, ch: i32) -> Image {
    if crop.h <= 0 || crop.w <= 0 || ch <= 0 || cw <= 0 {
        return crop.clone();
    }
    let max_w = (crop.h as i64) * (cw as i64) / (ch as i64);
    if (crop.w as i64) <= max_w {
        return crop.clone();
    }
    let ww = 1i64.max(max_w) as i32;
    let x0 = (crop.w - ww) / 2;
    let c = crop.c as usize;
    let mut out = Image {
        w: ww,
        h: crop.h,
        c: crop.c,
        orig_w: crop.orig_w,
        orig_h: crop.orig_h,
        data: vec![0u8; (ww as usize) * (crop.h as usize) * c],
    };
    for y in 0..crop.h {
        let src_row = crop.row(y);
        let dst_row = &mut out.data[(y as usize) * (ww as usize) * c..][..(ww as usize) * c];
        dst_row.copy_from_slice(&src_row[(x0 as usize) * c..(x0 as usize) * c + (ww as usize) * c]);
    }
    out
}
