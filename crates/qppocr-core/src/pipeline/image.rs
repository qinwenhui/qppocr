//! 图像与基础图像算子。
//!
//! 不引 image crate：流水线核心只吃已解码的 RGB 像素（，
//! 解码与推理解耦）。解码是门面（`qppocr` crate）feature 门控的事。

/// 已解码图像：RGB、行主序。
#[derive(Clone, Debug)]
pub struct Image {
    /// 宽（像素）。
    pub w: i32,
    /// 高（像素）。
    pub h: i32,
    /// 通道数（恒 3，保留字段以镜像 ）。
    pub c: i32,
    /// 磁盘上的原始尺寸（解码期间无缩放，当前恒等于 w/h）。
    pub orig_w: i32,
    /// 磁盘上的原始高。
    pub orig_h: i32,
    /// RGB 数据，`w * h * 3` 字节。
    pub data: Vec<u8>,
}

impl Image {
    /// 全黑图。
    pub fn black(w: i32, h: i32) -> Self {
        Image {
            w,
            h,
            c: 3,
            orig_w: w,
            orig_h: h,
            data: vec![0u8; (w as usize) * (h as usize) * 3],
        }
    }
    /// 行起点指针。
    pub fn row(&self, y: i32) -> &[u8] {
        let off = (y as usize) * (self.w as usize) * (self.c as usize);
        &self.data[off..off + (self.w as usize) * (self.c as usize)]
    }
    /// 可变行。
    fn row_mut(&mut self, y: i32) -> &mut [u8] {
        let off = (y as usize) * (self.w as usize) * (self.c as usize);
        let len = (self.w as usize) * (self.c as usize);
        &mut self.data[off..off + len]
    }
}

/// 双线性缩放，半像素中心（cv2.INTER_LINEAR 语义）。u8 进 u8 出，
/// 结果 `+0.5` 后截断——与 基准的 resize_bilinear_img 逐位一致。
pub fn resize_bilinear_img(src: &Image, dw: i32, dh: i32) -> Image {
    let mut dst = Image {
        w: dw,
        h: dh,
        c: src.c,
        orig_w: src.orig_w,
        orig_h: src.orig_h,
        data: vec![0u8; (dw as usize) * (dh as usize) * (src.c as usize)],
    };
    if dw <= 0 || dh <= 0 {
        return dst;
    }
    // ★ 快路径：RGB 紧凑排列交给 kernels 的**逐行并行**版本（与下面标量
    //   分支逐位相同）。det 输入准备的全图缩放实测 6.9 ms 串行，是热点。
    if src.c == 3 {
        qppocr_kernels::resize::resize_bilinear_rgb_u8(
            &src.data,
            src.w as usize,
            src.h as usize,
            &mut dst.data,
            dw as usize,
            dh as usize,
        );
        return dst;
    }
    let sx = src.w as f32 / dw as f32;
    let sy = src.h as f32 / dh as f32;
    let c = src.c as usize;
    let mut x0 = vec![0i32; dw as usize];
    let mut x1 = vec![0i32; dw as usize];
    let mut fx = vec![0f32; dw as usize];
    // 预计算源坐标映射（zip 的嵌套解构容易错位，展开写）
    for x in 0..dw as usize {
        let v = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
        let i0 = (v as i32).min(src.w - 1);
        x0[x] = i0;
        x1[x] = (i0 + 1).min(src.w - 1);
        fx[x] = v - i0 as f32;
    }
    let sw = src.w as usize;
    for y in 0..dh {
        let v = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
        let j0 = (v as i32).min(src.h - 1);
        let j1 = (j0 + 1).min(src.h - 1);
        let fy = v - j0 as f32;
        let r0 = src.row(j0);
        let r1 = src.row(j1);
        let out_off = (y as usize) * (dw as usize) * c;
        let out = &mut dst.data[out_off..out_off + (dw as usize) * c];
        for x in 0..dw {
            let (a, b) = (x0[x as usize], x1[x as usize]);
            let lx = fx[x as usize];
            for ch in 0..c {
                let p00 = r0[(a as usize) * c + ch] as f32;
                let p01 = r0[(b as usize) * c + ch] as f32;
                let p10 = r1[(a as usize) * c + ch] as f32;
                let p11 = r1[(b as usize) * c + ch] as f32;
                // 位级镜像：GCC 收缩形态——首项两个乘，其余 fma
                let t = p00 * (1.0 - lx) * (1.0 - fy);
                let t = (p01 * lx).mul_add(1.0 - fy, t);
                let t = (p10 * (1.0 - lx)).mul_add(fy, t);
                let val = (p11 * lx).mul_add(fy, t);
                // ：(uint8_t)min(255, max(0, val + 0.5f))——四舍五入后截断
                out[x as usize * c + ch] = (val + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
        let _ = sw;
    }
    dst
}

/// 顺时针旋转 90/180/270（GUI 的旋转按钮与方向分类的 180° 翻正共用）。
pub fn rotate_image(src: &Image, degrees: i32) -> Image {
    let degrees = degrees.rem_euclid(360);
    if degrees == 0 || src.w <= 0 {
        return src.clone();
    }
    let c = src.c as usize;
    let mut dst = Image {
        w: 0,
        h: 0,
        c: src.c,
        orig_w: src.orig_w,
        orig_h: src.orig_h,
        data: Vec::new(),
    };
    if degrees == 180 {
        dst.w = src.w;
        dst.h = src.h;
        dst.data = vec![0u8; src.data.len()];
        for y in 0..src.h {
            let r = src.row(y);
            let o = &mut dst.data[((src.h - 1 - y) as usize) * (dst.w as usize) * c..];
            for x in 0..src.w {
                let s = (x as usize) * c;
                let d = ((src.w as usize) - 1 - x as usize) * c;
                o[d..d + c].copy_from_slice(&r[s..s + c]);
            }
        }
        return dst;
    }
    dst.w = src.h;
    dst.h = src.w;
    dst.data = vec![0u8; (dst.w as usize) * (dst.h as usize) * c];
    for y in 0..src.h {
        let r = src.row(y);
        for x in 0..src.w {
            let (nx, ny) = if degrees == 90 {
                (src.h - 1 - y, x) // 顺时针
            } else {
                (y, src.w - 1 - x) // 270 = 逆时针 90
            };
            let s = (x as usize) * c;
            let d = ((ny as usize) * (dst.w as usize) + nx as usize) * c;
            dst.data[d..d + c].copy_from_slice(&r[s..s + c]);
        }
    }
    dst
}

/// 自动色阶：把亮度直方图的 0.5/99.5 百分位拉到 0/255。淡淡的
/// 水印会让墨迹离背景只有 20-40 级而不是 200 级，DB 的响应随差值缩放。
/// 直方图每 4 像素采样一个（百分位不需要更多），12 MP 上 ~1 ms。
/// 一条按亮度建的 LUT 施加到三个通道——保色平衡（各通道独立拉伸会偏色）。
pub fn auto_levels(img: &mut Image) {
    if img.w < 16 || img.h < 16 {
        return;
    }
    const STEP: i32 = 4;
    let mut hist = [0u32; 256];
    let mut n: u64 = 0;
    let mut y = 0;
    while y < img.h {
        let r = img.row(y);
        let mut x = 0;
        while x < img.w {
            let p = (x as usize) * (img.c as usize);
            hist[(((r[p] as u32) * 77 + (r[p + 1] as u32) * 151 + (r[p + 2] as u32) * 28) >> 8)
                as usize] += 1;
            n += 1;
            x += STEP;
        }
        y += STEP;
    }
    if n == 0 {
        return;
    }
    let clipped = n / 200; // 0.5%
    // 基准的 for(v; (acc += hist[v]) < kClipped; ++v) lo = v + 1;
    // ——条件处在累加后判断，为真（acc 还小）才更新 lo 并继续
    let mut lo = 0usize;
    let mut acc: u64 = 0;
    for v in 0..256 {
        acc += hist[v] as u64;
        if acc < clipped {
            lo = v + 1;
        } else {
            break;
        }
    }
    let mut hi = 255usize;
    acc = 0;
    for v in 0..256 {
        acc += hist[v] as u64;
        if acc + clipped >= n {
            hi = v;
            break;
        }
    }
    if hi < lo || (hi - lo) < 24 {
        return;
    }
    let scale = 255.0f32 / (hi - lo) as f32;
    let mut lut = [0u8; 256];
    for (v, l) in lut.iter_mut().enumerate() {
        let t = (v as i32 - lo as i32) as f32 * scale;
        *l = t.clamp(0.0, 255.0) as u8;
    }
    for y in 0..img.h {
        let r = img.row_mut(y);
        for b in r.iter_mut() {
            *b = lut[*b as usize];
        }
    }
}
