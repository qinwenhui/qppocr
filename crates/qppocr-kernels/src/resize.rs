//! Resize：nearest（floor）与 bilinear。

use crate::buf::F32Buf;
use crate::par;

/// 最近邻缩放。`iy = (int)((float)oy * H / oh)`，钳到 `[0, H-1]`。
#[allow(clippy::too_many_arguments)] // ONNX 属性逐项入参，压成 struct 反而藏默认值
pub fn resize_nearest(
    x: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    out: &mut F32Buf,
) {
    // SAFETY: 每输出元素写一次。
    unsafe { out.resize_uninit(n * c * oh * ow) };
    let op = par::SyncPtr::new(out.as_mut_slice().as_mut_ptr());
    par::parallel_for_units(n * c, |b, e| {
        for nc in b..e {
            let xnc = &x[nc * h * w..(nc + 1) * h * w];
            // SAFETY: 通道 nc 的输出平面与其他块不相交。
            let ync = unsafe { op.offset(nc * oh * ow).slice(oh * ow) };
            for oy in 0..oh {
                // (int)((float)oy * H / oh)：float 乘除再截断，与基准一致
                let iy = ((oy as f32 * h as f32 / oh as f32) as usize).min(h - 1);
                let xr = &xnc[iy * w..(iy + 1) * w];
                let yr = &mut ync[oy * ow..(oy + 1) * ow];
                for (oxx, yv) in yr.iter_mut().enumerate() {
                    let ix = ((oxx as f32 * w as f32 / ow as f32) as usize).min(w - 1);
                    *yv = xr[ix];
                }
            }
        }
    });
}

/// 双线性缩放。`align_corners` 语义按 ONNX：false 用半像素中心
/// `(i+0.5)*scale - 0.5`（钳非负），true 用角点对齐 `i*(H-1)/(oh-1)`。
#[allow(clippy::too_many_arguments)] // 同上
pub fn resize_bilinear(
    x: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    align_corners: bool,
    out: &mut F32Buf,
) {
    // SAFETY: 每输出元素写一次。
    unsafe { out.resize_uninit(n * c * oh * ow) };
    let op = par::SyncPtr::new(out.as_mut_slice().as_mut_ptr());

    let mut iy0 = vec![0usize; oh];
    let mut iy1 = vec![0usize; oh];
    let mut fy = vec![0f32; oh];
    let mut ix0 = vec![0i32; ow];
    let mut ix1 = vec![0i32; ow];
    let mut fx = vec![0f32; ow];
    for (i, (a, b)) in iy0.iter_mut().zip(&mut iy1).enumerate() {
        let v = if align_corners && oh > 1 {
            i as f32 * (h - 1) as f32 / (oh - 1) as f32
        } else {
            ((i as f32 + 0.5) * h as f32 / oh as f32 - 0.5).max(0.0)
        };
        let base = (v.floor() as isize).clamp(0, h as isize - 1) as usize;
        *a = base;
        *b = (base + 1).min(h - 1);
        fy[i] = v - base as f32;
    }
    for (i, (a, b)) in ix0.iter_mut().zip(&mut ix1).enumerate() {
        let v = if align_corners && ow > 1 {
            i as f32 * (w - 1) as f32 / (ow - 1) as f32
        } else {
            ((i as f32 + 0.5) * w as f32 / ow as f32 - 0.5).max(0.0)
        };
        let base = (v.floor() as isize).clamp(0, w as isize - 1) as usize;
        *a = base as i32;
        *b = ((base + 1).min(w - 1)) as i32;
        fx[i] = v - base as f32;
    }

    par::parallel_for_units(n * c, |b, e| {
        for nc in b..e {
            let xnc = &x[nc * h * w..(nc + 1) * h * w];
            // SAFETY: 通道 nc 的输出平面与其他块不相交。
            let ync = unsafe { op.offset(nc * oh * ow).slice(oh * ow) };
            for (oy, yr) in ync.chunks_mut(ow).enumerate() {
                let r0 = &xnc[iy0[oy] * w..(iy0[oy] + 1) * w];
                let r1 = &xnc[iy1[oy] * w..(iy1[oy] + 1) * w];
                let ly = fy[oy];
                #[cfg(target_arch = "x86_64")]
                if crate::use_avx2() {
                    // SAFETY: 输出行 nc 的这段与其他并行块不相交；映射表
                    // 长度为 ow，源列界内（预计算时已钳制）。
                    unsafe {
                        crate::x86::bilinear_row_vec(
                            r0.as_ptr(),
                            r1.as_ptr(),
                            ix0.as_ptr(),
                            ix1.as_ptr(),
                            fx.as_ptr(),
                            ly,
                            yr.as_mut_ptr(),
                            0,
                            ow,
                        )
                    };
                    continue;
                }
                for (oxx, yv) in yr.iter_mut().enumerate() {
                    let (x0, x1) = (ix0[oxx] as usize, ix1[oxx] as usize);
                    let lx = fx[oxx];
                    // 四角加权——项与顺序与基准一致（位级）。★ 收缩形态：
                    // GCC 把后续三项的「积 + 累加」收缩成 FMA（首项两个乘、
                    // 每项的第一乘保留、第二乘折进 fma），Rust 显式 mul_add
                    let t = r0[x0] * (1.0 - lx) * (1.0 - ly);
                    let t = (r0[x1] * lx).mul_add(1.0 - ly, t);
                    let t = (r1[x0] * (1.0 - lx)).mul_add(ly, t);
                    *yv = (r1[x1] * lx).mul_add(ly, t);
                }
            }
        }
    });
}

/// 图像级双线性缩放（u8 RGB 紧凑排列）——**逐行并行**。
///
/// 与 `qppocr_core::pipeline::image::resize_bilinear_img` 的标量实现
/// **逐位相同**（同一套半像素中心映射与「首项两乘、其余 fma」的收缩形态），
/// 只是把行循环交给池子。det 输入准备（848×816 → 832×832）实测 6.9 ms，
/// 全图缩放是串行热点。
///
/// 输出按 clamp(0,255) 取整；`dst` 长度必须为 `dw*dh*3`。
pub fn resize_bilinear_rgb_u8(
    src: &[u8],
    sw: usize,
    sh: usize,
    dst: &mut [u8],
    dw: usize,
    dh: usize,
) {
    assert!(sw > 0 && sh > 0 && dw > 0 && dh > 0, "resize: zero dim");
    assert_eq!(dst.len(), dw * dh * 3, "resize: dst size");
    assert_eq!(src.len(), sw * sh * 3, "resize: src size");
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    // 列映射（所有行共用）
    let mut x0 = vec![0usize; dw];
    let mut x1 = vec![0usize; dw];
    let mut fx = vec![0f32; dw];
    for x in 0..dw {
        let v = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
        let i0 = (v as usize).min(sw - 1);
        x0[x] = i0;
        x1[x] = (i0 + 1).min(sw - 1);
        fx[x] = v - i0 as f32;
    }
    let dp = par::SyncPtr::new(dst.as_mut_ptr());
    par::parallel_for(dh, 64, |yb, ye| {
        for y in yb..ye {
            let v = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
            let j0 = (v as usize).min(sh - 1);
            let j1 = (j0 + 1).min(sh - 1);
            let fy = v - j0 as f32;
            let r0 = &src[j0 * sw * 3..(j0 + 1) * sw * 3];
            let r1 = &src[j1 * sw * 3..(j1 + 1) * sw * 3];
            // SAFETY: 输出行 y 与其他并行块不相交。
            let out = unsafe { dp.offset(y * dw * 3).slice(dw * 3) };
            for x in 0..dw {
                let (a, b) = (x0[x] * 3, x1[x] * 3);
                let lx = fx[x];
                for ch in 0..3 {
                    let p00 = r0[a + ch] as f32;
                    let p01 = r0[b + ch] as f32;
                    let p10 = r1[a + ch] as f32;
                    let p11 = r1[b + ch] as f32;
                    // 位级镜像标量版：首项两乘，其余 fma
                    let t = p00 * (1.0 - lx) * (1.0 - fy);
                    let t = (p01 * lx).mul_add(1.0 - fy, t);
                    let t = (p10 * (1.0 - lx)).mul_add(fy, t);
                    let val = (p11 * lx).mul_add(fy, t);
                    out[x * 3 + ch] = (val + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        }
    });
}

/// DB 后处理的 2×2 膨胀（`cv2.dilate` 2×2 核、anchor=(1,1)、越界忽略）。
///
/// 逐位语义：`out[y][x] = max(窗口内所有界内样本)`。max 可交换，
/// 所以向量化不改变结果——与标量三重循环**逐位相同**。
///
/// 标量版在 832×832 掩码上实测 3.8 ms（DB 后处理里最大的一项）。
pub fn dilate2x2_max(mask: &[u8], dst: &mut [u8], h: usize, w: usize) {
    assert_eq!(mask.len(), h * w);
    assert_eq!(dst.len(), h * w);
    let dp = par::SyncPtr::new(dst.as_mut_ptr());
    par::parallel_for(h, 16, |yb, ye| {
        for y in yb..ye {
            let cur = &mask[y * w..(y + 1) * w];
            let up = if y > 0 {
                Some(&mask[(y - 1) * w..y * w])
            } else {
                None
            };
            // SAFETY: 输出行 y 与其他并行块不相交。
            let out = unsafe { dp.offset(y * w).slice(w) };
            // x = 0：左侧越界，只有 (y,x) 与 (y-1,x)
            out[0] = match up {
                Some(u) => cur[0].max(u[0]),
                None => cur[0],
            };
            #[cfg(target_arch = "x86_64")]
            if crate::use_avx2() {
                // SAFETY: x ∈ [1, w) 段内所有访问都在界内（w 由调用方保证）。
                unsafe { crate::x86::dilate2x2_row_avx2(cur, up, out.as_mut_ptr(), w) };
                continue;
            }
            for x in 1..w {
                let mut v = cur[x - 1].max(cur[x]);
                if let Some(u) = up {
                    v = v.max(u[x - 1]).max(u[x]);
                }
                out[x] = v;
            }
        }
    });
}
