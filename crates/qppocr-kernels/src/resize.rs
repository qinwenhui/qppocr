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
