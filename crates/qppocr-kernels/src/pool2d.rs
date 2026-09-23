//! 池化：global_avg_pool 与 pool2d（max/avg）。

use crate::par;

/// 全局平均池化：`[N,C,H,W] -> [N,C,1,1]`。
///
/// 累加用 **f64**（C++ 同款）：长归约在 f32 下会吃掉精度，test_ops 的
/// 1e-5 相对容差过不去。返回长度 N*C 的输出。
pub fn global_avg_pool(x: &[f32], n: usize, c: usize, out: &mut Vec<f32>) {
    let plane = usize::checked_div(x.len(), n * c).unwrap_or(0);
    out.clear();
    out.resize(n * c, 0.0);
    let op = par::SyncPtr::new(out.as_mut_ptr());
    par::parallel_for_units(n * c, |b, e| {
        for i in b..e {
            let p = &x[i * plane..(i + 1) * plane];
            let mut s = 0f64;
            for &v in p {
                s += v as f64;
            }
            // SAFETY: 元素 i 与其他块不相交。
            unsafe { *op.get().add(i) = (s / plane as f64) as f32 };
        }
    });
}

/// 2D 池化（max 或 avg）。输出形状 `[N, C, oh, ow]`，
/// `oh = (H + ph + peh - kh)/sh + 1`（池化输出不含 kernel 越界钳制）。
#[allow(clippy::too_many_arguments)]
pub fn pool2d(
    x: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    peh: usize,
    pew: usize,
    max_pool: bool,
    out: &mut Vec<f32>,
) -> (usize, usize) {
    let oh = (h + ph + peh).saturating_sub(kh) / sh + 1;
    let ow = (w + pw + pew).saturating_sub(kw) / sw + 1;
    out.clear();
    out.resize(n * c * oh * ow, 0.0);
    let op = par::SyncPtr::new(out.as_mut_ptr());

    // 无 padding 且 kernel == stride：iy = oy*sh+ky、ix = ox*sw+kx 必然在界内，
    // 每输出元素 6 次边界测试消失。
    let in_range = ph == 0
        && pw == 0
        && peh == 0
        && pew == 0
        && kh == sh
        && kw == sw
        && oh * sh == h
        && ow * sw == w;
    par::parallel_for_units(n * c, |b, e| {
        for nc in b..e {
            let xnc = &x[nc * h * w..(nc + 1) * h * w];
            // SAFETY: 通道 nc 的输出平面与其他块不相交。
            let ync = unsafe { op.offset(nc * oh * ow).slice(oh * ow) };
            if in_range {
                // 保留除法：乘 1/(kh*kw) 在例如 9 时不是同一个数，
                // 这条路径必须与通用路径逐位一致
                let karea = (kh * kw) as f32;
                for oy in 0..oh {
                    for oxx in 0..ow {
                        let mut acc = if max_pool { -3.4e38f32 } else { 0.0f32 };
                        let col = &xnc[oxx * sw..];
                        for ky in 0..kh {
                            let r = &col[(oy * sh + ky) * w..];
                            if max_pool {
                                for kx in 0..kw {
                                    if r[kx] > acc {
                                        acc = r[kx];
                                    }
                                }
                            } else {
                                for kx in 0..kw {
                                    acc += r[kx];
                                }
                            }
                        }
                        ync[oy * ow + oxx] = if max_pool { acc } else { acc / karea };
                    }
                }
                continue;
            }
            for oy in 0..oh {
                for oxx in 0..ow {
                    let mut acc = if max_pool { -3.4e38f32 } else { 0.0f32 };
                    for ky in 0..kh {
                        let iy = oy as isize * sh as isize - ph as isize + ky as isize;
                        if iy < 0 || iy as usize >= h {
                            continue;
                        }
                        for kx in 0..kw {
                            let ix = oxx as isize * sw as isize - pw as isize + kx as isize;
                            if ix < 0 || ix as usize >= w {
                                continue;
                            }
                            let v = xnc[iy as usize * w + ix as usize];
                            acc = if max_pool {
                                if v > acc { v } else { acc }
                            } else {
                                acc + v
                            };
                        }
                    }
                    ync[oy * ow + oxx] = if max_pool {
                        acc
                    } else {
                        acc / (kh * kw) as f32
                    };
                }
            }
        }
    });
    (oh, ow)
}
