//! 池化：global_avg_pool 与 pool2d（max/avg）。

use crate::buf::F32Buf;
use crate::par;

/// 全局平均池化：`[N,C,H,W] -> [N,C,1,1]`。
///
/// 累加用 **f64**：长归约在 f32 下会吃掉精度，参考实现的
/// 1e-5 相对容差过不去。返回长度 N*C 的输出。
pub fn global_avg_pool(x: &[f32], n: usize, c: usize, out: &mut F32Buf) {
    let plane = usize::checked_div(x.len(), n * c).unwrap_or(0);
    // SAFETY: 下方每元素写一次。
    unsafe { out.resize_uninit(n * c) };
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
    out: &mut F32Buf,
) -> (usize, usize) {
    let oh = (h + ph + peh).saturating_sub(kh) / sh + 1;
    let ow = (w + pw + pew).saturating_sub(kw) / sw + 1;
    // SAFETY: 每输出元素写一次。
    unsafe { out.resize_uninit(n * c * oh * ow) };
    let op = par::SyncPtr::new(out.as_mut_ptr());

    // 2x2 max pool s1 SAME_UPPER——v6 每一个 MaxPool 的形状（det 的是
    // 1x16x752x992）。max 可分：先垂直 max 进 scratch 行，再水平 max，
    // 两半都能向量化。标量版做 4 load、3 compare、4 边界测试/输出元素，
    // 实测 12.7 ms，约是这张 op 实际搬运量的 9 倍。
    // ph==pw==0、peh==pew==1：输出行 oy 读输入行 {oy, oy+1}（钳制），
    // 列同理；scratch 行尾放一个 -inf 哨兵让钳制落进普通 max。
    let pool2x2_s1 = max_pool
        && kh == 2
        && kw == 2
        && sh == 1
        && sw == 1
        && ph == 0
        && pw == 0
        && peh == 1
        && pew == 1
        && oh == h
        && ow == w;
    if pool2x2_s1 {
        par::parallel_for_units(n * c * oh, |b, e| {
            let mut vm = vec![0f32; w + 1]; // 每块一行，跨本块的行复用
            for u in b..e {
                let oy = u % oh;
                let plane = u / oh;
                // SAFETY: 输出行 (plane, oy) 与其他并行块不相交；
                // 读侧 x 平面 (plane, {oy, oy+1}) 界内。
                unsafe {
                    let r0 = x.as_ptr().add((plane * h + oy) * w);
                    let outp = op.get().add((plane * oh + oy) * ow);
                    // 最后一行没有 r1：传 null（读 r0+W 会跨进下一个通道）
                    let r1 = if oy + 1 < h {
                        r0.add(w)
                    } else {
                        std::ptr::null()
                    };
                    crate::arch_dispatch!(
                        crate::x86::pool2x2_row_vec(r0, r1, vm.as_mut_ptr(), outp, w),
                        {
                            for j in 0..w {
                                vm[j] = match r1 {
                                    p if !p.is_null() => (*r0.add(j)).max(*p.add(j)),
                                    _ => *r0.add(j),
                                };
                            }
                            vm[w] = -f32::MAX;
                            for oxx in 0..w {
                                *outp.add(oxx) = vm[oxx].max(vm[oxx + 1]);
                            }
                        }
                    );
                }
            }
        });
        return (oh, ow);
    }

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
