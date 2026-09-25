//! 卷积：设计文档 conv 部分。
//!
//! 三条路径（按命中顺序）：
//!
//! 1. **1x1 s1 g1 → sgemm**。N>1 时把批次折进并行单元（一次 fork 跑完整个
//!    批次，而不是每批一次 fork——rec 的 6x3x48x320 条带上每批 GEMM 只有
//!    ~18 us 工作量，而 16 线程的 fork 要 97 us，fork 是它并行工作量的 5 倍）。
//! 2. **depthwise（g>1 且 Cg==1）→ 直接累加**，任意核大小（v6 det 用 7x7；
//!    早版本把这个排除在外送去 im2col，花了 ~6 倍代价）。同样整批一次 fork。
//! 3. **通用：分组 im2col + GEMM，按输出行 tile**。补丁矩阵整图一次性建
//!    要几百 MB（v6 的 2x2 conv 在 992x752 图上 ~190 MB），tile 限制工作集。
//!
//! 只有 depthwise 分支累加进 y（tap 上 `+=`）；GEMM 分支在寄存器里算满输出
//! 再 store，先清零是浪费的一趟（det 1x1 conv 47 MB、~4 ms/节点）。
//! ⚠ 当前实现 `resize` 清零——缓冲池（阶段 2/3）落地时换成不清零分配。

use crate::activation::Activation;
use crate::buf::F32Buf;
use crate::gemm::{im2col, sgemm, sgemm_serial};
use crate::par;

/// 卷积参数（对应 ONNX Conv 属性；`peh/pew` 是 end padding，
/// `auto_pad=SAME_*` 下 begin ≠ end）。
#[derive(Clone, Copy, Debug)]
pub struct ConvParams {
    /// 竖直步长。
    pub sh: usize,
    /// 水平步长。
    pub sw: usize,
    /// 竖直 begin padding。
    pub ph: usize,
    /// 水平 begin padding。
    pub pw: usize,
    /// 竖直 end padding。
    pub peh: usize,
    /// 水平 end padding。
    pub pew: usize,
    /// 竖直膨胀（仅支持 1）。
    pub dh: usize,
    /// 水平膨胀（仅支持 1）。
    pub dw: usize,
    /// 分组。
    pub group: usize,
}

/// 输出形状 `[N, M, oh, ow]`。
pub fn conv2d_out_shape(x_shape: &[i64; 4], w_shape: &[i64; 4], p: &ConvParams) -> [i64; 4] {
    let (n, c, h, w) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    let (m, cg, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
    assert!(p.dh == 1 && p.dw == 1, "dilation not supported");
    assert!(p.group as i64 * cg == c, "conv group mismatch");
    assert!(m % p.group as i64 == 0, "conv out channels % group");
    let oh = (h + p.ph as i64 + p.peh as i64 - kh) / p.sh as i64 + 1;
    let ow = (w + p.pw as i64 + p.pew as i64 - kw) / p.sw as i64 + 1;
    assert!(oh > 0 && ow > 0, "conv output empty");
    [n, m, oh, ow]
}

/// 2D 卷积。`y` 会被重设为输出形状对应的长度；返回输出形状。
///
/// GEMM 路径的输出**不清零**（`resize_uninit`）：sgemm 每元素都在
/// 寄存器里算满再 store。只有 depthwise 累加路径需要预清零。
///
/// bias 在两条 GEMM 路径的 store 阶段折入；只有 depthwise 没经过 GEMM，
/// 保留独立的加 bias pass。
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    x: &[f32],
    x_shape: &[i64; 4],
    w: &[f32],
    w_shape: &[i64; 4],
    bias: Option<&[f32]>,
    p: &ConvParams,
    act: &Activation,
    y: &mut F32Buf,
) -> [i64; 4] {
    let out_shape = conv2d_out_shape(x_shape, w_shape, p);
    let (n, c, h, wdim) = (
        x_shape[0] as usize,
        x_shape[1] as usize,
        x_shape[2] as usize,
        x_shape[3] as usize,
    );
    let (m, cg, kh, kw) = (
        w_shape[0] as usize,
        w_shape[1] as usize,
        w_shape[2] as usize,
        w_shape[3] as usize,
    );
    let (oh, ow) = (out_shape[2] as usize, out_shape[3] as usize);
    let group = p.group;
    let ohw = oh * ow;
    let out_elems = n * m * ohw;

    let depthwise = group > 1 && cg == 1;
    if depthwise {
        y.resize_zeroed(out_elems); // 唯一需要预清零的路径（tap 上 +=）
    } else {
        // SAFETY: GEMM 路径（1x1/分块 im2col）每元素都由 sgemm 的 store
        // 写满，无任何先读。
        unsafe { y.resize_uninit(out_elems) };
    }
    let yp = par::SyncPtr::new(y.as_mut_slice().as_mut_ptr());
    let wt = w;

    let mut bias_in_gemm = true;
    if group == 1 && kh == 1 && kw == 1 && p.sh == 1 && p.sw == 1 {
        // C[n] = W * X[n]，整个批次一次 fork
        let rows = n * m;
        if n > 1
            && par::threads() > 1
            && 2.0 * (rows * ohw * c) as f64 >= par::thresholds().gemm_par_min
        {
            par::parallel_for(rows, 1, |rb, re| {
                let mut r = rb;
                while r < re {
                    let bn = r / m;
                    let m0 = r - bn * m;
                    // 一个批次的行在 C 里连续，工作项在批次边界停住
                    let m1 = m.min(re - bn * m);
                    // SAFETY: 本块写的行 [(bn*m+m0), (bn*m+m1)) 与其他块不相交。
                    let ysub = unsafe { yp.offset((bn * m + m0) * ohw).slice((m1 - m0) * ohw) };
                    {
                        sgemm_serial(
                            &wt[m0 * c..],
                            &x[bn * c * ohw..],
                            ysub,
                            m1 - m0,
                            ohw,
                            c,
                            ohw,
                            bias.map(|b| &b[m0..]),
                            act,
                        );
                    }
                    r = bn * m + m1;
                }
            });
        } else {
            for bn in 0..n {
                // SAFETY: 各批次写的行段不相交（这里是串行调用，直接取子片）。
                let ysub = &mut y[bn * m * ohw..(bn + 1) * m * ohw];
                sgemm(wt, &x[bn * c * ohw..], ysub, m, ohw, c, ohw, bias, act);
            }
        }
    } else if depthwise {
        // depthwise 直积。x 向快路径只看 sw（sh 已折进 iy 查表，两者可以不同，
        // 例如 s[1,2]）。中段无边界检查可向量化；段外贡献为零。
        bias_in_gemm = false;
        let chan_body = |ub: usize, ue: usize| {
            for u in ub..ue {
                let (bn, ch) = (u / c, u % c);
                let xn = &x[bn * c * h * wdim..];
                let xch = &xn[ch * h * wdim..(ch + 1) * h * wdim];
                let wc = &wt[ch * kh * kw..(ch + 1) * kh * kw];
                // SAFETY: (n, c) 通道平面与其他块不相交。
                let yc = unsafe { yp.offset((bn * m + ch) * ohw).slice(ohw) };
                for oy in 0..oh {
                    let yr = &mut yc[oy * ow..(oy + 1) * ow];
                    for ky in 0..kh {
                        let iy = oy as isize * p.sh as isize - p.ph as isize + ky as isize;
                        if iy < 0 || iy as usize >= h {
                            continue;
                        }
                        let xr = &xch[iy as usize * wdim..(iy as usize + 1) * wdim];
                        let wrow = &wc[ky * kw..];
                        if p.sw == 1 {
                            for kx in 0..kw {
                                let wv = wrow[kx];
                                // 段外整条贡献为零；中段连续可向量化
                                let ox0 = p.pw.saturating_sub(kx);
                                let ox1 = ow.min(wdim + p.pw - kx);
                                if ox1 <= ox0 {
                                    continue;
                                }
                                let yseg = &mut yr[ox0..ox1];
                                let xseg = &xr[ox0 + kx - p.pw..ox1 + kx - p.pw];
                                // SAFETY: 段 [ox0, ox1) 属于本通道平面，
                                // 与其他并行块不相交；xseg 同长。
                                crate::arch_dispatch!(
                                    // SAFETY: 段 [ox0, ox1) 属于本通道平面，
                                    // 与其他并行块不相交；xseg 同长。
                                    unsafe {
                                        crate::x86::depthwise_fma_vec(
                                            yseg.as_mut_ptr(),
                                            xseg.as_ptr(),
                                            ox1 - ox0,
                                            wv,
                                        )
                                    },
                                    {
                                        for (d, sv) in yseg.iter_mut().zip(xseg) {
                                            // y += w * x（fma，与 AVX2 一致）
                                            *d = wv.mul_add(*sv, *d);
                                        }
                                    }
                                );
                            }
                        } else {
                            for kx in 0..kw {
                                let wv = wrow[kx];
                                for (ox, d) in yr.iter_mut().enumerate() {
                                    let ix =
                                        ox as isize * p.sw as isize - p.pw as isize + kx as isize;
                                    if ix >= 0 && (ix as usize) < wdim {
                                        *d = wv.mul_add(xr[ix as usize], *d);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        let units = n * c;
        if par::worth_forking((units * kh * kw * oh * ow) as f64) {
            par::parallel_for(units, 1, chan_body);
        } else {
            chan_body(0, units);
        }
    } else {
        // 通用：分组 im2col + GEMM，输出行 tile
        bias_in_gemm = true;
        let col_budget = 1usize << 19; // tuning.hpp: col_budget
        let kk = cg * kh * kw;
        // tile 高度由 cache 预算定，再缩到 (批×组×tile) 工作项能填满线程池。
        // 预算按并发块数分摊——每个块带自己的 scratch，不除会让总足迹
        // 乘上线程数（det 曾因此回退 2.5%）。
        let nchunks = 1usize.max(par::threads().min(n * group * 1usize.max(oh)));
        let mut tile = 1usize.max(col_budget / (kk * ow).max(1) / nchunks);
        tile = tile.min(oh);
        {
            let ng = n * group;
            let want = 1usize.max(par::threads().div_ceil(ng));
            tile = tile.min(1usize.max(oh.div_ceil(want)));
        }
        let ntiles = oh.div_ceil(tile);
        let units = n * group * ntiles;

        let mg = m / group;
        let tile_body = |ub: usize, ue: usize| {
            // 每块自己的 scratch，跨本块的 tile 复用。
            // （试过共享池化的 Buf，实测更慢——0.878 over 13 对 @16 线程，
            // 单线程也 0.966，不只是池锁的竞争。保持 std::vector 语义。）
            let mut cols: Vec<f32> = Vec::new();
            for u in ub..ue {
                let t = u % ntiles;
                let g = (u / ntiles) % group;
                let bn = u / (ntiles * group);
                let oy0 = t * tile;
                let rows = tile.min(oh - oy0);
                let xg = &x[((bn * c + g * cg) * h * wdim)..];
                let wg = &wt[g * mg * kk..];
                // SAFETY: (n, g, [oy0, oy0+rows)) 的行块与其他块不相交。
                // 片长必须覆盖 sgemm 的触达范围：mg 行、行距 ohw、每行 n 个
                //（rows*ow），即 (mg-1)*ohw + rows*ow——只给 rows*ow 会截尾。
                let yg = unsafe {
                    yp.offset((bn * m + g * mg) * ohw + oy0 * ow)
                        .slice((mg - 1) * ohw + rows * ow)
                };
                cols.resize(kk * rows * ow, 0.0);
                im2col(
                    xg, cg, h, wdim, kh, kw, p.sh, p.sw, p.ph, p.pw, ow, oy0, rows, &mut cols,
                );
                sgemm_serial(
                    wg,
                    &cols,
                    yg,
                    mg,
                    rows * ow,
                    kk,
                    ohw,
                    bias.map(|b| &b[g * mg..]),
                    act,
                );
            }
        };
        // GEMM 是 N*M*oh*ow*K MAC；im2col 是纯搬运叠加，略低估工作量，
        // 偏向不 fork。
        if par::worth_forking((n * m * ohw * kk) as f64) {
            par::parallel_for(units, 1, tile_body);
        } else {
            tile_body(0, units);
        }
    }

    if act.is_on() && depthwise {
        // 图优化只融合 group==1 的卷积，这里理论上不触发——但静默丢掉
        // 融合激活比接住它糟糕得多，所以宁可处理。
        crate::activation::apply_act(y, act);
    }

    if let Some(b) = bias {
        if !bias_in_gemm {
            // depthwise 等不经过 sgemm 的路径补加 bias。
            // 单元是整 (channel, plane)——单元数 N*M 远低于元素 grain。
            let plane = ohw;
            let bias_body = |b0: usize, e0: usize| {
                for cm in b0..e0 {
                    let bv = b[cm % m];
                    // SAFETY: 平面 [cm*plane, ...) 与其他块不相交。
                    let seg = unsafe { yp.offset(cm * plane).slice(plane) };
                    for v in seg.iter_mut() {
                        *v += bv;
                    }
                }
            };
            if par::worth_forking((n * m * plane) as f64) {
                par::parallel_for_units(n * m, bias_body);
            } else {
                bias_body(0, n * m);
            }
        }
    }
    out_shape
}

/// ConvTranspose，仅支持 2x2 s2 p0（OCR det 用的唯一形状）。
///
/// ONNX 权重布局是 `[C_in, C_out/group, kh, kw]`（**不是** Conv 的
/// `[out, in, ...]`）。k=2,s=2,p=0 时每个输入像素 (iy,ix) 散射到
/// out(2iy+ky, 2ix+kx)。
///
/// 每个输出元素恰好被赋值一次（行 2iy+ky 覆盖所有行、列 2j+kx 所有列），
/// 不需要先清零。同一输出行的两个 kx 奇偶来自**相同**的输入像素：
/// 两个奇偶都在寄存器里累加，作为相邻对一次写出。朴素写法（4 个奇偶各走
/// 一遍、`out[..] += w·x` 对 c 累加）每个输出元素重读重写一次、隔一个
/// float 写一个，每条 64B cache line 每趟碰两次——det 里两个
/// ConvTranspose 节点 149 ms、扩展比 ~1.0 就是它的全部代价。
#[allow(clippy::too_many_arguments)]
pub fn convtranspose2d(
    x: &[f32],
    x_shape: &[i64; 4],
    w: &[f32],
    w_shape: &[i64; 4],
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    y: &mut F32Buf,
) -> [i64; 4] {
    let (n, c, h, wdim) = (
        x_shape[0] as usize,
        x_shape[1] as usize,
        x_shape[2] as usize,
        x_shape[3] as usize,
    );
    let (cin, m, kh, kw) = (
        w_shape[0] as usize,
        w_shape[1] as usize,
        w_shape[2] as usize,
        w_shape[3] as usize,
    );
    assert!(cin == c, "convT weight C_in mismatch");
    assert!(
        cin == c && kh == 2 && kw == 2 && sh == 2 && sw == 2 && ph == 0 && pw == 0,
        "convtranspose: only 2x2 s2 p0 supported"
    );
    let (oh, ow) = (h * 2, wdim * 2);
    // 每个输出元素恰好被赋值一次，无需清零
    // SAFETY: 下方循环对 (n, co, oy, ox) 全空间赋值。
    unsafe { y.resize_uninit(n * m * oh * ow) };
    let yp = par::SyncPtr::new(y.as_mut_slice().as_mut_ptr());

    for bn in 0..n {
        let xn = &x[bn * c * h * wdim..(bn + 1) * c * h * wdim];
        // 单元是输出通道；这些节点的通道数远低于默认 grain。
        let body = |mb: usize, me: usize| {
            let mut w0 = vec![0.0f32; c];
            let mut w1 = vec![0.0f32; c];
            for co in mb..me {
                // SAFETY: 输出通道 co 的平面与其他块不相交。
                let yo = unsafe { yp.offset((bn * m + co) * oh * ow).slice(oh * ow) };
                for ky in 0..2 {
                    // w[c][co][ky][0..1] 沿 c 步长 2*M；先gather成连续两列
                    for ch in 0..c {
                        let wc = &w[((ch * m + co) * 4 + ky * 2)..];
                        w0[ch] = wc[0];
                        w1[ch] = wc[1];
                    }
                    for iy in 0..h {
                        let orow = &mut yo[(iy * 2 + ky) * ow..(iy * 2 + ky + 1) * ow];
                        let rowbase = iy * wdim; // 通道内行起点（xn 是 [c][h][w]）
                        let mut j = 0usize;
                        #[cfg(target_arch = "x86_64")]
                        if crate::use_avx2() {
                            while j + 8 <= wdim {
                                // SAFETY: 输出段 [j*2, j*2+16) 属于本输出通道，
                                // 与其他并行块不相交；x 读 [rowbase+j, +8)。
                                unsafe {
                                    crate::x86::convt_row_vec(
                                        xn.as_ptr().add(rowbase),
                                        h * wdim,
                                        w0.as_ptr(),
                                        w1.as_ptr(),
                                        c,
                                        j,
                                        orow.as_mut_ptr(),
                                    )
                                };
                                j += 8;
                            }
                        }
                        while j < wdim {
                            let o2 = &mut orow[j * 2..j * 2 + 2];
                            // a -> 偶数输出列，b -> 奇数；c 升序累加（位级与
                            // AVX2 的 interleave 写法一致）
                            let mut a = 0.0f32;
                            let mut b = 0.0f32;
                            for ch in 0..c {
                                let xv = xn[ch * h * wdim + rowbase + j];
                                a = xv.mul_add(w0[ch], a);
                                b = xv.mul_add(w1[ch], b);
                            }
                            o2[0] = a;
                            o2[1] = b;
                            j += 1;
                        }
                    }
                }
            }
        };
        par::parallel_for(m, 1, body);
    }
    [n as i64, m as i64, oh as i64, ow as i64]
}
