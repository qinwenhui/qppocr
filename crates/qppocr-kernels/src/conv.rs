//! 卷积：im2col + GEMM、分组/深度卷积、转置卷积。
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
//! 三条路径都在寄存器里算满输出再 store，没有一条需要预清零 y
//! （det 1x1 conv 47 MB、~4 ms/节点，白扫一趟）。

use crate::activation::Activation;
use crate::buf::F32Buf;
use crate::gemm::sgemm_bptrs_serial;
use crate::gemm::{im2col, sgemm_serial_res};
use crate::par;

/// 深度卷积一个 (n, channel) 平面的**标量参考实现**（非 x86 的兜底，
/// 也是 AVX2 路径位级对拍的基准）。输入是**已补零的平面**。
///
/// 补零平面：`(ph + h + peh) × pwidth`，行距 `pwidth`，原始像素放在
/// `[ph + y][pw + x]`，其余全是 0。这样卷积本身可以完全不带边界判断——
/// 越界的 tap 读到的就是 0，贡献 `w * 0.0`。旧写法是逐 tap 判断越界再跳过，
/// 那几列（占列数 4%）实测吃掉 30-60% 的时间。
///
/// 累加顺序：ky 外层、kx 内层、从 0 起，最后加 bias。
#[allow(clippy::too_many_arguments)]
pub fn depthwise_plane_scalar(
    yc: &mut [f32],
    xp: &[f32],
    wc: &[f32],
    oh: usize,
    ow: usize,
    pwidth: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    bias: f32,
) {
    for oy in 0..oh {
        let base = oy * sh * pwidth;
        for ox in 0..ow {
            let mut acc = 0.0f32;
            for ky in 0..kh {
                let row = &xp[base + ky * pwidth..];
                for kx in 0..kw {
                    acc = wc[ky * kw + kx].mul_add(row[ox * sw + kx], acc);
                }
            }
            yc[oy * ow + ox] = acc + bias;
        }
    }
}

/// 把一个 (h, wdim) 通道平面补零成 `(ph+h+peh) × pwidth`（行距 pwidth）。
///
/// 边距只在**首次分配**时清一次零：内区每次都被原样覆盖，边距永远是 0，
/// 所以复用时不需要重新 memset。返回的行距由调用方按需要算好传进来。
#[allow(clippy::too_many_arguments)]
fn pad_channel(
    pad: &mut F32Buf,
    xch: &[f32],
    h: usize,
    wdim: usize,
    ph: usize,
    pw: usize,
    pheight: usize,
    pwidth: usize,
    pew: usize,
) {
    let need = pheight * pwidth;
    if pad.len() != need {
        // ★ 只清**边距**，不清整片。内区每次都被下面的 copy 覆盖，清它
        //   是白烧内存带宽：s2 的深度卷积一个平面就是 403×513 = 827 KB，
        //   而 `units = n·c` 会让每个并行块各分配一次——32 个块就是 26 MB
        //   的 memset，实测占那个算子 16 线程耗时（2.09 ms）的**全部**。
        //   边距只有 (ph+peh)·pwidth + h·(pw + pwidth-pw-wdim) ≈ 55 KB。
        // SAFETY: 下面把每个元素都写满——边距清零、内区拷入。
        unsafe { pad.resize_uninit(need) };
        let p = pad.as_mut_slice();
        let (top, bot) = (ph * pwidth, (ph + h) * pwidth);
        p[..top].fill(0.0);
        p[bot..].fill(0.0);
        for y in 0..h {
            let row = (ph + y) * pwidth;
            p[row..row + pw].fill(0.0);
            let right = row + pw + wdim;
            p[right..row + pwidth].fill(0.0);
        }
        let _ = pew;
    }
    let p = pad.as_mut_slice();
    for y in 0..h {
        let dst = (ph + y) * pwidth + pw;
        p[dst..dst + wdim].copy_from_slice(&xch[y * wdim..(y + 1) * wdim]);
    }
    crate::gemm::DW_PAD_BYTES.fetch_add(
        (h * wdim * 4 * 2) as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
}

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
    conv2d_res(x, x_shape, w, w_shape, bias, p, act, y, None)
}

/// [`conv2d`] 的残差融合版：`residual` 在激活之后逐元素加上（与被融合的
/// 独立 Add 同值同序）。1×1 GEMM 路径折进 `finish_res`（省一趟冷读写），
/// 其余路径在尾部一趟完成（至少省掉独立节点与中间张量分配）。
#[allow(clippy::too_many_arguments)]
pub fn conv2d_res(
    x: &[f32],
    x_shape: &[i64; 4],
    w: &[f32],
    w_shape: &[i64; 4],
    bias: Option<&[f32]>,
    p: &ConvParams,
    act: &Activation,
    y: &mut F32Buf,
    residual: Option<&[f32]>,
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
    // SAFETY: 两条路径都**写满**每一个输出元素——GEMM 路径由 sgemm 的 store
    // 写满，depthwise 路径由 `depthwise_plane_*` 逐 ox 写满（内点与两端
    // 三段合起来恰好覆盖 [0, ow)）。没有先读。
    unsafe { y.resize_uninit(out_elems) };
    let yp = par::SyncPtr::new(y.as_mut_slice().as_mut_ptr());
    let wt = w;

    // 非 1x1 路径的 residual 在出口统一加（与独立 Add 同值同序：
    // 各路径写完 y（含激活）之后逐元素 += residual）。
    let tail_residual = if group == 1 && kh == 1 && kw == 1 && p.sh == 1 && p.sw == 1 {
        None // 1x1 已折进 sgemm_res 的 finish_res
    } else {
        residual
    };
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
                        let rsub = residual.map(|r| &r[(bn * m + m0) * ohw..(bn * m + m1) * ohw]);
                        sgemm_serial_res(
                            &wt[m0 * c..],
                            &x[bn * c * ohw..],
                            ysub,
                            m1 - m0,
                            ohw,
                            c,
                            ohw,
                            bias.map(|b| &b[m0..]),
                            act,
                            rsub,
                        );
                    }
                    r = bn * m + m1;
                }
            });
        } else {
            for bn in 0..n {
                // SAFETY: 各批次写的行段不相交（这里是串行调用，直接取子片）。
                let ysub = &mut y[bn * m * ohw..(bn + 1) * m * ohw];
                let rsub = residual.map(|r| &r[bn * m * ohw..(bn + 1) * m * ohw]);
                crate::gemm::sgemm_res(
                    wt,
                    &x[bn * c * ohw..],
                    ysub,
                    m,
                    ohw,
                    c,
                    ohw,
                    bias,
                    act,
                    rsub,
                );
            }
        }
    } else if depthwise {
        // depthwise 直积：先把这个通道平面补零，再交给**完全无边界判断**
        // 的内核。累加器在寄存器里，每个输出只写一次。
        //
        // 数值与旧的逐 tap 写法等价：顺序仍是 ky 外层、kx 内层、从 0 起，
        // bias 最后加。越界的 tap 从「跳过」变成「加 w*0.0」——有限权重下
        // `x + 0.0 == x`（唯一例外 `-0.0 + 0.0 == +0.0`，靠逐字符对拍兜底）。
        let pheight = p.ph + h + p.peh;
        // 行距要够向量读：sw=2 时一发读 16 宽，最右一发在 ox=ow-8。
        let pwidth = (p.pw + wdim + p.pew).max(ow * p.sw + kw + 8);
        let chan_body = |ub: usize, ue: usize| {
            let mut pad = F32Buf::new();
            for u in ub..ue {
                let (bn, ch) = (u / c, u % c);
                let xn = &x[bn * c * h * wdim..];
                let xch = &xn[ch * h * wdim..(ch + 1) * h * wdim];
                let wc = &wt[ch * kh * kw..(ch + 1) * kh * kw];
                let bv = bias.map_or(0.0, |b| b[ch]);
                pad_channel(&mut pad, xch, h, wdim, p.ph, p.pw, pheight, pwidth, p.pew);
                // SAFETY: (n, c) 通道平面与其他块不相交；pad 长度 = pheight*pwidth。
                let yc = unsafe { yp.offset((bn * m + ch) * ohw).slice(ohw) };
                // SAFETY: yc 是本块独占的通道平面；pad/wc 只读且不与之重叠。
                unsafe {
                    crate::arch::depthwise_plane_padded(
                        yc.as_mut_ptr(),
                        pad.as_ptr(),
                        wc.as_ptr(),
                        oh,
                        ow,
                        pwidth,
                        kh,
                        kw,
                        p.sh,
                        p.sw,
                        bv,
                    )
                };
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
        let col_budget = 1usize << 19; // im2col 单块 scratch 预算（f32 个数）
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
                // ★ implicit GEMM：sw ∈ {1,2} 时**不再 materialize patch 矩阵**。
                //   patch 是输入的 kh·kw 倍（k3x3 就是 9 倍），实测 det 单图
                //   光 im2col 就搬 359 MB、占全部搬运的 30%，而整机已经跑在
                //   DRAM 带宽上限上。改成「把这一块要用到的输入行补零成一张
                //   slab，B 的每个 k 行给一个指向 slab 的指针」——一个字节的
                //   patch 都不用写。
                if p.sw == 1 || p.sw == 2 {
                    // slab：cg × slab_rows × pwidth，行距 pwidth，原始像素落在
                    // [pw, pw+wdim)，其余（上下补边、左右补边、右端 slack）是 0。
                    let slab_rows = (rows - 1) * p.sh + kh;
                    let pwidth = (ow - 1) * p.sw + kw + 32;
                    let mut slab = vec![0f32; cg * slab_rows * pwidth];
                    for ch in 0..cg {
                        let xch = &xg[ch * h * wdim..];
                        for y in 0..slab_rows {
                            let iy = (oy0 * p.sh) as isize - p.ph as isize + y as isize;
                            if iy < 0 || iy as usize >= h {
                                continue;
                            }
                            let src = &xch[iy as usize * wdim..(iy as usize + 1) * wdim];
                            let dst = (ch * slab_rows + y) * pwidth + p.pw;
                            slab[dst..dst + wdim].copy_from_slice(src);
                        }
                    }
                    // 每个输出行一次 GEMM，n = ow。B 的第 k 行 = slab 里
                    // (ch, r*sh+ky, ·+kx) 那一行，k 序与 im2col 完全一致
                    // （ch → ky → kx），所以逐位相同。
                    let mut ptrs: Vec<*const f32> = Vec::with_capacity(kk);
                    for r in 0..rows {
                        ptrs.clear();
                        for ch in 0..cg {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    // SAFETY: slab 在本次循环内不变；偏移落在
                                    // cg × slab_rows × pwidth 之内（r < rows）。
                                    ptrs.push(unsafe {
                                        slab.as_ptr()
                                            .add((ch * slab_rows + r * p.sh + ky) * pwidth + kx)
                                    });
                                }
                            }
                        }
                        // SAFETY: 行 (bn, g·mg.., oy0+r) 与其他块不相交；
                        // 片长覆盖 mg 行、行距 ohw、末行 ow 个。
                        let ysub = unsafe {
                            yp.offset((bn * m + g * mg) * ohw + (oy0 + r) * ow)
                                .slice((mg - 1) * ohw + ow)
                        };
                        sgemm_bptrs_serial(
                            wg,
                            &ptrs,
                            ysub,
                            mg,
                            ow,
                            kk,
                            ohw,
                            bias.map(|b| &b[g * mg..]),
                            act,
                            p.sw,
                        );
                    }
                    continue;
                }
                cols.resize(kk * rows * ow, 0.0);
                im2col(
                    xg, cg, h, wdim, kh, kw, p.sh, p.sw, p.ph, p.pw, ow, oy0, rows, &mut cols,
                );
                sgemm_serial_res(
                    wg,
                    &cols,
                    yg,
                    mg,
                    rows * ow,
                    kk,
                    ohw,
                    bias.map(|b| &b[g * mg..]),
                    act,
                    None, // 残差由函数尾部的统一趟处理（非 1x1 路径）
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
    // 非 1x1 路径的残差：bias 之后逐元素加（与独立 Add 同值同序）。
    if let Some(r) = tail_residual {
        let total = out_elems;
        let yp2 = par::SyncPtr::new(y.as_mut_slice().as_mut_ptr());
        par::parallel_for_elems(total, total * 4, |b0, e0| {
            // SAFETY: 元素区间 [b0, e0) 与其他并行块不相交；r 同长。
            let seg = unsafe { yp2.offset(b0).slice(e0 - b0) };
            let rs = &r[b0..e0];
            for (d, sv) in seg.iter_mut().zip(rs) {
                *d += *sv;
            }
        });
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
                        // SAFETY: 输出行 (iy*2+ky) 的这一段属于本输出通道，与
                        // 其他并行块不相交；x 读各通道同一行 [0, wdim)。
                        unsafe {
                            crate::arch::convt_rows(
                                xn.as_ptr().add(rowbase),
                                h * wdim,
                                w0.as_ptr(),
                                w1.as_ptr(),
                                c,
                                0,
                                wdim,
                                orow.as_mut_ptr(),
                            )
                        };
                    }
                }
            }
        };
        par::parallel_for(m, 1, body);
    }
    [n as i64, m as i64, oh as i64, ow as i64]
}

/// ConvTranspose 行内积的标量判据：输入列 `[j0, j1)` 每列散射出相邻两个
/// 输出（kx 奇偶），c 通道升序 FMA 链。向量后端按本架构宽度分批后，
/// 不足一批的尾巴也走这里——每元素的运算串与向量路径一致。
///
/// # Safety
///
/// `xr` 起有 `c*ch_plane` 个可读元素；`w0`/`w1` 各 `c` 个；`orow` 起有
/// `2*(j1-j0)` 个可写元素。
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_op_in_unsafe_fn)] // 裸指针循环：前置条件见 # Safety 段
pub(crate) unsafe fn convt_row_scalar(
    xr: *const f32,
    ch_plane: usize,
    w0: *const f32,
    w1: *const f32,
    c: usize,
    j0: usize,
    j1: usize,
    orow: *mut f32,
) {
    for j in j0..j1 {
        // a -> 偶数输出列，b -> 奇数；c 升序累加
        let mut a = 0.0f32;
        let mut b = 0.0f32;
        for ch in 0..c {
            let xv = *xr.add(ch * ch_plane + j);
            a = xv.mul_add(*w0.add(ch), a);
            b = xv.mul_add(*w1.add(ch), b);
        }
        orow.add(j * 2).write(a);
        orow.add(j * 2 + 1).write(b);
    }
}
