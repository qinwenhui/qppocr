//! 卷积：im2col + GEMM、分组/深度卷积、转置卷积。
//!
//! 三条路径（按命中顺序）：
//!
//! 1. **1x1 s1 g1 → sgemm**。N>1 时把批次折进并行单元（一次 fork 跑完整个
//!    批次，而不是每批一次 fork——rec 的 6x3x48x320 条带上每批 GEMM 只有
//!    ~18 us 工作量，而 16 线程的 fork 要 97 us，fork 是它并行工作量的 5 倍）。
//! 2. **depthwise（g>1 且 Cg==1）→ 直接累加**，任意核大小（v6 det 用 7x7；
//!    送 im2col 要多付 kh·kw 倍的搬运）。同样整批一次 fork。
//! 3. **通用：分组 im2col + GEMM，按输出行 tile**。补丁矩阵整图一次性建
//!    要几百 MB（v6 的 2x2 conv 在 992x752 图上 ~190 MB），tile 限制工作集。
//!
//! 三条路径都在寄存器里算满输出再 store，没有一条需要预清零 y（大输出
//! 的逐节点预清零是纯浪费）。

use crate::activation::Activation;
use crate::buf::F32Buf;
use crate::gemm::sgemm;
use crate::gemm::sgemm_bptrs_serial;
use crate::gemm::{im2col, sgemm_serial_res};
use crate::par;

thread_local! {
    /// tile 级 scratch 的每线程缓存（slab / im2col cols / bptrs）：
    /// 同 worker 线程跨 tile、跨图、跨引擎复用——只增不缩，热页零 fault。
    /// 之前「共享池化 Buf 更慢」的教训是**全局锁**争用；thread-local 没有
    /// 锁。缓存容量 = 该线程历史最大请求，上界可控（每线程至多一份）。
    static TILE_SLAB: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    static TILE_COLS: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    static TILE_PTRS: std::cell::RefCell<Vec<*const f32>> = const { std::cell::RefCell::new(Vec::new()) };
    /// NHWC depthwise 的补零输入缓冲（[n, h+ph+peh, w+pw+pew, c]）。
    static NHWC_PAD: std::cell::RefCell<F32Buf> = std::cell::RefCell::new(F32Buf::new());
}

/// 深度卷积一个 (n, channel) 平面的**标量参考实现**（无向量后端时的
/// 执行路径，也是位级对拍的基准）。输入是**已补零的平面**。
///
/// 补零平面：`(ph + h + peh) × pwidth`，行距 `pwidth`，原始像素放在
/// `[ph + y][pw + x]`，其余全是 0。这样卷积本身可以完全不带边界判断——
/// 越界的 tap 读到的就是 0，贡献 `w * 0.0`。逐 tap 判断越界再跳过的
/// 写法，边界几列的分支成本很高。
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
    let out_elems = out_shape.iter().product::<i64>().max(0) as usize;
    // SAFETY: `conv2d_res_into` 的所有路径都写满每一个输出元素（见其
    // SAFETY 注释），没有先读。
    unsafe { y.resize_uninit(out_elems) };
    conv2d_res_into(
        x,
        x_shape,
        w,
        w_shape,
        bias,
        p,
        act,
        out_shape,
        y.as_mut_slice(),
        residual,
    )
}

/// [`conv2d_res`] 的「调用方给缓冲」版：内核直接写进 `y`（长度必须恰好是
/// 输出元素数），不做任何分配/清零。`out_shape` 由调用方用
/// [`conv2d_out_shape`] 预先算好传入——别名执行器（Concat 零拷贝）用它把
/// 输出直接落到共享缓冲的通道槽位上。
///
/// SAFETY 契约（沿用 [`conv2d_res`]）：所有路径都**写满**每一个输出元素
/// ——GEMM 路径由 sgemm 的 store 写满，depthwise 路径由 `depthwise_plane_*`
/// 逐 ox 写满（内点与两端三段合起来恰好覆盖 `[0, ow)`）。没有先读。
#[allow(clippy::too_many_arguments)]
pub fn conv2d_res_into(
    x: &[f32],
    x_shape: &[i64; 4],
    w: &[f32],
    w_shape: &[i64; 4],
    bias: Option<&[f32]>,
    p: &ConvParams,
    act: &Activation,
    out_shape: [i64; 4],
    y: &mut [f32],
    residual: Option<&[f32]>,
) -> [i64; 4] {
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
    debug_assert_eq!(y.len(), out_elems, "conv2d_res_into: 输出切片长度");

    let depthwise = group > 1 && cg == 1;
    let yp = par::SyncPtr::new(y.as_mut_ptr());
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
        // 转置稠密路径（2026-11-04，预打包立项 Phase 1）：
        //
        //   Yᵀ[R, m] = Xᵀ[R, c] × Wᵀ[c, m]，R = n·ohw
        //
        // 旧路径按批项算 C[m_out, ohw]——rec 热点是 m_out=512~1536、ohw=30
        // 的**高瘦** GEMM（np=1），并行分支落到 `sgemm_mrows` 标量广播
        // （gemm.rs 文档：比面板内核慢 2.6~6.4×），批内分支即便走面板内核
        // 也只有 N=30 的窄面板。微基准（gemm_bench::rec_dense_transposed_
        // shootout）：现行形态 ~86 GFLOP/s（串行）/引擎内聚合 ~100，
        // 转置形态 np=48 宽面板 **4 路并行 267-283 GFLOP/s**——分片结构
        // 不动，热算子 ~2.7×。这正是 NHWC 形态的 GEMM（对方实现同构）。
        //
        // 位级：k 仍升序、每输出元素寄存器内全 k 累加；bias 移到回写循环
        // 「累加后最后加」——与窄面板路径的 bias-末加同序（与 mrows 的
        // bias-起种不同序，n==1 单行批低位可能变化，CTC 文本层验证兜底，
        // 先例：gemm.rs 的 T≥3 路径拉平）。act 与 residual 的次序照旧
        // （先 act 后 +residual，同 finish_res）。
        //
        // Wᵀ 每调用转置一次（~1ms 级）；装载期缓存是判决门通过后的优化，
        // 不在本步（安全落点在 session 层，避免 ptr 键缓存的一致性风险）。
        //
        // ★ 判决结果（2026-11-04，整页语料交错 3 轮）：**默认关**
        //   （`QPPOCR_DENSE_T=1` 显式开）。文本 24/24 逐字一致，但 rec 相位
        //   12.2s vs 旧 10.7s（**慢 15%**）——微基准赢（4 路 267 GFLOP/s vs
        //   引擎聚合 ~100）引擎里输，机理：微基准是单 GEMM 独占缓存；引擎里
        //   **4 个 shard 并发跑 4 个不同层**，转置形态让每个 worker 都流全量
        //   权重（4×4.7MB ≫ L3，加上各自 Xt/Yt ≈ 47MB 工作集），而旧路径的
        //   M 切分把权重按行私有化——每个 worker 只摸 1/16 切片，并发下
        //   缓存优势压倒形态优势。**教训（勿重试同款）：分片模型里
        //   「共享权重流」是负资产；权重私有的 M 切分才是分片的朋友。**
        //   Phase 2（NHWC 全链，转置被链上消化而非每次转进转出）不受此
        //   判决否定——它改变的是工作集结构本身；但须自带并发层的 L3 论证。
        let dense_t = std::env::var("QPPOCR_DENSE_T").ok().as_deref() == Some("1");
        if dense_t && m >= 64 {
            let r_total = n * ohw;
            let mut xt = crate::buf::F32Buf::new();
            let mut yt = crate::buf::F32Buf::new();
            // SAFETY: 两个 scratch 都被下方步骤写满（Xt 由转置循环、Yt 由
            // sgemm 的 store 写满），没有先读。
            unsafe {
                xt.resize_uninit(r_total * c);
                yt.resize_uninit(r_total * m);
            }
            // Wᵀ[c, m]：权重 [m, c] 转置（contiguous B，供面板内核宽面板流）。
            let mut wtp = crate::buf::F32Buf::new();
            // SAFETY: scratch 由下方转置循环写满，没有先读（resize_uninit 契约）。
            unsafe {
                wtp.resize_uninit(c * m);
            }
            {
                let wt_s = wtp.as_mut_slice();
                for ch in 0..c {
                    for o in 0..m {
                        wt_s[ch * m + o] = w[o * c + ch];
                    }
                }
            }
            // Xᵀ：Xt[bn·ohw+pos, ch] = x[bn·c·ohw + ch·ohw + pos]（按行并行）。
            {
                let xts = xt.as_mut_slice();
                let xpp = par::SyncPtr::new(xts.as_mut_ptr());
                par::parallel_for_elems(r_total, r_total * c, |rb, re| {
                    for r in rb..re {
                        let bn = r / ohw;
                        let pos = r - bn * ohw;
                        // SAFETY: 各行区间不相交；x 只读。
                        let dst = unsafe { xpp.offset(r * c).slice(c) };
                        let src = &x[bn * c * ohw..];
                        for ch in 0..c {
                            dst[ch] = src[ch * ohw + pos];
                        }
                    }
                });
            }
            // 宽面板 GEMM（np = m/32 ≥ 2 → 面板内核并行路径）。无 bias 无
            // act——两者在回写循环按原次序施加。
            {
                let yts = yt.as_mut_slice();
                crate::gemm::sgemm(
                    xt.as_slice(),
                    wtp.as_slice(),
                    yts,
                    r_total,
                    m,
                    c,
                    m,
                    None,
                    &crate::activation::Activation::default(),
                );
            }
            // 回写（按通道平面，y 的每个 ch 平面是 ohw 长的连续段）：
            //   y[(bn·m+ch)·ohw .. +ohw] = act( Yt[bn·ohw+pos, ch] + bias[ch] )
            //   之后逐元素 + residual（与 finish_res 同序：先 act 后 +res）。
            //   bias 在「累加完成后」加，与窄面板路径的 bias-末加同序。
            {
                let yts = yt.as_slice();
                let ypp = yp;
                par::parallel_for_elems(n * m, n * m * ohw, |b0, e0| {
                    for cm in b0..e0 {
                        let bn = cm / m;
                        let ch = cm - bn * m;
                        // SAFETY: 各通道平面 [cm·ohw, ...) 两两不相交。
                        let yplane = unsafe { ypp.offset(cm * ohw).slice(ohw) };
                        for pos in 0..ohw {
                            let mut v = yts[(bn * ohw + pos) * m + ch];
                            if let Some(b) = bias {
                                v += b[ch];
                            }
                            yplane[pos] = v;
                        }
                        if act.is_on() {
                            crate::activation::apply_act(yplane, act);
                        }
                        if let Some(rs) = residual {
                            let rs = &rs[cm * ohw..cm * ohw + ohw];
                            for pos in 0..ohw {
                                yplane[pos] += rs[pos];
                            }
                        }
                    }
                });
            }
            return out_shape;
        }
        // ---- 旧路径（m<64 的微小形状 / QPPOCR_DENSE_T=0 回退）----
        // C[n] = W * X[n]，整个批次一次 fork
        //
        // ★ fork 粒度必须是大块（chunk=1 的单通道小作业派发风暴实测让
        //   batch=8 慢 2×）；n==1 保持串行——分片上下文里 4 行并发再各自
        //   切 fork 会把 L3 冲爆（多行 X 流×fork 切片数），实测慢 2~3×。
        //   批内并行（n>1）摊薄权重/X 的流式读，是带宽墙的正解。
        //
        //   ⚠ 已验证的负结果（2026-11-03，勿重试同款）：「n==1 也开 fork +
        //   ROW_THREADS=1 逐行顺序」复刻「单线程跑满全机」的模型，本机实测比
        //   分片串行基线慢 1.5~2.5×（三种轮次均负）——fork_join 的
        //   发布/唤醒/join-spin 成本在这种 0.5~1ms 级切片上压过了收益。
        //   要走「单线全机」得给 rec 配专门的常驻 worker 队伍 + 静态
        //   stride 切分（零派发），是 pool 层的结构工程。
        let rows = n * m;
        if n > 1
            && par::threads() > 1
            && 2.0 * (rows * ohw * c) as f64 >= par::thresholds().gemm_par_min
        {
            let chunk = rows.div_ceil(par::threads() * 2).max(1);
            par::parallel_for(rows, chunk, |rb, re| {
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
            // np=1（ohw ≤ 32）落 `sgemm_mrows` 标量广播内核——曾试「X 右
            // 填充到 64 列换面板内核」（微基准 4 路 7.2×），引擎交错实测
            // 整页样张 0.993、全语料 1.000（纯持平）——微基准单 GEMM 独占
            // 缓存，引擎里 4 shard 并发跑不同层把每次调用打到缓存冷，
            // 内核形态不再是变量（「缓存冷墙」效应）。
            // 已回退（2026-11-04）。np=1_pad_shootout 微基准留档。
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
            // 每块自己的 scratch，跨本块的 tile 复用；cols/slab/ptrs 走
            // thread-local 缓存（跨 conv 调用、跨图存活，只增不缩）。
            // （试过「共享池化」的 Buf，实测更慢——0.878 over 13 对 @16
            // 线程，单线程也 0.966——那是全局锁争用；thread-local 无锁。）

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
                    // slab 缓存：fill(0) 与 vec![0f32; ..] 数值等价（padding
                    // 与行尾 slack 必须是真零；slack 列的垃圾可容忍——只进
                    // 被丢弃的累加器 lane，不进 store）。
                    TILE_SLAB.with(|sc| {
                        let slab_buf = &mut *sc.borrow_mut();
                        let slab_need = cg * slab_rows * pwidth;
                        if slab_buf.len() < slab_need {
                            slab_buf.resize(slab_need, 0.0);
                        }
                        // ★ 行级最小写入：旧做法先 fill(0) 整个 slab 再拷贝有效行
                        //   ——有效行写两遍（先零后拷）、32 列 slack 也白清（slack
                        //   垃圾本就可容忍，只进被丢弃的累加器 lane）。改成每行
                        //   只写「左补边零 + 有效拷贝 + 右补边零」：上/下越界行
                        //   整行清零到内核触达宽度。det 单图补零搬运 401.8MB
                        //   里的绝大部分正是这块的双重写（QPPOCR_PROF 实测）。
                        //   位级：零的语义与位置和整块 fill 完全一致。
                        let rpad_to = (ow - 1) * p.sw + kw; // 内核实际触达的行宽
                        let slab: &mut [f32] = &mut slab_buf[..slab_need];
                        for ch in 0..cg {
                            let xch = &xg[ch * h * wdim..];
                            for y in 0..slab_rows {
                                let iy = (oy0 * p.sh) as isize - p.ph as isize + y as isize;
                                let dst = (ch * slab_rows + y) * pwidth;
                                if iy < 0 || iy as usize >= h {
                                    // 上下越界行：纯补边，清到触达宽（slack 不写）
                                    slab[dst..dst + rpad_to].fill(0.0);
                                    continue;
                                }
                                let iy = iy as usize;
                                slab[dst..dst + p.pw].fill(0.0);
                                slab[dst + p.pw..dst + p.pw + wdim]
                                    .copy_from_slice(&xch[iy * wdim..(iy + 1) * wdim]);
                                if rpad_to > p.pw + wdim {
                                    slab[dst + p.pw + wdim..dst + rpad_to].fill(0.0);
                                }
                            }
                        }
                        // 每个输出行一次 GEMM，n = ow。B 的第 k 行 = slab 里
                        // (ch, r*sh+ky, ·+kx) 那一行，k 序与 im2col 完全一致
                        // （ch → ky → kx），所以逐位相同。
                        TILE_PTRS.with(|pc| {
                            let ptrs_buf = &mut *pc.borrow_mut();
                            ptrs_buf.clear();
                            for r in 0..rows {
                                ptrs_buf.clear();
                                for ch in 0..cg {
                                    for ky in 0..kh {
                                        for kx in 0..kw {
                                            // SAFETY: slab 在本次循环内不变；偏移落在
                                            // cg × slab_rows × pwidth 之内（r < rows）。
                                            ptrs_buf.push(unsafe {
                                                slab.as_ptr().add(
                                                    (ch * slab_rows + r * p.sh + ky) * pwidth + kx,
                                                )
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
                                    ptrs_buf,
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
                        });
                    });
                    continue;
                }
                TILE_COLS.with(|cc| {
                    let cols_buf = &mut *cc.borrow_mut();
                    let col_need = kk * rows * ow;
                    if cols_buf.len() < col_need {
                        cols_buf.resize(col_need, 0.0);
                    }
                    let cols: &mut [f32] = &mut cols_buf[..col_need];
                    im2col(
                        xg, cg, h, wdim, kh, kw, p.sh, p.sw, p.ph, p.pw, ow, oy0, rows, cols,
                    );
                    sgemm_serial_res(
                        wg,
                        cols,
                        yg,
                        mg,
                        rows * ow,
                        kk,
                        ohw,
                        bias.map(|b| &b[g * mg..]),
                        act,
                        None, // 残差由函数尾部的统一趟处理（非 1x1 路径）
                    );
                });
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

    if let Some(b) = bias
        && !bias_in_gemm
    {
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
    // 非 1x1 路径的残差：bias 之后逐元素加（与独立 Add 同值同序）。
    if let Some(r) = tail_residual {
        let total = out_elems;
        let yp2 = par::SyncPtr::new(y.as_mut_ptr());
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

// ── NHWC 稠密/depthwise（Phase 1 rec 主路径，docs/rec-layout-refactor.md）──
//
// Phase 0 微链对决（examples/nhwc_chain.rs）的判决形态落地：激活 [R,C]、
// 权重装载期转置预打包成 Wᵀ[K,M] 当 B——np=M/32 宽面板 + 权重按面板切片
// 私有（每 worker ~0.79MB，L2 装得下），4 shard 并发各停不同层也打不冷它。
// 与 NCHW 现状（np=1 落 mrows 标量）的对决已判：本路径胜出为默认。

/// NHWC 稠密 1×1：`x` 是 `[R,K]`（n·h·w 折平，channels-last 物理序），
/// `wt` 是装载期预打包的 `Wᵀ[K,M]`。bias/act/residual 的施加次序与
/// NCHW 路径一致：GEMM → bias → act → +residual。输出 `[R,M]` 写满
/// `y`（调用方负责形状元数据）。
///
/// 位级：k 升序累加与面板内核一致；bias 是「加在全 k 和之后」而 mrows
/// 是「起种」——低位可与旧路径不同（CTC 对低位差免疫，判据 = 文本
/// 逐字一致 + GT CER 不回归）。
#[allow(clippy::too_many_arguments)]
pub fn conv1x1_nhwc(
    x: &[f32],
    wt: &[f32],
    r: usize,
    k: usize,
    m: usize,
    bias: Option<&[f32]>,
    act: &Activation,
    residual: Option<&[f32]>,
    y: &mut F32Buf,
) {
    if r == 0 || m == 0 {
        return;
    }
    // SAFETY: sgemm 写满 r*m 个输出（每元素寄存器算满再 store），无先读。
    unsafe { y.resize_uninit(r * m) };
    let ys = y.as_mut_slice();
    sgemm(x, wt, ys, r, m, k, m, None, &Activation::default());
    // bias → act → residual 融合为**单趟**（与 NCHW finish_res 同序同 fork
    // 语义：三次独立扫会让宽行输出（364×1024×4B≈1.5MB）多付 ~2×3 趟冷读写，
    // 实测把面板收益吃成净负）。SyncPtr 模式镜像 gemm::finish_res。
    let cp = par::SyncPtr::new(ys.as_mut_ptr());
    let rp = residual.map(|r_| par::SyncPtr::new(r_.as_ptr() as *mut f32));
    let finish = |b: usize, e: usize| {
        for row in b..e {
            // SAFETY: 行区间 [b, e) 两两不相交，每块只写自己的行。
            let s = unsafe { cp.offset(row * m).slice(m) };
            if let Some(bv) = bias {
                for (v, b0) in s.iter_mut().zip(bv) {
                    *v += b0;
                }
            }
            if act.is_on() {
                crate::activation::apply_act(s, act);
            }
            if let Some(rr) = rp {
                // SAFETY: residual 与 c 同形，行区间 [b, e) 不相交且只读。
                let rs = unsafe { rr.offset(row * m).slice(m) };
                for (v, r0) in s.iter_mut().zip(rs) {
                    *v += *r0;
                }
            }
        }
    };
    crate::par::parallel_for_elems(r, r * m, finish);
}

/// NHWC depthwise（3×3 等）：`x` 是 `[n,h,w,c]` 物理序，权重是**装载期
/// tap-major 预打包**的 `wt[kh*kw, c]`（rec_nhwc_island 把 [c,1,kh,kw]
/// 重排过来——固定 tap 下 c 连续，内层循环跨通道向量化）。
///
/// 与 NCHW `depthwise_plane_*` 同一条乘加链：tap 序 ky 外层 kx 内层、
/// 逐通道 fma 累加、bias 在全 tap 和之后加——逐位可比（越界 tap 走
/// 补零读，`fma(w,0,acc)=acc`，-0/+0 边角由逐字符对拍兜底）。
/// 输出 `[n,oh,ow,c]`；返回输出形状。
#[allow(clippy::too_many_arguments)]
pub fn depthwise_nhwc_packed(
    x: &[f32],
    x4: &[i64; 4],
    wt: &[f32],
    kh: usize,
    kw: usize,
    p: &ConvParams,
    bias: Option<&[f32]>,
    act: &Activation,
    residual: Option<&[f32]>,
    y: &mut F32Buf,
) -> [i64; 4] {
    let (n, h, wd, c) = (
        x4[0] as usize,
        x4[1] as usize,
        x4[2] as usize,
        x4[3] as usize,
    );
    let taps = kh * kw;
    let oh = (h + p.ph + p.peh - kh) / p.sh + 1;
    let ow = (wd + p.pw + p.pew - kw) / p.sw + 1;
    let h2 = h + p.ph + p.peh;
    let w2 = wd + p.pw + p.pew;
    let plane = wd * c;
    let ohw = oh * ow;
    let need_pad = n * h2 * w2 * c;
    // 补零输入：**只清边距**（上下补边行 + 左右补边列），内区由下面的逐行
    // copy 全覆盖。旧做法在尺寸变化时整块 fill(0)——det 里多个 depthwise
    // 形状交替（k9x9 的 43.7MB、k7x7 的 5.5MB……），每次交替都把整块重清
    // 一遍；边距体积只有整块的 ~5%（k9x9 档 ~2MB），且「每趟清边」对历史
    // 脏数据免疫（换形状后旧内区可能落在新边距位置）。
    NHWC_PAD.with(|pb| {
        let mut pad = pb.borrow_mut();
        if pad.len() != need_pad {
            // SAFETY: 下方立即清边距、内区每行 copy 写满，没有先读。
            unsafe { pad.resize_uninit(need_pad) };
        }
        {
            let pm = pad.as_mut_slice();
            let rows = h2 * w2 * c;
            for bnn in 0..n {
                let pb_ = &mut pm[bnn * rows..(bnn + 1) * rows];
                // 上下补边行（整行）
                for r in 0..p.ph {
                    pb_[r * w2 * c..(r + 1) * w2 * c].fill(0.0);
                }
                for r in h2 - p.peh..h2 {
                    pb_[r * w2 * c..(r + 1) * w2 * c].fill(0.0);
                }
                // 有效行的左右补边列
                for r in p.ph..p.ph + h {
                    let row = r * w2 * c;
                    pb_[row..row + p.pw * c].fill(0.0);
                    pb_[row + (p.pw + wd) * c..row + w2 * c].fill(0.0);
                }
            }
        }
        {
            let pm = pad.as_mut_slice();
            let rows = h2 * w2 * c;
            for bnn in 0..n {
                for ih in 0..h {
                    let dst = bnn * rows + (p.ph + ih) * w2 * c + p.pw * c;
                    let src = bnn * plane + ih * plane;
                    pm[dst..dst + plane].copy_from_slice(&x[src..src + plane]);
                }
            }
        }
        // body 里只读共享切片（&mut 过不了 fork 闭包的 Sync 检查）。
        let pp: &[f32] = pad.as_slice();
        let rows = h2 * w2 * c;

        // SAFETY: 下方 body 写满每个输出元素（首 tap 写、余 tap 累加、
        // 尾加 bias），没有先读。
        unsafe { y.resize_uninit(n * ohw * c) };
        let yp = par::SyncPtr::new(y.as_mut_slice().as_mut_ptr());
        let body = |ub: usize, ue: usize| {
            for u in ub..ue {
                let (bnn, orow_, ocol) = (u / ohw, (u / ow) % oh, u % ow);
                let dst = (bnn * ohw + orow_ * ow + ocol) * c;
                for t in 0..taps {
                    let (ky, kx) = (t / kw, t % kw);
                    let sy = orow_ * p.sh + ky * p.dh; // 已在补零坐标系内
                    let sx = ocol * p.sw + kx * p.dw;
                    let src = bnn * rows + sy * w2 * c + sx * c;
                    let wtp = &wt[t * c..(t + 1) * c];
                    for ci in 0..c {
                        // SAFETY: 各 u 的输出像素互不相交；px 只读。
                        unsafe {
                            let yp2 = yp.offset(dst + ci).get();
                            if t == 0 {
                                *yp2 = wtp[ci] * pp[src + ci];
                            } else {
                                *yp2 = wtp[ci].mul_add(pp[src + ci], *yp2);
                            }
                        }
                    }
                }
                if let Some(bv) = bias {
                    for ci in 0..c {
                        // SAFETY: 同上，像素不相交。
                        unsafe {
                            *yp.offset(dst + ci).get() += bv[ci];
                        }
                    }
                }
            }
        };
        let units = n * ohw;
        if par::worth_forking((units * taps * c) as f64) {
            par::parallel_for(units, 1, body);
        } else {
            body(0, units);
        }
    });
    // act/residual：与 conv2d_res 的非 GEMM 路径同序（写完 y 含 bias，
    // 再施加 act，最后加 residual）——单趟融合。
    let ys = y.as_mut_slice();
    let cp = par::SyncPtr::new(ys.as_mut_ptr());
    let rp = residual.map(|r_| par::SyncPtr::new(r_.as_ptr() as *mut f32));
    let finish = |b: usize, e: usize| {
        for px_ in b..e {
            // SAFETY: 像素区间 [b, e) 两两不相交。
            let s2 = unsafe { cp.offset(px_ * c).slice(c) };
            if act.is_on() {
                crate::activation::apply_act(s2, act);
            }
            if let Some(rr) = rp {
                // SAFETY: residual 与 y 同形，像素区间 [b, e) 不相交且只读。
                let rs = unsafe { rr.offset(px_ * c).slice(c) };
                for (v, r0) in s2.iter_mut().zip(rs) {
                    *v += *r0;
                }
            }
        }
    };
    let pxn = n * ohw;
    crate::par::parallel_for_elems(pxn, pxn * c, finish);
    [n as i64, oh as i64, ow as i64, c as i64]
}
