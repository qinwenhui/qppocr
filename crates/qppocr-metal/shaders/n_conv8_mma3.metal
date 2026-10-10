// 稠密 kxk conv（k>1，NHWC f32，**任意 stride / pad**）——simdgroup_matrix MMA 版。
//
// # 为什么要有这个内核
//
// `n_conv8` 的 k×k 路和 1×1 路是**同一个病根**：m4×n8 瓦片里每条权重装载
// 只被 4 个输出行复用。变体探针在 3×3 形状上给的判决（`conv8_variant_scan`）：
//
//   det 480x432 c32->32 k3：p_base 1157 GFLOPS、p_fmaonly **2758**（81% 峰值）、
//   p_wconst 1800、**p_wsmall 1125（权重掩进 1 KB 无收益）**、
//   **p_colmaj 1152（列主序无收益）**、p_nocoal 1240（输入合并只值 +7%）。
//
// 即：既不是权重**流量**（3×3 的权重只有几十 KB，本来就 L2 常驻），也不是
// 输入合并，更不是 lane 排布——就是**装载指令条数**。去掉重量装载 1.55×，
// 再去掉输入装载 2.38×。`simdgroup_float8x8` 把装载变成 warp 协作（每 lane
// 只搬自己那一小片），正是这条处方；1×1 上它已经兑现 2.0~2.5×。
//
// # 与 n_conv8_mma 的关系
//
// 累加器布局、瓦片映射、尾声、参数语义的**结构**与 `n_conv8_mma` 同源，
// 只多一层 taps 循环与一个「块 = 一行内的 64 个连续输出 x」的映射。见下。
//
// # m 的映射：为什么是「行内 64 个连续 x」
//
// k×k 与 1×1 的根本差别是 **A 操作数不再是恒等映射**：输出像素 m 的第 t 个
// tap 要读输入像素 m 的空间邻居。若让 m 沿整幅图行主序线性铺开（1×1 那样），
// 每行的行首/行尾会**横向绕行**——x+dx-padL = -1 读到的是上一行最后一个
// 像素（**错值，不是零**），而且 y=0 时再往前就越过输入平面基址了（那就是
// 未绑定的 BIF0 page fault 那条致瘸路径）。
//
// 处方：**一个块只覆盖一行内的 64 个连续输出 x**。这样对固定的 (y,t)，
//   输入下标 = (y*sH+ky-padT)*inW + (x*sW+kx-padL)
//            = [常量] + i*sW       （i = x - x0 ∈ [0,64)）
// 是 i 的**一次函数**，A 瓦片的 64 行正好是**同一行内等距的 64 个像素**
// （行 stride = sW*kf = 步长 × 每像素通道数）——一条 `simdgroup_load` 就够，
// 零 gather。**stride 只改这个 stride，不改结构**：`simdgroup_load` 接受任意
// 行 stride，所以 stride≠1 并不需要 im2col/gather（竞品 `mm_ic_sg` 那套）。
// 权重侧零重排的理由同 `n_conv8_mma`（taps 布局 [tap][k][n] 就是 K×N 行主序）。
//
// # 只收「全内点块」——越界这一条是硬约束，不是优化
//
// 计划层只把**满足下式**的块派给本内核（内点行 × 块安全的列）：
//
//   y ∈ [yd_lo, yd_hi]，yd_lo = ceil(padT/sH)、yd_hi = (inH-1-(kH-1)+padT)/sH
//   x0 ∈ [xa, xb-63]，xa = ceil(padL/sW)、xb = (inW-1-(kW-1)+padL)/sW
//
// 于是对任意 tap、任意 i ∈ [0,64)，读到的输入下标都落在 [0, inH*inW) 内
// ——**由数学保证**，不需要任何对齐/担保，也绝不触碰输入平面之外（
// `n_conv8_mma` 那条「尾块读到 arena 里别的段」的推理在这里不适用也不需要）。
// 真正的边框由 `n_conv8_edge` 补，两者写的是**互不相交**的输出像素。
//
// stride1 + 对称 pad 时这组端点退化成 `[padT, padT+nrows-1]` 与
// `[padL, padL+inner_w-1]`（旧式），所以那次泛化对既有形状**逐位等价**；
// 新放开的只有 stride≠1 与非对称 pad。**三份拷贝必须一致**：这里、
// `n_conv8.metal` 的 `conv8_body<EDGE>`、`plan.rs` 的 `KxkCover::new`。
//
// # 行尾余数：为什么值得一个「读重叠、写裁掉」的尾块
//
// 内点列宽 `inner_w` **不是** 64 的倍数（det 的 240 → 238、rec 的 544 → 542）。
// 若按 `nfb = inner_w/64` 取整铺块，余下的 `inner_w mod 64` 列（240 那档是
// **48 列 = 全图 20% 的像素**）就全落到 `n_conv8_edge` 的慢路上。
//
// 实测这代价很大：`conv8_mma3_throughput` 上 det 136x240 c256->64 k3
// 一开始是 `mma3 2.995 + edge 2.695 ms = 5.69 ms`（1.49×）——edge 每像素比
// `n_conv8` 还贵约 1.5×，因为它按 4 像素一瓦片发射**全部**权重装载，而边框
// 瓦片大多只填 1~2 个像素。**慢路的像素占比就是主动收益的漏点。**
//
// 处方：本内核多铺**一个**尾块，锚在**最后 64 个内点列**上——
//
//   xblk <  nfb → x0 = padL + 64*xblk，       写整块 64 列
//   xblk == nfb → x0 = padL + inner_w - 64，  只写最后 `inner_w mod 64` 列
//
// 关键在于**读和写解耦**：A 的 64 列读会与前面的块重叠，但（a）数学上仍在
// 平面内——x0 + kx - padL ≤ inner_w - 64 + (kW-1) = inW - 64 ≤ inW-1，且
// x0 ≥ padL 因为 nfb ≥ 1 ⇒ inner_w ≥ 64；（b）**写入被掩码裁到未覆盖的
// 那几列**（`i >= i_lo`），所以输出像素仍恰好被写一次、无重叠写（两个线程组
// 写同一地址即使值相同也是竞争，不去碰）。
//
// 尾块只在 `inner_w % 64 != 0` 时铺（计划层决定），所以余数为 0 的形状一格
// 都不多。尾块在**行尾**上，网格里是 `wg.x` 的最后 `nch` 个值。
//
// # 网格
//
//   groups = [nx * nch, outH-(taps/kW - 1), batch]，nch = nDim/64，
//   **nx = nfb + (inner_w % 64 != 0)**（余数为 0 时没有尾块）。
//   **通道块在 wg.x 里变化最快**（`wg.x % nch`）——相邻线程组取同一片 A
//   （A 是 DRAM 大头），权重小得多且本就 L2 常驻。
//
//   `outH` 由 `mDim/outW` 导出、`kH` 由 `taps/kW` 导出（参数块里没有这两
//   个数，见下一节）；**计划层的网格公式必须与本内核逐字一致**，否则覆盖
//   矩形错位——见文件头末段。
//
// 线程组内：NSG 个 simdgroup 排成 (NSG/2)×2，sgid 认领 (64/(NSG/2)) 行
// （像素）× NCOLS/2 列（通道）。**覆盖矩形恒为 64 行像素 × NCOLS 列通道，
// 与 NSG 无关**——这正是三档（两条实例化 × 两种 NSG）能共用尾块/边框推理
// 的原因。
//
// # 装载/MMA 比值：为什么 n32 那条用 128 线程（4 个 simdgroup）
//
// 一个 simdgroup 的瓦片是 M 行 × N 列，每走一步 K=8 要装 M/8 个 A 分片 +
// N/8 个 B 分片，换 (M/8)(N/8) 条 MMA。**每 MMA 的装载数 = (M+N)/8 ÷ (MN/64)
// = 8(M+N)/(MN)**，即装载发射槽与 MMA 的配比。按 M+N 一样本：

//   16×16（256 格）→ 1.00      ← n32 8 组：装 4 条、做 4 条 MMA
//   16×32（512 格）→ 0.75      ← n64 8 组（现状）
//   32×16（512 格）→ 0.75      ← n32 4 组（**本档**，同一块 64×32 瓦片，线程减半）
//   32×32（1024 格）→ 0.50     ← n64 4 组（**实测 0.35×，被寄存器压力打死**）
//
// 这是**量出来**的瓶颈，不是推的：`microbench::conv8_mma3_load_attribution`
// 把三种「拆掉装载」的假想内核与真内核同台量（六条 arm 的判读规则见探针头），
// 结论是三行——
//
//   · n32 的 base/hoist = **2.1×**、base/small = 1.2× ⇒ 卡在**装载发射槽**，
//     于是抬瓦片（1.00 → 0.75）**值得做**，实测 **1.26×**。
//   · n64 的 base/hoist = 1.35× ⇒ 那条实例化只有三成时间在装载上，而它的
//     0.75 已经和 n32 的目标档同值 ⇒ **没有可抬的余地**。
//   · 唯一剩下的比 0.75 更低的一档（n64 抬到 0.50）实测 **0.35×**：累加器
//     分片从 8 涨到 16（每 lane 16 → 32 个 f32 寄存器）直接把内核打崩。
//     **别照着比值表再试这一档**——这就是 `n_conv_dw` 那条教训的形状。
//
// 代价是**宿主侧要按内核名给线程组尺寸**（`pipeline.rs::threads_per_group`）：
// 网格组数、覆盖矩形、尾块锚点与掩码、参数、`ost` 尺寸全都一字不改。
//
// # 参数块：**与 `n_conv8` 逐字段同一套**（P+0..20，见 n_conv8.metal 的
// `conv8_body` 读法）。理由不是省事，是**必须**：
//
//   · 计划层一次 conv 要派**两条** dispatch（本内核 + `n_conv8_edge`），
//     两条在同一条 command buffer 里背靠背；参数块是 arena 里的一段普通
//     内存，要在两次 dispatch 之间改它就是「编码期 CPU 写 GPU 正在读的
//     缓冲」——数据竞争，不是慢一点的问题。
//   · 于是两条必须**读同一块参数**。这里需要的每个字段（mDim/outW/inW/
//     inH/nDim/cin4v/taps/kW/padT/padL/act/c1..c3/r_off）标准块里都有，
//     `outH` 还能由 `mDim/outW` 导出（输出平面恒为 outH*outW），sH/sW 由
//     启用门保证恒为 1 而**不被本内核读取**。所以共享是可行的。
//   · 权重/输入/输出的字节语义与 `n_conv8` 完全一致（同一份 repack 布局、
//     同一个 arena），两支持同一张权重表。
//
// 代价：`outH = mDim/outW` 这类导出式在**计划层与内核里各写一份**，两份
// 对不上就是「覆盖矩形错位」（漏算或算两遍，都是静默错）。这是本设计唯一
// 的新耦合面，`conv8_mma3_matches_cpu` 用独立硬编码的第三份做交叉验证。
//
// # 32 通道块（`n_conv8_mma3_n32`）：为什么必须有
//
// 第一版只收 `Co%64==0`，端到端一量——**medium det 整张图只有 1 条 conv
// 点火**（`QPPOCR_GPU_KERNEL_PROF=1` 带节点名：`Conv.121`，Co=64 Ci=256
// 240x216）。把模型的 Conv 全扫一遍（`bench/gpuhealth/kxk_scan.py`）才看清
// 门外站着谁：
//
//   Conv.111/114/117（k=7/5/3，**Co=32 Ci=32**，同一层的三连）在 **240x216**
//   ——`outW-(k-1) = 234 >= 64` 空间门**过关**，只被 `Co%64≠0` 一条卡住；
//   Conv.100/103/106 同构、在 120x108，同样只差这一条。
//
// 即：**挡住的不是几何，是通道块宽度**。而 simdgroup_matrix 的 8x8 分片本来
// 就允许把 N 开窄——把 64 列块换成 32 列块，A 的复用从 4 个分片降到 2 个，
// 但**每个分片仍是 8x8 的 MMA**，ALU 照样喂满。
//
// N 开窄的代价是每 MMA 的装载数从 0.75 涨到 1.00（见「装载/MMA 比值」一节），
// 那条实例化因此**卡在装载发射槽**上；处方是把它的 simdgroup 从 8 个减到 4 个
// （行方向每 sg 16 → 32），瓦片不变、比值回到 0.75，实测 1.26×。
//
// 计划层按 `Co%64==0 ? 64 : 32` 选内核名（`co4%32 != 0` 时回落 `n_conv8`）。
// 两条都是本模板的实例化，瓦片映射/覆盖矩形/尾块/边框的推理**逐字共用**。
#include <metal_stdlib>
using namespace metal;

constant uint OFF_NONE = 0xFFFFFFFFu;

// exp1/erf1/act1：`n_conv8_mma` 的逐字拷贝（同一套与 CPU 位级一致的多项式）。
float exp1(float x) {
    const float LN2_HI = 0.6931472;
    const float LN2_LO = -2.9802322e-8;
    const float INV_LN2 = 1.442695;
    x = clamp(x, -88.0f, 88.0f);
    float kf = rint(x * INV_LN2);
    float r = fma(-kf, LN2_HI, x);
    r = fma(-kf, LN2_LO, r);
    float p = 1.0 / 720.0;
    p = fma(p, r, 1.0 / 120.0);
    p = fma(p, r, 1.0 / 24.0);
    p = fma(p, r, 1.0 / 6.0);
    p = fma(p, r, 0.5);
    p = fma(p, r, 1.0);
    p = fma(p, r, 1.0);
    int ki = int(kf) + 127;
    return as_type<float>(uint(ki) << 23) * p;
}

float erf1(float x) {
    float ax = fabs(x);
    float t = 1.0 / fma(0.3275911, ax, 1.0);
    float p = 1.0614054;
    p = fma(p, t, -1.453152027);
    p = fma(p, t, 1.421413741);
    p = fma(p, t, -0.284496736);
    p = fma(p, t, 0.2548296);
    p *= t;
    float e = exp1(-ax * ax);
    float r = fma(-p, e, 1.0);
    return as_type<float>(as_type<uint>(r) | (as_type<uint>(x) & 0x80000000u));
}

float act1(float v, uint act, float c1, float c2, float c3) {
    if (act == 2u) return max(v, 0.0f);
    if (act == 1u) {
        float inv = 1.0 / c1;
        return c3 * v * (erf1(v * inv) + c2);
    }
    return v;
}

// NCOLS = 输出通道块宽（64 或 32）；NSG = 本线程组里 simdgroup 的个数
//（8 → 256 线程，4 → 128 线程）。文件头「32 通道块」与「装载/MMA 比值」两节
// 说明这两档各是什么、为什么。除这两行的参数与由它们派生的四处（nch / sn /
// 分片数 / 尾声步长）外，两条实例化的推理**逐字共用**——瓦片映射、覆盖矩形、
// 尾块掩码、`outH = mDim/outW` 的导出全在下面这一段里，改一处两条同时改。
// `ost`（分片落地的 threadgroup 暂存）**必须由内核入口声明、以指针传进来**：
// MSL 硬规则——`threadgroup` 变量只能在 `kernel` 函数里声明（"variables in
// the threadgroup address space cannot be declared in a non-qualified
// function"）。于是各入口声明自己那份 `float[64*NCOLS]`，尺寸耦合由两处的
// static_assert 守住。
//
// ⚠️ **NSG 必须与宿主侧的线程组尺寸一致**（`pipeline.rs::threads_per_group`）。
// 内核按 `tid>>5` 认 simdgroup，线程组给大了就会有线程认领到不存在的 sg、
// 去写 `ost` 的界外行——静默算错。这道耦合由 `conv8_mma3_matches_cpu`
// 拿独立硬编码的第三份做交叉验证兜住（同网格公式那道耦合）。
template <uint NCOLS, uint NSG>
static void mma3_body(
    device float4* d,
    device const uint* params,
    uint P,
    uint tid,
    uint3 wg,
    threadgroup float* ost)
{
    // 分片数：每 16 个通道一个 8x8 分片列。
    constexpr uint NB = NCOLS / 16u;      // 64 -> 4，32 -> 2
    // 行方向：64 行像素均分给 NSG/2 个行组，每组再切成 8 行的分片。
    constexpr uint RPSG = 64u / (NSG / 2u);   // 每 simdgroup 的行数（8 -> 16，4 -> 32）
    constexpr uint NR = RPSG / 8u;            // 每 simdgroup 的行分片数（8 -> 2，4 -> 4）
    constexpr uint NTH = NSG * 32u;           // 线程组大小（8 -> 256，4 -> 128）

    device const float* f32v = (device const float*)d;
    device float* of = (device float*)d;

    // 与 `n_conv8` 同一套参数字（见文件头）。
    uint in_off = params[P + 0], w_off = params[P + 1], b_off = params[P + 2];
    uint out_off = params[P + 3];
    uint mDim = params[P + 4], nDim = params[P + 5], outW = params[P + 6];
    uint inW = params[P + 7], inH = params[P + 8], cin4v = params[P + 9];
    uint taps = params[P + 10], kW = params[P + 11];
    // sH/sW 现在**要读了**（P+12/P+13）：覆盖几何与 A 瓦片的行 stride 都
    // 由它们决定。见下面「覆盖几何」那段。
    uint sH = params[P + 12], sW = params[P + 13];
    uint padT = params[P + 14], padL = params[P + 15];
    uint act = params[P + 16];
    float c1 = as_type<float>(params[P + 17]);
    float c2 = as_type<float>(params[P + 18]);
    float c3 = as_type<float>(params[P + 19]);
    uint r_off = params[P + 20];

    uint b = wg.z;
    if (params[0] != 0u && b >= params[0]) return; // 批上限早退（同 n_conv8）

    uint nch = nDim / NCOLS;               // 通道块数（nDim 是 NCOLS 的倍数）
    uint nblk = wg.x % nch;                // 本线程组的通道块（变化最快）
    uint xblk = wg.x / nch;                // 本线程组的空间块

    uint kf = cin4v * 4u;                  // 每像素通道数（= K 的长度）

    // ── 覆盖几何（三份一致的第 2 份：plan.rs 的 `KxkCover::new`、
    //    n_conv8.metal 的 `conv8_body<EDGE>` 里那段）───────────────
    //
    // 内点 ⟺ 对任意 tap 读到的输入下标都在平面内：
    //   y*sH + ky - padT ∈ [0, inH)   ∀ky ∈ [0,kH)
    //   x*sW + kx - padL ∈ [0, inW)   ∀kx ∈ [0,kW)
    // 下界一侧是 `y*sH ≥ padT`，上界一侧是 `y*sH ≤ inH-1-(kH-1)+padT`；
    // 两边都是「乘 sH 后比较」，同除 sH 取整即得闭区间端点。
    // **stride1 + 对称 pad 时逐项退化成旧式**（`yd_lo = padT`、
    // `xb = padL + inner_w - 1`、`nfb = inner_w/64`、`i_lo = 64 - tail`），
    // 所以这次泛化对既有形状是逐位等价的，新放开的只有 stride≠1 与非对称
    // pad（det 的 Conv.1/2 是 2×2 SAME、Conv.3 是 3×3 stride2）。
    //
    // 行方向的**上界不在这里出现**：内点行数 `ny` 由网格的 y 维给出，
    // 本内核只认起点。（`kH = taps/kW` 也不读——它只参与上界的推导。）
    uint yd_lo = (padT + sH - 1u) / sH;                    // 内点行起点
    uint xa = (padL + sW - 1u) / sW;                       // 单点安全列下界
    uint xb = (inW - 1u - (kW - 1u) + padL) / sW;          // 单点安全列上界
    uint y = yd_lo + wg.y;                  // 本块的输出行（内点行）

    // 内点列 [xa, xb]：整块 64 列的起点还要求 `x0 ≤ xb-63`（**整块读**都在
    // 界内，不只是单点）。前 nfb 块从 `xa` 起步，最后一块（只在铺不满时
    // 存在）锚在 `xb-63` 上、写掩码裁到尾部。读重叠写在文件头「行尾余数」
    // 一节；写入始终无重叠。计划层的网格公式与这里逐字一致。
    uint xb_blk = xb - 63u;                 // ≥ xa（启用门保证 xb ≥ xa+63）
    uint nfb = (xb_blk - xa) / 64u + 1u;
    uint cover_end = xa + 64u * nfb - 1u;   // 整块已覆盖的最大输出 x
    uint x0, i_lo;
    if (xblk < nfb) {
        x0 = xa + xblk * 64u;
        i_lo = 0u;
    } else {                                // 尾块（xblk == nfb，仅在 cover_end < xb 时被派发）
        x0 = xb_blk;                        // ≥ xa，因为 nfb ≥ 1
        i_lo = (cover_end + 1u) - x0;       // 只写 [i_lo, 64)，且 ≤ 64
    }
    uint n0 = nblk * NCOLS;                // 本块的输出通道起点

    uint sgid = tid >> 5;                  // NSG 个 simdgroup / 线程组
    uint sr = (sgid >> 1) * RPSG;          // 本 simdgroup 的 RPSG 行（像素）基
    uint sn = (sgid & 1u) * (NCOLS / 2u);  // 本 simdgroup 的 NCOLS/2 列（通道）基

    uint ib = (in_off >> 2) * 4u + b * inH * inW * kf;   // 输入段 float 基
    uint wb = (w_off >> 2) * 4u;                         // 权重段 float 基
    uint ob = (out_off >> 2) * 4u + b * mDim * nDim;     // 输出平面恒 outH*outW 个像素

    simdgroup_float8x8 acc[NR][NB];
    for (uint i = 0u; i < NR; i++)
        for (uint j = 0u; j < NB; j++)
            acc[i][j] = simdgroup_float8x8(0.0f);

    // taps 外层、K 内层（步长 8）。A 的 64 行 = 本块 64 个输出 x 各自在
    // **同一行 (y*sH+ky-padT)** 上读到的那个像素；相邻输出 x 的输入相隔
    // `sW` 个像素 ⇒ A 的**行 stride = sW*kf**（stride1 时退化成 kf）。
    // `simdgroup_load` 接受任意行 stride，所以 stride≠1 不需要 gather。
    // 横向绕行（读到上一行/下一行的像素）在内点块上不可能发生——见文件头。
    for (uint t = 0u; t < taps; t++) {
        uint ky = t / kW;
        uint kx = t - ky * kW;
        uint abase = ib + ((y * sH + ky - padT) * inW + (x0 * sW + kx - padL)) * kf;
        uint wbase = wb + t * kf * nDim;   // [tap][k][n]，行 stride = nDim
        uint astep = sW * kf;              // A 的行 stride（输出 x 的一步）
        for (uint kb = 0u; kb < kf; kb += 8u) {
            simdgroup_float8x8 a[NR];
            for (uint i = 0u; i < NR; i++)
                simdgroup_load(a[i], f32v + abase + (sr + i * 8u) * astep + kb, astep);
            // **只装 NB 个分片**（不是恒装 4 个再挑着用）：NCOLS=32 时多装的那
            // 两个会越过本通道块、在最后一个块上直接越出权重段——那正是
            // 「未绑定/越界读 → 静默打瘸」那条路的形状。省下的也确实是 ALU。
            simdgroup_float8x8 b[NB];
            for (uint j = 0u; j < NB; j++)
                simdgroup_load(b[j], f32v + wbase + kb * nDim + n0 + sn + j * 8u, nDim);
            for (uint i = 0u; i < NR; i++)
                for (uint j = 0u; j < NB; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
    }

    // 尾声：分片落 threadgroup（NCOLS=64 -> 16 KB/组，32 -> 8 KB/组），再
    // NTH 线程逐元素做 bias/act/残差/写回。语义与 n_conv8 的尾声逐字段对齐。
    // 暂存区由入口声明（见上面那条 MSL 规则），尺寸 = 64*NCOLS。
    for (uint i = 0u; i < NR; i++)
        for (uint j = 0u; j < NB; j++)
            simdgroup_store(acc[i][j], ost + (sr + i * 8u) * NCOLS + sn + j * 8u, NCOLS);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tid; e < 64u * NCOLS; e += NTH) {
        uint i = e / NCOLS;     // 输出 x 偏移（本块内）
        if (i < i_lo) continue; // 尾块：与前一档重叠的那几列不写（见文件头）
        uint j = e % NCOLS;     // 输出通道偏移（本通道块内）
        uint n = n0 + j;
        float v = ost[e];
        if (b_off != OFF_NONE) v += f32v[(b_off >> 2) * 4u + n];
        v = act1(v, act, c1, c2, c3);
        uint m = y * outW + x0 + i;
        if (r_off != OFF_NONE)
            v += f32v[(r_off >> 2) * 4u + b * mDim * nDim + m * nDim + n];
        of[ob + m * nDim + n] = v;
    }
}

// 两条实例化。计划层按 `cpad4(Co)%64==0 ? 64 : 32` 选名（见 plan.rs）。
// 网格公式、参数块、覆盖矩形对两条完全一致——**只有通道块宽与线程组大小不同**。
kernel void n_conv8_mma3(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    threadgroup float ost[64u * 64u];
    static_assert(sizeof(ost) == 64u * 64u * sizeof(float), "暂存区尺寸");
    // 带属性的内核参数必须同为标量或同为同宽向量（MSL 硬约束），故 tid 走 uint3。
    // 8 个 simdgroup = **256 线程**（`pipeline.rs::threads_per_group` 的默认档）。
    mma3_body<64u, 8u>(d, params, p_off, tid3.x, wg, ost);
}

kernel void n_conv8_mma3_n32(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    // 8 KB/组（NCOLS=64 那支是 16 KB）——**N32 的收益一半来自这里**：
    // 通道块宽减半，暂存区跟着减半，每个核上能驻留的线程组数翻倍。
    threadgroup float ost[64u * 32u];
    static_assert(sizeof(ost) == 64u * 32u * sizeof(float), "暂存区尺寸");
    // **4 个 simdgroup = 128 线程**，不是 256：见文件头「装载/MMA 比值」一节。
    mma3_body<32u, 4u>(d, params, p_off, tid3.x, wg, ost);
}
