// 稠密 kxk conv（k>1，NHWC f32，stride 1、对称 pad）——simdgroup_matrix MMA 版。
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
//   输入下标 = (y+ky-padT)*inW + (x+dx-padL)
//            = [常量] + i          （i = x - x0 ∈ [0,64)）
// 是 i 的**一次函数**，A 瓦片的 64 行正好是平面上连续的 64 个像素
// （行 stride = kf = 每像素通道数）——一条 `simdgroup_load` 就够，零 gather。
// 权重侧零重排的理由同 `n_conv8_mma`（taps 布局 [tap][k][n] 就是 K×N 行主序）。
//
// # 只收「全内点块」——越界这一条是硬约束，不是优化
//
// 计划层只把**满足下式**的块派给本内核（内点行 × 内点列）：
//
//   y ∈ [padT, outH-1-(kH-1-padT)]
//   x ∈ [padL, padL + inner_w)，inner_w = outW - (kW-1)
//
// 于是对任意 tap，读到的输入下标都落在 [0, inH*inW) 内——**由数学保证**，
// 不需要任何对齐/担保，也绝不触碰输入平面之外（`n_conv8_mma` 那条「尾块读
// 到 arena 里别的段」的推理在这里不适用也不需要）。真正的边框（两行整行 +
// 两列整列）由 `n_conv8_edge` 补，两者写的是**互不相交**的输出像素。
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
//   groups = [nx * nch, outH-(taps/kW - 1), batch]，线程组恒 256 线程
//   （8 个 simdgroup），nch = nDim/64，**nx = nfb + (inner_w % 64 != 0)**
//   （余数为 0 时没有尾块）。**通道块在 wg.x 里变化最快**（`wg.x % nch`）
//   ——相邻线程组取同一片 A（A 是 DRAM 大头），权重小得多且本就 L2 常驻。
//
//   `outH` 由 `mDim/outW` 导出、`kH` 由 `taps/kW` 导出（参数块里没有这两
//   个数，见下一节）；**计划层的网格公式必须与本内核逐字一致**，否则覆盖
//   矩形错位——见文件头末段。
//
// 线程组内：8 个 simdgroup 排成 4×2，sgid 认领 16 行（像素）× NCOLS/2 列
// （通道）——NCOLS=64 时与 `n_conv8_mma` 逐字相同；NCOLS=32 时列方向窄一半，
// 行方向不变，于是**覆的还是同一块 64 行像素**（覆盖矩形与 NCOLS 无关，
// 这正是两条能共用尾块/边框推理的原因）。
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
// 就允许把 N 开窄——把 64 列块换成 32 列块，线程组内 8 个 simdgroup 排成
// 「4 行组 × 2 列组、每组 16 行 × 16 列」，A 的复用从 4 个分片降到 2 个，
// 但**每个分片仍是 8x8 的 MMA**，ALU 照样喂满。
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

// NCOLS = 输出通道块宽（64 或 32）。文件头「32 通道块」一节说明为什么要有
// 32 这一档。除本行的 NCOLS 与由它派生的三处（nch / sn / 分片数 + 尾声步长）
// 外，两条实例化的推理**逐字共用**——瓦片映射、覆盖矩形、尾块掩码、
// `outH = mDim/outW` 的导出全在下面这一段里，改一处两条同时改，不会漂。
// `ost`（分片落地的 threadgroup 暂存）**必须由内核入口声明、以指针传进来**：
// MSL 硬规则——`threadgroup` 变量只能在 `kernel` 函数里声明（"variables in
// the threadgroup address space cannot be declared in a non-qualified
// function"）。于是各入口声明自己那份 `float[64*NCOLS]`，尺寸耦合由两处的
// static_assert 守住。
template <uint NCOLS>
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

    device const float* f32v = (device const float*)d;
    device float* of = (device float*)d;

    // 与 `n_conv8` 同一套参数字（见文件头）；sH/sW（P+12/P+13）不读——
    // 启用门保证它们是 1。
    uint in_off = params[P + 0], w_off = params[P + 1], b_off = params[P + 2];
    uint out_off = params[P + 3];
    uint mDim = params[P + 4], nDim = params[P + 5], outW = params[P + 6];
    uint inW = params[P + 7], inH = params[P + 8], cin4v = params[P + 9];
    uint taps = params[P + 10], kW = params[P + 11];
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
    uint y = wg.y + padT;                  // 本块的输出行（内点行）

    // 内点列 [padL, padL+inner_w)：前 nfb 块 64 列起步，最后一块（只在
    // inner_w%64 != 0 时存在）锚在**最后 64 个内点列**上、写掩码裁到尾部。
    // 读重叠写在文件头「行尾余数」一节；写入始终无重叠。
    uint inner_w = outW - (kW - 1u);
    uint nfb = inner_w / 64u;
    uint tail = inner_w - nfb * 64u;        // 行尾余数（0 或 1..63）
    uint x0, i_lo;
    if (xblk < nfb) {
        x0 = padL + xblk * 64u;
        i_lo = 0u;
    } else {                                // 尾块（xblk == nfb，仅在 tail>0 时被派发）
        x0 = padL + inner_w - 64u;          // ≥ padL，因为 nfb ≥ 1
        i_lo = 64u - tail;                  // 只写 [i_lo, 64)
    }
    uint n0 = nblk * NCOLS;                // 本块的输出通道起点

    uint sgid = tid >> 5;                  // 8 个 simdgroup / 线程组
    uint sr = (sgid >> 1) * 16u;           // 本 simdgroup 的 16 行（像素）基
    uint sn = (sgid & 1u) * (NCOLS / 2u);  // 本 simdgroup 的 NCOLS/2 列（通道）基

    uint ib = (in_off >> 2) * 4u + b * inH * inW * kf;   // 输入段 float 基
    uint wb = (w_off >> 2) * 4u;                         // 权重段 float 基
    uint ob = (out_off >> 2) * 4u + b * mDim * nDim;     // 输出平面恒 outH*outW 个像素

    simdgroup_float8x8 acc[2][NB];
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < NB; j++)
            acc[i][j] = simdgroup_float8x8(0.0f);

    // taps 外层、K 内层（步长 8）。A 的 64 行 = 平面上从 abase 起的 64 个
    // 连续像素（行 stride = kf）：横向绕行在内点块上不可能发生，见文件头。
    for (uint t = 0u; t < taps; t++) {
        uint ky = t / kW;
        uint kx = t - ky * kW;
        uint abase = ib + ((y + ky - padT) * inW + (x0 + kx - padL)) * kf;
        uint wbase = wb + t * kf * nDim;   // [tap][k][n]，行 stride = nDim
        for (uint kb = 0u; kb < kf; kb += 8u) {
            simdgroup_float8x8 a0, a1;
            simdgroup_load(a0, f32v + abase + sr * kf + kb, kf);
            simdgroup_load(a1, f32v + abase + (sr + 8u) * kf + kb, kf);
            // **只装 NB 个分片**（不是恒装 4 个再挑着用）：NCOLS=32 时多装的那
            // 两个会越过本通道块、在最后一个块上直接越出权重段——那正是
            // 「未绑定/越界读 → 静默打瘸」那条路的形状。省下的也确实是 ALU。
            simdgroup_float8x8 b[NB];
            for (uint j = 0u; j < NB; j++)
                simdgroup_load(b[j], f32v + wbase + kb * nDim + n0 + sn + j * 8u, nDim);
            for (uint j = 0u; j < NB; j++) {
                simdgroup_multiply_accumulate(acc[0][j], a0, b[j], acc[0][j]);
                simdgroup_multiply_accumulate(acc[1][j], a1, b[j], acc[1][j]);
            }
        }
    }

    // 尾声：分片落 threadgroup（NCOLS=64 -> 16 KB/组，32 -> 8 KB/组），再
    // 256 线程逐元素做 bias/act/残差/写回。语义与 n_conv8 的尾声逐字段对齐。
    // 暂存区由入口声明（见上面那条 MSL 规则），尺寸 = 64*NCOLS。
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < NB; j++)
            simdgroup_store(acc[i][j], ost + (sr + i * 8u) * NCOLS + sn + j * 8u, NCOLS);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tid; e < 64u * NCOLS; e += 256u) {
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
// 网格公式、参数块、覆盖矩形对两条完全一致——**只有通道块宽不同**。
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
    mma3_body<64u>(d, params, p_off, tid3.x, wg, ost);
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
    mma3_body<32u>(d, params, p_off, tid3.x, wg, ost);
}
