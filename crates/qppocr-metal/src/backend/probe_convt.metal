// n_convt 的**变体二分**（仅测试；参数块语义与 shaders/n_convt.metal 逐字一致）。
//
// ## 为什么单开一份（2026-10-10）
//
// n_convt 在 det 里 2 条 = 10.1 ms（设备端 90 ms 的 11.3%），而它**真实**只做了
// 2.12 GFLOP（见下），也就是 210 GFLOPS —— 峰值 6.3%。三条下界：
//
//   · DRAM 下界   读输入 13.3 MB + 读权重 64 KB + 写输出 53.1 MB = 66.4 MB
//                 ÷ 77 GB/s（本机实测带宽）= **0.86 ms**
//   · FMA 下界    849 M MAC ÷ 1670 GMAC/s（本机实测峰值）= **0.51 ms**
//   · 实测        ConvTranspose.0 = **8.20 ms**
//
// 差 9.5×（对带宽）/ 16×（对算力）。既不是喂不上也不是算不动，那就只可能是
// **装载指令条数**：热循环每 ci 迭代是 2 条 float4 装载换 4 条 FMA（比值 2.0），
// 而且输入那条是 `d[...][ci & 3u]` —— **动态 float4 下标**，且同一地址每 4 个
// ci 重复一次（4× 冗余装载）。
//
// ## FLOPs 修正（先记下来，免得再被 flopsmap 骗一次）
//
// `bench/gpuhealth/flopsmap.py` 把 n_convt 记成 6.80 GFLOP，是**按 taps 记 MAC**；
// 但 kH≤sH 且 kW≤sW 时每个输出**只吃一个 tap**，真实 MAC = outH×outW×Co×Ci，
// 正好被放大 4×。真实值：ConvTranspose.0 = 1.699 GFLOP、ConvTranspose.2 = 0.425，
// 合计 2.124 GFLOP / 10.1 ms。
//
// ## 变体表
//
// | 名字 | 与 n_convt 的差 | 结果正确？ | 要回答的问题 |
// |---|---|---|---|
// | `p_ct_base` | 无（逐行抄） | 是 | 量法本身可信吗（须与生产同时间） |
// | `p_ct_ci4` | ci 按 4 展开、用 float4 四个 lane（消掉动态下标与 4× 冗余） | 是（位级同） | 那两条浪费值多少 |
// | `p_ct_noin` | 输入地址 → 常量 | 否（诊断） | 输入流占多少 |
// | `p_ct_now` | 权重地址 → 常量 | 否（诊断） | 权重流占多少 |
// | `p_ct_fmaonly` | 输入与权重全换常量，只留 FMA | 否（诊断） | 卡在「算不动」还是「喂不上」 |
// | `p_ct_ismall` | 输入地址掩进 1 KB | 否（诊断） | 输入卡在**字节**还是**装载条数** |
// | `p_ct_wsmall` | 权重地址掩进 1 KB | 否（诊断） | 权重卡在**字节**还是**装载条数** |
// | `p_ct_osmall` | **输出**地址掩进 1 KB | 否（诊断） | 53 MB 写回占多少（尾流三板斧的第三条） |
// | `p_ct_tap4` | 一线程吃掉 4 个 tap（同一 (ih,iw) 的 2×2 输出） | 是 | 空间分块能否把输入装载摊到 4 个 tap 上 |
// | `p_ct_tap4w4` | 同上 + 4 个相邻 iw（即 8 个 float4 输出/线程） | 是 | 权重能否跨空间位置复用 |
//
// `p_ct_ismall`/`p_ct_wsmall`/`p_ct_osmall` 是这组里最要紧的三条：如果掩掉字节
// 而时间基本不动，那杠杆就是**装载条数**（只能靠寄存器分块），不是**带宽**
// （靠暂存/精度档位）。这正是 [[activation-traffic-is-not-the-constraint]] 那条
// 结论的复核——那次管的是 n_conv8，这次是 n_convt，**形状完全不同**（K 只有
// 64、每输出一个 tap、N=64/4），不能默认沿用。
//
// ## 网格
//
// `p_ct_base` 家族 = `div256(mDim * nv)`（与 plan.rs 逐字一致）；
// `p_ct_tap4` = `div256(inH*inW*nv)`；`p_ct_tap4w4` = `div256(inH*inW/4*nv)`。
// 调用方必须按变体选网格（见 conv_probe.rs 的 `ct_groups_for`），否则覆盖不全。
//
// ## 纪律
//
// 本文件**不在** `KernelSet::SHADERS` 里，只在本模块编译、只被测试调用，
// 不进任何生产计划。探针胜势**不可信**、否证**可信**——落地前必须用真计划复核
// （[[probe-wins-dont-transfer-to-production]]）。
#include <metal_stdlib>
using namespace metal;

constant uint OFF_NONE = 0xFFFFFFFFu;

// exp1/erf1：与 shaders/n_convt.metal 逐字相同的拷贝（act=1 的 gelu 分支用）。
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

// 尾声：逐字等于 n_convt 的那三行。act=0 时是空操作（det 的 FPN 就是 0）。
inline float4 ct_epilogue(float4 acc, uint act, float c1, float c2, float c3) {
    if (act == 2u) {
        acc = max(acc, float4(0.0));
    } else if (act == 1u) {
        float inv = 1.0 / c1;
        acc = c3 * acc * (float4(erf1(acc.x * inv), erf1(acc.y * inv),
                                 erf1(acc.z * inv), erf1(acc.w * inv)) + c2);
    }
    return acc;
}

// 参数头（与 n_convt 逐字段同义）。
struct CtParams {
    uint in_off, w_off, b_off, out_off;
    uint mDim, nDim, outW, inW, inH, cin, ci4v, sH, sW, kW, act;
    float c1, c2, c3;
};

inline CtParams ct_load(device const uint* params, constant uint& p_off) {
    uint P = p_off;
    CtParams p;
    p.in_off = params[P + 0];   p.w_off = params[P + 1];
    p.b_off  = params[P + 2];   p.out_off = params[P + 3];
    p.mDim   = params[P + 4];   p.nDim = params[P + 5];
    p.outW   = params[P + 6];   p.inW = params[P + 7];
    p.inH    = params[P + 8];   p.cin = params[P + 9];
    p.ci4v   = params[P + 10];  p.sH = params[P + 11];
    p.sW     = params[P + 12];  p.kW = params[P + 13];
    p.act    = params[P + 14];
    p.c1 = as_type<float>(params[P + 15]);
    p.c2 = as_type<float>(params[P + 16]);
    p.c3 = as_type<float>(params[P + 17]);
    return p;
}

// 诊断用的「常量」输入/权重：由循环变量算出，编译器消不掉。
inline float4 ct_fake_x(uint i) {
    return float4(1.0f + float(i & 3u), 0.5f, 0.25f, 0.125f);
}
inline float4 ct_fake_w(uint i) {
    return float4(0.25f, 1.0f + float(i & 7u), 0.5f, 2.0f);
}

// 地址掩码：条数不变、字节≈0（掩进 1 KB = 64 个 float4）。
inline uint ct_mask(uint addr) { return addr & (63u); }

// ---------------------------------------------------------------- 控制组

kernel void p_ct_base(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    CtParams p = ct_load(params, p_off);
    uint nv = p.nDim >> 2;
    uint gid = gid3.x;
    uint tiles = p.mDim * nv;
    if (gid >= tiles) return;
    uint n4 = gid % nv;
    uint m = gid / nv;
    uint oh = m / p.outW, ow = m - oh * p.outW;
    uint ih = oh / p.sH, ky = oh - ih * p.sH;
    uint iw = ow / p.sW, kx = ow - iw * p.sW;
    uint tap = ky * p.kW + kx;
    uint ib = (p.in_off >> 2) + wg.y * p.inH * p.inW * p.ci4v;
    uint wb = p.w_off >> 2;
    uint ob = (p.out_off >> 2) + wg.y * p.mDim * nv;

    uint px = ih * p.inW + iw;
    float4 acc = float4(0.0);
    for (uint ci = 0u; ci < p.cin; ci++) {
        float xv = d[ib + px * p.ci4v + (ci >> 2)][ci & 3u];
        acc += xv * d[wb + tap * p.cin * nv + ci * nv + n4];
    }
    if (p.b_off != OFF_NONE) acc += d[(p.b_off >> 2) + n4];
    acc = ct_epilogue(acc, p.act, p.c1, p.c2, p.c3);
    d[ob + m * nv + n4] = acc;
}

// 控制组 + ci 按 4 展开：一次读整个 float4，四条 lane 静态取用。
// 消掉两样东西：**动态 float4 下标**、**4× 冗余的输入装载**。数学与 p_ct_base
// 的累加顺序逐位相同（acc + x.w0 + y.w1 + z.w2 + w.w3 与标量逐 ci 相加同序）。
kernel void p_ct_ci4(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    CtParams p = ct_load(params, p_off);
    uint nv = p.nDim >> 2;
    uint gid = gid3.x;
    if (gid >= p.mDim * nv) return;
    uint n4 = gid % nv;
    uint m = gid / nv;
    uint oh = m / p.outW, ow = m - oh * p.outW;
    uint ih = oh / p.sH, ky = oh - ih * p.sH;
    uint iw = ow / p.sW, kx = ow - iw * p.sW;
    uint tap = ky * p.kW + kx;
    uint ib = (p.in_off >> 2) + wg.y * p.inH * p.inW * p.ci4v;
    uint wb = (p.w_off >> 2) + tap * p.cin * nv + n4;
    uint ob = (p.out_off >> 2) + wg.y * p.mDim * nv;

    uint px = ih * p.inW + iw;
    float4 acc = float4(0.0);
    for (uint ci4 = 0u; ci4 < p.ci4v; ci4++) {
        float4 x = d[ib + px * p.ci4v + ci4];
        float4 w = d[wb + (ci4 << 2) * nv];
        acc = fma(x.x, w, acc);
        w = d[wb + ((ci4 << 2) + 1u) * nv];
        acc = fma(x.y, w, acc);
        w = d[wb + ((ci4 << 2) + 2u) * nv];
        acc = fma(x.z, w, acc);
        w = d[wb + ((ci4 << 2) + 3u) * nv];
        acc = fma(x.w, w, acc);
    }
    if (p.b_off != OFF_NONE) acc += d[(p.b_off >> 2) + n4];
    acc = ct_epilogue(acc, p.act, p.c1, p.c2, p.c3);
    d[ob + m * nv + n4] = acc;
}

// ---------------------------------------------------------------- 诊断（结果故意错）

// 输入地址钉死（同组 lane 读同一地址 → 广播）。看输入流占多少。
kernel void p_ct_noin(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    CtParams p = ct_load(params, p_off);
    uint nv = p.nDim >> 2;
    uint gid = gid3.x;
    if (gid >= p.mDim * nv) return;
    uint n4 = gid % nv;
    uint m = gid / nv;
    uint oh = m / p.outW, ow = m - oh * p.outW;
    uint ih = oh / p.sH, ky = oh - ih * p.sH;
    uint iw = ow / p.sW, kx = ow - iw * p.sW;
    uint tap = ky * p.kW + kx;
    uint ib = (p.in_off >> 2) + wg.y * p.inH * p.inW * p.ci4v;
    uint wb = p.w_off >> 2;
    uint ob = (p.out_off >> 2) + wg.y * p.mDim * nv;
    float4 acc = float4(0.0);
    for (uint ci = 0u; ci < p.cin; ci++) {
        float xv = d[ib + (ci >> 2)][ci & 3u];   // ← 只剩这一处差异：px 去掉
        acc += xv * d[wb + tap * p.cin * nv + ci * nv + n4];
    }
    if (p.b_off != OFF_NONE) acc += d[(p.b_off >> 2) + n4];
    acc = ct_epilogue(acc, p.act, p.c1, p.c2, p.c3);
    d[ob + m * nv + n4] = acc;
}

// 权重装载换常量。
kernel void p_ct_now(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    CtParams p = ct_load(params, p_off);
    uint nv = p.nDim >> 2;
    uint gid = gid3.x;
    if (gid >= p.mDim * nv) return;
    uint n4 = gid % nv;
    uint m = gid / nv;
    uint oh = m / p.outW, ow = m - oh * p.outW;
    uint ih = oh / p.sH, ky = oh - ih * p.sH;
    uint iw = ow / p.sW, kx = ow - iw * p.sW;
    uint ib = (p.in_off >> 2) + wg.y * p.inH * p.inW * p.ci4v;
    uint ob = (p.out_off >> 2) + wg.y * p.mDim * nv;
    uint px = ih * p.inW + iw;
    float4 acc = float4(0.0);
    for (uint ci = 0u; ci < p.cin; ci++) {
        float xv = d[ib + px * p.ci4v + (ci >> 2)][ci & 3u];
        acc += xv * ct_fake_w(ci + n4);
    }
    if (p.b_off != OFF_NONE) acc += d[(p.b_off >> 2) + n4];
    acc = ct_epilogue(acc, p.act, p.c1, p.c2, p.c3);
    d[ob + m * nv + n4] = acc;
}

// 访存全砍，只留 FMA（累加器做成数组，防止编译器把整条循环消掉）。
kernel void p_ct_fmaonly(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    CtParams p = ct_load(params, p_off);
    uint nv = p.nDim >> 2;
    uint gid = gid3.x;
    if (gid >= p.mDim * nv) return;
    uint n4 = gid % nv;
    uint m = gid / nv;
    uint ob = (p.out_off >> 2) + wg.y * p.mDim * nv;
    float4 acc = float4(0.0);
    for (uint ci = 0u; ci < p.cin; ci++) {
        acc += ct_fake_x(ci) * ct_fake_w(ci + n4);
    }
    if (p.b_off != OFF_NONE) acc += d[(p.b_off >> 2) + n4];
    acc = ct_epilogue(acc, p.act, p.c1, p.c2, p.c3);
    d[ob + m * nv + n4] = acc;
}

// 输入地址掩进 1 KB：装载条数/地址算术全同，只有字节塌掉。
kernel void p_ct_ismall(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    CtParams p = ct_load(params, p_off);
    uint nv = p.nDim >> 2;
    uint gid = gid3.x;
    if (gid >= p.mDim * nv) return;
    uint n4 = gid % nv;
    uint m = gid / nv;
    uint oh = m / p.outW, ow = m - oh * p.outW;
    uint ih = oh / p.sH, ky = oh - ih * p.sH;
    uint iw = ow / p.sW, kx = ow - iw * p.sW;
    uint tap = ky * p.kW + kx;
    uint ib = (p.in_off >> 2) + wg.y * p.inH * p.inW * p.ci4v;
    uint wb = p.w_off >> 2;
    uint ob = (p.out_off >> 2) + wg.y * p.mDim * nv;
    uint px = ih * p.inW + iw;
    float4 acc = float4(0.0);
    for (uint ci = 0u; ci < p.cin; ci++) {
        float xv = d[ib + ct_mask(px * p.ci4v + (ci >> 2))][ci & 3u];
        acc += xv * d[wb + tap * p.cin * nv + ci * nv + n4];
    }
    if (p.b_off != OFF_NONE) acc += d[(p.b_off >> 2) + n4];
    acc = ct_epilogue(acc, p.act, p.c1, p.c2, p.c3);
    d[ob + m * nv + n4] = acc;
}

// 权重地址掩进 1 KB。
kernel void p_ct_wsmall(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    CtParams p = ct_load(params, p_off);
    uint nv = p.nDim >> 2;
    uint gid = gid3.x;
    if (gid >= p.mDim * nv) return;
    uint n4 = gid % nv;
    uint m = gid / nv;
    uint oh = m / p.outW, ow = m - oh * p.outW;
    uint ih = oh / p.sH, ky = oh - ih * p.sH;
    uint iw = ow / p.sW, kx = ow - iw * p.sW;
    uint tap = ky * p.kW + kx;
    uint ib = (p.in_off >> 2) + wg.y * p.inH * p.inW * p.ci4v;
    uint wb = (p.w_off >> 2) + tap * p.cin * nv + n4;
    uint ob = (p.out_off >> 2) + wg.y * p.mDim * nv;
    uint px = ih * p.inW + iw;
    float4 acc = float4(0.0);
    for (uint ci = 0u; ci < p.cin; ci++) {
        float xv = d[ib + px * p.ci4v + (ci >> 2)][ci & 3u];
        acc += xv * d[wb + ct_mask(ci * nv)];
    }
    if (p.b_off != OFF_NONE) acc += d[(p.b_off >> 2) + n4];
    acc = ct_epilogue(acc, p.act, p.c1, p.c2, p.c3);
    d[ob + m * nv + n4] = acc;
}

// 输出地址掩进 1 KB（写 53 MB 那条流的对照）。
kernel void p_ct_osmall(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    CtParams p = ct_load(params, p_off);
    uint nv = p.nDim >> 2;
    uint gid = gid3.x;
    if (gid >= p.mDim * nv) return;
    uint n4 = gid % nv;
    uint m = gid / nv;
    uint oh = m / p.outW, ow = m - oh * p.outW;
    uint ih = oh / p.sH, ky = oh - ih * p.sH;
    uint iw = ow / p.sW, kx = ow - iw * p.sW;
    uint tap = ky * p.kW + kx;
    uint ib = (p.in_off >> 2) + wg.y * p.inH * p.inW * p.ci4v;
    uint wb = p.w_off >> 2;
    uint ob = (p.out_off >> 2) + wg.y * p.mDim * nv;
    uint px = ih * p.inW + iw;
    float4 acc = float4(0.0);
    for (uint ci = 0u; ci < p.cin; ci++) {
        float xv = d[ib + px * p.ci4v + (ci >> 2)][ci & 3u];
        acc += xv * d[wb + tap * p.cin * nv + ci * nv + n4];
    }
    if (p.b_off != OFF_NONE) acc += d[(p.b_off >> 2) + n4];
    acc = ct_epilogue(acc, p.act, p.c1, p.c2, p.c3);
    d[ob + ct_mask(m * nv + n4)] = acc;
}

// ---------------------------------------------------------------- 可落地候选

// 4 个 tap / 线程：同一个 (ih,iw) 的 2×2 输出块。**输入装载被 4 个 tap 共用**
// （1 读 4 用），且输出写回天然成对（同一行两个相邻 float4）。
// 网格 = div256(inH*inW*nv)，即 p_ct_base 的 1/(sH*sW)。
// 只对 (sH,sW)==(2,2) 且 kW==2 成立（4 个 tap 的枚举方式写死）。
kernel void p_ct_tap4(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    CtParams p = ct_load(params, p_off);
    uint nv = p.nDim >> 2;
    uint gid = gid3.x;
    uint nIn = p.inH * p.inW;
    if (gid >= nIn * nv) return;
    uint n4 = gid % nv;
    uint px = gid / nv;
    uint ih = px / p.inW, iw = px - ih * p.inW;
    uint ib = (p.in_off >> 2) + wg.y * nIn * p.ci4v;
    uint wb = p.w_off >> 2;
    uint ob = (p.out_off >> 2) + wg.y * p.mDim * nv;

    // tap = ky*kW + kx，(ky,kx) ∈ {0,1}²
    uint wt0 = wb + 0u * p.cin * nv + n4;
    uint wt1 = wb + 1u * p.cin * nv + n4;
    uint wt2 = wb + 2u * p.cin * nv + n4;
    uint wt3 = wb + 3u * p.cin * nv + n4;

    float4 a0 = float4(0.0), a1 = float4(0.0), a2 = float4(0.0), a3 = float4(0.0);
    for (uint ci4 = 0u; ci4 < p.ci4v; ci4++) {
        float4 x = d[ib + px * p.ci4v + ci4];      // 1 条输入装载，4 个 tap 共用
        uint c0 = (ci4 << 2) * nv;
        float4 w;
        w = d[wt0 + c0 + 0u * nv]; a0 = fma(x.x, w, a0);
        w = d[wt0 + c0 + 1u * nv]; a0 = fma(x.y, w, a0);
        w = d[wt0 + c0 + 2u * nv]; a0 = fma(x.z, w, a0);
        w = d[wt0 + c0 + 3u * nv]; a0 = fma(x.w, w, a0);
        w = d[wt1 + c0 + 0u * nv]; a1 = fma(x.x, w, a1);
        w = d[wt1 + c0 + 1u * nv]; a1 = fma(x.y, w, a1);
        w = d[wt1 + c0 + 2u * nv]; a1 = fma(x.z, w, a1);
        w = d[wt1 + c0 + 3u * nv]; a1 = fma(x.w, w, a1);
        w = d[wt2 + c0 + 0u * nv]; a2 = fma(x.x, w, a2);
        w = d[wt2 + c0 + 1u * nv]; a2 = fma(x.y, w, a2);
        w = d[wt2 + c0 + 2u * nv]; a2 = fma(x.z, w, a2);
        w = d[wt2 + c0 + 3u * nv]; a2 = fma(x.w, w, a2);
        w = d[wt3 + c0 + 0u * nv]; a3 = fma(x.x, w, a3);
        w = d[wt3 + c0 + 1u * nv]; a3 = fma(x.y, w, a3);
        w = d[wt3 + c0 + 2u * nv]; a3 = fma(x.z, w, a3);
        w = d[wt3 + c0 + 3u * nv]; a3 = fma(x.w, w, a3);
    }
    float4 bv = float4(0.0);
    if (p.b_off != OFF_NONE) bv = d[(p.b_off >> 2) + n4];
    a0 = ct_epilogue(a0 + bv, p.act, p.c1, p.c2, p.c3);
    a1 = ct_epilogue(a1 + bv, p.act, p.c1, p.c2, p.c3);
    a2 = ct_epilogue(a2 + bv, p.act, p.c1, p.c2, p.c3);
    a3 = ct_epilogue(a3 + bv, p.act, p.c1, p.c2, p.c3);
    // tap0=(0,0) tap1=(0,1) tap2=(1,0) tap3=(1,1) ⇒ oh = 2ih+ky, ow = 2iw+kx
    uint oh = ih * 2u, ow = iw * 2u;
    uint o00 = oh * p.outW + ow;
    d[ob + (o00 + 0u) * nv + n4] = a0;
    d[ob + (o00 + 1u) * nv + n4] = a1;
    d[ob + ((oh + 1u) * p.outW + ow) * nv + n4] = a2;
    d[ob + ((oh + 1u) * p.outW + ow + 1u) * nv + n4] = a3;
}

// 4 个 tap × 4 个相邻 iw / 线程（8 个 float4 输出）。加上空间方向的复用后，
// 同一条权重装载能喂 4 个位置 ⇒ 装载/ FMA 从 17/16 降到 20/64。
// 网格 = div256(inH*(inW/4)*nv)。要求 inW % 4 == 0。
kernel void p_ct_tap4w4(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    CtParams p = ct_load(params, p_off);
    uint nv = p.nDim >> 2;
    uint gid = gid3.x;
    uint inW4 = p.inW >> 2;
    uint nIn = p.inH * inW4;
    if (gid >= nIn * nv) return;
    uint n4 = gid % nv;
    uint q = gid / nv;
    uint ih = q / inW4, iwq = (q - ih * inW4) << 2;   // iwq = 4*组号
    uint ib = (p.in_off >> 2) + wg.y * p.inH * p.inW * p.ci4v;
    uint wb = p.w_off >> 2;
    uint ob = (p.out_off >> 2) + wg.y * p.mDim * nv;

    uint wt0 = wb + 0u * p.cin * nv + n4;
    uint wt1 = wb + 1u * p.cin * nv + n4;
    uint wt2 = wb + 2u * p.cin * nv + n4;
    uint wt3 = wb + 3u * p.cin * nv + n4;

    // acc[tap][j]，j = 4 个相邻 iw
    float4 a0[4], a1[4], a2[4], a3[4];
    for (uint j = 0u; j < 4u; j++) {
        a0[j] = float4(0.0); a1[j] = float4(0.0);
        a2[j] = float4(0.0); a3[j] = float4(0.0);
    }
    uint xb = ib + (ih * p.inW + iwq) * p.ci4v;
    for (uint ci4 = 0u; ci4 < p.ci4v; ci4++) {
        float4 x0 = d[xb + 0u * p.ci4v + ci4];
        float4 x1 = d[xb + 1u * p.ci4v + ci4];
        float4 x2 = d[xb + 2u * p.ci4v + ci4];
        float4 x3 = d[xb + 3u * p.ci4v + ci4];
        uint c0 = (ci4 << 2) * nv;
        uint c1 = c0 + nv, c2 = c0 + 2u * nv, c3 = c0 + 3u * nv;
#define CT_WSTEP(WBASE, ACC)                                              \
        {                                                                 \
            float4 w;                                                     \
            w = d[WBASE + c0]; ACC[0] = fma(x0.x, w, ACC[0]);             \
            ACC[1] = fma(x1.x, w, ACC[1]);                                \
            ACC[2] = fma(x2.x, w, ACC[2]);                                \
            ACC[3] = fma(x3.x, w, ACC[3]);                                \
            w = d[WBASE + c1]; ACC[0] = fma(x0.y, w, ACC[0]);             \
            ACC[1] = fma(x1.y, w, ACC[1]);                                \
            ACC[2] = fma(x2.y, w, ACC[2]);                                \
            ACC[3] = fma(x3.y, w, ACC[3]);                                \
            w = d[WBASE + c2]; ACC[0] = fma(x0.z, w, ACC[0]);             \
            ACC[1] = fma(x1.z, w, ACC[1]);                                \
            ACC[2] = fma(x2.z, w, ACC[2]);                                \
            ACC[3] = fma(x3.z, w, ACC[3]);                                \
            w = d[WBASE + c3]; ACC[0] = fma(x0.w, w, ACC[0]);             \
            ACC[1] = fma(x1.w, w, ACC[1]);                                \
            ACC[2] = fma(x2.w, w, ACC[2]);                                \
            ACC[3] = fma(x3.w, w, ACC[3]);                                \
        }
        CT_WSTEP(wt0, a0)
        CT_WSTEP(wt1, a1)
        CT_WSTEP(wt2, a2)
        CT_WSTEP(wt3, a3)
#undef CT_WSTEP
    }
    float4 bv = float4(0.0);
    if (p.b_off != OFF_NONE) bv = d[(p.b_off >> 2) + n4];
    uint oh = ih * 2u;
    // 相邻输出列在 arena 里差 **nv 个 float4**（NHWC 的通道维），不是 1 —— 只有
    // nv==1 时两者巧合相等，所以 nv=16 的形状上写 `(j<<1)` 会串位。
    uint cstep = nv << 1;
    uint row0 = (oh * p.outW + (iwq << 1)) * nv + n4;
    uint row1 = ((oh + 1u) * p.outW + (iwq << 1)) * nv + n4;
    for (uint j = 0u; j < 4u; j++) {
        uint co = j * cstep;
        d[ob + row0 + co] = ct_epilogue(a0[j] + bv, p.act, p.c1, p.c2, p.c3);
        d[ob + row0 + co + nv] = ct_epilogue(a1[j] + bv, p.act, p.c1, p.c2, p.c3);
        d[ob + row1 + co] = ct_epilogue(a2[j] + bv, p.act, p.c1, p.c2, p.c3);
        d[ob + row1 + co + nv] = ct_epilogue(a3[j] + bv, p.act, p.c1, p.c2, p.c3);
    }
}
