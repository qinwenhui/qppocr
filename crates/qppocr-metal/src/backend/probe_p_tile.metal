// 瓦片空间探针：把 `n_conv8` 的寄存器分块从写死的「4 行 × 2 列（2 个
// float4 通道块）」参数化成 ROWS × NBLK，用来量**权重复用度**的边际收益。
//
// ## 动机（同轮变体扫描的直接结论）
//
// CTC 分类层（`MatMul.12`，折批后 216×1 c120→18712 k1）占整个 rec 计划的
// 27–41%，扫描结果是：
//
//     p_base 2.925 ms   p_wconst 0.496 ms   p_fmaonly 0.354 ms
//     p_nocoal 2.924 ms（输入访存钉死——和 p_base 一样快）
//
// 即：**83% 的时间在等权重流，输入装载与算力都不是瓶颈**。而权重流量
// ∝ 1/ROWS —— 每个权重 float4 被多少行输出复用，就直接决定权重读多少遍
// （本层权重 9 MB，被 54 个行瓦片各读一遍 = 485 MB）。所以 ROWS 是主
// 变量；NBLK 是次变量，它摊薄的是**输入**装载（输入只有 104 KB，本来
// 就在 L1 里，收益小得多）。
//
// 同一个扫描里 3×3 的形状恰好相反（p_r8/p_m8n4 比 p_base 慢一倍）——
// 所以这组瓦片**只能按形状启用**，不能全局换掉 `n_conv8` 的几何。
//
// ## 网格仍用计划的 m4n8 公式
//
// 这是关键设计：本文件的内核**不要求换网格**。它自己算本瓦片的线程数
// `tiles`，并先按计划的 `tiles_plan` 兜一道，再对 `gid >= tiles` 早退——
// 于是量到的是「生产里换瓦片」的真实收益，而不是「顺便换个网格」的收益。
// （候选瓦片的线程数必须 ≤ 计划给的线程数，否则覆盖不全；`tiles` 与
// `tiles_plan` 谁大谁小的判据写在 `tile_body` 里。）
//
// ## 为什么用模板而不是逐行抄
//
// 行数 4/8/16 的差别只在几个常量上；逐行抄四份会让「只差一个变量」的
// 纪律被抄写错误冲掉。`conv_probe.rs` 的头注释警告过数组形式没被标量化
// （`acc[ROWS][2]` 落进线程私有内存，比 `n_conv8` 慢 1.6 倍）——那时的
// 循环上界是**运行期**值。这里 ROWS/NBLK 是模板常量，`#pragma unroll`
// 后索引全为常量，累加器必须被提升到寄存器；`p_t4n2` 这个实例就是控制
// 组：它必须与 `p_base` 同时间（跑出来的表里第一行可以自校验）。
#include <metal_stdlib>
using namespace metal;

constant uint OFF_NONE = 0xFFFFFFFFu;

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

float4 apply_act(float4 v, uint act, float c1, float c2, float c3) {
    if (act == 2u) return max(v, float4(0.0));
    if (act == 1u) {
        float inv = 1.0 / c1;
        return c3 * v * (float4(erf1(v.x * inv), erf1(v.y * inv),
                                erf1(v.z * inv), erf1(v.w * inv)) + c2);
    }
    return v;
}

// 参数解析与 `n_conv8` 逐字段一致（多出来的只有 ROWS/NBLK 两个常量）。
template <uint ROWS, uint NBLK>
inline void tile_body(device float4* d, device const uint* params, uint P, uint gid, uint b) {
    uint in_off = params[P + 0], w_off = params[P + 1], b_off = params[P + 2];
    uint out_off = params[P + 3];
    uint mDim = params[P + 4], nDim = params[P + 5], outW = params[P + 6];
    uint inW = params[P + 7], inH = params[P + 8], cin4v = params[P + 9];
    uint taps = params[P + 10], kW = params[P + 11];
    uint sH = params[P + 12], sW = params[P + 13];
    uint padT = params[P + 14], padL = params[P + 15];
    uint act = params[P + 16];
    float c1 = as_type<float>(params[P + 17]);
    float c2 = as_type<float>(params[P + 18]);
    float c3 = as_type<float>(params[P + 19]);
    uint r_off = params[P + 20];

    uint nv = nDim >> 2;
    if (taps != 1u || (nv % NBLK) != 0u) return; // 只对 1×1 且能整分的形状有意义
    uint tiles_plan = ((mDim + 3u) >> 2) * (nv >> 1);
    uint nt = nv / NBLK;
    uint tiles = ((mDim + ROWS - 1u) / ROWS) * nt;
    // 覆盖性判据：候选瓦片的线程数必须 ≤ 计划给的。计划网格是
    // div256(tiles_plan)，所以 gid < tiles_plan 的线程一定存在；这里
    // 再早退 gid >= tiles，剩下的线程就恰好铺满本瓦片。
    if (gid >= tiles_plan || gid >= tiles) return;
    uint n4 = (gid % nt) * NBLK;
    uint m0 = (gid / nt) * ROWS;
    uint ib = (in_off >> 2) + b * inH * inW * cin4v;
    uint wb = w_off >> 2;
    uint ob = (out_off >> 2) + b * mDim * nv;

    float4 acc[ROWS][NBLK];
    uint oh[ROWS], ow[ROWS];
    bool ok[ROWS];
    #pragma unroll
    for (uint r = 0u; r < ROWS; r++) {
        uint m = m0 + r;
        ok[r] = m < mDim;
        oh[r] = m / outW;
        ow[r] = m - oh[r] * outW;
        #pragma unroll
        for (uint c = 0u; c < NBLK; c++) acc[r][c] = float4(0.0);
    }

    for (uint t = 0u; t < taps; t++) {
        uint ky = t / kW, kx = t - ky * kW;
        for (uint ci4 = 0u; ci4 < cin4v; ci4++) {
            float4 g[ROWS];
            #pragma unroll
            for (uint r = 0u; r < ROWS; r++) {
                g[r] = float4(0.0);
                if (ok[r]) {
                    int ih = int(oh[r] * sH + ky) - int(padT);
                    int iw = int(ow[r] * sW + kx) - int(padL);
                    if (ih >= 0 && ih < int(inH) && iw >= 0 && iw < int(inW)) {
                        g[r] = d[ib + (uint(ih) * inW + uint(iw)) * cin4v + ci4];
                    }
                }
            }
            uint wRow = wb + (t * cin4v + ci4) * 4u * nv + n4;
            #pragma unroll
            for (uint c = 0u; c < NBLK; c++) {
                #pragma unroll
                for (uint i = 0u; i < 4u; i++) {
                    float4 wv = d[wRow + c + i * nv];
                    #pragma unroll
                    for (uint r = 0u; r < ROWS; r++) acc[r][c] += g[r][i] * wv;
                }
            }
        }
    }

    if (b_off != OFF_NONE) {
        #pragma unroll
        for (uint c = 0u; c < NBLK; c++) {
            float4 bv = d[(b_off >> 2) + n4 + c];
            #pragma unroll
            for (uint r = 0u; r < ROWS; r++) acc[r][c] += bv;
        }
    }
    if (act != 0u) {
        #pragma unroll
        for (uint r = 0u; r < ROWS; r++)
            #pragma unroll
            for (uint c = 0u; c < NBLK; c++) acc[r][c] = apply_act(acc[r][c], act, c1, c2, c3);
    }
    if (r_off != OFF_NONE) {
        uint rb = (r_off >> 2) + b * mDim * nv;
        #pragma unroll
        for (uint r = 0u; r < ROWS; r++)
            if (ok[r])
                #pragma unroll
                for (uint c = 0u; c < NBLK; c++)
                    acc[r][c] += d[rb + (m0 + r) * nv + n4 + c];
    }
    #pragma unroll
    for (uint r = 0u; r < ROWS; r++) {
        if (!ok[r]) continue;
        #pragma unroll
        for (uint c = 0u; c < NBLK; c++) d[ob + (m0 + r) * nv + n4 + c] = acc[r][c];
    }
}

#define TILE_ENTRY(NAME, ROWS, NBLK)                                            \
    kernel void NAME(                                                           \
        device float4* d [[buffer(0)]],                                         \
        device const uint* params [[buffer(1)]],                                \
        constant uint& p_off [[buffer(3)]],                                     \
        uint3 gid3 [[thread_position_in_grid]],                                 \
        uint3 wg [[threadgroup_position_in_grid]])                              \
    {                                                                           \
        uint b = wg.y;                                                          \
        if (params[0] != 0u && b >= params[0]) return;                          \
        tile_body<ROWS, NBLK>(d, params, p_off, gid3.x, b);                     \
    }

// 控制（= p_base 的几何，验证模板没有把累加器赶进线程私有内存）
TILE_ENTRY(p_t4n2, 4, 2)
// 权重复用 ×2、输入装载翻倍
TILE_ENTRY(p_t8n1, 8, 1)
TILE_ENTRY(p_t8n2, 8, 2)
// 权重复用 ×4（16 个 float4 累加器 = 64 寄存器，且输入装载翻 4 倍）
TILE_ENTRY(p_t16n1, 16, 1)
// 权重复用 ×4 且输入装载只翻 2 倍——但 32 个 float4 累加器 = 128 寄存器，
// 预期溢出；列在这里是为了让「寄存器预算」这条边界有实测数字。
TILE_ENTRY(p_t16n2, 16, 2)
