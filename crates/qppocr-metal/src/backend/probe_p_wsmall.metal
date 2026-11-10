// 诊断变体：权重地址钉在 1 KB 窗口里（**故意算错**，只看时间）。
//
// ## 要分开的两个假设
//
// `p_base` 与 `p_wconst` 的差（CTC 分类层 2.935 → 0.496 ms）说明「权重流」
// 吃掉 83% 的时间，但这**两种完全不同的机制**都会给出这个差：
//
//   (a) **流量**：权重 9 MB，被 54 个行瓦片各读一遍 = 485 MB，超出
//       DRAM/L2 的供给能力 → 卡在带宽上。
//   (b) **装载指令**：每 32 个 FMA 要 8 条 `LDG.128`（权重），LSU 槽位
//       被占满 → 卡在指令发射上。
//
// 两者的处方**方向完全相反**：
//   (a) → 用线程组内存把权重**暂存后就地复用**（DRAM 只读一遍 = 9 MB），
//         装载指令数不变，但每条的代价从 L2 往返降到片上。
//   (b) → 唯一出路是**减少每 FMA 的装载条数**（加大行分块），而寄存器
//         预算又把它卡死在 8 个 float4 累加器上——那就到此为止。
//
// ## 怎么分开
//
// 本变体保留 `p_base` **一模一样的装载条数与地址算术**，只把权重地址
// 掩进 64 个 float4（1 KB）的窗口：装载指令一条不少，流量≈0（首次之后
// 全在 L1）。于是：
//
//   p_wsmall ≈ p_wconst  → 是 (a) 流量：线程组暂存有大钱可赚
//   p_wsmall ≈ p_base    → 是 (b) 指令：别再想暂存的事
//
// 掩码安全性：`wRow` 最大 63，加 `i*nv`（i≤3、nv≤4678）最大 14,097 个
// float4，远在权重段（本层 561k 个）之内；`n4` 是 co4 下标，也被掩掉。
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

kernel void p_wsmall(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    uint P = p_off;
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

    uint b = wg.y;
    if (params[0] != 0u && b >= params[0]) return;
    uint nv = nDim >> 2;
    uint nh = nv >> 1;
    uint gid = gid3.x;
    uint tiles = ((mDim + 3u) >> 2) * nh;
    if (gid >= tiles) return;
    uint n4 = (gid % nh) * 2u;
    uint m0 = (gid / nh) * 4u;
    uint ib = (in_off >> 2) + b * inH * inW * cin4v;
    uint wb = w_off >> 2;
    uint ob = (out_off >> 2) + b * mDim * nv;

    float4 a0c0 = float4(0.0), a1c0 = float4(0.0), a2c0 = float4(0.0), a3c0 = float4(0.0);
    float4 a0c1 = float4(0.0), a1c1 = float4(0.0), a2c1 = float4(0.0), a3c1 = float4(0.0);
    uint oh[4], ow[4];
    bool ok[4];
    for (uint i = 0u; i < 4u; i++) {
        uint m = m0 + i;
        ok[i] = m < mDim;
        oh[i] = m / outW;
        ow[i] = m - oh[i] * outW;
    }

    for (uint t = 0u; t < taps; t++) {
        uint ky = t / kW, kx = t - ky * kW;
        for (uint ci4 = 0u; ci4 < cin4v; ci4++) {
            float4 g0 = float4(0.0), g1 = float4(0.0), g2 = float4(0.0), g3 = float4(0.0);
            for (uint i = 0u; i < 4u; i++) {
                if (!ok[i]) continue;
                int ih = int(oh[i] * sH + ky) - int(padT);
                int iw = int(ow[i] * sW + kx) - int(padL);
                if (ih < 0 || ih >= int(inH) || iw < 0 || iw >= int(inW)) continue;
                uint px = uint(ih) * inW + uint(iw);
                float4 v = d[ib + px * cin4v + ci4];
                if (i == 0u) g0 = v; else if (i == 1u) g1 = v;
                else if (i == 2u) g2 = v; else g3 = v;
            }
            // ★ 唯一改动：地址掩进 64 个 float4 的窗口（1 KB，L1 常驻）。
            //   条数、地址算术、累加序全同 p_base。
            uint wRow = (wb + (t * cin4v + ci4) * 4u * nv + n4) & 0x3Fu;
            for (uint i = 0u; i < 4u; i++) {
                float4 wv0 = d[wRow + i * nv];
                float4 wv1 = d[wRow + i * nv + 1u];
                float s0 = g0[i], s1 = g1[i], s2 = g2[i], s3 = g3[i];
                a0c0 += s0 * wv0; a1c0 += s1 * wv0; a2c0 += s2 * wv0; a3c0 += s3 * wv0;
                a0c1 += s0 * wv1; a1c1 += s1 * wv1; a2c1 += s2 * wv1; a3c1 += s3 * wv1;
            }
        }
    }
    if (b_off != OFF_NONE) {
        float4 bv0 = d[(b_off >> 2) + n4];
        float4 bv1 = d[(b_off >> 2) + n4 + 1u];
        a0c0 += bv0; a1c0 += bv0; a2c0 += bv0; a3c0 += bv0;
        a0c1 += bv1; a1c1 += bv1; a2c1 += bv1; a3c1 += bv1;
    }
    if (act != 0u) {
        a0c0 = apply_act(a0c0, act, c1, c2, c3);
        a1c0 = apply_act(a1c0, act, c1, c2, c3);
        a2c0 = apply_act(a2c0, act, c1, c2, c3);
        a3c0 = apply_act(a3c0, act, c1, c2, c3);
        a0c1 = apply_act(a0c1, act, c1, c2, c3);
        a1c1 = apply_act(a1c1, act, c1, c2, c3);
        a2c1 = apply_act(a2c1, act, c1, c2, c3);
        a3c1 = apply_act(a3c1, act, c1, c2, c3);
    }
    if (r_off != OFF_NONE) {
        uint rb = (r_off >> 2) + b * mDim * nv;
        if (ok[0]) { a0c0 += d[rb + m0 * nv + n4]; a0c1 += d[rb + m0 * nv + n4 + 1u]; }
        if (ok[1]) { a1c0 += d[rb + (m0 + 1u) * nv + n4]; a1c1 += d[rb + (m0 + 1u) * nv + n4 + 1u]; }
        if (ok[2]) { a2c0 += d[rb + (m0 + 2u) * nv + n4]; a2c1 += d[rb + (m0 + 2u) * nv + n4 + 1u]; }
        if (ok[3]) { a3c0 += d[rb + (m0 + 3u) * nv + n4]; a3c1 += d[rb + (m0 + 3u) * nv + n4 + 1u]; }
    }
    if (ok[0]) { d[ob + m0 * nv + n4] = a0c0; d[ob + m0 * nv + n4 + 1u] = a0c1; }
    if (ok[1]) { d[ob + (m0 + 1u) * nv + n4] = a1c0; d[ob + (m0 + 1u) * nv + n4 + 1u] = a1c1; }
    if (ok[2]) { d[ob + (m0 + 2u) * nv + n4] = a2c0; d[ob + (m0 + 2u) * nv + n4 + 1u] = a2c1; }
    if (ok[3]) { d[ob + (m0 + 3u) * nv + n4] = a3c0; d[ob + (m0 + 3u) * nv + n4 + 1u] = a3c1; }
}
