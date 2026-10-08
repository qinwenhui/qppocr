// 变体：**m8×n4 瓦片 + 窗口序**（两个改动叠加）。
//
// `p_m8n4` 单独拿到 1.53×（2.956 → 2.033 ms），`p_window` 单独验证
// 「复用距离」是不是那 485 MB 的根因。这一条把两者叠在一起：瓦片仍是
// m8×n1（8 行 × 1 块 = 8 个 float4 累加器，权重每 float4 换 8 次 FMA），
// 映射换成窗口序。
//
// 窗口序在 m8×n1 下的形状：一个 warp = 32 个 lane × 1 个通道块 = 32 个
// **co4**（不是 co4 对），所以窗口是 32 个 co4 = 128 通道；行组数
// `ceil(m/8) = 27`。`nv = 4678 = 146*32 + 6`，尾部 6 个 co4 × 27 行组
// = 162 个 tile，同样由 `full` 之后的那个分支铺掉（对任意形状都成立）。
//
// 网格仍是计划的 m4n8 公式（`tiles = 27*4678 = 126306 = 54*2339`，恰好
// 与计划等值），所以零计划改动。
#include <metal_stdlib>
using namespace metal;
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

kernel void p_window8(
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
    uint gid = gid3.x;
    uint rows_g = (mDim + 7u) >> 3;
    uint tiles = rows_g * nv;
    if (gid >= tiles) return;
    // ★ 窗口序双射（窗口 = 32 个 co4 = 一个 warp 的通道跨度）
    uint full = (tiles / (32u * rows_g)) * 32u * rows_g;
    uint n4, m0;
    if (gid < full) {
        uint w = gid / (32u * rows_g);
        uint rem = gid - w * (32u * rows_g);
        n4 = (w << 5u) + (rem & 31u);
        m0 = (rem >> 5u) * 8u;
    } else {
        uint rem = gid - full;
        uint wb = full / (32u * rows_g) * 32u;
        n4 = wb + rem / rows_g;
        m0 = (rem % rows_g) * 8u;
    }
    uint ib = (in_off >> 2) + b * inH * inW * cin4v;
    uint wb = w_off >> 2;
    uint ob = (out_off >> 2) + b * mDim * nv;

    float4 a0 = float4(0.0), a1 = float4(0.0), a2 = float4(0.0), a3 = float4(0.0);
    float4 a4 = float4(0.0), a5 = float4(0.0), a6 = float4(0.0), a7 = float4(0.0);
    uint oh[8], ow[8];
    bool ok[8];
    for (uint i = 0u; i < 8u; i++) {
        uint m = m0 + i;
        ok[i] = m < mDim;
        oh[i] = m / outW;
        ow[i] = m - oh[i] * outW;
    }

    for (uint t = 0u; t < taps; t++) {
        uint ky = t / kW, kx = t - ky * kW;
        for (uint ci4 = 0u; ci4 < cin4v; ci4++) {
            float4 g0 = float4(0.0), g1 = float4(0.0), g2 = float4(0.0), g3 = float4(0.0);
            float4 g4 = float4(0.0), g5 = float4(0.0), g6 = float4(0.0), g7 = float4(0.0);
            for (uint i = 0u; i < 8u; i++) {
                if (!ok[i]) continue;
                int ih = int(oh[i] * sH + ky) - int(padT);
                int iw = int(ow[i] * sW + kx) - int(padL);
                if (ih < 0 || ih >= int(inH) || iw < 0 || iw >= int(inW)) continue;
                uint px = uint(ih) * inW + uint(iw);
                float4 v = d[ib + px * cin4v + ci4];
                if (i == 0u) g0 = v; else if (i == 1u) g1 = v;
                else if (i == 2u) g2 = v; else if (i == 3u) g3 = v;
                else if (i == 4u) g4 = v; else if (i == 5u) g5 = v;
                else if (i == 6u) g6 = v; else g7 = v;
            }
            uint wRow = wb + (t * cin4v + ci4) * 4u * nv + n4;
            for (uint i = 0u; i < 4u; i++) {
                float4 wv = d[wRow + i * nv];
                float s0 = g0[i], s1 = g1[i], s2 = g2[i], s3 = g3[i];
                float s4 = g4[i], s5 = g5[i], s6 = g6[i], s7 = g7[i];
                a0 += s0 * wv; a1 += s1 * wv; a2 += s2 * wv; a3 += s3 * wv;
                a4 += s4 * wv; a5 += s5 * wv; a6 += s6 * wv; a7 += s7 * wv;
            }
        }
    }
    if (b_off != OFF_NONE) {
        float4 bv = d[(b_off >> 2) + n4];
        a0 += bv; a1 += bv; a2 += bv; a3 += bv;
        a4 += bv; a5 += bv; a6 += bv; a7 += bv;
    }
    if (act != 0u) {
        a0 = apply_act(a0, act, c1, c2, c3); a1 = apply_act(a1, act, c1, c2, c3);
        a2 = apply_act(a2, act, c1, c2, c3); a3 = apply_act(a3, act, c1, c2, c3);
        a4 = apply_act(a4, act, c1, c2, c3); a5 = apply_act(a5, act, c1, c2, c3);
        a6 = apply_act(a6, act, c1, c2, c3); a7 = apply_act(a7, act, c1, c2, c3);
    }
    if (r_off != OFF_NONE) {
        uint rb = (r_off >> 2) + b * mDim * nv + n4;
        if (ok[0]) a0 += d[rb + (m0 + 0u) * nv];
        if (ok[1]) a1 += d[rb + (m0 + 1u) * nv];
        if (ok[2]) a2 += d[rb + (m0 + 2u) * nv];
        if (ok[3]) a3 += d[rb + (m0 + 3u) * nv];
        if (ok[4]) a4 += d[rb + (m0 + 4u) * nv];
        if (ok[5]) a5 += d[rb + (m0 + 5u) * nv];
        if (ok[6]) a6 += d[rb + (m0 + 6u) * nv];
        if (ok[7]) a7 += d[rb + (m0 + 7u) * nv];
    }
    uint obn = ob + n4;
    if (ok[0]) d[obn + (m0 + 0u) * nv] = a0;
    if (ok[1]) d[obn + (m0 + 1u) * nv] = a1;
    if (ok[2]) d[obn + (m0 + 2u) * nv] = a2;
    if (ok[3]) d[obn + (m0 + 3u) * nv] = a3;
    if (ok[4]) d[obn + (m0 + 4u) * nv] = a4;
    if (ok[5]) d[obn + (m0 + 5u) * nv] = a5;
    if (ok[6]) d[obn + (m0 + 6u) * nv] = a6;
    if (ok[7]) d[obn + (m0 + 7u) * nv] = a7;
}
