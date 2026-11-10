// 变体：**1×1 同分辨率快路**——把 gather 地址整体提出热循环。
//
// ## 病根（读数指过来的）
//
// `p_fmaonly`（访存全砍）比 `p_base` 快 3 倍，而 `p_wsmall`（权重地址掩进
// 1 KB，条数与地址算术全保留）只快 8%：权重那侧不是瓶颈，输入那侧才是。
// 但「输入那侧」有两种病：流量、还是**装载指令 + 伴随的地址算术**。看
// `p_base` 的热循环，每 (t, ci4)、每行都要重算一遍：
//
//   oh[i] = m / outW（除法）、ow[i] = m - oh[i]*outW、
//   ih = oh[i]*sH + ky - padT、iw = ow[i]*sW + kx - padL、
//   两组越界比较 + 分支、px = ih*inW + iw、再乘 cin4v
//
// 这些量**与 ci4 完全无关**，却被重算了 `taps × cin4v` 遍（分类层
// cin4v=30 → 每线程 120 遍）。指令流里塞满 INT 除法/乘法/比较，FMA 反而
// 排不上队——这正是「快 3 倍」的那 3 倍从哪来的最省事解释。
//
// ## 处方：同分辨率 1×1 时 `px 恒等于 m`
//
// sH=sW=1、pad=0、inW=outW 时：ih = oh[i]、iw = ow[i]，
// 而 m = oh*outW + ow = oh*inW + ow = px——**逐位相等**。于是
//
//   ibase[i] = ib + (m0+i)*cin4v      （4 个寄存器，循环外算一次）
//   内层 g 装载 = d[ibase[i] + ci4]   （纯线性，无除法无比较无分支）
//
// 覆盖判据只留 `ok[i]`（尾块的 m ≥ mDim 行），越界判断可证明恒假：sH=1、
// padT=0 ⇒ ih = oh[i] < outH（m < mDim 保证）；iw = ow[i] < outW = inW。
//
// **累加顺序与 `p_base` 逐字相同**（t 只有 0、ci4 外层、i 内层），所以
// 结果位级一致——这不是近似，是同一算式的等价改写。
//
// ## 与 `p_base` 的其余差别：无
//
// 头注释、文件名、内核名、外加一个 `if`。网格、覆盖、epilogue（bias/act/
// 残差/存储）与 `p_base` 共用同一段代码。
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

kernel void p_k1fast(
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
    bool ok[4];
    for (uint i = 0u; i < 4u; i++) {
        ok[i] = m0 + i < mDim;
    }

    // ── 快路判据（全部是计划期已知的形状常量，逐 dispatch 恒定）──
    if (taps == 1u && sH == 1u && sW == 1u && padT == 0u && padL == 0u && inW == outW) {
        uint ib0 = ib + m0 * cin4v;
        uint ib1 = ib0 + cin4v, ib2 = ib1 + cin4v, ib3 = ib2 + cin4v;
        for (uint ci4 = 0u; ci4 < cin4v; ci4++) {
            // 4 行地址 = 4 个寄存器 + ci4；无除法、无乘法、无越界比较。
            float4 g0 = float4(0.0), g1 = float4(0.0), g2 = float4(0.0), g3 = float4(0.0);
            if (ok[0]) g0 = d[ib0 + ci4];
            if (ok[1]) g1 = d[ib1 + ci4];
            if (ok[2]) g2 = d[ib2 + ci4];
            if (ok[3]) g3 = d[ib3 + ci4];
            uint wRow = wb + ci4 * 4u * nv + n4;
            for (uint i = 0u; i < 4u; i++) {
                float4 wv0 = d[wRow + i * nv];
                float4 wv1 = d[wRow + i * nv + 1u];
                float s0 = g0[i], s1 = g1[i], s2 = g2[i], s3 = g3[i];
                a0c0 += s0 * wv0; a1c0 += s1 * wv0; a2c0 += s2 * wv0; a3c0 += s3 * wv0;
                a0c1 += s0 * wv1; a1c1 += s1 * wv1; a2c1 += s2 * wv1; a3c1 += s3 * wv1;
            }
        }
    } else {
        uint oh[4], ow[4];
        for (uint i = 0u; i < 4u; i++) {
            uint m = m0 + i;
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
                uint wRow = wb + (t * cin4v + ci4) * 4u * nv + n4;
                for (uint i = 0u; i < 4u; i++) {
                    float4 wv0 = d[wRow + i * nv];
                    float4 wv1 = d[wRow + i * nv + 1u];
                    float s0 = g0[i], s1 = g1[i], s2 = g2[i], s3 = g3[i];
                    a0c0 += s0 * wv0; a1c0 += s1 * wv0; a2c0 += s2 * wv0; a3c0 += s3 * wv0;
                    a0c1 += s0 * wv1; a1c1 += s1 * wv1; a2c1 += s2 * wv1; a3c1 += s3 * wv1;
                }
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
