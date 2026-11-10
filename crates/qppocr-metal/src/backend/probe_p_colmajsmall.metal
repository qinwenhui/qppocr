// 变体 `p_colmajsmall`：**列主序 + 权重地址掩进常驻窗口**（故意算错，只看时间）。
//
// ## 要回答的问题：`p_wsmall` 那 1.56× 的余量，列主序吃掉了吗？
//
// 第 4 节定案「卡在权重装载的 cache line 数 / 局部性」，处方是线程组暂存
// （任务 #12），上限取自 `p_wsmall` 的 **1.56×**。但那个 1.56× 是拿
// **`p_base`** 当基线量的——那时列主序还没进生产。列主序改善的正是
// 「一条装载指令横跨几条 line」，而 `p_wsmall` 改善的是「地址footprint
// 太小所以常驻」；**两者可能大量重叠**。若不先验这一条就去写线程组暂存，
// 很可能辛苦半天拿 0%。
//
// 判读：
//   ① `p_colmajsmall` 相对 `p_base` ≈ 1.56×  ⇒ 余量还在，任务 #12 值得做；
//   ② 相对 `p_colmajsmall` ≈ 1.0×（即相对 p_base ≈ p_colmaj 的 1.25×）
//      ⇒ **列主序已经把这份余量吃掉了**，线程组暂存这条路的上限所剩无几，
//      任务 #12 该降级或改向。
//
// ## 唯一改动
//
// 相对 `p_colmaj` 只把权重行的基址 `& 0x3Fu`（掩进 64 个 float4 = 1 KB）。
// 与 `p_wsmall` 逐字同款掩法（`wRow + i*nv` 里的 `i*nv` 不掩，所以实际
// footprint 是 4 行 × 64 列 × 16 B = 4 KB，每次迭代都是同一片 ⇒ L1 常驻）。
// **装载指令条数、地址算术、累加顺序全部不变**——这正是它比
// `p_wconst`（整条装载消失）干净的地方。
//
// ⚠️ 第二列仍用 `+ nh`（列主序的原偏移）不掩。footprint 因此是两片 4 KB，
// 分居 `nh` 两侧——都是常驻的，仍然量的是「流量≈0」那个方向。
//
// ----以下是 `p_colmaj` 的原始说明，映射与它完全一致----
//
// 变体：**列主序 warp 排布**——只改「一个 warp 的 32 个 lane 落在哪些
// 通道上」，累加器、循环、指令条数、网格、覆盖全与 `p_base` 一致。
//
// ## 假设：权重装载的「行数」是 3×3 的真瓶颈，而不是字节数
//
// `p_base` 的映射 `n4 = (gid % nh) * 2` 让**相邻 lane 的通道相距 2 个
// float4**（32 B）。一个 warp 的一条权重装载指令因此横跨 32 lane × 32 B
// = 1024 B ≈ 8 条 cache line，却只用掉每条 line 的一半（512 B）——
// 两条指令（第 0 列、第 1 列）合起来才把 8 条 line 填满。L1 要为**每条
// 指令**过 8 条 line，于是装载的 wavefront 数（= L1 事务数）翻倍。
//
// 这解释了 `det 480x432 c32->32 k3` 上的一组夹逼：
//   p_base 3.301 / p_fmaonly 1.325（FMA 地板，81% 峰值）
//   p_wconst 1.958（砍权重装载，省 1.34 = 41%）
//   p_nocoal 3.032（钉死输入地址，只省 8%）
// 权重只有 36 KB、**整段 L1 常驻**，`p_wsmall`（0.435 ≈ p_base 0.450）
// 也印证了不是字节流量问题 ⇒ 剩下的 1.34 ms 只能是装载**指令/事务**的
// 开销，而 stride-2 的 lane 排布正好把事务数翻倍。
//
// ## 处方
//
// 一个 warp 的 32 个 lane 改为读**连续的 32 个 float4**（4 条 line 填满），
// 每个线程的第二列不再放 `n4+1`（相邻），而是放到 `n4 + nh`（nv/2 之外
// 的另一半）。两列合起来仍然恰好覆盖 `[0, nv)`：
//
//     lane q ∈ [0,32) → 列 A = q，列 B = q + nh        （nh = nv/2）
//     全局 q ∈ [0, nh) → 列 {q, q + nh} 双射覆盖 [0, nv)
//
// 网格公式不动（仍是 `ceil(m/4)*nh`，每线程仍是 2 列 8 通道），只要
// `nv` 为偶（内核既有契约）。输入侧完全不受影响：输入地址与 lane 无关，
// 仍是广播。
//
// ## 实测：**假设被否**（5 个形状 p_colmaj ≡ p_base，误差 1e-8 级）
//
//     分类层  3.217 vs p_base（p_t4n2 3.147 同源参照）
//     rec 48x544 k3  0.449 vs 0.452
//     rec 24x272 k3  0.481 vs 0.485
//     det 480x432 k3 3.303 vs 3.308
//     rec 6x68 k1    0.077 vs 0.078
//
// 全部在噪声内 ⇒ M4 的 L1 不为 stride-2 的 lane 排布付额外代价（按 32 B
// sector 取数，两条指令各自用的半条 line 合起来正好填满，没有浪费）。
// **列主序这条路不用再试**。3×3 那 41% 的权重装载开销是「装载指令数」
// 本身（或它引发的调度停顿），不是 line 事务数——要降只能降条数，见
// probe_p_hoist / probe_p_slide 的方向。
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

kernel void p_colmajsmall(
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
    // ★ 唯一改动：列主序——相邻 lane 的 n4 连续（warp 一条指令 = 32 个
    // 连续 float4 = 512 B 连续 = 4 条 line 填满），第二列放到 +nh。
    uint n4 = gid % nh;
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
            // ★ 相对 p_colmaj 的唯一改动：基址掩进 64 个 float4（1 KB）。
            // 掩法与 probe_p_wsmall.metal 逐字相同；`i*nv` 不掩，故实际
            // footprint = 4 行 × 64 列（+ 第二列那片），每次都一样 ⇒ 常驻。
            uint wRow = (wb + (t * cin4v + ci4) * 4u * nv + n4) & 0x3Fu;
            for (uint i = 0u; i < 4u; i++) {
                float4 wv0 = d[wRow + i * nv];
                float4 wv1 = d[wRow + i * nv + nh];
                float s0 = g0[i], s1 = g1[i], s2 = g2[i], s3 = g3[i];
                a0c0 += s0 * wv0; a1c0 += s1 * wv0; a2c0 += s2 * wv0; a3c0 += s3 * wv0;
                a0c1 += s0 * wv1; a1c1 += s1 * wv1; a2c1 += s2 * wv1; a3c1 += s3 * wv1;
            }
        }
    }
    if (b_off != OFF_NONE) {
        float4 bv0 = d[(b_off >> 2) + n4];
        float4 bv1 = d[(b_off >> 2) + n4 + nh];
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
        if (ok[0]) { a0c0 += d[rb + m0 * nv + n4]; a0c1 += d[rb + m0 * nv + n4 + nh]; }
        if (ok[1]) { a1c0 += d[rb + (m0 + 1u) * nv + n4]; a1c1 += d[rb + (m0 + 1u) * nv + n4 + nh]; }
        if (ok[2]) { a2c0 += d[rb + (m0 + 2u) * nv + n4]; a2c1 += d[rb + (m0 + 2u) * nv + n4 + nh]; }
        if (ok[3]) { a3c0 += d[rb + (m0 + 3u) * nv + n4]; a3c1 += d[rb + (m0 + 3u) * nv + n4 + nh]; }
    }
    if (ok[0]) { d[ob + m0 * nv + n4] = a0c0; d[ob + m0 * nv + n4 + nh] = a0c1; }
    if (ok[1]) { d[ob + (m0 + 1u) * nv + n4] = a1c0; d[ob + (m0 + 1u) * nv + n4 + nh] = a1c1; }
    if (ok[2]) { d[ob + (m0 + 2u) * nv + n4] = a2c0; d[ob + (m0 + 2u) * nv + n4 + nh] = a2c1; }
    if (ok[3]) { d[ob + (m0 + 3u) * nv + n4] = a3c0; d[ob + (m0 + 3u) * nv + n4 + nh] = a3c1; }
}
