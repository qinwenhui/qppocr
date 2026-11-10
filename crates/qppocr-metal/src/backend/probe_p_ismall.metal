// 诊断变体：**输入（激活）地址**钉在 1 KB 窗口里（**故意算错**，只看时间）。
//
// ## 要回答的问题
//
// `p_wsmall` 量的是「权重足迹砍到≈0 值多少」。**激活这一侧一直没有对应的
// 对照**——而激活恰恰是卷积访存的大头：最小流量式 `4·(cin·m + co·m)`
// （激活）+ `4·cin·co·taps`（权重）里，1×1 层 m 有几千、cin/co 才几百，
// 激活占绝对多数。
//
// 于是「把激活也改成 fp16」这个改造（足迹砍半）的价值无从判断。
// 本变体就是那条对照：保留 `p_base` **一模一样的装载条数与地址算术**，
// 只把输入地址掩进 64 个 float4（1 KB）的窗口，于是激活流量≈0（首次之后
// 全在 L1），而权重侧**一字不改**（注意：与 `p_wsmall` 相反的那一半）。
//
// ## 怎么读
//
//   p_ismall ≈ p_base    → 激活**流量**根本不是约束：fp16 激活（减字节、
//                          减 line 数）没有可赚的钱，别做那个改造；
//   p_ismall ≪ p_base    → 激活流量是真约束，减半足迹值得做。
//
// 与同轮的 `p_wsmall`（权重掩码）、`p_fmaonly`（全部 load 砍掉）一起看，
// 三者的份额就能把「激活 / 权重 / 其余」劈开。
//
// ## 这个探针**只掩读侧**（诚实边界）
//
// 掩的是热循环里的**激活装载**（`taps × cin4v` 次/线程），**不掩输出写出**
// （每线程 8 条 store）。所以量到的是「激活读流量 → 0」的收益，不是「激活
// 流量全 → 0」。刻意如此：把输出也掩进 1 KB 会让所有 warp 砸同一条 L1
// line，那是写出争用的人造瓶颈，会把数往上带。
// ⇒ 读数应当理解成**下界**：真的全砍只会更大一点，但不会翻倍。
//
// **2026-10-10 实测（10 个生产形状）**：`p_base/p_ismall` = **1.00~1.35×**
// （中位 ~1.05×），而同轮 `p_wsmall` 到 4.5×、`p_fmaonly` 到 9.9×。
// 最有力的一条是 `det 480x432 c32->32 k3`：那里激活占流量 **99.9%**（输入+
// 输出各 26.5 MB，权重才 36 KB），把激活读流量全砍掉也**只快 1.07×**。
// ⇒ 结论：**激活流量不是约束，fp16 激活这条路不宜走**。见
// `logs/activation_traffic_not_the_constraint_20261010.md`。
//
// ## 掩码安全性
//
// `px*cin4v + ci4` 最大 = `inH*inW*cin4v - 1`（输入段的全部 float4）。
// 掩进 64 个 float4 后落在 `[ib, ib+63]`，只要输入段 ≥ 64 个 float4 就
// 一定在段内——比 `p_wsmall` 更保守（那边靠「权重段远大于 63」）。更小的
// 输入退化成不掩（等价 `p_base`），避免读到输入段之外。
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

kernel void p_ismall(
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
    // 激活掩码：输入段装得下 1 KB 窗口才掩；装不下就退化成不掩（等价
    // p_base）——否则会读到输入段之外。掩码是热循环外的常量，不进热路径。
    uint amask = (inH * inW * cin4v >= 64u) ? 0x3Fu : 0xFFFFFFFFu;

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
                // ★ 唯一改动：激活地址掩进 64 个 float4 的窗口（1 KB，L1
                //   常驻）。装载条数、地址算术、累加序全同 p_base；**权重
                //   侧一字不改**（与 p_wsmall 恰好互补的那一半）。
                float4 v = d[ib + ((px * cin4v + ci4) & amask)];
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
