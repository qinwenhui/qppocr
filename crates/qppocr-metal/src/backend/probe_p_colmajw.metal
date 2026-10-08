// 变体：**窗口序 + 列主序**——把 `p_window` 与 `p_colmaj` 两个映射叠起来。
// 这是下面那份「窗口序」源码的副本，**唯一差别是窗内 lane→列 的排布**。
//
// ## 为什么要把两个叠起来（2026-10-08）
//
// 生产 `n_conv8` 现在按门 `taps==1 && cin4v*nv >= 4096`（权重段 ≥ 256 KB）
// 走窗口序。可实测那张权重尺寸阶梯上，**两个映射各管一段、且互相打架**：
//
//   288 KB  （rec 6x208 c192->384 等，**频率最高的两档** 94+80 条 dispatch）
//           p_window 0.239 ≈ p_base 0.252（只值 5%），p_colmaj **0.201 = 1.25×**
//           —— 权重 288 KB 本来就躺在 L2 里，窗口序无益；赚的是 lane 连续。
//   8771 KB（CTC 头 104x1，12 条）
//           p_window 0.733 = 1.9×，p_colmaj 1.464 = **0.92×（更差）**
//           —— 这里复用距离才是命门。
//
// 于是要么按权重尺寸切两套映射（阈值取多少**没有数据**，573 KB~1152 KB
// 那几档一个都没测），要么把两者**一起用**：窗口序管复用距离、列主序管
// lane 排布，各修各的病。本变体就是后者——**若它在两端都不劣于各自的
// 胜者，就不需要阈值了**，一条路覆盖全部 taps==1。
//
// ## 机制上为什么不冲突
//
// 窗口序动的是「哪个 gid 拿哪片通道窗」，列主序动的是「窗内 32 个 lane 拿
// 哪 32 个 float4」。窗宽 32 块 = 64 个 float4，lane q 取第 q 个与第 q+32
// 个，正好把这一窗读成两条**满载**的 4-line 装载（`p_base` 是两条半数
// 利用的 8-line 装载）。尾数分支不用动：那里一个 warp 的 32 个 lane 本就
// 跨行组、同通道 ⇒ 权重装载已经是广播（1 条事务），没有可省的。
//
// ⚠️ 判读只看**误差是否为 1e-8 量级**（必须与 `p_base` 同位），以及
// `p_colmajw` 是否 ≥ max(`p_window`, `p_colmaj`)。
//
// ## 假设
//
// `p_wsmall` 把「权重 9 MB 被 54 个行瓦片各读一遍 = 485 MB」判成了**流量**
// 问题而不是装载条数问题（同一形状：p_base 2.956 → p_wsmall 0.784，条数与
// 地址算术完全一致，只把地址掩进 1 KB）。若真是流量，那么**不动指令、
// 只把复用的时间距离压短**就该拿到接近 0.784 的数。
//
// ## 现状的复用距离
//
// 现映射 `n4=(gid%nh)*2, m0=(gid/nh)*4` 是**通道最快变化**：一个 256 线程
// 工作组 = 1 个行组 × 512 个通道块（1024 通道）。同一片权重要等 54 个
// 行组各走完一遍才被再读一次——**复用距离 = 整个 9 MB 权重段**。
//
// ## 本变体
//
// 把「通道窗」提到行组外面：先让同一片 32 通道块的**所有行组**背靠背走完，
// 再推进到下一片。映射仍是 126306 个 tile 的双射（网格一字不改）：
//
//     gid = w*(32*rows_g) + r_g*32 + cl      （w = 通道窗号）
//     cp  = w*32 + cl                        （cp = co4 对下标）
//
// 尾部：`nh = 2339 = 73*32 + 3`，前 73 个窗整满，余下 3 个通道 × 54 行组
// = 162 个 tile 正好是剩下的尾数（`rows_g*nh - 73*32*rows_g = 3*54`），
// 用一个分支单独铺掉。**这份巧合并偶然**：`126306 = 2339*54`，而
// `126306 = 73*1728 + 162`、`162 = 3*54`——任何形状都成立（见代码里
// `full` 的算法），所以它是个对任意 (m, n) 都成立的通用双射。
//
// ## 为什么行组数取 54（m4×n2）而不是 27
//
// 这是纯窗口序实验：瓦片保持 m4×n2，与 `p_base` 只差映射，差异才能干净
// 归因到「复用距离」。瓦片与窗口序**可以叠加**，见 `p_window8`。
//
// ## 负载形态没变
//
// 一个 warp 的 32 个 lane 仍然是 32 个**连续通道块**（cl = lane）：输入
// 地址与块号无关 ⇒ 32 lane 读同一地址（广播 1 条事务，同 `p_base`）；
// 权重的 32 个地址跨 512 B（同 `p_base` 的 4 行）。变的是**这片权重
// 多久之后还会被用到**。
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

kernel void p_colmajw(
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
    uint rows_g = (mDim + 3u) >> 2;            // 行组数（= 计划网格的一维）
    uint tiles = rows_g * nh;
    if (gid >= tiles) return;
    // ★ 唯一改动：窗口序双射（见文件头）+ 窗内列主序
    uint full = (tiles / (32u * rows_g)) * 32u * rows_g;
    uint n4, m0, n4b; // n4b = 第二个通道块（float4 列）的下标
    if (gid < full) {
        uint w = gid / (32u * rows_g);
        uint rem = gid - w * (32u * rows_g);
        // 窗宽 32 块 = 64 个 float4。lane q 取第 q 与第 q+32 个 float4：
        // 相邻 lane 的地址连续 ⇒ 一条装载 = 32×16 B = 512 B = 4 条**满载**
        // 的 line（原式 `(w*32+q)*2` 是 stride-2，跨 8 条 line 各用一半）。
        // 双射仍在：q ∈ [0,32) 的两列合起来覆盖本窗 [0,64) 全部 float4。
        n4 = (w << 6u) + (rem & 31u);
        n4b = n4 + 32u;
        m0 = (rem >> 5u) * 4u;
    } else {
        // 尾数：一个 warp 的 lane 跨行组、同通道 ⇒ 权重装载本就是广播，
        // 列主序无从下手，保持与原窗口序逐字一致（第二列 = 相邻的 +1）。
        uint rem = gid - full;
        uint wb = full / (32u * rows_g) * 32u;
        n4 = (wb + rem / rows_g) * 2u;
        n4b = n4 + 1u;
        m0 = (rem % rows_g) * 4u;
    }
    // 第二列相对第一列的偏移（wconst 循环外算一次，热循环里只加这个常量）。
    uint dcol = n4b - n4;
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
            uint wRow = wb + (t * cin4v + ci4) * 4u * nv + n4;
            for (uint i = 0u; i < 4u; i++) {
                float4 wv0 = d[wRow + i * nv];
                float4 wv1 = d[wRow + i * nv + dcol];
                float s0 = g0[i], s1 = g1[i], s2 = g2[i], s3 = g3[i];
                a0c0 += s0 * wv0; a1c0 += s1 * wv0; a2c0 += s2 * wv0; a3c0 += s3 * wv0;
                a0c1 += s0 * wv1; a1c1 += s1 * wv1; a2c1 += s2 * wv1; a3c1 += s3 * wv1;
            }
        }
    }
    if (b_off != OFF_NONE) {
        float4 bv0 = d[(b_off >> 2) + n4];
        float4 bv1 = d[(b_off >> 2) + n4b];
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
        if (ok[0]) { a0c0 += d[rb + m0 * nv + n4]; a0c1 += d[rb + m0 * nv + n4b]; }
        if (ok[1]) { a1c0 += d[rb + (m0 + 1u) * nv + n4]; a1c1 += d[rb + (m0 + 1u) * nv + n4b]; }
        if (ok[2]) { a2c0 += d[rb + (m0 + 2u) * nv + n4]; a2c1 += d[rb + (m0 + 2u) * nv + n4b]; }
        if (ok[3]) { a3c0 += d[rb + (m0 + 3u) * nv + n4]; a3c1 += d[rb + (m0 + 3u) * nv + n4b]; }
    }
    if (ok[0]) { d[ob + m0 * nv + n4] = a0c0; d[ob + m0 * nv + n4b] = a0c1; }
    if (ok[1]) { d[ob + (m0 + 1u) * nv + n4] = a1c0; d[ob + (m0 + 1u) * nv + n4b] = a1c1; }
    if (ok[2]) { d[ob + (m0 + 2u) * nv + n4] = a2c0; d[ob + (m0 + 2u) * nv + n4b] = a2c1; }
    if (ok[3]) { d[ob + (m0 + 3u) * nv + n4] = a3c0; d[ob + (m0 + 3u) * nv + n4b] = a3c1; }
}
