// 变体 `p_wstage`：**列主序 + 权重切片暂存进线程组内存**。
//
// ## 为什么做它
//
// 5.6 节用 `p_colmajsmall` 验明：在已落地的列主序**之上**，把权重地址掩进
// 常驻小窗还值 **1.13–1.23×**（每个热 1×1 形状），且两者**乘性叠加**——
// 是「同一片权重被重复搬运」这个独立瓶颈。本变体就是那份余量的**可落地
// 处方**（掩地址只是诊断，不能抄进生产）。
//
// ## 机制
//
// 每个 `(t, ci4)` 迭代里，一层权重的 **4 行 × nv 列**是一段**连续**内存
// （`wb + (t*cin4v+ci4)*4*nv + [0, 4*nv)`）。让整个线程组把它**协作搬运**
// 进 threadgroup 内存，热循环改从组内存读：
//
//     全局流量：256 线程 × 8 float4 = 2048 个/迭代
//       ↓ 协作装载
//     4*nv 个/迭代（nv=96 ⇒ 384 个）≈ **5.3× 下降**，且完全连续合并
//
// 热形状的 `nv ∈ {48, 96, 192}` ⇒ 切片 4*nv ∈ {768, 1536, 3072} 字节×16
// = **3/6/12 KB**，一个固定 `threadgroup float4 slab[4*256]`（16 KB）够用。
// `nv > 256` 的层**回退**到直接全局装载（与 `p_colmaj` 逐字一致）。
//
// ## 位级一致（这是比性能更要紧的判据）
//
// 搬进 slab 的是**纯连续拷贝**：`slab[idx] = d[wbase + idx]`，而原来读的是
// `d[wbase + i*nv + n4]`。令 `idx = i*nv + n4` 两者**逐元素相同**。累加顺序
// （t → ci4 → i）一字未动 ⇒ **结果与 `p_colmaj` / `p_base` 位级一致**。
// 所以本变体的误差必须**逐位等于 `p_colmaj`**（如热形状的 3.73e-8）——
// 对不上就是搬运写错了，性能数字一概不看。
//
// ## ⚠️ 唯一的挂死风险，以及为什么它不会发生
//
// 线程组屏障**必须被组内全部线程**执行到。原来开头那两句早退
// （`gid >= tiles`、`b >= params[0]`）在带屏障的内核里是**致命**的：
// 网格是把 tiles 向上取整到 256 的倍数，尾组里 inactive 的线程若 `return`，
// 其余线程就卡在屏障上**永不返回 → 命令缓冲永不完成 → GPU 打瘸**。
// 所以这里把早退改成 **`active` 标志**：`return` 全部删掉，工作放进
// `if (active)`，而**屏障在 `if` 外面**——inactive 线程照常走到屏障。
// 协作装载那一段也不依赖 `gid`（`wbase` 只由 t/ci4 决定），inactive 线程
// 参与搬运是安全的。`nv <= 256` 是**组内一致**的判据，不会出现半个组
// 有屏障、半个组没有的情况。
//
// ## 相对 `p_colmaj` 的改动清单
//
//   1. 早退 → `active` 标志（见上）。
//   2. `threadgroup float4 slab[4*256]` + 协作装载 + 一对屏障。
//   3. 热循环里权重读 `slab[…]` 而不是 `d[…]`。
// 其余（映射、网格、输入 gather、bias/残差/激活/写回）**一字未改**。
#include <metal_stdlib>
using namespace metal;

constant uint OFF_NONE = 0xFFFFFFFFu;
// 一层权重的暂存容量：4 行 × 256 列。hot 层 nv ≤ 192，够。
constant uint WSTAGE_COLS = 256u;

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

kernel void p_wstage(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]])
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
    // ★ 早退改成标志：`return` 会让尾组里 inactive 的线程缺席屏障 → 挂死。
    bool active = (params[0] == 0u || b < params[0]);
    uint nv = nDim >> 2;
    uint nh = nv >> 1;
    uint gid = gid3.x;
    uint tiles = ((mDim + 3u) >> 2) * nh;
    active = active && (gid < tiles);

    // 列主序（同 p_colmaj）：相邻 lane 的 n4 连续，第二列放到 +nh。
    uint n4 = gid % nh;
    uint m0 = (gid / nh) * 4u;
    // 一层权重的暂存容量够不够——**组内一致**（只取决于 nv）。
    bool staged = (nv <= WSTAGE_COLS);

    uint ib = (in_off >> 2) + b * inH * inW * cin4v;
    uint wb = w_off >> 2;
    uint ob = (out_off >> 2) + b * mDim * nv;

    // 协作装载用的线程号与组大小。
    uint tx = tid3.x + tid3.y * tpg.x;
    uint nthr = tpg.x * tpg.y;
    threadgroup float4 slab[4u * WSTAGE_COLS];

    float4 a0c0 = float4(0.0), a1c0 = float4(0.0), a2c0 = float4(0.0), a3c0 = float4(0.0);
    float4 a0c1 = float4(0.0), a1c1 = float4(0.0), a2c1 = float4(0.0), a3c1 = float4(0.0);
    uint oh[4], ow[4];
    bool ok[4];
    for (uint i = 0u; i < 4u; i++) {
        uint m = m0 + i;
        ok[i] = active && (m < mDim);
        oh[i] = m / outW;
        ow[i] = m - oh[i] * outW;
    }

    for (uint t = 0u; t < taps; t++) {
        uint ky = t / kW, kx = t - ky * kW;
        for (uint ci4 = 0u; ci4 < cin4v; ci4++) {
            // ★ 协作装载：一段**连续**的 4*nv 个 float4。
            uint wbase = wb + (t * cin4v + ci4) * 4u * nv;
            if (staged) {
                for (uint idx = tx; idx < 4u * nv; idx += nthr) {
                    slab[idx] = d[wbase + idx];
                }
                // 装载可见之后才允许读。
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (active) {
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
                for (uint i = 0u; i < 4u; i++) {
                    // staged 时读组内存；否则回退到直接全局装载（逐字同 p_colmaj）。
                    float4 wv0 = staged ? slab[i * nv + n4] : d[wbase + i * nv + n4];
                    float4 wv1 = staged ? slab[i * nv + n4 + nh]
                                        : d[wbase + i * nv + n4 + nh];
                    float s0 = g0[i], s1 = g1[i], s2 = g2[i], s3 = g3[i];
                    a0c0 += s0 * wv0; a1c0 += s1 * wv0; a2c0 += s2 * wv0; a3c0 += s3 * wv0;
                    a0c1 += s0 * wv1; a1c1 += s1 * wv1; a2c1 += s2 * wv1; a3c1 += s3 * wv1;
                }
            }
            if (staged) {
                // 全部读完才允许下一轮覆盖。
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
    if (!active) return;
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
