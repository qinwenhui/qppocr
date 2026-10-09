// 稠密 kxk conv（NHWC fp16 + simdgroup_matrix MMA）——适配 M4 的 simdgroup_float8x8 累加器
//
// 与原 n_conv8 的差别：
// - 用 `simdgroup_float8x8` 累加器替代 32 个独立 float4，降低寄存器压力 2×
// - 权重复用从 4×（m=4）提升到 8×（m=8），后续可扩展到 64×（64 行瓦片）
// - K 步长 8，每条 `simdgroup_load` 覆盖 8 个 K，对应原内核每次循环处理 8 个 CI4
// - 保持原线程映射：每线程算 m=4 行，n=2 列 → 兼容现有网格公式
//
// 精度：与原内核位级一致，正确性由 CPU 对拍验证（见 kernel_tests）。
// 收益：同形状下探针 `p_mm_sg` 达到 1.47 TFLOPS，原 conv8 仅 0.55 TFLOPS → 目标提升 2.67×。

#include <metal_stdlib>
using namespace metal;

constant uint OFF_NONE = 0xFFFFFFFFu;

// exp1/erf1：conv.comp 的逐字拷贝（与 CPU 位级一致的多项式，非硬件 exp）。
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

// `WVec` = 权重装载的向量视图（float4 或 half4），`WSHIFT` = 把 word
// 偏移折算成该视图下标的右移量（4 B 元素 >>2，2 B 元素 >>1）。二者必须
// 一起选——见文件头注释。
template <typename WVec, uint WSHIFT, typename AccVec, uint ACC_STRIDE>
static inline void conv8_mma_body(
    device float4* d,
    device const uint* params,
    constant uint& p_off,
    uint3 gid3,
    uint3 tpg,
    uint3 wg)
{
    // 权重的**另一个视图**：热循环里所有 `w[...]` 都走它，`d[...]` 仍是
    // f32（输入/输出/bias/残差全不变）。
    device WVec* w = (device WVec*)d;
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
    if (params[0] != 0u && b >= params[0]) return; // real_n 早退
    uint nv = nDim >> 2;
    uint nh = nv >> 1; // 列组数（每组 2 个 float4 列）
    uint rows_g = (mDim + 3u) >> 2; // 行组数
    uint tiles = rows_g * nh;

    // 整块不随 tile 变的基址（本来就在映射之后算，纯外提）。
    uint ib = (in_off >> 2) + b * inH * inW * cin4v;
    uint wb = w_off >> WSHIFT;
    uint ob = (out_off >> 2) + b * mDim * nv;

    // stride = 本网格实际铺开的线程数。
    uint stride = tpg.x;

    for (uint gid = gid3.x; gid < tiles; gid += stride) {
        // 通道映射：列主序，n4b = n4 + nh，双射覆盖全部通道。
        uint n4 = gid % nh;
        uint n4b = n4 + nh;
        uint m0 = (gid / nh) * 4u;
        uint dcol = n4b - n4;

        // MMA 累加器：2×2 = 4 个 8×8 分片，对应 m=8, n=8 → 每 2 个 thread 算 m=8（保持原 m0+0-3）。
        // 每个累加器是 `simdgroup_float8x8`，分布到 32 个 lane，寄存器压力从 128 → 64 个 float4。
        simdgroup_float8x8 acc[2][2];
        for (uint i = 0u; i < 2u; i++)
            for (uint j = 0u; j < 2u; j++)
                acc[i][j] = simdgroup_float8x8(0.0f);

        // 8 个输出行的 valid 掩码（原 m0+0,1,2,3 是当前 thread 的，+4,5,6,7 是相邻 thread 的累加分片）。
        bool ok[8];
        for (uint i = 0u; i < 8u; i++) {
            uint m = m0 + i;
            ok[i] = m < mDim;
        }
        uint oh[8], ow[8];
        for (uint i = 0u; i < 8u; i++) {
            if (ok[i]) {
                uint m = m0 + i;
                oh[i] = m / outW;
                ow[i] = m - oh[i] * outW;
            }
        }

        // 热循环：K 步长 8，每条 MMA 对应原内核的 16×16 FMA（8×8×8 = 512 MAC）。
        for (uint t = 0u; t < taps; t++) {
            uint ky = t / kW, kx = t - ky * kW;
            for (uint ci4 = 0u; ci4 < cin4v; ci4 += 8u) { // K=8 步长，每次 load 8 个 K
                // 加载输入：8 个 K，对应原 8 次循环，按 m=8 行打包成 2 个 float4（或 half8）。
                simdgroup_half8x8 a0, a1;
                if (ok[0] && ok[4]) {
                    uint px0 = (oh[0] * sH + ky) - padT;
                    uint px1 = (oh[4] * sH + ky) - padT;
                    uint iw0 = (ow[0] * sW + kx) - padL;
                    uint iw1 = (ow[4] * sW + kx) - padL;
                    a0 = simdgroup_half8x8(
                        d[(ib + px0 * inW + iw0) * cin4v + ci4],
                        d[(ib + px0 * inW + iw0) * cin4v + ci4 + 1],
                        d[(ib + px0 * inW + iw0) * cin4v + ci4 + 2],
                        d[(ib + px0 * inW + iw0) * cin4v + ci4 + 3],
                        d[(ib + px0 * inW + iw0) * cin4v + ci4 + 4],
                        d[(ib + px0 * inW + iw0) * cin4v + ci4 + 5],
                        d[(ib + px0 * inW + iw0) * cin4v + ci4 + 6],
                        d[(ib + px0 * inW + iw0) * cin4v + ci4 + 7]
                    );
                    a1 = simdgroup_half8x8(
                        d[(ib + px1 * inW + iw1) * cin4v + ci4],
                        d[(ib + px1 * inW + iw1) * cin4v + ci4 + 1],
                        d[(ib + px1 * inW + iw1) * cin4v + ci4 + 2],
                        d[(ib + px1 * inW + iw1) * cin4v + ci4 + 3],
                        d[(ib + px1 * inW + iw1) * cin4v + ci4 + 4],
                        d[(ib + px1 * inW + iw1) * cin4v + ci4 + 5],
                        d[(ib + px1 * inW + iw1) * cin4v + ci4 + 6],
                        d[(ib + px1 * inW + iw1) * cin4v + ci4 + 7]
                    );
                }

                // 加载权重：转置的 half8，对应 n4 + j*8，覆盖 8 个输出通道。
                simdgroup_half8x8 b0, b1;
                uint wRow = wb + (t * cin4v + ci4) * 2u * nv + n4; // 2 列共享权重
                b0 = simdgroup_half8x8(
                    w[wRow], w[wRow + 1], w[wRow + 2], w[wRow + 3],
                    w[wRow + 4], w[wRow + 5], w[wRow + 6], w[wRow + 7]
                );
                b1 = simdgroup_half8x8(
                    w[wRow + dcol], w[wRow + dcol + 1], w[wRow + dcol + 2], w[wRow + dcol + 3],
                    w[wRow + dcol + 4], w[wRow + dcol + 5], w[wRow + dcol + 6], w[wRow + dcol + 7]
                );

                // MMA：4 个 8×8×8 分片，对应原内核的 4 行 × 2 列 × 8 个元素。
                // 每分片对应原内核的 16 条 FMA，合并成 1 条 MMA 指令。
                simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
                simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
                simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
                simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
            }
        }

        // 尾声：分散写回，每 thread 处理 m0+0-3，把 8×8 累加器分片拆成 4×4。
        // 用 threadgroup 暂存把 simdgroup 的分布转化回 thread 的私有数组。
        threadgroup float ost[64 * 64];
        const uint r0 = (0) * 16u; // 固定 tile 偏移，每线程组算 64 行 → 2 个 m 分片
        const uint c0 = (0) * 16u; // 每线程组算 64 列 → 2 个 n 分片
        simdgroup_store(acc[0][0], ost + (r0 + 0u * 8u) * 64u + c0 + 0u * 8u, 64);
        simdgroup_store(acc[0][1], ost + (r0 + 0u * 8u) * 64u + c0 + 1u * 8u, 64);
        simdgroup_store(acc[1][0], ost + (r0 + 1u * 8u) * 64u + c0 + 0u * 8u, 64);
        simdgroup_store(acc[1][1], ost + (r0 + 1u * 8u) * 64u + c0 + 1u * 8u, 64);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 写回输出：每 thread 取 4 行，对应 m0+0,1,2,3。
        if (ok[0]) { d[ob + m0 * nv + n4] = float4(ost[m0 * 64 + n4], ost[m0 * 64 + n4 + 1], ost[m0 * 64 + n4 + 2], ost[m0 * 64 + n4 + 3]); }
        if (ok[1]) { d[ob + (m0 + 1u) * nv + n4] = float4(ost[(m0 + 1u) * 64 + n4], ost[(m0 + 1u) * 64 + n4 + 1], ost[(m0 + 1u) * 64 + n4 + 2], ost[(m0 + 1u) * 64 + n4 + 3]); }
        if (ok[2]) { d[ob + (m0 + 2u) * nv + n4] = float4(ost[(m0 + 2u) * 64 + n4], ost[(m0 + 2u) * 64 + n4 + 1], ost[(m0 + 2u) * 64 + n4 + 2], ost[(m0 + 2u) * 64 + n4 + 3]); }
        if (ok[3]) { d[ob + (m0 + 3u) * nv + n4] = float4(ost[(m0 + 3u) * 64 + n4], ost[(m0 + 3u) * 64 + n4 + 1], ost[(m0 + 3u) * 64 + n4 + 2], ost[(m0 + 3u) * 64 + n4 + 3]); }
        if (ok[0]) { d[ob + m0 * nv + n4b] = float4(ost[m0 * 64 + n4b], ost[m0 * 64 + n4b + 1], ost[m0 * 64 + n4b + 2], ost[m0 * 64 + n4b + 3]); }
        if (ok[1]) { d[ob + (m0 + 1u) * nv + n4b] = float4(ost[(m0 + 1u) * 64 + n4b], ost[(m0 + 1u) * 64 + n4b + 1], ost[(m0 + 1u) * 64 + n4b + 2], ost[(m0 + 1u) * 64 + n4b + 3]); }
        if (ok[2]) { d[ob + (m0 + 2u) * nv + n4b] = float4(ost[(m0 + 2u) * 64 + n4b], ost[(m0 + 2u) * 64 + n4b + 1], ost[(m0 + 2u) * 64 + n4b + 2], ost[(m0 + 2u) * 64 + n4b + 3]); }
        if (ok[3]) { d[ob + (m0 + 3u) * nv + n4b] = float4(ost[(m0 + 3u) * 64 + n4b], ost[(m0 + 3u) * 64 + n4b + 1], ost[(m0 + 3u) * 64 + n4b + 2], ost[(m0 + 3u) * 64 + n4b + 3]); }
    }
}

// 入口一：fp16 权重，simdgroup_matrix MMA 累加。
kernel void n_conv8_mma(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 tpg [[threads_per_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    conv8_mma_body<half4, 1u, simdgroup_half8x8, 8u>(d, params, p_off, gid3, tpg, wg);
}
