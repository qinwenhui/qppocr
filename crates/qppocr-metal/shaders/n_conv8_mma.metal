// 稠密 1×1 conv（NHWC f32）——simdgroup_matrix MMA 版。
//
// # 为什么要有这个内核
//
// `n_conv8` 的全部单因子变体已穷举（见 `probe_p_*` 系列），结论钉死在
// 「load 指令发射瓶颈」：m4×n8 瓦片的 load:FMA = 12:8，每条权重装载只被
// 4 个输出行复用（权重字节/FLOP = 0.5，形状无关）。提高复用（= 减少 load）
// 的所有标量手段都撞在寄存器文件上——m8/m16 的累加器 = m×8 个 float4。
//
// `simdgroup_float8x8` 把一个 8×8 累加分片摊在 **32 个 lane** 上（每 lane
// 2 个寄存器），于是 16 行 × 32 列 = 512 个输出元素只占每 lane 64 个寄存器，
// 瓦片能开到 64 行（`acc[2][4]` × 8 个 simdgroup/线程组）——权重复用从
// 4× 提到 64×。探针 `p_mm_sg` 在同结构下量到 ~1476 GFLOPS（44% fp32 峰值），
// 生产 `n_conv8` 同形状只有 554 GFLOPS。
//
// # 适配（与竞品 `mm_ic_sg` 同型，但零精度损失）
//
// - **纯 f32**：A/B/acc 全是 `simdgroup_float8x8`（fp32 MMA），输入/权重/
//   输出语义与 `n_conv8` 逐位一致，无 fp16 精度代价。
// - **权重零重排**：现有 repack 布局 `[tap][ci4][i(4)][n4]` 在 taps==1 时
//   就是 `[k][n]` k-主序（行 stride = nv 个 float4）——正是 MMA 的 B 操作
//   数想要的 K×N 行主序，直接 `simdgroup_load` 不转置。
// - **输入零重排**：NHWC 下位置 m 的 kf 个通道连续 = A 的行（行主序，
//   行 stride = kf），`simdgroup_load(a, ...)` 直接可用。
//
// # 启用条件（计划层保证，不满足则派发 n_conv8 原内核）
//
// - `taps == 1 && kW == 1 && sH == sW == 1 && padT == padL == 0`
//   （1×1 无 pad：输出位置 m 与输入像素 m 一一对应，A 无 gather）；
// - `nDim % 64 == 0`（64 列块不越界；nDim 是补 4 后的 Co，float 元素数）；
// - `cin4v % 2 == 0`（kf % 8 == 0：K 步长 8 不越界；Cin%8==0 即满足）。
//
// M 方向尾块会读 m >= mDim 的输入（越界行只进被丢弃的累加器，且落在
// arena 内——输出段跟在输入段之后，无 fault 风险），写回按 m < mDim 掩码。
//
// # 网格（与 n_conv8 的 `ceil(m/4)*nh` 完全不同，计划层单算）
//
//   groups = [ceil(mDim/64), ceil(nDim/64), batch]，线程组恒 256 线程。
//   x = 64 行块，y = 64 列块（批从 n_conv8 的 wg.y 挪到 wg.z）。
//
// 线程组内：8 个 simdgroup，sgid 认领 16 行 × 32 列
// （`(sgid>>1)*16 + {0,8}` 行 × `(sgid&1)*32 + {0,8,16,24}` 列），
// `acc[2][4]` 共 512 元素/simdgroup，8 个合起来覆盖 64×64。
//
// # 参数块语义：与 n_conv8 逐字段一致（P+0..20），零改动。

#include <metal_stdlib>
using namespace metal;

constant uint OFF_NONE = 0xFFFFFFFFu;

// exp1/erf1：n_conv8.metal 的逐字拷贝（与 CPU 位级一致的多项式）。
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

// apply_act 的标量版（MMA 尾声按 thread 逐元素处理，不再是 float4）。
float act1(float v, uint act, float c1, float c2, float c3) {
    if (act == 2u) return max(v, 0.0f);
    if (act == 1u) {
        float inv = 1.0 / c1;
        return c3 * v * (erf1(v * inv) + c2);
    }
    return v;
}

kernel void n_conv8_mma(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    // 带属性的内核参数必须同为标量或同为同宽向量（MSL 硬约束），故 tid 走 uint3。
    uint tid = tid3.x;
    device const float* f32v = (device const float*)d;
    device float* of = (device float*)d;

    uint P = p_off;
    uint in_off = params[P + 0], w_off = params[P + 1], b_off = params[P + 2];
    uint out_off = params[P + 3];
    uint mDim = params[P + 4], nDim = params[P + 5];
    uint inH = params[P + 7], inW = params[P + 8], cin4v = params[P + 9];
    uint act = params[P + 16];
    float c1 = as_type<float>(params[P + 17]);
    float c2 = as_type<float>(params[P + 18]);
    float c3 = as_type<float>(params[P + 19]);
    uint r_off = params[P + 20];

    uint b = wg.z;
    if (params[0] != 0u && b >= params[0]) return; // 批上限早退（同 n_conv8）

    uint nv = nDim >> 2;        // float4 列数（列 stride 用）
    uint kf = cin4v * 4u;        // K 的 float 元素数（行 stride 用）

    uint m0 = wg.x * 64u;       // 64 行基（输出位置）
    uint n0 = wg.y * 64u;       // 64 列基（输出通道，float 元素）

    uint sgid = tid >> 5;       // 8 个 simdgroup / 线程组
    uint sm = m0 + (sgid >> 1) * 16u;   // 本 simdgroup 的 16 行基
    uint sn = n0 + (sgid & 1u) * 32u;  // 本 simdgroup 的 32 列基

    uint ib = (in_off >> 2) * 4u + b * inH * inW * kf;  // 输入段 float 基
    uint wb = (w_off >> 2) * 4u;                        // 权重段 float 基
    uint ob = (out_off >> 2) * 4u + b * mDim * nDim;    // 输出段 float 基

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++)
            acc[i][j] = simdgroup_float8x8(0.0f);

    // 热循环：K 步长 8。每步 2 条 A 装载 + 4 条 B 装载 + 8 条 MMA，
    // 服务 16 行 × 32 列 × 8 K = 4096 MAC —— load:FMA 从 n_conv8 的
    // 12:8 掉到 6:4096（SIMD 协作装载，每 lane 只搬自己分片）。
    for (uint kb = 0u; kb < kf; kb += 8u) {
        simdgroup_float8x8 a0, a1;
        simdgroup_load(a0, f32v + ib + sm * kf + kb, kf);
        simdgroup_load(a1, f32v + ib + (sm + 8u) * kf + kb, kf);
        simdgroup_float8x8 b0, b1, b2, b3;
        // 权重 [k][n] k-主序（行 stride = nDim float），无需转置。
        simdgroup_load(b0, f32v + wb + kb * nDim + sn, nDim);
        simdgroup_load(b1, f32v + wb + kb * nDim + sn + 8u, nDim);
        simdgroup_load(b2, f32v + wb + kb * nDim + sn + 16u, nDim);
        simdgroup_load(b3, f32v + wb + kb * nDim + sn + 24u, nDim);
        simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
        simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
        simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
        simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
        simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
        simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
        simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
    }

    // 尾声：分片落 threadgroup（16 KB/组），再 256 线程逐元素做
    // bias/act/残差/写回。语义与 n_conv8 的尾声逐字段对齐（bias 加在
    // 通道、act 逐元素、残差按 (m,n) 读、写回带 m 掩码）。
    threadgroup float ost[64u * 64u];
    uint r0 = (sgid >> 1) * 16u;   // 相对本 64×64 块的行基
    uint c0 = (sgid & 1u) * 32u;   // 列基
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++)
            simdgroup_store(acc[i][j], ost + (r0 + i * 8u) * 64u + c0 + j * 8u, 64u);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tid; e < 64u * 64u; e += 256u) {
        uint m = m0 + (e >> 6);
        uint n = n0 + (e & 63u);
        if (m >= mDim) continue;
        float v = ost[e];
        if (b_off != OFF_NONE) v += f32v[(b_off >> 2) * 4u + n];
        v = act1(v, act, c1, c2, c3);
        if (r_off != OFF_NONE)
            v += f32v[(r_off >> 2) * 4u + b * mDim * nDim + m * nDim + n];
        of[ob + m * nDim + n] = v;
    }
}
