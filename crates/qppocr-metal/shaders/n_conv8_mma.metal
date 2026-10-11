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
// - `nDim % 64 == 0` 或 `nDim % 32 == 0`（列块不越界；nDim 是补 4 后的 Co，
//   float 元素数）。两条选不同入口，见下面「两档」一节；
// - `cin4v % 2 == 0`（kf % 8 == 0：K 步长 8 不越界；Cin%8==0 即满足）。
//
// M 方向尾块会读 m >= mDim 的输入（越界行只进被丢弃的累加器，且落在
// arena 内——输出段跟在输入段之后，无 fault 风险），写回按 m < mDim 掩码。
//
// # 网格（与 n_conv8 的 `ceil(m/4)*nh` 完全不同，计划层单算）
//
//   groups = [ceil(mDim/64), ceil(nDim/NCOLS), batch]。
//   x = 64 行块，y = NCOLS 列块（批从 n_conv8 的 wg.y 挪到 wg.z）。
//
// 线程组内：NSG 个 simdgroup，sgid 认领 RPSG 行 × NCOLS/2 列，
// `acc[NR][NB]` 共 8 个分片/simdgroup，NSG 个合起来覆盖 64×NCOLS。
//
// # 两档：NCOLS=64/NSG=8（256 线程）与 NCOLS=32/NSG=4（128 线程）
//
// **为什么不只有 64 那一档**：`nDim % 64 == 0` 这条判据把 rec/det 里成片的
// 真实层挡在门外——rec 的 Conv.4/9/12/15/47 与 det 的 CSP 支线都是 **Co=96**
// （`cpad4(96)=96`，`96%64=32`），它们合计占 `n_conv8` 总时的大头，却只能
// 走标量内核。Co=96 是 3 个 32 列块，正落在 NCOLS=32 这一档上。
// 这与 `n_conv8_mma3` 早就有 `_n32` 实例化是**同一个理由、同一套做法**：
// 除 NCOLS/NSG 两行参数与由它们派生的四处（nch / sn / 分片数 / 尾声步长）外，
// 两条实例化的推理逐字共用——改一处两条同时改。
//
// 与 `n_conv8_mma3_n32` 同款的两点副作用：① `ost` 从 16 KB 降到 8 KB/组，
// 每核可驻留的线程组数翻倍；② 线程组 128 而非 256。B 分片在 NCOLS=32 时
// 每条只被 NR=4 个行分片复用（NCOLS=64 时是 2 个复用 4 条）——单条 MMA 的
// 装载比变差，但瓦片更小、驻留更多。**实测这两档都赢**（`small` 档端到端
// A/B 4 轮中位 −2.40 ms/图；逐内核归因：det `n_conv8` 8.35→5.20、`_n32`
// 只花 1.96 = **1.61×**，rec 14.78→11.56、`_n32` 花 1.60 = **2.01×**，
// 100 图 1041 行文本/框逐位不变）。
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

// NCOLS = 输出通道块宽（64 或 32）；NSG = 本线程组里 simdgroup 的个数
// （8 → 256 线程，4 → 128 线程）。两档的理由见文件头「两档」一节。
//
// `ost`（分片落地的 threadgroup 暂存）**必须由内核入口声明、以指针传进来**：
// MSL 硬规则——`threadgroup` 变量只能在 `kernel` 函数里声明（"variables in
// the threadgroup address space cannot be declared in a non-qualified
// function"）。于是各入口声明自己那份 `float[64*NCOLS]`，尺寸耦合由入口的
// static_assert 守住。
//
// ⚠️ **NSG 必须与宿主侧的线程组尺寸一致**（`pipeline.rs::threads_per_group`）。
// 内核按 `tid>>5` 认 simdgroup，线程组给大了就会有线程认领到不存在的 sg、
// 去写 `ost` 的界外行——静默算错。这道耦合由 `conv8_mma_matches_cpu`
// 那条 CPU 对拍兜住（它对两条实例化都跑）。
template <uint NCOLS, uint NSG>
static void mma1_body(
    device float4* d,
    device const uint* params,
    uint P,
    uint tid,
    uint3 wg,
    threadgroup float* ost)
{
    // 分片数：每 16 个通道一个 8x8 分片列。
    constexpr uint NB = NCOLS / 16u;          // 64 -> 4，32 -> 2
    // 行方向：64 行像素均分给 NSG/2 个行组，每组再切成 8 行的分片。
    constexpr uint RPSG = 64u / (NSG / 2u);   // 每 simdgroup 的行数（8 -> 16，4 -> 32）
    constexpr uint NR = RPSG / 8u;            // 每 simdgroup 的行分片数（8 -> 2，4 -> 4）
    constexpr uint NTH = NSG * 32u;           // 线程组大小（8 -> 256，4 -> 128）

    device const float* f32v = (device const float*)d;
    device float* of = (device float*)d;

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

    uint m0 = wg.x * 64u;         // 64 行基（输出位置）
    uint n0 = wg.y * NCOLS;       // NCOLS 列基（输出通道，float 元素）

    uint sgid = tid >> 5;         // NSG 个 simdgroup / 线程组
    uint sm = m0 + (sgid >> 1) * RPSG;        // 本 simdgroup 的行基
    uint sn = n0 + (sgid & 1u) * (NCOLS / 2u); // 本 simdgroup 的 NCOLS/2 列基

    uint ib = (in_off >> 2) * 4u + b * inH * inW * kf;  // 输入段 float 基
    uint wb = (w_off >> 2) * 4u;                        // 权重段 float 基
    uint ob = (out_off >> 2) * 4u + b * mDim * nDim;    // 输出段 float 基

    simdgroup_float8x8 acc[NR][NB];
    for (uint i = 0u; i < NR; i++)
        for (uint j = 0u; j < NB; j++)
            acc[i][j] = simdgroup_float8x8(0.0f);

    // 热循环：K 步长 8。每步 NR 条 A 装载 + NB 条 B 装载 + NR*NB 条 MMA，
    // 服务 64 行 × NCOLS 列 × 8 K —— load:FMA 从 n_conv8 的 12:8 掉到
    // 6:4096（SIMD 协作装载，每 lane 只搬自己分片）。
    for (uint kb = 0u; kb < kf; kb += 8u) {
        simdgroup_float8x8 a[NR], b[NB];
        for (uint i = 0u; i < NR; i++)
            simdgroup_load(a[i], f32v + ib + (sm + i * 8u) * kf + kb, kf);
        // 权重 [k][n] k-主序（行 stride = nDim float），无需转置。
        for (uint j = 0u; j < NB; j++)
            simdgroup_load(b[j], f32v + wb + kb * nDim + sn + j * 8u, nDim);
        for (uint i = 0u; i < NR; i++)
            for (uint j = 0u; j < NB; j++)
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }

    // 尾声：分片落 threadgroup（NCOLS=64 -> 16 KB/组，32 -> 8 KB/组），再
    // NTH 线程逐元素做 bias/act/残差/写回。语义与 n_conv8 的尾声逐字段对齐
    // （bias 加在通道、act 逐元素、残差按 (m,n) 读、写回带 m 掩码）。
    // 暂存区由入口声明（见上面那条 MSL 规则），尺寸 = 64*NCOLS。
    uint r0 = (sgid >> 1) * RPSG;   // 相对本 64×NCOLS 块的行基
    uint c0 = (sgid & 1u) * (NCOLS / 2u);   // 列基
    for (uint i = 0u; i < NR; i++)
        for (uint j = 0u; j < NB; j++)
            simdgroup_store(acc[i][j], ost + (r0 + i * 8u) * NCOLS + c0 + j * 8u, NCOLS);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tid; e < 64u * NCOLS; e += NTH) {
        uint m = m0 + (e / NCOLS);
        uint n = n0 + (e % NCOLS);
        if (m >= mDim) continue;
        float v = ost[e];
        if (b_off != OFF_NONE) v += f32v[(b_off >> 2) * 4u + n];
        v = act1(v, act, c1, c2, c3);
        if (r_off != OFF_NONE)
            v += f32v[(r_off >> 2) * 4u + b * mDim * nDim + m * nDim + n];
        of[ob + m * nDim + n] = v;
    }
}

// 两条实例化。计划层按 `cpad4(Co)%64==0 ? 64 : 32` 选名（见 plan.rs）。
// 网格公式、参数块对两条完全一致——**只有通道块宽与线程组大小不同**。
kernel void n_conv8_mma(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    threadgroup float ost[64u * 64u];
    static_assert(sizeof(ost) == 64u * 64u * sizeof(float), "暂存区尺寸");
    // 带属性的内核参数必须同为标量或同为同宽向量（MSL 硬约束），故 tid 走 uint3。
    // 8 个 simdgroup = **256 线程**（`pipeline.rs::threads_per_group` 的默认档）。
    mma1_body<64u, 8u>(d, params, p_off, tid3.x, wg, ost);
}

kernel void n_conv8_mma_n32(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    // 8 KB/组（NCOLS=64 那支是 16 KB）：通道块宽减半，暂存区跟着减半。
    threadgroup float ost[64u * 32u];
    static_assert(sizeof(ost) == 64u * 32u * sizeof(float), "暂存区尺寸");
    // **4 个 simdgroup = 128 线程**，不是 256（见文件头「两档」一节）。
    mma1_body<32u, 4u>(d, params, p_off, tid3.x, wg, ost);
}
