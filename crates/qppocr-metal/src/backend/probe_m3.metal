// `n_conv8_mma3` 家族的**三臂诊断**——把 `probe_mma_split.metal` 在 1×1 上
// 用熟的那套（base / small / hoist / noepi）搬到 k×k 的 MMA 内核上，并且
// **同时在 NCOLS=64 与 NCOLS=32 两条实例化上跑**。
//
// ## 为什么要单独做一份
//
// 1×1 那边的结论**不能直接搬**：k×k 多一层 taps 循环、多一个尾块、多一条
// NCOLS=32 的实例化，热循环的装载条数完全不同。
//
//   每 kb 步、每 simdgroup 的装载:MMA 条数
//     NCOLS=64（8 组 = 4 行组 × 2 列组，每组 16 行 × 32 列）：2 A + 4 B : 8 MMA = 0.75
//     NCOLS=32（8 组 = 4 行组 × 2 列组，每组 16 行 × 16 列）：2 A + 2 B : 4 MMA = 1.00
//
//   即 **n32 每条 MMA 要多花 33% 的装载**。这正好对着端到端看到的差：
//   64 宽那族纯 MMA 发射率 **2024~2455 GFLOPS**（已到家族天花板 2044~2444），
//   而 n32 只有 **1503~1821**。所以本探针的第一要务是回答：**n32 慢是不是
//   真的慢在这 0.25 的比值上**（若是，处方是把每 simdgroup 的瓦片从 16×16
//   抬到 512 格 = 32×16 或 16×32，比值回到 0.75）。
//
// 2. 本族时间里有 **14.5% 是「算完就被尾块掩码丢掉」的冗余**（逐条算过：
//   Conv.121 算出 256 列只需 214 列 ⇒ 1.196×）。这笔账用探针不好量（它是
//   网格性质不是热循环性质），所以**单独用「少发一个 xblk」的网格对照**量，
//   见 `conv8_mma3_load_attribution` 里的 `tail` 两臂。
//
// ## 四臂（单变量，逐条对照）
//
//   MODE 0 base    与生产 `mma3_body` 的**热循环 + 尾声逐字节相同**（对照组）。
//   MODE 1 small   A/B 的基址**与行距**都收进 1 KB（条数/类型/MMA 全不变）
//                  ⇒ 劈出「**流量/延迟**」那一半。行距也要收的理由同
//                  `probe_mma_split.metal`（只掩基址的话 8 行照样散在几十 KB）。
//   MODE 2 hoist   全部 `simdgroup_load` **挪出 taps×kb 两层循环**，MMA 条数与
//                  累加链**一字不改** ⇒ 劈出「**发射槽**」那一半。
//   MODE 3 noepi   删掉分片落 threadgroup + 屏障 + 尾声 + 输出写 ⇒ 给「动尾声」
//                  的**上界**（同时去掉算术、写流量、8~16 KB 线程组内存三样）。
//
// ## 怎么读
//
//   base ≈ hoist                ⇒ 热循环不是装载发射槽卡的 ⇒ n32 的 1.00 比值
//                                 **不是**主因，别再改瓦片映射；
//   base ≫ hoist（≈ small）     ⇒ 卡在**装载发射槽** ⇒ 值得把每 simdgroup 瓦片
//                                 抬到 512 格（n32: 16×16 → 32×16，比值 1.00 → 0.75）；
//   base ≫ small，hoist ≈ base  ⇒ 卡在**流量/延迟**（权重 footprint / 起步行跨步）
//                                 ⇒ 该动的是 L2 局部性，不是条数。
//
// ## 诚实边界
//
// * `MODE 1/2/3` **数值全错**（诊断件，不是能落地的内核）；`MODE 0` 与生产
//   同义但**没参加过对拍**——它只用来当对照，别拿它当正确性证据。
// * 四臂共用 arena 与参数块，段布局与生产 `build_conv_case` 逐字段一致；
//   尾声在同一模板里共用，保证「尾声」是常量、差分只落在热循环上。
// * 与 `n_conv8_mma3` 一样**只跑内点几何**（启用门由计划层保证）；这里不重
//   实现门禁，网格由测试按生产公式给出。
#include <metal_stdlib>
using namespace metal;

constant uint OFF_NONE = 0xFFFFFFFFu;

// exp1/erf1/act1：`n_conv8_mma3.metal` 的逐字拷贝。
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

float act1(float v, uint act, float c1, float c2, float c3) {
    if (act == 2u) return max(v, 0.0f);
    if (act == 1u) {
        float inv = 1.0 / c1;
        return c3 * v * (erf1(v * inv) + c2);
    }
    return v;
}

// NCOLS = 输出通道块宽（64 或 32），MODE 见文件头。几何解码与生产
// `mma3_body` 逐字一致（覆盖矩形、尾块锚点与掩码、sgid 映射、参数读法）。
template <uint NCOLS, uint MODE>
static void m3_body(
    device float4* d,
    device const uint* params,
    uint P,
    uint tid,
    uint3 wg,
    threadgroup float* ost)
{
    constexpr uint NB = NCOLS / 16u;

    device const float* f32v = (device const float*)d;
    device float* of = (device float*)d;

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

    uint b = wg.z;
    if (params[0] != 0u && b >= params[0]) return;

    uint nch = nDim / NCOLS;
    uint nblk = wg.x % nch;
    uint xblk = wg.x / nch;

    uint kf = cin4v * 4u;

    uint yd_lo = (padT + sH - 1u) / sH;
    uint xa = (padL + sW - 1u) / sW;
    uint xb = (inW - 1u - (kW - 1u) + padL) / sW;
    uint y = yd_lo + wg.y;

    uint xb_blk = xb - 63u;
    uint nfb = (xb_blk - xa) / 64u + 1u;
    uint cover_end = xa + 64u * nfb - 1u;
    uint x0, i_lo;
    if (xblk < nfb) {
        x0 = xa + xblk * 64u;
        i_lo = 0u;
    } else {
        x0 = xb_blk;
        i_lo = (cover_end + 1u) - x0;
    }
    uint n0 = nblk * NCOLS;

    // ── simdgroup 布局 ────────────────────────────────────────────────
    //
    // 不变式（**每 simdgroup 的瓦片面积恒 512 格 = 8 个 8x8 分片**）：
    //
    //   NCOLS=64（8 组 = 4 行组 × 2 列组，每组 16 行 × 32 列）：2A + 4B : 8 MMA = 0.75
    //   NCOLS=32（8 组 = 4 行组 × 2 列组，每组 16 行 × 16 列）：2A + 2B : 4 MMA = 1.00
    //   NCOLS=32 MODE 4（**4 组** = 2 行组 × 2 列组，每组 32 行 × 16 列）：4A + 2B : 8 MMA = 0.75
    //
    // MODE 4 就是本文件要证的处方：**把线程组砍到 128（4 个 simdgroup），
    // 瓦片仍是 64×32、网格仍是同一组数、尾块与掩码一字不改**，只是每个
    // simdgroup 多扛 16 行 ⇒ 每条 MMA 的装载数从 1.00 压回 0.75。
    constexpr bool FOUR = (MODE == 4u);
    constexpr uint NSG = FOUR ? 4u : 8u;
    constexpr uint NTH = NSG * 32u;                 // 线程组大小
    constexpr uint RPSG = 64u / (NSG / 2u);         // 每 simdgroup 的行数
    constexpr uint NR = RPSG / 8u;                  // 每 simdgroup 的行分片数

    uint sgid = tid >> 5;
    uint sr = (sgid >> 1) * RPSG;
    uint sn = (sgid & 1u) * (NCOLS / 2u);

    uint ib = (in_off >> 2) * 4u + b * inH * inW * kf;
    uint wb = (w_off >> 2) * 4u;
    uint ob = (out_off >> 2) * 4u + b * mDim * nDim;

    simdgroup_float8x8 acc[NR][NB];
    for (uint i = 0u; i < NR; i++)
        for (uint j = 0u; j < NB; j++)
            acc[i][j] = simdgroup_float8x8(0.0f);

    if (MODE == 2u) {
        // ── hoist：装载整个挪出 taps×kb 两层循环（MMA 条数与累加链不变）──
        // 地址取循环起点（t=0, kb=0）——只回答「删掉装载指令能省多少」。
        simdgroup_float8x8 a[NR];
        for (uint i = 0u; i < NR; i++)
            simdgroup_load(a[i], f32v + ib + ((y * sH - padT) * inW + (x0 * sW - padL)) * kf + (sr + i * 8u) * (sW * kf), sW * kf);
        simdgroup_float8x8 b[NB];
        for (uint j = 0u; j < NB; j++)
            simdgroup_load(b[j], f32v + wb + n0 + sn + j * 8u, nDim);
        for (uint t = 0u; t < taps; t++) {
            for (uint kb = 0u; kb < kf; kb += 8u) {
                for (uint i = 0u; i < NR; i++)
                    for (uint j = 0u; j < NB; j++)
                        simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
    } else {
        uint astep = sW * kf;
        for (uint t = 0u; t < taps; t++) {
            uint ky = t / kW;
            uint kx = t - ky * kW;
            uint abase = ib + ((y * sH + ky - padT) * inW + (x0 * sW + kx - padL)) * kf;
            uint wbase = wb + t * kf * nDim;
            for (uint kb = 0u; kb < kf; kb += 8u) {
                simdgroup_float8x8 a[NR];
                simdgroup_float8x8 b[NB];
                if (MODE == 1u) {
                    // 基址与行距一起收进 1 KB（≈1.3 KB 全 L1 常驻）。
                    for (uint i = 0u; i < NR; i++)
                        simdgroup_load(a[i], f32v + ((abase + (sr + i * 8u) * astep + kb) & 0xFFu), 8u);
                    for (uint j = 0u; j < NB; j++)
                        simdgroup_load(b[j], f32v + ((wbase + kb * nDim + n0 + sn + j * 8u) & 0xFFu), 8u);
                } else {
                    for (uint i = 0u; i < NR; i++)
                        simdgroup_load(a[i], f32v + abase + (sr + i * 8u) * astep + kb, astep);
                    for (uint j = 0u; j < NB; j++)
                        simdgroup_load(b[j], f32v + wbase + kb * nDim + n0 + sn + j * 8u, nDim);
                }
                for (uint i = 0u; i < NR; i++)
                    for (uint j = 0u; j < NB; j++)
                        simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
    }

    // 分片落地（各臂共用——noepi 也要留着它，否则整个热循环会被**常量传播
    // 整段删掉**：acc 无副作用，编译器直接删空，量到 0.003 ms 的假数。
    // 这正是 [[compiler-ate-the-benchmark]] 那条坑。）
    for (uint i = 0u; i < NR; i++)
        for (uint j = 0u; j < NB; j++)
            simdgroup_store(acc[i][j], ost + (sr + i * 8u) * NCOLS + sn + j * 8u, NCOLS);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (MODE == 3u) {
        // noepi：**只**删尾声算术（act/bias/残差）与输出写，留下分片落地与
        // 一个 sink 防 DCE。⇒ 给「动尾声」的上界。
        if (tid == 0u) of[ob] = ost[0];
        return;
    }

    for (uint e = tid; e < 64u * NCOLS; e += NTH) {
        uint i = e / NCOLS;
        if (i < i_lo) continue;
        uint j = e % NCOLS;
        uint n = n0 + j;
        float v = ost[e];
        if (b_off != OFF_NONE) v += f32v[(b_off >> 2) * 4u + n];
        v = act1(v, act, c1, c2, c3);
        uint m = y * outW + x0 + i;
        if (r_off != OFF_NONE)
            v += f32v[(r_off >> 2) * 4u + b * mDim * nDim + m * nDim + n];
        of[ob + m * nDim + n] = v;
    }
}

#define M3_SIG                                                             \
    device float4* d [[buffer(0)]],                                        \
    device const uint* params [[buffer(1)]],                               \
    constant uint& p_off [[buffer(3)]],                                    \
    uint3 tid3 [[thread_position_in_threadgroup]],                         \
    uint3 wg [[threadgroup_position_in_grid]]

// NCOLS=64（对应生产 `n_conv8_mma3`）
kernel void p_m3_base(M3_SIG) {
    threadgroup float ost[64u * 64u];
    m3_body<64u, 0u>(d, params, p_off, tid3.x, wg, ost);
}
kernel void p_m3_small(M3_SIG) {
    threadgroup float ost[64u * 64u];
    m3_body<64u, 1u>(d, params, p_off, tid3.x, wg, ost);
}
kernel void p_m3_hoist(M3_SIG) {
    threadgroup float ost[64u * 64u];
    m3_body<64u, 2u>(d, params, p_off, tid3.x, wg, ost);
}
kernel void p_m3_noepi(M3_SIG) {
    threadgroup float ost[64u * 64u];
    m3_body<64u, 3u>(d, params, p_off, tid3.x, wg, ost);
}

// NCOLS=32（对应生产 `n_conv8_mma3_n32`）
kernel void p_m3n_base(M3_SIG) {
    threadgroup float ost[64u * 32u];
    m3_body<32u, 0u>(d, params, p_off, tid3.x, wg, ost);
}
kernel void p_m3n_small(M3_SIG) {
    threadgroup float ost[64u * 32u];
    m3_body<32u, 1u>(d, params, p_off, tid3.x, wg, ost);
}
kernel void p_m3n_hoist(M3_SIG) {
    threadgroup float ost[64u * 32u];
    m3_body<32u, 2u>(d, params, p_off, tid3.x, wg, ost);
}
kernel void p_m3n_noepi(M3_SIG) {
    threadgroup float ost[64u * 32u];
    m3_body<32u, 3u>(d, params, p_off, tid3.x, wg, ost);
}

// ── 候选（MODE 4）：n32 用 **4 个 simdgroup / 128 线程**，瓦片仍是 64×32 ──
//
// 与 `p_m3n_base` 的差别**只有 simdgroup 布局**：网格组数、覆盖矩形、尾块
// 锚点与掩码、参数、ost 尺寸全部一字不改；每个 simdgroup 从「16 行 × 16 列」
// 变成「32 行 × 16 列」，于是每条 MMA 的装载数从 2A+2B:4 = **1.00** 压回
// 2A... 这里是 4A+2B:8 = **0.75**，与 NCOLS=64 那条实例化同档。
//
// ⚠️ **它要 128 线程/组**，而生产 `KernelSet::dispatch` 恒给 256 ⇒ 落地需要
// 按内核名给线程组尺寸（`dispatch_with_tg` 已存在，见 pipeline.rs）。本探针
// 用 `dispatch_with_tg(..., 128)` 派发；**网格组数不变**，所以内核里
// `threadgroup_position_in_grid` 的解码不受影响。
kernel void p_m3n_half(M3_SIG) {
    threadgroup float ost[64u * 32u];
    m3_body<32u, 4u>(d, params, p_off, tid3.x, wg, ost);
}

// # `p_m3_half`：同一个「抬瓦片」搬到 NCOLS=64 上
//
// 64 宽当前是「16 行 × 32 列」= 2A+4B:8 = **0.75** 装载/MMA；抬到 4 组之后
// 每 simdgroup 是「32 行 × 32 列」= 4A+4B:16 = **0.50**——**比值砍掉三分之一**，
// 比 n32 那一档（1.00 → 0.75）省得更多。代价是累加器从 2×4=8 个分片涨到
// 4×4=16 个（每 lane 16 → 32 个 f32 寄存器），**这是本档唯一的风险**：
// `n_conv_dw` 那次教训正是「同一个改写多了几个寄存器就在生产里翻车」。
// 所以这一档**必须**和 `base` 同台量，不能只凭比值表推断。
//
// 读数规则同前：half ≪ base 才说明装载发射槽真的是瓶颈；若 half ≥ base，
// 就是寄存器压力/并行度先到顶了（那时 64 宽这条实例化照旧不动）。
kernel void p_m3_half(M3_SIG) {
    threadgroup float ost[64u * 64u];
    m3_body<64u, 4u>(d, params, p_off, tid3.x, wg, ost);
}
