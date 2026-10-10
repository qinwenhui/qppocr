// 探针：**MMA 的装载路径** —— 分片从 device 直读 vs 经 threadgroup 中转。
//
// ## 要回答的问题
//
// 全图的账压在同一个数上：
//
//   * 纯 ALU 的 f32 FMA 峰值（`gpualu2`，零访存独立 FMA 链）…… **3340 GFLOPS**
//   * 我们 f32 `simdgroup_matrix` 的孤立探针天花板…… **2204 ~ 3326**
//   * 生产的 rec 卷积钉在这个天花板上（~2200）…… 图上 §17.4 读作「已到顶」
//   * 我们的 fp16 MMA 探针（`p_mm_sg`）…… **1517 ~ 1535**
//
// 三件事对不上：
//
// ① **「到我们自己探针的天花板」≠「到硬件的天花板」**。2200/3340 = 66%。
//    `simdgroup_matrix` 走的就是同一条 FMA 流水线，正常的 MMA GEMM 应当在
//    85~95%。所以 §17.4 那个「已到顶」应该是**问题**，不是结论。
// ② **fp16 是反的**：Apple 的 fp16 FMA 是 f32 的 2×，我们测出 0.62×。
// ③ 两个图、几十个内核**齐刷刷**落在 36~66% —— 这是「一个共享病根」的签名，
//    不是几十个独立的病根。
//
// ## 被检验的假设
//
// 读遍全仓四个 MMA 内核（`n_conv8_mma.metal:146-153`、`n_conv8_mma3.metal:333,339`、
// 本文件前身 `probe_mm_sg.metal:60-65`），**A/B 分片全部是 `simdgroup_load`
// 从 device 内存直读的**，`threadgroup` 只用在**尾声写回**。而 Apple GPU 的
// 标准快路（Apple 自己的 GEMM 样例）是：
//
//     协作式向量化拷贝 A/B 瓦片 → threadgroup → barrier
//           → `simdgroup_load` 从 threadgroup 取分片 → MMA
//
// 差别在于：device 直读时每个 K 步的 6 条分片装载都是**全延迟的 gathered
// 访存**，K 循环没有寄存器里的软件流水，流水线每隔几步就等内存。
//
// ## 反证据（必须摆出来，不许靠推理定案）
//
// 图上 §4 的 batch 扫描「加 16 倍常驻 warp 一分钱不赚」——**纯延迟说本该被
// 更多 warp 救**。同图 §4 还记着上一条被我推理出来、又因为「期待它成立而
// 放过反例」的假说（那个 364 GB/s 的「权重带宽墙」）。所以这条**必须直测**。
//
// ## 本文件的臂
//
// | 臂 | dtype | 分片来源 | 缓冲 |
// |---|---|---|---|
// | `p_mm_dev`     | half | device      | 单 |
// | `p_mm_tg`      | half | threadgroup | 单 |
// | `p_mm_tg2`     | half | threadgroup | 双（预取下一块） |
// | `p_mm_f32_dev` | float| device      | 单 |
// | `p_mm_f32_tg2` | float| threadgroup | 双 |
//
// `p_mm_dev` / `p_mm_f32_dev` 是**同 run 内的对照**（与 `p_mm_sg` /
// `n_conv8_mma` 同构），没有它们就分不清「变快」是改动的功劳还是机器的漂移。
//
// 端点与 `probe_mm_sg.metal` 完全一致：A `[M,K]` 行主序、W `[N,K]` 行主序、
// C `[M,N]` 行主序；M/N 是 64 的倍数、K 是 8 的倍数；256 线程 = 8 个
// simdgroup，64×64 瓦片，每 simdgroup 认领 16 行 × 32 列。
//
// ⚠️ 探针只**否证**不拍板（memory `probe-wins-dont-transfer-to-production`）：
// 这里赢了也必须回生产用 `env_ab.sh` 交错复测才算数。
//
// ===========================================================================
// 实测结果（2026-10-10）——**这个探针的「赢」是假的，别照着它改生产**
// ===========================================================================
//
// 五臂同 run 对照（`microbench::mm_tg_scan`，det 两个形状、best-of-15）：
//
//   | 臂 | c512→1024 | c1024→512 |
//   |---|---|---|
//   | fp16 device  | 1494 | 1511 |
//   | fp16 tg 单缓冲 | **2182** | **2241** |
//   | fp16 tg 双缓冲 | 1854 | 1808 |
//   | f32  device  | 1437 | 1454 |
//   | f32  tg 单缓冲 | **1830** | **1873** |
//   | f32  tg 双缓冲 | 1555 | 1491 |
//
// 读起来像「全族 +27~46%」。**它是假的。**
//
// 把同一招搬到生产（`n_conv8_mma` → 加了 `n_conv8_mma_st`，同网格/同参数/
// 同尾声/同数学、只换装载路径，CPU 对拍 5 个形状**位级一致 0.000e0**），
// 六个生产形状上实测 **0.55 ~ 0.75×（慢 25~45%）**：
//
//   det m32640 c128->256  2209.9 → 1235.0 GF (0.559×)
//   det 34x60 c1024->512  2467.1 → 1358.8 GF (0.551×)
//   det 34x60 c512->1024  2421.1 → 1446.6 GF (0.597×)
//   det m32640 c32->64    1033.4 →  712.2 GF (0.689×)
//   rec 6x68  c256->256   1602.3 → 1203.4 GF (0.751×)
//   rec 6x208 c192->384   2040.4 → 1276.4 GF (0.626×)
//
// ## 探针为什么会骗人 —— **B 操作数的转置**
//
// 看那一列数就现形了：**生产的 device 直读本来就有 2209~2467 GFLOPS**，
// 比本探针「中转」臂的 1830 还高。所以探针的 device 臂（1437~1510）不是
// 「生产的 device 直读」，它比生产**慢 1.5 倍**——那 1.5 倍是探针自己塞进去的。
//
// 差在哪：本文件按竞品 `mm_sg_dd` 的接口走，**B 是 `[N,K]` 且用
// `transpose = true` 取分片** —— 那是一次**跨步 gather**，本来就慢，
// 而中转（低延迟线程组里的跨步读）恰好把这次 gather 的代价抹掉。
// 生产的 B 是 `[k][n]`（K×N、行 stride = nDim），**正是 `simdgroup_load`
// 想要的朝向，不转置、不 gather**，所以生产根本没有这个缺陷可修，
// 中转只剩「多一次协作拷贝 + 多两道屏障 + 8 KB 线程组占用」的净亏。
//
// ⚠️ **这是一类新的探针陷阱**：`probe-wins-dont-transfer-to-production`
// 记的是「探针**剥掉了**生产里存在的成本」；这里是「探针**加上了**生产里
// 不存在的成本」——探针在**被自己弄坏的基线**上对照，赢的是修自己捅的娄子。
// 判据：**大赢先查基线臂**。同一形状下如果生产内核的孤立读数已经超过探针的
// 「优化」臂，那这个探针量的一定不是它自己声称的东西。
//
// 仍然成立的两条副产物：
//   · **双缓冲在两个 dtype 上都慢 15~20%** —— Apple 教科书的双缓冲配方在本
//     结构下不划算（暂存翻倍到顶 → 常驻线程组数掉）。这是真结论，但生产里
//     没有中转，所以也用不上。
//   · fp16 中转 2182~2241 ≈ 生产 f32 device 直读 2209~2467 ⇒ **fp16 相对
//     我们的 f32 没有净胜**，与 memory `fp16-dead-after-mma` 一致。
//
// 生产的 `n_conv8_mma` 孤立天花板因此是 **2209~2467 GFLOPS = f32 ALU 峰值的
// 66~74%**（长 K 的 det 形状）—— 这才是拿它当尺子的那个数。

#include <metal_stdlib>
using namespace metal;

// ==================== fp16 ====================

constant uint BK_H = 32u; // fp16 K 块：64×32 half = 4 KB

// 单缓冲：拷瓦片 → barrier → 分片从 threadgroup 取。
kernel void p_mm_tg(device const half *x [[buffer(0)]],
                    device const half *w [[buffer(1)]],
                    device half *o [[buffer(2)]],
                    constant uint *p [[buffer(3)]],
                    uint2 wg [[threadgroup_position_in_grid]],
                    uint2 tp [[thread_position_in_threadgroup]]) {
    const uint M = p[0], N = p[1], K = p[2];
    const uint tid = tp.x;
    const uint sgid = tid >> 5;

    const uint m0 = wg.x * 64u + (sgid >> 1) * 16u;
    const uint n0 = wg.y * 64u + (sgid & 1u) * 32u;

    threadgroup half as[BK_H * 64u];
    threadgroup half bs[BK_H * 64u];

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint kb0 = 0u; kb0 < K; kb0 += BK_H) {
        // 协作拷贝：64 行 × BK 列，256 线程各摊 8 个 half（= 2 个 half4）。
        // `i % BK` / `i / BK` 的除法是常数位宽，编译器出移位。
        for (uint i = tid; i < 64u * BK_H; i += 256u) {
            const uint r = i >> 5u, c = i & 31u;
            as[r * BK_H + c] = x[(ulong)(wg.x * 64u + r) * K + kb0 + c];
            bs[r * BK_H + c] = w[(ulong)(wg.y * 64u + r) * K + kb0 + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < BK_H; kk += 8u) {
            simdgroup_half8x8 a0, a1, b0, b1, b2, b3;
            simdgroup_load(a0, as + ((sgid >> 1) * 16u) * BK_H + kk, BK_H);
            simdgroup_load(a1, as + ((sgid >> 1) * 16u + 8u) * BK_H + kk, BK_H);
            simdgroup_load(b0, bs + ((sgid & 1u) * 32u + 0u) * BK_H + kk, BK_H, ulong2(0, 0), true);
            simdgroup_load(b1, bs + ((sgid & 1u) * 32u + 8u) * BK_H + kk, BK_H, ulong2(0, 0), true);
            simdgroup_load(b2, bs + ((sgid & 1u) * 32u + 16u) * BK_H + kk, BK_H, ulong2(0, 0), true);
            simdgroup_load(b3, bs + ((sgid & 1u) * 32u + 24u) * BK_H + kk, BK_H, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float ost[64 * 64];
    const uint r0 = (sgid >> 1) * 16u;
    const uint c0 = (sgid & 1u) * 32u;
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++)
            simdgroup_store(acc[i][j], ost + (r0 + i * 8u) * 64u + c0 + j * 8u, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint mb = wg.x * 64u, nb = wg.y * 64u;
    for (uint i = tid; i < 64u * 64u; i += 256u)
        o[(ulong)(mb + i / 64u) * N + nb + (i % 64u)] = half(ost[i]);
}

// 双缓冲：迭代 i 里**先发下一块的协作拷贝**，再算当前块 —— 拷贝的延迟被
// MMA 盖住。屏障只放在「算完之后」，让下一轮一定能看到新数据。
kernel void p_mm_tg2(device const half *x [[buffer(0)]],
                     device const half *w [[buffer(1)]],
                     device half *o [[buffer(2)]],
                     constant uint *p [[buffer(3)]],
                     uint2 wg [[threadgroup_position_in_grid]],
                     uint2 tp [[thread_position_in_threadgroup]]) {
    const uint M = p[0], N = p[1], K = p[2];
    const uint tid = tp.x;
    const uint sgid = tid >> 5;

    threadgroup half as[2][BK_H * 64u];
    threadgroup half bs[2][BK_H * 64u];

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    // 前导：块 0 进 buf 0
    for (uint i = tid; i < 64u * BK_H; i += 256u) {
        const uint r = i >> 5u, c = i & 31u;
        as[0][r * BK_H + c] = x[(ulong)(wg.x * 64u + r) * K + c];
        bs[0][r * BK_H + c] = w[(ulong)(wg.y * 64u + r) * K + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0u;
    for (uint kb0 = 0u; kb0 < K; kb0 += BK_H) {
        const uint nbuf = buf ^ 1u;
        if (kb0 + BK_H < K) {
            for (uint i = tid; i < 64u * BK_H; i += 256u) {
                const uint r = i >> 5u, c = i & 31u;
                as[nbuf][r * BK_H + c] = x[(ulong)(wg.x * 64u + r) * K + kb0 + BK_H + c];
                bs[nbuf][r * BK_H + c] = w[(ulong)(wg.y * 64u + r) * K + kb0 + BK_H + c];
            }
        }
        for (uint kk = 0u; kk < BK_H; kk += 8u) {
            simdgroup_half8x8 a0, a1, b0, b1, b2, b3;
            simdgroup_load(a0, as[buf] + ((sgid >> 1) * 16u) * BK_H + kk, BK_H);
            simdgroup_load(a1, as[buf] + ((sgid >> 1) * 16u + 8u) * BK_H + kk, BK_H);
            simdgroup_load(b0, bs[buf] + ((sgid & 1u) * 32u + 0u) * BK_H + kk, BK_H, ulong2(0, 0), true);
            simdgroup_load(b1, bs[buf] + ((sgid & 1u) * 32u + 8u) * BK_H + kk, BK_H, ulong2(0, 0), true);
            simdgroup_load(b2, bs[buf] + ((sgid & 1u) * 32u + 16u) * BK_H + kk, BK_H, ulong2(0, 0), true);
            simdgroup_load(b3, bs[buf] + ((sgid & 1u) * 32u + 24u) * BK_H + kk, BK_H, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = nbuf;
    }

    threadgroup float ost[64 * 64];
    const uint r0 = (sgid >> 1) * 16u;
    const uint c0 = (sgid & 1u) * 32u;
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++)
            simdgroup_store(acc[i][j], ost + (r0 + i * 8u) * 64u + c0 + j * 8u, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint mb = wg.x * 64u, nb = wg.y * 64u;
    for (uint i = tid; i < 64u * 64u; i += 256u)
        o[(ulong)(mb + i / 64u) * N + nb + (i % 64u)] = half(ost[i]);
}

// ==================== f32 ====================
//
// 与 `n_conv8_mma.metal:136-153` 同构（2×4 分片、8 simdgroup、64×64 瓦片），
// 只有分片来源不同。f32 的瓦片是 fp16 的两倍大，BK 取 16 以把双缓冲压在
// 16 KB 线程组内存以内。

constant uint BK_F = 16u; // float K 块：64×16 float = 4 KB

// f32 单缓冲：与 `p_mm_tg` 逐行同构，只换 dtype。**生产就是 f32**，
// 所以这一条才是决定「要不要改四个生产内核」的那一票。
kernel void p_mm_f32_tg(device const float *x [[buffer(0)]],
                        device const float *w [[buffer(1)]],
                        device float *o [[buffer(2)]],
                        constant uint *p [[buffer(3)]],
                        uint2 wg [[threadgroup_position_in_grid]],
                        uint2 tp [[thread_position_in_threadgroup]]) {
    const uint M = p[0], N = p[1], K = p[2];
    const uint tid = tp.x;
    const uint sgid = tid >> 5;

    threadgroup float as[BK_F * 64u];
    threadgroup float bs[BK_F * 64u];

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint kb0 = 0u; kb0 < K; kb0 += BK_F) {
        for (uint i = tid; i < 64u * BK_F; i += 256u) {
            const uint r = i >> 4u, c = i & 15u;
            as[r * BK_F + c] = x[(ulong)(wg.x * 64u + r) * K + kb0 + c];
            bs[r * BK_F + c] = w[(ulong)(wg.y * 64u + r) * K + kb0 + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < BK_F; kk += 8u) {
            simdgroup_float8x8 a0, a1, b0, b1, b2, b3;
            simdgroup_load(a0, as + ((sgid >> 1) * 16u) * BK_F + kk, BK_F);
            simdgroup_load(a1, as + ((sgid >> 1) * 16u + 8u) * BK_F + kk, BK_F);
            simdgroup_load(b0, bs + ((sgid & 1u) * 32u + 0u) * BK_F + kk, BK_F, ulong2(0, 0), true);
            simdgroup_load(b1, bs + ((sgid & 1u) * 32u + 8u) * BK_F + kk, BK_F, ulong2(0, 0), true);
            simdgroup_load(b2, bs + ((sgid & 1u) * 32u + 16u) * BK_F + kk, BK_F, ulong2(0, 0), true);
            simdgroup_load(b3, bs + ((sgid & 1u) * 32u + 24u) * BK_F + kk, BK_F, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float ost[64 * 64];
    const uint r0 = (sgid >> 1) * 16u;
    const uint c0 = (sgid & 1u) * 32u;
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++)
            simdgroup_store(acc[i][j], ost + (r0 + i * 8u) * 64u + c0 + j * 8u, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint mb = wg.x * 64u, nb = wg.y * 64u;
    for (uint i = tid; i < 64u * 64u; i += 256u)
        o[(ulong)(mb + i / 64u) * N + nb + (i % 64u)] = ost[i];
}

kernel void p_mm_f32_dev(device const float *x [[buffer(0)]],
                         device const float *w [[buffer(1)]],
                         device float *o [[buffer(2)]],
                         constant uint *p [[buffer(3)]],
                         uint2 wg [[threadgroup_position_in_grid]],
                         uint2 tp [[thread_position_in_threadgroup]]) {
    const uint M = p[0], N = p[1], K = p[2];
    const uint tid = tp.x;
    const uint sgid = tid >> 5;

    const uint m0 = wg.x * 64u + (sgid >> 1) * 16u;
    const uint n0 = wg.y * 64u + (sgid & 1u) * 32u;

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint kb = 0u; kb < K; kb += 8u) {
        simdgroup_float8x8 a0, a1, b0, b1, b2, b3;
        simdgroup_load(a0, x + (ulong)m0 * K + kb, K);
        simdgroup_load(a1, x + (ulong)(m0 + 8u) * K + kb, K);
        simdgroup_load(b0, w + (ulong)(n0 + 0u) * K + kb, K, ulong2(0, 0), true);
        simdgroup_load(b1, w + (ulong)(n0 + 8u) * K + kb, K, ulong2(0, 0), true);
        simdgroup_load(b2, w + (ulong)(n0 + 16u) * K + kb, K, ulong2(0, 0), true);
        simdgroup_load(b3, w + (ulong)(n0 + 24u) * K + kb, K, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
        simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
        simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
        simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
        simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
        simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
        simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
    }

    threadgroup float ost[64 * 64];
    const uint r0 = (sgid >> 1) * 16u;
    const uint c0 = (sgid & 1u) * 32u;
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++)
            simdgroup_store(acc[i][j], ost + (r0 + i * 8u) * 64u + c0 + j * 8u, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint mb = wg.x * 64u, nb = wg.y * 64u;
    for (uint i = tid; i < 64u * 64u; i += 256u)
        o[(ulong)(mb + i / 64u) * N + nb + (i % 64u)] = ost[i];
}

kernel void p_mm_f32_tg2(device const float *x [[buffer(0)]],
                         device const float *w [[buffer(1)]],
                         device float *o [[buffer(2)]],
                         constant uint *p [[buffer(3)]],
                         uint2 wg [[threadgroup_position_in_grid]],
                         uint2 tp [[thread_position_in_threadgroup]]) {
    const uint M = p[0], N = p[1], K = p[2];
    const uint tid = tp.x;
    const uint sgid = tid >> 5;

    threadgroup float as[2][BK_F * 64u];
    threadgroup float bs[2][BK_F * 64u];

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint i = tid; i < 64u * BK_F; i += 256u) {
        const uint r = i >> 4u, c = i & 15u;
        as[0][r * BK_F + c] = x[(ulong)(wg.x * 64u + r) * K + c];
        bs[0][r * BK_F + c] = w[(ulong)(wg.y * 64u + r) * K + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0u;
    for (uint kb0 = 0u; kb0 < K; kb0 += BK_F) {
        const uint nbuf = buf ^ 1u;
        if (kb0 + BK_F < K) {
            for (uint i = tid; i < 64u * BK_F; i += 256u) {
                const uint r = i >> 4u, c = i & 15u;
                as[nbuf][r * BK_F + c] = x[(ulong)(wg.x * 64u + r) * K + kb0 + BK_F + c];
                bs[nbuf][r * BK_F + c] = w[(ulong)(wg.y * 64u + r) * K + kb0 + BK_F + c];
            }
        }
        for (uint kk = 0u; kk < BK_F; kk += 8u) {
            simdgroup_float8x8 a0, a1, b0, b1, b2, b3;
            simdgroup_load(a0, as[buf] + ((sgid >> 1) * 16u) * BK_F + kk, BK_F);
            simdgroup_load(a1, as[buf] + ((sgid >> 1) * 16u + 8u) * BK_F + kk, BK_F);
            simdgroup_load(b0, bs[buf] + ((sgid & 1u) * 32u + 0u) * BK_F + kk, BK_F, ulong2(0, 0), true);
            simdgroup_load(b1, bs[buf] + ((sgid & 1u) * 32u + 8u) * BK_F + kk, BK_F, ulong2(0, 0), true);
            simdgroup_load(b2, bs[buf] + ((sgid & 1u) * 32u + 16u) * BK_F + kk, BK_F, ulong2(0, 0), true);
            simdgroup_load(b3, bs[buf] + ((sgid & 1u) * 32u + 24u) * BK_F + kk, BK_F, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = nbuf;
    }

    threadgroup float ost[64 * 64];
    const uint r0 = (sgid >> 1) * 16u;
    const uint c0 = (sgid & 1u) * 32u;
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++)
            simdgroup_store(acc[i][j], ost + (r0 + i * 8u) * 64u + c0 + j * 8u, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint mb = wg.x * 64u, nb = wg.y * 64u;
    for (uint i = tid; i < 64u * 64u; i += 256u)
        o[(ulong)(mb + i / 64u) * N + nb + (i % 64u)] = ost[i];
}
