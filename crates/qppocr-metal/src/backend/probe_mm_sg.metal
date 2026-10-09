// 探针：**竞品结构的 GEMM** —— 64×64 瓦片 + `simdgroup_matrix` MMA。
//
// ## 为什么要有这条
//
// 本仓 `conv8_variant_scan` 把 n_conv8 的单因子路径穷举完了，结论是：
//
//   * `p_fmaonly`（删掉全部 load）**3.2~3.4×**  —— ALU 够快；
//   * `p_wconst`（只删权重 load）2.35×；
//   * `p_wsmall`（权重流量归零、指令数不变）只值 1.56×；
//   * `p_wstage`（线程组暂存权重）**0.71×** —— 屏障成本吃掉收益；
//   * m8/m16 大瓦片全部更慢 —— 累加器 = `m×8` 个 float4，寄存器文件爆。
//
// 于是留下的矛盾是：**瓶颈是 load 指令发射，而提高复用（= 减少 load）
// 的所有手段都撞在寄存器文件上**。
//
// 竞品绕开它的办法是换**累加器的排布**：`simdgroup_float8x8` 把一个 8×8
// 结果摊在 32 个 lane 上（每 lane 2 个数），于是 64×64 的瓦片每 lane 只占
// `2 行分片 × 4 列分片 × 8 寄存器 = 64` 个寄存器，而 m16 标量瓦片要
// `16×8 = 128` 个 float4。**瓦片能开到 64 行，权重就是被复用 64 次而不是
// 4 次**，`load 指令 / FMA` 从 12:8 掉到远小于 1。
//
// 这条探针只量**结构**：A/W 用理想布局（A 行主序、W n-主序），
// 与 n_conv8 现产布局无关。要回答的是「值不值得为这个结构重写取数路径」。
//
// ## 端点（与 `mm_sg_dd` 同形）
//
// A: half [M, K] 行主序（M 是输出位置，K 是归约维）
// W: half [N, K] n-主序（`simdgroup_load` 带转置取 8(K)×8(N) 分片）
// C: half [M, N] 行主序
// M/N 必须是 64 的倍数、K 必须是 8 的倍数（探针自己补齐，生产要掩码）。
//
// 线程组 256 线程 = 8 个 simdgroup；瓦片 64 行 × 64 列，每个 simdgroup
// 认领 16 行 × 32 列（`acc[2][4]`，与竞品 `mm_sg` 一致）。

#include <metal_stdlib>
using namespace metal;

kernel void p_mm_sg(device const half *x [[buffer(0)]],
                    device const half *w [[buffer(1)]],
                    device half *o [[buffer(2)]],
                    constant uint *p [[buffer(3)]],
                    uint2 wg [[threadgroup_position_in_grid]],
                    uint2 tp [[thread_position_in_threadgroup]]) {
    const uint M = p[0], N = p[1], K = p[2];
    const uint tid = tp.x;

    const uint sgid = tid >> 5;
    // 8 个 simdgroup 铺成 4 行块 × 2 列块 → 64 行 × 32 列（每块 16×32）
    const uint m0 = wg.x * 64u + (sgid >> 1) * 16u;
    const uint n0 = wg.y * 64u + (sgid & 1u) * 32u;

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    // K 步长 8：一条 `simdgroup_load` 换一条 8×8 分片，
    // 一条 `simdgroup_multiply_accumulate` 做 8×8×8 = 512 次 MAC。
    for (uint kb = 0u; kb < K; kb += 8u) {
        simdgroup_half8x8 a0, a1, b0, b1, b2, b3;
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

    // 尾声：分片落到线程组 fp32 缓冲，再整体写成 half（与 mm_sg 同形）。
    threadgroup float ost[64 * 64];
    const uint r0 = (sgid >> 1) * 16u;
    const uint c0 = (sgid & 1u) * 32u;
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++)
            simdgroup_store(acc[i][j], ost + (r0 + i * 8u) * 64u + c0 + j * 8u, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint mb = wg.x * 64u, nb = wg.y * 64u;
    for (uint i = tid; i < 64u * 64u; i += 256u) {
        o[(ulong)(mb + i / 64u) * N + nb + (i % 64u)] = half(ost[i]);
    }
}
