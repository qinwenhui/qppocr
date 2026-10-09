// 探针：**基于 conv8 形状的 simdgroup_matrix MMA** —— 64×64 瓦片，适配 conv8 的 m4×n8 分块。
//
// ## 为什么要有这条
//
// 竞品的核心是 `simdgroup_float8x8` 累加器，让 Tile 能开到 64×64 而寄存器压力仍在 M4 的耐受范围内。
// 我们的 `conv8` 用的是 4×8 tile，但用的是普通 float4 累加器，寄存器数 = m×8 float4 = 4×8 = 32 个 float4 = 128 个 32 位寄存器，
// 刚好撞在 M4 的 register-file wall 上，没法再提高瓦片尺寸（窗口序 9 MB 权重复用时也没超过这堵墙）。
//
// 这条探针把 `conv8` 的热循环**直接替换成竞品的 MMA 结构**，但保留我们的取数路径（列主序、每个 thread 算 4 个 m0）。
// 回答：在我们现有的输入/权重布局和取数序列下，fp16 MMA 能跑多快？是不是真的需要重写所有取数路径？
//
// ## 映射规则（与 n_conv8_body 逐字对齐）
//
// - m0 = gid / nh * 4u → 4 个输出行，与原内核完全相同
// - n4 = gid % nh → 通道块，原内核的列主序映射（n4b = n4 + nh）仍保留
// - 每个 thread 算 (m0:4, n4:8) → 正好是 64×64 瓦片的 1/16：
//   每个 simdgroup 负责 16×32 = (4×4) × (8×4) = 16 个 thread，共 8 个 simdgroup/线程组。
//
// ## MMA 映射（与竞品 `p_mm_sg` 一致）
//
// - `acc[2][4]` = 2 个 m 分片 × 4 个 n 分片 = 8 个 8×8 累加器，对应：
//   m 分片 = m0 + i×8（每 thread 算 2 个 m=8 行）
//   n 分片 = n4 + j×8（每个分片算 8 个 n=8 块）
// - K 步长 8，每条 `simdgroup_load` 加载 8 个 K 元素的 A（行主序）/B（转置 n-主序）
// - 热循环内仅 1 次 `simdgroup_multiply_accumulate` per 分片，替代原内核的 32 条独立 FMA。

#include <metal_stdlib>
using namespace metal;

// conv8 的 m=4 行 → 适配 2×8 累加器分片：每 thread 取 m0 + 0,8 → 不对，我们仍用 m0+0,1,2,3，但累加器跨 8 行的 m 分片？
// 不，`conv8` 的 m 是 4，刚好对应 `simdgroup_float8x8` 里的 8？不对，应该是每 2 个 thread 合作算 8 行？
// 不对，原内核每 thread 算 4 行（m0 = rem % rows_g * 4），而 64 行瓦片需要 16 个 thread 沿 m 方向，所以 m 方向的 thread 数 = 64 / 4 = 16。
// 线程组沿 m 方向的 threadg 数 = 16 / 8 个 simdgroup → 2 个 simdgroup，刚好匹配 acc[2] = 2 个 m 分片。

kernel void p_mm_sg_conv8(device const half *x [[buffer(0)]],
                          device const half *w [[buffer(1)]],
                          device half *o [[buffer(2)]],
                          constant uint *p [[buffer(3)]],
                          uint2 wg [[threadgroup_position_in_grid]],
                          uint2 tp [[thread_position_in_threadgroup]]) {
    const uint M = p[0], N = p[1], K = p[2];
    const uint tid = tp.x + tp.y * 256; // tpg = (64/4)*(64/8) = 16*8 = 128 threads？不，原线程组是 256 = 8*32 simdgroups。
    // 重置：回到 conv8 的网格公式，每线程算 m=4，n=8，线程组 = (64/4)*(64/8) = 16*8 = 128？不，竞品是 256 线程 = 8 simdgroups = 8*32 = 256。
    // 重新匹配：conv8 一个 tile = (4 行) × (8 通道块) → 32 个 float4，对应：
    // simdgroup 算 (16 行) × (32 通道) → 4× 宽 tile，对应 4×8 thread，8 simdgroups = 32 线程？不对，直接用竞品的 256 线程，映射到 conv8 的 tile。

    const uint simd_id = tp.x >> 5; // 8 个 simdgroups/线程组
    const uint lane = tp.x & 31;
    // 64 行瓦片：m = wg.x * 64 + (simd_id >> 1) * 8 → 2 个 simdgroup 共享 16 行，对应原 conv8 的 4 个 thread（每个算 4 行）
    const uint m_base = wg.x * 64 + (simd_id >> 1) * 8;
    // 64 列瓦片：n = wg.y * 64 + (simd_id & 1) * 32 → 2 个 simdgroup 共享 64 列？不，n=8 通道块 → 32 个通道 = 4 个块
    const uint n_base = wg.y * 64 + (simd_id & 1) * 32;

    // 每个 lane 算 m = m_base + (lane >> 3)，n = n_base + (lane & 7)？不对，simdgroup_float8x8 是 8×8，所以 8×8 = 64 个元素/lane？
    // 直接用竞品的 simdgroup 映射：acc[2][4]，每个 acc 是 simdgroup_float8x8，对应 m=8, n=8。
    // 所以 total m = 2*8 = 16, n = 4*8 = 32，对应线程组内 8 simdgroups → 每个 simd 算 16×32。
    // 而 conv8 的 m 是 4 → 所以每个 m 行需要 2 个 thread 叠加？不，原 conv8 用的是 float4，这里直接用 half8，每 2 个 float4 = 1 half8。
    // 简化：按竞品的 ABI，但在 K 循环内模拟 conv8 的 CI4 展开，看看实际的取数与 MMA 结合的开销。

    simdgroup_float8x8 acc[2][4]; // m=2*8, n=4*8
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++) acc[i][j] = simdgroup_float8x8(0.0f);

    // K 步长 8，每条 load 对应 8 个 K，与 conv8 的 CI4 循环数对齐（K=512 → 64 次循环）
    for (uint kb = 0u; kb < K; kb += 8u) {
        simdgroup_half8x8 a0, a1, b0, b1, b2, b3;
        // A: [M,K] 行主序 → m_base + i*8, K=kb → 与原输入取数逐字对齐：原 conv8 每次 load g0-g3，这里 load a0-a1 对应 m=8 行
        simdgroup_load(a0, x + (m_base + 0u) * K + kb, K);
        simdgroup_load(a1, x + (m_base + 8u) * K + kb, K);
        // B: [N,K] n-主序，转置 → 与原权重取数的 n4 + j*8 对齐（原每次 load w[wRow + i*nv]）
        simdgroup_load(b0, w + (n_base + 0u) * K + kb, K, ulong2(0, 0), true);
        simdgroup_load(b1, w + (n_base + 8u) * K + kb, K, ulong2(0, 0), true);
        simdgroup_load(b2, w + (n_base + 16u) * K + kb, K, ulong2(0, 0), true);
        simdgroup_load(b3, w + (n_base + 24u) * K + kb, K, ulong2(0, 0), true);
        // MMA：每条 FMA 对应原内核的 4×4 FMA，这里 8×8×8 = 512 MAC，替代 32 条
        simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
        simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
        simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);
        simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
        simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
        simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);
        simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
    }

    // 尾声：存回 half，按 conv8 的 m=4 行分组 → 每 2 个 acc 分片对应 4 行，写到 m_base + (i/2)*4
    threadgroup float ost[64 * 64]; // 线程组暂存，写回 half 前的 fp32 累加
    const uint r0 = (simd_id >> 1) * 16u; // 每个 m 分片对应 8 行 → 2 个分片 = 16 行，对应 4 个 conv8 m=4 块
    const uint c0 = (simd_id & 1u) * 32u;
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < 4u; j++)
            simdgroup_store(acc[i][j], ost + (r0 + i * 8u) * 64u + c0 + j * 8u, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 最终写回：把 64×64 的 fp32 → half，按 conv8 的 m0 = tid / nh *4，n4 = tid % nh 映射
    const uint mb = wg.x * 64u, nb = wg.y * 64u;
    for (uint i = tid; i < 64u * 64u; i += 256u) {
        o[(ulong)(mb + i / 64u) * N + nb + (i % 64u)] = half(ost[i]);
    }
}
