// 批量动态 GEMM（注意力头）：out[batch, M, N] = A[batch, M, K] ×
// B[batch, K, N]，全动态（QK^T 与 scores×V）。
// 精确连续存储（无通道 pad）。一线程跨步算一 WG 一输出行：M、N ≤ 数百
//（T 量级）、K=15 或 T——朴素点积足够（张量小、launch 主导）。
//   params: a_off, b_off, out_off, m, k, n, batch, heads
// batch 维 = B×heads；real_n 早退按 B（params[0]×heads 之外空批行）。
// 累加序 = k 升序标量（与 CPU MatMul 一致）。
// 与 qppocr-gpu/shaders/n_attn_mm.comp 逐行对应——改语义必须两边同步。
// GLSL 的 `precise` 由本 crate 的禁 fast-math 编译选项承担（不收缩
// a*b+= 成 fma——CPU MatMul 是标量 mul+add 不收缩；收缩曾以 ~1e-4
// 概率在近阈值 CTC 候选翻转单字）。
#include <metal_stdlib>
using namespace metal;

kernel void n_attn_mm(
    device float* data [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 wg [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    uint P = p_off;
    uint a_off = params[P + 0], b_off = params[P + 1], out_off = params[P + 2];
    uint m = params[P + 3], k = params[P + 4], n = params[P + 5];
    uint batch = params[P + 6], heads = params[P + 7];

    // 一 WG = batch × M 的一行（grid.y = batch*M）
    uint row = wg.y;
    uint b = row / m, mi = row % m;
    // real_n 早退：params[0]=真实 B（补齐空批行零算力；0=不限）
    if (params[0] != 0u && b >= params[0] * heads) return;

    uint a_base = a_off + (b * m + mi) * k;
    uint o_base = out_off + (b * m + mi) * n;
    for (uint j = lid; j < n; j += 256u) {
        float acc = 0.0;
        uint b_base = b_off + (b * k) * n + j; // B[b, ki, j] 行主序
        for (uint ki = 0u; ki < k; ki++) {
            acc += data[a_base + ki] * data[b_base + ki * n];
        }
        data[o_base + j] = acc;
    }
}
