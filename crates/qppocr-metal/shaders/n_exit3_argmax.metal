// rank-3 出口的 argmax 变体（rec 的 CTC 头）：softmax 输出 [rows × cpad]
// f32，每行取**首个严格最大**（平局取最小下标——与 CPU ctc_decode 的
// 逐元素 `>` 首胜语义逐字等价），写 (val, idx) 对（每行 2 word：
// val + idx 的位型）。
// 读回从 T×V×4 B/行（~3 MB）降到 8 B/行——CTC 只吃 argmax 路径，
// 全量概率矩阵回主机纯属带宽浪费。
//   params: in_off, out_off, rows, cols, cpad, rpb（rows/批，空行早退）
// 与 qppocr-gpu/shaders/n_exit3_argmax.comp 逐行对应——改语义必须两边同步。
#include <metal_stdlib>
using namespace metal;

kernel void n_exit3_argmax(
    device float* data [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 wg [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup float2 red[256]; // (max 值, 持有者最小下标——位型存 float)

    uint P = p_off;
    uint in_off = params[P + 0], out_off = params[P + 1];
    uint rows = params[P + 2], cols = params[P + 3], cpad = params[P + 4];
    uint rpb = params[P + 5]; // rows/批（rows=N_pad*T；批是外维）

    uint row = wg.x;
    if (row >= rows) return;
    if (params[0] != 0u && row >= params[0] * rpb) return; // real_n 早退：批维补齐的空行零算力；0=不限（裸 dispatch 安全默认）
    uint base = in_off + row * cpad;
    uint sgw = threads_per_simdgroup;
    uint nsg = 256u / sgw;
    uint sg = lid / sgw;

    // 1) 线程内跨步扫描：首个严格最大（v > m 才换——平局留先者）
    float m = -3.4e38;
    uint mi = 0u;
    for (uint j = lid; j < cols; j += 256u) {
        float v = data[base + j];
        if (v > m) {
            m = v;
            mi = j;
        }
    }
    // 2) 子组归约：值取 max；等值线程里取最小下标
    float wm = simd_max(m);
    uint cand = (m == wm) ? mi : 0xFFFFFFFFu;
    uint wmi = simd_min(cand);
    if (simd_is_first()) red[sg] = float2(wm, as_type<float>(wmi));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 3) 跨子组（softmax 同款尾归约）：值优先、平局最小下标
    if (lid < nsg && lid % 4u == 0u) {
        float v = red[lid].x;
        uint ix = as_type<uint>(red[lid].y);
        for (uint k = 1u; k < 4u && lid + k < nsg; k++) {
            float v2 = red[lid + k].x;
            uint ix2 = as_type<uint>(red[lid + k].y);
            if (v2 > v || (v2 == v && ix2 < ix)) {
                v = v2;
                ix = ix2;
            }
        }
        red[lid / 4u] = float2(v, as_type<float>(ix));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float bv = red[0].x;
    uint bi = as_type<uint>(red[0].y);
    for (uint g = 1u; g < (nsg + 3u) / 4u; g++) {
        float v2 = red[g].x;
        uint ix2 = as_type<uint>(red[g].y);
        if (v2 > bv || (v2 == bv && ix2 < bi)) {
            bv = v2;
            bi = ix2;
        }
    }
    if (lid == 0u) {
        data[out_off + row * 2u] = bv;
        data[out_off + row * 2u + 1u] = as_type<float>(bi);
    }
}
