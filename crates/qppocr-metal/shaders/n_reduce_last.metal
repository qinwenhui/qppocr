// 末轴均值（rank-3 [B,T,C] 的 LayerNorm 分解链，ReduceMean axes 归一化
// 到最后一维）：每行 sum(cols) → mean，写出 [rows,4]（lane0=均值、旁
// lane 写 0——下游行向量广播 8/11 只读 lane0）。
// 一 WG（256 线程）一行：simdgroup 归约 sum（softmax 同款骨架）。
//   params: in_off, out_off, rows, cols, cpad（存储列数）, rpb（rows/批）
// 求和序 = 子组树归约（CPU 的逐元素序不同——GPU↔CPU 浮点序差异类）。
// 与 qppocr-gpu/shaders/n_reduce_last.comp 逐行对应——改语义必须两边同步。
#include <metal_stdlib>
using namespace metal;

kernel void n_reduce_last(
    device float* data [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 wg [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup float red[256];

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

    float s = 0.0;
    for (uint j = lid; j < cols; j += 256u) {
        s += data[base + j];
    }
    float ws = simd_sum(s);
    if (simd_is_first()) red[sg] = ws;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < nsg && lid % 4u == 0u) {
        float v = red[lid];
        if (lid + 1u < nsg) v += red[lid + 1u];
        if (lid + 2u < nsg) v += red[lid + 2u];
        if (lid + 3u < nsg) v += red[lid + 3u];
        red[lid / 4u] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowsum = red[0];
    for (uint g = 1u; g < (nsg + 3u) / 4u; g++) {
        rowsum += red[g];
    }
    if (lid == 0u) {
        data[out_off + row * 4u] = rowsum / float(cols);
        data[out_off + row * 4u + 1u] = 0.0;
        data[out_off + row * 4u + 2u] = 0.0;
        data[out_off + row * 4u + 3u] = 0.0;
    }
}
