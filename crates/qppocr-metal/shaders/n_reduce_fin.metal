// 空间均值阶段 2：scratch[mBlocks][Cpad] f32 → gate[Cpad]（f32 float4 写）。
// 通道按 256 分组跨步（Cpad 可 > 256，如 small 的 SE 通道 384）：
// 每组全 WG 并行累加 + 经线程组内存拼 float4 写出（直接读改写 float4 有
// 跨线程 lane 竞争）。组数 = ceil(Cpad/256) 是 WG 一致值——组内 barrier
// 合法；Cpad %4==0 且组界 4 对齐（分配保证），末组的 float4 不越界。
// 与 qppocr-gpu/shaders/n_reduce_fin.comp 逐行对应——改语义必须两边同步。
#include <metal_stdlib>
using namespace metal;

kernel void n_reduce_fin(
    device float* data [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    device float4* d [[buffer(2)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup float accs[256];

    uint P = p_off;
    uint scratch_off = params[P + 0], out_off = params[P + 1];
    uint mBlocks = params[P + 2], cDim = params[P + 3], mDim = params[P + 4];

    uint nb_idx = gid3.x / 256u; // grid.x=nb，每 WG 一批
    if (params[0] != 0u && nb_idx >= params[0]) return; // real_n 早退：批维补齐的空行零算力；0=不限（裸 dispatch 安全默认）
    for (uint c0 = 0u; c0 < cDim; c0 += 256u) {
        uint c = c0 + lid;
        float acc = 0.0;
        if (c < cDim) {
            for (uint b = 0u; b < mBlocks; b++) {
                acc += data[scratch_off + (nb_idx * mBlocks + b) * cDim + c];
            }
        }
        accs[lid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 每 4 通道一线程拼 float4 写出（cDim 为 %4 对齐——分配保证）
        if (lid % 4u == 0u && c0 + lid + 4u <= cDim) {
            float4 v = float4(accs[lid], accs[lid + 1u], accs[lid + 2u], accs[lid + 3u]);
            d[(out_off >> 2) + nb_idx * (cDim >> 2) + ((c0 + lid) >> 2)] =
                v / float(mDim);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
