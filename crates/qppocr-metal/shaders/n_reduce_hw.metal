// 空间均值阶段 1（NHWC f32，axes={2,3}）：out[c] = Σ_m x[m*C + c] 的
// m 分块部分和（f32 scratch[mb][Cpad]）。
//
// 访存：256 线程 = (256/cp4) 个位置组 × cp4 个通道组——每步 16 个
// 连续位置 × 连续通道 float4 = 2KB 连续读（曾按 C 步长逐线程扫，
// 128B 间隔带宽放大 64×、单次归约 3.4ms）。
// mDim 每块 1024 位置；Cpad ≤ 1024（cp4 ≤ 256）。cp4 不整除 256 时
//（如 small 的 SE 通道 384 → cp4=96、npg=2）尾线程 pg≥npg：**必须
// 跳过 m 循环**——它们的页序列与 pg=0 重叠会双计；red[t] 写 0 参与
// barrier（不能提前 return，屏障发散是 UB）。
// scratch 用 f32（buffer 0 视图）写——同为 f32 区（不相交）。
// 与 qppocr-gpu/shaders/n_reduce_hw.comp 逐行对应——改语义必须两边同步。
#include <metal_stdlib>
using namespace metal;

kernel void n_reduce_hw(
    device float* data [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    device float4* d [[buffer(2)]],
    constant uint& p_off [[buffer(3)]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    threadgroup float4 red[256];

    uint P = p_off;
    uint in_off = params[P + 0], scratch_off = params[P + 1];
    uint mDim = params[P + 2], cDim = params[P + 3];

    uint cp4 = cDim >> 2;
    uint t = lid3.x;
    uint c4 = t % cp4;
    uint pg = t / cp4;
    uint npg = 256 / cp4;
    uint nb_idx = wg.y; // 批维：每批独立归约
    if (params[0] != 0u && nb_idx >= params[0]) return; // real_n 早退：批维补齐的空行零算力；0=不限（裸 dispatch 安全默认）
    uint in_base = (in_off >> 2) + nb_idx * mDim * cp4;
    uint m_blocks_total = (mDim + 1023u) / 1024u; // scratch 行距
    uint m0 = wg.x * 1024u;
    uint mEnd = min(m0 + 1024u, mDim);

    float4 acc = float4(0.0);
    if (pg < npg) {
        for (uint m = m0 + pg; m < mEnd; m += npg) {
            acc += d[in_base + m * cp4 + c4];
        }
    }
    red[t] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (pg == 0) {
        float4 s = red[t];
        for (uint g = 1u; g < npg; g++) {
            s += red[c4 + g * cp4];
        }
        uint sb = scratch_off + (nb_idx * m_blocks_total + wg.x) * cDim + c4 * 4u;
        data[sb + 0u] = s.x;
        data[sb + 1u] = s.y;
        data[sb + 2u] = s.z;
        data[sb + 3u] = s.w;
    }
}
