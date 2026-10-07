// 出口转换：f32 NHWC [N,H,W,Cpad4] → f32 NCHW [N,C,H,W]，可选 act
//（0=无，4=sigmoid——det 的图输出 Sigmoid 在此融合，省一整趟读写）。
// pad 通道不写出。
// 与 qppocr-gpu/shaders/n_exit.comp 逐行对应——改语义必须两边同步。
#include <metal_stdlib>
using namespace metal;

kernel void n_exit(
    device float* data [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    device float4* d [[buffer(2)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]])
{
    uint P = p_off;
    uint in_off = params[P + 0];   // word 偏移（f32）
    uint out_off = params[P + 1];  // float 元素偏移
    uint nb = params[P + 2], c = params[P + 3], h = params[P + 4], w = params[P + 5];
    uint cpad = params[P + 6], act = params[P + 7];
    uint cp4 = cpad >> 2;

    uint hw = h * w;
    uint gid = gid3.x;
    if (gid >= nb * hw) return;
    uint n = gid / hw, pos = gid - n * hw;
    uint plane = c * hw;
    for (uint c4 = 0u; c4 < cp4; c4++) {
        float4 v = d[(in_off >> 2) + (n * hw + pos) * cp4 + c4];
        uint base = c4 * 4u;
        if (base + 0u < c) data[out_off + n * plane + (base + 0u) * hw + pos] = act == 4u ? 1.0 / (1.0 + exp(-v.x)) : v.x;
        if (base + 1u < c) data[out_off + n * plane + (base + 1u) * hw + pos] = act == 4u ? 1.0 / (1.0 + exp(-v.y)) : v.y;
        if (base + 2u < c) data[out_off + n * plane + (base + 2u) * hw + pos] = act == 4u ? 1.0 / (1.0 + exp(-v.z)) : v.z;
        if (base + 3u < c) data[out_off + n * plane + (base + 3u) * hw + pos] = act == 4u ? 1.0 / (1.0 + exp(-v.w)) : v.w;
    }
}
