// 池化（NHWC f32，max/avg 通用带 pad）：越界跳过；avg 除满核面积。
// 一线程一 (输出位置, 通道组 c4)。
// 与 qppocr-gpu/shaders/n_pool.comp 逐行对应——改语义必须两边同步。
#include <metal_stdlib>
using namespace metal;

kernel void n_pool(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    uint P = p_off;
    uint in_off = params[P + 0], out_off = params[P + 1];
    uint mDim = params[P + 2], cDim = params[P + 3], outW = params[P + 4];
    uint inW = params[P + 5], inH = params[P + 6];
    uint kh = params[P + 7], kw = params[P + 8];
    uint sH = params[P + 9], sW = params[P + 10];
    uint padT = params[P + 11], padL = params[P + 12];
    uint isMax = params[P + 13];

    uint b = wg.y;
    if (params[0] != 0u && b >= params[0]) return; // real_n 早退：批维补齐的空行零算力；0=不限（裸 dispatch 安全默认）
    uint cv4 = cDim >> 2;
    uint gid = gid3.x;
    uint tiles = mDim * cv4;
    if (gid >= tiles) return;
    uint c4 = gid % cv4;
    uint m = gid / cv4;
    uint oh = m / outW, ow = m - oh * outW;
    uint ib = (in_off >> 2) + b * inH * inW * cv4;
    uint ob = (out_off >> 2) + b * mDim * cv4;

    float4 acc = isMax != 0u ? float4(-3.4e38) : float4(0.0);
    for (uint ky = 0u; ky < kh; ky++) {
        int iy = int(oh * sH + ky) - int(padT);
        if (iy < 0 || iy >= int(inH)) continue;
        for (uint kx = 0u; kx < kw; kx++) {
            int ix = int(ow * sW + kx) - int(padL);
            if (ix < 0 || ix >= int(inW)) continue;
            float4 v = d[ib + (uint(iy) * inW + uint(ix)) * cv4 + c4];
            if (isMax != 0u) acc = max(acc, v); else acc += v;
        }
    }
    if (isMax == 0u) acc /= float(kh * kw);
    d[ob + m * cv4 + c4] = acc;
}
