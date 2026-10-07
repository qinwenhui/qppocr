// 最近邻上/下采样（NHWC f32）：iy = trunc(oy * sy) 钳 [0,H-1]。
// sy/sx 由主机端 f64 预计算入参（P+7/P+8）——内核只乘不除：GPU 浮点
// 除法可能走倒数近似，trunc(oy*H/oh) 在整除边界偶发少 1（Arc Pro 实测
// 曾致 det 图 29/60 行错位）；整数除法版正确但慢。乘法版：2 的幂比例
// f32 精确，其余 ≤0.5ulp（min 钳兜底）。
// 与 qppocr-gpu/shaders/n_resize.comp 逐行对应——改语义必须两边同步。
#include <metal_stdlib>
using namespace metal;

kernel void n_resize(
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
    float sy = as_type<float>(params[P + 7]);
    float sx = as_type<float>(params[P + 8]);

    uint b = wg.y;
    uint cv4 = cDim >> 2;
    uint gid = gid3.x;
    uint tiles = mDim * cv4;
    if (gid >= tiles) return;
    uint c4 = gid % cv4;
    uint m = gid / cv4;
    uint oy = m / outW, ox = m - oy * outW;
    uint ib = (in_off >> 2) + b * inH * inW * cv4;
    uint ob = (out_off >> 2) + b * mDim * cv4;

    uint iy = min(uint(float(oy) * sy), inH - 1u);
    uint ix = min(uint(float(ox) * sx), inW - 1u);
    d[ob + m * cv4 + c4] = d[ib + (iy * inW + ix) * cv4 + c4];
}
