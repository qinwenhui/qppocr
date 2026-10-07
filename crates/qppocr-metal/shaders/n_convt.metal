// ConvTranspose（NHWC f32，kH≤sH 且 kW≤sW——det 的 FPN 2×2 s2 上采样）：
// 每输出恰由一个 (ih,iw,ky,kx) tap 决定。ONNX 权重 [Ci,Co,kh,kw] 重排为
// 每 tap 一张 [Ci][nv] 的 float4 矩阵（元素 = 4 个输出通道）。
// act：0=无、1=gelu(c1,c2,c3)、2=relu。
// 与 qppocr-gpu/shaders/n_convt.comp 逐行对应——改语义必须两边同步。
#include <metal_stdlib>
using namespace metal;

constant uint OFF_NONE = 0xFFFFFFFFu;

// exp1/erf1：conv.comp 的逐字拷贝（与 CPU 位级一致的多项式，非硬件 exp）。
float exp1(float x) {
    const float LN2_HI = 0.6931472;
    const float LN2_LO = -2.9802322e-8;
    const float INV_LN2 = 1.442695;
    x = clamp(x, -88.0f, 88.0f);
    float kf = rint(x * INV_LN2);
    float r = fma(-kf, LN2_HI, x);
    r = fma(-kf, LN2_LO, r);
    float p = 1.0 / 720.0;
    p = fma(p, r, 1.0 / 120.0);
    p = fma(p, r, 1.0 / 24.0);
    p = fma(p, r, 1.0 / 6.0);
    p = fma(p, r, 0.5);
    p = fma(p, r, 1.0);
    p = fma(p, r, 1.0);
    int ki = int(kf) + 127;
    return as_type<float>(uint(ki) << 23) * p;
}

float erf1(float x) {
    float ax = fabs(x);
    float t = 1.0 / fma(0.3275911, ax, 1.0);
    float p = 1.0614054;
    p = fma(p, t, -1.453152027);
    p = fma(p, t, 1.421413741);
    p = fma(p, t, -0.284496736);
    p = fma(p, t, 0.2548296);
    p *= t;
    float e = exp1(-ax * ax);
    float r = fma(-p, e, 1.0);
    return as_type<float>(as_type<uint>(r) | (as_type<uint>(x) & 0x80000000u));
}

kernel void n_convt(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    uint P = p_off;
    uint in_off = params[P + 0], w_off = params[P + 1], b_off = params[P + 2];
    uint out_off = params[P + 3];
    uint mDim = params[P + 4], nDim = params[P + 5], outW = params[P + 6];
    uint inW = params[P + 7], inH = params[P + 8], cin = params[P + 9];
    uint ci4v = params[P + 10], sH = params[P + 11], sW = params[P + 12];
    uint kW = params[P + 13], act = params[P + 14];
    float c1 = as_type<float>(params[P + 15]);
    float c2 = as_type<float>(params[P + 16]);
    float c3 = as_type<float>(params[P + 17]);

    uint b = wg.y;
    uint nv = nDim >> 2;
    uint gid = gid3.x;
    uint tiles = mDim * nv;
    if (gid >= tiles) return;
    uint n4 = gid % nv;
    uint m = gid / nv;
    uint oh = m / outW, ow = m - oh * outW;
    uint ih = oh / sH, ky = oh - ih * sH;
    uint iw = ow / sW, kx = ow - iw * sW;
    uint tap = ky * kW + kx;
    uint ib = (in_off >> 2) + b * inH * inW * ci4v;
    uint wb = w_off >> 2;
    uint ob = (out_off >> 2) + b * mDim * nv;

    uint px = ih * inW + iw;
    float4 acc = float4(0.0);
    for (uint ci = 0u; ci < cin; ci++) {
        float xv = d[ib + px * ci4v + (ci >> 2)][ci & 3u];
        acc += xv * d[wb + tap * cin * nv + ci * nv + n4];
    }
    if (b_off != OFF_NONE) acc += d[(b_off >> 2) + n4]; // bias 全批共享（曾有 b*nv 越界）
    if (act == 2u) acc = max(acc, float4(0.0));
    else if (act == 1u) {
        float inv = 1.0 / c1;
        acc = c3 * acc * (float4(erf1(acc.x * inv), erf1(acc.y * inv),
                                 erf1(acc.z * inv), erf1(acc.w * inv)) + c2);
    }
    d[ob + m * nv + n4] = acc;
}
