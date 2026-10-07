// depthwise conv（NHWC f32，group=Ci=Co）：一线程一 (输出位置, 通道组 c4)，
// 逐 tap 边界检查。权重重排 [t][c4]：float4 = 4 个通道该 tap 的权重。
// act：0=无、1=gelu(c1,c2,c3)、2=relu。
// 与 qppocr-gpu/shaders/n_conv_dw.comp 逐行对应——改语义必须两边同步。
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

kernel void n_conv_dw(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    uint P = p_off;
    uint in_off = params[P + 0], w_off = params[P + 1], b_off = params[P + 2];
    uint out_off = params[P + 3];
    uint mDim = params[P + 4], cDim = params[P + 5], outW = params[P + 6];
    uint inW = params[P + 7], inH = params[P + 8];
    uint taps = params[P + 9], kW = params[P + 10];
    uint sH = params[P + 11], sW = params[P + 12];
    uint padT = params[P + 13], padL = params[P + 14];
    uint act = params[P + 15];
    float c1 = as_type<float>(params[P + 16]);
    float c2 = as_type<float>(params[P + 17]);
    float c3 = as_type<float>(params[P + 18]);
    uint r_off = params[P + 19]; // 残差（NHWC 同布局；OFF_NONE=无）

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
    uint wb = w_off >> 2;
    uint ob = (out_off >> 2) + b * mDim * cv4;

    float4 acc = float4(0.0);
    for (uint t = 0u; t < taps; t++) {
        uint ky = t / kW, kx = t - ky * kW;
        int ih = int(oh * sH + ky) - int(padT);
        int iw = int(ow * sW + kx) - int(padL);
        if (ih < 0 || ih >= int(inH) || iw < 0 || iw >= int(inW)) continue;
        uint px = uint(ih) * inW + uint(iw);
        float4 x = d[ib + px * cv4 + c4];
        float4 w = d[wb + t * cv4 + c4];
        acc += x * w;
    }
    if (b_off != OFF_NONE) acc += d[(b_off >> 2) + c4]; // bias 全批共享（曾有 b*cv4 越界）
    if (act == 2u) acc = max(acc, float4(0.0));
    else if (act == 1u) {
        float inv = 1.0 / c1;
        acc = c3 * acc * (float4(erf1(acc.x * inv), erf1(acc.y * inv),
                                 erf1(acc.z * inv), erf1(acc.w * inv)) + c2);
    }
    if (r_off != OFF_NONE) acc += d[(r_off >> 2) + b * mDim * cv4 + m * cv4 + c4];
    d[ob + m * cv4 + c4] = acc;
}
