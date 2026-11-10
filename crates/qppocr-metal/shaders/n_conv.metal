// 稠密 kxk conv（NHWC f32，group=1，任意 stride/pad）——m4×n4 dot 瓦片。
//
// 存储：激活 NHWC f32（行 = 空间位置，行距 cin4v 个 float4）；权重
// **k-major 重排**：元素 (t*cin4v + ci4)*4*nv + i*nv + n4 = float4，
// 装 4 个输出通道在 k=t*Ci+ci4*4+i 的权重（Ci/Co 均 pad 到 %4 补 0）。
// f32 累加；bias/act epilogue。每元素边界检查（pad 处贡献 0）。
// act：0=无、1=gelu(c1,c2,c3)、2=relu。
// 与 qppocr-gpu/shaders/n_conv.comp 逐行对应——改语义必须两边同步。
// 绑定：D32=0、Params=1；p_off 经 set_bytes（buffer 3）。
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

float4 apply_act(float4 v, uint act, float c1, float c2, float c3) {
    if (act == 2u) return max(v, float4(0.0));
    if (act == 1u) {
        float inv = 1.0 / c1;
        return c3 * v * (float4(erf1(v.x * inv), erf1(v.y * inv),
                                erf1(v.z * inv), erf1(v.w * inv)) + c2);
    }
    return v;
}

kernel void n_conv(
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
    uint inW = params[P + 7], inH = params[P + 8], cin4v = params[P + 9];
    uint taps = params[P + 10], kW = params[P + 11];
    uint sH = params[P + 12], sW = params[P + 13];
    uint padT = params[P + 14], padL = params[P + 15];
    uint act = params[P + 16];
    float c1 = as_type<float>(params[P + 17]);
    float c2 = as_type<float>(params[P + 18]);
    float c3 = as_type<float>(params[P + 19]);
    uint r_off = params[P + 20]; // 残差（NHWC 同布局；OFF_NONE=无）

    uint b = wg.y;
    if (params[0] != 0u && b >= params[0]) return; // real_n 早退：批维补齐的空行零算力；0=不限（裸 dispatch 安全默认）
    uint nv = nDim >> 2;
    uint gid = gid3.x;
    uint tiles = ((mDim + 3u) >> 2) * nv;
    if (gid >= tiles) return;
    uint n4 = gid % nv;
    uint m0 = (gid / nv) * 4u;
    uint ib = (in_off >> 2) + b * inH * inW * cin4v;
    uint wb = w_off >> 2;
    uint ob = (out_off >> 2) + b * mDim * nv;

    float4 acc0 = float4(0.0), acc1 = float4(0.0), acc2 = float4(0.0), acc3 = float4(0.0);
    uint oh[4], ow[4];
    bool ok[4];
    for (uint i = 0u; i < 4u; i++) {
        uint m = m0 + i;
        ok[i] = m < mDim;
        oh[i] = m / outW;
        ow[i] = m - oh[i] * outW;
    }

    for (uint t = 0u; t < taps; t++) {
        uint ky = t / kW, kx = t - ky * kW;
        for (uint ci4 = 0u; ci4 < cin4v; ci4++) {
            float4 a0 = float4(0.0), a1 = float4(0.0), a2 = float4(0.0), a3 = float4(0.0);
            for (uint i = 0u; i < 4u; i++) {
                if (!ok[i]) continue;
                int ih = int(oh[i] * sH + ky) - int(padT);
                int iw = int(ow[i] * sW + kx) - int(padL);
                if (ih < 0 || ih >= int(inH) || iw < 0 || iw >= int(inW)) continue;
                uint px = uint(ih) * inW + uint(iw);
                float4 v = d[ib + px * cin4v + ci4];
                if (i == 0u) a0 = v; else if (i == 1u) a1 = v;
                else if (i == 2u) a2 = v; else a3 = v;
            }
            uint wRow = wb + (t * cin4v + ci4) * 4u * nv + n4;
            for (uint i = 0u; i < 4u; i++) {
                float4 wv = d[wRow + i * nv];
                acc0 += a0[i] * wv; acc1 += a1[i] * wv;
                acc2 += a2[i] * wv; acc3 += a3[i] * wv;
            }
        }
    }
    if (b_off != OFF_NONE) {
        float4 bv = d[(b_off >> 2) + n4]; // bias 全批共享——无 b 项（曾有 b*nv 越界读未初始化内存，cls 批 NaN 的根因）
        acc0 += bv; acc1 += bv; acc2 += bv; acc3 += bv;
    }
    if (act != 0u) {
        acc0 = apply_act(acc0, act, c1, c2, c3);
        acc1 = apply_act(acc1, act, c1, c2, c3);
        acc2 = apply_act(acc2, act, c1, c2, c3);
        acc3 = apply_act(acc3, act, c1, c2, c3);
    }
    // 残差在 act 之后（镜像 CPU conv2d_res：acc+bias → act → +residual）
    if (r_off != OFF_NONE) {
        uint rb = (r_off >> 2) + b * mDim * nv;
        if (ok[0]) acc0 += d[rb + m0 * nv + n4];
        if (ok[1]) acc1 += d[rb + (m0 + 1u) * nv + n4];
        if (ok[2]) acc2 += d[rb + (m0 + 2u) * nv + n4];
        if (ok[3]) acc3 += d[rb + (m0 + 3u) * nv + n4];
    }
    if (ok[0]) d[ob + m0 * nv + n4] = acc0;
    if (ok[1]) d[ob + (m0 + 1u) * nv + n4] = acc1;
    if (ok[2]) d[ob + (m0 + 2u) * nv + n4] = acc2;
    if (ok[3]) d[ob + (m0 + 3u) * nv + n4] = acc3;
}
