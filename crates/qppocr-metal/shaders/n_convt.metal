// ConvTranspose（NHWC f32，kH≤sH 且 kW≤sW——det 的 FPN 2×2 s2 上采样）：
// 每输出恰由一个 (ih,iw,ky,kx) tap 决定。ONNX 权重 [Ci,Co,kh,kw] 重排为
// 每 tap 一张 [Ci][nv] 的 float4 矩阵（元素 = 4 个输出通道）。
// act：0=无、1=gelu(c1,c2,c3)、2=relu。
//
// ## 与 GLSL 侧的关系（**已分叉，此处是唯一准确的说法**）
//
// `qppocr-gpu/shaders/n_convt.comp` 仍是**标量热循环**（`d[..][ci & 3u]`），
// 本文件换成了下面的 ci4 展开。两者**语义/结果位级一致**（累加次序相同），
// 但**源码不再逐行对应**——本机没有 SPIR-V 工具链（glslangValidator/glslc
// 都没有），改 `.comp` 只会让它与签入的 `spirv/n_convt.spv` 脱钩，而真正
// 跑的是 `.spv`。所以：**GLSL/Vulkan 侧保持原样（正确但慢），分叉是有意的**。
// 哪天要同步，必须同时重编 `spirv/n_convt.spv`，不能只改 `.comp`。
//
// 2×2/s2/**nv≥4** 的形状也不走这里，走空间分块的 `n_convt4w`（见该文件头注释：
// 探针 `convt_probe_attribution` 实测 3.45×，本内核同为 1×1 通道块时只有 1.28×）。
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
    uint wt = wb + tap * cin * nv + n4;
    float4 acc = float4(0.0);
    // ci 按 4 展开：一次读整个 float4，四条 lane **静态**取用。这一步同时消掉
    // 两样东西——`d[...][ci & 3u]` 的**动态 float4 下标**，以及同一地址每 4 个
    // ci 重复一次的 **4× 冗余装载**。累加次序与标量版逐 ci 相同（ci = 4k+0..3
    // 依次 fma），故**位级一致**：`probe_convt.metal` 的 `p_ct_ci4` 与 `p_ct_base`
    // 实测差 0.000e0。
    //
    // 为什么值得（同一次探针，`convt_probe_attribution`）：nv=1 那条
    // （ConvTranspose.2，`ismall` 2.13×）瓶颈就是输入侧这 4× 冗余 + 装载条数，
    // 本改写单独给 **2.01×**，且**不改网格**。两条 real 形状上 ci4 是
    // 1.28× / 2.01×。
    //
    // `cin` 恒为 4 的倍数（plan.rs 传 `cpad4(Ci)`，`ci4v` = 其 1/4），
    // 故遍历 `ci4v` 覆盖全部 ci。
    for (uint ci4 = 0u; ci4 < ci4v; ci4++) {
        float4 x = d[ib + px * ci4v + ci4];
        acc += x.x * d[wt + (ci4 << 2) * nv];
        acc += x.y * d[wt + ((ci4 << 2) + 1u) * nv];
        acc += x.z * d[wt + ((ci4 << 2) + 2u) * nv];
        acc += x.w * d[wt + ((ci4 << 2) + 3u) * nv];
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
