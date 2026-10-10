// ConvTranspose 的**空间分块**版本（NHWC f32，2×2 stride 2，无 pad）。参数块
// 与 `n_convt` **逐字段同义**（同一份 plan.rs 发射逻辑，只换内核名与网格），
// 所以这一支是 `n_convt` 的drop-in 替换——**不是**另一套语义。
//
// ## 为什么（2026-10-10 探针 `probe_convt.metal`）
//
// det 里 `n_convt` 两条分别是 ConvTranspose.0（8.20 ms）与 ConvTranspose.2
// （1.91 ms）。ConvTranspose.0 真实只做 1.70 GFLOP（**flopsmap 按 taps 记 MAC
// 把 FLOPs 放大了 4×**——kH≤sH 时每个输出只吃一个 tap），即 8.20 ms 里
// 210 GFLOPS。三条下界（下界流量 66.4 MB / 77 GB/s = 0.86 ms；849 M MAC /
// 1670 GMAC/s = 0.51 ms）说明它既不是喂不上也不是算不动。
//
// 探针九臂的归因（`convt_probe_attribution`，同形状同网格）：
//
//   · 每 ci 迭代 2 条 float4 装载换 4 条 FMA（比值 **2.0**），其中输入那条是
//     `d[...][ci & 3u]`——**动态 float4 下标**，且同一地址每 4 个 ci 重复一次；
//   · 三个「掩地址」的臂把**字节**与**条数**劈开：权重掩进 1 KB → 1.82×、
//     去掉权重装载 → 2.14×、访存全砍 → 4.37×；输入掩进 1 KB 只 1.04×、
//     输出掩进 1 KB 只 1.14×。⇒ **瓶颈是权重装载的访存路径**，不是输入/输出
//     流量，也不是算力。
//
// 这一支把权重复用从「1 条装载喂 1 个 FMA」提到「1 条装载喂 4 个 FMA」：
// 同一个 (ih,iw) 的 2×2 输出块由 4 个 tap 分别决定，4 个 tap 的**输入相同、
// 权重不同**，于是输入 1 读 4 用、输出天然成对写。再叠 4 个相邻 iw（权重
// 跨 4 个空间位置复用）：
//
//   装载/float4-FMA   17/16 = 1.06（原）→ **20/64 = 0.31**（本内核）
//
// 探针同台实测 **3.45×**（5.023 → 1.456 ms），已到该形状 fmaonly 地板
// （1.148 ms）的 79%。
//
// ## 网格
//
//   线程 = `inH * (inW/4) * nv`，`tiles` 同上，`groups = div256(tiles)`
//   （`gid3.y` 仍是 batch，与 `n_convt` 的 `wg.y` 同义）
//
// 每个线程覆盖 `(ih, iw..iw+3)` 四个输入位置的 2×2 输出 = 16 个输出 float4。
//
// ## 启用判据（plan.rs 侧，缺一不可）
//
// - `kh==kw==2 && sh==sw==2`：4 个 tap 的枚举方式写死在这里；
// - `inW % 4 == 0`：空间块不跨行；
// - `nv >= 4`：nv=1（ConvTranspose.2 那条）上本内核实测 **1.73×**，不如
//   `n_convt` 自己的 ci4 快路（2.01×）——累加器 16 个 float4 换不来通道维
//   的并行度，寄存器压力反噬。那条走 `n_convt`。
//
// ## 同源
//
// **Metal 独有，GLSL 侧没有对应物**——本机没有 SPIR-V 工具链
// （glslangValidator/glslc 均无），签入的 `.spv` 才是 Vulkan 真正跑的东西，
// 只写 `.comp` 不改 `.spv` 等于造一份脱钩的源码。Vulkan 侧靠
// `PlanBuilder::conv_t_tile = false` 在计划层就关掉（与 `conv1x1_mma`
// 同款：见 `crates/qppocr-gpu/src/plan.rs`）。
#include <metal_stdlib>
using namespace metal;

constant uint OFF_NONE = 0xFFFFFFFFu;

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

kernel void n_convt4w(
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

    uint nv = nDim >> 2;
    uint gid = gid3.x;
    uint inW4 = inW >> 2;
    uint tiles = inH * inW4 * nv;
    if (gid >= tiles) return;
    uint n4 = gid % nv;
    uint q = gid / nv;
    uint ih = q / inW4, iwq = (q - ih * inW4) << 2; // iwq = 4 × 块号
    uint ib = (in_off >> 2) + wg.y * inH * inW * ci4v;
    uint wb = w_off >> 2;
    uint ob = (out_off >> 2) + wg.y * mDim * nv;

    // 权重每 tap 一张 [cin][nv]，此处把 tap 偏移与 n4 预先并进基址。
    uint wt0 = wb + 0u * cin * nv + n4;
    uint wt1 = wb + 1u * cin * nv + n4;
    uint wt2 = wb + 2u * cin * nv + n4;
    uint wt3 = wb + 3u * cin * nv + n4;

    // acc[tap][j]：tap 的 (ky,kx) = (t>>1, t&1)，j = 4 个相邻 iw。
    float4 a0[4], a1[4], a2[4], a3[4];
    for (uint j = 0u; j < 4u; j++) {
        a0[j] = float4(0.0); a1[j] = float4(0.0);
        a2[j] = float4(0.0); a3[j] = float4(0.0);
    }
    uint xb = ib + (ih * inW + iwq) * ci4v;
    for (uint ci4 = 0u; ci4 < ci4v; ci4++) {
        float4 x0 = d[xb + 0u * ci4v + ci4];
        float4 x1 = d[xb + 1u * ci4v + ci4];
        float4 x2 = d[xb + 2u * ci4v + ci4];
        float4 x3 = d[xb + 3u * ci4v + ci4];
        uint c0 = (ci4 << 2) * nv;
        uint c1 = c0 + nv, c2 = c0 + 2u * nv, c3 = c0 + 3u * nv;
        // 4 条 lane 静态取用（`x0.x` 等）——这一步同时消掉 `n_convt` 里
        // `d[...][ci & 3u]` 的动态 float4 下标与那 4× 冗余装载。
#define CT_WSTEP(WBASE, ACC)                                          \
        {                                                             \
            float4 w;                                                 \
            w = d[WBASE + c0]; ACC[0] = fma(x0.x, w, ACC[0]);         \
            ACC[1] = fma(x1.x, w, ACC[1]);                            \
            ACC[2] = fma(x2.x, w, ACC[2]);                            \
            ACC[3] = fma(x3.x, w, ACC[3]);                            \
            w = d[WBASE + c1]; ACC[0] = fma(x0.y, w, ACC[0]);         \
            ACC[1] = fma(x1.y, w, ACC[1]);                            \
            ACC[2] = fma(x2.y, w, ACC[2]);                            \
            ACC[3] = fma(x3.y, w, ACC[3]);                            \
            w = d[WBASE + c2]; ACC[0] = fma(x0.z, w, ACC[0]);         \
            ACC[1] = fma(x1.z, w, ACC[1]);                            \
            ACC[2] = fma(x2.z, w, ACC[2]);                            \
            ACC[3] = fma(x3.z, w, ACC[3]);                            \
            w = d[WBASE + c3]; ACC[0] = fma(x0.w, w, ACC[0]);         \
            ACC[1] = fma(x1.w, w, ACC[1]);                            \
            ACC[2] = fma(x2.w, w, ACC[2]);                            \
            ACC[3] = fma(x3.w, w, ACC[3]);                            \
        }
        CT_WSTEP(wt0, a0)
        CT_WSTEP(wt1, a1)
        CT_WSTEP(wt2, a2)
        CT_WSTEP(wt3, a3)
#undef CT_WSTEP
    }
    float4 bv = float4(0.0);
    if (b_off != OFF_NONE) bv = d[(b_off >> 2) + n4]; // bias 全批共享
    uint oh = ih * 2u;
    // 相邻输出列在 arena 里差 **nv 个 float4**（NHWC 的通道维）。
    uint cstep = nv << 1;
    uint row0 = (oh * outW + (iwq << 1)) * nv + n4;
    uint row1 = ((oh + 1u) * outW + (iwq << 1)) * nv + n4;
    for (uint j = 0u; j < 4u; j++) {
        uint co = j * cstep;
        float4 r0 = a0[j] + bv, r1 = a1[j] + bv, r2 = a2[j] + bv, r3 = a3[j] + bv;
        if (act == 2u) {
            r0 = max(r0, float4(0.0)); r1 = max(r1, float4(0.0));
            r2 = max(r2, float4(0.0)); r3 = max(r3, float4(0.0));
        } else if (act == 1u) {
            float inv = 1.0 / c1;
            r0 = c3 * r0 * (float4(erf1(r0.x * inv), erf1(r0.y * inv),
                                   erf1(r0.z * inv), erf1(r0.w * inv)) + c2);
            r1 = c3 * r1 * (float4(erf1(r1.x * inv), erf1(r1.y * inv),
                                   erf1(r1.z * inv), erf1(r1.w * inv)) + c2);
            r2 = c3 * r2 * (float4(erf1(r2.x * inv), erf1(r2.y * inv),
                                   erf1(r2.z * inv), erf1(r2.w * inv)) + c2);
            r3 = c3 * r3 * (float4(erf1(r3.x * inv), erf1(r3.y * inv),
                                   erf1(r3.z * inv), erf1(r3.w * inv)) + c2);
        }
        d[ob + row0 + co] = r0;
        d[ob + row0 + co + nv] = r1;
        d[ob + row1 + co] = r2;
        d[ob + row1 + co + nv] = r3;
    }
    // sH/sW/kW 未用：判据已由 plan.rs 钉死为 2/2/2（见文件头「启用判据」）。
    (void)sH; (void)sW; (void)kW;
}
