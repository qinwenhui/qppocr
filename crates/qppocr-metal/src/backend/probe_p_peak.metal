// 变体：**纯 FMA 峰值探针**——没有访存、没有分支、没有索引运算，只有
// 若干条互相独立的 `fma`，用来量这块 M4 在**当前 dispatch 形状下**到底
// 能跑多少 fp32 FLOPs。
//
// ## 为什么必须先量它
///
// 常识里的「M4 十大核 GPU ≈ 4 TFLOPS fp32」是标称值。实测内核只跑到
// 150–400 GFLOPS，在不知道天花板的情况下没法判断这是「内核烂」还是
// 「标称值不适用」。这个探针给出分母。
//
// 两个累加器数量（4 与 16 个 float4 = 16 / 64 寄存器）各测一遍：两者
// 算数指令完全一样，只差寄存器占用——**差值就是占用率的代价**。
#include <metal_stdlib>
using namespace metal;

/// 4 个 float4 累加器（16 寄存器）：占用率友好，ILP 16 条独立链。
kernel void p_peak4(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]])
{
    uint iters = params[p_off];
    float t = float(gid3.x & 255u) * 1e-6f + 1.0f;
    float4 w = float4(t);
    float4 a0 = float4(1.0, 2.0, 3.0, 4.0);
    float4 a1 = float4(1.1, 2.1, 3.1, 4.1);
    float4 a2 = float4(1.2, 2.2, 3.2, 4.2);
    float4 a3 = float4(1.3, 2.3, 3.3, 4.3);
    for (uint it = 0u; it < iters; it++) {
        a0 = fma(a0, w, a0);
        a1 = fma(a1, w, a1);
        a2 = fma(a2, w, a2);
        a3 = fma(a3, w, a3);
    }
    d[(p_off + 16u) >> 2] = a0 + a1 + a2 + a3;
}

/// 16 个 float4 累加器（64 寄存器）：ILP 64 条，占用率差。
kernel void p_peak16(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]])
{
    uint iters = params[p_off];
    float t = float(gid3.x & 255u) * 1e-6f + 1.0f;
    float4 w = float4(t);
    float4 a0 = float4(1.0, 2.0, 3.0, 4.0);
    float4 a1 = float4(1.1, 2.1, 3.1, 4.1);
    float4 a2 = float4(1.2, 2.2, 3.2, 4.2);
    float4 a3 = float4(1.3, 2.3, 3.3, 4.3);
    float4 a4 = float4(1.4, 2.4, 3.4, 4.4);
    float4 a5 = float4(1.5, 2.5, 3.5, 4.5);
    float4 a6 = float4(1.6, 2.6, 3.6, 4.6);
    float4 a7 = float4(1.7, 2.7, 3.7, 4.7);
    float4 a8 = float4(1.8, 2.8, 3.8, 4.8);
    float4 a9 = float4(1.9, 2.9, 3.9, 4.9);
    float4 a10 = float4(2.0, 3.0, 4.0, 5.0);
    float4 a11 = float4(2.1, 3.1, 4.1, 5.1);
    float4 a12 = float4(2.2, 3.2, 4.2, 5.2);
    float4 a13 = float4(2.3, 3.3, 4.3, 5.3);
    float4 a14 = float4(2.4, 3.4, 4.4, 5.4);
    float4 a15 = float4(2.5, 3.5, 4.5, 5.5);
    for (uint it = 0u; it < iters; it++) {
        a0 = fma(a0, w, a0);
        a1 = fma(a1, w, a1);
        a2 = fma(a2, w, a2);
        a3 = fma(a3, w, a3);
        a4 = fma(a4, w, a4);
        a5 = fma(a5, w, a5);
        a6 = fma(a6, w, a6);
        a7 = fma(a7, w, a7);
        a8 = fma(a8, w, a8);
        a9 = fma(a9, w, a9);
        a10 = fma(a10, w, a10);
        a11 = fma(a11, w, a11);
        a12 = fma(a12, w, a12);
        a13 = fma(a13, w, a13);
        a14 = fma(a14, w, a14);
        a15 = fma(a15, w, a15);
    }
    d[(p_off + 16u) >> 2] =
        a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
}
