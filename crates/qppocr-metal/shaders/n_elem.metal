// NHWC-f32 逐元素族：op 由参数块给出（0=relu 1=sigmoid 2=hardsigmoid
// 3=clip 4=add 5=mul 6=标量add 7=标量mul 8=减行向量 9=square
// 10=sqrt 11=除行向量 12=sub 13=div）。float4 一次 4 元素，f32 计算。
// 布局：op, in_off, in2_off, out_off, n, p1(f32bits), p2(f32bits)
//（in2 仅双目用；hardsigmoid p1=alpha p2=beta；clip p1=min p2=max；
//  行向量广播 8/11 的 p1 = 每行 float4 数 c4 的位型——b 是 [rows,4]
//  存储、只有 lane0 有效，行内 4 通道同减/除该值——LN 的 x-mean 与
//  (x-mean)/std 用）。
// 与 qppocr-gpu/shaders/n_elem.comp 逐行对应——改语义必须两边同步。
// 绑定编号沿用 GLSL 侧（D32=0、Params=1）；p_off 经 set_bytes 注入。
#include <metal_stdlib>
using namespace metal;

kernel void n_elem(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint P = p_off;
    uint op = params[P + 0];
    uint in_off = params[P + 1], in2_off = params[P + 2], out_off = params[P + 3];
    uint n = params[P + 4];
    float p1 = as_type<float>(params[P + 5]);
    float p2 = as_type<float>(params[P + 6]);

    uint nv = n >> 2;
    if (gid >= nv) return;
    uint ib = in_off >> 2;

    if (op == 6u || op == 7u) { // 标量广播（cls 的 Mul([1,C,1,1],[1])）
        float4 a = d[ib + gid];
        float4 sv = d[in2_off >> 2]; // 标量在 lane0（旁 lane 是分配对齐垃圾）
        d[(out_off >> 2) + gid] = (op == 6u) ? a + sv.x : a * sv.x;
    } else if (op == 4u) { // add
        float4 a = d[ib + gid], b = d[(in2_off >> 2) + gid];
        d[(out_off >> 2) + gid] = a + b;
    } else if (op == 5u) { // mul
        float4 a = d[ib + gid], b = d[(in2_off >> 2) + gid];
        d[(out_off >> 2) + gid] = a * b;
    } else if (op == 12u) { // sub
        float4 a = d[ib + gid], b = d[(in2_off >> 2) + gid];
        d[(out_off >> 2) + gid] = a - b;
    } else if (op == 13u) { // div
        float4 a = d[ib + gid], b = d[(in2_off >> 2) + gid];
        d[(out_off >> 2) + gid] = a / b;
    } else if (op == 8u || op == 11u) { // 行向量广播 sub/div（LN 链）
        uint c4 = as_type<uint>(p1);
        uint row = gid / c4;
        float4 a = d[ib + gid];
        // [rows,4] 的 lane0：d 是 float4 数组，行 r 的 lane0 在第 r 个
        // float4（= word in2_off + r*4，与 n_reduce_last 的写侧对齐）。
        float s = d[(in2_off >> 2) + row].x;
        d[(out_off >> 2) + gid] = (op == 8u) ? a - s : a / s;
    } else {
        float4 a = d[ib + gid];
        if (op == 0u) a = max(a, float4(0.0));
        else if (op == 1u) a = 1.0 / (1.0 + exp(-a));
        else if (op == 2u) a = clamp(a * p1 + p2, float4(0.0), float4(1.0));
        else if (op == 9u) a = a * a;       // square（Pow 常量 2，LN 方差）
        else if (op == 10u) a = sqrt(a);    // LN 的 std
        else a = clamp(a, float4(p1), float4(p2)); // clip
        d[(out_off >> 2) + gid] = a;
    }
}
