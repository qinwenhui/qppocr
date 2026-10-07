// NHWC-f32 通道广播族：op 由参数块给出（0=mul_c 1=add_c 2=muladd_scale
// 3=fused_hardsigmoid_mul 4=fused_sigmoid_mul 5=bn(a*g+b 全广播)）。
// NHWC 里通道恒 = float4 下标 % C——每 float4 恰是同一位置的 4 个通道，
// gate 直接按 float4 取（gate[Cpad4]）。
// 语义镜像：mul_c/add_c（广播）、muladd_scale（out=f*gate+r）、
// fused_*_mul（out=f*act(gate)，gate 为前激活值）。
// 布局：op, a_off, gate_off, r_off, out_off, n(元素数), cpad, p1(f32bits), p2,
//       hw(每门行的特征行数——[N,C,H,W] 特征 × [N,C,1,1] 门 = H*W；
//       [V] 单行门 = 全部行。曾有 bug：门恒读批 0 的行——det N=1 恰好
//       恒对，cls B=17 全批用了批 0 的 SE 门）
// 与 qppocr-gpu/shaders/n_channel.comp 逐行对应——改语义必须两边同步。
#include <metal_stdlib>
using namespace metal;

kernel void n_channel(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]])
{
    uint P = p_off;
    uint op = params[P + 0];
    uint a_off = params[P + 1], gate_off = params[P + 2], r_off = params[P + 3];
    uint out_off = params[P + 4], n = params[P + 5], cpad = params[P + 6];
    float p1 = as_type<float>(params[P + 7]);
    float p2 = as_type<float>(params[P + 8]);
    uint hw = params[P + 9]; // 每门行的特征行数

    uint nv = n >> 2;
    uint c4 = cpad >> 2;
    uint gid = gid3.x;
    if (gid >= nv) return;
    uint rowC4 = gid % c4; // 该位置的通道 float4 下标

    float4 a = d[(a_off >> 2) + gid];
    // 门行 = 特征行 / hw（批维）；det N=1 时 hw=特征行数 → 恒第 0 行
    uint featRow = gid / c4;
    uint gateRow = featRow / hw;
    // real_n 早退（按特征行判批，两种门形都覆盖：[N,C,1,1] 门 featRow/hw
    // 即批下标；[V] 单行门 hw=批内全部行 → featRow ≥ rn*hw 即空批行，
    // 旧守卫 gateRow>=rn 对它恒不触发、曾白算全部补齐行）
    if (params[0] != 0u && featRow >= params[0] * hw) return; // real_n=0 = 不限（裸 dispatch 安全默认）
    float4 g = d[(gate_off >> 2) + gateRow * c4 + rowC4];
    float4 o;
    if (op == 0u) o = a * g;
    else if (op == 1u) o = a + g;
    else if (op == 2u) o = a * g + d[(r_off >> 2) + gid];
    else if (op == 5u) {
        float4 rb = d[(r_off >> 2) + gateRow * c4 + rowC4]; // BN shift 同批寻址
        o = a * g + rb;
    }
    else if (op == 3u) o = a * clamp(g * p1 + p2, float4(0.0), float4(1.0));
    else o = a * (1.0 / (1.0 + exp(-g)));
    d[(out_off >> 2) + gid] = o;
}
