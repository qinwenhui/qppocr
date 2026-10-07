// 入口转换：f32 NCHW [N,C,H,W] → f32 NHWC [N,H,W,Cpad4]（pad 通道补 0）。
// 与 qppocr-gpu/shaders/n_entry.comp 逐行对应——改语义必须两边同步。
// 绑定编号沿用 GLSL 侧：0=F32 输入区、1=参数区（u32 视图）、2=D32 输出
// 区（float4 视图）；三者是同一 arena 缓冲的别名视图，Rust 侧统一把
// arena 绑到 0/1/2。p_off 经 set_bytes 注入（push constant 的 Metal 对应物）。
#include <metal_stdlib>
using namespace metal;

kernel void n_entry(
    device const float* data [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    device float4* d [[buffer(2)]],
    constant uint& p_off [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint P = p_off;
    uint in_off = params[P + 0];   // float 元素偏移
    uint out_off = params[P + 1];  // word 偏移（f32）
    uint nb = params[P + 2], c = params[P + 3], h = params[P + 4], w = params[P + 5];
    uint cpad = params[P + 6];     // 补到 %4 的通道数
    uint cp4 = cpad >> 2;

    uint hw = h * w;
    if (gid >= nb * hw) return;
    uint n = gid / hw, pos = gid - n * hw;
    if (params[0] != 0u && n >= params[0]) return; // real_n 早退：批维补齐的空行零算力；0=不限（裸 dispatch 安全默认）
    uint plane = c * hw;
    for (uint c4 = 0u; c4 < cp4; c4++) {
        float4 v = float4(0.0);
        uint base = c4 * 4u;
        if (base + 0u < c) v.x = data[in_off + n * plane + (base + 0u) * hw + pos];
        if (base + 1u < c) v.y = data[in_off + n * plane + (base + 1u) * hw + pos];
        if (base + 2u < c) v.z = data[in_off + n * plane + (base + 2u) * hw + pos];
        if (base + 3u < c) v.w = data[in_off + n * plane + (base + 3u) * hw + pos];
        d[(out_off >> 2) + (n * hw + pos) * cp4 + c4] = v;
    }
}
