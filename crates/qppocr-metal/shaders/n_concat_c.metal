// 通道拼接（NHWC f32，axis=1）：out 行 = 各输入行按通道段串接。
// 一线程一 (空间位置, 输出 c4)；线性扫 ≤8 段找源。
// 与 qppocr-gpu/shaders/n_concat_c.comp 逐行对应——改语义必须两边同步。
#include <metal_stdlib>
using namespace metal;

kernel void n_concat_c(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]])
{
    uint P = p_off;
    uint out_off = params[P + 0];
    uint hw = params[P + 1];        // 空间位置数（N*H*W）
    uint nIn = params[P + 2];
    uint outC4 = params[P + 3];     // 总通道 / 4（pad 后）

    uint gid = gid3.x;
    uint tiles = hw * outC4;
    if (gid >= tiles) return;
    uint c4 = gid % outC4;
    uint pos = gid / outC4;

    // 段表在参数块尾部：{src_off, c4_len} × nIn
    uint acc = 0u;
    for (uint i = 0u; i < nIn; i++) {
        uint sOff = params[P + 4 + i * 2];
        uint sC4 = params[P + 5 + i * 2];
        if (c4 < acc + sC4) {
            d[(out_off >> 2) + pos * outC4 + c4] =
                d[(sOff >> 2) + pos * sC4 + (c4 - acc)];
            return;
        }
        acc += sC4;
    }
}
