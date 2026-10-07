// rank-3 出口（rec 的 [B,T,V]）：存储 [rows × cpad] → f32 NCHW 恒等拷贝。
// cols=V 是 NCHW 最后轴——逐元素同序；pad 列不写出。
// rpb = rows/批（批维补齐后 rows=N_pad*T；row ≥ real_n*rpb = 空行早退）。
// 与 qppocr-gpu/shaders/n_exit3.comp 逐行对应——改语义必须两边同步。
#include <metal_stdlib>
using namespace metal;

kernel void n_exit3(
    device float* data [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]])
{
    uint P = p_off;
    uint in_off = params[P + 0], out_off = params[P + 1];
    uint rows = params[P + 2], cols = params[P + 3], cpad = params[P + 4];
    uint rpb = params[P + 5];

    uint gid = gid3.x;
    if (gid >= rows * cols) return;
    uint row = gid / cols;
    uint col = gid - row * cols;
    if (params[0] != 0u && row >= params[0] * rpb) return; // real_n 早退：批维补齐的空行零算力；0=不限（裸 dispatch 安全默认）
    data[out_off + gid] = data[in_off + row * cpad + col];
}
