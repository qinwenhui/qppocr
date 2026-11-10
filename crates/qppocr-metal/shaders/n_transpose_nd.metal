// 通用 ND 转置（rank ≤ 5，精确连续存储、无通道 pad）——注意力头的
// [2,0,3,1,4] / [0,1,3,2] / [0,2,1,3] 等实转置。一线程一元素：
// 输出线性下标 → 按输出维分解 → 经 perm 映射回输入坐标。张量小
//（[B,T,3,8,15] 级），朴素拷贝足够（launch 主导）。
//   params: in_off, out_off, total, rank, dims[5]（**输入**逻辑维）,
//           perm[5]
// 存储定向陷阱：perm=[0,2,1] 常是存储恒等（rc 按生产者记账，见
// 计划模型的别名臂）——本内核只接真转置，别名由那边短路。
// 与 qppocr-gpu/shaders/n_transpose_nd.comp 逐行对应——改语义必须两边同步。
#include <metal_stdlib>
using namespace metal;

kernel void n_transpose_nd(
    device float* data [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]])
{
    uint P = p_off;
    uint in_off = params[P + 0], out_off = params[P + 1];
    uint total = params[P + 2], rank = params[P + 3];
    uint i = gid3.x;
    if (i >= total) return;

    uint dims[5]; uint perm[5];
    for (uint d = 0u; d < rank; d++) {
        dims[d] = params[P + 4 + d];
        perm[d] = params[P + 9 + d];
    }
    // 输出坐标：按输出维（= dims 按 perm 重排）分解线性 i
    uint oc[5] = {0u, 0u, 0u, 0u, 0u};
    uint rem = i;
    for (int d = int(rank) - 1; d >= 0; d--) {
        uint od = dims[perm[d]]; // 输出第 d 维的长度
        oc[d] = rem % od;
        rem /= od;
    }
    // 输入坐标：in[perm[d]] = oc[d]，再按输入维（dims）行主序 Horner 合成
    uint ic[5] = {0u, 0u, 0u, 0u, 0u};
    for (uint d = 0u; d < rank; d++) {
        ic[perm[d]] = oc[d];
    }
    uint src = 0;
    for (uint k = 0u; k < rank; k++) {
        src = src * dims[k] + ic[k];
    }
    data[out_off + i] = data[in_off + src];
}
