// rec 头的融合出口（**Metal 独有**）：一条内核同时做完 softmax 的数值与
// CTC 的 (val, idx)，取代原来 `n_softmax` + `n_exit3_argmax` **两条**
// dispatch（最宽那份计划里这两条合计 2.03 ms / 15.71 ms = 13%）。
//
// ## 语义与那两条**逐位一致**，不是"近似等价"
//   * rowmax / exp1 / 求和 / 归一 全部照抄 n_softmax：exp1 多项式、
//     8 条 lane 累加链（lane k 累加 j ≡ k (mod 8)、链内串行）、水平树
//     `((l0+l4)+(l1+l5))+((l2+l6)+(l3+l7))`、标量尾按 j 递增并入、
//     归一用 `×(1/s)` 不是除。
//   * argmax 照抄 n_exit3_argmax，且**必须对归一后的值取**首个严格最大
//     （平局取最小下标）——不可以在求和阶段顺手对 exp 值比大小：乘 inv
//     是一次舍入，能把两个原本不等的概率压成相等，从而造出新的平局，
//     平局的下标语义就变了。
//
// ## 与 n_softmax 的唯一差别：exp **就地**写回输入平面
// n_softmax 把 exp 写进另一个输出平面、归一、再由 exit3 读回来；这里
// 直接覆盖 logits —— 省掉一次 [rows×cpad] 的分配和一趟来回。**只在
// 本内核是该平面唯一消费者时启用**（plan 侧的守卫负责，见
// `PlanBuilder::head_fuse`）。
//
// ## 为什么要把 8 条链拆给 8 个线程
// phase 3 原本是**单线程**跑 `q = cols>>3 = 2339` 轮、每轮 8 次设备内存
// 装载 —— 这是整条内核的大头。那 8 条链**互相独立**（l_k 只是子序列
// {k, k+8, k+16, …} 的串行和），谁算都一样 ⇒ 拆给 8 个线程，墙钟 ~8×，
// **位级结果一字不差**。再往下拆就改序了，所以 8 是这条路的并行度上限。
//
// 与 `qppocr-gpu/shaders/*.comp` **无对应文件**（Metal 独有，Vulkan 侧
// 仍走 n_softmax + n_exit3_argmax）——所以改这里不会与 Vulkan 脱钩。
//
//   params: in_off, out_off, rows, cols, cpad, rpb（rows/批，空行早退）
#include <metal_stdlib>
using namespace metal;

// exp1：n_softmax.metal 的逐字拷贝（与 CPU softmax.rs 位级一致的多项式）。
// 曾用硬件 exp：概率差 ~1e-5 在近阈值候选（0/l/I、空格边界）上翻转 CTC
// argmax——small 100 图 12 行文本差（GT 975 vs CPU 978）。
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

kernel void n_head_argmax(
    device float* data [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint wg [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgw [[threads_per_simdgroup]])
{
    threadgroup float red[256];      // 阶段 1 的 max 归约 + 阶段 3 的 8 条链
    threadgroup float2 ared[256];    // 阶段 4 的 argmax 归约（(值, 下标位型)）

    uint P = p_off;
    uint in_off = params[P + 0], out_off = params[P + 1];
    uint rows = params[P + 2], cols = params[P + 3], cpad = params[P + 4];
    uint rpb = params[P + 5]; // rows/批（rows=N_pad*T；批是外维）

    // `wg` 是标量（MSL 禁同内核混用标量/向量位置属性）——grid 恒为
    // [rows, 1, 1]，第 0 维即行号；sgw 用 `threads_per_simdgroup`
    // 属性取，Apple=32 / Mac-AMD=64 都对（别写死 32）。
    uint row = wg;
    if (row >= rows) return;
    if (params[0] != 0u && row >= params[0] * rpb) return; // real_n 早退：批维补齐的空行零算力；0=不限（裸 dispatch 安全默认）
    uint base = in_off + row * cpad;
    uint nsg = 256u / sgw;
    uint sg = lid / sgw;

    // 1) 行 max（pad 列视 -inf）——n_softmax phase 1 逐字
    float m = -3.4e38;
    for (uint j = lid; j < cols; j += 256u) {
        m = max(m, data[base + j]);
    }
    float wm = simd_max(m);
    if (simd_is_first()) red[sg] = wm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < nsg && lid % 4u == 0u) {
        float v = red[lid];
        if (lid + 1u < nsg) v = max(v, red[lid + 1u]);
        if (lid + 2u < nsg) v = max(v, red[lid + 2u]);
        if (lid + 3u < nsg) v = max(v, red[lid + 3u]);
        red[lid / 4u] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowmax = red[0];
    for (uint g = 1u; g < (nsg + 3u) / 4u; g++) {
        rowmax = max(rowmax, red[g]);
    }
    // red[] 复用边界：所有线程读完 max，才允许阶段 3 的链写改写 red。
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) exp 就地（= n_softmax phase 2，只是 obase == base）
    for (uint j = lid; j < cols; j += 256u) {
        data[base + j] = exp1(data[base + j] - rowmax);
    }
    // 写 → 单线程读是**设备内存**跨线程依赖：屏障带 mem_device
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // 3) 8 条 lane 链：lane k 累加 j = k, k+8, k+16, …（CPU x86
    //    softmax_row_vec 的 lane 序）。链内仍是同一个串行和 ⇒ 逐位不变。
    uint q = cols >> 3;
    if (lid < 8u) {
        float l = 0.0;
        for (uint t = 0u; t < q; t++) {
            l += data[base + t * 8u + lid];
        }
        red[lid] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        // 水平树 = hadd 序；再用 ×(1/s) 归一（CPU 是 inv 乘，不是除）；
        // 标量尾按 j 递增并入（n_softmax phase 2b 逐字）。
        float s = ((red[0] + red[4]) + (red[1] + red[5])) + ((red[2] + red[6]) + (red[3] + red[7]));
        for (uint j = q * 8u; j < cols; j++) {
            s += data[base + j];
        }
        red[0] = 1.0 / s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = red[0];

    // 4) argmax（n_exit3_argmax 逐字；值改成归一后的 data[base+j]*inv）。
    //    必须等 inv 落地再扫——见文件头「不能顺手比 exp」。
    float bm = -3.4e38;
    uint mi = 0u;
    for (uint j = lid; j < cols; j += 256u) {
        float v = data[base + j] * inv;
        if (v > bm) {
            bm = v;
            mi = j;
        }
    }
    // 子组归约：值取 max；等值线程里取最小下标
    float wbv = simd_max(bm);
    uint cand = (bm == wbv) ? mi : 0xFFFFFFFFu;
    uint wbi = simd_min(cand);
    if (simd_is_first()) ared[sg] = float2(wbv, as_type<float>(wbi));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 跨子组（与 n_softmax/n_exit3_argmax 同款尾归约）
    if (lid < nsg && lid % 4u == 0u) {
        float v = ared[lid].x;
        uint ix = as_type<uint>(ared[lid].y);
        for (uint k = 1u; k < 4u && lid + k < nsg; k++) {
            float v2 = ared[lid + k].x;
            uint ix2 = as_type<uint>(ared[lid + k].y);
            if (v2 > v || (v2 == v && ix2 < ix)) {
                v = v2;
                ix = ix2;
            }
        }
        ared[lid / 4u] = float2(v, as_type<float>(ix));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float bv = ared[0].x;
    uint bi = as_type<uint>(ared[0].y);
    for (uint g = 1u; g < (nsg + 3u) / 4u; g++) {
        float v2 = ared[g].x;
        uint ix2 = as_type<uint>(ared[g].y);
        if (v2 > bv || (v2 == bv && ix2 < bi)) {
            bv = v2;
            bi = ix2;
        }
    }
    if (lid == 0u) {
        // val = 归一后的最大概率（逐位等于 n_softmax phase 3 写下的那一格）；
        // idx 按 exit3 的约定存**位型**（主机侧 as u32 读回）。
        data[out_off + row * 2u] = bv;
        data[out_off + row * 2u + 1u] = as_type<float>(bi);
    }
}
