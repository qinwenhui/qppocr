// 逐行 softmax（rec 头：[rows × vocab]，最后轴；pad 列按 -inf 处理）。
// 一 WG（256 线程）一行：simdgroup 归约 max → exp → sum → 归一。
//   params: in_off, out_off, rows, cols, cpad（存储列数，cols ≤ cpad）
// 与 qppocr-gpu/shaders/n_softmax.comp 逐行对应——改语义必须两边同步。
// subgroup → simdgroup 映射：max/add → simd_max/simd_sum、elect →
// simd_is_first；simdgroup 宽度运行期自适应（Apple=32 / Mac-AMD=64）。
#include <metal_stdlib>
using namespace metal;

constant float RED_INIT = -3.4e38;

// exp1：conv.comp 的逐字拷贝（与 CPU softmax.rs 位级一致的多项式）。
// 曾用硬件 exp：概率差 ~1e-5 在近阈值候选（0/l/I、空格边界）上翻转
// CTC argmax——small 100 图 12 行文本差（GT 975 vs CPU 978）。
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

kernel void n_softmax(
    device float* data [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 wg [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup float red[256];

    uint P = p_off;
    uint in_off = params[P + 0], out_off = params[P + 1];
    uint rows = params[P + 2], cols = params[P + 3], cpad = params[P + 4];
    uint rpb = params[P + 5]; // rows/批（rows=N_pad*T；批是外维）

    uint row = wg.x;
    if (row >= rows) return;
    if (params[0] != 0u && row >= params[0] * rpb) return; // real_n 早退：批维补齐的空行零算力；0=不限（裸 dispatch 安全默认）
    uint base = in_off + row * cpad;
    uint obase = out_off + row * cpad;
    uint sgw = threads_per_simdgroup;
    uint nsg = 256u / sgw;
    uint sg = lid / sgw;

    // 1) 行 max（pad 列视 -inf）
    float m = RED_INIT;
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
    // red[] 复用边界：所有线程读完 max，才允许 phase 2 的 elect 写改写
    // red。缺此屏障时快线程的 sum 写会覆盖慢线程尚未读的 max（T=120
    // 实测整行级非确定：max 读到 sum 值 → exp 溢出 → 整行概率错）。
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) exp(x - max)（并行；exp1 与 CPU exp256_ps 逐位同构——硬件 exp
    // 会差 ~1e-5，剃刀薄时间步翻 CTC 单字）
    for (uint j = lid; j < cols; j += 256u) {
        data[obase + j] = exp1(data[base + j] - rowmax);
    }
    // 写 → 单线程读是**设备内存**跨线程依赖：屏障带 mem_device
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    // 2b) 求和：单线程按 CPU x86 softmax_row_vec 的 lane 序（树归约序差
    // ~1e-6 同样翻 CTC）——lane k = 元素 k+8t 的顺序和；水平树
    // ((l0+l4)+(l1+l5))+((l2+l6)+(l3+l7))（=hadd 序）；标量尾顺序并入；
    // 归一用 ×(1/sum)（CPU 是 inv 乘，不是除）。
    if (lid == 0u) {
        float l0 = 0.0, l1 = 0.0, l2 = 0.0, l3 = 0.0;
        float l4 = 0.0, l5 = 0.0, l6 = 0.0, l7 = 0.0;
        uint q = cols >> 3;
        for (uint t = 0u; t < q; t++) {
            uint b8 = obase + t * 8u;
            l0 += data[b8];
            l1 += data[b8 + 1u];
            l2 += data[b8 + 2u];
            l3 += data[b8 + 3u];
            l4 += data[b8 + 4u];
            l5 += data[b8 + 5u];
            l6 += data[b8 + 6u];
            l7 += data[b8 + 7u];
        }
        float s = ((l0 + l4) + (l1 + l5)) + ((l2 + l6) + (l3 + l7));
        for (uint j = q * 8u; j < cols; j++) {
            s += data[obase + j];
        }
        red[0] = 1.0 / s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = red[0];

    // 3) 归一（×inv）；pad 列写 0
    for (uint j = lid; j < cpad; j += 256u) {
        data[obase + j] = j < cols ? data[obase + j] * inv : 0.0;
    }
}
