// `n_head_argmax` 的**线程组大小**单因子扫描（仅测试，不进任何生产计划）。
//
// ## 为什么要问这个
//
// small 档 rec 的四份计划里，`n_head_argmax` 是**单条最贵**的 dispatch
// （0.739 / 0.761 / 0.435 / 1.277 ms，合计 3.22 ms/图 = rec 的 9.4%），比
// 任何一条 conv 都贵。它网格只有 `[rows,1,1]`（plan1 是 160 组），而**同一
// 份计划里**的 `n_channel Add.204` 用 2924 组跑 0.247 ms —— 按「每元素每次
// 触碰」折算，head 比它差 ~1.8×。
//
// 真形状（plan1）：`rows=160`、`cols=18710`（plan.rs 注释里的 `[T×18710]`，
// 与 `n_channel` 的 2924 组 = 160×18712/4/256 互证）、`cpad=18712`。一行
// = 74.8 KB。内核每行要**来回四趟**：行 max → 就地 exp（读+写）→ 8 条 lane
// 链求和 → 归一后 argmax。四趟都只有 256 个线程在搬 18712 个数。
//
// **并行度是死的**：phase 3 的 8 条链是 CPU 的 lane 序（lane k 累加
// j ≡ k (mod 8)），拆到别处就改序 —— 见 `n_head_argmax.metal` 头注释，
// 8 是上限。但 phase 1/2/4 是**乱序无关**的（max 结合律；exp 逐元素；
// argmax 的「首个严格最大、平局取最小下标」在任意划分下都等于全局解），
// 所以**换线程组大小是位级安全的**。
//
// 这是最便宜的一根杠杆：不动算法、不动网格，只把 256 提到 512/1024。
//
// ## 变体表
//
// | 名字 | 线程组 | 阶段 | exp | 要回答的问题 |
// |---|---|---|---|---|
// | `p_h_t256` | 256 | 全 | exp1 | 与生产逐字一致（对照组；应≈生产读数） |
// | `p_h_t512` | 512 | 全 | exp1 | 线程翻倍值多少 |
// | `p_h_t1024` | 1024 | 全 | exp1 | 顶到 Apple 上限值多少 |
// | `p_h_s1` | 256 | 1 | — | 只跑行 max：**一趟读**的地板 |
// | `p_h_s2` | 256 | 1+2 | exp1 | 多一趟读 + 一趟写 + exp 值多少 |
// | `p_h_s3` | 256 | 1+2+3 | exp1 | 8 条 lane 链值多少 |
// | `p_h_noexp` | 256 | 全 | 恒等 | exp1 多项式本身值多少 |
//
// `p_h_s*` 用**编译期常量** `STAGES` 门控，早退点对所有线程一致（无发散的
// barrier），但结果只对 `s4`（= `t256`）成立——其余三个只量时间。
//
// ## 纪律
//
// 本内核**不在** `KernelSet::SHADERS` 清单里，只在本模块编译、只被测试
// 调用。所有变体与 `n_head_argmax` **位级相同**（归约序变了但 max/argmax
// 与序无关，8 条链一字不动），所以测试里拿 `p_h_t256` 的输出当基准、
// 断言其余变体逐位相等。

#include <metal_stdlib>
using namespace metal;

// exp1：与 `n_head_argmax.metal` 逐字（那是 `n_softmax.metal` 的逐字拷贝）。
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

// 恒等「exp」：只为把 exp1 多项式本身的成本从 phase 2 里劈出来。
float exp_id(float x) { return x; }

// phase 3 的 8 条 lane 链，**按 `U` 展开**。
//
// ## 为什么展开是位级安全的
//
// 链本身（lane k 累加 j = k, k+8, k+16, …）**一个加法都没少、次序一个字
// 都没改**——展开只是让 U 条**地址无关**的装载先发出去，再按原顺序累加。
// 依赖链仍是 `(((l+a0)+a1)+a2)…`，浮点结合序不变 ⇒ 逐位相同。
//
// ## 为什么要展开（阶段归因的读数，rows=160，cols=18710）
//
//   s1（行 max，一趟读，256 线程）            0.122 ms
//   s2（+ 就地 exp，读+写）                   0.309 ms
//   s3（+ 8 条链，**一趟读，8 线程**）          0.668 ms   ← 增量 0.359
//   t256（+ argmax，一趟读，256 线程）         0.750 ms   ← 增量 0.082
//
// **同样搬一趟 18712 个数，8 个线程要 0.359 ms，256 个线程只要 0.082 ms
// （argmax）**——4.4× 的差距不是带宽（该趟合计只到 ~33 GB/s，不到本机
// 90 GB/s 的一半），是**每线程一条串行依赖装载**：2339 次 × ~38 ns 的
// 访存延迟，一次只有一个在飞。展开就是把 MLP 抬上去。
//
// `U == 1` 分支保持逐字原样，当对照。
template <int U>
static inline float chain_sum(
    device const float* data, uint base, uint q, uint lid)
{
    float l = 0.0;
    uint t = 0u;
    if (U == 1) {
        for (; t < q; t++) {
            l += data[base + t * 8u + lid];
        }
    } else {
        for (; t + uint(U) <= q; t += uint(U)) {
            float a[U];
            #pragma unroll
            for (int u = 0; u < U; u++) {
                a[u] = data[base + (t + uint(u)) * 8u + lid];
            }
            #pragma unroll
            for (int u = 0; u < U; u++) {
                l += a[u];
            }
        }
        for (; t < q; t++) {
            l += data[base + t * 8u + lid];
        }
    }
    return l;
}

// NT 直接传字面量（`256u` 这样带后缀）：数组维度与步长都用它，省掉 token
// 拼接。归约用 4 路折叠（`lid % 4u`），对任何 `nsg = NT/sgw` 都成立。
// `STAGES` 是编译期常量（1..4），早退点对所有线程一致。
#define HEAD_PROBE(NAME, NT, STAGES, EXPFN, UNROLL)                            \
    kernel void NAME(                                                          \
        device float* data [[buffer(0)]],                                      \
        device const uint* params [[buffer(1)]],                               \
        constant uint& p_off [[buffer(3)]],                                    \
        uint wg [[threadgroup_position_in_grid]],                              \
        uint lid [[thread_position_in_threadgroup]],                           \
        uint sgw [[threads_per_simdgroup]])                                    \
    {                                                                          \
        threadgroup float red[NT];                                             \
        threadgroup float2 ared[NT];                                           \
        uint P = p_off;                                                        \
        uint in_off = params[P + 0], out_off = params[P + 1];                  \
        uint rows = params[P + 2], cols = params[P + 3], cpad = params[P + 4]; \
        uint rpb = params[P + 5];                                              \
        uint row = wg;                                                         \
        if (row >= rows) return;                                               \
        if (params[0] != 0u && row >= params[0] * rpb) return;                 \
        uint base = in_off + row * cpad;                                       \
        uint nsg = NT / sgw;                                                   \
        uint sg = lid / sgw;                                                   \
                                                                               \
        float m = -3.4e38;                                                     \
        for (uint j = lid; j < cols; j += NT) {                                \
            m = max(m, data[base + j]);                                        \
        }                                                                      \
        float wm = simd_max(m);                                                \
        if (simd_is_first()) red[sg] = wm;                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        if (lid < nsg && lid % 4u == 0u) {                                     \
            float v = red[lid];                                                \
            if (lid + 1u < nsg) v = max(v, red[lid + 1u]);                     \
            if (lid + 2u < nsg) v = max(v, red[lid + 2u]);                     \
            if (lid + 3u < nsg) v = max(v, red[lid + 3u]);                     \
            red[lid / 4u] = v;                                                 \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        float rowmax = red[0];                                                 \
        for (uint g = 1u; g < (nsg + 3u) / 4u; g++) {                          \
            rowmax = max(rowmax, red[g]);                                      \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        if (STAGES == 1) {                                                     \
            if (lid == 0u) data[out_off + row * 2u] = rowmax;                  \
            return;                                                            \
        }                                                                      \
                                                                               \
        for (uint j = lid; j < cols; j += NT) {                                \
            data[base + j] = EXPFN(data[base + j] - rowmax);                   \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup); \
        if (STAGES == 2) {                                                     \
            if (lid == 0u) data[out_off + row * 2u] = rowmax;                  \
            return;                                                            \
        }                                                                      \
                                                                               \
        uint q = cols >> 3;                                                    \
        if (lid < 8u) {                                                        \
            red[lid] = chain_sum<UNROLL>(data, base, q, lid);                  \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        if (lid == 0u) {                                                       \
            float s = ((red[0] + red[4]) + (red[1] + red[5])) +                \
                      ((red[2] + red[6]) + (red[3] + red[7]));                 \
            for (uint j = q * 8u; j < cols; j++) {                             \
                s += data[base + j];                                           \
            }                                                                  \
            red[0] = 1.0 / s;                                                  \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        float inv = red[0];                                                    \
        if (STAGES == 3) {                                                     \
            if (lid == 0u) data[out_off + row * 2u] = inv;                     \
            return;                                                            \
        }                                                                      \
                                                                               \
        float bm = -3.4e38;                                                    \
        uint mi = 0u;                                                          \
        for (uint j = lid; j < cols; j += NT) {                                \
            float v = data[base + j] * inv;                                    \
            if (v > bm) {                                                      \
                bm = v;                                                        \
                mi = j;                                                        \
            }                                                                  \
        }                                                                      \
        float wbv = simd_max(bm);                                              \
        uint cand = (bm == wbv) ? mi : 0xFFFFFFFFu;                            \
        uint wbi = simd_min(cand);                                             \
        if (simd_is_first()) ared[sg] = float2(wbv, as_type<float>(wbi));      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        if (lid < nsg && lid % 4u == 0u) {                                     \
            float v = ared[lid].x;                                             \
            uint ix = as_type<uint>(ared[lid].y);                              \
            for (uint k = 1u; k < 4u && lid + k < nsg; k++) {                  \
                float v2 = ared[lid + k].x;                                    \
                uint ix2 = as_type<uint>(ared[lid + k].y);                     \
                if (v2 > v || (v2 == v && ix2 < ix)) {                         \
                    v = v2;                                                    \
                    ix = ix2;                                                  \
                }                                                              \
            }                                                                  \
            ared[lid / 4u] = float2(v, as_type<float>(ix));                    \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        float bv = ared[0].x;                                                  \
        uint bi = as_type<uint>(ared[0].y);                                    \
        for (uint g = 1u; g < (nsg + 3u) / 4u; g++) {                          \
            float v2 = ared[g].x;                                              \
            uint ix2 = as_type<uint>(ared[g].y);                               \
            if (v2 > bv || (v2 == bv && ix2 < bi)) {                           \
                bv = v2;                                                       \
                bi = ix2;                                                      \
            }                                                                  \
        }                                                                      \
        if (lid == 0u) {                                                       \
            data[out_off + row * 2u] = bv;                                     \
            data[out_off + row * 2u + 1u] = as_type<float>(bi);                \
        }                                                                      \
    }

HEAD_PROBE(p_h_t256, 256u, 4, exp1, 1)
HEAD_PROBE(p_h_t512, 512u, 4, exp1, 1)
HEAD_PROBE(p_h_t1024, 1024u, 4, exp1, 1)
HEAD_PROBE(p_h_s1, 256u, 1, exp1, 1)
HEAD_PROBE(p_h_s2, 256u, 2, exp1, 1)
HEAD_PROBE(p_h_s3, 256u, 3, exp1, 1)
HEAD_PROBE(p_h_noexp, 256u, 4, exp_id, 1)
// 链展开（位级安全，见 chain_sum 的注释）：U = 4 / 8 / 16。
HEAD_PROBE(p_h_u4, 256u, 4, exp1, 4)
HEAD_PROBE(p_h_u8, 256u, 4, exp1, 8)
HEAD_PROBE(p_h_u16, 256u, 4, exp1, 16)
HEAD_PROBE(p_h_u32, 256u, 4, exp1, 32)
HEAD_PROBE(p_h_u64, 256u, 4, exp1, 64)
