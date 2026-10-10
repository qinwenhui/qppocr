// `n_conv8_mma` 的**四臂诊断**：把热循环的 75% 天花板劈成「装载指令条数」
// 与「装载流量/延迟」两半，再给后半找一条**能落地**的处方。
//
// 前三个（`p_mma_base` / `p_mma_small` / `p_mma_hoist`）是诊断件，**故意算错，
// 只看时间**；第四个 `p_mma_wfrag` **结果正确**，是候选内核。
//
// ## 要回答的问题
//
// `conv8_mma_ksweep` 把这条内核的固定成本（尾声）与边际成本（热循环）分开了：
// 边际斜率 = **0.0262 µs / K步（每步 8 个 K）**，而理想（3340 GFLOPS 峰）
// 只要 0.0196 ⇒ **热循环只跑到 75%**。这个 25% 的缺口是哪来的？
//
// 标量 `n_conv8` 那边的答案早就钉死了（`p_fmaonly` 2.19~9.95×、
// `p_wsmall` 1.01~4.49×）——是**装载指令条数**，不是字节。`n_conv8_mma`
// 把装载压到 6 条/8 MMA（load:FMA 从 12:8 掉到 6:4096），但**没有量过**
// 剩下这 6 条值多少。本探针补上这一格。
//
// ## 三臂（单变量，逐条对照）
//
//   p_mma_base   与生产 `n_conv8_mma` **逐字节相同**（对照组）。
//   p_mma_small  装载条数与 MMA 条数**一字不改**，把 A 与 B 的基址**与行距**
//                都收进 1 KB ⇒ **足迹≈0、延迟≈L1**。它劈出的是
//                「**字节/行数/延迟**」那一半（行距也收的理由见内核注释）。
//   p_mma_hoist  6 条 `simdgroup_load` **整个挪出 K 循环**（只在循环前装一次），
//                MMA 条数与累加链**一字不改** ⇒ **装载指令≈0**。它劈出的是
//                「**发射槽**」那一半。
//
// ## 怎么读
//
//   base ≈ small ≈ hoist  ⇒ 75% 不是装载造成的：是 MMA 自身在这台机器上的
//                            发射率上限 ⇒ **这条内核已经到顶，别动它**；
//   base ≫ hoist（≈ small）⇒ 卡在**装载指令的发射槽** ⇒ 值得换更大的每
//                            simdgroup 瓦片（16×32 → 16×64 / 32×32，
//                            每 MMA 的装载条数降 17~33%）；
//   base ≫ small，而 hoist ≈ base ⇒ 卡在**流量/延迟**（权重 footprint 或
//                            输入行跨步）⇒ 该动的是 L2 局部性 / 权重驻留，
//                            不是条数。
//
// ## 诚实边界
//
// * `p_mma_hoist` **数值全错**（K 维每一段用的是同一组操作数）——它只回答
//   「删掉装载指令能省多少」，**不是**一条能落地的内核。
// * `p_mma_small` 的数应当理解成**下界**：掩掉读侧流量不一定顺带掩掉
//   输出的写流量（尾声没动）。
// * 三臂都在同一个 arena 上跑，`d` 的段布局与生产逐字段一致。
#include <metal_stdlib>
using namespace metal;

constant uint OFF_NONE = 0xFFFFFFFFu;

// exp1/erf1/act1：`n_conv8_mma.metal` 的逐字拷贝（与 CPU 位级一致的多项式）。
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

float erf1(float x) {
    float ax = fabs(x);
    float t = 1.0 / fma(0.3275911, ax, 1.0);
    float p = 1.0614054;
    p = fma(p, t, -1.453152027);
    p = fma(p, t, 1.421413741);
    p = fma(p, t, -0.284496736);
    p = fma(p, t, 0.2548296);
    p *= t;
    float e = exp1(-ax * ax);
    float r = fma(-p, e, 1.0);
    return as_type<float>(as_type<uint>(r) | (as_type<uint>(x) & 0x80000000u));
}

float act1(float v, uint act, float c1, float c2, float c3) {
    if (act == 2u) return max(v, 0.0f);
    if (act == 1u) {
        float inv = 1.0 / c1;
        return c3 * v * (erf1(v * inv) + c2);
    }
    return v;
}

// 尾声：与 `n_conv8_mma.metal` 的循环**逐字段同义**。三臂共用，保证
// 「尾声」在两臂之间是同一个常量，差分只落在热循环上。
inline void mma_epilogue(threadgroup float* ost,
                         device float* of,
                         device const float* f32v,
                         uint m0,
                         uint n0,
                         uint mDim,
                         uint nDim,
                         uint ob,
                         uint b_off,
                         uint r_off,
                         uint batch,
                         uint act,
                         float c1,
                         float c2,
                         float c3,
                         uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 64u * 64u; e += 256u) {
        uint m = m0 + (e >> 6);
        uint n = n0 + (e & 63u);
        if (m >= mDim) continue;
        float v = ost[e];
        if (b_off != OFF_NONE) v += f32v[(b_off >> 2) * 4u + n];
        v = act1(v, act, c1, c2, c3);
        if (r_off != OFF_NONE)
            v += f32v[(r_off >> 2) * 4u + batch * mDim * nDim + m * nDim + n];
        of[ob + m * nDim + n] = v;
    }
}

// 参数解包 + 分片落 threadgroup：三臂共用的前半段。
#define MMA_PROLOGUE                                                       \
    device const float* f32v = (device const float*)d;                     \
    device float* of = (device float*)d;                                   \
    uint P = p_off;                                                        \
    uint in_off = params[P + 0], w_off = params[P + 1], b_off = params[P + 2]; \
    uint out_off = params[P + 3];                                          \
    uint mDim = params[P + 4], nDim = params[P + 5];                       \
    uint inH = params[P + 7], inW = params[P + 8], cin4v = params[P + 9];  \
    uint act = params[P + 16];                                             \
    float c1 = as_type<float>(params[P + 17]);                             \
    float c2 = as_type<float>(params[P + 18]);                             \
    float c3 = as_type<float>(params[P + 19]);                             \
    uint r_off = params[P + 20];                                           \
    uint b = wg.z;                                                         \
    if (params[0] != 0u && b >= params[0]) return;                         \
    uint nv = nDim >> 2;                                                   \
    uint kf = cin4v * 4u;                                                  \
    uint m0 = wg.x * 64u;                                                  \
    uint n0 = wg.y * 64u;                                                  \
    uint tid = tid3.x;                                                     \
    uint sgid = tid >> 5;                                                  \
    uint sm = m0 + (sgid >> 1) * 16u;                                      \
    uint sn = n0 + (sgid & 1u) * 32u;                                      \
    uint ib = (in_off >> 2) * 4u + b * inH * inW * kf;                      \
    uint wb = (w_off >> 2) * 4u;                                           \
    uint ob = (out_off >> 2) * 4u + b * mDim * nDim;                       \
    simdgroup_float8x8 acc[2][4];                                          \
    for (uint i = 0u; i < 2u; i++)                                         \
        for (uint j = 0u; j < 4u; j++)                                     \
            acc[i][j] = simdgroup_float8x8(0.0f);

#define MMA_ACCS                                                           \
    simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);           \
    simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);           \
    simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]);           \
    simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);           \
    simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);           \
    simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);           \
    simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]);           \
    simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);

#define MMA_EPILOGUE_CALL                                                  \
    threadgroup float ost[64u * 64u];                                      \
    uint r0 = (sgid >> 1) * 16u;                                           \
    uint c0 = (sgid & 1u) * 32u;                                           \
    for (uint i = 0u; i < 2u; i++)                                         \
        for (uint j = 0u; j < 4u; j++)                                     \
            simdgroup_store(acc[i][j], ost + (r0 + i * 8u) * 64u + c0 + j * 8u, 64u); \
    mma_epilogue(ost, of, f32v, m0, n0, mDim, nDim, ob, b_off, r_off, b,   \
                 act, c1, c2, c3, tid);

#define MMA_SIG                                                            \
    device float4* d [[buffer(0)]],                                        \
    device const uint* params [[buffer(1)]],                               \
    constant uint& p_off [[buffer(3)]],                                    \
    uint3 tid3 [[thread_position_in_threadgroup]],                         \
    uint3 wg [[threadgroup_position_in_grid]]

// ── 对照组：与生产 `n_conv8_mma` 逐字节相同 ──────────────────────────────
kernel void p_mma_base(MMA_SIG) {
    MMA_PROLOGUE
    for (uint kb = 0u; kb < kf; kb += 8u) {
        simdgroup_float8x8 a0, a1;
        simdgroup_load(a0, f32v + ib + sm * kf + kb, kf);
        simdgroup_load(a1, f32v + ib + (sm + 8u) * kf + kb, kf);
        simdgroup_float8x8 b0, b1, b2, b3;
        simdgroup_load(b0, f32v + wb + kb * nDim + sn, nDim);
        simdgroup_load(b1, f32v + wb + kb * nDim + sn + 8u, nDim);
        simdgroup_load(b2, f32v + wb + kb * nDim + sn + 16u, nDim);
        simdgroup_load(b3, f32v + wb + kb * nDim + sn + 24u, nDim);
        MMA_ACCS
    }
    MMA_EPILOGUE_CALL
}

// ── 流量≈0：条数不变，把基址与**行距**都收进 1 KB ─────────────────────────
//
// **为什么行距也要收**：`simdgroup_load(m, ptr, stride)` 取的是
// `ptr + row*stride + col`。只掩 `ptr` 的话 8 行的**行距仍是 kf / nDim**
// （几千个 float），8 行照样散在几十 KB 上——**流量根本没塌**，那样量到的
// 是「基址重排」不是「足迹归零」。所以这里行距一并给 8：整块分片落进
// `[0, 320)` 个 float（≈1.3 KB）里，全 L1 常驻。
//
// 代价：**行距这个操作数变了**（`kf`/`nDim` → 常量 8）。装载指令的**条数、
// 类型、MMA 条数**都不变，但严格说它已不是「只改地址」——记在这里，
// 读表时按「流量≈0 的近似下界」用。
//
// 掩到 `[0,255]` 是绝对下标，落在 arena 最前 1 KB（必然在缓冲区内）。
kernel void p_mma_small(MMA_SIG) {
    MMA_PROLOGUE
    for (uint kb = 0u; kb < kf; kb += 8u) {
        simdgroup_float8x8 a0, a1;
        simdgroup_load(a0, f32v + ((ib + sm * kf + kb) & 0xFFu), 8u);
        simdgroup_load(a1, f32v + ((ib + (sm + 8u) * kf + kb) & 0xFFu), 8u);
        simdgroup_float8x8 b0, b1, b2, b3;
        simdgroup_load(b0, f32v + ((wb + kb * nDim + sn) & 0xFFu), 8u);
        simdgroup_load(b1, f32v + ((wb + kb * nDim + sn + 8u) & 0xFFu), 8u);
        simdgroup_load(b2, f32v + ((wb + kb * nDim + sn + 16u) & 0xFFu), 8u);
        simdgroup_load(b3, f32v + ((wb + kb * nDim + sn + 24u) & 0xFFu), 8u);
        MMA_ACCS
    }
    MMA_EPILOGUE_CALL
}

// ── 权重分片分块（fragment-blocked）：只改 B 的行距，A 一字不动 ─────────────
//
// `p_mma_base` 的 4 条 B 装载是 `wb + kb*nDim + sn`，**行距 = nDim**（几百到
// 几千个 float）。`simdgroup_load` 取 8 行 × 8 列，每行只有 32 B，而 8 行彼此
// 相距 `nDim*4` 字节 ⇒ **一条装载打散成 8 段各自的 32 B**，并且每段只用到
// 所属 cache line 的一小块。
//
// 这里把权重**离线**重排成「每个 8×8 分片连续 64 个 float」：
//
//   wf[((ci_blk) * nv8 + co_blk) * 64 + ci_off * 8 + co_off]
//
// 于是同一条 `simdgroup_load` 变成**一段连续的 256 B**（行距 = 8 个 float =
// 32 B，8 行严丝合缝）。
//
// **单变量**：权重是**静态**量（离线重排一次，与既有的 `repack_conv_w` 同族
// 成本），输入/bias/输出段与 `p_mma_base` 逐字节相同，A 的 2 条装载一字不改。
// 落在「4/6 条装载」上，是 `p_mma_small`（6/6 条全收，1.22×）的**部分版**。
//
// 前置条件（生产门禁已保证）：`kf % 8 == 0`（`cpad4(cin) % 8 == 0`）与
// `nDim % 64 == 0` ⇒ `nv8 = nDim/8` 整除、`sn >> 3` 整除。
kernel void p_mma_wfrag(MMA_SIG) {
    MMA_PROLOGUE
    uint nv8 = nDim >> 3;
    for (uint kb = 0u; kb < kf; kb += 8u) {
        simdgroup_float8x8 a0, a1;
        simdgroup_load(a0, f32v + ib + sm * kf + kb, kf);
        simdgroup_load(a1, f32v + ib + (sm + 8u) * kf + kb, kf);
        simdgroup_float8x8 b0, b1, b2, b3;
        uint bb = ((kb >> 3) * nv8 + (sn >> 3)) * 64u;
        simdgroup_load(b0, f32v + wb + bb, 8u);
        simdgroup_load(b1, f32v + wb + bb + 64u, 8u);
        simdgroup_load(b2, f32v + wb + bb + 128u, 8u);
        simdgroup_load(b3, f32v + wb + bb + 192u, 8u);
        MMA_ACCS
    }
    MMA_EPILOGUE_CALL
}

// ── 单侧收窄：把 `p_mma_small` 的 1.22× 劈成 A 侧与 B 侧 ────────────────
//
// `p_mma_wfrag` 已经证伪了「B 侧是行数/合并问题」：在 nDim=64（行距 256 B）
// 上权重分片连续 **1.00×**。所以 1.22× 要么在 **A 侧（激活）**，要么是
// **延迟**（`p_mma_small` 让 510 个线程组全部命中同一块 1.3 KB L1，延迟≈0），
// 与行数无关。这两条处方完全不同（前者 = 改 A 的访存结构，后者 = 提高占用率
// 藏延迟），必须先劈开。
//
// 两臂都只收**一侧**（另一侧与 `p_mma_base` 逐字节相同、含行距），
// 于是：
//
//   asmall ≈ base 而 bsmall ≪ base  ⇒ 全在 B 侧；**再按 wfrag 的 1.00× 往下查**；
//   bsmall ≈ base 而 asmall ≪ base  ⇒ 全在 A 侧 ⇒ 处方是激活侧的访存结构；
//   两侧各占一半                      ⇒ 是**延迟**，与哪一侧无关。
//
// 数值都是错的（诊断件，只看时间）。
kernel void p_mma_asmall(MMA_SIG) {
    MMA_PROLOGUE
    for (uint kb = 0u; kb < kf; kb += 8u) {
        simdgroup_float8x8 a0, a1;
        simdgroup_load(a0, f32v + ((ib + sm * kf + kb) & 0xFFu), 8u);
        simdgroup_load(a1, f32v + ((ib + (sm + 8u) * kf + kb) & 0xFFu), 8u);
        simdgroup_float8x8 b0, b1, b2, b3;
        simdgroup_load(b0, f32v + wb + kb * nDim + sn, nDim);
        simdgroup_load(b1, f32v + wb + kb * nDim + sn + 8u, nDim);
        simdgroup_load(b2, f32v + wb + kb * nDim + sn + 16u, nDim);
        simdgroup_load(b3, f32v + wb + kb * nDim + sn + 24u, nDim);
        MMA_ACCS
    }
    MMA_EPILOGUE_CALL
}

kernel void p_mma_bsmall(MMA_SIG) {
    MMA_PROLOGUE
    for (uint kb = 0u; kb < kf; kb += 8u) {
        simdgroup_float8x8 a0, a1;
        simdgroup_load(a0, f32v + ib + sm * kf + kb, kf);
        simdgroup_load(a1, f32v + ib + (sm + 8u) * kf + kb, kf);
        simdgroup_float8x8 b0, b1, b2, b3;
        simdgroup_load(b0, f32v + ((wb + kb * nDim + sn) & 0xFFu), 8u);
        simdgroup_load(b1, f32v + ((wb + kb * nDim + sn + 8u) & 0xFFu), 8u);
        simdgroup_load(b2, f32v + ((wb + kb * nDim + sn + 16u) & 0xFFu), 8u);
        simdgroup_load(b3, f32v + ((wb + kb * nDim + sn + 24u) & 0xFFu), 8u);
        MMA_ACCS
    }
    MMA_EPILOGUE_CALL
}

// ── 尾声：劈成「算术+输出写」与「16 KB 线程组内存」两半 ──────────────────
//
// `p_mma_hoist` 给热循环钉了上界（1.44×），尾声这条路同样需要一个**分母**，
// 否则「去暂存 / 去屏障 / 逐步长尾声」全是没法估值的处方。
//
// **第一版的教训**：直接删掉尾声（只 `simdgroup_store(acc[0][0], …)` 再读一个
// 元素）量出 2.28×——**这个数是不可能的**：折算 5445 GFLOPS，超过本机 3340 的
// 纯 FMA 峰。原因：只有 `acc[0][0]` 被消费，**另外 7 条累加链连同它们的 B 装载
// 被死代码消除掉了**。比的是「删掉 7/8 的 MMA」，不是「删掉尾声」。
//
// 修法：8 个分片**全都存**（都活着），并把每线程的收尾压到 512 个元素 + 1 次
// 写。两臂唯一的差别是**声明出来的线程组数组大小**：
//
//   `p_mma_ost16`  `threadgroup float ost[64*64]`（16 KB，与生产同款）
//   `p_mma_ost2`   `threadgroup float ost[64*8]` （2 KB，只有声明变了）
//
// 于是：
//
// | 比值 | 含义 |
// |---|---|
// | `base / ost16` | 尾声**算术 + 输出写流量**的上界 |
// | `ost16 / ost2` | **16 KB 线程组内存**卡掉多少常驻线程组（并行度） |
//
// 两臂都**不是**生产内核（不读 bias/residual、不做激活、只写 1 个输出元素）。
#define MMA_TAIL(OSTSZ, LAST)                                             \
    threadgroup float ost[OSTSZ];                                         \
    for (uint i = 0u; i < 2u; i++)                                        \
        for (uint j = 0u; j < 4u; j++)                                    \
            simdgroup_store(acc[i][j], ost + (i * 4u + j) * 64u, 8u);      \
    if (tid == 0u) ost[LAST] = 0.0f;                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    float s = 0.0f;                                                       \
    for (uint e = tid; e < 512u; e += 256u) s += ost[e];                   \
    if (tid == 0u) of[ob + sm * nDim + sn] = s + ost[LAST];

// 热循环与 `p_mma_base` 逐字节相同。
#define MMA_HOT                                                            \
    for (uint kb = 0u; kb < kf; kb += 8u) {                                \
        simdgroup_float8x8 a0, a1;                                         \
        simdgroup_load(a0, f32v + ib + sm * kf + kb, kf);                  \
        simdgroup_load(a1, f32v + ib + (sm + 8u) * kf + kb, kf);           \
        simdgroup_float8x8 b0, b1, b2, b3;                                 \
        simdgroup_load(b0, f32v + wb + kb * nDim + sn, nDim);              \
        simdgroup_load(b1, f32v + wb + kb * nDim + sn + 8u, nDim);         \
        simdgroup_load(b2, f32v + wb + kb * nDim + sn + 16u, nDim);        \
        simdgroup_load(b3, f32v + wb + kb * nDim + sn + 24u, nDim);        \
        MMA_ACCS                                                           \
    }

kernel void p_mma_ost16(MMA_SIG) {
    MMA_PROLOGUE
    MMA_HOT
    MMA_TAIL(64u * 64u, 4095u)
}

kernel void p_mma_ost2(MMA_SIG) {
    MMA_PROLOGUE
    MMA_HOT
    MMA_TAIL(64u * 8u, 511u)
}

// ── 装载指令≈0：6 条 load 挪出 K 循环，MMA 条数与累加链一字不改 ──────────
//
// **数值是错的**（K 维每段用同一组操作数）——这是诊断件，不是内核。
kernel void p_mma_hoist(MMA_SIG) {
    MMA_PROLOGUE
    simdgroup_float8x8 a0, a1, b0, b1, b2, b3;
    simdgroup_load(a0, f32v + ib + sm * kf, kf);
    simdgroup_load(a1, f32v + ib + (sm + 8u) * kf, kf);
    simdgroup_load(b0, f32v + wb + sn, nDim);
    simdgroup_load(b1, f32v + wb + sn + 8u, nDim);
    simdgroup_load(b2, f32v + wb + sn + 16u, nDim);
    simdgroup_load(b3, f32v + wb + sn + 24u, nDim);
    for (uint kb = 0u; kb < kf; kb += 8u) {
        MMA_ACCS
    }
    MMA_EPILOGUE_CALL
}
