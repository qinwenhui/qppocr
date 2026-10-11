// depthwise conv 的**线程组瓦片暂存版**（NHWC f32，group=Ci=Co）——**Metal 独有**。
//
// 语义与 `n_conv_dw` **逐位相同**（证明见下「为什么是逐位的」），只是把热循环
// 的数据来源从设备内存换成线程组内存。plan 侧按形状选路，判据见
// `PlanBuilder::dw_tile`；**3×3 及以下不走这条**（实测更慢，见下）。
//
// ## 为什么挖这里（三轮探针的结论，`logs/n_conv_dw_dig_20261010.md` + 第三轮）
//
// 前两轮钉下的事实：
//   * `p_dw_fma`（地址数学与装载全去、只留 FMA）= **4.66×** ⇒ 靶心是「每 tap
//     那十几条整数/分支指令」，不是带宽；
//   * 但**逐项**去掉都不值钱：去输入装载 1.01×、去权重装载 1.05×、
//     去边界判断 1.15× ⇒ 钱不在某一条指令上，在那**一整段**上；
//   * 所有「换个写法去掉除法/加宽每线程」的结构改动**全部更慢**
//     （`nodiv` 0.74×、`u9` 0.72×、`wide4` 0.84×、`intun` 0.59×）——因为
//     循环体变大 ⇒ 寄存器压力/取指吃掉收益。
//
// 唯一没试过、且**同时把循环体变小**的一条：输入瓦片先搬进线程组内存，
// 热循环里就没有运行期除法、没有 `ih/iw` 的乘加与两个 compare+分支、没有
// `px*cv4+c4` 的设备地址数学；权重也进共享内存，热循环那次「每线程不同 c4
// ⇒ 全分叉 gather」变成每线程组只搬一趟。代价是一个 prologue + 一次 barrier。
//
// 竞品 `conv_dw4t`（`SimdPaddleOCR/.../Shaders/conv.metal`）正是这个结构，
// 所以不是纸上谈兵。差别只在他们**顺带用了 fp16**，这里保持 f32 —— 自变量
// 只留「暂存」一个。
//
// ## 实测（探针 `p_dw_tile`，`cargo test --release -p qppocr-metal conv_dw_variant_scan`）
//
//   9×9 c256 136×240（det 首恶 Conv.64）  5.324 → **2.863 ms**  **1.86×**
//   9×9 c256  68×120（Conv.62）           1.339 → **0.761**      **1.76×**
//   9×9 c64  136×240（Conv.69）           1.325 → **0.713**      **1.86×**
//   3×3 c128 136×240                      0.365 →  0.466         0.78×  ← 更慢
//   3×3 c512  34×60                       0.098 →  0.113         0.86×  ← 更慢
//
// 3×3 更慢是**可解释的、不是噪声**：那两条算下来 0.365 ms / 0.36 ms 的
// DRAM 下界 —— 已经贴着访存天花板，暂存只会加上一层来回。**所以判据里
// 必须排除小 tap**（见 `PlanBuilder::dw_tile` 的 `taps >= 25`）。
//
// ## 为什么是逐位的（不是"数值等价"）
//
// 1. **累加次序一字未改**：base 是 `for t in 0..taps`（`ky = t/kW`，
//    行主序），这里是 `for dy / for dx` —— 同一个行主序序列。
// 2. **每个在界 tap 的乘数逐位相同**：输入取 `d[ib + (gy*inW+gx)*cv4 + c4]`、
//    权重取 `d[wb + t*cv4 + c4]`，与 base 逐字同式。
// 3. **越界 tap 的差别是安全 0**：base 是 `continue`（不加）；这里是加
//    `0.0 * w`。`w` 有限 ⇒ `0.0*w = ±0.0`，而 `acc` 从 `+0.0` 起步，且
//    IEEE 加法只有「两个 -0.0 相加」才得 -0.0 ⇒ **acc 取值域里没有 -0.0**，
//    于是 `acc + ±0.0 == acc` 恒成立（含 acc 恰为 +0.0 时）。
//    ⇒ 越界 tap 一个都不改结果。唯一能破这条的是权重含 inf/NaN，模型里没有。
//
// 于是本内核可以直接要求 **tiny/small 全语料逐位不变**。
//
// ## 与 Vulkan 的关系
//
// `n_conv_dw.metal` 与 `qppocr-gpu/shaders/n_conv_dw.comp` 逐行对应，但
// **本文件是 Metal 独有**：`PlanBuilder::dw_tile` 在 Vulkan 侧恒 false
// （与 `n_head_argmax` / `n_convt4w` 同款）。误开会让 Vulkan 派发到不存在的
// 内核——硬失败，非静默降级。
//
// ## 网格
//
// 一个线程组 = 一个 **16×16 输出瓦片 × 一个 c4 组**，恒 256 线程。
// `threadgroup_position_in_grid = (tile*cv4 + c4, batch, 1)`，
// 组数 = `tilesX · tilesY · (cDim/4)`（`plan.rs` 的 `dw_tile_grid` 现算）。
// 与 base 的 `[div256(m*cv4), nb, 1]` 完全不同 —— 改这里必须两边一起改。
#include <metal_stdlib>
using namespace metal;

constant uint OFF_NONE = 0xFFFFFFFFu;

// exp1/erf1：conv.comp 的逐字拷贝（与 CPU 位级一致的多项式，非硬件 exp）。
// 标准库没有这两个函数，每个用到 act 的内核各带一份（`n_head_argmax.metal`
// 同款），不跨文件共享 —— 每个 .metal 独立成库。
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

// 瓦片边长与支持的最大核 —— 都必须是编译期常量（数组尺寸）。
// 24×24 float4 = 9 KB 瓦片 + 81×16 B = 1.3 KB 权重 ⇒ 每组 ~10.3 KB 共享内存。
#define DW_TILE 16
#define DW_MAXK 9

kernel void n_conv_dw_tile(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 wg [[threadgroup_position_in_grid]],
    // MSL 要求同内核的位置属性**同型**（不能 uint3 混 uint），故这里也取
    // uint3 再取 `.x` —— `n_head_argmax` 头注释里记过同一个坑。
    uint3 tp3 [[thread_position_in_threadgroup]])
{
    uint tid = tp3.x;

    threadgroup float4 shTile[(DW_TILE + DW_MAXK - 1) * (DW_TILE + DW_MAXK - 1)];
    threadgroup float4 shW[DW_MAXK * DW_MAXK];

    // 参数块与 `n_conv_dw` 逐字同布局（P+0..P+19）。
    uint P = p_off;
    uint in_off = params[P + 0], w_off = params[P + 1], b_off = params[P + 2];
    uint out_off = params[P + 3];
    uint mDim = params[P + 4], cDim = params[P + 5], outW = params[P + 6];
    uint inW = params[P + 7], inH = params[P + 8];
    uint taps = params[P + 9], kW = params[P + 10];
    uint sH = params[P + 11], sW = params[P + 12];
    uint padT = params[P + 13], padL = params[P + 14];
    uint act = params[P + 15];
    float c1 = as_type<float>(params[P + 16]);
    float c2 = as_type<float>(params[P + 17]);
    float c3 = as_type<float>(params[P + 18]);
    uint r_off = params[P + 19]; // 残差（NHWC 同布局；OFF_NONE=无）

    uint b = wg.y;
    // real_n 早退：批维补齐的空行零算力。注意**必须在 barrier 之前就返回**
    // —— 本内核每个线程组自带 barrier，提前返回的线程组不参与别人的 barrier
    // （组内所有线程走同一条路径，所以组内一致）。
    if (params[0] != 0u && b >= params[0]) return;
    // 形状自锁：三条都与线程无关（参数块全组同值），所以早退不会让组内分叉。
    // 好处是把「plan 侧判据写错」从**读飞打瘸 GPU** 降级成「算出旧结果」——
    // `stW*stH` 一旦超过 24×24 数组就会越界读线程组内存（不报错、只出错值），
    // `taps > 81` 则直接越界**写**。plan 侧的 `dw_tile` 判据是唯一调用方约定，
    // 这里是兜底。
    if (sH != 1u || sW != 1u) return;
    if (kW == 0u || kW > DW_MAXK || taps > DW_MAXK * DW_MAXK) return;
    if (taps / kW > DW_MAXK) return;

    uint cv4 = cDim >> 2;
    uint outH = mDim / outW;
    uint kH = taps / kW;
    uint tilesX = (outW + DW_TILE - 1u) / DW_TILE;

    uint g = wg.x;
    uint c4 = g % cv4;
    uint tile = g / cv4;
    uint tx = tile % tilesX, ty = tile / tilesX;

    uint stW = DW_TILE * sW + kW - 1u;
    uint stH = DW_TILE * sH + kH - 1u;
    int iyT = int(ty * DW_TILE * sH) - int(padT);
    int ixT = int(tx * DW_TILE * sW) - int(padL);

    uint ib = (in_off >> 2) + b * inH * inW * cv4;
    uint wb = w_off >> 2;
    uint ob = (out_off >> 2) + b * mDim * cv4;

    // ── prologue：暂存输入瓦片（越界填 0）与权重 ──────────────────────
    uint tileElems = stW * stH;
    for (uint e = tid; e < tileElems; e += 256u) {
        uint iy = e / stW, ix = e - iy * stW;
        int gy = iyT + int(iy), gx = ixT + int(ix);
        float4 v = float4(0.0);
        if (gy >= 0 && uint(gy) < inH && gx >= 0 && uint(gx) < inW) {
            v = d[ib + (uint(gy) * inW + uint(gx)) * cv4 + c4];
        }
        shTile[e] = v;
    }
    // 权重：本线程组的 c4 固定 ⇒ 这 81 次装载全组只搬一趟
    // （base 里是每线程各搬 taps 次，且同 warp 内地址全分叉）。
    for (uint e = tid; e < taps; e += 256u) {
        shW[e] = d[wb + e * cv4 + c4];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 屏障之后才允许早退（否则瓦片没填完就有线程跑了）。
    uint ox = tx * DW_TILE + (tid & (DW_TILE - 1u));
    uint oy = ty * DW_TILE + (tid >> 4);
    if (ox >= outW || oy >= outH) return;

    uint lx0 = (tid & (DW_TILE - 1u)) * sW;
    uint ly0 = (tid >> 4) * sH;

    // ── 热循环：全部来自线程组内存，无除法、无边界判断、无设备地址数学。
    // 次序与 base 逐位相同（t 行主序 ⇒ dy 外层 / dx 内层）。
    float4 acc = float4(0.0);
    for (uint dy = 0u; dy < kH; dy++) {
        for (uint dx = 0u; dx < kW; dx++) {
            acc += shW[dy * kW + dx] * shTile[(ly0 + dy) * stW + lx0 + dx];
        }
    }

    // 尾部（bias / act / 残差 / 写回）与 `n_conv_dw` 逐字相同。
    uint m = oy * outW + ox;
    if (b_off != OFF_NONE) acc += d[(b_off >> 2) + c4]; // bias 全批共享
    if (act == 2u) acc = max(acc, float4(0.0));
    else if (act == 1u) {
        float inv = 1.0 / c1;
        acc = c3 * acc * (float4(erf1(acc.x * inv), erf1(acc.y * inv),
                                 erf1(acc.z * inv), erf1(acc.w * inv)) + c2);
    }
    if (r_off != OFF_NONE) acc += d[(r_off >> 2) + b * mDim * cv4 + m * cv4 + c4];
    d[ob + m * cv4 + c4] = acc;
}
