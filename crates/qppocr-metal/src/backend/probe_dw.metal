// depthwise 探针：把 `n_conv_dw` 的热循环拆开量。
//
// ## 动机
//
// medium det 里 `n_conv_dw` 只做 **2.60 GFLOP（全 det 的 3.2%）**，却占
// **15% 的设备时间**（12.6 / 82.4 ms）。最大的那条（`Conv.64`，9×9，
// m=32640，c=256，grid=[8160,1,1]）单条 **6.54 ms** 做 1.354 GFLOP
// = **207 GFLOPS**，而 `n_conv8h` 同机 1400+。而它自己那条 dispatch 的
// 下界（读 33 MB + 写 33 MB @ 93 GB/s）+ 纯 FMA 只有 ~0.7 ms ——
// **离下界 9 倍**。
//
// fp16 权重对它几乎无效（`QPPOCR_GPU_KERNEL_PROF`：23.5 → 21.9 ms），
// 因为它的权重只有 81×64×16 = 83 KB，本来就不缺带宽。
//
// ## 热循环每 tap（9×9 要付 81 次）
//
//     ky = t / kW        // kW 是**运行期**值 —— 真整数除法（倒数乘法序列）
//     kx = t - ky*kW
//     ih = oh*sH + ky - padT;  iw = ow*sW + kx - padL
//     if (ih<0 || ih>=inH || iw<0 || iw>=inW) continue;   // 2 比较 + 分支
//     px = ih*inW + iw                                    // 1 乘法
//     x = d[ib + px*cv4 + c4]                             // 1 乘法 + 装载
//     w = d[wb + t*cv4 + c4]                              // 装载
//     acc += x*w                                          // 4 FMA
//
// **每 4 个 FMA 要配十几条整数/分支指令和 2 次装载。** 拿 conv8 比：那边是
// 8 个 FMA 配 3 次装载、地址增量推进。如果 dw 真被整数流水线/分支吃掉，
// 那「加宽装载」这类访存优化压根打不中靶子 —— 得先知道靶子在哪。
//
// ## 五个变体切法（同网格、同覆盖、同迭代次数）
//
//   p_dw_base   控制组，热循环**逐字** = 生产 `n_conv_dw`
//   p_dw_nodiv  只把 `t/kW` 换成嵌套 ky/kx 循环（去掉运行期除法）
//   p_dw_now    权重装载换成常数（去掉 1 条装载 + 那份流量，指令数只减 1）
//   p_dw_fma    地址数学与装载全去、只留 FMA —— **这一档全部可动的量**
//   p_dw_nobc   去掉边界判断（只在**无 padding** 的配置上跑，见下）
//
// `p_dw_nobc` 在带 padding 的配置上会读越界（ih 为负 → uint 回绕成天文
// 数字），所以它只在 `build_dw_case(..., no_pad=true)` 那份上跑：pad=0、
// outH = inH-kH+1，此时每个 tap 恒在界内，删掉判断不会越界。它**不与
// base 的绝对值可比**（m 变了），只与**同无 pad 配置上的 `p_dw_base`**
// 配对读差 —— 那才是分支的代价。
#include <metal_stdlib>
using namespace metal;

constant uint OFF_NONE = 0xFFFFFFFFu;

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

// 尾部（bias / act / 残差 / 写回）与 `n_conv_dw` 逐字相同 —— 变体之间
// 唯一的自变量必须落在热循环里，尾部抄错会把噪声算进结论。
inline void dw_tail(device float4* d, float4 acc, uint P, device const uint* params,
                    uint b_off, uint r_off, uint act, float c1, float c2, float c3,
                    uint ob, uint m, uint cv4, uint c4, uint b) {
    if (b_off != OFF_NONE) acc += d[(b_off >> 2) + c4];
    if (act == 2u) acc = max(acc, float4(0.0));
    else if (act == 1u) {
        float inv = 1.0 / c1;
        acc = c3 * acc * (float4(erf1(acc.x * inv), erf1(acc.y * inv),
                                 erf1(acc.z * inv), erf1(acc.w * inv)) + c2);
    }
    if (r_off != OFF_NONE) acc += d[(r_off >> 2) + b * (params[P + 4] * cv4) + m * cv4 + c4];
    d[ob + m * cv4 + c4] = acc;
}

// MODE: 0=base 1=nodiv 2=nobc 3=now 4=fma 5=wide4 6=noin 7=win
template <int MODE>
inline void dw_body(device float4* d, device const uint* params, uint P, uint gid, uint b) {
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
    uint r_off = params[P + 19];

    uint cv4 = cDim >> 2;

    if (MODE == 5) {
        // ── 4 个通道块（16 通道）/线程 ─────────────────────────────
        //
        // 动机：`p_dw_fma` 比 base 快 6.65×，而 `p_dw_now` 只快 1.06×
        // —— 开销不在装载、不在权重，在**每 tap 那十几条地址数学**
        // （一个 tap 只有 4 个 FMA，摊到的整数指令比 conv8 差一个量级）。
        //
        // 于是把 `m` 固定、让一个线程吃掉 4 个连续的 `c4` 块：`ih/iw/px`
        // 一个 tap 只算**一次**，却喂 4 个 float4 FMA。装载条数与 FMA
        // 条数**之比不变**（4 x + 4 w 对 16 FMA，与 4 个线程各自的
        // 1 x + 1 w 对 4 FMA 相同），所以这条变体量的是**纯摊薄**：
        // 若它提速接近 4×，说明地址数学确实是靶子；若几乎不动，说明
        // 靶子另有其人。
        //
        // 4 个 x 与 4 个 w 在内存里各自连续（相差 1 个 float4），地址
        // 也只差一个 `+1` —— 不引入新的乘法。
        //
        // **网格不同**：线程数 = mDim * cv4 / 4，由调用方按
        // `dw_groups_for_wide` 给。CPU 对拍必须过（结果应与 base 逐位同）。
        uint cg = cv4 >> 2;
        uint tiles = mDim * cg;
        if (gid >= tiles) return;
        uint c4b = (gid % cg) << 2;
        uint m = gid / cg;
        uint oh = m / outW, ow = m - oh * outW;
        uint ib = (in_off >> 2) + b * inH * inW * cv4;
        uint wb = w_off >> 2;
        uint ob = (out_off >> 2) + b * mDim * cv4;

        float4 a0 = float4(0.0), a1 = float4(0.0), a2 = float4(0.0), a3 = float4(0.0);
        for (uint t = 0u; t < taps; t++) {
            uint ky = t / kW, kx = t - ky * kW;
            int ih = int(oh * sH + ky) - int(padT);
            int iw = int(ow * sW + kx) - int(padL);
            if (ih < 0 || ih >= int(inH) || iw < 0 || iw >= int(inW)) continue;
            uint p = ib + (uint(ih) * inW + uint(iw)) * cv4 + c4b;
            uint q = wb + t * cv4 + c4b;
            a0 += d[p] * d[q];
            a1 += d[p + 1u] * d[q + 1u];
            a2 += d[p + 2u] * d[q + 2u];
            a3 += d[p + 3u] * d[q + 3u];
        }
        if (b_off != OFF_NONE) {
            uint bo = (b_off >> 2) + c4b;
            a0 += d[bo]; a1 += d[bo + 1u]; a2 += d[bo + 2u]; a3 += d[bo + 3u];
        }
        // act 恒为 0（探针只量卷积本体），残差恒无 —— 与 dw_tail 同语义。
        uint o = ob + m * cv4 + c4b;
        d[o] = a0; d[o + 1u] = a1; d[o + 2u] = a2; d[o + 3u] = a3;
        return;
    }

    uint tiles = mDim * cv4;
    if (gid >= tiles) return;
    uint c4 = gid % cv4;
    uint m = gid / cv4;
    uint oh = m / outW, ow = m - oh * outW;
    uint ib = (in_off >> 2) + b * inH * inW * cv4;
    uint wb = w_off >> 2;
    uint ob = (out_off >> 2) + b * mDim * cv4;

    float4 acc = float4(0.0);

    if (MODE == 4) {
        // 纯 FMA：地址数学与装载全去，**迭代次数与真内核逐一对应**。
        //
        // **被乘数必须随迭代变化**，否则整条循环会被闭式折叠掉：
        // `acc = fma(x, w, acc)` 跑 n 次恒等于 `acc = x*w*n`，这是教科书
        // 级的循环惯用法，clang 会把循环整个消掉 —— 那样量到的是
        // 「写输出的访存下界」而不是算力下界，会凭空造出一个不存在的
        // 6.5× 余量。（初版就是这样写的，0.799 ms 与本行的访存下界
        // 0.72 ms 几乎相等，就是这一条的旁证。）`float(t)` 让被乘数每轮
        // 不同，线性关系被破坏，折叠不成立。
        float4 x = d[ib + c4];
        float4 w = d[wb + c4];
        for (uint t = 0u; t < taps; t++) acc = fma(x, w + float4(float(t)), acc);
    } else if (MODE == 1) {
        // 嵌套 ky/kx：热循环里没有运行期除法。`ky*kW + kx` 仍是 1 条乘法
        // （变体只针对**除法**这一项）。
        uint kH = taps / kW;
        for (uint ky = 0u; ky < kH; ky++) {
            for (uint kx = 0u; kx < kW; kx++) {
                int ih = int(oh * sH + ky) - int(padT);
                int iw = int(ow * sW + kx) - int(padL);
                if (ih < 0 || ih >= int(inH) || iw < 0 || iw >= int(inW)) continue;
                uint px = uint(ih) * inW + uint(iw);
                float4 x = d[ib + px * cv4 + c4];
                float4 w = d[wb + (ky * kW + kx) * cv4 + c4];
                acc += x * w;
            }
        }
    } else {
        for (uint t = 0u; t < taps; t++) {
            uint ky = t / kW;
            uint kx = t - ky * kW;
            int ih = int(oh * sH + ky) - int(padT);
            int iw = int(ow * sW + kx) - int(padL);
            // MODE 2 删掉判断：调用方保证 grid 只覆盖恒在界内的位置。
            if (MODE != 2 && (ih < 0 || ih >= int(inH) || iw < 0 || iw >= int(inW))) continue;
            uint px = uint(ih) * inW + uint(iw);
            // MODE 7：把 `px` 折进一个**小窗口**（地址数学一条不少、lane
            // 内的合并访问也不破坏），于是输入变成 L1 常驻。这一刀切的是
            // 「**工作集/缓存未命中**」与「**装载管线本身**」——
            //   · 折窗后接近 `p_dw_fma` ⇒ 靶子是**输入工作集**（33 MB 的
            //     输入，每 tap 重读一遍 ≈ 2.7 GB 请求），处方是**复用/分块**
            //     （让同一片输入在寄存器或 L1 里被多 tap、多输出列共享）；
            //   · 折窗后仍接近 base ⇒ 靶子是装载指令自身的发射/延迟，
            //     换布局或分块都没用。
            uint addr = ib + ((MODE == 7) ? (px & 1023u) : px) * cv4 + c4;
            // MODE 6：**删掉输入装载**，但把地址算出来的值本身喂进累加器
            // ——地址数学一条不少地留着（`addr` 被用到，编译器删不掉），
            // 少掉的只有那一条 `d[addr]`。
            //
            // 这是「输入装载 vs 地址数学」的决定性一刀：`p_dw_now` 删的是
            // **权重**装载（同一条 line 被所有 m 复用，本来就近乎免费，
            // 只值 1.06×），没能试到真正的嫌疑人。若本变体接近
            // `p_dw_fma`，则靶子是**输入 gather 本身**（→ 处方是复用，
            // 不是摊薄）；若接近 base，则靶子是那条**串行地址依赖链**。
            float4 x = (MODE == 6) ? float4(as_type<float>(addr)) : d[addr];
            float4 w = (MODE == 3) ? float4(0.25) : d[wb + t * cv4 + c4];
            acc += x * w;
        }
    }

    dw_tail(d, acc, P, params, b_off, r_off, act, c1, c2, c3, ob, m, cv4, c4, b);
}

// ── 全展开特化：KH/KW/PAD/STRIDE 变成编译期常量 ─────────────────────
//
// 动机：上面那一整排单变量变体把「输入装载」「权重装载」「地址数学」
// 「整数除法」「边界分支」逐个**单独**拿掉，结果是**每一个都只值 1~6%**
// （`p_dw_noin` 1.02×、`p_dw_now` 1.06×、`p_dw_win` 0.98×、
// `p_dw_nodiv` 0.74×、`p_dw_wide4` 0.84×），而把它们**一起**拿掉
// （`p_dw_fma`）值 **6.5×**。这不是任何单项的锅 —— 是**指令发射**：
// 热循环每 tap 约 17 条指令里只有 4 条是有用的 FMA，而最小循环体
// （`p_dw_fma`）每 tap 只有 3 条，两者之比 ≈ 5.7，与实测 6.5× 同量级。
//
// 于是让编译器替我们把那 13 条干掉：`t / kW` 的除法序列随常量折叠整个
// 消失（索引全成常量）、行级边界判断只留 KH 次、地址由
// `px = 常量 + kx` 强度削减成指针自增。
//
// **结果必须与 `p_dw_base` 逐位相同**（累加顺序一行没动），所以这条是
// 可以过 CPU 对拍的候选，不是诊断变体。
template <uint KH, uint KW, uint PAD, uint STRIDE>
inline void dw_spec(device float4* d, device const uint* params, uint P, uint gid, uint b) {
    uint in_off = params[P + 0], w_off = params[P + 1], b_off = params[P + 2];
    uint out_off = params[P + 3];
    uint mDim = params[P + 4], cDim = params[P + 5], outW = params[P + 6];
    uint inW = params[P + 7], inH = params[P + 8];
    uint act = params[P + 15];
    float c1 = as_type<float>(params[P + 16]);
    float c2 = as_type<float>(params[P + 17]);
    float c3 = as_type<float>(params[P + 18]);
    uint r_off = params[P + 19];

    uint cv4 = cDim >> 2;
    uint tiles = mDim * cv4;
    if (gid >= tiles) return;
    uint c4 = gid % cv4;
    uint m = gid / cv4;
    uint oh = m / outW, ow = m - oh * outW;
    uint ib = (in_off >> 2) + b * inH * inW * cv4;
    uint wb = w_off >> 2;
    uint ob = (out_off >> 2) + b * mDim * cv4;

    float4 acc = float4(0.0);
    #pragma unroll
    for (uint ky = 0u; ky < KH; ky++) {
        int ih = int(oh * STRIDE + ky) - int(PAD);
        if (ih < 0 || ih >= int(inH)) continue;
        uint rowb = ib + (uint(ih) * inW) * cv4 + c4;
        #pragma unroll
        for (uint kx = 0u; kx < KW; kx++) {
            int iw = int(ow * STRIDE + kx) - int(PAD);
            if (iw < 0 || iw >= int(inW)) continue;
            acc += d[rowb + uint(iw) * cv4] * d[wb + (ky * KW + kx) * cv4 + c4];
        }
    }

    dw_tail(d, acc, P, params, b_off, r_off, act, c1, c2, c3, ob, m, cv4, c4, b);
}

#define DW_SPEC_ENTRY(NAME, KH, KW, PAD, STRIDE)                                \
    kernel void NAME(                                                           \
        device float4* d [[buffer(0)]],                                         \
        device const uint* params [[buffer(1)]],                                \
        constant uint& p_off [[buffer(3)]],                                     \
        uint3 gid3 [[thread_position_in_grid]],                                 \
        uint3 wg [[threadgroup_position_in_grid]])                              \
    {                                                                           \
        uint b = wg.y;                                                          \
        if (params[0] != 0u && b >= params[0]) return;                          \
        dw_spec<KH, KW, PAD, STRIDE>(d, params, p_off, gid3.x, b);              \
    }

DW_SPEC_ENTRY(p_dw_u9, 9, 9, 4, 1)
DW_SPEC_ENTRY(p_dw_u3, 3, 3, 1, 1)

#define DW_ENTRY(NAME, MODE)                                                    \
    kernel void NAME(                                                           \
        device float4* d [[buffer(0)]],                                         \
        device const uint* params [[buffer(1)]],                                \
        constant uint& p_off [[buffer(3)]],                                     \
        uint3 gid3 [[thread_position_in_grid]],                                 \
        uint3 wg [[threadgroup_position_in_grid]])                              \
    {                                                                           \
        uint b = wg.y;                                                          \
        if (params[0] != 0u && b >= params[0]) return;                          \
        dw_body<MODE>(d, params, p_off, gid3.x, b);                             \
    }

DW_ENTRY(p_dw_base, 0)
DW_ENTRY(p_dw_nodiv, 1)
DW_ENTRY(p_dw_nobc, 2)
DW_ENTRY(p_dw_now, 3)
DW_ENTRY(p_dw_fma, 4)
DW_ENTRY(p_dw_wide4, 5)
DW_ENTRY(p_dw_noin, 6)
DW_ENTRY(p_dw_win, 7)
