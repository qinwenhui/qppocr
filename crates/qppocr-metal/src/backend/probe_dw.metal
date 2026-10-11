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
// 第二轮加的（见 `dw_body` 里 MODE 8/9/10 的注释）：
//   p_dw_interior 内点/边界双路径 —— `p_dw_nobc` 的可落地形态，位级一致、
//                 可过对拍、不需要无 pad 配置。**但它在生产内核里是负的**
//                 （见下），留着当反面样本。
//   p_dw_tap1o3 / p_dw_tap2o3  只跑 1/3、2/3 个 tap（故意算错）——
//                 量时间对 **tap 条数** 是否线性，给「换分离式」定价
//
// ## ⚠️ 这块探针最重要的产出：**孤立探针的胜势不迁移到生产**
//
// `p_dw_interior` 在这个文件里同台量到 **1.12~1.17×**（9×9 那批），位级一致、
// 过 CPU 对拍 —— 看着是白捡的。把它原样搬进 `shaders/n_conv_dw.metal` 之后，
// 用**真计划**（medium det，`QPPOCR_GPU_KERNEL_PROF`）三轮交错复现：
//
//     Conv.64（9×9, c256, m=51840, grid=12960）  13.80 → **16.38 ms（+19%）**
//     n_conv_dw 合计                              24.16 → **27.52 ms**
//     三轮交错：new 26.1/26.1/26.2  vs  old 24.1/24.7/24.6 ms
//
// 同一个改写，孤立探针里 +14%，真内核里 −16%。差别只在**周围那份代码**
// （act/残差/写回的尾部、以及整个内核一起过寄存器分配与取指）——
// 循环体翻倍的代价盖过了分支的收益。**已回退，生产内核一字未改。**
//
// 教训：这条内核上的结论必须用**真计划**复核。探针只能用来**划掉**候选
// （它说「没用」基本可信），不能用来**拍板**（它说「有用」不算数）。
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
//       8=interior（内点/边界双路径，**结果正确**）
//       9=tap1o3 10=tap2o3（只跑 1/3、2/3 个 tap，**结果故意错**）
//       11=incr（平铺循环+增量 ky/kx，**结果正确**）
//       12=intun 13=intinc（内点 + 部分展开 / 增量，**结果正确**）
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
    } else if (MODE == 8 || MODE == 12 || MODE == 13) {
        // ── 内点 / 边界**双路径**（可落地候选）────────────────────────
        //
        // 上轮钉下的唯一可动项是**边界判断**：无 pad 配置上 `p_dw_nobc`
        // 比 base 快 **1.16×**，但那是「删掉判断」，带 padding 的产线配置
        // 上不能删 —— 于是上轮把它记成「不可落地」。
        //
        // 其实可落地：按**整窗口是否全在界内**一次判掉，内点走一条
        // 没有那 4 条 compare/branch 的循环体，边界像素原样走 base 那条。
        // 9×9 pad=4 时内点占 `(H-8)(W-8)/(H·W)`（240×216 → 93%）。
        //
        // 与 base **逐位相同**：tap 顺序、`ih/iw/px` 算法、`acc += x*w`
        // 一字未动，内点上 base 的判断本来就永真（`continue` 从不触发）。
        // 所以这三条都能过 CPU 对拍，是候选不是诊断变体。
        //
        // 内点谓词只判**首尾两个 tap**（`t=0` 与 `t=taps-1`）：行号
        // `ih = oh*sH + ky - padT` 对 `ky` 单调，故两端都在界内 ⟺ 全部
        // 在界内。这与 base 里「那 4 条 compare 从来没触发过」**逐位等价**，
        // 不是保守近似。
        //
        // ⚠️ 判宽了（谓词比实际界内更松）会让 `uint` 回绕成天文数字地址、
        // 直接把 GPU 打瘸。调用方在**派发前**用同一算式在 CPU 上逐个输出
        // 位置暴力验过「内点 ⟹ 每个 tap 都在界内」，不成立就 panic。
        //
        // 12 = 内点 + `#pragma unroll 4`；13 = 内点 + 增量 ky/kx。
        // 动机：内点路径已经没有分支了，可**时间仍只快了 1.15×**，
        // 说明剩下的瓶颈是每 tap 那条**地址依赖链**本身（`t/kW` 的倒数
        // 乘法序列 → ky/kx → ih/iw → px → 地址），而不是它的条数 ——
        // 上轮把链上任何**单项**拿掉都只值 1.0~1.1×，全拿掉才 4.7×。
        // 展平或缩短这条链 = 让相邻几次迭代的地址互相独立。
        uint kH = taps / kW;
        int ih0 = int(oh) * int(sH) - int(padT);
        int iw0 = int(ow) * int(sW) - int(padL);
        bool inner = (ih0 >= 0) && (ih0 + int(kH - 1u) < int(inH))
                  && (iw0 >= 0) && (iw0 + int(kW - 1u) < int(inW));
        if (inner) {
            if (MODE == 13) {
                uint ky = 0u, kx = 0u;
                for (uint t = 0u; t < taps; t++) {
                    int ih = int(oh * sH + ky) - int(padT);
                    int iw = int(ow * sW + kx) - int(padL);
                    uint px = uint(ih) * inW + uint(iw);
                    acc += d[ib + px * cv4 + c4] * d[wb + t * cv4 + c4];
                    if (++kx == kW) { kx = 0u; ky++; }
                }
            } else if (MODE == 12) {
                #pragma unroll 4
                for (uint t = 0u; t < taps; t++) {
                    uint ky = t / kW;
                    uint kx = t - ky * kW;
                    int ih = int(oh * sH + ky) - int(padT);
                    int iw = int(ow * sW + kx) - int(padL);
                    uint px = uint(ih) * inW + uint(iw);
                    acc += d[ib + px * cv4 + c4] * d[wb + t * cv4 + c4];
                }
            } else {
                for (uint t = 0u; t < taps; t++) {
                    uint ky = t / kW;
                    uint kx = t - ky * kW;
                    int ih = int(oh * sH + ky) - int(padT);
                    int iw = int(ow * sW + kx) - int(padL);
                    uint px = uint(ih) * inW + uint(iw);
                    acc += d[ib + px * cv4 + c4] * d[wb + t * cv4 + c4];
                }
            }
        } else {
            for (uint t = 0u; t < taps; t++) {
                uint ky = t / kW;
                uint kx = t - ky * kW;
                int ih = int(oh * sH + ky) - int(padT);
                int iw = int(ow * sW + kx) - int(padL);
                if (ih < 0 || ih >= int(inH) || iw < 0 || iw >= int(inW)) continue;
                uint px = uint(ih) * inW + uint(iw);
                acc += d[ib + px * cv4 + c4] * d[wb + t * cv4 + c4];
            }
        }
    } else if (MODE == 11) {
        // 平铺计数循环 + **增量 ky/kx**，边界判断保留。结构与 `p_dw_nodiv`
        // 的关键区别：那条换成了**嵌套循环**（实测 0.74×），输的可能是
        // 嵌套结构本身（内层 9 次的循环无法软件流水、寄存器压力上升），
        // 而不是除法。这条把「除法」与「嵌套」两件事分开量。
        uint ky = 0u, kx = 0u;
        for (uint t = 0u; t < taps; t++) {
            int ih = int(oh * sH + ky) - int(padT);
            int iw = int(ow * sW + kx) - int(padL);
            if (ih >= 0 && ih < int(inH) && iw >= 0 && iw < int(inW)) {
                uint px = uint(ih) * inW + uint(iw);
                acc += d[ib + px * cv4 + c4] * d[wb + t * cv4 + c4];
            }
            if (++kx == kW) { kx = 0u; ky++; }
        }
    } else {
        // MODE 9/10：只跑前 1/3、2/3 个 tap（**结果故意错**）。循环体与
        // base **逐字相同**，只有迭代次数不同 —— 用来量 d(时间)/d(tap)，
        // 也就是「把 81 个 tap 换成 18 个（分离式 9×1+1×9）能拿到多少」
        // 的换算尺子。上一轮只知道「削到近乎空循环体值 4.77×」，不知道
        // 时间对 tap **条数**是不是线性的 —— 这条直接回答。
        uint ntap = (MODE == 9) ? (taps / 3u) : (MODE == 10) ? (taps * 2u / 3u) : taps;
        for (uint t = 0u; t < ntap; t++) {
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
DW_ENTRY(p_dw_interior, 8)
DW_ENTRY(p_dw_tap1o3, 9)
DW_ENTRY(p_dw_tap2o3, 10)
DW_ENTRY(p_dw_incr, 11)
DW_ENTRY(p_dw_intun, 12)
DW_ENTRY(p_dw_intinc, 13)

// ── 第三轮：**线程组瓦片暂存**（`p_dw_tile`）──────────────────────────
//
// 这是唯一一条**没被前两轮 11 个候选覆盖**的结构，而且它直接打在被量实的
// 靶心上。
//
// 前两轮钉下的事实：
//   * `p_dw_fma`（地址数学与装载全去、只留 FMA）= **4.66×** ⇒ 靶心是「每 tap
//     那十几条整数/分支指令」，不是带宽；
//   * 但**逐项**去掉都不值钱：去输入装载 1.01×、去权重装载 1.05×、
//     去边界判断 1.15×。⇒ 钱不在某一条指令上，在那**一整段**上；
//   * 所有「换个写法去掉除法/加宽每线程」的结构改动**全部更慢**
//     （`nodiv` 0.74×、`u9` 0.72×、`wide4` 0.84×、`intun` 0.59×）——
//     因为循环体变大 ⇒ 寄存器压力/占用率下降吃掉收益。
//
// 前两轮**都没试过**的一条：把输入瓦片先搬进**线程组内存**，再让热循环从
// 共享内存读。这样热循环里：
//   * 没有运行期除法（`dy`/`dx` 双层循环，外层循环上界 `kH` 只算一次）；
//   * 没有 `ih/iw` 的乘加、没有 2 个 compare + 分支（边界在**暂存时**判掉，
//     越界填 0）；
//   * 没有 `px*cv4 + c4` 的乘加与 4 路分叉的设备地址；
//   * 权重也进了共享内存，热循环那次「每线程不同 c4 ⇒ 全分叉 gather」变成
//     每线程组只搬一趟。
//
// 这条同时**把循环体变小**（这正与前两轮「变大就输」的规律同向），代价是
// 一个 prologue + 一次 `threadgroup_barrier`。
//
// 竞品 `conv_dw4t`（`SimdPaddleOCR/.../Shaders/conv.metal`）正是这个结构，
// 所以它不是纸上谈兵。差别只在他们**顺带用了 fp16**，这里保持 f32 ——
// 自变量只留「暂存」一个。
//
// 语义：累加次序与 base **逐位相同**（`t` 行主序 = `dy` 外层 / `dx` 内层）。
// 唯一差别是越界 tap：base 是 `continue`（不加），这里是加 `0.0*w`。
// 两者只可能在 `acc` 恰为 `-0.0` 时有位差（`-0.0 + 0.0 = +0.0`），数值相等。
//
// 限制：`kW,kH ≤ DW_MAXK` 且 **stride == 1**（瓦片尺寸是编译期常量）。
// 调用方必须先判 `dw_tile_ok`，否则 `stW*stH` 会超出数组、读飞打瘸 GPU。
#define DW_TILE 16
#define DW_MAXK 9

inline bool dw_tile_ok(uint kh, uint kw, uint sh, uint sw, uint outh, uint outw, uint cdim) {
    if (kh > DW_MAXK || kw > DW_MAXK || sh != 1u || sw != 1u) return false;
    uint stW = DW_TILE + kw - 1u, stH = DW_TILE + kh - 1u;
    if (stW * stH > (DW_TILE + DW_MAXK - 1u) * (DW_TILE + DW_MAXK - 1u)) return false;
    return outh > 0u && outw > 0u && cdim >= 4u;
}

kernel void p_dw_tile(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 wg [[threadgroup_position_in_grid]],
    // MSL 要求同内核的位置属性**同型**（不能 uint3 混 uint），故这里也取
    // uint3 再取 `.x` —— `n_head_argmax` 头注释里记过同一个坑。
    uint3 tp3 [[thread_position_in_threadgroup]])
{
    uint tid = tp3.x;
    // 9×9 s=1 ⇒ 24×24 = 576 个 float4 = 9 KB；权重 81×16 B = 1.3 KB。
    threadgroup float4 shTile[(DW_TILE + DW_MAXK - 1) * (DW_TILE + DW_MAXK - 1)];
    threadgroup float4 shW[DW_MAXK * DW_MAXK];

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
    uint r_off = params[P + 19];

    uint b = wg.y;
    if (params[0] != 0u && b >= params[0]) return; // 与生产同款 real_n 早退

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

    // 暂存输入瓦片（越界填 0）
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
    // 暂存权重：本线程组的 c4 是固定的，所以 81 次装载全组只搬一趟
    for (uint e = tid; e < taps; e += 256u) {
        shW[e] = d[wb + e * cv4 + c4];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint ox = tx * DW_TILE + (tid & (DW_TILE - 1u));
    uint oy = ty * DW_TILE + (tid >> 4);
    if (ox >= outW || oy >= outH) return; // 屏障之后才允许早退

    uint lx0 = (tid & (DW_TILE - 1u)) * sW;
    uint ly0 = (tid >> 4) * sH;

    // 热循环：全部来自线程组内存，无除法、无边界判断、无设备地址数学。
    // 次序与 base 逐位相同（t 行主序）。
    float4 acc = float4(0.0);
    for (uint dy = 0u; dy < kH; dy++) {
        for (uint dx = 0u; dx < kW; dx++) {
            acc += shW[dy * kW + dx] * shTile[(ly0 + dy) * stW + lx0 + dx];
        }
    }
    dw_tail(d, acc, P, params, b_off, r_off, act, c1, c2, c3, ob, oy * outW + ox, cv4, c4, b);
}

// ── 第三轮续：瓦片版还能往哪走（`p_dw_tile9` / `p_dw_tilew`）──────────
//
// `p_dw_tile` 是 1.86×，但离 `p_dw_fma` 还有 **2.55×**（473 vs 1207 GFLOPS）。
// 热循环每轮是「2 次线程组装载 + 1 次 FMA」，所以剩下这 2.55× 只可能出在
// 那两条装载上。两个互相独立的疑点，各做一个变体：
//
//   `p_dw_tile9`：K 烤成编译期常量 9 并全展开。`kW`/`kH` 现在是**运行期**
//     的，于是每轮都要算一次 `(ly0+dy)*stW + lx0+dx` 与 `dy*kW+dx` 两条
//     乘加（外加循环判断）；烤死之后这 81 个下标全变成「一个基址 + 立即
//     数」，装载也可能被流水化。代价是只对 9×9 成立。
//
//   `p_dw_tilew`：权重**不进线程组**，热循环里从设备内存读。权重在瓦片版
//     里是「全组同地址」的广播装载，本来就不贵（前两轮量过：去掉权重装载
//     只值 1.05×）；而搬进 `shW` 要付 prologue（81 次全组装载）与一次
//     barrier 里的写可见性。如果这条更快，说明 `shW` 那一段是净亏 ——
//     与 memory `mma-conv8-weights-are-free-parallelism-is-dead` 同款。
//
// 自变量仍然只有一个（K 是否编译期 / 权重在哪一级存储），其余与 `p_dw_tile`
// 逐字相同。
#define DW_TILE9_K 9

kernel void p_dw_tile9(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 wg [[threadgroup_position_in_grid]],
    uint3 tp3 [[thread_position_in_threadgroup]])
{
    uint tid = tp3.x;
    threadgroup float4 shTile[(DW_TILE + DW_MAXK - 1) * (DW_TILE + DW_MAXK - 1)];
    threadgroup float4 shW[DW_TILE9_K * DW_TILE9_K];

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
    uint r_off = params[P + 19];

    uint b = wg.y;
    if (params[0] != 0u && b >= params[0]) return;
    // 这条变体只在 9×9 s1 成立；判宽了就是读飞（`dw_spec_ok` 是唯一约定）。
    if (kW != DW_TILE9_K || taps != DW_TILE9_K * DW_TILE9_K || sH != 1u || sW != 1u) return;

    uint cv4 = cDim >> 2;
    uint outH = mDim / outW;
    uint tilesX = (outW + DW_TILE - 1u) / DW_TILE;
    uint g = wg.x;
    uint c4 = g % cv4;
    uint tile = g / cv4;
    uint tx = tile % tilesX, ty = tile / tilesX;

    uint stW = DW_TILE + DW_TILE9_K - 1u;
    uint stH = DW_TILE + DW_TILE9_K - 1u;
    int iyT = int(ty * DW_TILE) - int(padT);
    int ixT = int(tx * DW_TILE) - int(padL);
    uint ib = (in_off >> 2) + b * inH * inW * cv4;
    uint wb = w_off >> 2;
    uint ob = (out_off >> 2) + b * mDim * cv4;

    for (uint e = tid; e < stW * stH; e += 256u) {
        uint iy = e / stW, ix = e - iy * stW;
        int gy = iyT + int(iy), gx = ixT + int(ix);
        float4 v = float4(0.0);
        if (gy >= 0 && uint(gy) < inH && gx >= 0 && uint(gx) < inW) {
            v = d[ib + (uint(gy) * inW + uint(gx)) * cv4 + c4];
        }
        shTile[e] = v;
    }
    for (uint e = tid; e < DW_TILE9_K * DW_TILE9_K; e += 256u) shW[e] = d[wb + e * cv4 + c4];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint ox = tx * DW_TILE + (tid & (DW_TILE - 1u));
    uint oy = ty * DW_TILE + (tid >> 4);
    if (ox >= outW || oy >= outH) return;
    uint lx0 = (tid & (DW_TILE - 1u));
    uint ly0 = (tid >> 4);

    float4 acc = float4(0.0);
    #pragma unroll
    for (uint dy = 0u; dy < DW_TILE9_K; dy++) {
        #pragma unroll
        for (uint dx = 0u; dx < DW_TILE9_K; dx++) {
            acc += shW[dy * DW_TILE9_K + dx] * shTile[(ly0 + dy) * stW + lx0 + dx];
        }
    }
    dw_tail(d, acc, P, params, b_off, r_off, act, c1, c2, c3, ob, oy * outW + ox, cv4, c4, b);
}

kernel void p_dw_tilew(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 wg [[threadgroup_position_in_grid]],
    uint3 tp3 [[thread_position_in_threadgroup]])
{
    uint tid = tp3.x;
    threadgroup float4 shTile[(DW_TILE + DW_MAXK - 1) * (DW_TILE + DW_MAXK - 1)];

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
    uint r_off = params[P + 19];

    uint b = wg.y;
    if (params[0] != 0u && b >= params[0]) return;

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

    for (uint e = tid; e < stW * stH; e += 256u) {
        uint iy = e / stW, ix = e - iy * stW;
        int gy = iyT + int(iy), gx = ixT + int(ix);
        float4 v = float4(0.0);
        if (gy >= 0 && uint(gy) < inH && gx >= 0 && uint(gx) < inW) {
            v = d[ib + (uint(gy) * inW + uint(gx)) * cv4 + c4];
        }
        shTile[e] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint ox = tx * DW_TILE + (tid & (DW_TILE - 1u));
    uint oy = ty * DW_TILE + (tid >> 4);
    if (ox >= outW || oy >= outH) return;
    uint lx0 = (tid & (DW_TILE - 1u)) * sW;
    uint ly0 = (tid >> 4) * sH;

    // 权重直接读设备内存：地址在全组内**同一个**（c4 固定、t 随循环推进），
    // 所以每次是一个广播装载，而不是 256 路 gather。
    float4 acc = float4(0.0);
    for (uint dy = 0u; dy < kH; dy++) {
        for (uint dx = 0u; dx < kW; dx++) {
            acc += d[wb + (dy * kW + dx) * cv4 + c4] * shTile[(ly0 + dy) * stW + lx0 + dx];
        }
    }
    dw_tail(d, acc, P, params, b_off, r_off, act, c1, c2, c3, ob, oy * outW + ox, cv4, c4, b);
}
