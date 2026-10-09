// 稠密 kxk conv（NHWC f32，group=1，任意 stride/pad）——m4×n8 寄存器
// 分块瓦片：每线程算 4 个输出位置 × 8 个输出通道（两个 float4 列）。
//
// 与 n_conv（m4×n4）的差别：两列共享同一组输入 gather（a0..a3）——
// 每输入 float4 的 FMA 数翻倍，输入装载指令与 DRAM 放大（列组数）
// 减半。权重装载同样两列共享。适用 nv%2==0（Co%8==0，det 的
// 16/32/64 通道 conv 全满足）。
//
// 存储/重排/act/残差语义与 n_conv 完全一致（见其头注释）。
// 与 qppocr-gpu/shaders/n_conv8.comp 逐行对应——改语义必须两边同步。
//
// # 两个入口：`n_conv8`（f32 权重）与 `n_conv8h`（fp16 权重）
//
// 内核体只有一份（`conv8_body` 模板），两个 `kernel` 入口分别用
// `float4`/`half4` 实例化——**热循环、网格、瓦片映射、累加顺序逐字相同**，
// 唯一变量是「一条权重装载覆盖几个字节」：f32 下一个 `float4` 是 16 B，
// fp16 下一个 `half4` 是 8 B，而**两者覆盖的是同样的 4 个输出通道**
// （权重段内的元素序没变，只换了每个元素的宽度）。
//
// 所以内核里所有权重下标表达式一字不动，只有基址的折算变：
// `w_off >> 2`（float4 下标）→ `w_off >> 1`（half4 下标）。`w_off` 以
// u32 word 计，而静态段起始恒 16 B 对齐（`Layout::alloc_static`），
// 故 `>>1` 是精确的、不需要额外对齐。
//
// 这支 fp16 路子的价值是**先量过才写的**：探针 `p_whalf` 与该模板的
// 唯一语义差就是这一处 `>>`，实测热点 1×1 上 1.7~2.6×，吃掉整个权重
// 装载成本的 73~88%（见 `bench/gpuhealth/logs/` 与 conv_probe.rs 的
// 变体表）。精度代价在 CPU 侧量过：三层模型 CER 0.072~0.199%。
//
// 选择哪一支是**计划层**的事（`PlanBuilder::weight_f16`，默认 false），
// 并且与权重怎么重排**必须成对**：`n_conv8h` 读 half4、`n_conv8` 读
// float4，同一个 `w_off` 两种解释，错配就是静默读垃圾。
//
// ⚠️ Vulkan 侧的 `n_conv8.comp` **没有** fp16 变体（GLSL 版是离线编译的
// SPIR-V 产物链，要同步加）。所以 Vulkan 的 `weight_f16` 恒 false——
// 两边不同步时计划会派发到不存在的内核，是硬失败而非静默降级。
//
// **瓦片到线程的映射有两套**（见内核里 `taps == 1u && ...` 那一段的
// 注释）：通道最快变化的默认序，与「窗口序」（1×1 且权重段 ≥ 256 KB
// 时启用，把同一片权重的所有行组铺在一起，复用距离 9 MB → 123 KB）。
// 两套的 tile 数与网格公式**完全相同**（`ceil(m/4)*nh`），所以计划层
// 零改动、覆盖无风险；热循环与寄存器瓦片也完全相同。
//
// # grid-stride：保留机制，但实测无收益
//
// 曾每线程恰好吃一个 tile（网格 = `ceil(tiles/256)`）。一度认为「线程组
// 周转本身极贵」，据此在这个内核里加了 stride 循环、计划侧加了网格上限，
// 好让计划把网格夹小。**结论不成立，机制保留、默认关闭**：
//
// - det 前向 5 次取最小：cap=0（不夹）**1474.7 ms**、64 组 1506.1、
//   256 组 1559.4、32 组 1596.2、16 组 1875.3。只有 16 组明显变差，
//   其余都在噪声里，而不夹名义最好。`GROUP_CAP_DEFAULT` 因此回到 0。
// - 原因：真实内核是**短突发**（det 154 条 / 1475 ms ≈ 9.6 ms 一条），
//   吃的是每条 dispatch 的固定开销与访存停顿，不是长跑稳态带宽。
// - 旧记录里「空转内核 0.7~2.7 µs/组、与内核算什么无关」来自一个有缺陷
//   的探针；用取自 det 真实网格的 v1 探针重测是 **0.15~3.4 ms/条，且与
//   组数不同向**（9720 组 0.15 ms、288 组 2.7 ms）——斜率法在这个粒度上
//   的噪声地板就有 ~1 ms/条，比信号还大。
//
// ⚠️ **已作废（2026-10-09 复测）**——原记录是：
//
//   「长跑带宽曲线（256 MB 合并访存 + 手工展开 + 逐字节校验）：8K 线程
//     24 GB/s → 32K 13 → 262K 8.8 → 2M 3.9 GB/s；同机 CPU memcpy 136 GB/s」，
//   并据此刻下了「这块 GPU 跑不赢自己的 CPU」的结论。
//
// 这条**不是在健康 GPU 上量的**。同款探针（`microbench::mem_bandwidth_sweep`，
// n_elem 原地读+写，健康门禁 `bench/gpuhealth/health.sh` 退出码 0、
// recoveryCount=0）当天复测：1 MB 332.7 → 4 MB 216.2 → 16 MB 83.5 →
// 64 MB 94.5 → **256 MB 93.8 GB/s**，且随线程数是**上升**的
// （旧曲线说 2M 线程只有 3.9 GB/s，新测 16.7M 线程是 93.8）。
// 同一台机器、同一个缓冲区大小，差 24 倍——旧曲线是「访存通路被打瘸」
// 那个状态的读数，不是这台 M4 的能力。
//
// 所以：**别再引用 24 GB/s / 「跑不赢 CPU」**。GPU 访存通路的可用带宽与
// 苹果自家 blit 拷贝同量级（blit 实测 80 GB/s，计算内核 94 GB/s），
// 任何「GPU 天生慢」的推论都要先跑健康门禁再谈。数据本身仍留着，
// 是「瘸了之后所有数字都会静默地变小」这条教训的实证材料。
//
// stride 循环留着是为了让计划侧的上限机制仍能工作（`PlanBuilder::grid_cap`，
// 默认 0 = 不夹），也便于以后扫参。**数值逐位不变**：每个 tile 仍由一个
// 线程用同一套累加顺序算完，stride 只改变「哪个线程算哪个 tile」，不改变
// 任何输出的加法次序。`grid_cap = 0` 时 `stride >= tiles`，循环恰好跑一趟，
// 与改写前逐字等价。
#include <metal_stdlib>
using namespace metal;

constant uint OFF_NONE = 0xFFFFFFFFu;

// exp1/erf1：conv.comp 的逐字拷贝（与 CPU 位级一致的多项式，非硬件 exp）。
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

float4 apply_act(float4 v, uint act, float c1, float c2, float c3) {
    if (act == 2u) return max(v, float4(0.0));
    if (act == 1u) {
        float inv = 1.0 / c1;
        return c3 * v * (float4(erf1(v.x * inv), erf1(v.y * inv),
                                erf1(v.z * inv), erf1(v.w * inv)) + c2);
    }
    return v;
}

// `WVec` = 权重装载的向量视图（`float4` 或 `half4`），`WSHIFT` = 把 word
// 偏移折算成该视图下标的右移量（4 B 元素 >>2，2 B 元素 >>1）。二者必须
// 一起选——见文件头注释。
//
// `EDGE` = 边框变体（见 `n_conv8_edge`）：把「已被 MMA 覆盖」的矩形
// （y_lo, y_hi, x_lo, x_hi，闭区间，**由标准参数块现算**，见正文里那段）
// 里的输出像素跳过——那些由 `n_conv8_mma3` 写，本内核写**互不相交**的补集。
// 参数块与另两支**逐字相同**（21 词），因为一次 conv 的两条 dispatch 共用
// 一块参数（编码期不能改它，见 n_conv8_mma3.metal 文件头）。
// `EDGE == false` 时整段矩形计算是编译期死代码，热循环一条指令都不多。
template <typename WVec, uint WSHIFT, bool EDGE>
static inline void conv8_body(
    device float4* d,
    device const uint* params,
    constant uint& p_off,
    uint3 gid3,
    uint3 tpg,
    uint3 wg)
{
    // 权重的**另一个视图**：热循环里所有 `w[...]` 都走它，`d[...]` 仍是
    // f32（输入/输出/bias/残差全不变）。
    device WVec* w = (device WVec*)d;
    uint P = p_off;
    uint in_off = params[P + 0], w_off = params[P + 1], b_off = params[P + 2];
    uint out_off = params[P + 3];
    uint mDim = params[P + 4], nDim = params[P + 5], outW = params[P + 6];
    uint inW = params[P + 7], inH = params[P + 8], cin4v = params[P + 9];
    uint taps = params[P + 10], kW = params[P + 11];
    uint sH = params[P + 12], sW = params[P + 13];
    uint padT = params[P + 14], padL = params[P + 15];
    uint act = params[P + 16];
    float c1 = as_type<float>(params[P + 17]);
    float c2 = as_type<float>(params[P + 18]);
    float c3 = as_type<float>(params[P + 19]);
    uint r_off = params[P + 20];

    // 覆盖矩形（`n_conv8_mma3` 写这些像素，本内核写补集）。**由标准参数块
    // 现算**，不额外占参数词——理由见 n_conv8_mma3.metal 文件头：两条 dispatch
    // 读同一块参数，参数块在编码期不能改。算式与计划层的派发网格、与
    // n_conv8_mma3 内核里的同一组导出式**必须三份一致**。
    //
    //   y_lo = padT     y_hi = padT + (mDim/outW - (taps/kW - 1)) - 1
    //   x_lo = padL     x_hi = padL + (outW - (kW-1)) - 1
    //
    // 注意 x 上用**内点列全宽**而不是「64 的整数倍块」——`n_conv8_mma3` 用
    // 一个尾部块（锚在最后 64 个内点列、写掩码裁到行尾）吃掉了余数，所以
    // 覆盖矩形一直铺到 padL + inner_w - 1。用旧的 64 取整式会把那 48 列
    // 算两遍（MMA 写一次、edge 再写一次，静默错）。
    //
    // 非 EDGE 实例化下整段是编译期死代码（`EDGE` 是模板常量）。
    uint y_lo = 0u, y_hi = 0u, x_lo = 0u, x_hi = 0u, e_tiles = 0u, M_e = 0u;
    // ── 紧凑网格（EDGE 专用）────────────────────────────────────────────
    //
    // 第一版让 edge 复用 `n_conv8` 的原网格（铺满整张输出平面）再逐 tile
    // 早退。实测**空转本身就是成本**：Conv.121（240x216 k=7 Co=64）的 edge
    // 要 1.75 ms，而真正含边框像素的 tile 只有 6%——每个空 tile 仍要付
    // `gid%nh`、`gid/nh` 和 4 次 `m/outW` 共 6~8 次整数除法（GPU 上整数除
    // 法是几十条的软实现）才走到早退那一行，103680 个线程一起做就是毫秒级。
    //
    // 处方：网格只铺**真正含边框像素的行组**（行组 = 一行内连续 4 个输出
    // x，与 base 映射同一单位）。边框恰好是三段之并：
    //
    //   T 上台：y ∈ [0, padT)            整行，每行 gh=outW/4 个行组
    //   M 中带：y ∈ [padT, outH-padT)    每行头 gL 个 + 尾 gL 个行组
    //   B 下台：y ∈ [outH-padT, outH)    整行
    //
    //   gL = ceil(padL/4)。对称 pad 下`x_hi = outW-1-padL`，所以右带
    //   `x ≥ outW-padL` 恰好落在**最后 gL 个行组**里。
    //
    // 上台/下台不能省成「只铺头尾 gL」——那些行的**全部**像素都在覆盖矩形
    // 之外（`y_lo = padT`），必须整行补。
    //
    // 落进集合的行组仍可能含被覆盖的像素（比如中带头部那个行组的 x ∈ [padL,4)
    // 部分），那几个像素由下面**一字未动**的逐像素 `ok[]` 谓词裁掉——所以
    // 「每个像素恰好写一次」的证明还是原来那一条：本集合是补集的**子集**，
    // 且并上 MMA 的覆盖矩形 = 全平面。
    //
    // 契约（计划层第三份拷贝，见 plan.rs 的 `n_conv8_edge` 派发式）：
    //   tiles_edge = (2*T + M) * nh，**替换**原来的 rows_g*nh；
    // 门禁另加两条：`outW % 4 == 0`（行组不跨行，否则枚举与平面映射错位）、
    // `2*gL <= gh`（中带头尾不重叠，否则同一行组被枚举两次＝写两次）。
    uint gh = outW >> 2;                  // 每行行组数
    uint gL = (padL + 3u) >> 2;           // 头/尾各铺几个行组
    if (EDGE) {
        uint outH = mDim / outW;          // 输出平面恒 outH*outW 个像素
        uint kH = taps / kW;
        y_lo = padT;
        y_hi = padT + (outH - (kH - 1u)) - 1u;
        x_lo = padL;
        x_hi = padL + (outW - (kW - 1u)) - 1u;
        // 紧凑瓦片数（`nh = nDim >> 3`，与下面那个 `nv >> 1` 恒等）。
        M_e = (outH - 2u * padT) * 2u * gL;
        e_tiles = (2u * (padT * gh) + M_e) * (nDim >> 3u);
    }

    uint b = wg.y;
    if (params[0] != 0u && b >= params[0]) return; // real_n 早退：批维补齐的空行零算力；0=不限（裸 dispatch 安全默认）
    uint nv = nDim >> 2;
    uint nh = nv >> 1; // 列组数（每组 2 个 float4 列）
    uint rows_g = (mDim + 3u) >> 2; // 行组数
    // EDGE 用紧凑网格（见文件头「紧凑网格」）；非 EDGE 分支是编译期常量选择，
    // 原式一字未动。
    uint tiles = EDGE ? e_tiles : rows_g * nh;

    // 整块不随 tile 变的基址（本来就在映射之后算，纯外提）。
    uint ib = (in_off >> 2) + b * inH * inW * cin4v;
    uint wb = w_off >> WSHIFT;
    uint ob = (out_off >> 2) + b * mDim * nv;

    // stride = 本网格实际铺开的线程数（计划侧可能夹过）。不夹时
    // `stride >= tiles`，循环恰好跑一趟，与改前逐字等价。
    uint stride = tpg.x;

    for (uint gid = gid3.x; gid < tiles; gid += stride) {
        // `n4b` = 第二个通道块（float4 列）的下标。**它不再恒等于 `n4+1`**
        // ——列主序把第二列放到 `n4 + nh`（见下面 else-if 分支）。所有用到
        // 第二列的地方（权重、bias、残差、写回）都必须用 `n4b`，用错就是
        // 静默写坏一半通道。
        uint n4, m0, n4b;
        if (EDGE) {
            // ── 紧凑解码：rg → (y, 行组) 三段 ──
            //
            // 与文件头「紧凑网格」逐字对应。**不跨行**：`outW%4==0` 由门禁
            // 保证，所以行组 r 覆盖的 4 个像素恒在同一行 y 内（`ok[]` 里那个
            // `m/outW` 仍按平面线性下标算，与这里给的行是同一个 y）。
            uint rg = gid / nh;
            n4 = (gid % nh) * 2u;
            n4b = n4 + 1u;
            uint T = padT * gh;
            uint y, r;
            if (rg < T) {                          // 上台：整行，rg 直接是行内序号
                y = rg / gh;
                r = rg - y * gh;
            } else if (rg < T + M_e) {             // 中带：每行头 gL + 尾 gL
                uint q = rg - T;
                uint per = 2u * gL;
                y = padT + q / per;
                uint s = q - (q / per) * per;
                r = (s < gL) ? s : (gh - gL + (s - gL));
            } else {                               // 下台：整行
                uint q = rg - T - M_e;
                uint yy = q / gh;
                y = (mDim / outW) - padT + yy;
                r = q - yy * gh;
            }
            m0 = y * outW + r * 4u;
        } else if (taps == 1u && cin4v * nv >= 65536u) {
            // ── 窗口序（**只换瓦片到线程的映射，热循环一字不动**）──
            //
            // 病根：权重段（本例 9 MB）被每个行组各读一遍 = 485 MB，而原
            // 映射是**通道最快变化**——同一片权重要等全部 54 个行组走完
            // （整整 9 MB 流量）才被再读一次，远超 L2，于是卡在 DRAM 上
            // （实测：同形状 p_base 2.92 ms vs 权重钉进 1 KB 的 p_wsmall
            // 0.78 ms，条数地址算术全同）。
            //
            // 处方：把「32 列组的窗」提到行组外面——先让同一片窗口的**所
            // 有行组**背靠背走完，再推进下一片。复用距离 9 MB → 一片窗口
            // 的权重（120×256×4 ≈ 123 KB，L2 内）。网格不变（`tiles` 与
            // 原式同为 `ceil(m/4)*nh`），一个 warp 的 32 个 lane 仍是 32 个
            // 连续列组 ⇒ 输入仍是广播、权重的行数与跨度也不变，变的**只是
            // 这片权重多久之后还会被用到**。实测分类层 2.92 → 1.20 ms。
            //
            // 尾部：`nh` 不一定是 32 的倍数（本层 2339 = 73*32 + 3）。前
            // `full` 个 tile 铺满 73 个整窗，剩下的 3 个通道 × 54 行组
            // = 162 个 tile 正好是尾数，用下面 else 分支单独铺掉。对任意
            // (m, n) 都成立（`tiles - full == (nh%32)*rows_g` 恒等）。
            //
            // 门：仅 1×1（taps>1 时相邻输出共享输入像素，换序会砸掉输入的
            // 局部性——实测 3×3 上反而慢一倍）且权重段 ≥ 4 MB（低于这档的
            // 权重本来就躺在 L2 里，缩复用距离无益——2026-10-08 在 288 KB /
            // 576 KB / 1152 KB 三档上实测窗口序只值 1.05 / 1.00 / 0.97，而
            // 那三档改用列主序值 1.25 / 1.17 / 1.04；门的旧值是 256 KB，
            // 把这三档一起误判进了窗口序。生产里 1152 KB 与 8771 KB 之间
            // **没有别的权重档**（见 bench/gpuhealth/logs/bottleneck_map_*.md
            // 的尺寸分布），所以阈值落在这段区间里任何位置都等价）。
            // 另：本分支的窗内排布也已改为列主序（CTC 头 8771 KB 上
            // 1.51× → **2.00×**）。旧注释保留的"≥ 256 KB"已作废。
            //
            // 映射仍是 `gid → tile` 的函数，与 stride 无关（stride 循环
            // 只是换一批 gid 来算），故夹网格不影响这里的全覆盖。
            uint full = (tiles / (32u * rows_g)) * (32u * rows_g);
            if (gid < full) {
                uint w = gid / (32u * rows_g);
                uint rem = gid - w * (32u * rows_g);
                // 窗内**列主序**（见 else-if 分支的解释）：窗宽 32 块 =
                // 64 个 float4，lane q 取第 q 与第 q+32 个，两条装载各是
                // 4 条满载 line。双射仍在（q ∈ [0,32) 两列合起来覆盖本窗）。
                n4 = (w << 6u) + (rem & 31u);
                n4b = n4 + 32u;
                m0 = (rem >> 5u) * 4u;
            } else {
                // 尾数：一个 warp 的 lane 跨行组、同通道 ⇒ 权重装载本就是
                // 广播（1 条事务），列主序无从下手，保持相邻的 +1。
                uint rem = gid - full;
                uint wb2 = full / (32u * rows_g) << 5u;
                n4 = (wb2 + rem / rows_g) * 2u;
                n4b = n4 + 1u;
                m0 = (rem % rows_g) * 4u;
            }
        } else if (taps == 1u) {
            // ── 列主序（1×1 且权重 < 4 MB）──────────────────────────
            //
            // 上面那条窗口序的门从 256 KB 提到了 4 MB：实测 288 KB / 576 KB
            // / 1152 KB 三档上窗口序分别只值 1.05 / 1.00 / 0.97，而列主序
            // 值 1.25 / 1.17 / 1.04 —— 权重本来就躺在 L2 里时，缩短复用
            // 距离没有意义，赚的是 **lane 排布**。
            //
            // 原映射 `n4 = (gid % nh) * 2` 让相邻 lane 的通道相距 2 个
            // float4，一个 warp 的权重装载横跨 8 条 cache line 却各用一半。
            // 改成相邻 lane 连续取列（第二列放到 +nh）后，一条装载 = 32 个
            // 连续 float4 = 512 B = **4 条满载 line**。输入侧不受影响：输入
            // 地址与列号无关，仍是组内广播。
            //
            // 双射：`n4 = gid % nh ∈ [0, nh)`、`n4b = n4 + nh ∈ [nh, nv)`，
            // 两列合起来恰好覆盖 `[0, nv)`（要求 `nv` 为偶，内核既有契约）。
            // **每个输出元素的累加顺序一字未变**（仍按 ci4 递增），所以结果
            // 与 base 映射位级一致——探针里误差 2.24e-8 与 `p_base` 完全相同。
            n4 = gid % nh;
            n4b = n4 + nh;
            m0 = (gid / nh) * 4u;
        } else {
            n4 = (gid % nh) * 2u; // 第一列的 float4 下标（第二列 = n4+1）
            n4b = n4 + 1u;
            m0 = (gid / nh) * 4u;
        }
        uint dcol = n4b - n4; // 第二列相对第一列的偏移（循环外算一次）

        // 8 个累加器：行 × 2 列
        float4 a0c0 = float4(0.0), a1c0 = float4(0.0), a2c0 = float4(0.0), a3c0 = float4(0.0);
        float4 a0c1 = float4(0.0), a1c1 = float4(0.0), a2c1 = float4(0.0), a3c1 = float4(0.0);
        uint oh[4], ow[4];
        bool ok[4];
        for (uint i = 0u; i < 4u; i++) {
            uint m = m0 + i;
            ok[i] = m < mDim;
            oh[i] = m / outW;
            ow[i] = m - oh[i] * outW;
            if (EDGE && oh[i] >= y_lo && oh[i] <= y_hi && ow[i] >= x_lo && ow[i] <= x_hi)
                ok[i] = false; // 已被 n_conv8_mma3 覆盖
        }
        // 整块落在覆盖矩形内 ⇒ 本 tile 无事可做。紧凑网格之后这不是**全覆盖**
        // 的兜底了（网格已是补集的子集），而是「子集里仍含被覆盖像素的那部分
        // tile」的最后一道过滤：中带头部那个行组、以及左右边界不对齐 4 的行
        // 组，都可能整块落在矩形内或部分落在矩形内。留着它，覆盖命题的证明
        // 就仍然是「网格 ⊆ 补集、逐像素 ok[] 裁到补集」两段，与网格公式解耦。
        if (EDGE && !(ok[0] || ok[1] || ok[2] || ok[3])) continue;

        for (uint t = 0u; t < taps; t++) {
            uint ky = t / kW, kx = t - ky * kW;
            for (uint ci4 = 0u; ci4 < cin4v; ci4++) {
                float4 g0 = float4(0.0), g1 = float4(0.0), g2 = float4(0.0), g3 = float4(0.0);
                for (uint i = 0u; i < 4u; i++) {
                    if (!ok[i]) continue;
                    int ih = int(oh[i] * sH + ky) - int(padT);
                    int iw = int(ow[i] * sW + kx) - int(padL);
                    if (ih < 0 || ih >= int(inH) || iw < 0 || iw >= int(inW)) continue;
                    uint px = uint(ih) * inW + uint(iw);
                    float4 v = d[ib + px * cin4v + ci4];
                    if (i == 0u) g0 = v; else if (i == 1u) g1 = v;
                    else if (i == 2u) g2 = v; else g3 = v;
                }
                uint wRow = wb + (t * cin4v + ci4) * 4u * nv + n4;
                for (uint i = 0u; i < 4u; i++) {
                    // WVec → float4：f32 视图下是恒等，fp16 视图下是
                    // half4→float4 的升宽。这是整个 fp16 路径**唯一**的
                    // 语义差，其余（下标、条数、累加顺序）逐字相同。
                    float4 wv0 = float4(w[wRow + i * nv]);
                    float4 wv1 = float4(w[wRow + i * nv + dcol]);
                    float s0 = g0[i], s1 = g1[i], s2 = g2[i], s3 = g3[i];
                    a0c0 += s0 * wv0; a1c0 += s1 * wv0; a2c0 += s2 * wv0; a3c0 += s3 * wv0;
                    a0c1 += s0 * wv1; a1c1 += s1 * wv1; a2c1 += s2 * wv1; a3c1 += s3 * wv1;
                }
            }
        }
        if (b_off != OFF_NONE) {
            float4 bv0 = d[(b_off >> 2) + n4]; // bias 全批共享（曾有 b*nv 越界）
            float4 bv1 = d[(b_off >> 2) + n4b];
            a0c0 += bv0; a1c0 += bv0; a2c0 += bv0; a3c0 += bv0;
            a0c1 += bv1; a1c1 += bv1; a2c1 += bv1; a3c1 += bv1;
        }
        if (act != 0u) {
            a0c0 = apply_act(a0c0, act, c1, c2, c3);
            a1c0 = apply_act(a1c0, act, c1, c2, c3);
            a2c0 = apply_act(a2c0, act, c1, c2, c3);
            a3c0 = apply_act(a3c0, act, c1, c2, c3);
            a0c1 = apply_act(a0c1, act, c1, c2, c3);
            a1c1 = apply_act(a1c1, act, c1, c2, c3);
            a2c1 = apply_act(a2c1, act, c1, c2, c3);
            a3c1 = apply_act(a3c1, act, c1, c2, c3);
        }
        if (r_off != OFF_NONE) {
            uint rb = (r_off >> 2) + b * mDim * nv;
            if (ok[0]) { a0c0 += d[rb + m0 * nv + n4]; a0c1 += d[rb + m0 * nv + n4b]; }
            if (ok[1]) { a1c0 += d[rb + (m0 + 1u) * nv + n4]; a1c1 += d[rb + (m0 + 1u) * nv + n4b]; }
            if (ok[2]) { a2c0 += d[rb + (m0 + 2u) * nv + n4]; a2c1 += d[rb + (m0 + 2u) * nv + n4b]; }
            if (ok[3]) { a3c0 += d[rb + (m0 + 3u) * nv + n4]; a3c1 += d[rb + (m0 + 3u) * nv + n4b]; }
        }
        if (ok[0]) { d[ob + m0 * nv + n4] = a0c0; d[ob + m0 * nv + n4b] = a0c1; }
        if (ok[1]) { d[ob + (m0 + 1u) * nv + n4] = a1c0; d[ob + (m0 + 1u) * nv + n4b] = a1c1; }
        if (ok[2]) { d[ob + (m0 + 2u) * nv + n4] = a2c0; d[ob + (m0 + 2u) * nv + n4b] = a2c1; }
        if (ok[3]) { d[ob + (m0 + 3u) * nv + n4] = a3c0; d[ob + (m0 + 3u) * nv + n4b] = a3c1; }
    }
}

// 入口一：f32 权重（默认）。参数块语义、网格公式与改写前逐字相同。
kernel void n_conv8(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 tpg [[threads_per_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    conv8_body<float4, 2u, false>(d, params, p_off, gid3, tpg, wg);
}

// 入口二：fp16 权重。**参数块完全一样**——`w_off` 还是同一个 word 偏移，
// 只是内核按 half4 去折它（权重段由 `repack_conv_w_f16` 压成半个长度）。
// 计划层用 `PlanBuilder::weight_f16` 选，见文件头。
kernel void n_conv8h(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 tpg [[threads_per_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    conv8_body<half4, 1u, false>(d, params, p_off, gid3, tpg, wg);
}

// 入口三：**边框补算**。与 `n_conv8_mma3` 配套——MMA 那支只吃「全内点块」
// （每块 = 一行内 64 个连续输出 x，9 个 tap 的读全落在平面内），本入口负责
// 剩下的补集：两行整行、两列整列、以及每行尾不到 64 的余数。
//
// **参数块与 `n_conv8` 逐字相同**（同一次 conv 的两条 dispatch 读同一块
// 参数，编码期不能改；覆盖矩形由内核从标准字段现算）。计划层保证这两支的
// 覆盖**恰好互补、无交**——交了的后果是「同一个输出像素被两个内核写」，
// 谁后写谁赢且结果取决于 dispatch 顺序，是那种只在特定形状上偶发的静默错。
//
// 网格与 `n_conv8` 同形（`ceil(m/4)*nh`），因为 tile→像素的映射一字未动；
// 代价是全图 tile 数的一次空转，靠内核里的整块早退压到指令级。
kernel void n_conv8_edge(
    device float4* d [[buffer(0)]],
    device const uint* params [[buffer(1)]],
    constant uint& p_off [[buffer(3)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 tpg [[threads_per_grid]],
    uint3 wg [[threadgroup_position_in_grid]])
{
    conv8_body<float4, 2u, true>(d, params, p_off, gid3, tpg, wg);
}
