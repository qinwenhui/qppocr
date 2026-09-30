# 对标基准：对象、数据、口径、已证伪清单

> 这份文档存在的唯一目的：**换会话/换人之后不用重新踩一遍坑**。
> 每一条都是实测出来的，不是推断。改动前先读「已证伪」一节。

## 1. 对标的是谁

| | 路径 | 是什么 |
|---|---|---|
| **对手** | `D:\qinwh\code\myself\SimdPaddleOCR` | 别人的 **C#** 项目（Sdcb.SimdPaddleOCR），**这才是对标对象** |
| 移植来源 | `D:\qinwh\code\myself\ocr-demo` | 我们**自己**的 C++ 原型，qppocr 就是照着它移植的。**拿它当基准只能做内部回归，不能当对外结论** |
| 本项目 | `D:\qinwh\code\myself\qppocr` | 纯 Rust 手写内核的 PP-OCRv6 推理引擎 |

⚠ 曾经因为搞错对象，在 CHANGELOG 里写过「我们快 1.43x」——实际对着 SimdPaddleOCR 是慢 1.5–1.8x。

## 2. 数据与模型

- **语料**：`D:\qinwh\code\myself\sku-manager\ocr-tool\bench\simdpaddleocr-dataset-v1\dataset`
  —— 100 张 `img-001.jpg … img-100.jpg` + `metadata.json`（GT 文本 / `cls_degrees` / `bbox`）。
  这是**对方自己的评测集**。
- **模型**：`models/{tiny,small}/det.onnx` 与对方 `models/` 下的
  **逐字节相同**（SHA-256 已核：det `193bab7a…`、rec `9ef676d6…`）。
- **评分口径**：用对方 `BenchSummary.cs` 的算法复刻在
  `C:\Users\qin_w\AppData\Local\Temp\claude\score_both.py`
  （exact_lines 逐行精确、exact_img、CER、cls 按 IoU≥0.3 配对）。
  **准确率一直是我们高**：tiny 逐行精确 83.3% vs 71.5%、small 94.9% vs 91.6%。

## 3. 怎么跑（复现命令）

对方 runner：
```
cd ../SimdPaddleOCR/test/Sdcb.SimdPaddleOCR.Tests/bin/Release/net10.0
./Sdcb.SimdPaddleOCR.Tests.exe --benchmark --benchmark-kind simd \
  --engine sharp --workers 8 --model tiny \
  --input <dataset> --count 100 --warmup 1 --case-id X --out out.json
```
- `--workers` 是**行（cls/rec）worker 数**，不是多图批量。det 另有 `DetIntraOpThreads`
  （默认 auto、上限 16），可用 `PPOCR_DET_THREADS` 覆盖。
- 它的 `timing` 块**分项取自最后一次迭代**（它自己的注释也承认），只有
  `best_total_ms` 可信。
- 结果 JSON **不落 boxes**，存量 JSON 重打分只能按文本配 cls。
- `PPOCR_DUMP_NODES=1` 会 dump 逐节点带形状的耗时（`244n`=tiny det、`163n`=rec、`107n`=cls）。

我们的 CLI：
```
./target/release/qppocr.exe <img1> <img2> ... --tier tiny --json --quiet --workers 1
```
- `--json` 里 `total_ms` = **best_ms**（`--bench N` 时是 N 次里的最优），分项也取自最快那次。
- `--rec-shards N` 覆盖 rec 分片数：`N` 显式分片、`0` 关、不传 = 按档位自动。
- **必须一次进程传完 100 张图**，理由见 §4。

## 4. 测量协议（血的教训）

**这台机器（Intel Core Ultra 5 125H，6P+8E / 18 逻辑核）在分钟级上有 20–40% 的频率漂移。**

- 任何「先跑完 A 再跑完 B」的协议都是错的：第二个配置接手的是已经被烧热的机器。
  **我因此连续报错三次结论**（分片 1.26x、深度卷积不换布局、稳态 1.87x），全部复现不出来。
- **唯一可靠的做法**：`python bench/ab.py vs` —— 每轮交替跑两边、**算当轮比值**、
  最后取比值中位。4–6 轮就有稳定结果。
- 反直觉但重要：**对方比我们抖**（连跑 6 轮极差 11.8% vs 3.6%）。他们的均值
  在 59–80 ms 之间跳，我们的稳定在 89–95。所以只看一次运行的比值没有意义。
- 对方 `--warmup 1` 只丢第一张，而它前 ~20 张在逐档建按宽度缓存的会话
  （`rec_graph` 从 231 ms 降到 119 ms）。**要比较稳态就得两边都丢掉前 20 张。**

### 4.1 ★ 两边都必须是「热进程」（本条曾经错了整整一轮）

**一图一进程会把冷启动开销平摊到每一张上。** 同图同配置实测：

| | 同进程第 2 张起 | 每张一个独立进程 |
|---|---|---|
| det | **33.7 ms** | **44.4 ms** |
| total | **75 ms** | **89 ms** |

对方的 runner 是**一个进程跑 100 张**（`--count 100`），拿我们的冷进程去比它
等于白送 19%。`ab.py` 早期版本就是每张起一个进程，于是报出的 tiny 1.47
是虚高的；改成同进程 100 张后是 **1.25**。

自检：同一张图在**一个进程里**跑 8 遍，`total` 从 89 掉到 74 就说明中招了。
（成因是首次前向的冷页/分配器/频率爬升，不是引擎缺陷——重复两遍 200 张，
第 2 遍 71.7 vs 第 1 遍 72.9，没有持续劣化。）

- 每次实验前先跑 `scaling_probe`（1% 以内才开工）。
- `python bench/abx.py <tier> <轮数> --a "<参数>" --b "<参数>"` 做任意配置的
  交错 A/B；`python bench/verify.py save/diff <名字>` 做 100 图逐字符对拍。

机器特性（实测，探针在 `crates/qppocr-kernels/examples/scaling_probe.rs`）：
| 项 | 值 |
|---|---|
| 纯计算扩展 | 1/2/4/8/16 线程 = 1.00/1.82/2.72/4.26/**5.05x** |
| 内存带宽 | 1/2/4/8/16 线程 = 29.6/50.1/45.2/37.7/**26.1 GB/s**（峰值在 2 线程） |
| fork 协调 | 2 线程 **0.6 µs**、4 线程 2.2、8 线程 12.1、16 线程 26.1 µs |
| 单核 FMA | 纯寄存器探针 54.5 GFLOPS；而 sgemm 实测 **82–97 GFLOPS** |

## 5. 当前位置（`bench/ab.py vs`，4 轮交错中位，热进程口径）

### 5.1 我方口径 = `--preset speed`（产品速度模式）

- **区域重试必须关**（speed 里就是关的）：重试是本引擎的应用层功能，
  对方没有；开着比等于我们平均多跑 9/100 张图的第二遍 det（+2.5ms 均值）。
- **rec_height 40**（speed 的值，对方 48）：rec 计算量 −31%，精度代价见下。
- 精度（对方自己的评测集、对方算法的评分口径）：
  | | 我们 speed | 我们 balanced | 对方 |
  |---|---|---|---|
  | tiny 逐行精确 | **79.2%** | 83.3% | 71.5% |
  | small 逐行精确 | **91.9%**（h44=94.5%） | 94.9% | 91.6% |
  即 speed 模式下两档精度仍高于对方；small 想留余量用 `--rec-height 44`。
- **rec_height 32 不可用**（tiny 59.2%，击穿对方下限 71.5%）——CJK 高度
  敏感，32 只够拉丁。


**⚠ 比值随机器状态变，同一个协议同一天量出过两个档位：**

| 机器状态/口径 | tiny | small | 备注 |
|---|---|---|---|
| 最初（冷进程 + balanced） | 1.47 | 1.26 | 口径错的，只作历史参照 |
| 热进程 + balanced | 1.25 | 1.33 | |
| **热进程 + speed（当前）** | **1.07–1.10** | **1.12** | 已收：mimalloc、cc 免 label、`--rec-shards 0` 修复、分片默认开 |

> **★ 2026-09-27 终局：两档全部低于对方，8 轮全胜。**
> | | 比值中位 | 4 轮明细 |
> |---|---|---|
> | tiny | **0.965** | 0.952 / 0.971 / 0.958 / 0.973 |
> | small | **0.937** | 0.923 / 0.931 / 0.956 / 0.943 |
>
> 最后一块拼图是 **Conv+ResidualAdd 残差融合**（rec 每图折 ~58 个 Add、
> det 双分支合并同理，两档文本零差异）。此前：BN 折叠后 tiny 1.088 /
> small 0.984；最初 1.47 / 1.33。
> 全天同一协议、同一评测集；精度全程高于对方（tiny 79.2% vs 71.5%、
> small 91.9% vs 91.6%）。
> 剩余差距全在行阶段的执行器结构：对方 8 个单线程行 worker 拿 5.7x，
> 我们最好的分片配置 ~4x（裸内核并发上限 5.9-6.4x，探针已证）。

两次都是 4 轮交错，轮内比值很稳（极差 ≤0.06），**不是噪声**。变的是机器：
`scaling_probe` 的纯计算扩展从 1/2/4/8/16 = 1.00/1.82/2.72/4.26/**5.05x**
掉到 1.00/1.95/3.34/3.72/**4.46x**。而且两边的**绝对值都变慢**（我们 +7%、
对方 +18%），所以这是机器状态、不是哪一方的改动。

**结论：报比值时必须同时报当时的 `scaling_probe`。** 拿早期那个 5.05x 状态
量出来的 1.25 去和现在的实现比是无效的。
→ 待办：把 `ab.py` 每次运行时的 scaling_probe 结果一起落盘。

> 改口径之前的报数是 tiny 1.47 / small 1.26，两个数都不可比：tiny 是冷进程
> 虚高，small 反倒在同进程下变慢（272 → 285，原因未查清，见 §8）。

tiny 分阶段：det 28.5 vs ~24（1.2x）、行阶段 35 vs ~22（1.6x）。
small 分阶段：det 70 vs ~70（**持平**）、行阶段 203 vs ~147（1.38x）。

**结论**：small 的差距 100% 在行阶段（cls+rec）；tiny 的差距三七开，
det 已经不多了、行阶段是大头。

## 6. 差距的根因（已定位到这一层）

1. **内核不是问题**：单算子对比我们常常更快（例如首层 `3→16 k3x3 s2`，
   对方单图 26.2 ms、我们 2.5 ms）；sgemm 也超过纯 FMA 探针。**所以不是「某个算子写得慢」。**
2. **差距在「他们干的活少」**：他们 9 类节点融合（我们 4 类：
   Conv+ReLU、MulAddScale、GELU 子图、Identity），加上整段 channels-last。
   同样 177/244 个节点跑同样的形状，他们的中间张量产生与搬运更少。
3. 我们的 det 并行扩展 3.73x（上限 5.05x）、行阶段 3.75x——**41% 的 det 时间花在扩展 <3x 的算子上**
   （MaxPool 2.7 ms、Concat 2.6、深度卷积 4.7）。

## 7. 已证伪清单（**别再走一遍**）

| 方向 | 结果 | 备注 |
|---|---|---|
| **NHWC / channels-last**（1×1 卷积改写成 `Y[p][co]=ΣX[p][ci]·Wt[ci][co]`） | **0.94x（更慢）** | 见 `examples/nhwc_probe.rs`。他们的 NHWC 收益是**在修他们自己的问题**：`Conv1x1PackedNhwc` 每算子要转两趟 NCHW↔NHWC，`LayoutPlanner` 只是消掉这些转置。我们的 NCHW 1×1 本来就是干净的 GEMM，没有可省的转置。 |
| **深度卷积不补零平面**（照他们源码改成直接在原平面做） | 慢 7.6%，已回退 | in-situ 27.5 → 29.6 ms。深度卷积**不是带宽瓶颈**，省下的搬运抵不过内层寻址变复杂。 |
| ~~**两级并行 / 多池分片**：0.998x，无差别~~ | **搞错了，见下** | 这条是**假的**：`--rec-shards 0` 当时被 CLI 吞掉（`if o.rec_shards > 0` 守卫），对照实验两边跑的是**同一个配置**。修好后重测：分片对 tiny 值 **1.12x 整体 / 1.29x 行阶段**（4 轮交错中位 0.890，极差 0.869–0.913）。默认（tiny 12 / small 4）保留。 |
| `rec_batch > 1`（批内同行宽差 ≤10% 才并批） | **1.10x，但改用** | 见 §8：它改了 8/1099 行的文本。 |
| **并行门槛调整**（`fork_min_macs` 4M→8M/16M/32M） | 旧结论不可信 | 那一轮同样是拿 `--rec-shards 0` 当对照组。见 §8 的 8M 线索。 |
| cls 批化（cls_batch 1→2/4/8/16） | 更慢（4.12→11.95 ms/图） | 用旧口径量的，**没重测**。 |
| 线程局部小块缓冲缓存 | 慢 2.7% | 全局池锁不是瓶颈。 |
| 拿 `ocr-demo` 当对标 | 口径错 | 见 §1。 |

## 8. 未完成 / 下一步候选

按估计收益排（**都没验证过，别当结论**）：

1. **★ 行阶段（cls + rec）——扩展其实没问题，问题在绝对工作量。**

   **⚠⚠ 这一节前后写错过三次，基线都取错了。** 记下来：
   1. 第一次用 `QPPOCR_ROW_THREADS=1` 当串行——但 `run_batches` 在
      `nthreads == 1` 时走**提前返回的分支，那里没有 `set_serial_execution`**，
      每个算子照样 fork 满池；
   2. 第二次用 `--threads 1` 当串行——但**分片的外层 11 个线程是
      `run_batches_sharded` 自己 spawn 的，跟池大小无关**，照样并行；
   3. 第三次才对：**`--rec-shards 0 --threads 1`**。

   **自检规则（务必照做）**：真串行的判据是
   **profiler 的「内核合计」必须 ≤ 该次墙钟**。前两次内核合计都远大于墙钟
   （143 ms vs 44.9 ms），一眼就能看出不是串行。

   **订正后的对照**（tiny、20 图、同进程热态、`--rec-shards 0 --threads 1`
   为基线）：

   | 配置 | rec ms | cls ms | 行阶段 | vs 真串行 |
   |---|---|---|---|---|
   | **真串行** | 122.1 | 13.4 | 135.5 | 1.00x |
   | 行并行 ×11 | 40.5 | 4.2 | 44.7 | 3.03x |
   | **默认（分片 12，每片 2 线程）** | **30.1** | 4.4 | **34.5** | **3.93x** |

   **读法：rec 4.06x、cls 3.05x。而机器当前的并行上限（`scaling_probe`
   16 线程）只有 4.49x——行阶段已经吃到 87%。** 这里没有多少可捞的了。

   **参照**：同一个 sgemm 内核、同一个 rec 形状，11 个独立线程 5.9x
   （`examples/gemm_shape_probe.rs mt:160x180x320`）——说明内核与机器都正常。

   **已排除的候选**（都实测过，别再走）：`fork_join` 没检查 `serial_exec`
   （检查了，`pool.rs:335`）；分配器竞争（mimalloc 只帮 det）；缓冲池锁里
   做 malloc/free（已移出，无差别）；形状太小（`gemm_shape_probe` 单线程扫过，
   34–40 GMAC/s 无形状悬崖）；每线程工作集超 L3（放大到 3 MB/线程仍 6.4x）。

   **真串行下的卷积效率**（`--rec-shards 0 --threads 1`，按行独立计数，
   别用跨会话聚合的脚本——那个会把调用次数乘错，量出过 3 倍的假差）：

   | 模型 | Conv 合计 | MMAC | 平均 |
   |---|---|---|---|
   | rec | 61.7 ms | 2333 | **37.8 GMAC/s** |
   | det | 115.5 ms | 3484 | 30.2 GMAC/s |
   | cls | 4.7 ms | 211 | 45.0 GMAC/s |

   参照 `gemm_shape_probe` 的单线程上限 **34–40 GMAC/s**：
   **rec 已经贴着内核上限（~95%），det 是 30.2（被小 M 的 conv 拖）。**
   所以「内核还有 1.5x 余量」也是错的——两侧都没剩多少。

   **那么 tiny 的 ~73 ms 是这样构成的**：det 28.5 + 行阶段 34.5 + 其余 ~10。
   行阶段的机器上限是 135.5/4.49 = 30.2 ms，我们在 34.5（差 13%）；
   det 的并行扩展 3.7x 对上限 4.49x（差 18%）。**没有一处是病态，
   是一堆 10–20% 叠起来。**

   > 顺手量到、**可以直接拿**的东西：mimalloc 全局分配器（只加在 CLI 上）
   > det 稳定快 ~1 ms、整体 4 轮中位 0.983。**没采纳**是因为它会削弱
   > 「纯 Rust」这个卖点，而收益只有 1.7%。

2. **`rec_batch > 1`**：实测 total 74.6 → 67.8（1.10x）。**但代价是改了
   8/1099 行的文本**（批内按最宽行 pad，宽度一变识别器读法就变）。
   逐条看过：3 条变好（`Paddle0CR→PaddleOCR`、补出正确空格）、4 条变坏、
   1 条中性。**在「逐字符不变」的纪律下不予采纳**，留在 `QPPOCR_REC_BATCH`
   后面当实验开关。
   - 只并**完全相同宽度**的行（`QPPOCR_REC_BATCH_RATIO=1.0`）是**逐位不变**的，
     但这种行太少，实测没有收益（rec 26.4 vs 26.8）。
3. **★ `QPPOCR_GEMM_MIN=8000000` 是一个量到了、但没搞清机制的线索。**
   6 轮交错（tiny，100 图）：比值中位 **0.983**，后 5 轮全部落在
   0.978–0.999，**行阶段 40.4–41.1 → 38.0–38.6（+7%）**，但
   **det 一致地 32.0 → 33.2（−3%）**，净 +1.7%。small 3 轮中位 0.988。
   - 试过的解释都**不成立**：按「分片池只有 2 个线程 → 门槛同比放大」
     （`fork_scale`，两种标定：`g/p` 与 40 倍）A/B 都测不出收益
     （中位 1.015 / 1.012），已回退。
   - 所以**没有采纳**：机制不明 + det 明确变慢，不够格进默认值。
     留给下次：先用 `QPPOCR_PROF` 把 8M 前后**逐算子**的差异打出来，
     看是哪几个节点进/出了并行分支，再决定。
4. **`small` 档同进程反而变慢**（272 冷 → 285 热）没查清。两遍 200 张的
   复测（tiny）显示没有持续劣化（71.7 vs 72.9），所以不像内存增长；
   small 上没做同样复测——**先补这个**，它是 small 从 1.26 变 1.33 的原因。
5. **更多节点融合**（对方有 9 类，我们 4 类）：`Conv + Bias + ResidualAdd`、
   `ConvTranspose + Bias + Activation`、`Conv + HardSwish`。det 的 `Add`
   有 15 处、93 MB，融进生产者卷积能省掉「写中间张量 + 再读回来」两趟。
6. **det 里 <15 GB/s 的算子**：MaxPool 8.8、Resize 13.2、Sigmoid 4.6 GB/s。
   合计约 5 ms，**但这些是带宽绑定的，不是内核写得差**：
   - MaxPool 已试过**单趟化**（去掉 scratch 行）——**无收益**（2.6-2.9 →
     2.5-3.1 ms）。它读 2 行 + 写 1 行 = 38 MB 实际流量 @ ~13 GB/s，
     瓶颈就是 DRAM，记账口径的 25.6 MB 低估了它。
   - `connected_components` 2.6 ms 试过拿掉整除——**没用**（LLVM 已强度
     归约），占 DB 后处理的 70%，真要动得换算法。
   → 这一类的正确做法是**减少流量**（融合掉上游的写），不是再抠内核。
7. **区域重试触发率**：我们 9/100、对方 7/100，触发一次该图 det 翻倍。
   ⚠ 但**测试时用的是关掉重试的配置**（`Preset::Speed`），这是应用层功能，别当性能问题。

## 9. 仓库状态

- 分支 `internal-main`（**含内部移植叙述，绝不推送**）；
  公开快照是孤儿分支 `release-vX.Y.Z`（单提交、无父）+ tag。
- 当前 tag `v0.1.2`，快照见 `git log --oneline -1 release-v0.1.2`。
- **推送被网络阻断**（github.com:443 连不上），提交都在本地。
- 全程纪律：**每次改动都必须过 100 图逐字符 + 每字坐标对拍**——
  `python bench/verify.py save <名字>` 存基线、`diff <名字>` 比。
  两档基线：`base`（tiny，1099 行）、`pre-small`（small，1042 行）。
  性能改动一律用 `python bench/abx.py` 交错 A/B，不比单跑。

## 2026-09-28 对方 GPU 路线数据（外部报告，未同机复测）

机器：AMD Ryzen AI H365 笔记本，Radeon 880M 核显（RDNA 3.5, 12CU，
LPDDR5X 共享带宽 ~120GB/s 级）。对方 Vulkan 后端，4 workers，据称同
100 图语料：tiny 30ms / small 46ms / medium 117ms（口径疑似吞吐归一）。

拟合 t = a + b·FLOPs：固定开销 a ≈ 19–25ms/图，与档位无关——dispatch/
barrier/传输/同步/后处理是对方的大头，tiny 尤其被压住。medium 在其
GPU 路线上首次「可用」（<120ms）。

我方含义：
- 笔记本/核显场景 CPU-only 无胜算（我方桌面 16 线程 small ~250ms，
  对方核显 46ms）——GPU 后端是必争之地，优先级上调；
- 攻击面 = 固定开销：mega-kernel 融合 + 异步 staging + 全异步提交，
  目标把 a 压到 ~5ms 级；带宽墙 → 隐式 GEMM（免 im2col）+ 后置 fp16；
- 验收：同 H365 同 100 图，bench --device gpu 对拍三档全胜；
  开发用任意 Vulkan 卡，880M 数据可由社区/实机回传。

## 2026-09-29 我方 Vulkan det 路线落地（Intel Arc Pro 核显，本机实测）

后端：ash + 计算着色器（glslc 编译、SPIR-V 签入），NHWC f32 存储 +
k-major 权重预排（f16 曾实测 GPU 总 16.9ms，但 SE 门乘性放大逐层
舍入、真实图概率图衰减——det 上 fp16 存储不可行，已转 f32）。

单 det 前向（960×864，微基准 det_session_end_to_end，GPU 二跑）：

| 路径 | 墙钟 | GPU 总时间 |
|---|---|---|
| 旧 NCHW f32（conv/conv_gemm） | 77.2ms | — |
| n_ NHWC f32（本轮） | 28.4ms | 24.7ms |

分解（QPPOCR_GPU_PROF）：n_conv 12.8ms / n_conv_dw 2.2 / n_channel
2.1 / n_elem 1.7 / concat 1.6 / convT 1.2 / resize 0.9 / reduce 0.7。

正确性口径：两条 GPU 路径末图 mean|d| = 2.1e-8（逐位一致）；对 CPU
executor 的浮点序差异（概率图 mean|d| ~5e-2、>0.3 像素 52508 vs
51538）与旧路径相同——引擎级 verify.py 表现一致（文本同、conf 浮点差）。

端到端（ab.py tiny 4，--workers 1，QPPOCR_GPU_STAGES=det）：比值中位
1.088（GPU 整链比 CPU 慢 9%）——det 本身 43→31ms，但 GPU 会话的
prefers_host_parallelism=false 保守门控把 rec/cls 批退化为串行，吃掉
det 的收益。Phase 2 待办：rec/cls 上 GPU 或门控改按会话粒度。

途中修的三个横切 bug（值得记录）：
- reduce_hw 曾单 WG 串行（8.4ms）；staging 非 HOST_CACHED 型主机读回
  慢百倍（26ms）——两者都曾伪装成「GPU 慢」；
- fuse_conv_residual 折进 conv 的 inputs[3] 残差，n_ 内核曾整个漏读
  （随机输入对拍不暴露，真实图才发散）——跨会话对拍要两条 GPU 路径
  互比，CPU dump 的节点序与 GPU 计划不同源不可直接配对。

## 2026-09-29（二）端到端反超 CPU + 计划重建三级缓存

上一节的「ab tiny 1.088」已翻到 **0.93-0.95**（GPU 快 5-9%，交错
ab.py，--workers 1；small 0.75-0.90）。两个独立根因：

1. **EngineBuilder 一刀切**：--device gpu 时无条件 rec_shards=0，
   把 QPPOCR_GPU_STAGES=det 下仍是 CPU 会话的 rec/cls 行并行也关了
   （CPU rec 拖慢 37%，吃掉 det 的全部收益）。运行时
   prefers_host_parallelism() 三处门控本就按会话判断——构造期这道
   是冗余保险，删掉。
2. **形状多样性 × 计划逐出 → 每帧重建**：100 图语料 91 个 distinct
   形状、重建 11-18ms，det p50 曾被推到 52ms。三级缓存：
   VkPipelineCache（38 管线 5-9→1-2ms）、权重 k-major 重排缓存
   （与形状无关）、arena 池（逐出计划整块 reset 复用，免 137MB 页
   提交 ~7ms）；计划缓存计数 LRU→字节预算。重建 → 2.5-3.2ms。

教训存档：计时开关与日志开关必须分离（QPPOCR_GPU_STEP_DEBUG 的
日志自身 ~6ms，曾把节点环误判成热点）。

## 2026-09-30 首次 GPU 模式对决（n_resize 修复后，GPU 输出与 CPU 逐字符全同）

机器状态：scaling_probe 16T = **4.36x**（参考静基线 5.05x，当日劣化 ~14%，
两边绝对值均偏慢，比值与 09-27 的 0.965/0.937 不可直接比）。

| 场 | 比值中位 | 4 轮明细 | 备注 |
|---|---|---|---|
| CPU `ab.py vs` tiny | 1.058 | 1.03–1.07 | 对方 59–64ms 处其波动区间快端 |
| CPU `ab.py vs` small | ~1.13 | 1.12–1.20 | 我方绝对值比 09-27 慢 ~18%（机器） |
| **GPU tiny（首次）** | **1.009** | 0.97–1.04 | det 22.1 vs 对方 ~24；GPU 把 CPU 场的 +5.8% 拉回 ±0 |

- **GPU small 不支持**：small det 的 SE 归约通道 384 > 256（n_ 两阶段内核
  上限），`--device gpu --tier small` 干净报错。要么扩内核上限要么按图
  回退 CPU（当前 neither）。
- GPU 场曾在连续 CPU 基准后偶发一次整批空输出（进程级失败，重跑 8+ 次
  未复现；嫌疑驱动状态，未定位——再遇到先看 stderr 是否 device lost）。
- 当日绝对值参考（同机同轮）：我们 CPU 63–67 / GPU 59.7–60.3（r4 一次
  66.6）；对方 59–64（对方仍比我们抖，与 §4 一致）。

## 2026-09-30（二）detrec 分级：rec 上 GPU 的精度平价与性能地板

- **`QPPOCR_GPU_STAGES=detrec`**（新增：det+rec 上 GPU、cls 留 CPU）：
  GT 评分 **860/1036 与 CPU 逐位全同**（cer/char_acc/cls 全部同数字）——
  rec 宽度分桶（`DeviceSession::bucket_grain`=64，右侧零填充 ≤63 列）零精度代价。
- **挖出历史 bug**：`open_bytes_with` 建 cls 会话时 `opts.model` 残留 Rec——
  cls 一直被设备层按 Rec 路由（det-only 时代无感，detrec 分级当天暴露为
  cls 误上 GPU → 翻转边界分歧毁 ~14% 行文本）。已修。
- **cls 永久留 CPU 的两条实证**：① GPU cls argmax 与 CPU 在真实图上有
  ~14% 行的边界翻转分歧（翻转错=整行毁，153/1096 行旋转不同）；② GPU cls
  5.3ms > CPU 4.2ms，无收益纯风险。
- **GPU rec 性能地板（未反超，本轮定量）**：暖机单行 GPU 总 1.7–4ms +
  提交往返 ≈ 5–6ms/行，串行 11 行 ≈ 68ms vs CPU 并行 37ms。两个直觉被
  数据否掉：批维合批 per=8 → (bsz∈1..8)×宽桶 ≈ **106 形状/百图、重建摊销
  88ms/图**（364 次构建）；批维补齐到 8 → 宽桶上纯 8× 算力（bsz=1 时宽行
  网格已吃满 EU，实测 GPU 总 = 8× 单行）。**bsz=1 + 宽桶（~20 形状）是
  平衡点**，rec 68ms——反超 CPU 需 rec 内核本体的 det 级优化（60 dispatch/行）。
- 引擎默认维持 det-only（0.955–0.968 不变）；detrec 为后续 rec 内核优化
  的现成地基。

## 2026-09-30（三）rec 合批 + argmax 出口 + 两个一致性缺口

机器状态：scaling_probe 16T = 4.73x（参考静基线 5.05x）。

**每行开销构成的实测订正**（QPPOCR_GPU_PROF，img-001 暖机）：11 行 GPU 忙算
合计仅 **21.9 ms**（每行 1.3–3.2），提交往返 ~0.85 ms/行 ×11；真正的
大头是**读回**——T×V×4 B/行的概率矩阵全量拷出 ≈ **18 ms/图**
（clflush 读回 ~1 ms/MB）。上一节「5–6 ms/行」的记法把读回算进了提交。

三件套（全部 60 dispatch/batch 不变）：
1. **批维合批 + 早退**：同桶行合批至 8（`DeviceSession::batch_grain`），
   会话把批维补齐到 8 建计划，real_n 写 arena word 0，10 个 n_ 内核
   `params[0]` 早退守卫（0=不限，裸 dispatch 安全默认）——空行工作组
   零算力，(bsz,W) 形状塌缩成 (8,W桶)。上一节两条否决（per=8 形状爆炸 /
   补齐=8× 算力）的死刑都被早退豁免。
2. **argmax 出口**（`n_exit3_argmax`）：CTC 只吃每时间步 (val, idx) 对，
   读回 ~3 MB/行 → 8 B/行。解码语义与概率路径逐字等价（首个严格最大、
   平局最小下标——`ctc_decode_pairs`，单测位级背书）。
3. **计划预算 768→2048 MB**：rec 桶 24 个 × ~60 MB ≈ 1.4 GB，旧默认下
   每图互相驱逐（rec 71 ms/图、93 次重建）→ 全命中（24 次构建，p50 37）。

**两个新的一致性缺口（本机驱动，img-003 间歇整行空文本定位）**：
- GPU 侧：CB 头部全量内存屏障（每次重放执行）——GPU 缓存的旧行遮蔽
  新写，主机侧怎么 flush 都没用（无屏障 9/20 复现，加屏障 0/20）。
- 主机侧：real_n 单元的 4 字节小写要 `publish_clean`（SFENCE+clflush
  写回逐出；CLFLUSH 与 store 弱有序、必须隔 SFENCE）——大块 memcpy 靠
  容量自我逐出掩盖了同款问题。两道 A/B 实证缺一复现。与 09-29 的
  clflush 读回屏障合称本驱动三个一致性缺口；上会话「偶发整批空输出」
  疑即同类。

**精度**：GT 860/1036（=CPU、=（二）节 detrec）；CHUNK=1 ↔ 合批 100 图
1096 行逐字符+坐标+置信度 **IDENTICAL**；`rec_batch_pad_early_exit` 单测：
补齐 3 行@grain8 与逐行逐位一致 + argmax 对解码等价。宽桶 grain 128/256
更快（行阶段 37/35 vs 45）但丢精度（GT 859/853）——按逐字符纪律定格 64。

**性能**（abx 4 轮中位，--workers 1）：detrec vs det-only(rec CPU)
**1.136**（（二）节口径 1.57）——行阶段 GPU 45 vs CPU 37；单图暖机
detrec 68.3 vs 全 CPU 72.4（赢）。rec 语料 p50 37 / p90 54 ms。引擎默认
仍 det-only。**下一刀（已定量）**：批间流水（全部批提交→末批等待，
pack/读回/解码与 GPU 执行重叠，可藏 ~10 ms）；空行工作组的调度税
~3 ms（网格烤死在 CB，需可重录命令池）。

## 2026-09-30（四）批间流水：detrec 全链反超纯 CPU 26%

- **`DeviceSession::run_deferred` / `DeferredRun`**：提交不等、
  `complete()` 等信号+读回。CPU 默认实现 = 同步 run（语义等价零成本）；
  Vulkan 侧计划改 `Arc<Plan>`（deferred 持引用防预算逐出悬空），每计划
  `inflight` 原子标记兜底「同形状两条在飞 = 同址互踩」。
- **引擎流水循环**：pack(k+1)+提交(k+1) 与 GPU(k) 重叠、收账(k-1)
  （等+读回+解码）也与 GPU(k) 重叠；同桶连续批 = 同计划，先收账再提交。
  GPU 批间空隙从「pack+提交+等待+读回+解码」缩到 pack+提交。
- **精度不变**：GT **860/1036**（=CPU）；100 图 1096 行对逐行基线仍
  **IDENTICAL**；`rec_deferred_pipeline` 单测：双计划在飞 + 同计划
  收后重发，与同步逐位一致。34 测试绿。
- **性能**：
  - detrec vs **纯 CPU 全链**：**0.734**（79 vs 107 ms，4 轮极差 ±0.01，
    同源发热对照）——det 25 vs 41、行阶段 41 vs 55。
  - detrec vs det-only：机器状态依赖（CPU rec 16 线程自加热 37→60ms，
    GPU rec 恒 40-41ms 稳定）——冷机 det-only 略快（行 36.6 vs 40.4），
    持续负载下 detrec 反超（0.79-0.81）。**detrec 的卖点 = 稳定 + 释放
    全部 CPU**。
  - 单图暖机：rec 25.9ms（（三）节 30.7），总 64.8。
- 引擎默认仍 det-only（detrec 改默认需冷机对照补数——待定）。

## 2026-09-30（五）detrec 转正为 `--device gpu` 默认

机器状态：scaling_probe 16T = 4.69x。转正前补齐的最后两块证据：

| 场景 | det-only | detrec | 判定 |
|---|---|---|---|
| 冷进程单图（一次性 CLI，5 次中位） | **84.8 ms** | 157.7 | det-only 大胜——rec 计划构建税 ~13ms×6 桶 |
| 冷机语料（abx 4 轮中位） | 67.7-68.2 | 69.7-70.0（**1.028**） | 平手（构建税摊薄后 ~2-3%） |
| 持续负载（机器热态，见（四）） | 95-99（CPU rec 劣化 37→60） | 78-80（恒稳） | detrec 大胜 ~20% |
| CPU 占用 | rec 吃满线程 | 全释放 | detrec |
| 精度 | 860/1036 | 860/1036 | 平 |

计划构建税的构成（QPPOCR_GPU_BUILD_TIME，rec 计划 13.6ms）：节点环
8.7（map_node_n 4.7——CPU 侧图处理与权重重排）+ arena/KernelSet 2.7 +
其余 ~2——**管线创建只占 1-2ms，磁盘管线缓存救不了冷启动**；大头是
build_plan 本体，列为后续优化项。

**决定：默认 detrec**。产品形态（GUI 常驻、批量）都在热区且受益最大；
一次性 CLI 冷进程的 73ms 代价以 `QPPOCR_GPU_STAGES=det` 作为显式退路
并在此记录。验证：翻默认后无环境变量跑 100 图 1096 行对 detrec 基线
IDENTICAL；34 测试绿。

## 2026-09-30（六）small 档 GPU 通车：SE 384 + LayerNorm 链 + 探针分级

`--tier small --device gpu` 从干净报错 → **可用**（GT 978/1036 与 CPU
逐位全同，所有指标同数字）。三件事：

1. **SE 归约通道 384 > 256 解禁**：n_reduce_hw 的线程平铺要求 cp4 整除
   256——cp4=96 时尾线程 pg≥npg 的页序列与 pg=0 **重叠双计**（这才是
   旧上限的根源）。修法：m 循环挂 `pg < npg` 条件（尾线程空转但照常
   写 red/过 barrier——屏障发散是 UB）。n_reduce_fin 改通道按 256 分组
   跨步（组数 WG 一致、组内双 barrier）。上限放宽到 Cpad ≤ 1024
   （cp4 ≤ 256；超出静默算零，必须显式拒绝）。
2. **LayerNorm 分解链上 GPU**（small rec 的 SVTR 头，tiny 没有）：
   `ReduceMean(-1)` 新内核 n_reduce_last（softmax 同款行归约骨架，出
   [rows,4] lane0）；n_elem 扩 8=减行向量/9=square/10=sqrt/11=除行向量/
   12=sub/13=div（行向量广播 p1 携带 c4 位型）；rank-0 标量 Add（LN 的
   +eps，paddle2onnx 导出无维 helper.constant）；rank-3 MulAddScale
   （LN 的 ×scale+shift）复用 n_channel op5（bn 的双 [C] 广播，单
   dispatch 无临时区）。
3. **rank-5 注意力转置族（[2,0,3,1,4] 等 7 种排列 + MatMul ×13）不支持
   → 探针分级**：create_session 用一次性探针会话（克隆图/权重）小形状
   试建计划，失败则 stderr 声明后 rec 退回 CPU（非静默，cls 同款纪律）；
   补齐内核后探针自动放行。det 全 GPU 不受影响。

small 分账（img-001 单图）：det GPU 43.8ms（CPU ~70）；rec 140ms CPU
（注意力头是 small rec 上 GPU 的下一步）。34 测试绿。

## 2026-09-30（七）第二轮对决：两档全胜（tiny 0.682 / small 0.799）

机器：scaling_probe 16T = 4.32x（劣化态；对方绝对值 ~104ms 比首战的
59-64 慢——其已知波动 + 机器态；交错取比值控制）。协议：ab.py vs
4 轮交错，我们侧 `--device gpu`（tiny = detrec 全 GPU；small = det GPU
+ rec 探针退 CPU），--preset speed --workers 1，同进程 100 张丢首图。

| 档 | 我们 GPU | 对方 | **比值中位** | 极差 | 同机态 CPU 参考 |
|---|---|---|---|---|---|
| tiny | 70.2ms（det 24.1 行 32.7） | ~104 | **0.682** | 0.67-0.75 | 0.945 |
| small | 307ms（det 49 行 244） | ~388 | **0.799** | 0.79-0.81 | 1.123（极差 0.72-1.51） |

- **tiny 从首战 1.009 → 0.682**：detrec 转正 + 合批/argmax/流水四刀的
  合力，赢 32%。
- **small 从不支持 → 0.799 赢 20%**，且 rec 还在 CPU——注意力头上 GPU
  是后续余量。
- CPU 参考轮同时量化 GPU 贡献：tiny 0.945→0.682（~28 点）；small CPU
  轮两边的 16 线程互相加热、极差 0.72-1.51 不可用——GPU 模式反而恒稳
  （0.79-0.81），复证（四）节的发热不对称。

## 2026-09-30（八）★ 口径纠正：竞品有 GPU 后端（feature/2.0），GPU vs GPU 仍两档全胜 2.1-2.2×

**（七）节的「竞品全 CPU」结论是错的**——只看了对方 main 检出。竞品仓库
分支：`feature/2.0`（Vulkan+Metal 双 GPU 后端，DET/CLS/REC 全图上 GPU、
sg32l/coopmat/SE/CTC 调优、880M 实测报告）与 `devin/...-vulkan-gpu-backend`
开发分支。**cls 也上了 GPU**（PaddleOcrClassifier._gpuCapable + ClsGraph
档）——回答了「竞品 cls 用没用 GPU」：2.0 分支用了，main 没有。

对决设置：竞品 worktree（`../SimdPaddleOCR-gpu`，feature/2.0 @6298596）
dotnet build 后 `--engine vulkan`；我们 `--device gpu`。跑器协议同（七）。
机器 16T = 4.33x。工具：ab.py 增 `AB_SIMD_EXE`/`AB_THEIR_ENGINE` 透传。

| 档 | 竞品 GPU(vulkan) | 我们 GPU | **比值中位** | 极差 |
|---|---|---|---|---|
| tiny | 133 ms | 62-70 ms | **0.472** | 0.47-0.52 |
| small | 428-433 ms | 190-234 ms | **0.460** | 0.44-0.55 |

他们 vulkan（30 图阶段分解）：det_graph 42.5 / **cls_graph 11.5** /
rec_graph 76.4 / lines_wall 90.4——对照我们 det 22.6 / cls 4.2(CPU) /
行阶段 30.5。精度：他们 GPU 740/1036（=其 CPU 水位 71.4%，GPU 无损）vs
我们 860/1036。

**关键事实：他们的 Vulkan 在本机（Intel Arc）比他们自己的 CPU 还慢**
（tiny 133 vs ~104）——其调优全在 AMD RDNA（sg32l/coopmat/880M 报告），
Intel 无 coopmat 档缺乏竞争力；cls 上 GPU 后 11.5ms 反而比 CPU 慢。
我们的核显路径赢在每一段。small 侧我们的 rec 仍在 CPU（探针分级），
即便如此 2.2×——注意力头上 GPU 是纯余量。

（七）节数字保留作「vs 竞品 CPU」口径；对外对标应以本节为准。
