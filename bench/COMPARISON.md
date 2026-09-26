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
./target/release/qppocr.exe <img> --tier tiny --json --quiet --bench N --workers 1
```
- `--json` 里 `total_ms` = **best_ms**（N 次引擎墙钟取最优），分项也取自最快那次。
- `--rec-shards N` 覆盖 rec 分片数。

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
- 每次实验前先跑 `scaling_probe`（1% 以内才开工）。

机器特性（实测，探针在 `crates/qppocr-kernels/examples/scaling_probe.rs`）：
| 项 | 值 |
|---|---|
| 纯计算扩展 | 1/2/4/8/16 线程 = 1.00/1.82/2.72/4.26/**5.05x** |
| 内存带宽 | 1/2/4/8/16 线程 = 29.6/50.1/45.2/37.7/**26.1 GB/s**（峰值在 2 线程） |
| fork 协调 | 2 线程 **0.6 µs**、4 线程 2.2、8 线程 12.1、16 线程 26.1 µs |
| 单核 FMA | 纯寄存器探针 54.5 GFLOPS；而 sgemm 实测 **82–97 GFLOPS** |

## 5. 当前位置（`bench/ab.py vs`，4 轮交错中位）

| | 比值 |
|---|---|
| **tiny** | **1.49**（各轮 1.47/1.52/1.48/1.50） |
| **small** | **1.28**（各轮 1.29/1.28/1.25/1.31） |

tiny 分阶段：det 41 ms vs 24（1.7x）、行阶段 37 vs ~22（1.7x）。
small 分阶段：det 74 vs 67（1.1x，基本持平）、行阶段 191 vs 116（1.65x）。

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
| **两级并行 / 多池分片** | 0.998x，无差别 | 实现完整（`rec_shards`，tiny 12 / small 4），默认开着也无害，但**没有收益**。基础设施（多池）保留备用。 |
| **并行门槛调整**（fork_min_macs 4M→8M/16M/32M） | 更慢（27.40/27.68/29.15/29.88） | 默认 4M 已是最优。 |
| cls 批化（cls_batch 1→2/4/8/16） | 更慢（4.12→11.95 ms/图） | |
| rec_batch > 1 | 逐行扇出下批起来只降并行度 | |
| 线程局部小块缓冲缓存 | 慢 2.7% | 全局池锁不是瓶颈。 |
| 拿 `ocr-demo` 当对标 | 口径错 | 见 §1。 |

## 8. 未完成 / 下一步候选

按估计收益排（**都没验证过，别当结论**）：

1. **更多节点融合**（对方有 9 类，我们 4 类）：
   - `Conv + Bias + ResidualAdd`（det 里 5 个双分支合并 + 3 个 FPN 的 `Resize+Add`）
   - `ConvTranspose + Bias + Activation`
   - `Conv + HardSwish`、LayerNorm
   估 det 的 3–5%。
2. **det 里扩展 <3x 的算子**（占 41% 时间）：MaxPool / Concat / 深度卷积。
   但对方同样慢（MaxPool 2.1 vs 我们 2.7），**不完全是差距来源**。
3. **区域重试触发率**：我们 9/100、对方 7/100，触发一次该图 det 翻倍。
   ⚠ 但**测试时用的是关掉重试的配置**（`Preset::Speed`），这是应用层功能，别当性能问题。

## 9. 仓库状态

- 分支 `internal-main`（**含内部移植叙述，绝不推送**）；
  公开快照是孤儿分支 `release-vX.Y.Z`（单提交、无父）+ tag。
- 当前 tag `v0.1.2`，快照见 `git log --oneline -1 release-v0.1.2`。
- **推送被网络阻断**（github.com:443 连不上），提交都在本地。
- 全程纪律：**每次改动都必须过 100 图逐字符对拍**（`dump_texts` 输出与
  `$TEMP/before.txt` diff）。
