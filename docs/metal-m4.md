# qppocr — Apple Silicon Metal 后端实测（M4 真机）

实测机：**Mac（Apple M4 真机，10 核 CPU / 10 核 GPU，macOS 26，Darwin 25.3.0）**，
纯 Rust，`qppocr bench`。语料两套，各 100 张——**套内张张尺寸不同**（最不利
GPU 的形状多样场景），**两套之间逐张一一同尺寸**：

| 语料 | 用途 | 说明 |
|---|---|---|
| `bench/gpuhealth/corpus100/` | **性能**（本文所有 ms/图） | 竞品确定性生成器（`Seed=20260830`）的输出，与竞品 `dataset/` 同一批尺寸 |
| `bench/gpuhealth/gen100/` | **精度**（只有它配得上真值） | 同 seed 的**另一次**渲染 + `metadata.json`（逐行 `text` / `bbox` / `orientation_degrees`） |

两套的**图像像素尺寸 100/100 全同、内容 sha256 100/100 全不同**（验证：
`PIL.Image.size` 逐张相等，文件字节数逐张不等）。同尺寸正是它们能直接对拍性能
的前提；而内容不同意味着**拿 corpus100 去配 gen100 的真值就是拿错答案对卷子**，
所以**精度表只跑 gen100**。

构建：`cargo build --release -p qppocr-cli --features qppocr/gpu` → `target/release/qppocr`。
本文对应 commit `b6471e8`（`metal-backend` 分支）。**测前过 `bench/gpuhealth/health.sh`**
（判据是**实测带宽**，不是 `recoveryCount`——它不重启不清零）。

**结论：三档 Metal 全部快于本机 CPU——端到端中位 1.92× / 4.30× / 5.92×，
四路并发吞吐 2.00× / 2.76× / 4.82×；检出框与 CPU 逐图完全一致（300 图 0 差），
空文本行全档为 0，进程内存不高于 CPU，识别精度与 CPU 在真值上同分**
（small 995 = 995，medium Metal 反而多对 1 行）。

**两条必须一起说的反向结论**：
① **模型越大 det 越钝**（medium 的 rec 前向快 6.3×，而 det 只快 3.1×）；
② **跟竞品比精度，原样我们三档全胜、去掉空格三档全输**——那 6~8pp 全是
`rec_space_gap` 补出来的空格，**不是识别能力**（§3.2）。

> ⚠️ 本文所有数字都在**这一台**机器上、同一天、`--workers 1`（除吞吐表）测出。
> 这台机器的分钟级频率漂移是 20~40%，跨天比数字一律无效——`bench/gpuhealth/`
> 下的 A/B 工具因此内建了漂移门禁。**别把本文的毫秒数当跨机常数。**

---

## 一、端到端

### 1.1 单图串行（引擎计时口径，越低越好）

`qppocr bench bench/gpuhealth/corpus100/img-*.jpg --tier T --sweep device=cpu,metal --rounds 3`
——同进程跑完整语料、1 轮 warmup 丢弃、CPU 与 Metal **逐轮交错**、各取中位。
`total` 不含图像解码（`wall` 含）。

| 模型 | CPU total | Metal total | 比值 | CPU wall | Metal wall | CPU img/s | Metal img/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny   | 61.9  | **32.3**  | **1.92×** | 64.3  | 34.6  | 15.5 | 28.9 |
| small  | 288.1 | **67.1**  | **4.30×** | 290.7 | 69.3  | 3.4  | 14.4 |
| medium | 2017.8| **340.7** | **5.92×** | 2021.9| 344.7 | 0.5  | 2.9  |

这里的 CPU 是**单图串行**（引擎内部多线程铺满 10 核），不是「一个图一个线程」
的批量口径——与下表的 4 worker 直接比会低估 CPU。

### 1.2 四路并发吞吐（对齐上游 `--workers 4` 口径）

`--workers N` = **N 个子进程**各持一份引擎、各分 `总线程数 / N` 线程，连续切块。
同一语料各跑两轮取好的一轮：

| 模型 | CPU | Metal | 比值 |
|---|---:|---:|---:|
| tiny   | 38.5   | **19.3**  | **2.00×** |
| small  | 163.8  | **59.3**  | **2.76×** |
| medium | 1555.5 | **322.5** | **4.82×** |

两条口径给的比值不一样，这是**真实差异不是矛盾**：四路并发把 CPU 侧的
空闲核吃掉了（small CPU 288.1 → 163.8，1.76×），而 Metal 侧本来就接近
打满（67.1 → 59.3，1.13×）。**要报「快多少倍」必须先说并发度。**

**并发度不是越多越好**（这台 10 核机上，`--workers N` 是拿总线程数去除 N 的）。
100 图整语料实测，`--workers 2` 在**每一档**都输给 `--workers 4`：
tiny 50.0 vs **38.5**、small 226.8 vs **163.8**、medium 1656~1866 vs **1555.5** ms/图。
4 是这里的最优点。

> 一个踩过的坑：先在前 25 张（`img-001..025`）上扫，结论是 medium CPU
> `--workers 2` 最好（1002 vs `--workers 4` 的 1177）。**子集结论与整语料相反**
> ——25 张的形状多样性远低于 100 张，冷启动摊薄方式完全不同。
> **并发度扫描必须在整语料上做**，子集只配当「没崩」的冒烟。

---

## 二、分阶段耗时（gen100，100 图，每图一次 run 的均值，ms/图）

`crates/qppocr-core` 的 `Timings` 逐字段聚合（`bench/gpuhealth/stage_table.py`）。
本表如实留了三个 0：`decode_ms`（图像解码）、`rec_pre` / `rec_post`。
`rec_pre` / `rec_post` 的工作实际发生在 det 的裁剪与 rec 的 CTC 出口里，
不走这两个计时器；`decode_ms` 在这条路径上**根本没有计时**（六份 JSON
里 600 张图的 `decode_ms` 全是 0.0，不是「解码很快」）——所以**这张表是引擎
内部的相对成本，不是墙钟分解**。含解码的总墙钟看 §1.1 的 `wall` 列，
两设备都约 +2.3 ms/图（解码是两侧共用的 CPU 侧工作）。

| 阶段 | tiny CPU | tiny Metal | small CPU | small Metal | medium CPU | medium Metal |
|---|---:|---:|---:|---:|---:|---:|
| det 预处理 | 0.6 | 0.6 | 0.6 | 0.6 | 0.7 | 0.7 |
| det 前向 | 21.0 | **11.1** | 46.4 | **17.6** | 359.3 | **114.5** |
| det 后处理 | 1.1 | 1.1 | 1.2 | 1.1 | 1.6 | 1.2 |
| 裁剪 | 0.6 | 0.6 | 0.7 | 0.6 | 0.9 | 0.7 |
| cls | 4.8 | **3.1** | 4.9 | **3.0** | 6.4 | **4.4** |
| rec 前向 | 30.1 | **14.1** | 198.7 | **41.4** | 1227.2 | **195.5** |
| rec 预处理 / 后处理 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **引擎合计** | 59.6 | **31.5** | 255.2 | **65.5** | 1599.5 | **318.4** |

Metal 相对 CPU 的倍率：

| 模型 | det 前向 | cls | rec 前向 | 合计 |
|---|---:|---:|---:|---:|
| tiny | 1.9× | 1.5× | 2.1× | 1.9× |
| small | 2.6× | 1.6× | 4.8× | 3.9× |
| medium | 3.1× | 1.5× | 6.3× | 5.0× |

**`det` 是这条线里最钝的一段**：模型越大越明显（medium det 只快 3.1×，而
rec 快 6.3×）。原因见 §六：「我们现在能拿到的 det 天花板」不是 GPU 峰值。

---

## 三、精度（gen100 真值，100 图 / 1022 GT 行，IoU≥0.5 贪心一对一）

**报精度必须先声明空格算不算。** 字典 `models/dict_small_medium.txt` 里
**根本没有空格字符**，CTC 吐不出来；输出里的空格是
`crates/qppocr-core/src/pipeline/rec.rs` 的 `add_gap_spaces()` 按几何补的启发式
（相邻字符在裁切图上空出 ≥ `rec_space_gap · crop_h` 就插一个）。同一份代码，
真值约定含不含中英边界空格会给出**差 6 个百分点**的结论。

| 模型 | 口径 | CPU | Metal |
|---|---|---:|---:|
| tiny | 原样 | 891/1011 = 88.13% | 889/1011 = 87.93% |
| tiny | **去空格** | 995/1011 = 98.42% | 997/1011 = 98.62% |
| small | 原样 | 995/1016 = 97.93% | 995/1016 = 97.93% |
| small | **去空格** | 1012/1016 = 99.61% | 1011/1016 = 99.51% |
| medium | 原样 | 1014/1017 = 99.71% | 1015/1017 = 99.80% |
| medium | **去空格** | 1015/1017 = 99.80% | 1015/1017 = 99.80% |

CER：tiny 0.76% / 0.77%、small 0.30% / 0.32%、medium 0.21% / 0.20%
（去空格口径：tiny 0.33% / 0.32%、small 0.24% / 0.25%、medium 0.21% / 0.21%）。

有错的图：tiny 73/73、small 17/16、medium 3/2（CPU/Metal）。

三条与后端无关的硬结论：

- **检出框逐图完全一致**：1034 / 1020 / 1020，300 张图 0 差。
- **空文本行全档为 0**（CPU 与 Metal 都是）——曾经约 2.7% 的图会有一行
  `text="" conf=0.0`，根因是账本拿 `Arc::as_ptr` 当身份（地址被回收再分配
  ⇒ 假命中 ⇒ 跳过写参数），已修。
- **Metal 与 CPU 在真值上同分**。

### 3.1 Metal vs CPU 的逐图文本差（同真值语料，同轮）

| 模型 | det 逐图不同 | 行数/框不同 | 实质不同的行 | 其中的编辑距离 | 只差空白的行 | 受影响图 | 旋转角不同 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny | **0** | **0** | 10 | 19 | 29 | 30/100 | 0 |
| small | **0** | **0** | 2 | 5 | 7 | 7/100 | 0 |
| medium | **0** | **0** | 2 | 8 | 1 | 3/100 | 0 |

「只差空白」那列每行恰好差一个空格（编辑距离 = 行数），它们是
`rec_space_gap` 的边界启发式，**与后端无关**；真正属于浮点次序噪声的是
「实质不同」那一小撮——tiny 10 行共 19 个字符、small 2 行共 5 个、
medium 2 行共 8 个。

**没有一行「多了/少了」**；差异全在字符级，两个方向都有（Metal 也会比 CPU 多认
出一个字），且几乎全是近形替换。逐条（`CPU ↔ Metal`）：

    img-001  conf 0.96  OpenVINOo dynamic width   ↔  OpenVINO dynamic width    （O/o）
    img-001  conf 0.83  这是一                     ↔  这是
    img-073  conf 0.89  OpenVINO                   ↔  OpenVINo
    img-083  conf 1.00  方向分类 180               ↔  方向分类 180度            （Metal 多）
    img-066  conf 0.97  PaddleOCR 官方结果          ↔  PadleOCR 官方结果         （dd/dl）
    img-078  conf 0.98  …2026-08–30…               ↔  …2026-08-30…             （en dash / 连字符）
    img-080  conf 0.85  foX                        ↔  fox                       （X/x）
    img-033  conf 0.51  井                          ↔  进                        （低置信行）
    img-068  conf 0.91  天气晴朗                    ↔  天气睛朗                   （晴/睛，small）
    img-052  conf 0.59  …a rnd uae z               ↔  …n n ua z                 （垃圾行，small）
    img-060  conf 0.39  浦务路西一                   ↔  浦务堂路各品                （垃圾行，medium）
    img-056  conf 0.79  度序与填充 点济质房度          ↔  宽度序与填充 点房济度贸       （垃圾行，medium）

可以看到两类：**conf ≥ 0.83 的近形替换**（字认对了、只是字形相近选错一个），
和 **conf ≤ 0.79 的垃圾行**——后者本来就是被压到 <8px、连 CPU 也读不对的行，
两边各自乱猜，差异不构成谁的损失。

判据是「去掉空白后是否相等」：**只差空白的那些是 `rec_space_gap` 的边界启发式**
（良性，且与后端无关），实质不同的那些是并行归约的浮点末位噪声落在近阈值行上。
这类改动「改前改后输出是否一致」不能比字符串——见 §七。

### 3.2 同一份真值上的**跨引擎**精度（对竞品）

上面两张表是「我们 vs 真值」和「Metal vs 我们自己的 CPU」。要回答「谁识别得准」，
得拿**同一份真值**喂两个引擎。这份对照在 `bench/gpuhealth/logs/acc_h2h/SUMMARY.md`，
口径：gen100 真值 + **文本多重集**打分（竞品 harness 的 `texts` 不带框），
模型是同一批（竞品 `models/` 是指向我们 `models/` 的符号链接）。

| 模型 | 口径 | 我们 | 竞品 | Δ |
|---|---|---:|---:|---:|
| tiny | 原样 | **880/1022 = 86.11%** | 798/1022 = 78.08% | **+8.02pp** |
| tiny | **去空格** | 988/1022 = 96.67% | 993/1022 = **97.16%** | −0.49pp |
| small | 原样 | **987/1022 = 96.58%** | 925/1022 = 90.51% | **+6.07pp** |
| small | **去空格** | 1004/1022 = 98.24% | 1012/1022 = **99.02%** | −0.78pp |
| medium | 原样 | **1014/1022 = 99.22%** | 1007/1022 = 98.53% | +0.68pp |
| medium | **去空格** | 1014/1022 = 99.22% | 1019/1022 = **99.71%** | −0.49pp |

**原样看我们三档全胜，去掉空格三档全输。那 6~8pp 全是空格，不是字。**
逐行归属（small）：两边都对 912 / 只有我们对 75 / 只有竞品对 13 / 两边都错 22；
我们赢的 **75 行全是中英、数字边界带空格的行**，竞品赢的全是长英文行（各差 1 行）。

⇒ **绝不能报「识别能力领先 6 个百分点」。** 去掉空格那 0.5~0.8pp 与我们
Metal-vs-CPU 的浮点次序噪声同量级，不构成识别能力的差别。

> ⚠️ 两个诚实的边界：① 这张表的 small 行是 **cls 第二意见落地之前**的读数；
> 那 −0.78pp 的来源已定位（cls 硬门漏翻，180° 裁切原样进 rec）并已修，
> 修完后 small 去空格追平——机制见 §六。② **这一场语料里空格真的画了**
> （由 `bbox` 反推 ink 宽 ÷ 同字体 PIL 渲染：有空格 **1.0132**、无空格 1.0771，
> 11/11 命中），所以补空格是赚；真值若不把中英边界空格算作内容，同一份代码
> 就变成扣分项。**别把它说成普适的质量优势。**

---

## 四、内存

### 4.1 进程峰值 RSS（corpus100 全部 100 张，`/usr/bin/time -l` 的 `ru_maxrss`）

| 模型 | CPU | Metal | Metal/CPU |
|---|---:|---:|---:|
| tiny   | 245.0 | **231.8** | 0.95× |
| small  | 404.3 | **314.2** | 0.78× |
| medium | 819.2 | **785.7** | 0.96× |

（上表取三轮交错里同一轮的数，三轮之间每格只差 ±1~3%。**同轮比较才是有效的**
——把 CPU 和 Metal 分两次跑，机器状态会变。）

**三条能说的**：

- **Metal 的 RSS 不高于 CPU**（tiny/medium 打平，small 低 22%），尽管它还要
  额外管 GPU 侧的资源。**device buffer 本来就不进进程 RSS**（走 IOSurface
  wire memory）。
- **RSS 与模型大小不成比例**：模型目录只有 6 / 30 / 132 MB（`du -sh models/*`），
  而 RSS 是 245 / 405 / 819 MB——差不多是「一个 ~200 MB 的固定底 + 模型大小的
  4~6 倍」。**别拿模型文件大小估内存占用。**
- **Metal 侧每个计划在宿主内存里还留一份静态区副本**：`MetalSession` 的
  `statics: Vec<(u32, Arc<Vec<u32>>)>`——`ensure_statics` 要靠它把静态区
  在每次派发前放回共享 arena（§六 那条挂死 bug 的修法）。这是**确定存在的
  一笔开销**（medium 的十几个形状计划静态段合起来约 500 MB），只是它在
  上面的总账里被与 CPU 打平这件事盖住了。

> ⚠️ **这张表的绝对值不能当常数用。** 同一批命令在跑过一次带宽测试之后
> 复测，读数整体掉到 **55 / 169 / 615 MB**（Metal）与 **25 / 81 / 305 MB**（CPU）
> ——又是一个稳定档，但低了 2~4×。原因未查明（最可能是模型 mmap 的
> file-backed 页在那时被回收了）。**所以：RSS 只报「本轮同一状态下的
> CPU vs Metal」，别把绝对值写进任何跨环境的对比**，尤其是 §七.3 那张
> 跨机内存表。

> **测量方法本身踩过一个坑**：`ps -o rss=` 每 20 ms 轮询取峰值，会比同一次
> 运行的 `ru_maxrss` 少 30~55%（cpu medium 138 vs 305 MB）——RSS 的峰值是
> 模型解析期一个很窄的尖峰，轮询采不到。**峰值一律用 `getrusage` 口径
> （`/usr/bin/time -l` 的 `ru_maxrss`），别用轮询。**

### 4.2 Metal 计划占用（单张图触发的全部形状计划）

| 模型 | 计划份数 | 整块合计 | 最大一份 |
|---|---:|---:|---:|
| tiny   | 10 | 250.2 MB | 137.8 MB |
| small  | 9  | 536.4 MB | 222.2 MB |
| medium | 10 | **1360.5 MB** | **536.5 MB**（det）|

「整块」= 该形状的静态段（权重/参数）+ 动态区。按角色给**字节预算**
（`crates/qppocr-metal/src/backend/mod.rs`）：det 形状逐图一次性、暖缓存重建便宜
→ **384 MB**；rec 桶形状每图复用 → **1536 MB**（全进、零驱逐）。超预算按 LRU
驱逐。`QPPOCR_GPU_PLAN_MB` 可覆写。

> 同一台机器上，medium det 单份 536.4 MB / 静态 62.0 MB 这个数在另一份账本
> （`bench/gpuhealth/logs/phase_global_optimum_20261011.md`）里逐数对上——
> 两份独立测量互为核对。

---

## 五、GPU 能力实测（**真机**，不是虚拟化）

`cargo test -p qppocr-metal --release -- --nocapture <name>`。

| 项 | 实测 | 探针 |
|---|---:|---|
| fp32 FMA 峰值（16 个 float4 累加器，64 寄存器） | **3578 GFLOPS** | `fma_peak_reference` |
| fp32 FMA 峰值（4 个 float4 累加器，16 寄存器） | 3433 GFLOPS | 同上 |
| 片上带宽（1 MB，读+写） | 308 GB/s | `mem_bandwidth_sweep` |
| 流式带宽（≥16 MB，读+写） | ~102 GB/s | 同上 |
| blit 拷贝（256 MB，`health.sh`） | 90.5~91.1 GB/s | `membw` |
| 缓冲分配 | 1 MB 0.005 ms … 320 MB 0.030 ms | `buffer_alloc_cost` |
| 空命令缓冲 commit+wait 往返 | 0.017 ms | `cmd_buffer_roundtrip` |
| 单条 dispatch（含逐条 barrier 与首次绑定） | 0.383 ms | `dispatch_marginal_cost` |
| dispatch **边际**成本 | **0.006 ms/条**（256 条内） | 同上 |

两条由此推出的工程结论，都是当初**先被假设、后被否证**的：

1. **FP32 峰值 ≈ 3.58 TFLOPS**，所以「离峰值还差 8.6×」这类说法多半是拿错了
   分母——我们某些内核孤立跑最好只有 ~1022 GFLOPS，**余量要拿内核自证的天花板
   当尺子，理论峰值只配当上界脚注**。
2. **dispatch 条数与 barrier 基本免费**：整条 180 条计划的调度开销只有 1.33 ms，
   `NO_BARRIER` 端到端 Δ = −0.10 ms。所以「瓶颈是 dispatch 太多」这条**作废**。
   一个内核慢，要在**访存路径 / 装载指令条数 / 循环体大小**里找，不在调度里。

---

## 六、为拿到这些数字做的改动（保留在树上的）

| 改动 | 机制 | 实测 |
|---|---|---|
| `n_conv8_mma` / `n_conv8_mma3` | `simdgroup_matrix` 直读 device 内存做 GEMM；泛化到 stride≠1 / 非对称 pad，覆盖 det 头部 4 条 conv | det 头部 conv 家族 **13.3 ms** |
| `n_conv_dw_tile` | depthwise 的瓦片暂存版（只收 5~9 方形 s1） | dw 家族 **17.27 → 11.77 ms** |
| **64 B 相位对齐** | 静态段与激活段两条分配路径都抬到 64 B（含 `alloc`/`free` 尺寸与起点）；起点字节 mod 64 曾让同形状同网格的内核差 **1.47×** | medium 端到端 **−4.6%**，再 **−2.1 ms/图** |
| `n_head_argmax` | rec 出口 `Softmax.2 + argmax` 并成一条 Metal 独有内核（8 条串行累加链拆 8 线程 ⇒ 加法次序不变） | small **−3.5 ms/图**，全语料**逐位不变** |
| rec 桶宽粒度 32 → 64 | 合批省的是**整整一个设备端地板**（一次 rec run ≈ 2.27 ms + 0.006·w） | small **−3 ms/图**（**要花精度买**：1041 行里 7 行变） |
| cls 合批上限 8 → 32 | 每图 2 个 run 合成 1 个 | 逐位不变 |
| **cls 第二意见**（`cls_retry_conf`，默认 0.85） | `cls` 存疑**且**首次解码置信度低时翻转 180° 重读，只在更自信时采纳 | small **+8 行**、CER 1.82%→**0.32%**、三档 `orientation=0°` **一行没动** |
| 共享 arena + 计划缓存 + 静态区重放 | 修掉「后建计划的静态区压掉先前计划的参数块」⇒ shader 读垃圾 ⇒ 巨轮询 ⇒ GPU 挂死/打瘸 | `fixcheck.sh` 5/5；打瘸后曾要重启才能恢复 |
| 账本改钉 `Arc` | `Arc::as_ptr` 是整数，所有者一 drop 地址就被复用 ⇒ 假命中 | 空文本行 0（互锁对照：不钉 61 轮撞上 / 钉住 450 轮干净）|

被**否证**而没上树的（写在这里免得后人重挖）：

- **fp16 权重/激活**：fp16 在 MMA 落地后从 1.79× 收益反转成 **1.14× 净亏**，
  fp16 MMA 探针也只到 1535 GFLOPS < 我们 f32 MMA 的 2204~3326 ⇒ **这台 M4 上
  「精度档位换性能」这条路关掉了**。代码已删，只留 `QPPOCR_GPU_*` 旋钮。
- **MMA 累加分片到 16**（`acc[4][4]`）：8 形状全负、中位 **0.331×**，数值逐位相同。
  判别式是**分片数**：8 片的都赢，16 片的崩（每 lane 48 vs 28 个分片寄存器）。
- **dw 的 9×9 可分离**（9×1 + 1×9）：不是难做而是**前提不成立**——要求核矩阵秩 1，
  实测 σ2/σ1 = **0.233~0.998**（`onnx_dw_sep.py`）。
- **放大循环体**：这台 M4 上凡让热点循环体变大的结构改动**五连败**
  （0.59× / 0.72× / 0.75× / 0.84× / 0.86~0.91×）。
- **工程侧**（`cls/img` 分列、rec 分桶、建计划、host 开销）：2026-10-12 逐条量死，
  整条 180 条计划的调度开销只有 1.33 ms。

---

## 七、已知限制 / 未闭合

1. **这台机器的漂移是 20~40%**：`bench` 靠同进程多轮交错取中位抵抗，A/B 靠
   `bench/gpuhealth/env_ab.sh` 的**漂移门禁**（按 A 臂自身离散分类轮次，
   绝**不**按 Δ 筛）。本文的绝对毫秒数请当**同一台机器同一天**的相对量看。
2. **Metal 的 rec 输出逐次运行本来就不确定**：同一份二进制连跑两遍，10 图里
   7 图**文本**不同（近形替换、两个方向都有）、**框逐字相同**。所以「改动前后
   输出是否一致」不能比字符串，判据只能是行数/框数/unread + 去空格后的实质差。
   tiny/small 跨 run 是 0 差（可要求逐位一致），medium 不是。
3. **跨机对照没有闭合，而且这台机器上大概率闭不了**。上游公开的 M4
   paravirt VM 数字（`docs/metal-m4.md`，他们仓库，自报 `--workers 4`）：
   tiny/small/medium Metal **39.3 / 54.6 / 107.0** ms、CPU **68.8 / 218.4 / 693.0** ms。
   我们真机对应（同为 `--workers 4`）**19.3 / 59.3 / 322.5** ms 与
   **38.5 / 163.8 / 1555.5** ms。⇒ tiny 快 2.0×、small 慢 1.09×、
   **medium 慢 3.0×**；medium 的 CPU 我们反而慢 2.2×。

   **已有三处可核的口径不一致，任一处都足以解释这个量级的缺口，所以这张跨机表
   不该被当成结论：**
   - **CPU 侧基线不同**：他们是 8 vCPU 上的 .NET `sharp` 引擎，我们是 10 核上的
     纯 Rust；`--workers 4` 在两种核数下分到的线程数不同。
   - **det 的规模行为完全不同**：他们的 `det_graph` 从 tiny→small 几乎不动
     （13.0 → 13.6 ms），我们 11.1 → 17.6 ms；而 medium 他们 33.1 ms、我们
     114.5 ms。**同一批 100 张图（逐张像素尺寸完全相同）**下，两种 harness 的
     det 对不同档位的伸缩方式不一致 ⇒ 至少有一边在 det 上跑的不是同一件事
     （输入分辨率上限 / 档位与 det 模型的对应关系不同）。
   - **内存那一列不是同一种米**：他们的 "peak WS" 是 .NET 进程的 working set
     （688/955/1709 MB），我们的是纯 Rust 的 `ru_maxrss`（§四.1，百 MB 量级）。
     差的 2~3 倍里混着**计量口径、运行时（.NET vs Rust）和我们自己那个未查明的
     状态依赖**——不是内存占用本身的差别。

   **记为未闭合，不判虚标。** 想真判，得让两边跑同一份可核的输入分辨率配置 +
   同一种内存计量——不在本轮范围内。
4. **det 是 Metal 线上最钝的一段**（medium 只快 3.1×）。它已经贴着**内核自证
   的天花板**（纯访存那批 84~101 GB/s 贴 95 屋顶、MMA 家族贴自己上限），
   这不是「还没优化」，而是「按现在的结构就到这里」。
5. **`medium` 的 CPU 吞吐最差**（4 worker 1555 ms/图）。§1.2 的并发度扫描说明
   这不是并发度没调对（`--workers 2` 比 4 更差），是模型本身在 CPU 上贵。
6. **`rec_space_gap` 是按图源调出来的启发式，不是普适优势**：本文这一场语料
   里空格**真画了**（由 `bbox` 反推渲染 ink 宽 ÷ 同字体 PIL 渲染：有空格
   **1.0132**、无空格 1.0771，11/11 命中）所以补空格是赚；真值约定不含中英
   边界空格时，同一份代码会变扣分项。
7. **不是「Metal 总是更快」**：`--workers 1` 且形状极度多样、模型又小时，
   首次遇到新形状要一次性建执行计划，冷进程首图会比 CPU 慢（后续同形状零构建）。

---

## 八、复现

**以下命令都在仓库根目录执行**（`bench/gpuhealth/*.sh` 会自己 `cd` 回根目录）。

```bash
cargo build --release -p qppocr-cli --features qppocr/gpu

# 0) 先过健康门禁——**任何 GPU 结论之前都必须过**，判据是实测带宽
./bench/gpuhealth/health.sh

# 1) 端到端（同进程、CPU/Metal 逐轮交错、取中位）
./target/release/qppocr bench bench/gpuhealth/corpus100/img-*.jpg \
    --models models --tier small --sweep device=cpu,metal --rounds 3

# 2) 吞吐（四路并发；注意 `bench` 子命令是串行的、**忽略 --workers**，
#    所以这一步走普通路径，逐档把 small 换成 tiny|medium）
./target/release/qppocr bench/gpuhealth/corpus100/img-*.jpg \
    --models models --tier small --device metal --workers 4 --quiet

# 3) 一次 run 同时拿分阶段计时与逐行输出（§二 表 + §三 精度都吃它）
./target/release/qppocr bench/gpuhealth/gen100/img-*.jpg \
    --models models --tier small --device metal --workers 1 --quiet --json \
    > bench/gpuhealth/logs/deliverable/metal-small.json

# 4) 分阶段表 / 精度（原样与去空格两种口径都要报）
./bench/gpuhealth/stage_table.py \
    CPU=bench/gpuhealth/logs/deliverable/cpu-small.json \
    Metal=bench/gpuhealth/logs/deliverable/metal-small.json
./bench/gpuhealth/acc_truth.py --tier small --device metal \
    --from-json bench/gpuhealth/logs/deliverable/metal-small.json
./bench/gpuhealth/acc_truth.py --tier small --device metal \
    --from-json bench/gpuhealth/logs/deliverable/metal-small.json --strip-space

# 5) 内存峰值（**必须用 ru_maxrss，别用 ps 轮询**，见 §四.1）
/usr/bin/time -l ./target/release/qppocr bench/gpuhealth/corpus100/img-*.jpg \
    --models models --tier medium --device metal --workers 1 --quiet 2>&1 >/dev/null \
  | awk '/maximum resident set size/{printf "%.1f MB\n", $1/1048576}'

# 6) GPU 能力微基准
cargo test -p qppocr-metal --release -- --nocapture --test-threads=1 \
    fma_peak_reference mem_bandwidth_sweep cmd_buffer_roundtrip dispatch_marginal_cost
```

本轮六份 JSON（`cpu|metal` × `tiny|small|medium`）落在
`bench/gpuhealth/logs/deliverable/`，与本文各表逐数对应；
精度对拍与成本账本在 `bench/gpuhealth/logs/acc_h2h/`、`logs/cls_retry/`。

**引用本文时的最小声明**：机器 = 这台 M4 真机（**真机，不是 VM**）；
并发度 = 1 或 4（表上标了）；语料 = 性能用 corpus100、精度用 gen100；
**精度口径 = 空格算不算**；跨机数字（§七.3）不可当结论。
