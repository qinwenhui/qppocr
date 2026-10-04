# Changelog

本项目的全部显著变化都记录在这个文件里。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。
每个版本写清**行为变化**，尤其是默认值变化。

## [Unreleased]

### 变更（Changed）

- **GPU 路径不再按单机性能数据默认降级**：0.3.0 曾把注意力结构的识别
  模型（small 档 rec）默认留在 CPU（依据开发机上的相对性能实测）。
  显式选择 `--device gpu` 现在意味着三段（det/cls/rec）全部上 GPU——
  性能权衡随硬件差异太大，应由使用者判断；`QPPOCR_GPU_STAGES` 分级
  与算子覆盖面探测（不支持时声明并退回 CPU）不变。影响：small 档
  在 GPU 上的识别输出与 CPU 存在浮点末位噪声导致的个别行差异
  （评测语料 100 图约 12 行文本不同，以方向两可的近似字符翻转为
  主）。

## [0.3.0] - 2026-10-01

**GPU 版本**：新增可选的 Vulkan 计算后端（det/cls/rec 全图上 GPU），
aarch64 升级为 NEON 向量执行，CLI 新增 `bench` 基准子命令。CPU 路径
行为不变（100 图逐字符对拍与 0.2.1 一致）。

### 新增（Added）

- **GPU 后端（`qppocr-gpu` crate，feature `gpu`）**：纯 Rust（ash 绑定）
  的 Vulkan 计算后端，不引入 C++ 工具链——GLSL 着色器编译为 SPIR-V 后
  签入仓库（`tools/shader-build` 为独立编译工具），构建零着色器依赖。
  - 执行模型：装载期完成形状推理与常量折叠、算子到内核的映射、显存
    区域布局与权重重排，整图 dispatch 序列录进一条可复用命令缓冲；
    运行时一次提交 + 等信号，无逐算子往返。
  - 内核覆盖 conv 全家族（1x1/3x3/5x5、分组、深度、ConvTranspose，
    bias/激活/残差收尾）、池化、resize、通道拼接、SE 门融合、
    LayerNorm 分解链、softmax、注意力所需的批量矩阵乘与通用转置、
    CTC argmax 出口（每时间步只读回 8 字节/行）。
  - 批处理：同形状行合批 + 批维补齐，空行由内核按真实行数早退（零
    算力）；识别行按宽度分桶复用计划；`run_deferred` 支持批间流水
    （提交下一批与 GPU 执行当前批重叠）。
  - 三级缓存控制重建开销：管线缓存（驱动复用编译产物）、权重重排
    缓存、显存块池；计划缓存按字节预算 LRU。
  - 精度：GPU 与 CPU 路径在评测语料上逐字符一致（100 图对拍）；算术
    内核与 CPU 参考逐位对齐（exp/erf 多项式逐字镜像、FMA 链一致）。
  - 分级部署 `QPPOCR_GPU_STAGES=all|detrec|det`；会话在装载期探测算子
    覆盖面，不支持的模型显式声明并留在 CPU，不静默降级。**small 档
    识别模型（SVTR 注意力结构）默认留在 CPU**：该结构对批处理的宽度
    填充敏感，按自然宽度执行才保精度，而逐行形状的代价目前大于收益
    （`QPPOCR_GPU_FORCE_REC=1` 可强制上 GPU）。
  - 开发与验证在 Intel Arc 核显（Vulkan 1.4）完成；其它 GPU 未经测试。
    发现并适配了一个驱动侧的缓存一致性问题（HOST_COHERENT 映射读回
    需显式 clflush；已按架构 gate，不影响其它平台语义）。
- **设备抽象（`qppocr-core::device`）**：`DeviceContext` / `DeviceSession`
  trait 把设备边界定在 Session 层；门面新增 `DeviceChoice` 与
  `EngineBuilder::device()`；CLI 新增 `--device cpu|gpu|vulkan[:N]|cuda[:N]`
  （cuda 为枚举级预留，计算后端尚未实现）。`qppocr-core::Error` 新增
  `Device` 变体（下游 exhaustive match 需跟进）。
- **aarch64 NEON 后端**：sgemm 面板（含 implicit-GEMM 指针面板）、窄 N
  路径、深度卷积、ConvTranspose、激活（GELU/erf/exp/ReLU/clip/
  HardSigmoid/sigmoid）、softmax、二元算子、2x2 池化、双线性缩放、
  2x2 膨胀——全部与标量判据逐位一致。ARM 设备从纯标量回退升级为向量
  执行；f32 NEON 是 aarch64 基线指令集，无需运行时探测。`Backend` 枚举
  新增 `Neon` 档。CI 新增 aarch64-linux 交叉编译 + QEMU 测试 job。
- **CLI `bench` 子命令**：测量纪律内建——同进程跑完整语料、warmup
  丢弃、多配置逐轮交错取中位，报告内核后端与生效配置；`--sweep` 对
  自己的图集扫 preset / rec-height / threads / rec-shards / device。

### 修复（Fixed）

- **区域重试自 0.2.0 起实际默认开启**（与 0.2.0 发布说明相反）：当时的
  「默认关闭」只改了门面默认值，漏改 core 的 `PipelineConfig::default()`，
  默认配置仍在约 9/100 张图上隐式跑双倍 det。现默认真正关闭；Accuracy
  预设与显式 `Advanced::retry_conf` 的开启路径不受影响。已补回归断言
  锁住「文档声明的默认值」。
- **线程池嵌套并行死锁**：并行区内的 `fork_join` 改为就地串行执行，
  消灭「并行区内再次并行」争锁的整类挂起（疑似即 0.2.0 已知问题中
  medium 多进程批量偶发停滞的根因）。触发时每线程提示一次。

### 变更（Changed）

- 内核分发层重构为单一入口（`qppocr-kernels::arch`）：架构接线集中到
  一个模块，内核文件不再含架构条件；标量参考实现抽成命名函数。
  x86-64 输出逐位不变。
- implicit-GEMM（卷积免 im2col 物化）不再限定特定架构：无向量后端的
  目标同样走指针面板。

## [0.2.0] - 2026-09-27

**性能大版本**：默认档逐行精确 tiny 83.3% / small 94.9% 保持不变；
单图端到端（100 图语料、同进程热态）tiny ~62 ms / small ~250 ms，
持续负载下的长跑稳定性显著优于短跑（频率漂移敏感度 ~7%）。

### 已知问题

- **medium 档 + CLI `--workers 4/8`（多进程批量）偶发挂起**（100 图集上
  有报告：子进程 CPU 停滞；`--workers 1` 正常）。本地复测 medium w8 全集
  通过（100/100），未复现；疑似特定环境下的
  嵌套并行死锁（`pool.rs` 明确禁止嵌套 `fork_join`），继续排查。
  临时规避：medium 用 `--workers 1`。
- 本版开发中曾引入一个 medium 图损坏（残差折叠把拓扑在后的张量折进
  卷积成环，子进程报 unresolvable nodes/停滞）——已用「R 的生产者必须在
  拓扑在前」的保守条件修复，medium w1/w8 全集验证通过。

### 破坏性变更（BREAKING）

- **区域重试默认关闭**（`retry_conf` 0.85 → 0）。实测在评测语料上开关
  重试的逐行精度**完全一致**，而重试让 9/100 张图的 det 翻倍。要开用
  `--preset accuracy` 或 `Advanced::retry_conf`。低置信区域图的输出
  可能与旧版不同。
- CLI 的 `--rec-shards` 语义修正：显式 `0` 现在真的是"关"（曾被 `> 0`
  守卫吞掉）；不传 = 按档位自动（tiny 12 / small 4）。

### 性能

- **Conv+ResidualAdd 残差融合**：`Add(conv_out, R)` 折进卷积收尾（激活
  之后逐元素加，与原 Add 同值同序——逐位不变）；1x1 GEMM 路径折进
  sgemm 的收尾趟，省掉独立 Add 的整趟冷读冷写与中间张量。
- **Conv→BatchNormalization 折叠**：BN 仿射直接乘进卷积权重与 bias，
  rec 每图折掉 ~38 个节点、cls ~30 个。
- 检测输入归一化通道在外层（det_pre 2.35 → 1.52 ms）；DB 连通域免
  label 数组（2.6 → 1.45 ms）；MaxPool 2x2 s1 单趟化（无收益，留档）。

### 文档

- README 新增「档位与预设怎么选」：场景→配置对照、测量纪律
  （同进程跑全量、逐轮交错取中位）、`--rec-height` 红线（32 不可用）。

## [0.1.2] - 2026-09-26

单图性能补丁。同一台机器、同一份上游模型、同一份 100 图语料，
**每一项改动都验证过输出逐字符一致**。

### 算子层的账（本轮的重点）

单线程 `conv_bench`（真实 shape、9 轮取最好；本机 sgemm 的实测上限约
48 GMAC/s）：

| 算子 | 优化前 | 现在 |
|---|---|---|
| dw 160→160 k3x3 @25x30 | 2.79 | **8.90** GMAC/s |
| dw 64→64 k5x5 @200x240 | 9.50 | **18.15** |
| dw 96→96 k7x7 @50x60 | 4.39 | **17.35** |
| g1 64→16 k3x3 @200x240 | 19.39 | **32.75** |
| g1 32→16 k3x3 @400x480 s2 | 14.06 | **21.66** |
| g1 64→16 k1x1 @200x240 | 22.71 | **36.33** |
| g1 64→128 k1x1 @25x30 | — | **42.43**（88% 上限） |

三处瓶颈：**深度卷积**（逐 tap 读改写 + 边界列逐元素判断）、**im2col**
（逐元素越界判断，占 k3x3 卷积 50–65%，吞吐只有 6–10 GB/s）、**窄 N 面板**
（nn≤16 时一半 FMA 空转）。

**rec 的内核已经不是瓶颈**：small 档单行 38.8 ms 里，`192→384 k1x1 @6x268`
这类跑 40 GMAC/s（83% 上限）。剩下的**并行效率**是机器本身的墙：纯计算
探针（`scaling_probe`，N 个完全独立的线程各跑一份同规模 GEMM）实测本机
16 线程只有 **5.25x**，每线程吞吐从 40.9 掉到 13.4 GMAC/s。对照我们的
det 拿到 4.5x（已贴近上限），rec 2.5–3.2x——rec 的每行是一条串行链，
行间并行下墙钟就是最长那一行乘以降频系数。

### 本轮改动

| 改动 | 效果 |
|---|---|
| cls/rec **逐行扇出**（行内串行，靠池的执行模式开关） | -12.6%；批串行+批内并行只拿到 3.2x 扩展，逐行扇出拿满机器上限 4.6x |
| **图像缩放逐行并行**（det 输入准备） | 6.95 → 0.63 ms |
| **DB 2×2 膨胀 SIMD 化** | 3.81 → 0.29 ms |
| **凸包计算并行化** | per-box 2.36 → 1.31 ms |
| **裁剪阶段并行化** | 4.2 → 1.3 ms |
| rec_batch 6→16 | -3.6% |
| 执行器就绪调度改「入边计数 + 小顶堆」 | 结构改动，tiny/small 上实测中性；CLI 分项耗时改取最快那一次的 |
| **深度卷积重写**（补零平面 + 寄存器累加） | det -11%；单线程最差档 2.8→8.9 GMAC/s |
| **im2col 内层定区间 + memcpy** | k3x3 卷积单线程 1.5–1.7x；im2col 单项 11.1→2.9 ms |
| **窄面板微内核**（nn≤16 不再空转 FMA） | 窄 1x1 卷积 22.7→36.3 GMAC/s |
| rec_batch 16→1 | rec 阶段 -10%（逐行扇出下批起来只会降并行度） |
| **两级并行**（多池分片，tiny 12 片 / small·medium 4 片） | 行阶段 tiny 1.26x、small 1.33x |
| **Conv→ReLU 融合**（之前只融了 GELU） | det 搬运 -5.3% |
| **门控块坍缩**（Add(Mul,·) → 一趟三读一写） | det 搬运再 -2.6% |
| **implicit GEMM**（卷积不再摊 patch 矩阵） | det 搬运再 **-30%**；s2 的 k3x3 单线程 -11~15% |
| **深度卷积补零平面只清边距** | s2 深度卷积 16 线程 2.09 → 1.04 ms；det -2.1% |

新增 `par::set_serial_execution`（池的执行模式开关：只改执行方式、不改
切分决策，`threads()` 仍报真实池大小 ⇒ 与并行路径位级一致）、
`geometry::par_map`、`kernels::resize::{resize_bilinear_rgb_u8,
dilate2x2_max}`；诊断 example 增加 `dump_texts` / `resize_bench` /
`det_input_dump` / `cls_batch_sweep` / `im2col_split`（同一 shape 分别只跑
im2col、只跑切 tile 的 sgemm、完整 conv2d，用来判断该动哪边）。

**修复**：`Advanced::default()` 的 `rec_batch` 是 6，与
`PipelineConfig::default()` 的 16 不一致——显式传 `Advanced::default()`
的人会悄悄退回旧默认。CLI 的九项分项耗时此前取自**最后一次**迭代而总时间
取自最快那次，导致分项加不出总数。

**已证伪的方向**（都量过，记下来免得再走）：

- cls 批大小调大只会更慢：16 图 ms/图 1→4.12、2→4.76、4→6.09、8→10.09、
  16→11.95。单行 cls 前向本来就只有 ~1.5 ms，省下的固定开销盖不过批间
  并行度的损失。（翻转结论在所有档位完全一致。）
- **rec 的并行只能切成「行间」，不能切成「行内」**。池不支持嵌套 fork，
  所以两种切法只能二选一：一批一个线程、批内串行（现状）vs 批次顺序
  执行、算子内部 fork 铺满池。100 图逐图交错实测（rec 阶段中位 ms）：

  | 方案 | tiny | small |
  |---|---|---|
  | 行间并行（现状） | **40.2** | **221.8** |
  | 行内并行，每批 1 行 | 66.3 | 262.7 |
  | 行内并行，每批 4 行 | 57.3 | 234.1 |
  | 行内并行，每批 8 行 | 58.1 | 237.7 |
  | 行内并行，按宽度聚批 | 58.5 | 238.7 |

  rec 每行 ~80 个算子**顺序依赖**，每次 fork ~97 us 的协调开销乘上去
  就不划算；行间并行的墙钟是「最长那一行」，但行内并行连这个都追不上。
  同理，"按行数在线程数两侧自适应切换"这个方向也被否掉了。
- **（已推翻，见下）两级并行曾经被判定「不划算」——那是两个自己埋的 bug
  造成的假象。** 教训留在下面，因为两次都是「结论先于验证」：
  1. `par::threads()` 会顺手按**默认布局**把池建起来，而 engine 里是「先取
     线程数、再请求布局」，于是布局请求永远是 no-op——8 个分片线程全挤在
     0 号池上被 `fork_mu` 串行化（逐算子耗时翻倍）。加 `par::planned_threads()`
     （只算不建）修掉。
  2. 修完之后，`auto_rec_shards` 又被写成无条件 `return 0`，**连显式的
     `--rec-shards N` 也被吞掉**——连续几轮扫描全程都在测「不分片」，那些
     ±3% 的「无差别」当然是噪声。
  验证手段：`QPPOCR_FORK_DEBUG=1` 打印每次 fork 的池槽与参与者数。分片真正
  生效时能看到 `slot=1..N, p.threads=m`；没生效就全是 `slot=0, p.threads=16`。
  **改这类默认值时必须确认覆盖路径还通。**
- 全局缓冲池锁不是并发瓶颈：线程局部小块缓存实测**慢 2.7%**，已回退。
- rec 的并发膨胀是硬件墙：纯计算探针实测本机 9 线程时每线程降频 2x，
  叠加内存争用后共 4x。

## [0.1.1] - 2026-09-26

生产事故驱动的稳定性补丁（识别输出与 0.1.0 逐位一致，无行为变化）：

- **修复 copy_strided 的 dst 越界 panic**：调用方传相对切片而内核用
  绝对 o 索引——单块路径（小张量）恰好掩住，大图 + medium 的多块并行
  必 panic（3918×2772 照片生产事故）。大张量转置逐元素对拍回归。
- **池的 panic 隔离**：worker/主侧 chunk panic 不再杀死线程或带着
  fork_mu 守卫展开（标毒）——屏障照常完成、原 panic 重抛给调用方、
  池毫发无损继续服务。毒化回归测试（panic 后池仍正确工作）。
- **EXIF Orientation 引擎侧自动应用**（JPEG，对齐 PaddleOCR 官方与
  浏览器显示）：竖拍照片的原始像素横躺，此前检测框与显示坐标系是两套
  （应用侧各自兜底）。零依赖手写 APP1/IFD0 解析，1-8 全方向转正；
  一张 3918×2772 的竖拍照片解码即 2772×3918。
- **修复 depthwise 快路径 `ox1` 的 usize 下溢 panic**：输入小于核宽
  （如 1×1 防盗链占位图遇 3×3 核 + 尾部 padding）时整批识别中断；
  越界列本就零贡献，饱和后落入既有跳过分支，含固定数字回归测试。
- **executor 错误传播**：34 处 unwrap/直接索引改为 `Err(Graph)`（带
  算子与输入名）——极端输入变成「单张失败 + 原因」，不再 panic。

## [0.1.0] - 2026-09-25

首个公开版本。

### 引擎

- 纯 Rust 的 PP-OCRv6 推理：手写 ONNX 读取器（opset 7/11/14）、图优化
  （常量折叠/Identity 消除/GELU 子图坍缩/bias 折叠/conv+act 融合）、
  带引用计数释放的 arena 执行器。
- 手写算子内核：AVX2+FMA 的 4×4 分块 GEMM、im2col/conv2d/convtranspose、
  广播二元、激活五件套、深度卷积、池化、双线性 resize、softmax 等；
  NEON/WASM 走同一套标量判据。`unsafe` 集中在 qppocr-kernels 一个
  crate，内核 crate 零第三方依赖。
- 自研 fork-join 线程池：批量唤醒、主线程参与、自旋 join、参与者计数
  屏障；并行阈值全部来自实测。
- det → cls → rec 完整流水线：DB 后处理（二值化/膨胀/连通域/旋转卡壳/
  unclip）、透视裁剪（8×8 单应）、凸包框合并、0/180 方向分类、按宽
  分批、CTC 解码、像素空格、弱行区域重试。36 个行为参数全部实测标定。
- 准确率：多语料 + 100 图独立合成数据集复核，tiny/small 文本输出
  逐字符稳定。
- 性能（16 逻辑核桌面机）：tiny 单图 ~80ms、small ~200ms；100 图批量
  5.9s（CLI 进程扇出）；内存峰值 tiny ~115MB / small ~181MB。

### 公开 API（qppocr 门面）

- `Engine::new(tier, dir)` 三行上手；`EngineBuilder`
  （tier/preset/config/advanced/threads/detect_orientation/
  verify_sha256）；`Config`（Option 字段 = 预设覆盖语义）+ `Preset`
  （Speed/Balanced/Accuracy）+ `Advanced`（29 项调参常数，不承诺稳定）。
- `TextLine`：文本/置信度/rotation/原图四角点 + **逐字坐标**
  `chars`（CTC 时间步映射，性能损耗 ≈ 0，含空格真实空白区间）+
  **区域重试标记** `retried`。
- `OcrResult`：九项分阶段计时、boxes/merged/flipped/unread 统计。
- `ModelSource::Dir/Bytes` + 上游官方 SHA-256 校验（手写 FIPS 180-4，
  NIST 向量测试；失败返回 Err 不 panic）。
- `serde` feature：结果/配置可序列化；`Engine` 为 `Send + Sync`。
- feature 门控（parallel/image-decode/serde/std），七组组合 CI 矩阵。

### CLI

- 参考命令行：单图/批量（`--workers` 进程扇出自动分图）、`--json`
  含九项阶段耗时、场景参数透传。

- 监控截图场景配方（`monitor_preset` example）：`unclip_perp=0.5` +
  `unclip_margin_thresh=0.45` + 关方向分类——白字压栏杆/栅栏纹理的
  时间戳行 0.44 → 0.94 完整读出。
- 诊断工具：`QPPOCR_DUMP_DIR` 按会话分文件、`QPPOCR_SAVE_CROPS`
  落盘裁剪与方向分类输入、`QPPOCR_DEBUG_MARGIN` 逐框边跟能量。


### 工程面

- CI：三平台构建+测试、MSRV 1.85、clippy、rustfmt、cargo-deny、
  feature 矩阵。
- 文档：rustdoc 全覆盖；六个可运行 examples
  （api_smoke/api_verify/char_boxes/retry_flags/phase_breakdown）。
