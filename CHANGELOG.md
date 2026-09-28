# Changelog

本项目的全部显著变化都记录在这个文件里。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。
每个版本写清**行为变化**，尤其是默认值变化。

## [Unreleased]

### 新增（Added）

- **GPU 计算管线与首批内核（Phase 1-F）**：单 SSBO + push constant
  偏移寻址的绑定模型（一个会话一个描述符集，零描述符churn）；
  **可复用命令缓冲**——整图 dispatch 序列装载期录一次、每帧只
  submit+等信号（Intel Arc 实测 10-dispatch 序列重提交中位
  0.355 ms，这是把每图固定开销压到 ~5ms 级的载体）。首批 10 个
  compute 内核（sigmoid/hardsigmoid/relu/clip/add/mul/mul_c/
  muladd_scale + 融合 SE 链 fused_sigmoid_mul/fused_hardsigmoid_mul
  ——SE 门与特征图的 Mul 融成一个内核，省一整趟 NCHW 读写），全部
  与 CPU 判据对拍：纯算术内核逐位一致，exp 族 ≤1.8e-7，fma 收缩
  ≤1.9e-6。着色器工具链：naga（纯 Rust）的 GLSL→SPIR-V 编译工具
  （`tools/shader-build`，独立于 workspace），SPIR-V 签入，仓库仍零
  C++ 依赖。

- **GPU planner（`qppocr-gpu::planner`，Phase 1-E）**：装载期静态形状
  推理 + i64 常量折叠（Shape/Cast/Slice/Concat/Transpose/Unsqueeze 链
  求值——Resize 的 sizes、Reshape 目标、各类 axes 的来源）。语义逐字
  镜像 executor/内核（conv/pool/ConvTranspose 输出公式、auto_pad 的
  SAME_UPPER/LOWER、右对齐广播、slice 钳位、0/-1 reshape、右对齐批次
  matmul、keepdims 归约），判据是真实 det 模型的 oracle 对拍：三个输入
  尺寸（960×960/640×640/960×512）下逐节点与 `QPPOCR_DUMP_DIR` 落盘的
  运行期形状**全部一致**（各 140 节点）。该测试即 Phase F 内核规划的
  准入门；缺模型环境自动跳过。
- **GPU 后端 crate 落地（`qppocr-gpu`，Phase 1 第一步）**：Vulkan 设备
  枚举与打开（ash；实例按 1.1 请求最大化枚举面，可用性按物理设备
  `apiVersion` 判定 1.4 基线，低于基线的设备枚举可见并标注）+ CUDA
  枚举级预留（dlopen 驱动，零编译期 SDK 依赖）。门面 feature `gpu` /
  `gpu-cuda` 接线：`--device gpu`（Auto：Vulkan 优先、CUDA 兜底）、
  `--device vulkan[:N]`、`--device cuda[:N]` 全部到达真实设备事实；
  `--device cpu` 与默认构建零变化（不引 GPU 依赖）。计算内核尚未实现
  ——显式要求 GPU 的会话构造给出明确说明，不静默回退。已在本机
  Intel Arc（Vulkan 1.4.335）实测枚举/打开。CI feature 矩阵补
  `gpu` 与 `gpu-cuda` 组合。
- **设备接缝（GPU 地基）**：`qppocr-core` 新增 `device` 模块——
  `DeviceContext`（装载期工厂，权重按值一次性移交设备）与
  `DeviceSession`（与 CPU `Session::run` 同签名的执行入口）两个 trait
  把设备边界定在 Session 层（kernels 层的 `Backend` 枚举保持纯 CPU 概念）；
  pipeline 的 det/cls/rec 会话字段改为 `Arc<dyn DeviceSession>`，CPU
  路径行为零变化（100 图逐字符对拍双基线 IDENTICAL、`--device cpu`
  abx 比值 1.008）。会话不受益于宿主侧扇出时（`prefers_host_parallelism`
  = false）cls/rec 批自动退化为串行提交、跳过分片池布局。门面新增
  `DeviceChoice` / `EngineBuilder::device()` / `Engine::device()`；CLI
  新增 `--device cpu|gpu|vulkan[:N]|cuda[:N]`（多进程 worker 回传，
  bench `--sweep device=` 可用）。GPU 实现本体（`qppocr-gpu` crate，
  Vulkan 计算后端）后续版本接入；显式要 GPU 而未编译支持时构造报错，
  不静默回退 CPU。`Session` 新增 `from_parts` / `into_parts`；
  `qppocr-core::Error` 新增 `Device` 变体（下游 exhaustive match 需跟进）。
- **CLI `bench` 子命令**：测量纪律内建——同进程跑完整语料、1 轮 warmup
  丢弃、多配置逐轮交错、中位数汇总，报告探测到的内核后端与生效配置；
  `--sweep` 对自己的图集扫 preset / rec-height / threads / rec-shards。
  `qppocr` 门面新增 `detect_backend` / `Backend` 再导出。
- 新增 `RELEASING.md`：引用模型（internal-main 私有主线 / 公开孤儿
  快照 / v tag）与发布 checklist、改动准入纪律（「声明 = 断言」、
  verify.py 对拍、abx 性能口径）全部文档化；废止长命 release-prep
  分支（0.3.0 的过期分支已删）。
- CI 新增 aarch64-linux 测试 job（交叉编译 + QEMU 用户态执行）：
  NEON↔标量逐位对拍在 linux-arm64 上持续验证，不依赖 Apple 硬件。
- **aarch64 NEON 后端**：sgemm 面板（含 implicit-GEMM 指针面板）、窄 N
  路径、深度卷积、ConvTranspose、激活（GELU/erf/exp/ReLU/clip/
  HardSigmoid/sigmoid）、softmax、二元算子、2x2 池化、双线性缩放、
  2x2 膨胀——全部与标量判据逐位一致（`bitexact` 对拍在 aarch64 上自动
  选 NEON 侧）。Apple Silicon（M 系列）与各家 ARM 服务器/手机 SoC 从
  纯标量回退升级为向量执行。f32 NEON 是 aarch64 基线指令集，无需运行时
  探测；`enable_flush_denormals` 在 aarch64 上经 FPCR 置 FZ 位。
- `Backend` 枚举新增 `Neon` 档；`detect_backend()` 在 aarch64 上返回它。

### 修复（Fixed）

- **区域重试自 0.2.0 起实际默认开启**（与 0.2.0 发布说明及 README 宣称
  相反）：当时的「默认关闭」只改了门面（`Advanced::default` 与 Balanced
  预设），漏改 core 的 `PipelineConfig::default()`——门面 `resolve_config`
  以 core 默认值为起点、Balanced 预设不清零，于是默认配置仍在约 9/100
  张图上隐式跑双倍 det（0.2.0/0.2.1 发布版均如此）。core 默认值现归零，
  默认真正关闭；Accuracy 预设与显式 `Advanced::retry_conf` 的开启路径
  不受影响。已补回归断言（门面预设契约 + core 默认值各一条测试），
  「文档声明的默认值必须被测试锁住」，防同类脱节再犯。
  逐行精度开关一致（0.2.0 发布说明的实测依据不变）；对拍基线已随
  本行为修正重建（tiny 1099→1096 行、small 1042→1041 行）。
- **池的嵌套 fork 结构性降级**：worker 线程内的 `fork_join` 改为就地
  串行执行（切分决策不变，与并行路径逐位一致），消灭「并行区内再次
  并行」争 `fork_mu` 的整类死锁。此前内核依赖「收尾激活的字节数低于
  fork 阈值」这一形状巧合避开嵌套——大通道形状越过阈值即挂死，疑似即
  medium 多进程批量（`--workers 4/8`）偶发停滞的根因；50ms 唤醒兜底
  对它无效（卡的是互斥锁不是条件变量）。触发时每线程提示一次。

### 变更（Changed）

- **内核分发层重构为单一入口**（`qppocr-kernels::arch`）：全部
  `target_arch` 接线集中到一个模块，内核文件不再含架构条件；原先
  「分发宏 + cfg-if」两套写法合一。标量参考实现从调用点内联代码抽成
  命名函数（仍是逐位判据）。x86-64 输出逐位不变。
- implicit-GEMM（卷积免 im2col 物化路径）不再限定特定架构：无向量后端
  的目标同样走指针面板，省掉 patch 矩阵的内存搬运。

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
