# Changelog

本项目的全部显著变化都记录在这个文件里。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。
每个版本写清**行为变化**，尤其是默认值变化。

## [Unreleased]

（尚无）

## [0.1.2] - 2026-09-26

### 性能：单图 -25%（相对另一款同语料手写内核实现的差距 2.2x → 1.2x）

同一台机器、同一份上游模型、同一 100 图语料，交错 A/B 实测（20 图 ×
4 轮）：**142 ms → 107 ms/图**。每一项都验证过**输出逐字符一致**。

| 改动 | 效果 |
|---|---|
| cls/rec **逐行扇出** + cls 并入同一扇出 | -12.6%（批串行+批内并行只拿到 3.2x 扩展；逐行扇出拿满机器上限 4.6x） |
| **图像缩放逐行并行**（det 输入准备） | 6.95 → 0.63 ms |
| **DB 2×2 膨胀 SIMD 化** | 3.81 → 0.29 ms |
| **凸包计算并行化** | per-box 2.36 → 1.31 ms |
| **裁剪阶段并行化** | 4.2 → 1.3 ms |
| rec_batch 6→16 | -3.6% |

新增 `par::set_serial_execution`（池的执行模式开关：只改执行方式、不改
切分决策，`threads()` 仍报真实池大小 ⇒ 与并行路径位级一致）、
`geometry::par_map`、`kernels::resize::{resize_bilinear_rgb_u8,
dilate2x2_max}`、`dump_texts`/`resize_bench` 两个诊断 example。

阶段对照（img-089）：det_post 8.7→4.0（对拍实现 4.1）、crop 4.2→1.3
（对拍实现 2.5，**反超**）、工作副本准备 7.3→1.2；det 推理与对拍实现
持平；剩余差距集中在 rec 一项。

**已证伪的方向**：全局缓冲池锁不是并发瓶颈（线程局部小块缓存实测
**慢 2.7%**，已回退）——rec 的并发膨胀是硬件墙（纯计算探针实测本机
9 线程时每线程降频 2x；叠加内存争用后共 4x）。


（尚无）

## [Unreleased]

### 性能

- **图像级缩放逐行并行**：det 输入准备（848×816 → 832×832 的全图缩放）
  实测 **6.95 ms 串行 → 0.63 ms**（-91%），rec 裁剪缩放 0.19 → 0.13 ms。
  新增 `kernels::resize::resize_bilinear_rgb_u8`（行循环交给池子），与
  core 的标量实现**逐位相同**（4 组形状含非 32 倍数，测试对拍）。
  这是同语料对拍时定位到的最大单项热点。
- 修复 rec 阶段计时双重计数（cls 墙钟曾进 rec_infer 两次）。

- **单图 -12.6%**（20 图交错 A/B 三轮一致）：cls/rec 从「批串行 + 批内并行」
  改为**逐行扇出**——每行在自己线程里串行完成「方向分类 → 识别」，行内
  串行靠池的**执行模式开关**（`par::set_serial_execution`：只改执行方式、
  不改切分决策，`threads()` 仍报真实池大小 ⇒ 位级与并行路径一致）。
  实测：`cls` 8.5→4.2 ms、`rec` 台面 38→39 但不再有串行批的 3.2x 天花板；
  单图 img-089 从 123 → 105 ms。
- 正确性：100 图输出与改造前**逐字符一致率 99.9%**（1 行空格差异，来自
  逐行批的自然宽度；行级 exact 864→863、CER 不变、完全正确图 15→16）。
  fanout=1 与 fanout=9 输出逐字符相同（扇出本身确定性）。
- 新增 `dump_texts` example：整目录跑批输出 TSV，供改造前后 diff 与评分。

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
  yyzz.jpg（3918×2772 竖拍）解码即 2772×3918。
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
