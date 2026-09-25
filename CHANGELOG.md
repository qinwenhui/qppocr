# Changelog

本项目的全部显著变化都记录在这个文件里。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。
每个版本写清**行为变化**，尤其是默认值变化。

## [Unreleased]

（尚无）

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
