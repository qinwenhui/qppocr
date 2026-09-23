# Changelog

本项目的全部显著变化都记录在这个文件里。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。
每个版本写清**行为变化**，尤其是默认值变化（DESIGN.md §10）。

## [Unreleased]

### 新增

- workspace 骨架：`qppocr`（门面）/ `qppocr-core`（引擎主体）/
  `qppocr-kernels`（唯一允许 `unsafe` 的算子内核）/ `qppocr-cli`（参考 CLI）。
- CI：构建 + 测试（三平台）、MSRV（1.85）、clippy、rustfmt、cargo-deny。
- 内核标量版全量移植（阶段 1a）：sgemm/im2col/conv2d/convtranspose、
  广播二元、激活（erf/exp 多项式系数照抄）、池化、resize、形状类、
  matmul/batchnorm。并行走 rayon（feature 门控），fork 阈值照搬 tuning.hpp。
- 内核测试：test_ops.cpp 用例集移植 + 补充覆盖（softmax/pool/resize/
  transpose/slice/concat/matmul/reduce_mean/convtranspose），自包含、
  无需模型与语料；并行与单线程两种配置都过。
- sgemm 的 AVX2+FMA 微内核（4×4 分块、尾部面板按 nn 门控加载）与
  逐位对拍测试：9 组形状 × bias/serial 组合，AVX2 与标量输出逐位相同。
- GEMM 性能对拍（交错 5 轮取最好）：Rust/C++ = 0.80~0.87x，判据
  ≤1.05x 大幅超额（tools/migration-bench/README.md）。
- 其余算子的 AVX2 向量路径与逐位对拍：激活五件套、二元平坦/run、
  depthwise 内层、convT interleave、softmax 向量相、pool 2x2 可分
  快路径（含钳制行哨兵修复）。5 组 bitexact 测试全绿。
- ONNX 解析 + 图优化 + 执行器（阶段 2）：手写 protobuf 读取器
  （opset 7/11/14 全实测）、Constant 折叠、Identity 消除、GELU 子图
  坍缩、bias 折叠、conv+act 融合、带引用计数释放的 arena 执行器。
  上游 det(opset 14)/rec 前向通过。
- 逐位对拍（docs/CROSSCHECK.md）：cls 251/251 节点逐字节一致；rec
  76/77、det 107/177（其余为同一 libm 尾巴根因的 ≤1 ulp 传播）。
  抓出并修复七类真错误，其中 GCC「intrinsic 之间也做 FMA 收缩」
  （汇编实证）是最大的一类。
- det → cls → rec 完整流水线（阶段 3）：DB 后处理（二值化/膨胀/连通域/
  旋转卡壳/box_score/unclip）、透视裁剪（8×8 单应求解）、凸包框合并、
  0/180 方向分类、按宽度分批、CTC 解码、像素空格（rec_space_gap）、
  区域重试。36 参数照搬 tuning.hpp。
- 文本级对拍：上游原件全套模型（C++ 跑不了的配置）在 testdata 15 张图
  114 行上与 C++（转换版）**113 行逐字符相同**；同时完成 §4.4 的
  上游参数复验。修复区域重试偏移、上游 cls 的 attribute 式 Slice
  （C++ 在此段错误）、cls_view 开窗对上游模型有害三个问题。
- 性能：池化缓冲（pool.hpp/buf.hpp 移植，阶段 3 遗留的短板）。
  `F32Buf` 按 2^k 分桶回收块（保守默认 64 MB 上限，§6.3 不照搬 C++
  的 256 MB）；GEMM/binary/conv 等写满型内核走 `resize_uninit`，
  权重不再每次 run 克隆（arena 两级查找）。mixed.png 端到端
  143→107 ms（C++ 80 ms，1.34x；此前 545 ms）。binary_op 曾三趟
  内存流量（清零分配 + 计算 + from_vec 拷贝）→ 单趟。附带
  QPPOCR_PROF=1 的 per-op/per-shape profile（对标 C++ LEAN_PROF2）。
  对拍无回归：逐节点 76/77、文本 113/114 维持。
- perf(conv): im2col 串行化修复嵌套并行——micro-bench 中 stride-2 的
  im2col 路径从 1.93x 慢反超到 0.74~0.83x 快。根因：C++ 的 im2col 带
  serial 参数（conv 的 tile 调用全部传 true），我的移植丢了这层语义，
  每个 2 行 tile 又嵌套进 rayon（160 个微任务吃掉 1.2ms）。附带
  conv_bench / im2col_split 两个分解计时 example。
- perf(pool): 移植 C++ fork-join 线程池（util.hpp `ThreadPool` Windows
  路径）替代 rayon——批量唤醒、主线程参与、自旋 join、参与者计数屏障
  （epoch 推进被 join 门控 ⇒ 栈上 Job 生存期安全）。跨线程并发
  `fork_join` 用 `fork_mu` 串行化而非 C++ 的挂死语义；drop rayon
  依赖（kernels 零第三方依赖）。TP_CHUNKS 回归 tuning.hpp 默认 8。
  端到端交错基准（bench v3）：Σ中位 rust 563.5 vs C++ 572.2 ms
  （**0.985x，整体反超**；v2 为 1.379x），det 单模型 33.5→26.9 ms
  （C++ 30.5），K=4 并发 0.959x，内存峰值 123 vs 130 MB。含
  resize 双线性 AVX2（bilinear_row_vec）与全局 FMA 构建旗标
  （.cargo/config.toml，对齐 C++ `-mavx2 -mfma`；workspace 级配置
  不传播给下游依赖者）。
- 公开 API（阶段 4）：`qppocr` 门面 crate——`Engine::new(tier, dir)`
  三行上手；`EngineBuilder`（tier/preset/config/advanced/threads/
  verify_sha256）；`Config`（Option 字段 = 预设覆盖语义）+ `Preset`
  （Speed/Balanced/Accuracy）+ `Advanced`（不承诺稳定的基准常数，
  只经 builder.advanced 进入）；`ModelSource::Dir/Bytes` + 上游官方
  SHA-256 校验（手写 FIPS 180-4，零依赖，NIST 向量测试）；字典三路
  查找（{tier}/dict.txt → dict.txt → ppocr_keys.txt → 内嵌）；
  feature 门控（parallel/image-decode/serde）。`#![deny(missing_docs)]`
  + `cargo doc` 零警告 + 不依赖模型的 doctest。
