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
