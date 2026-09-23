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
