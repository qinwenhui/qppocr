# qppocr

**纯 Rust、手写内核的 PP-OCRv6 推理引擎。** 不依赖 ONNX Runtime、不依赖
tract、不依赖任何原生库——ONNX 解析、图优化、算子（AVX2/NEON SIMD）、
并行调度、检测/方向分类/识别流水线、后处理，全部自己实现，零 FFI。

```rust
use qppocr::{Engine, Tier};

let engine = Engine::new(Tier::Small, "models/")?;
let result = engine.run_image_file("receipt.png")?;   // PNG/JPEG
for line in &result.lines {
    println!("[{:.2}] {}  @ {:?}", line.confidence, line.text, line.pts);
    for ch in &line.chars { /* 逐字坐标（原图空间） */ }
}
```

## 为什么是它

通用推理框架给你通用性，代价是你用不到的那部分。现有 Rust 方案多绑定
ONNX Runtime（一个 C 库：跨平台部署带工具链、有 CVE 跟随、行为随版本
漂移）。`qppocr` 只做一件事：**把 PP-OCRv6 上游官方模型跑到这台机器
能给的最快、最准**——为此我们手写了全部内核。

- **默认值就是最优值。** 上游检测阈值 0.5，我们实测 0.2 更好
  （exact 91.31% → 91.99%）；识别画布高 48 在 32/40/48/56/64 里是尖峰
  （87.16/91.02/**93.05**/90.06/84.65%）。这类结论有几百条，全部沉淀在
  默认值里——`cargo add qppocr` 之后不配任何东西，拿到的就是我们能给出
  的最好结果。
- **逐字坐标。** `TextLine::chars` 每个字符一个原图坐标四边形（含空格
  的真实空白区间），来自 CTC 时间步对齐，性能损耗 ≈ 0。
- **并发安全。** `Engine` 是 `Send + Sync`，多线程共享一个实例并发
  `run` 是官方支持的用法。
- **透明计时。** 每次结果自带九项分阶段毫秒（det/cls/rec 的前后向与
  后处理）——慢在哪一段，数据说话。
- **配置三档分层**：`Preset`（Speed/Balanced/Accuracy，整段基准验证）
  → `Config`（有 semver 承诺的公开项）→ `Advanced`（29 项调参常数，
  明确不承诺稳定）。
- **`unsafe` 只有一个 crate**（`qppocr-kernels`），其余每行编译期安全；
  内核 crate 零第三方依赖。

## 模型

仓库不含权重（上游 Apache-2.0，体积原因）。从 HuggingFace 组织
`PaddlePaddle` 下载（文件一律叫 `inference.onnx`），按此布局摆放：

```
models/
├── tiny/   det.onnx + rec.onnx     # 档位目录名 = Tier::dir_name()
├── small/  det.onnx + rec.onnx
├── cls.onnx                        # 可选：0/180 方向分类（三档共用）
└── dict.txt                        # 字典（上游 rec 不带内嵌字典，实际必需）
```

从 HuggingFace 组织 [`PaddlePaddle`](https://huggingface.co/PaddlePaddle)
下载（文件一律叫 `inference.onnx`，重命名为上表布局）：

| 档 | 模型 | HF 仓库名 | SHA-256（装载时自动校验） |
|---|---|---|---|
| tiny | det | `PP-OCRv6_tiny_det_onnx` | `193bab7a…9dafb19f8` |
| tiny | rec | `PP-OCRv6_tiny_rec_onnx` | `9ef676d6…091563e6` |
| small | det | `PP-OCRv6_small_det_onnx` | `d73e0058…fe9c9410e` |
| small | rec | `PP-OCRv6_small_rec_onnx` | `5435fd74…2fa24634` |
| medium | det | `PP-OCRv6_medium_det_onnx` | `eb13b44b…65d086e1` |
| medium | rec | `PP-OCRv6_medium_rec_onnx` | `9c09abf0…71b673ba` |
| 三档共用 | cls | `PP-LCNet_x0_25_textline_ori_onnx_infer` | `dd8b2b61…d74d2cf2` |

字典（上游 rec 不带内嵌字典，必需）：tiny 用 `PP-OCRv6_tiny_rec_onnx`
仓库内附的字典文件；small/medium 共用一份 18,708 行字典
（SHA `118d0f07…5b3365d8e`）。完整 SHA-256 值在引擎内建校验表里
（`qppocr/src/models.rs`），下载不符会直接报错并给出两个哈希——自备
重导出模型用 `EngineBuilder::verify_sha256(false)` 跳过。字典查找顺序：
`{tier}/dict.txt` → `dict.txt` → `ppocr_keys.txt` → rec 内嵌。

## 性能（16 逻辑核桌面机实测，多轮交错取中位）

| | tiny | small |
|---|---|---|
| 单图端到端（7 图语料） | ~80 ms | ~200 ms |
| 100 图批量（CLI `--workers` 8 进程扇出） | 5.9 s | — |
| 100 图批量（API 共享引擎 2 worker） | — | 18 s |
| 进程内存峰值 | 114~128 MB | 181~191 MB |

准确率在内部多语料 + 100 图独立合成数据集上复核（判据：行级精确
匹配与 CER），tiny 与 small 档文本输出逐字符稳定。

## CLI

```bash
cargo install --path crates/qppocr-cli
qppocr img.png --tier small --json        # 单图（--json 含九项分阶段耗时）
qppocr *.jpg --workers 8                  # 批量：进程扇出，自动分图
```

常用参数：`--models <dir>`（默认 `models/`）· `--tier tiny|small|medium`
（默认 small）· `--preset speed|balanced|accuracy` · `--threads <n>`
（0=自动）· `--workers <n>`（批量并发进程数，0=自动）· `--bench <n>`
（每图跑 n 次取最好）· `--det-only`（只要框）· `--boxes`（输出带坐标）
· `--no-cls`（关方向分类）。环境变量 `QPPOCR_THREADS` 等价 `--threads`；
`QPPOCR_PROF=1` 输出逐算子耗时剖析。

## 已知限制（如实）

- `medium` 档**未跑过精度基准**（能跑通，单图约 1.7 s）；预设值系从
  small 照搬，无独立依据。
- 位级输出依赖线程数（gemm 分轴规则）：同一图在不同 `--threads` 下
  约 1/100 概率出现 ≤1 ulp 级的 box 微差（聚合指标不变）。同会话固定
  线程数即稳定。
- `Advanced` 的 29 项常数**无稳定性承诺**，小版本可能变。
- WASM/嵌入式：`--no-default-features` 单线程构建可用；七组 feature
  组合在 CI 矩阵内。

## 文档与示例

- API 文档：rustdoc（`cargo doc --open`；发布 crates.io 后 docs.rs 自动生成）
- 可运行示例（`crates/qppocr/examples/`）：`api_smoke`（三行上手）、
  `api_verify`（serde/并发语义断言）、`char_boxes`（逐字坐标几何验证）、
  `retry_flags`（区域重试标记）、`phase_breakdown`（分阶段计时）、
  `monitor_preset`（监控截图场景完整配方）

## License

MIT OR Apache-2.0（下游任选其一遵守即可）。模型权重归 PaddleOCR 上游
（Apache-2.0），见 `NOTICE`。

## 作者

qinwh · [GitHub](https://github.com/qinwenhui) · [博客](http://qinwh.cn)
