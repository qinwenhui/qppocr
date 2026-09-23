# qppocr

**纯 Rust 的手写内核 PP-OCRv6 推理引擎。** 不依赖 ONNX Runtime、不依赖 tract、
不依赖任何 C/C++ 库——ONNX 解析、图优化、算子、调度、检测/方向/识别流水线、
后处理，全部是自己实现的。

> 🚧 **状态：设计中。** 本文档描述的是目标形态，代码尚未开始。
> 完整设计与开发文档见 [`docs/DESIGN.md`](docs/DESIGN.md)。

## 为什么

通用的推理框架给你通用性，代价是你用不到的那部分。`qppocr` 只做一件事：
**把 PP-OCRv6 跑到这台机器能给的最快、最准。**

上游模型默认的检测二值化阈值是 0.5，我们测出 0.2 更好（exact 91.31% → 91.99%）。
识别画布高度 48 在 32/40/48/56/64 里是个尖峰（87.16/91.02/**93.05**/90.06/84.65%）。
这类结论有几百条，全部沉淀在默认值里——**`cargo add qppocr` 之后不配任何东西，
拿到的就是我们能给出的最好结果。**

## 设计要点

- **引擎与 UI 无关。** 核心 API 是「一张图进、一个结果出」。批量、切片、多进程、
  GUI 都是上层的事。
- **`unsafe` 只有一个 crate。** `qppocr-kernels` 之外的每一行都编译期安全。
- **默认值就是最优值**，外加一组命名预设（`Speed` / `Balanced` / `Accuracy`）。
- **配置三档分层**：公开配置有 semver 承诺，调参常数在 `Advanced` 里明确不承诺。
- **三档模型**：tiny / small / medium。

## 计划中的 API

```rust
use qppocr::{Engine, Tier, Image};

let engine = Engine::new(Tier::Small, "models/")?;
let out = engine.run(&Image::open("receipt.png")?)?;

for line in &out.lines {
    println!("{:.2}  {}", line.confidence, line.text);
}
```

## 模型

**本仓库不包含模型文件。** PP-OCRv6 的权重来自
[PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0），
请自行获取，见 [`NOTICE`](NOTICE)。

## 许可

MIT OR Apache-2.0 双许可，任选其一。见 [`LICENSE-MIT`](LICENSE-MIT) 与
[`LICENSE-APACHE`](LICENSE-APACHE)。

---

作者：qinwh · <https://github.com/qinwenhui/>
