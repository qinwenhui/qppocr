# qppocr-gpu

qppocr 的 GPU 设备后端：实现 `qppocr-core` 的设备接缝
（`DeviceContext` / `DeviceSession`）。

- **Vulkan 计算后端**（feature `vulkan`，默认开，依赖 `ash`）：
  Vulkan 1.4 基线。构建无需 Vulkan SDK（运行时加载 loader）；
  无 loader / 无 ICD 的机器上枚举为空，显式要求 GPU 时才报错。
- **CUDA**（feature `cuda`）：枚举级预留（dlopen，无编译期 CUDA SDK
  依赖），计算内核后续版本接入。

通过 `qppocr` 门面的 `gpu` feature 使用：

```toml
qppocr = { version = "0.3", features = ["gpu"] }
```

```no_run
# use qppocr::{Engine, DeviceChoice, GpuApi, Tier};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let engine = Engine::builder()
    .tier(Tier::Small)
    .device(DeviceChoice::Gpu { api: GpuApi::Auto, index: None })
    .build("models/")?;
# Ok(())
# }
```

状态：Phase 1 开发中——设备枚举/执行基建/装载期形状推理已就绪；
计算内核逐批接入中。显式要求 GPU 而内核未就绪时会得到明确的错误
说明，不静默回退 CPU。

## 改着色器

SPIR-V **签入**（`shaders/spirv/`），构建零着色器依赖。改 `shaders/*.comp`
后必须重编译并提交产物：

```
cargo run --manifest-path tools/shader-build/Cargo.toml
```

编译器是 naga（纯 Rust）——仓库不依赖 Vulkan SDK / glslang。
