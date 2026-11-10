# qppocr-metal

qppocr 的 Metal 设备后端：实现 `qppocr-core` 的设备接缝
（`DeviceContext` / `DeviceSession`），Apple GPU 原生直连，仅 macOS。

- **Metal 计算后端**（feature `metal`，默认开，依赖 gfx-rs 的 `metal`
  绑定）：MSL 源码内嵌、运行时编译——构建零原生依赖（链系统
  Metal.framework，与 Vulkan 侧链 loader 同级）。
- 非 macOS 平台编译为空枚举 + 明确报错（类型面完整）——与 Vulkan 侧
  「无 loader 枚举为空、显式要求才报错」同一纪律。

架构与 Vulkan 后端（`qppocr-gpu`）镜像同构：单一大 arena +
参数块间接寻址，内核（`shaders/*.metal`）与 `qppocr-gpu/shaders/*.comp`
逐行对应——改语义必须两边同步。统一内存（Apple Silicon 全系 UMA）
下 `StorageModeShared` 缓冲 CPU/GPU 同址，无 staging 往返。

通过 `qppocr` 门面的 `gpu` feature 使用（依赖只挂 macOS target，其他
平台零成本）：

```toml
qppocr = { version = "0.3", features = ["gpu"] }
```

```no_run
# use qppocr::{Engine, DeviceChoice, GpuApi, Tier};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let engine = Engine::builder()
    .tier(Tier::Small)
    .device(DeviceChoice::Gpu { api: GpuApi::Metal, index: None })
    .build("models/")?;
# Ok(())
# }
```

状态：det/rec/cls 整图会话（计划构建与 Vulkan 共用 `qppocr_gpu::plan`
后端无关模型，参数编码同源）。批维补齐/real_n 早退/CTC argmax 出口/
精确宽/部署分级（`QPPOCR_GPU_STAGES`）与 Vulkan 侧同款语义；macOS 上
`GpuApi::Auto` 优先 Metal，MoltenVK 兜底。

## 改着色器

MSL 源码（`shaders/*.metal`）内嵌进二进制、运行时编译——无需离线
编译器，无产物签入步骤（区别于 Vulkan 侧的 SPIR-V 签入链）。
