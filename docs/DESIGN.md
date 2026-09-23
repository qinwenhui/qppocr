# qppocr · 设计与开发文档

> **纯 Rust、无第三方推理依赖的 PP-OCRv6 推理引擎。**
> 作者：qinwh · 目标：crates.io 公开发布 · 本文档面向实现者

---

## 0. 这份文档怎么用

**给实现会话的指令：**

1. 先通读 §1~§3（定位、原则、工程结构），这三节是硬约束，不要改。
2. §4（公开 API）和 §6（参数体系）是**设计决策已经拍板**的部分，照做；有异议先提出来再动。
3. §9 是分阶段路线图，**每一阶段都有明确的完成判据**，判据不过不进下一阶段。
4. §11 列了未决问题，遇到时停下来问，不要自己猜。

**不要做的事**：不要把参考实现的外壳（Win32 GUI、多进程池）搬进来——那些与引擎无关；
不要试图"顺便优化"任何参数——§6 的数值是三年基准测出来的，**照搬**。

---

## 1. 项目定位

### 1.1 是什么

一个**手写内核**的 OCR 推理引擎，直接读 PP-OCRv6 的 ONNX，不依赖 ONNX Runtime、
不依赖 tract、不依赖任何 C/C++ 库。整条链——ONNX 解析、图优化、算子、调度、
检测/方向/识别流水线、后处理——都是自己的。

### 1.2 与现有 Rust 方案的区别

| 方案 | 推理后端 | 与 qppocr 的关系 |
|---|---|---|
| `ort`（ONNX Runtime 绑定） | C++ 的 ORT | 依赖外部二进制，跨平台要发 dylib |
| `tract` | 纯 Rust 通用框架 | 通用换来的开销；**实测在 PP-OCRv6 上明显慢** |
| **qppocr** | **自己的 AVX2/NEON/WASM 内核** | **专门为 PP-OCRv6 调过**，见 §6 |

★ 这不是营销话术，是先导实验里量过的：同机、同语料、同档位下，
基于 `tract` 的实现单图 **809.8 ms**、基于 `ort` 的实现 **203.5 ms**、
而我们自己的 C++ 参考实现 **152.4 ms**。**专用内核比通用框架快，这是这个项目存在的理由。**

> ⚠ **发布前必须处理**：上面这三组数是**内部测量**——后两者量的是我们自己的应用，
> 不是裸 `ort` / `tract`。**直接把它们写成「ort 是 203ms」是过度声称。**
> 发布前要么做一份**公开可复现**的基准（用 `ort` 和 `tract` 的公开 API 分别跑同一个
> PP-OCRv6 模型 + 同一张图，代码放进 `tools/`），要么把这段改成定性描述。
> **README 里的性能数字只有公开可复现的才能写。**

### 1.3 名字与许可

- crate 名 / 仓库名：**`qppocr`**（`q` + `PP-OCR`）
- 作者：**qinwh**（<https://github.com/qinwenhui/>）
- 许可：**MIT OR Apache-2.0 双许可**（Rust 生态惯例，写进 `Cargo.toml` 的 `license` 字段，
  仓库里放 `LICENSE-MIT` 和 `LICENSE-APACHE` 两个文件）
- ⚠ **模型不是本项目的**：PP-OCRv6 权重来自 PaddlePaddle（Apache-2.0）。
  仓库里**不放模型文件**，只放获取脚本和校验值；`NOTICE` 里写明出处与许可。

---

## 2. 核心设计原则（铁律）

1. **单实例、单图、无 IO 副作用。** 引擎的核心 API 是「一张图进、一个结果出」。
   批量、切片、多进程、文件遍历**全是上层的事**（§7）。
2. **默认值就是最优值。** `Engine::new()` 出来的结果必须是最好的一档，
   不是"中性的默认"。用户不配任何东西就该拿到好结果（§4.2 有详细论证）。
3. **`unsafe` 只允许出现在 `qppocr-kernels` 一个 crate 里。** 其余 crate 顶部写
   `#![forbid(unsafe_code)]`。这条线要能 `grep` 出来。
4. **每个魔法数字都要有出处。** §6 的每个参数在代码注释里都要写明「这个值是实测出来的，
   依据是什么」。**这是本项目最重要的资产**，不是装饰。
5. **不静默降级。** 模型校验不过、字典不匹配、输入格式不对 —— **必须报错**，
   不许"尽力而为"。C++ 前身在这里有过教训（字典拿错静默出乱码）。
6. **平台相关的代码集中在一个 module 里。** 除了那一个 module，全仓库不该出现
   `cfg(windows)` 之类的东西。

---

## 3. 工程结构

### 3.1 workspace 划分

```
qppocr/                          workspace root
├── Cargo.toml                   [workspace]
├── README.md  LICENSE-MIT  LICENSE-APACHE  NOTICE  CHANGELOG.md
├── rust-toolchain.toml          钉 MSRV
├── deny.toml                    cargo-deny：许可与依赖审计
├── .github/workflows/ci.yml
├── docs/
│   ├── DESIGN.md                本文档
│   ├── TUNING.md                §6 的展开：每个参数的实测依据
│   └── PORTING.md               与 C++ 版本的逐模块对照
├── crates/
│   ├── qppocr/                  ★ 唯一面向用户的入口（发布）
│   ├── qppocr-core/             引擎主体（publish = false）
│   ├── qppocr-kernels/          算子内核，唯一允许 unsafe（publish = false）
│   └── qppocr-cli/              参考 CLI（发布二进制）
└── tools/                       内部基准与对拍（publish = false，见 §8）
```

**为什么拆这三个 crate 而不是一个大 crate：**

- `qppocr-kernels` 独立 → `unsafe` 边界可 `grep`、可审计；SIMD 后端（AVX2/NEON/WASM）
  按 target 条件编译，不污染上层。
- `qppocr-core` 独立 → 编译单元小、增量编译快；将来要做 Android/iOS 绑定时直接复用。
- `qppocr` 只做门面 → **公开 API 面和内部实现解耦**，内部随便重构不影响 semver。

`publish = false` 的两个 crate 是内部实现细节，**不进 crates.io**，所以它们的
API 不需要稳定承诺——这一点很重要，它让你能持续重构。

### 3.2 各 crate 的职责

#### `qppocr-kernels`

```
src/
├── lib.rs            #![deny(unsafe_op_in_unsafe_fn)]，导出后端选择
├── scalar.rs         参考实现（永远保留，是 SIMD 版的判据）
├── x86.rs            AVX2 + FMA
├── aarch64.rs        NEON（阶段 5）
├── wasm.rs           simd128（阶段 5）
├── gemm.rs           sgemm：分块 + 打包 + 微内核
├── conv.rs           im2col、conv2d、convtranspose2d
├── elementwise.rs    广播二元、就地二元
├── activation.rs     relu/sigmoid/hardsigmoid/gelu/erf/clip/softmax
├── pool.rs           global_avg_pool
├── resize.rs         nearest / bilinear
└── shape.rs          concat / slice / transpose / reshape
```

**要点：**

- **`scalar.rs` 不是备用方案，是判据。** 每个 SIMD 内核都要有对应的标量版本，
  单元测试逐位（或按明确容差）比对。参考实现里已经有一套这样的对拍测试，可以直接照抄它的用例集。
- **微内核保持和 C++ 一样的写法**：`sgemm` 用「4 行 × 4 个 AVX 向量、k 内层
  broadcast+FMA、16 路独立 FMA」；`conv` 走 im2col + `sgemm`。
  ★ **不要"顺手改成更 Rust 的写法"**——写成惯用的 slice 下标会引入 bounds check，
  而这些内核的性能正是靠裸指针 + 精确的寄存器分配拿到的。内核里用 `*const f32`/`*mut f32`。
- 每个内核文件顶部的注释要抄 C++ 原注释里的**实测结论**（例如"激活放在 store 之后
  比放在累加器上快 5.3%，因为后者把 erf 的除法和常数拖进已经占满 16 个 YMM 的循环"）。

#### `qppocr-core`

```
src/
├── lib.rs            #![forbid(unsafe_code)]
├── tensor.rs         Tensor / Shape / DType，连续、NCHW、行主序
├── buffer.rs         缓冲池（对应 C++ 的 pool.hpp / buf.hpp）
├── onnx/
│   ├── mod.rs
│   ├── parse.rs      protobuf 解析（不依赖 prost，手写最小解析器）
│   └── model.rs      Graph / Node / Attribute
├── graph/
│   ├── mod.rs
│   ├── optimize.rs   常量折叠、算子融合（conv+act 等）
│   └── schedule.rs   执行顺序、内存复用
├── executor.rs       逐节点执行
├── pipeline/
│   ├── mod.rs        det → cls → rec 主流程
│   ├── det.rs        预处理（短边规则、长边帽、垂直补边）
│   ├── cls.rs        0/180 方向分类
│   ├── rec.rs        裁剪、按宽度分批、canvas 归一化
│   └── post/
│       ├── db.rs     DB 二值化、连通域、unclip、框
│       ├── merge.rs  同行相邻框合并（凸包拟合）
│       ├── ctc.rs    贪心 CTC 解码
│       └── space.rs  从像素读间距插空格（★ 见 §6 的 rec_space_gap）
├── config.rs         Config / Preset / Advanced（§4.2）
└── error.rs
```

#### `qppocr`

```
src/
├── lib.rs            门面：Engine / EngineBuilder / 重导出
├── models.rs         ModelSource：目录/字节流/可选下载
└── image.rs          Image 输入抽象（从 bytes 解码，不依赖 image crate 的具体版本）
```

---

## 4. 公开 API 设计

### 4.1 主入口

```rust
use qppocr::{Engine, Tier, Preset, Image};

// 最简：三行
let engine = Engine::new(Tier::Small, "models/")?;
let out = engine.run(&Image::open("receipt.png")?)?;
for line in &out.lines {
    println!("{:.2}  {}", line.confidence, line.text);
}

// 定制
let engine = Engine::builder()
    .tier(Tier::Tiny)
    .preset(Preset::Speed)
    .threads(4)
    .build("models/")?;
```

**API 形状的几条约定：**

- `run(&self, ...)` 收 `&self` → **`Engine` 必须 `Send + Sync`**（一次构造，多线程共享）。
- 输入用 `Image`（已解码的像素）和 `Image::open`/`from_bytes` 两种构造方式。
  ★ **解码与推理解耦**：核心只吃像素，`image` crate 是 feature 门控的可选项。
  这样 WASM 用户可以用浏览器的解码器，移动端可以用系统解码器。
- 错误用 `thiserror` 风格的 `enum Error`，**不用 `anyhow`**（库不该强制调用方的错误类型）。

### 4.2 ★ 配置与预设：默认值就是最优值

**这一段是整个 API 设计的核心，先讲清楚为什么。**

#### 论证

有一类库，默认值是"中性的"，因为作者不知道用户要用在哪。
**qppocr 不是这类库。** 我们手上有三年的基准数据，知道 `det_thresh` 从上游默认的
0.5 改成 0.2 能让 exact 从 91.31% 涨到 91.99%，知道 `rec_height=48` 在 32/40/48/56/64
里是个尖峰（87.16/91.02/93.05/90.06/84.65%）。

**把这些值放进默认值，不是"替用户做决定"，这就是这个库本身。**
一个默认 `det_thresh=0.5` 的 qppocr 严格劣于一个默认 `0.2` 的 qppocr——
后者才是我们测出来的东西。**用户 `cargo add qppocr` 之后不配任何东西，
拿到的就该是我们能给出的最好结果。**

#### 三档分层

| 层 | 面向谁 | 包含什么 | 稳定性承诺 |
|---|---|---|---|
| **`Config`（公开）** | 所有用户 | 能凭常识判断该往哪调的：档位、画布高度、输入上限、线程数 | **稳定**，semver 保护 |
| **`Advanced`（隐藏）** | 我们 + 极少数高级用户 | 基准测出来的常数：阈值、unclip、合并、分批比例 | **不承诺**，文档明说别动 |
| **内部（不暴露）** | 只有我们 | 缓冲池参数、fork 阈值、诊断开关 | 无 |

判据是一句话：**用户不需要跑基准就能讲清楚该往哪边调吗？**

- `rec_height`：40 更快、48 更准、64 更慢 → **用户按延迟预算能自己选** → 公开
- `unclip_perp`：1.0 还是 1.8 取决于图像里有没有栏杆状背景噪声
  → **用户没有量具，调它只会变差** → 隐藏

#### 具体类型

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tier { Tiny, #[default] Small, Medium }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Preset {
    /// 速度优先：rec_height 40，关掉区域重试
    Speed,
    /// 默认：我们全部基准跑出来的那套值
    #[default]
    Balanced,
    /// 精度优先：rec_height 48，开区域重试、开边距判定
    Accuracy,
}

/// 公开配置。字段全部是 `Option`，`None` = 「用预设的值」。
///
/// 这样 `Preset::Speed` + 单独覆盖一项 的语义是明确的：
/// 只覆盖你点名的那一项，其余跟预设走。
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct Config {
    pub tier: Tier,
    pub preset: Preset,
    /// 识别画布高度。None = 跟预设（Speed 40 / Balanced 48 / Accuracy 48）。
    pub rec_height: Option<u32>,
    /// 检测输入长边上限。
    pub det_max_side: Option<u32>,
    /// 原图长边预缩上限。0 = 不缩。
    pub max_side_len: Option<u32>,
    /// 0 = 自动（按可用核数）。
    pub threads: usize,
    /// 是否跑 0/180 方向分类。
    pub detect_orientation: bool,
}
```

**两条硬性约定：**

1. **`#[non_exhaustive]` 是必须的。** 加了它，以后往 `Config` 加字段不算 breaking change。
   不加，加一个字段就要发大版本。
2. **不要用字符串当选项。** 用 `enum`。`"fast"` 拼错运行时才炸，`Preset::Fast` 编译期就报。
   （这条在 crates.io 上是硬要求，不是偏好。）

#### `Advanced` 逃生通道

```rust
/// 基准测出来的常数。**不要改**——每一项都有实测依据（见 docs/TUNING.md）。
///
/// 保留它是为了让我们自己能继续调参，以及给极少数有自己量具的用户一条路。
/// 这里的字段**不提供稳定性承诺**，任何一个小版本都可能变。
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Advanced {
    pub det_thresh: f32,          // 默认 0.2
    pub box_thresh: f32,          // 默认 0.5
    pub unclip_ratio: f32,        // 默认 1.6
    pub unclip_perp: f32,         // 默认 1.0
    pub unclip_margin_thresh: f32,// 默认 0.0（关）
    pub retry_conf: f32,          // 默认 0.85
    pub merge_line_gap: f32,      // 默认 0.5
    pub rec_batch_ratio: f32,     // 默认 1.1
    pub rec_space_gap: f32,       // 默认 0.3
    pub cls_thresh: f32,          // 默认 0.9
    // ...
}
```

只能通过 `EngineBuilder::advanced(|a| ...)` 进入。文档里那句「不要改」要写重。

### 4.3 结果类型

```rust
#[derive(Clone, Debug)]
pub struct OcrResult {
    pub lines: Vec<TextLine>,
    pub image_size: (u32, u32),
    pub timings: Timings,       // 各阶段毫秒，见下
}

#[derive(Clone, Debug)]
pub struct TextLine {
    pub text: String,
    pub confidence: f32,
    /// 四个角点，**按阅读顺序**（左上、右上、右下、左下），**原图坐标**
    pub quad: [(f32, f32); 4],
    /// 0 或 180。分类器判定为倒置并翻正过就是 180。
    pub rotation: Rotation,
}

/// 分阶段计时。字段名与 C++ 版本的 JSON 保持一致，便于两边对数。
#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    pub decode_ms: f32,
    pub det_pre_ms: f32,
    pub det_infer_ms: f32,
    pub det_post_ms: f32,
    pub crop_ms: f32,
    pub cls_ms: f32,
    pub rec_pre_ms: f32,
    pub rec_infer_ms: f32,
    pub rec_post_ms: f32,
    pub total_ms: f32,
}
```

★ **`Timings` 要公开。** 我们在 C++ 版本上吃了大亏：分阶段计时以前有一半是假的
（结构体里压根没有那些字段，`--json` 输出恒为 `null`），导致"该优化哪里"只能靠猜。
公开它既是给用户的，也是**逼我们自己把每一段都真算出来**。

★ **`text` 里带不带空格是有讲究的**（见 §6 `rec_space_gap`）：
我们从**像素**读间距，不套「CJK 与数字之间加空格」那类约定。
不要把空格"归一化"掉——归一化是调用方的事。

### 4.4 模型来源

crates.io 上**不能塞几十 MB 的模型**，也不能假定用户的网络环境。方案：

```rust
pub enum ModelSource {
    /// 从目录读，文件名固定（det/cls/rec 各一，按档位）
    Dir(PathBuf),
    /// 由调用方提供字节（WASM / 移动端从 asset 里取）
    Bytes { det: Vec<u8>, cls: Vec<u8>, rec: Vec<u8> },
}

// 可选便利：feature = "fetch"
impl ModelSource {
    pub fn fetch(tier: Tier) -> Result<Self, Error>;   // 从 HF 下载 + 校验 SHA-256
}
```

#### ★★ 必须面向**上游原件**，不要面向参考实现用的那套转换版

**这是移植评估里最值得记的一条，因为它差点让整个参数体系白调。**

参考实现训练/调参用的是**转换过的模型**，不是上游原件。两者差异很大：

| | 上游原件（HuggingFace） | 参考实现用的转换版 |
|---|---|---|
| det opset | **14**，242 节点，169 个权重 | **11**，**464 节点**，213 个权重 |
| rec opset | 11，219 节点 | 11，219 节点（图相同） |
| rec 字典 | **不带**元数据 | **内嵌在 `character` 元数据里** |
| det_tiny 大小 | 1,780,590 B | 1,829,618 B |
| rec_tiny 大小 | 4,462,639 B | 4,489,813 B |

转换的动因显然是适配参考实现自己的手写解析器（降到 opset 11）和它
"字典必须从模型里读"的假设（把字典塞进元数据）。

**为什么这不要紧（已实测）**：把两套模型跑同一张图，**11 行输出逐字符相同**。
转换是语义保持的，所以**参考实现调出来的 36 个参数对上游原件同样有效**。

**但这意味着 `qppocr` 必须：**

1. **解析器支持 opset 14**（上游 det 就是 opset 14）。
   ★ 已实测参考实现的解析器能吃下上游 opset 14 的 det，所以这不是新工作，
   但移植时不要假设"只支持到 11"。
2. **面向上游模型做验证，不要面向转换版**。用户 `cargo` 下来会拿到上游原件。
3. **发布前用上游原件复验一遍参数**。上面那次逐字符相同是**单张图**的，
   不能推广到全部 36 个参数在全部档位上的行为——**这是必须补的功课**，
   不要因为"图相同"就跳过。

#### ★ 字典必须是一等输入，不能"从模型里读，读不到就死"

**这是移植评估时实测出来的硬要求，不要照抄参考实现的做法。**

参考实现只从识别模型的 `character` 元数据取字典，取不到直接抛异常。实测三档模型：

| 模型 | 自带 `character` 元数据？ |
|---|---|
| `PP-OCRv6_rec_tiny.onnx` | ✅ 有 |
| `PP-OCRv6_rec_small.onnx` | ✅ 有（37415 字符） |
| **`PP-OCRv6_rec_medium.onnx`** | ❌ **一个元数据都没有** |

**后果：medium 档在参考实现上根本跑不起来**（报 `rec model has no 'character' metadata`），
而它的图和算子其实是完全支持的——把字典注入进去之后 medium 立刻正常出结果
（`det_input=[864,960]`，方向分类也正常）。

★ **medium 的字典与 small 是同一份**（18708 项）。上游模型清单里写的就是
`shared:small_rec_dictionary`。

所以 `ModelSource` 必须把字典当成**独立于模型的输入**：

```rust
pub enum Dictionary {
    /// 从识别模型的 metadata 里取（tiny / small 走这条）
    Embedded,
    /// 显式给一份（medium 走这条，或者用户想换字典）
    File(PathBuf),
    Bytes(Vec<u8>),
}
```

**默认策略**：先试 `Embedded`，**失败了不要直接死** —— 报错信息要说明
「这个模型不带字典，请用 `Dictionary::File` 指定」，并指出该档位对应的字典行数。
`ModelSource::fetch()` 则按官方清单自动把正确的字典一起拉下来。

#### 另外三件必须做的事

1. **校验 SHA-256**，不匹配就报错并说清楚期望值是什么。
2. **校验字典与识别模型匹配**。参考实现这条做对了（拿错字典硬报错
   `invalid_shape: recognizer model output and dictionary are incompatible`），
   而别的实现有静默出乱码的。**保持这个行为。**
3. **报错信息里给出修复建议**，例如「期望 18708 行的 small/medium 字典，
   拿到的是 6904 行的 tiny」。—— 这条是从真实翻车来的：字典与档位不匹配是最高频的错误。

### 4.5 feature 开关

```toml
[features]
default   = ["std", "parallel", "image-decode"]
std       = []
parallel  = ["dep:rayon"]          # 关掉 = 单线程，WASM/嵌入式用
image-decode = ["dep:image"]       # 关掉 = 只吃已解码像素
fetch     = ["dep:ureq", "dep:sha2"]
serde     = ["dep:serde"]          # 结果的序列化
```

★ **默认不能是 `no_std`**，但 `std` 要能关。这样 WASM 和裸机场景有路可走。

---

## 5. 引擎内部架构

### 5.1 分层

```
       qppocr             Engine / Config / Preset / OcrResult
         │
       qppocr-core
         │
   ┌─────┴──────────────────────────────────────┐
   │  pipeline   det → cls → rec → post         │  这一层是我们的优势所在
   ├────────────────────────────────────────────┤
   │  executor   按调度顺序逐节点执行            │
   ├────────────────────────────────────────────┤
   │  graph      常量折叠 / 算子融合 / 内存规划   │
   ├────────────────────────────────────────────┤
   │  onnx       解析 protobuf，建图             │
   ├────────────────────────────────────────────┤
   │  tensor     连续 NCHW 行主序 + 缓冲池       │
   └─────┬──────────────────────────────────────┘
         │
   qppocr-kernels     gemm / conv / elementwise / ...
```

### 5.2 张量与内存

- `Tensor`：**连续、NCHW、行主序**。不做步长/视图系统——OCR 的图用不到，
  而步长系统会让每个内核都要处理非连续输入，得不偿失。
- **缓冲池**要保留。C++ 版本里 `pool.hpp` / `buf.hpp` 的缓冲复用是实测有效的；
  ★ **但池上限要重调**：C++ 里的 `pool_bytes_mb = 256` 是**为这台机器调的单次跑分速度**，
  注释里自己承认"在产品里从没回本"（CLI 一次跑一张图，GUI 的池一直坐在上限）。
  **Rust 版默认应该是保守值**，让 `threads`/`max_buffers` 显式开大。见 §6。
- **所有张量分配走池子**，不要在内核里 `Vec::new()`。

### 5.3 ONNX 解析

- 手写最小 protobuf 解析器，**不引入 `prost`/`prost-build`**。
  理由：ONNX 用到的 protobuf 子集很小，而引入 `prost-build` 会让构建依赖 protoc，
  这对"纯 Rust、无外部依赖"的定位是自相矛盾的。
- 只支持 PP-OCRv6 用到的算子集（见附录 B）。**遇到不支持的算子要明确报错并点名**，
  不要静默跳过节点。
- 解析结果要能序列化缓存（`serde` feature），避免每次启动重新解析。

### 5.4 图优化

C++ 版本里已验证有效的几项，**逐条搬**：

| 优化 | 收益（C++ 实测） |
|---|---|
| conv + activation 融合 | 省掉一次完整的输出读改写；★ 激活作用在**已写出的输出**上，不是累加器——后者会把 erf 的除法拖进占满 16 个 YMM 的循环，det_small 慢 5.3% |
| bias 折进 GEMM 的 store 阶段 | 省掉最常见的一次 Add 的读改写 |
| 去掉冗余 memset | det 的 1x1 conv 输出 47 MB，零填充是 ~4ms/节点的无效功 |

★ **融合的边界条件要写测试**：融合后的数值必须与未融合**逐位相同**
（C++ 注释里专门强调"两者都精确，但代价不同"）。

### 5.5 算子内核

**移植时最重要的三条：**

1. **微内核逐句照抄 C++ 的结构**，包括循环顺序、分块大小、寄存器分配意图。
   这些不是随手写的——例如 `sgemm` 的 4×4 分块是为了 16 路独立 FMA 打满流水线。
2. **裸指针，不是 slice 下标。** 见 §3.2。
3. **`scalar.rs` 永远保留**，作为 SIMD 版的判据和所有平台的兜底。

**已实测**：把这个 GEMM 内核逐句搬到 Rust（同 `-mavx2 -mfma`），
**校验和逐位相同，速度 Rust 反而快约 5%**（交错 5 轮，7.190ms → 6.815ms）。
详见 §8.1。

### 5.6 流水线

```
输入图
  → 预缩放（max_side_len 帽）
  → det 预处理：短边放到 det_limit_side_len(736) + 长边帽 det_max_side(960)
                 + 极端宽高比时上下补边
  → det 推理 → DB 后处理（thresh 0.2）→ 连通域 → 框
  → unclip（ratio 1.6，垂直方向按 unclip_perp 缩放）
  → [可选] 边距判定：框外一圈是杂波就收窄重做
  → 批量裁剪（四角点透视变换）
  → cls 方向分类（阈值 0.9，判定倒置就翻 180°）
  → 按宽高比排序、**按宽度相近分批**（rec_batch_ratio 1.1 封顶）
  → canvas 归一化到 rec_height(48)
  → rec 推理 → CTC 贪心解码
  → [可选] 从像素读间距补空格（rec_space_gap 0.3）
  → 同一行相邻框合并（merge_line_gap 0.5，**凸包拟合**，见下）
  → [可选] 区域重试：最弱行置信度 < retry_conf(0.85) 时，把那块**区域**放大重跑
```

**几个必须照搬的细节：**

- **框合并要用凸包拟合，不是移动两个角点。** C++ 版本在这里翻过三次车：
  用轴对齐外接框判据会把斜行焊死（值 4.53 个点）；判据修好之后仍然是净负，
  因为**合并本身**是 bug——它只移动四个角点里的两个，只在两个碎片严格共基线时才对。
  用八个角点的凸包拟合旋转矩形就不会塌。
- **区域重试而不是整图重试。** 只把有疑问的那块裁出来重新检测，
  比整图重跑又准又便宜。
- **rec 分批要按宽度相近切。** 一批是一个张量，所有行被 padding 到最宽那行。
  padding 不是免费的：一行被撑到自身宽度 3 倍时，识别器读它的方式会变（CJK 与数字
  之间的空格不再输出，值 4.7 个点）。所以要封顶最宽/最窄的比例。
- **rec 的空白间距从像素读**，见 §6。

### 5.7 后处理

**DB 后处理**：二值化 → 膨胀 → 连通域 → 每域取外接旋转矩形 → 按 `box_thresh` 过滤
→ unclip 扩张。`unclip_perp` 控制**垂直方向**的扩张比例——沿行方向扩张无害，
垂直方向会吃进上下邻居。

**CTC 解码**：标准贪心。`best != 0 && best != prev` 时追加字符。
★ 字典末尾有一个空格项，是**能输出的**——不要在解码阶段 strip 掉它。

**空格**：见 §6 `rec_space_gap`。这是最容易做错的一块。

---

## 6. ★ 参数体系（本项目最重要的资产）

**36 个行为参数 + 3 个诊断开关。数值全部照搬 C++ 版本的 `tuning.hpp`，一个都不改。**

分成三类处置：

### 6.1 公开（进 `Config`，semver 保护）

| 参数 | 默认 | 公开的理由 |
|---|---|---|
| `tier` | `Small` | 用户明确知道自己要哪档 |
| `preset` | `Balanced` | 见 §4.2 |
| `rec_height` | `48` | 产品的质量/速度档位。sweep：32/40/48/56/64 → exact 87.16/91.02/**93.05**/90.06/84.65%；整图中位 277/348/**414** ms。**用户按延迟预算能自己选** |
| `det_max_side` | `960` | 检测输入长边帽。736~1024 之间测出来一样（957~965/1036），960 是 PP-OCR 自己的值 |
| `max_side_len` | `960` | 原图预缩上限。sweep 640~1152 → 86.29/86.00/86.49/86.87/**86.78**/85.14/85.52% |
| `threads` | `0`(自动) | 常规 |
| `detect_orientation` | `true` | 常规 |
| `vertical_padding` | `true` | PP-OCR 的 `use_vertical_padding`，极端宽高比时补边 |

### 6.2 隐藏（进 `Advanced`，不承诺稳定）

| 参数 | 默认 | 为什么不让用户调 |
|---|---|---|
| `det_thresh` | **0.2** | 上游默认 0.5。0.5→0.2 是帕累托改进：exact 91.31→**91.99%**、CER 1.08→0.98%，且 18 张 testdata 行数不变。再降到 0.1 语料上 93.63% 但**吃掉小票的独立数量列**（19→17 行）。**用户没有语料，调它只会变差** |
| `box_thresh` | 0.5 | 与 det_thresh 耦合 |
| `unclip_ratio` | 1.6 | 依赖图像内容 |
| `unclip_perp` | **1.0** | ⚠ **最敏感的一个**。1.6 在难图（栏杆背景）上会把 41px 文字框成 68px，读出 `2026-06-07-19129:18`；1.0 正确。但合成语料想要 1.8。**用户不可能自己判断** |
| `unclip_margin_thresh` | 0.0（关） | 边距杂波判定。判定量是框内非文字行的边缘能量：难图 0.48 vs 语料 0.00~0.06 |
| `retry_conf` | 0.85 | 区域重试触发阈值。语料上最弱行 0.876，所以 100 张零触发；难图 0.742 能救回。**零代价的牌，默认开** |
| `merge_line_gap` | 0.5 | 见 §5.6，翻过三次车 |
| `rec_min_width` | 16 | 识别批次宽度下限。曾设 320，在 UI 截图上浪费 3.4 倍 |
| `rec_width_grain` | 0 | 批次宽度对齐粒度 |
| `rec_pad_min_h` | 0.0 | 补白到画布高度的策略 |
| `rec_batch` | 6 | 每批几行（有固定开销，每批约 8.6ms） |
| `rec_batch_ratio` | **1.1** | 批内最宽/最窄比例上限。⚠ 不是越小越好：补边更省但切得更碎，最优点 1.3~1.6（receipt 1.1→1.6 识别 −28%） |
| `rec_space_gap` | **0.3** | ★ 见下，单独一节 |
| `cls_thresh` | 0.9 | 方向分类翻转阈值 |
| `det_limit_side_len` | 736 | 短边规则目标 |
| `det_pixel_budget` | 6.0 | 检测输入像素不超过原图多少倍 |
| `width_height_ratio` | 8.0 | 超过它才补边 |

### 6.3 内部（不暴露）

`max_candidates`(1000)、`use_dilation`(1)、缓冲池参数（`pool_enabled`/`pool_cap_*`/
`pool_bytes_mb`）、并行阈值（`fork_min_macs`/`gemm_par_min`/`gemm_split`/`col_budget`/
`elem_fork_min_bytes`/`tp_chunks`）、诊断（`prof`/`prof2`/`trace`）。

★ **缓冲池参数在 Rust 版要重调，不要照搬。** `pool_bytes_mb = 256` 的 C++ 注释
自己写着"是为这台机器调的单次跑分速度，在产品里从没回本"。**默认给保守值。**

### 6.4 ★★ `rec_space_gap`：单独说，因为它最容易做错

**问题**：CJK 与拉丁/数字之间的空格，识别器经常不输出，导致整行 exact 判错
（合成语料上 176 行是"除了空格全对"）。

**错误解法（已否决）**：全局规则「CJK↔拉丁/数字边界插空格」。
在合成语料上 +11.2 个点、**毛损失 0**、看起来完美——
但换到扫描件语料上把 **92.31% 打成 0.00%（48 行全毁）**。
因为**两份语料的约定正好相反**，而扫描件里**同一行**就同时存在 `第1行`（紧）
和 `和金额 3.50 元`（松）。

**正确解法**：把 CTC 的**时间步映射回裁剪图的像素列**，逐列量墨迹，
相邻两字之间如果有足够长的空白段，就在那里插一个空格。
**问图像，不问字符类别。**

**实现要点：**

1. `ctc_decode` 除了文本，还要输出每个字的**时间步**和**字节偏移**。
2. 时间步 → 张量列 → 内容区列 → 裁剪图像素列：
   ```
   内容宽度 = pad_only ? min(crop.w, imgW) : ceil(imgH * crop.w / crop.h)
   字符列   = (step / T) * imgW / 内容宽度 * crop.w
   ```
3. 逐列墨迹：用**直方图中位数**估背景，每列取「最极端的像素离背景多远」。
   **极性无关**——语料里深底白字和浅底黑字都有，背景是"这行像素里最多的那个值"。
4. 相邻两字列区间内找最长连续空白段，超过 `rec_space_gap × crop.h` 就插。

**两个必须处理的边界（都是实现时真踩过的）：**

- **不要挨着已有的空格再插一个。** 空格本身也是一个解码字符、也有时间步，
  它渲染出来就是空白，于是它前后两段各被当成一次间隙，
  `native dependency` 会变成 `native  dependency`。→ **已输出空格的地方不许再插。**
- **不要挨着全角标点插。** 全角逗号的字形框只有左下角有墨，
  剩下的空白是**它自己 em 框里的**，宽度和空格重叠。
  扫遍阈值没有可分窗口（0.3/0.5/0.7 三个档：帮了合成集就把扫描件打到 17.3%/90.4%/92.3%）。
  → 用 Unicode 区块判标点（`U+3000–303F`、`U+FF01–0F`、`U+FF1A–20`、`U+FF3B–40`、
  `U+FF5B–65`，加引号/破折号/省略号），**标点旁边不插**。
  ★ 这是全项目**唯一**用到字符类别的地方，而且它回答的是**另一个问题**：
  「这片空白是不是这个字形自己的」，不是「这两个字该不该有空格」。

**实测效果**（四份语料 × 两档）：

| 语料 | 档 | 关闭 | 默认 0.3 |
|---|---|---|---|
| 合成语料 | tiny | 73.57% | **83.25%** |
| 合成语料 | small | 93.13% | **94.87%** |
| 样例语料 | tiny | 27.59% | **34.48%** |
| scan | small | 92.31% | 92.31%（不回退） |

**代价 1~2%（噪声级）。阈值 0.3 是唯一「四份语料全不回退」的档**——
0.2/0.25 会把合成语料 small 打掉 0.7~3.1 个点。

---

## 7. 并发模型

**结论：核心 crate 只做「单实例 + 算子内线程池」，批量是上层的事。**

理由（也是用户拍板的方向）：核心要干净、可测、可移植。一旦核心背了进程池和 IPC，
WASM 和移动端就没法用同一份代码了。

```
qppocr-core        单实例。算子内用线程池把单张图跑满。
                   批量接口 run_batch() 只是 for 循环 + 线程池，不做进程。
      ↑
上层 / 外壳        切片、多进程、路由、队列 —— 这些是产品的事
```

**具体：**

- `parallel` feature 打开时用 `rayon`（或自建线程池）做**算子级并行**：
  GEMM 按 N-panel 切、逐元素按字节数阈值切。
- ★ **并行阈值要照搬**：C++ 版本里 `fork_min_macs = 4e6`、`gemm_par_min = 2e5`、
  `elem_fork_min_bytes = 1<<20` 这些不是拍脑袋的——小特征图上起线程池的固定开销
  比 GEMM 本身还大（注释：fork 一次约 0.3ms）。
- **嵌套并行是死锁**。C++ 版本里 `sgemm` 有个 `serial` 参数专门给已经在
  `parallel_for` 里的调用方用。Rust 版要用类型或作用域表达这个约束
  （例如 rayon 的 `scope`，或内部用一个"已在并行区"的标记）。
- `Engine` 必须 `Send + Sync`：模型权重是只读的，共享用 `Arc`。

---

## 8. 测试与基准

### 8.1 已完成的先导实验（结论可信，可直接引用）

**GEMM 内核 A/B**（把 C++ 的 sgemm 逐句搬到 Rust，同 `-mavx2 -mfma`，交错 5 轮取最好）：

| shape | C++ | Rust | 比 |
|---|---|---|---|
| M=3136 N=64 K=576 | 3.240 ms | 3.111 ms | 0.960x |
| M=1024 N=256 K=256 | 1.860 ms | 1.716 ms | 0.923x |
| M=512 N=256 K=576 | 2.083 ms | 1.982 ms | 0.952x |
| M=64 N=64 K=64 | 0.007 ms | 0.006 ms | 0.857x |
| **合计** | 7.190 ms | **6.815 ms** | **0.948x** |

**校验和四个形状逐位相同。** 结论：**移植无性能损失，Rust 侧反而略快（LLVM vs GCC）。**

**内存 A/B**（~80MB 常驻 + 反复申请释放中间张量，交错 4 轮）：
墙钟 1060.6 → 1003.6 ms（0.946x），峰值工作集 90.5 → 89.5 MB。**无内存损失。**

### 8.2 测试分层

| 层 | 判据 | 能否公开 |
|---|---|---|
| **内核单元测试** | 对拍 `scalar.rs` 参考实现，逐位或明确容差 | ✅ **能**（自包含，不需要模型和语料） |
| **图/解析测试** | 用小 onnx fixture，比对中间张量 | ✅ 能 |
| **end-to-end 测试** | 需要模型 | ⚠ 默认 `#[ignore]`，CI 里下载模型后跑 |
| **基准 / 精度回归** | 需要**别人的语料** | ❌ **内部，不发布** |

★ **用户的要求：语料是别人的，所以基准和精度回归只在内部做。**
但**内核单元测试必须公开**——那是一个开源库可信度的基础，
而且它不需要任何第三方资产。

### 8.3 移植期的对拍方法

**`tools/test_ops.cpp` 是现成的黄金判据**（算子对拍朴素参考实现）。
移植时：

1. 先把 `ops.cpp` 的每个算子在 Rust 里实现**标量版**，
   用 `test_ops.cpp` 的参考实现和随机输入对拍。
2. 再实现 SIMD 版，与标量版逐位对拍。
3. 三阶段判据：**标量版通过 → SIMD 版逐位一致 → 整图输出与 C++ 版一致**。

**整图对拍的判据**：同一模型、同一图片，Rust 版与 C++ 版的
`lines[].text` **逐字符相同**。不允许多一个字、少一个空格。

---

## 9. 移植路线图

**每一阶段有判据，不过不进下一阶段。**

### 阶段 0 · 骨架（半天）
- workspace、三个 crate、CI、`deny.toml`、许可文件、README 骨架
- **判据**：`cargo build` 通过，CI 绿

### 阶段 1 · 内核（工作量最大，价值最高）
- 移植 `ops.cpp`：先 `scalar.rs`，再 `x86.rs`（AVX2）
- 移植 `tools/test_ops.cpp` 作为测试
- **判据**：标量版通过参考实现对拍；AVX2 版与标量版**逐位一致**；
  GEMM 性能与 C++ 版持平（目标 ≤1.05x）

### 阶段 2 · 模型与执行器
- ONNX 解析 → 建图 → 常量折叠/融合 → 执行器
- **判据**：用同一份 onnx，中间张量与 C++ 版**逐位一致**

### 阶段 3 · 流水线（★ 我们的优势在这里）
- det 预处理 / DB 后处理 / unclip / cls / rec 分批 / CTC / 空格 / 合并 / 区域重试
- **判据**：内部语料上 `lines[].text` 与 C++ 版**逐字符相同**；
  exact / CER 与 C++ 版一致

### 阶段 4 · 公开 API
- `Config` / `Preset` / `Advanced` / `Engine` / `OcrResult`
- 文档 + doctest（不需要模型的例子）
- **判据**：`cargo doc` 无警告，`#![deny(missing_docs)]` 通过

### 阶段 5 · 其他后端与 CLI
- NEON（aarch64）、WASM simd128
- `qppocr-cli`（对应参考实现的命令行入口）
- **判据**：各后端输出与 AVX2 版一致；WASM 能在浏览器里跑通一张图

### 阶段 6 · 发布
- 版本号、CHANGELOG、crates.io 元数据、MSRV 声明
- **判据**：`cargo publish --dry-run` 干净；`cargo semver-checks` 通过

---

## 10. 工程规范（crates.io 的专业度）

| 项 | 要求 |
|---|---|
| **MSRV** | 在 `Cargo.toml` 里写 `rust-version`，CI 里单独测这个版本 |
| **semver** | `Config` / `OcrResult` 等公开类型全部 `#[non_exhaustive]`（枚举除外） |
| **docs** | `#![deny(missing_docs)]`；每个公开项都要有文档；**要有不用模型的 doctest 例子** |
| **lint** | `#![forbid(unsafe_code)]`（除 kernels）；CI 跑 `clippy -D warnings` 和 `rustfmt --check` |
| **依赖** | 尽量为零。`rayon` / `image` / `serde` / `sha2` 全部 feature 门控 |
| **审计** | `cargo-deny` 查许可证与已知漏洞 |
| **CHANGELOG** | 用 keep-a-changelog 格式；每个版本写清**行为变化**（尤其是默认值变化） |
| **README** | 顶部要有：一句话是什么、性能数字（与 ort/tract 的对比）、三行上手例子、许可 |

★ **性能数字要写进 README。** 这是这个库最大的卖点，
而且我们有实测数据（见 §8.1 与 §1.2），不用吹。

---

## 11. 未决问题（遇到时问，不要猜）

1. **`medium` 档没有跑过基准。** 模型已找到并核对（附录 D），也实测跑得通
   （`det_input=[864,960]` 正常出结果，代价是单图约 1.7 s，**比 small 慢约 4 倍**）。
   但**参考实现从来没有在 medium 上跑过精度基准**，所以：
   - `Balanced` 预设照搬 small 的那套值给 medium，**是没有依据的**；
   - 要么给 medium 单独调，要么在文档里明确说"medium 未经验证"。
   **不要假装三档都验过。**
2. **参数必须用上游原件复验一遍**（见 §4.4）。参考实现调参用的是转换过的模型，
   虽然实测单图输出逐字符相同，但那只证明了**一张图**。
   发布前要在上游原件上跑一遍完整回归。
3. **`Preset` 的档位与 `rec_height` 的组合是否够**：现在只有三档预设。
   要不要再加 `Preset::Receipt` 这类**场景预设**（针对小票/截图分别调过）？待定。
4. **缓冲池默认值**：C++ 的 256 MB 是为单机跑分调的，Rust 版的默认要给多少？见 §6.3。
5. **`no_std`**：现在设计成「能关 std」，但真正的 `no_std` 支持（嵌入式）要不要做？
6. **是否提供 C ABI**：方便非 Rust 产品接入。如果要，放独立 crate `qppocr-capi`，
   不要污染主 crate。

---

## 附录 A · 参数完整清单与实测依据

见 §6 的三张表。**每一项的详细依据要抄进代码注释**，
参考实现的参数表里带着每个旋钮的 long-help，**逐条抄进代码注释**。

★ **抄注释不是形式主义。** 这个项目在参数上翻过的车包括：
「在循环里 build，把最后一轮的中间状态留在了源码里」、
「复验时只跑了该零触发的语料，恰好把唯一有信息量的样本排除在外」。
**注释里的"为什么是这个值"是防止同类错误重演的唯一手段。**

## 附录 B · 需要支持的算子

**以下 32 个是 C++ 版本实际分派的全部**（从 `ops.cpp` / `executor.cpp` / `graph_opt.cpp`
的算子名表里抠出来的）。**照这张表实现即可，不要多也不要少。**

| 类 | 算子 |
|---|---|
| **矩阵** | `MatMul` |
| **卷积** | `Conv`、`ConvTranspose` |
| **逐元素（含 numpy 广播）** | `Add`、`Sub`、`Mul`、`Div`、`Pow`、`Sqrt` |
| **激活** | `Relu`、`Sigmoid`、`HardSigmoid`、`Erf`、`Clip`、`Softmax`、`FusedGelu` |
| **池化** | `GlobalAveragePool`、`AveragePool`、`MaxPool` |
| **归一化** | `BatchNormalization`（一般会在图优化里折进 Conv，但要能处理未折的情形） |
| **形状** | `Concat`、`Slice`、`Transpose`、`Reshape`、`Shape`、`Squeeze`、`Unsqueeze`、`ReduceMean` |
| **缩放** | `Resize`（nearest / bilinear） |
| **其他** | `Constant`、`Cast`、`Identity` |

**几条注意：**

- ⚠ **没有 `Gemm`**——只有 `MatMul`。别照 ONNX 文档去实现 Gemm。
- `Identity` / `Constant` / `Cast` 大多会在图优化阶段消掉或折叠，
  但**执行器仍要能处理它们**，否则遇到没优化掉的图会崩。
- ★ **`Erf` 和 `Sigmoid` 用手写多项式近似**（C++ 里是 `erf256_ps` / `exp256_ps`），
  移植时**连系数一起抄**，不要换成 `libm`——这既影响速度也影响数值逐位一致。
- `FusedGelu` 是**融合后的中间态**，不是标准 ONNX 算子。
  Rust 版可以叫别的名字，但要让"conv+gelu 融合"这条优化有落点。

## 附录 C · C++ 源文件对照

| C++ 文件 | 行数 | Rust 对应 |
|---|---|---|
| `src/ops.cpp` | 2067 | `qppocr-kernels` |
| `src/ocr.cpp` | 1869 | `qppocr-core::pipeline` |
| `src/main.cpp` | 1010 | `qppocr-cli` |
| `src/executor.cpp` | 665 | `qppocr-core::executor` |
| `src/tuning.hpp` | 622 | `qppocr-core::config`（§6） |
| `src/util.hpp` | 598 | `qppocr-core::buffer` + 线程池 |
| `src/onnx_parser.cpp` | 514 | `qppocr-core::onnx` |
| `src/proc_pool.hpp` | 501 | **不移植**（上层的事，见 §7） |
| `src/graph_opt.cpp` | 412 | `qppocr-core::graph::optimize` |
| `src/ocr.hpp` | 301 | `qppocr-core` 各模块的类型 |
| `src/pool.hpp` `buf.hpp` | 347 | `qppocr-core::buffer` |
| `src/gui_win32.cpp` | 3066 | **不移植**（UI 无关） |
| 其余小头文件 | ~500 | 分散 |

**总计约 12,362 行 C++，其中要移植的约 8,800 行**（去掉 GUI 和进程池）。

---

## 附录 D · 模型清单（`fetch` 用，全部已下载核对）

**HuggingFace 组织**：`PaddlePaddle`　**文件名**：一律是仓库里的 `inference.onnx`

| 档 | 模型 | HF 仓库名 | 字节 | SHA-256 |
|---|---|---|---|---|
| tiny | det | `PP-OCRv6_tiny_det_onnx` | 1,780,590 | `193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8` |
| tiny | rec | `PP-OCRv6_tiny_rec_onnx` | 4,462,639 | `9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6` |
| small | det | `PP-OCRv6_small_det_onnx` | 9,880,512 | `d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e` |
| small | rec | `PP-OCRv6_small_rec_onnx` | 21,159,378 | `5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634` |
| medium | det | `PP-OCRv6_medium_det_onnx` | 62,032,837 | `eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1` |
| medium | rec | `PP-OCRv6_medium_rec_onnx` | 76,554,979 | `9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba` |

**方向分类模型（三档共用同一个）**

| 模型 | HF 仓库名 | SHA-256 |
|---|---|---|
| cls | `PP-LCNet_x0_25_textline_ori_onnx_infer` | `dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2` |

**字典**

| 用于 | 行数 | 字节 | SHA-256 |
|---|---|---|---|
| tiny | 6,904 | 27,153 | （随 tiny rec 的 `character` 元数据，无需单独文件） |
| small / **medium** | 18,708 | 74,944 | `118d0f0714ad2a37668c23d6541f2c3feb65b8214041265b567f7fd5b3365d8e` |

★ **medium 的字典与 small 是同一份**，但 medium 的 rec 模型**不带**内嵌字典（见 §4.4）。

★ 全部六个模型的 SHA-256 都已下载核对，与上游发布一致。
`ModelSource::fetch()` 请对下载结果做校验，不匹配就拒绝使用。

---

## 附录 E · 一句话记住这个项目

**别人是"调用一个推理框架"，我们是"自己写了一个，并且为 PP-OCRv6 调了三年"。**

前者给你通用性，后者给你**在同一台机器上更快、更准、更小**。
`qppocr` 的存在就是为了把这三年变成一行 `cargo add`。
