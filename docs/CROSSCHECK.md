# 逐位对拍记录（阶段 2 判据）

**判据**（DESIGN.md §9）：同一份 onnx，中间张量与 C++ 参考实现逐位一致。

**方法**：两侧执行器都带逐节点落盘（C++ `LEAN_DUMP_DIR`，Rust
`QPPOCR_DUMP_DIR`，文件格式相同：`%06d.f32` = i32 rank + i64×rank 形状 +
f32 数据 + manifest.tsv）。同一份模型 + 同一份确定性输入（裸小端 f32 流），
逐文件比字节。

- C++ 侧：`ocr-demo` 的 `tools/engine_runner.cpp`（重编：
  `g++ -O2 -mavx2 -mfma -std=c++17 …`，见其 build.sh）
- Rust 侧：`cargo run --release -p qppocr-core --example dump_model --
  <model> <in.f32> <dims> <out.f32>`

## 结果（2026-09-23，tiny 档 + 参考版 cls）

| 模型 | 逐位一致节点 | 偏差 | 最终输出 |
|---|---|---|---|
| rec tiny（上游） | **76/77** | 最终 Softmax：80/276240 元素 ≤1 ulp | 差 ≤1 ulp |
| det tiny（上游） | 107/177* | 全部节点 ≤6.8e-7 相对差 | 概率图差 1.35e-8 |
| cls（参考版 ppocr_cls） | **251/251** | 无 | **逐字节相同** |

\* det 的 70 个偏差节点是**同一根因的传播**（首个偏差源 node 75 已定位），
不是 70 个独立问题。

## 唯一的剩余偏差类别：libm 标量尾巴（已声明，见 kernels/activation.rs 模块头）

C++ 内核的向量循环覆盖 8 的倍数；余数尾巴走**标量 libm**（`std::erff` /
`std::exp`）。Rust 侧统一用与向量主体相同的多项式。两类值差 ≤1 ulp。

- **rec**：softmax `inner=6906`，`6906%8=2` → 每行尾 2 元素走 libm exp
  （实测 80/276240 元素差，全部在尾巴位置）。
- **det**：融合 gelu 的 `apply_act` 按行施加，`N=690`（=23×30）同样
  `690%8=2` → 每通道平面末行末 2 列（node 75 实测差异精确落在 ox=28,29）。
- 验证过 libm 无法位级复刻：mingw 的 `expf` 既不等于"多项式"也不等于
  "double exp 再舍入"（80 个样本里后者也只中 57）。**不同编译器的 C++
  参考会有不同的 libm**——复刻没有稳定目标，此偏差维持声明。

### 为什么这不是问题

- 阶段 3 判据是**文本级**（`lines[].text` 逐字符相同）：CTC argmax 与
  DB 阈值比较对 ≤1 ulp 不敏感（除非精确平局）。
- det 最终概率图差 1.35e-8，阈值 0.2/0.5 下不可能翻转。

## 对拍过程中抓出并修复的真错误（对拍的价值所在）

1. **im2col 零填充只清了一段**：C++ 是 `for kx: memset`，漏掉 kx>0 的段
   会留复用缓冲的陈旧数据（1a）。
2. **conv tile 子片长度**：`(mg-1)*ohw + rows*ow`，只给 `rows*ow` 截尾（1a）。
3. **`serial` 标志的双重语义**：它决定走面板分支还是 M 行分支，而两分支
   的 bias 加法位置不同——路径选择本身进位（1a）。
4. **pool 2x2 钳制行越界**：向量内核读 `r0+W` 跨进下一个通道（1b）。
5. **GCC 的 FMA 收缩面**（本次，全部有汇编实证）：
   - `x*a + b` → vfmadd（batchnorm 主式、resize_bilinear 四项和）
   - `bias − mean*a` → vfnmadd
   - **intrinsic 之间也收缩**：`_mm256_sub_ps(x, _mm256_mul_ps(kf, hi))`
     → vfnmadd（exp256 的 ln2 减法两处、erf256 的 `1 − p·e` 一处）。
     源码形态的分立指令在 GCC 下**不存在**；Rust 侧显式 `mul_add`/`fnmadd`。
6. **erf 系数字面值**：C++ 写 `1.421413741`，抄成 `1.421_413_7` 会舍到
   相邻的 f32（三个系数各差 1 ulp）——字面值必须**逐字符**照抄。
7. **Resize 的 sizes 按 rank 索引**：C++ 按扁平数组末两位（rank-1 的 sizes
   会下溢）。

## 遗留

- **上游 cls（opset 7）在 C++ 参考实现上段错误**——黄金值不可得，用了
  参考实现自带的转换版 cls 做数值验证（251/251 一致）。上游 cls 待
  阶段 3 接入流水线后在文本级验证。
- small / medium 档的 dump 对拍未跑（判据按 tiny 达成；§11-2 要求的
  全参数复验在 Rust 侧属于阶段 3 之后的功课）。


## 阶段 3：文本级对拍（2026-09-23）

**判据**（DESIGN.md §9）：内部语料上 `lines[].text` 与 C++ 逐字符相同。

**方法**：C++ `q-lite-ocr-cli --json`（转换版模型）vs Rust `run_ocr`
（**上游原件**全套：det/rec/cls + 外部字典）跑 `testdata/` 全部 15 张图
（114 行文本），JSON 文本逐行对比。

**结果：113/114 行逐字符相同。** 唯一差异（edge_all_samedir.png 的一个
贴边小框：C++ 读出「右」、Rust 为空）属于阶段 2 已量化的 libm 尾巴
（≤1 ulp）在 CTC 边界上的个例翻转——两边都接近读不出。

这次对比同时完成了 §4.4 未决问题 2 的 Rust 侧要求：**36 个参数在上游
原件上的复验**（转换版调的参 → 上游模型 + Rust 引擎 → 与转换版 C++
输出逐字符一致）。

### 对拍中抓出并修复

1. **区域重试的裁剪偏移**：行内偏移算成了全图平铺偏移（C++ 是
   `row + rx0*3`），receipt.png 直接越界 panic。
2. **上游 cls（opset 7）的 attribute 式 Slice**：starts/ends/axes 在
   opset ≤9 是属性不是输入——C++ 参考在这里越界读**段错误**（它跑不了
   上游 cls 的根因）。Rust 版补上，上游 cls 从此可用。
3. **cls_view 开窗对上游 PP-LCNet 有害**：C++ 给转换版 cls 打的补丁
   （30:1 宽行压缩后答错）在上游模型上把倒置英文行的中段截掉后特征
   不足——实测 win=true 时 0.72 说「正立」、win=false 时 0.9998 说
   「倒置」。做成 `cls_window` 配置，默认关（上游），转换版用户可开。

### 上游 cls 的验证

- 语义与转换版一致：`out[1] > out[0]` = 倒置（需翻 180°）。
- 画布 48×192 无窗时 rot180/rot270/rot90 全部读对（此前 win=true 时
  rot180/rot270 的英文行读成倒置乱码）。
- 竖排文本（rot90）两套模型行为一致。

### 性能现状（未优化，判据外的观察）

receipt.png：Rust 439ms vs C++ 81ms。已知主因：executor 每次 run 克隆
initializers（权重）、conv2d 输出的 `resize` 清零（TODO 标记的 memset，
det 的 47 MB 输出 ~4 ms/节点）。缓冲池是下一阶段的事。


## 阶段 5 补充：small 档全语料（2026-09-23）

**上游 small 原件（Rust）vs 转换版 small（C++）：104/104 逐字符一致（100%）。**

这是 §11-2 功课的一半（small 完成；medium 体积大、无内嵌字典，
文本对拍待做——上游 cls 段错误使 C++ 侧只能对比转换版）。

### 过程中解决的两个真问题

1. **Slice 常量查找遗漏（4 处）**：「权重零克隆」改造后 initializers
   不再进 arena，Slice 的 starts/ends/axes/steps（通常是折叠常量）
   查不到。grab 补两级查找（arena → initializers）。
2. **sdcb small/medium 字典丢了前导空行项**：sdcb 提取版首行空、
   缺 `!`/`"`——不是官方字典的真实布局。实测上游 small rec 全部
   字符偏移 +1（`Email`→`Fnbjm`）；补一个前导空行后 104/104 一致。
   这同时解释了当初归一化后 SHA 不匹配的悬案（内容差一项）。
   ⚠ `models/dict_small_medium.txt` 已带修正的前导空行；上游官方
   字典的 SHA（附录 D `118d0f07…`）待 fetch 落地时重新核对。
