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
