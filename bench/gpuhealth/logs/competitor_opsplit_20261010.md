# 竞品分算子对拍：本机 Metal 逐 dispatch vs 作者 VM（2026-10-10）

任务 #35。用户诉求：**只验证竞品**（"先不用管我们的"），跑本机分算子、与作者 VM 数字
对照，然后给作者一个答复。

**一句话**：作者那张 det 表里有**一个真正的异常大头**——`convk_f32n ×1`（读 fp32 输入
缓冲的那条 stem conv，每个图恰好一条），在他 VM 上 **18.1 ms**、在本机 **1.667 ms（10.9×）**，
而且是全表**唯一不受串行化地板污染**的条目（×1）。他结论里的「GEMM 系 ~70%、没有异常
大头」靠把这条 conv + `convk_dot` 也算进 GEMM 才成立（纯 mm_* 只有 54.7%）。另外他的
det 表（Σ=131.2 ms / 124 条）与他的 det_graph 墙钟（30.1 ms）**互相矛盾**，他 doc 里
自己写的 0.4~0.5 ms/dispatch 地板与他自己表里的 `elem4 3.4ms ×13` 也**互相矛盾**。

---

## 0. 取证：我们跑的就是他说的那个 revision

* 本地 `/Users/qinwh/Projects/SimdPaddleOCR` 与 GitHub tree `9f49c7b1c` 逐文件哈希：
  **402/403 字节相同**，唯一差异是我们自己的 `models/rec_keys.txt` 软链。
* 当前 main `b52a09b62` 另有 12 文件差 + 4 缺失 ⇒ 作者引用的 revision 就是我们手里的。
* **dispatch 条数逐条复现**：det `recs=124` ✓（作者 124）；cls `recs=41` ✓；rec 每行
  ≈118.2（`1066/9`、`828/7`、`594/5`、`1184/10`）✓（作者 118）。

## 1. 本机墙钟（同 revision、corpus100、medium、engine=metal）

| | 作者 VM | 本机 workers=1 | 本机 workers=4 |
|---|---|---|---|
| det_graph | 30.1 | 73.3 | 72.5 |
| cls_graph | 2.0 | 2.6 | 2.7 |
| rec_graph | 57.0 | 184.8 | 181.1 |
| **median total** | **113.5** | **~270** | **269.9** |

`SIMD_OCR_GPU_TIME=1` 的墙钟分解（本机）：
`write` 0.10~0.79、`record+submit` 0.06~1.53、`wait+read+sink` 2.5~194 ms。
⇒ **作者「CPU 侧 write+record+submit 每 run <0.3 ms，时间全在 GPU」这条在本机成立**；
但它同时意味着本机 270 ms 里 ~256 ms（94%）是 GPU 等待，**CPU 胶水解释不了我们的差**。

## 2. 取数方法：9f49c7b 上 Metal 逐 dispatch 没有入口

* `--detprof` / `--recprof` 全部硬编码 `VkDevice.Create()` + `Vulkan.GpuDetGraph`
  （`GpuBench/DetBench.cs:131`、`RecBench.cs:203`、`AndroidBench/Prof.cs:47`）。
* 引擎路径开 `SIMD_OCR_GPU_PROF=1` **什么都不打印**，而且 `MetalRunner.cs:142`
  的 `interleave = n > 1 && !s_prof && …` ⇒ **开了它 rec 的行批 interleave 会关掉**，
  时间不可比（本机实测 det_graph 从 72.5 涨到 90.2）。
* 所以：作者那份 Metal 分算子表在本 revision **无法从仓库里复现**，只能来自本地改动。
  我们自己补了一个驱动（`test/Sdcb.SimdPaddleOCR.GpuBench/MetalProf.cs`，40 行，
  直接调 `MetalRunner.DumpProfile`）——**没有改他任何一行已有代码**，只加了
  `--metalprof` 一个分支 + `Program.cs` 一行路由。

跑法：
```
cargo 无关。在 SimdPaddleOCR/ 下：
export DOTNET_ROOT="$HOME/.dotnet"
dotnet build test/Sdcb.SimdPaddleOCR.GpuBench/…csproj -c Release
SIMD_OCR_GPU_PROF=1 bin/…/GpuBench --metalprof det models/medium_det.onnx 960 640 10
SIMD_OCR_GPU_PROF=1 bin/…/GpuBench --metalprof rec models/medium_rec.onnx 7 320 5
```

## 3. det 逐 dispatch 对照（本机 960×640，串行化口径）

| pipe | 作者 VM | 本机 | 条数 | 比（他/我们） | 备注 |
|---|---|---|---|---|---|
| mm_sg_dd | 40.4 | 49.062 | 43 | **0.82×** | 他更快 |
| mm_ic_sg | 27.6 | 22.786 | 18 | 1.21× | |
| **convk_f32n** | **18.1** | **1.667** | **1** | **10.86×** | **×1 ⇒ 无地板污染** |
| conv_dw4t | 15.2 | 11.200 | 21 | 1.36× | |
| **convk_dot** | **7.2** | **2.302** | **1** | **3.13×** | **×1 ⇒ 无地板污染** |
| mm_ps_sg | 3.8 | 3.362 | 2 | 1.13× | |
| elem4 | 3.4 | 2.902 | 13 | 1.17× | |
| mmdot_sk+reduce_hw | 4.3 | 1.251+0.809 | 6+3 | 2.09× | |
| se_part/join | 3.7 | 0.670+0.508 | 2+2 | 3.14× | |
| pool/resize/convt | ~7.5 | 7.06 | 12 | 1.06× | |

本机 Σ=**102.352**、best wall=**69.99** ⇒ 串行化/wall = **1.46×**。
作者 Σ=**131.2**、det_graph=**30.1** ⇒ 串行化/wall = **4.36×**。
（本机其余档：rec 1.52×；作者 rec 1.42×。**他 det 那个 4.36× 是全场唯一离群值。**）

## 4. 他的 det 两个数字不能同时成立

设每次 commit 的固定地板为 θ（同 revision 同一台机，逐 rec 一次 commit+wait，
地板对所有 rec 相同）。逐条实测 ≥ θ ⇒ **θ ≤ min(总/条数)**。

他自己的表里：`elem4 3.4 ms ×13` ⇒ **θ ≤ 0.262 ms**；
`conv_dw4t 15.2 ×21` ⇒ θ ≤ 0.724；`mm_sg_dd 40.4 ×43` ⇒ θ ≤ 0.940。
本机实测 θ = (102.352−69.99)/124 = **0.261 ms**（与他 elem4 那根线几乎重合）。

* 取 θ ≤ 0.262 ⇒ 他 det 的真实 GPU 工作量 **≥ 131.2 − 124×0.262 = 98.7 ms**，
  是他自己 det_graph（30.1）的 **3.3×**，也是本机（70.0）的 **1.41×**。
* 取他 doc 里写的 0.4~0.5 ms 地板 ⇒ 124 条纯地板就 **49.6~62 ms**，
  真实工作量 **69~82 ms** —— 与他自己的 30.1 ms 差了 2.3~2.7×，
  **但同时** `elem4 ×13 = 3.4 ms` 就变成不可能（13×0.4 = 5.2 > 3.4）。

⇒ **他的 `det_graph 30.1 ms` 与他的 det 分算子表（Σ=131.2）无论如何都对不上**；
而他的 rec 那一对是自洽的（81.0 − 118×0.25 = 51.5 ≈ 57.0，比值 1.42 ≈ 本机 1.52）。
**不一致只在 det，不是全局的。**

## 5. rec 逐 dispatch 对照（本机 batch=7/W=320，与作者 ~7 行/批对齐）

| pipe | 作者 VM | 本机 | 条数 | 比（他/我们） |
|---|---|---|---|---|
| mm_sg_dd | 39.0 | 56.544 | 37 | **0.69×**（他更快） |
| elem4 | 12.8 | 6.470 | 31 | 1.98× |
| conv_dw4t | 8.7 | 6.152 | 15 | 1.41× |
| **convk_dot** | **6.0** | **2.353** | **3** | **2.55×** |
| mmdot_sk | 4.9 | 3.158 | 13 | 1.55× |
| **convk_f32n** | **4.1** | **0.997** | **1** | **4.11×** |
| reduce_hw | 3.7 | 2.039 | 6 | 1.81× |
| layernorm | 1.3 | 0.966 | 5 | 1.35× |
| attn | 0.5 | 0.514 | 2 | 0.97× |

本机 Σ=80.49、wall=52.84。**同一条规律复现**：`mm_*`（MMA）那条他反而更快，
两个 `convk_*`（直卷积）就是 2.5~4.1×。

## 6. `convk_f32n` 是什么（机型差异的抓手）

`MetalGraph.cs:1478`：`Emit(f32In ? _pConvDF32 : _pConvD, …)`，绑定
`(f32In ? inF32 : arena, 0)`。
⇒ **`convk_f32n` 只用在「这条 conv 的输入是图的 fp32 输入缓冲（不是 fp16 arena）」**，
每个图恰好一条，就是 **stem**（det：`convd n0 153600x36x64`，gx=2400 gy=1 tx=256；
rec 同形）。det 里 M=153600、K=36、N=64。

它是全表**唯一 θ-free** 的条目，也是唯一的离群条目 ⇒ **这是「哪里出了问题」的答案**：
不是 GEMM 吃满，是**读 fp32 输入缓冲那一条**在他 VM 上被打了 10.9×。
本机同一 dispatch 707 MFLOP / 1.667 ms；他 707 MFLOP / 18.1 ms。
（对应每图输入 960×640×3×fp32 = 7.4 MB，两机同源同网格同内核。）

## 7. 「GEMM 系合计 ~70%」怎么来的

40.4 + 27.6 + 18.1 + 7.2 + 3.8 = **97.1 / 131.2 = 74%** —— 他把
`convk_f32n`(13.8%) 与 `convk_dot`(5.5%) **也算成 GEMM** 了。纯 `mm_*` 是
(40.4+27.6+3.8)/131.2 = **54.7%**。而这两条恰恰是全场仅有的 θ-free 离群项
⇒ 「GEMM 吃满、没有异常大头」这个结论把异常本身算进了 GEMM 里。

## 8. 悬而未决 / 下一步（要问作者的）

1. **9f49c7b 上 Metal 逐 dispatch 他是怎么跑出来的？**（仓库里没有入口；我们补了驱动）
2. **det 用 `SIMD_OCR_GPU_TIME=1` 再跑一次，把 det 的 `[t]` 行贴出来**（`wait+read+sink`
   是独立仪器）。它说 ~30 还是 ~75~99，直接把第 4 节那条矛盾钉死。
3. **看 `convk_f32n`（n0）**：按他自己的表它是 det_graph 的 59%，且比本机慢 10.9×。
