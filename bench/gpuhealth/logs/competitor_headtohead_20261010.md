# 竞品 SimdPaddleOCR 本机对拍（2026-10-10）

**背景**：`bench/gpuhealth/logs/bottleneck_map_20261008.md` §11.3 曾记「竞品快 2.4~6.5×」，
并据此把「换实现」当成主要出路。用户要求「直接拿竞品在我们这台机器测试」。
本节是实测结果，**推翻了 §11.3 的读法**。

---

## 18.1 结论先行

| | 我们 | 竞品 | 倍率 |
|---|---:|---:|---|
| medium 整图 `total/img` | **265.4 ms** | **251.6 ms** | 竞品快 **1.06×** |
| det/img | 97.5 ms | **70.8 ms** | 竞品快 **1.38×** |
| rec 段 `recinf/img` | **163.1 ms** | 171.8 ms | 我们快 1.05× |
| cls/img | 3.6 ms | 2.4 ms | — |

**同机、同 100 张图（corpus100）、同模型文件、同一次会话、跑完 health.sh 仍健康。**

⇒ **不是「快 2.4~6.5×」，是基本打平（整体差 6%），且真实差距只在 det 一段。**

---

## 18.2 §11.3 错在哪：把「竞品 vs 它自己的 CPU」读成了「竞品 vs 我们」

`docs/metal-m4.md`（竞品自述）原文：

> **结论：三档 Metal 全部净胜 CPU**（tiny 1.75× / small 4.00× / medium 6.47×）

表头是 `CPU(sharp) | Metal | 比值`，`比值` 那一列是
**他们自己的 Metal÷自己的 C# CPU 后端**，与我们毫无关系。
medium 那行 `693.0 | 107.0 | 6.47×` 即：**他们的 CPU 后端 693 ms，Metal 后端 107 ms**。

我此前把 `6.47×` 当成「他们对我们的优势」直接写进 §11.3。**这是纯粹的口径误读，不是他们的数据有问题。**

---

## 18.3 他们的 107.0 ms：本机复现不出来，但**无法判定为虚标**

用**他们自己的 harness、他们文档里的原命令**在本机跑：

```
test/Sdcb.SimdPaddleOCR.Tests -c Release -- \
  --workers 4 --model medium --engine metal --input dataset --out bench-out/metal-medium-4w.json
```

| 配置 | total mean | median | det_graph | rec_graph | cls_graph |
|---|---:|---:|---:|---:|---:|
| 本机 workers=1 | 250.3 | 253.2 | 70.78 | 171.08 | 2.40 |
| 本机 workers=4 | 251.6 | 255.7 | 70.84 | 171.79 | 2.44 |
| 他们文档（paravirt VM，4w） | — | **107.0** | **33.1** | **68.9** | 3.2 |

**差 2.4×，且 det 段差得最凶（33.1 → 70.8）。**

**但不能据此说他们虚标**，因为有一个无法排除的混杂：

- 他们的 `dataset/` **不在仓库里**（`api.github.com/repos/sdcb/SimdPaddleOCR/contents/dataset` → 404），
  是本地/CI 提供的。我下载的 394 个源文件里 **0 条 dataset 路径**。
- 我这次跑的是**我们的 corpus100** 当 `dataset`。corpus100 的图很大
  （864×976 / 1488×1248 / 1536×864 / 1632×1024 …，最大 1712×1324）。
  他们的图若明显更小，det/rec 都会按面积便宜下来，`33.1` 与 `70.8` 的差就能被解释。

⇒ **他们的绝对值对我们是不可验证的**：既不能证真也不能证伪。**唯一可比的，是同图同模型那组。**

**并发不是原因**：他们 workers 1→4 只从 250.3 动到 251.6（+0.5%），
`--conc` threads 1→2 也只从 238.5 到 229.5 ms/img。**他们是 GPU 绑定的，加并发不涨吞吐**，
所以 107.0 不可能是「4 worker 摊薄」出来的。

---

## 18.4 同图对拍（唯一公平的那组）

**图**：`bench/gpuhealth/corpus100/img-*.jpg`（100 张，同一次会话）
**模型**：`models/{medium/det,medium/rec,cls}.onnx`（他们的 9 个模型文件与我们的**字节大小完全相同**）

我们（`--workers 1`，rounds 3）：
```
config        total/img  det/img  line/img  wall/img  img/s  cls/img  recinf/img
base             265.4     97.5     166.6     267.7    3.7      3.6       163.1
```

他们（官方 harness，workers=4）：
```
total_ms mean=251.6 median=255.7 p95=314.4
det_graph 70.84 | cls_graph 2.44 | rec_graph 171.79 | lines_wall 175.67
```

**det 画布已核对为同级**：我们 det 的 entry tensor 实测
`out=[1, 3, 960, 864]`（`QPPOCR_METAL_PLAN_DUMP`，img-001 864×976）；
他们的 `ComputeDetSize` 对同图给 864×960。**同面积、同模型 ⇒ det 那 1.38× 是真的。**

（他们的 `LimitSideLength` 默认 960，与我们 `det_max_side_len: 960` 相同；
差别只在取整基数：他们 32，我们 33。）

---

## 18.5 他们快的结构性原因（读码所得）

1. **整条流水线是 fp16**：`gemm.metal` / `conv.metal` 的签名一律
   `device const half4* x` / `device const half4* o`。
   不是「fp16 权重」，是**激活也走 half**——DRAM 流量直接减半。
2. **`mm_sg_dd`**（`gemm.metal:126`）：64×64 tile，`simdgroup_half8x8` ×
   `simdgroup_multiply_accumulate` 累加到 **`simdgroup_float8x8`**（fp32 累加器）；
   A/B fragment **直接从 device memory `simdgroup_load`**，不做 threadgroup staging
   （他们的注释：这块 GPU 的 threadgroup 就是 DRAM 划出来的，staging 只多两个 barrier）。
3. **`mm_ic_sg`**（`gemm.metal:402`）：**隐式 GEMM 卷积**，im2col 内联进 A 的 gather，
   `tA/tB` 用 **+1 的 17 填充**避 bank conflict。他们自述 det medium 960² 86.2→80.0 ms，
   消掉 ~1.28 GB 无谓读写。
4. 每个 simdgroup 负责 16×32，B fragment 沿 M 复用、A fragment 沿 N 复用。

**对照我们**：`n_conv8` 每 `(t,ci4)` 是 12 次装载换 32 个 FMA（1:2.67），
且 f32 全精度 ⇒ 同样 FLOP 下 DRAM 流量是他们的 ~2×。

---

## 18.6 复现命令

```bash
# 一、源码（他们的模型 177 MB 与我们逐字节同尺寸，跳过）
#     /Users/qinwh/Projects/SimdPaddleOCR —— 394 文件 / 5.3 MB
# 二、.NET 10.0.401（brew bottle 走 ghcr.io 两次 PROTOCOL_ERROR，改用官方脚本）
curl -fsSL -o dotnet-install.sh https://dot.net/v1/dotnet-install.sh
./dotnet-install.sh --channel 10.0 --install-dir "$HOME/.dotnet"
export DOTNET_ROOT="$HOME/.dotnet"   # apphost 找不到 runtime 就报 "You must install .NET"

# 三、符号链接：他们的目录布局 ↔ 我们的模型
cd /Users/qinwh/Projects/SimdPaddleOCR
mkdir -p models bench-out/models
ln -sf /Users/qinwh/Projects/qppocr/models/{cls.onnx,medium/det.onnx,...} ...   # 见 18.4

# 四、构建与运行
$HOME/.dotnet/dotnet build test/Sdcb.SimdPaddleOCR.Tests/…csproj -c Release -f net10.0
```

```bash
# 他们（官方 harness）
./test/Sdcb.SimdPaddleOCR.Tests/bin/Release/net10.0/Sdcb.SimdPaddleOCR.Tests \
  --workers 4 --model medium --engine metal --input dataset --out bench-out/metal-medium-4w.json

# 我们
./target/release/qppocr bench bench/gpuhealth/corpus100/img-*.jpg \
  --models models --device metal --tier medium --preset balanced \
  --rounds 3 --quiet --workers 1
```

---

## 18.7 对优化方向的修正

- 「换实现/换架构」不再是主要出路：**整体只差 6%**，而 §16/§17 已经证明
  我们两段都已 ~94-98% 设备绑定、卷积主循环已到 fp32 侧天花板。
- **唯一有实测支撑的缺口是 det 的 1.38×**，且他们给的机制很清楚：
  **fp16 激活（DRAM 减半）+ 隐式 GEMM（省 1.28 GB 流量）+ device-direct fragment 装载**。
  这三样里前两样都属于「省带宽」，正好对上我们 det 的 `n_conv_dw`（218 GFLOPS，6.5%）
  与 `n_conv8` 非 MMA 段（1049 GFLOPS，31%）——**带宽受限**的那部分。
- 注意 [[fp16-dead-after-mma]] 的结论要重新审视：那条量的是
  「fp16 **权重** + MMA 门里带 `&& !fp16` ⇒ 掉回非 MMA 内核」，
  **不是**竞品做的「fp16 激活 + fp16 MMA」。两者不是一回事，此前被我混为一谈。
