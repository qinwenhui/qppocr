# 同真值 · 双引擎 **精度** 对拍（gen100，2026-10-12）

此前只对拍过**吞吐**。这一份是第一次拿**同一份生成器真值**
（`bench/gpuhealth/gen100/metadata.json` + `gen100/img-*.jpg`，自洽 sha256）
把两个引擎的**精度**摆在一起。

- 我们的：`target/release/qppocr --tier T --device metal --workers 1 --quiet --json`
- 竞品的：`Sdcb.SimdPaddleOCR.Tests --dataset <gen100> --out gentruth-T.json`
  （`texts` 每图一个数组，**不带框** ⇒ 两边统一按**文本多重集**打分）
- 模型：竞品 `models/` 是**指向我们 `models/` 的符号链接** ⇒ 同一批 f32 权重，无档位混杂。
- 评分器：`bench/gpuhealth/acc_head2head.py`（log 在 `h2h_20261012.txt`）。

## 一、口径：严格相等 / 去空格 / 子串包含

| tier | 口径 | 我们 | 竞品 | Δ |
|---|---|---|---|---|
| tiny | 原样 | **880**/1022 = 86.11% | 798/1022 = 78.08% | **+8.02pp** |
| tiny | **去空格** | 988/1022 = 96.67% | 993/1022 = **97.16%** | **−0.49pp** |
| small | 原样 | **987**/1022 = 96.58% | 925/1022 = 90.51% | **+6.07pp** |
| small | **去空格** | 1004/1022 = 98.24% | 1012/1022 = **99.02%** | **−0.78pp** |
| medium | 原样 | **1014**/1022 = 99.22% | 1007/1022 = 98.53% | **+0.68pp** |
| medium | **去空格** | 1014/1022 = 99.22% | 1019/1022 = **99.71%** | **−0.49pp** |

**这张表就是全部结论。** 原样看我们三档全胜，把空格去掉竞品三档全胜——
差的那 0.5~0.8pp 与之前量到的 Metal-vs-CPU 浮点次序噪声同量级。

## 二、那 +6pp 是**空格**，不是字

small 逐行归属（`decompose()`）：两边都对 912 | **只有我们对 75** | 只有竞品对 13 | 两边都错 22。

75 行按**去重字符串**归并后（同一句话重复几十遍会把分辨率伪装成几十个百分点）：

| GT 字符串 | 行数 | 我们 | 竞品 |
|---|---|---|---|
| `方向分类 180 度` | 53 | **49** | 16 |
| `这是用于性能基准的更长文本行包含中文 English 以及数字 1234567890` | 38 | **34** | 25 |
| `检测后处理与文字方向分类应该保持和 PaddleOCR 官方结果一致` | 37 | **36** | 35 |
| `纯托管 C# 推理` | 36 | **35** | 34 |
| `天气晴朗 温度 28C` | 29 | **28** | 27 |
| `读取发票金额 128.50` | 26 | **25** | **0** |
| `The quick brown fox jumps over…` | 35 | 30 | **35** |
| `OpenVINO and pure managed CSharp…` | 35 | 31 | **34** |
| `PPOCRSharp dynamic recognition benchmark…` | 33 | 32 | **33** |

我们赢的**全是中英/数字边界带空格的行**；竞品赢的**全是长英文行**（各差 1 行）。

### 那些空格在图上**真的画出来了**（决定性）

生成器 `.../TestData/Program.cs:309` 是 `canvas.DrawText(text, …)` ——
**逐字画 `text`**，空格是字体的空格 advance，不是记号。

独立验证：对 11 条**轴对齐**（`|angle|<0.5°`、`orientation=0`）且含空格的 GT 行，
由 `bbox_w − 20`（`rw = ceil((textW+20)·cos + (textH+20)·sin)`，angle≈0 时即 `textW+20`）
反推渲染的 ink 宽，与 PIL 按**同一个字体文件 + 同一个 `font_size`** 渲出来的
ink 宽对比：

    textW / 有空格 ink 宽  均值 = 1.0132     ← 逐条命中（11/11）
    textW / 无空格 ink 宽  均值 = 1.0771     ← 系统性偏大

⇒ 图上确实画了空格；我们的输出更接近真值，竞品的 `读取发票金额128.50` 是**真丢了字**。

## 三、我们靠什么拿到这个空格：`rec_space_gap`（自有启发式，非模型能力）

字典 `models/dict_small_medium.txt`（18708 行）里**根本没有空格字符**，
CTC 不可能吐出空格。空格是 `crates/qppocr-core/src/pipeline/rec.rs:218 add_gap_spaces()`
**按几何补的**：两个相邻解码字符之间裁剪图上空出了一段 ≥ `rec_space_gap · crop_h`
（默认 `0.3`，`config.rs:132`）的空白 run，就插一个空格。

竞品没有这一步（`SharpEngine.cs:81` 直接 `result.Lines.Select(x => x.Text)`，
无 trim、无补空格），所以它**一个空格都不会给**。

| | 严格命中 | 多给空格 | 少给空格 |
|---|---|---|---|
| 我们 (small) | 987 | **1** | 34 |
| 竞品 (small) | 925 | 0 | 97 |

⇒ 我们的启发式**精确率极高**（34 漏 vs 1 误），在**这份语料**上近乎纯赚。

## 四、诚实的边界

1. **+6pp 不能记成「识别能力强 6pp」。** 去掉空格后竞品略胜 0.5~0.8pp。
   两边的差距都落在之前量过的浮点次序噪声量级里。
2. **这 0.5~0.8pp 的形态**：small 上竞品赢的 13 行里，medium 档显形的
   `图像处理流水线` / `宽度排序与填充` 正是 `abs-accuracy-vs-generator-truth` 里
   已定性的**真值把一条视觉整行拆成两条条目**（img-056，间隙 4px）。
   medium 档竞品赢的 5 行里 2 行是这个来源 ⇒ **那部分不是它读得准，是它恰好多切了一刀**。
3. 我们的空格启发式在**别的标注约定**下会变成扣分项：若真值本身不含
   中英边界空格，我们反而多给空格。**这份语料**是「空格真画了」的那一档，
   所以是赚。
4. 竞品 harness **自带精度评分**，其 `exact_lines 925 / total_lines 1022`
   与本脚本独立算出的 925/1022 **逐数吻合** ⇒ 双方评分器互证。

## 复现

    bench/gpuhealth/health.sh -q
    cd ~/Projects/SimdPaddleOCR && dotnet run --project test/Sdcb.SimdPaddleOCR.Tests -- \
        --dataset <…>/bench/gpuhealth/gen100 --out bench-out/gentruth-small.json
    bench/gpuhealth/acc_head2head.py --tier small --ours metal \
        --comp ~/Projects/SimdPaddleOCR/bench-out/gentruth-small.json \
        --dump-ours logs/acc_h2h/ours-small.json
