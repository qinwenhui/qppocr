# rec 合批判据：绝对桶 → 相对跨度（2026-10-11，用户拍板落地）

## 1. 结论

把 rec 的同批分组判据从「两行落进**同一个 32 px 桶**」改成「组内**跨度 ≤32 px**」
（批宽 = 组内最大宽）。**填充上界完全不变（都是 ≤32 px）**，但不再被任意的桶
边界挡住：

| 图 | 行数 | 旧（绝对桶）run 数 | 新（相对跨度）run 数 |
|---|---:|---:|---:|
| img-001 | 10 | 9 | **6** |
| img-002 | 15 | 9 | **8** |
| img-003 | 7 | 7 | **5** |

端到端（`env_ab.sh` 奇偶轮换、`--workers 1`、30 图 × 6 轮）：

| 档位 | 对比 | Δtotal | Δline | Δdet（对照） | 符号 |
|---|---|---:|---:|---:|---|
| small | 新 vs 旧 | **+1.95** | +1.95 | +0.05 | 5/6 正 |
| medium | 新 vs 旧 | **+2.30** | **+4.00** | −2.10 | 5/6 正 |
| small | 跨度 64 vs 32 | −1.95 | −1.95 | +0.30 | 4/6 负 |

（Δ 的定义是「旧 − 新」，**正 = 新的更快**。small 那轮 Δdet = +0.05，对照干净；
medium 那轮 Δdet = −2.10 系统性为负 ⇒ 污染方向**对我们不利**（旧臂那几轮机身
更快），所以真实的 Δline 只会比 +4.00 更大；small 跨度 64 那轮第 4、5 轮 Δdet
跑 −0.90/+2.50，离散 11.8 ms，只有 1、2、6 轮可用，中位 −2.20。）

**跨度 64 只再多 ~2 ms**（边际递减），所以停在 32。

## 2. 为什么桶边界是错的

`QPPOCR_GPU_PROF=1` 的 `[metal] run#` 行只报**批宽**（`img_w`）与真实行数，
不报每行的裸宽，所以下面只说看得见的东西。img-001 的 10 行，两版的分批：

| | 各 run 的（行数 × 批宽） |
|---|---|
| 旧（绝对桶） | 1×224、**2×256**、1×288、1×384、1×416、1×448、1×896、1×992、1×1024 ⇒ **9 个 run** |
| 新（相对跨度） | **2×256**、**2×288**、1×384、**2×448**、1×896、**2×1024** ⇒ **6 个 run** |

读法：旧的 224 那行（桶 224）孤零零一个 run，而 256 桶里有 2 行——**两桶只差
32 px 却永远合不了**，因为 `bucket_w(a)==bucket_w(b)` 是相等判定，桶边界把它
们切开了。新判据放进来后 224 那行并进 256 的批（这正是它从「1×224」变成
「2×256」的来源），288 那批也多了一个成员。

跨度判据的语义是「组内 max−min ≤ grain」，批宽取**组内最大宽**，所以**最窄那行
最多被填 grain px**——填充上界与旧的 `bucket_w` 同量级（旧的 ≤31 px）。区别只在
旧的会被任意的桶边界挡住，新的不会。

## 3. 代价：逐行文本对拍（`grain_acc.py`，一图一进程）

| 档位 | 行数 | 文本不同 | 差异内容 |
|---|---:|---:|---|
| small | 1041 | **2** | `方向分类 180 度` ↔ `方向分类180 度`；`方向分类180 度` ↔ `方向分类180度` |
| medium | 411 | **3** | 一处尾随双空格；`纯托管 C# 推理` ↔ `纯托管 C#推理`；`No native dependecy` ↔ `No native **dependency**` |

**5 处里 4 处是纯空格**（都是 `rec_space_gap` 那个按列墨迹/置信度插空格的启发式
在阈值附近抖动，与 [[metal-rec-output-is-nondeterministic]] 记的
`validationtext ↔ validation text` 同一类）；第 5 处是把 `dependecy` **改对**成
`dependency`。**没有任何一处是字符认错。** 5 处里 3 处新侧置信度更高
（0.9321→0.9703、0.9592→1.0000、0.9234→0.9512）。

⚠️ 注意这与 grain 8→32 那步（1452 行 **0 差异**）不同：那步只是把**本来就会填充
的行**填得更齐；这一步让**本来不填充的行**（桶对齐的 224、384）也开始填充，
所以必然有可观测扰动。用户看过这张表后拍板落地。

## 4. 一条被否掉的解释：不是并发协议问题

头对头是「竞品 4 workers vs 我们 1 worker」（`cmp_metal.sh` 写死 `--workers 1`，
理由是并发 Metal 进程会致瘸，见 [[metal-unbound-buffer-wedges-gpu]]）。small 是
唯一输的档，而 small 的 rec 网格极小、GPU 有空隙，所以「他们是多进程填满了空隙」
是很自然的猜想。

**实测把它否掉了**：我们自己在 small 档跑 workers 1 vs workers 2（2 是这台机器的
实测健康点），交替 4 轮：

```
轮1  w1=78.3  w2=78.3  Δ=+0.00   det 21.5/21.5
轮2  w1=78.7  w2=78.7  Δ=+0.00   det 21.5/21.6
轮3  w1=78.8  w2=79.0  Δ=+0.20   det 21.5/21.6
轮4  w1=80.1  w2=78.7  Δ=-1.40   det 21.8/21.6
中位 Δ = +0.00 ms
```

⇒ **加并发一点不动**。small 的缺口是真的引擎工作量，不是量法吃亏。（竞品那边
在 medium 上 workers 1→4 也只有 +0.5%，同一形态。）

## 5. 剩下的缺口在哪：一个 rec 计划 173 条 dispatch 里 67 条是逐元素

`QPPOCR_METAL_PLAN_DUMP=1` 数一份 rec 计划（`[plan]` 表）：

| 内核族 | 条数 | |
|---|---:|---|
| `n_elem` | **48** | Add 9、Add(fused) 4、Sub/Sqrt/Sigmoid/Pow/Mul/HardSigmoid/Div 各 5 |
| `n_conv8_mma` | 27 | |
| `n_conv8` | 25 | |
| `n_channel` | **19** | Add 9、Add(fused) 5、Mul 5 |
| `n_conv_dw` | 14 | |
| `n_reduce_last` | 10 | |
| `n_transpose_nd` | 8 | |
| `n_reduce_hw`/`n_reduce_fin` | 5/5 | 5 个 SE 块 |
| `n_mm`/`n_softmax` | 4/3 | |
| 其余（pool/concat_c/entry/exit3） | 5 | |

**67/173 = 39% 是逐元素/逐通道的廉价核**，5 个 SE 块的 `HardSigmoid` 与 LayerNorm
的 `Sub/Sqrt/Div` 各自占一趟 dispatch。§11.5 早就点名把它们融进生产者的尾声是
「最大且最安全的一刀」，且**可以做逐位不变**。注意 `n_conv8`/`n_conv8_mma` 的
尾声**已经**融了 bias/act/残差（`尾: act=.. bias=true res=true`），所以那一刀指的
是这些**独立图节点**，不是 act/bias。

按 `real_plan_replay_cost` 的机制成本（仿 173 条真实名 dispatch、守卫早退 =
1.201 ms ⇒ ~6.9 µs/条）估，融掉 67 条 ≈ **0.3~0.6 ms/run**，乘每图 ~8~9.6 次 run
≈ **3~6 ms/图**。这是目前唯一还能一次拿几毫秒、且**输出逐位不变**的路子。

> 另一条更彻底的路：给 rec 的注意力加 mask（`n_attn_mm`/`n_softmax`），那样
> 右零填充就**完全不改输出**，桶宽可以随便放大（甚至直接按批拼最宽）。
> 但这两个是共享内核——本机没有 SPIR-V 工具链，改 `.metal` 会让 Vulkan
> **静默**跑旧 `.spv`（见 [[shared-shader-edits-desync-vulkan]]），所以先不动。

## 6. 复现

```bash
./bench/gpuhealth/health.sh                       # 先判健康
BIN=$PWD/target/release/qppocr TIER=small  ./bench/gpuhealth/env_ab.sh 30 6 QPPOCR_REC_BUCKET=1
BIN=$PWD/target/release/qppocr TIER=medium ./bench/gpuhealth/env_ab.sh 10 6 QPPOCR_REC_BUCKET=1
BIN=$PWD/target/release/qppocr TIER=small  ./bench/gpuhealth/env_ab.sh 30 6 QPPOCR_REC_GRAIN=64
# 逐行文本对拍（QPPOCR_REC_BUCKET 从环境里继承）
QPPOCR_REC_BUCKET=1 python3 bench/gpuhealth/grain_acc.py dump small 32 logs/acc_bucket32.txt 1 100
python3 bench/gpuhealth/grain_acc.py dump small 32 logs/acc_spread32.txt 1 100
# 计划构成
QPPOCR_METAL_PLAN_DUMP=1 target/release/qppocr bench/gpuhealth/corpus100/img-001.jpg \
    --models models --device metal --tier small --json 2>&1 >/dev/null | grep '^\[plan\]'
```

`QPPOCR_REC_BUCKET=1` 退回旧的绝对桶判据（唯一需要它的场合就是这里）。
