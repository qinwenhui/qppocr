#!/usr/bin/env python3
"""head_fuse 的**逐位对拍**：同一批图，`QPPOCR_METAL_HEAD_FUSE` 0/1 两臂各
跑一遍 `--json`，逐行比对 text / confidence / rotation / box。

依据：`n_head_argmax` 的设计是「逐位等于 n_softmax + n_exit3_argmax 的
组合」，所以这里要求的是**完全相等**，不是"容差内"。任何一个 confidence
的第 7 位小数不同都算失败——那就说明融合改动动了数值路径，得回去查，
而不是把差异记成"噪声"。

## 两个必须的纪律（2026-10-11 踩过，一次差点把结论判反）

1. **`--workers 1`**。不传时 bench 走 auto，在这台 10 核机上 = 5 个并发
   Metal 进程，**输出顺序**每次不同 ⇒ 按位置比对等于拿不同图的行互比。
2. **按图名对齐，不按位置**。见上。
3. 结论要跟**同臂跑两遍**的基线比：medium 档自己两遍就差 7/1034 行文本
   （pre-existing，未定位），所以 medium 上只能看"增量是否超出噪声"。
   tiny/small 是 0 差，**那两档可以直接要求逐位一致**。

用法： python3 bench/gpuhealth/head_fuse_check.py [图数] [档位]
"""
import json
import os
import subprocess
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
TIER = sys.argv[2] if len(sys.argv) > 2 else "small"
BIN = os.environ.get("BIN", "target/release/qppocr")


def run(fuse: bool):
    e = dict(os.environ)
    e["QPPOCR_METAL_HEAD_FUSE"] = "1" if fuse else "0"
    p = subprocess.run(
        [BIN, *[f"bench/gpuhealth/corpus100/img-{i:03d}.jpg" for i in range(1, N + 1)],
         "--models", "models", "--device", "metal", "--tier", TIER,
         "--json", "--workers", "1"],
        capture_output=True, text=True, env=e, check=True,
    )
    return {
        im["image"].split("/")[-1]: [
            (ln["text"], ln["confidence"], ln["rotation"], tuple(ln["box"]))
            for ln in im["lines"]
        ]
        for im in json.loads(p.stdout)
    }


def rows(doc):
    return [(img, *ln) for img, lns in sorted(doc.items()) for ln in lns]


a, b = rows(run(False)), rows(run(True))
print(f"行数：OFF={len(a)}  ON={len(b)}")
if len(a) != len(b):
    print("❌ 行数不等")
    sys.exit(1)
bad = 0
for i, (x, y) in enumerate(zip(a, b)):
    if x != y:
        bad += 1
        if bad <= 8:
            print(f"❌ [{i}] {x[0]}\n   OFF {x[1]!r} conf={x[2]!r}\n   ON  {y[1]!r} conf={y[2]!r}")
print("✅ 逐位一致" if bad == 0 else f"❌ {bad}/{len(a)} 行不同")
sys.exit(0 if bad == 0 else 1)
