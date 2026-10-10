#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rec 桶宽粒度（QPPOCR_REC_GRAIN）的识别结果对拍。

用法：
    python3 bench/gpuhealth/grain_acc.py dump <档位> <粒度> <输出文件> [起始] [张数]
    python3 bench/gpuhealth/grain_acc.py diff <文件A> <文件B>

为什么需要它：rec 的注意力头没有 mask，桶宽右零填充会经注意力混入真步概率
（`crates/qppocr-gpu/src/vulkan/n_kernel_tests.rs` 的 small_rec_pad_* 断言
「填充**会**改输出」）。所以「把桶宽粒度从 8 放大」这条性能路子的代价必须
用**逐行文本**量出来，不能只看 Δms。

逐行文本只在「同一次会话、同一档位、同一粒度」内稳定；跨粒度的差异就是
填充造成的差异本身。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OURS = ROOT / "target/release/qppocr"
CORPUS = ROOT / "bench/gpuhealth/corpus100"
MODELS = ROOT / "models"


def collect(tier, grain, n, first=1):
    env = dict(os.environ)
    if grain:
        env["QPPOCR_REC_GRAIN"] = str(grain)
    else:
        env.pop("QPPOCR_REC_GRAIN", None)
    imgs = [str(CORPUS / f"img-{i:03d}.jpg") for i in range(first, first + n)]
    # 注意：`bench` 子命令忽略 --json（它只打测量表）。要点逐行结果必须走
    # 普通模式（一次一图），这也更接近真实调用形态。
    docs = []
    for img in imgs:
        r = subprocess.run(
            [str(OURS), img, "--models", str(MODELS),
             "--device", "metal", "--tier", tier, "--preset", "balanced",
             "--json"],
            capture_output=True, text=True, env=env, cwd=str(ROOT),
        )
        if r.returncode != 0:
            sys.exit(f"qppocr 退出码 {r.returncode}（{img}）\n{r.stderr[-2000:]}")
        for ln in r.stdout.splitlines():
            ln = ln.strip()
            if not ln.startswith("["):
                continue
            try:
                docs.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    out = []
    for d in docs:
        for item in d:
            name = Path(item["image"]).name
            for k, line in enumerate(item["lines"]):
                out.append(f"{name}\t{k:02d}\t{line['confidence']:.4f}\t{line['text']}")
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "dump":
        tier, grain, path = sys.argv[2], sys.argv[3], sys.argv[4]
        first = int(sys.argv[5]) if len(sys.argv) > 5 else 1
        n = int(sys.argv[6]) if len(sys.argv) > 6 else 20
        g = 0 if grain in ("0", "off", "none") else int(grain)
        rows = collect(tier, g, n, first)
        Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"{path}: {len(rows)} 行（tier={tier} grain={g} first={first} n={n}）")
    elif cmd == "diff":
        a = Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
        b = Path(sys.argv[3]).read_text(encoding="utf-8").splitlines()
        na = sum(1 for x in a if x)
        nb = sum(1 for x in b if x)
        ta = [x.split("\t", 3)[3] if "\t" in x else x for x in a]
        tb = [x.split("\t", 3)[3] if "\t" in x else x for x in b]
        ndiff = sum(1 for x, y in zip(ta, tb) if x != y) + abs(len(ta) - len(tb))
        print(f"A {na} 行 / B {nb} 行；文本不同的行 {ndiff}（{ndiff / max(na, 1):.2%}）")
        shown = 0
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y and shown < 15:
                print(f"  [{i}] A: {x}\n      B: {y}")
                shown += 1
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
