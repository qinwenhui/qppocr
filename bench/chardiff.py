#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐字符对拍（macOS/Linux 版）：把一张或多张图的全部识别结果（含每字
坐标）落成可 diff 的文本。

与 `verify.py` 同一条纪律（改动内核/调度后必须与基线逐字符相同），差别
只在路径与调用方式——这里是本机自用：

    python3 bench/chardiff.py collect <名字> [--device d] [--env K=V ...] -- imgs...
    python3 bench/chardiff.py diff <a> <b>          # 逐行比，打印第一处差异

环境变量用 `--env` 逐项给，不进 shell 全局。
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "qppocr"
OUT = Path("/tmp/qppocr-chardiff")


def collect(name, images, device, tier, envs, extra):
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for kv in envs:
        k, _, v = kv.partition("=")
        env[k] = v
    lines = []
    for img in images:
        p = Path(img)
        r = subprocess.run(
            [str(BIN), str(p), "--tier", tier, "--json", "--quiet",
             "--workers", "1", "--device", device, *extra],
            capture_output=True, timeout=1800, cwd=str(ROOT), env=env)
        if r.returncode != 0:
            raise SystemExit(f"{p.name} 失败: {r.stderr.decode('utf-8', 'replace')[-800:]}")
        d = json.loads(r.stdout.decode("utf-8", "replace"))[0]
        for j, ln in enumerate(d["lines"]):
            ch = ";".join(f"{c['char']}@{c['x0']},{c['y0']},{c['x1']},{c['y1']}"
                          for c in ln.get("chars", []))
            lines.append(
                f"{p.name} #{j} rot={ln.get('rotation')} "
                f"conf={ln.get('confidence'):.6f} text={ln.get('text')!r} chars={ch}")
    dst = OUT / f"{name}.txt"
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[chardiff] {name}: {len(lines)} 行 → {dst}")


def diff(a, b):
    A = (OUT / f"{a}.txt").read_text(encoding="utf-8").splitlines()
    B = (OUT / f"{b}.txt").read_text(encoding="utf-8").splitlines()
    if len(A) != len(B):
        print(f"[chardiff] 行数不同: {a}={len(A)} {b}={len(B)}")
    n = 0
    for i, (x, y) in enumerate(zip(A, B)):
        if x != y:
            n += 1
            if n <= 20:
                print(f"[chardiff] 第 {i} 行不同")
                print(f"  {a}: {x}")
                print(f"  {b}: {y}")
    print(f"[chardiff] {a} vs {b}: {n} 处不同 / {max(len(A), len(B))} 行")
    return n


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("name")
    c.add_argument("images", nargs="+")
    c.add_argument("--device", default="cpu")
    c.add_argument("--tier", default="small")
    c.add_argument("--env", action="append", default=[])
    c.add_argument("--extra", action="append", default=[])
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.name, a.images, a.device, a.tier, a.env, a.extra)
    else:
        sys.exit(1 if diff(a.a, a.b) else 0)


if __name__ == "__main__":
    main()
