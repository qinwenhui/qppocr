#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐字符对拍：把 100 图的全部识别结果（含每字坐标）落成一份可 diff 的文本。

改动内核/调度后必须与改动前的基线**逐字符相同**——这是本项目每一条
性能改动的准入条件（浮点不可结合，任何重排都会在某个像素上显形）。

用法：
    python bench/verify.py save <名字>          # 存基线
    python bench/verify.py diff <名字>          # 与基线比，打印第一处差异

环境变量（QPPOCR_*）原样透传给引擎，可以给两个配置分别设。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

DS = Path(r"D:\qinwh\code\myself\sku-manager\ocr-tool\bench\simdpaddleocr-dataset-v1\dataset")
OURS = Path(r"D:\qinwh\code\myself\qppocr\target\release\qppocr.exe")
CWD = Path(r"D:\qinwh\code\myself\qppocr")
OUT = Path(os.environ.get("TEMP", ".")) / "qppocr-verify"


def collect(tier, extra, n=100):
    lines = []
    for i in range(1, n + 1):
        r = subprocess.run(
            [str(OURS), str(DS / f"img-{i:03d}.jpg"), "--tier", tier, "--json",
             "--quiet", "--workers", "1", *extra],
            capture_output=True, timeout=900, cwd=CWD)
        d = json.loads(r.stdout.decode("utf-8", "replace"))[0]
        for j, ln in enumerate(d["lines"]):
            ch = ";".join(f"{c['char']}@{c['x0']},{c['y0']},{c['x1']},{c['y1']}"
                          for c in ln.get("chars", []))
            lines.append(f"img-{i:03d} #{j} rot={ln.get('rotation')} "
                         f"conf={ln.get('confidence'):.6f} text={ln.get('text')!r} chars={ch}")
    return "\n".join(lines)


def main():
    mode = sys.argv[1]
    name = sys.argv[2]
    tier = sys.argv[3] if len(sys.argv) > 3 else "tiny"
    extra = sys.argv[4:]
    OUT.mkdir(exist_ok=True)
    body = collect(tier, extra)
    if mode == "save":
        (OUT / f"{name}.txt").write_text(body, encoding="utf-8")
        print(f"saved {len(body.splitlines())} lines -> {OUT / (name + '.txt')}")
    else:
        ref = (OUT / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        cur = body.splitlines()
        if ref == cur:
            print(f"IDENTICAL: {len(cur)} lines")
            return
        print(f"DIFF: baseline {len(ref)} lines, now {len(cur)} lines")
        for k in range(max(len(ref), len(cur))):
            a = ref[k] if k < len(ref) else "<missing>"
            b = cur[k] if k < len(cur) else "<missing>"
            if a != b:
                print(f"  first diff at line {k}:")
                print(f"    was: {a[:300]}")
                print(f"    now: {b[:300]}")
                break
        sys.exit(1)


if __name__ == "__main__":
    main()
