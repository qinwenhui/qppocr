#!/usr/bin/env python3
"""把 `qppocr ... --json` 的落盘结果汇成「分阶段 ms/图」表，供交付文档引用。

为什么要单独一个脚本：`qppocr bench` 的表**只给到 det/line 两级**（det 三阶段
被加总、`crop` 折进 `line`），而文档要对齐竞品 `docs/perf.md` 的粒度
（det_preprocess / det_graph / det_postprocess / crop / cls / rec_* 逐行）。
`--json` 里每个字段都在，聚合一下即可。

    ./stage_table.py 金属=metal-small.json  CPU=cpu-small.json
    ./stage_table.py --gen100 金属=...      # 顺带报检出/行数

约定：列标签用 `名字=路径.json`，第一列是**第一个**文件；`total` 行即
引擎总计时（不含图像解码，`decode_ms` 单独一行）。
"""
import argparse, json, os, sys

STAGES = [
    ("decode_ms", "图像解码"),
    ("det_pre_ms", "det 预处理"),
    ("det_infer_ms", "det 前向"),
    ("det_post_ms", "det 后处理"),
    ("crop_ms", "裁剪"),
    ("cls_ms", "cls"),
    ("rec_pre_ms", "rec 预处理"),
    ("rec_infer_ms", "rec 前向"),
    ("rec_post_ms", "rec 后处理"),
    ("total_ms", "引擎合计"),
]


def load(path):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        d = [d]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cols", nargs="+", help="名字=路径.json")
    ap.add_argument("--digits", type=int, default=1)
    a = ap.parse_args()
    cols = []
    for spec in a.cols:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"要写成 名字=路径.json，收到 {spec!r}")
        cols.append((name, load(path)))
    n = len(cols[0][1])
    print(f"# 每张图一次 run，按图求均值（{n} 张图）；单位 ms/图")
    print("| 阶段 | " + " | ".join(name for name, _ in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for key, label in STAGES:
        vals = []
        for _, imgs in cols:
            vals.append(sum(im.get("timing", {}).get(key, 0.0) for im in imgs) / len(imgs))
        print(f"| {label} | " + " | ".join(f"{v:.{a.digits}f}" for v in vals) + " |")
    # 检出/行数（与精度口径一致时才可比）
    print("\n| 项 | " + " | ".join(name for name, _ in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for label, fn in [
        ("图数", lambda im: 1),
        ("检出框数", lambda im: len(im.get("lines", []))),
        ("空文本行", lambda im: sum(1 for l in im.get("lines", []) if not (l.get("text") or ""))),
        ("cls 第二意见重试行", lambda im: im.get("num_cls_retry", 0)),
        ("cls 改判行", lambda im: im.get("num_cls_retry_flipped", 0)),
    ]:
        vals = [sum(fn(im) for im in imgs) for _, imgs in cols]
        print(f"| {label} | " + " | ".join(str(v) for v in vals) + " |")


if __name__ == "__main__":
    main()
