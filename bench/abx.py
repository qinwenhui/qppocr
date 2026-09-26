#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用的「我们自己的两个配置」交错 A/B。

与 `ab.py` 同一套纪律（每轮交替、算当轮比值、取中位），只是把两边换成
任意两组命令行参数/环境变量，用来给单个改动定生死。

用法：
    python bench/abx.py <tier> <轮数> --a "<额外参数>" --b "<额外参数>" \
        [--env-a K=V,K2=V2] [--env-b ...]

例：
    python bench/abx.py tiny 4 --b "--rec-shards 0"
    python bench/abx.py tiny 4 --env-a QPPOCR_REC_BATCH=1 --env-b QPPOCR_REC_BATCH=4,NOCLAMP=1

输出：每轮两边的均值 + 当轮比值，最后给比值中位与极差。**比值 <1 表示 A 更快**
（比值 = B / A）。
"""
import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

DS = Path(r"D:\qinwh\code\myself\sku-manager\ocr-tool\bench\simdpaddleocr-dataset-v1\dataset")
OURS = Path(r"D:\qinwh\code\myself\qppocr\target\release\qppocr.exe")
OURS_CWD = Path(r"D:\qinwh\code\myself\qppocr")


def run_once(tier, extra, env, n=100, drop=1):
    """跑 n 张，返回 (总均值, det 均值, 行阶段均值, 张/s)。

    ⚠ **一次进程跑完 n 张**——每张一个独立进程会把冷启动开销摊到每张上
    （实测 det 44.4 vs 33.7、整体 89 → 75），而对方 runner 是一个进程跑
    100 张。见 `ab.py::ours` 的同一段说明。
    """
    e = dict(os.environ)
    e.update(env)
    imgs = [str(DS / f"img-{i:03d}.jpg") for i in range(1, n + 1)]
    t0 = time.time()
    r = subprocess.run(
        [str(OURS), *imgs, "--tier", tier, "--json",
         "--quiet", "--bench", "1", "--workers", "1", *extra],
        capture_output=True, timeout=3600, cwd=OURS_CWD, env=e)
    tot, det, line = [], [], []
    for x in json.loads(r.stdout.decode("utf-8", "replace")):
        t = x["timing"]
        tot.append(t["total_ms"])
        det.append(t["det_infer_ms"])
        line.append(t["rec_infer_ms"] + t["cls_ms"])
    tot, det, line = tot[drop:], det[drop:], line[drop:]
    return (statistics.mean(tot), statistics.mean(det), statistics.mean(line),
            n / (time.time() - t0))


def parse_env(s):
    d = {}
    for kv in filter(None, (s or "").split(",")):
        k, _, v = kv.partition("=")
        d[k.strip()] = v.strip()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tier", nargs="?", default="tiny")
    ap.add_argument("rounds", nargs="?", type=int, default=3)
    ap.add_argument("--a", default="")
    ap.add_argument("--b", default="")
    ap.add_argument("--env-a", default="")
    ap.add_argument("--env-b", default="")
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--drop", type=int, default=1)
    o = ap.parse_args()

    ea, eb = parse_env(o.env_a), parse_env(o.env_b)
    aa, ab_ = shlex.split(o.a), shlex.split(o.b)
    ratios = []
    for r in range(o.rounds):
        A = run_once(o.tier, aa, ea, o.images, o.drop)
        B = run_once(o.tier, ab_, eb, o.images, o.drop)
        ratios.append(B[0] / A[0])
        print(f"  r{r+1}: A {A[0]:6.1f}（det {A[1]:5.1f} 行 {A[2]:5.1f}）   "
              f"B {B[0]:6.1f}（det {B[1]:5.1f} 行 {B[2]:5.1f}）   "
              f"B/A {B[0]/A[0]:.3f}", flush=True)
    print(f"  → {o.tier} B/A 中位 {statistics.median(ratios):.3f}   "
          f"极差 {min(ratios):.3f}–{max(ratios):.3f}", flush=True)


if __name__ == "__main__":
    main()
