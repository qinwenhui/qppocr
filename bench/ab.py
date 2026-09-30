#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交错 A/B：**每轮交替跑两边、算当轮比值，最后取比值的中位**。

为什么必须这样：这台机器（6P+8E 的笔记本）在**分钟级**上有 20-40% 的
频率漂移。任何「先跑完 A 再跑完 B」的协议，第二个配置接手的是已经被烧热的
机器，比出来的差值里混着热降频。只有把两个配置放进同一轮的同一段时间里，
比值才是干净的。

用法：
    python bench/ab.py ours            # 只跑我们（默认配置 vs 关分片）
    python bench/ab.py vs <对方exe>    # 我们 vs 另一个引擎

输出每轮的原始值 + 比值，最后给比值的中位数与极差。
"""
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

DS = Path(r"D:\qinwh\code\myself\sku-manager\ocr-tool\bench\simdpaddleocr-dataset-v1\dataset")
OURS = Path(r"D:\qinwh\code\myself\qppocr\target\release\qppocr.exe")
OURS_CWD = Path(r"D:\qinwh\code\myself\qppocr")
# AB_SIMD_EXE：对方跑器路径覆盖（默认主检出的 CPU 构建；GPU 对决指向
# feature/2.0 worktree 的构建产物，配 AB_THEIR_ENGINE=vulkan）
SIMD = Path(os.environ.get(
    "AB_SIMD_EXE",
    r"D:\qinwh\code\myself\SimdPaddleOCR\test\Sdcb.SimdPaddleOCR.Tests\bin\Release\net10.0\Sdcb.SimdPaddleOCR.Tests.exe"))
SIMD_CWD = SIMD.parent
ROUNDS = 4


def ours(tier, extra=(), quiet=True):
    """一遍 100 张，返回 (均值, 阶段均值, 每秒张数)。

    ⚠ **必须在一个进程里跑完 100 张。** 曾经每张图起一个独立进程，看着
    「更干净」，实际是把每次冷启动的开销平摊到每一张：同图同配置实测
    det 44.4 ms（独立进程）vs 33.7 ms（同进程第二张起），整体 89 → 75。
    对方的 runner 是一个进程跑 100 张（`--count 100`），拿我们的冷进程
    去比它等于白送 19%。两边都得是热的。
    """
    tot, det, line = [], [], []
    imgs = [str(DS / f"img-{i:03d}.jpg") for i in range(1, 101)]
    t0 = time.time()
    r = subprocess.run(
        [str(OURS), *imgs, "--tier", tier, "--preset", "speed", "--json",
         "--quiet", "--bench", "1", "--workers", "1", *extra],
        capture_output=True, timeout=3600, cwd=OURS_CWD)
    for x in json.loads(r.stdout.decode("utf-8", "replace")):
        t = x["timing"]
        tot.append(t["total_ms"])
        det.append(t["det_infer_ms"])
        line.append(t["rec_infer_ms"] + t["cls_ms"])
    # 去掉首图（模型加载后的第一张）
    tot, det, line = tot[1:], det[1:], line[1:]
    return statistics.mean(tot), statistics.mean(det), statistics.mean(line), 100 / (time.time() - t0)


def theirs(tier, tag):
    # AB_THEIR_ENGINE：对方引擎（默认 sharp=CPU；vulkan=他们的 GPU 后端，
    # 需先在竞品仓库 checkout feature/2.0 并把 SIMD 指到其构建产物）
    engine = os.environ.get("AB_THEIR_ENGINE", "sharp")
    r = subprocess.run(
        [str(SIMD), "--benchmark", "--benchmark-kind", "simd", "--engine", engine,
         "--workers", "8", "--model", tier, "--input", str(DS), "--count", "100",
         "--warmup", "1", "--case-id", tag,
         "--out", str(Path.home() / "AppData/Local/Temp" / f"ab-{tag}.json")],
        capture_output=True, timeout=1800, cwd=SIMD_CWD)
    for ln in r.stdout.decode("utf-8", "replace").splitlines():
        if ln.startswith("total_ms"):
            mean = float(ln.split("mean=")[1].split()[0])
            med = float(ln.split("median=")[1].split()[0])
            return mean, med
    # 失败时带输出尾巴（偶发设备初始化失败/中途崩溃的现场）
    out = r.stdout.decode("utf-8", "replace").splitlines()[-6:]
    err = r.stderr.decode("utf-8", "replace").splitlines()[-6:]
    raise RuntimeError(f"对方没有输出 total_ms（exit={r.returncode}）\nstdout 尾: {out}\nstderr 尾: {err}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "ours"
    for tier in ("tiny", "small"):
        ratios = []
        print(f"=== {tier} ===", flush=True)
        for r in range(ROUNDS):
            if mode == "ours":
                on = ours(tier)
                off = ours(tier, ("--rec-shards", "0"))
                a, b = off[0], on[0]
                print(f"  r{r+1}: 默认 {b:6.1f}（det {on[1]:5.1f} 行 {on[2]:5.1f}）"
                      f"   关分片 {a:6.1f}（det {off[1]:5.1f} 行 {off[2]:5.1f}）"
                      f"   比值 {b/a:.3f}", flush=True)
            else:
                mean, med = theirs(tier, f"{tier}{r}")
                # AB_OURS_EXTRA：我们一侧的附加参数（如 "--device gpu"）
                o = ours(tier, os.environ.get("AB_OURS_EXTRA", "").split())
                print(f"  r{r+1}: 对方 {mean:6.1f}(中位 {med:6.1f})   "
                      f"我们 {o[0]:6.1f}（det {o[1]:5.1f} 行 {o[2]:5.1f}, {o[3]:.2f} 张/s）"
                      f"   比值 {o[0]/mean:.3f}", flush=True)
                ratios.append(o[0] / mean)
        if ratios:
            print(f"  → 比值中位 {statistics.median(ratios):.3f}   "
                  f"极差 {min(ratios):.2f}–{max(ratios):.2f}", flush=True)


if __name__ == "__main__":
    main()
