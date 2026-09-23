#!/usr/bin/env python3
"""并行度扫描：{rust,cpp} × 线程数 × 并发 K，找延迟/吞吐最优配置。

用法：
  python tools/sweep_parallel.py                       # 单进程线程扫描（默认档）
  python tools/sweep_parallel.py 4,8,16                # 指定线程档
  python tools/sweep_parallel.py 4,8,16 --k 2          # K 进程并发
  python tools/sweep_parallel.py 8 --k 2 --engines cpp # 只扫 C++（LEAN_THREADS）

方法论（docs/BENCH.md §5 与 migration-bench 的教训）：
  - 档位在每轮内**旋转**（round-robin）——升序扫描里最后测的配置
    总是吃漂移亏，单次跑分波动可达 ±27%；
  - 1 轮 warmup 丢弃；单进程模式每图 bench=3 取最好，K 进程模式量墙钟。

跨设备复跑本脚本即可得到该机的最优（K, threads）组合。经验规则
（16 逻辑核实测，供先验）：吞吐最优在总线程 ≈ 1.2~1.5× 逻辑核，
即 threads_per_proc ≈ max(2, round(cores * 1.25 / K))。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

RUST_EXE = r"D:\qinwh\code\myself\qppocr\target\release\qppocr.exe"
CPP_EXE = r"D:\qinwh\code\myself\ocr-demo\q-lite-ocr-cli.exe"
TESTDATA = r"D:\qinwh\code\myself\ocr-demo\testdata"
CASES = json.load(open(f"{TESTDATA}/cases.json", encoding="utf-8"))
IMGS = [f"{TESTDATA}/{c['file'].split('/')[-1]}" for c in CASES][:7]
RUST_ARGS = ["--tier", "tiny"]
CPP_ARGS = [
    "--det", r"D:\qinwh\code\myself\ocr-demo\models\PP-OCRv6_det_tiny.onnx",
    "--rec", r"D:\qinwh\code\myself\ocr-demo\models\PP-OCRv6_rec_tiny.onnx",
    "--cls", r"D:\qinwh\code\myself\ocr-demo\models\ppocr_cls.onnx",
]


def env_for(engine, t):
    if engine == "rust":
        return dict(os.environ, QPPOCR_THREADS=str(t))
    return dict(os.environ, LEAN_THREADS=str(t))


def run_one_ms(engine, t, img):
    """单进程：一次调用（进程内 bench=3 取最好）。"""
    exe, extra = (RUST_EXE, RUST_ARGS) if engine == "rust" else (CPP_EXE, CPP_ARGS)
    r = subprocess.run([exe, img, "--json", "--quiet", "--bench", "3"] + extra,
                       capture_output=True, env=env_for(engine, t), timeout=600)
    m = re.search(r'"total_ms":\s*([0-9.]+)', r.stdout.decode("utf-8", "replace"))
    return float(m.group(1)) if m else None


def run_k_wall_ms(engine, t, k):
    """K 进程并发：同时起 k 个进程各跑全部图，量墙钟。"""
    exe, extra = (RUST_EXE, RUST_ARGS) if engine == "rust" else (CPP_EXE, CPP_ARGS)
    t0 = time.perf_counter()
    procs = [subprocess.Popen([exe] + IMGS + ["--quiet"] + extra, env=env_for(engine, t),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for _ in range(k)]
    for p in procs:
        p.wait()
    return (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("threads", nargs="?", default="4,6,8,10,12,14,16",
                    help="逗号分隔线程档（默认 4..16）")
    ap.add_argument("--k", type=int, default=1, help="并发进程数（默认 1）")
    ap.add_argument("--engines", default="rust", help="rust,cpp 子集（默认 rust）")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    ts = [int(x) for x in args.threads.split(",")]
    engines = args.engines.split(",")
    cfgs = [(e, t) for e in engines for t in ts]

    # warmup 丢弃
    if args.k == 1:
        for img in IMGS:
            run_one_ms(cfgs[0][0], cfgs[0][1], img)
    else:
        run_k_wall_ms(cfgs[0][0], cfgs[0][1], args.k)

    res = {c: [] for c in cfgs}
    for r in range(args.rounds):
        order = cfgs[r % len(cfgs):] + cfgs[:r % len(cfgs)]
        for e, t in order:
            if args.k == 1:
                vals = [run_one_ms(e, t, img) for img in IMGS]
                res[(e, t)].append(sum(v for v in vals if v is not None))
            else:
                res[(e, t)].append(run_k_wall_ms(e, t, args.k))
        print(f"  round {r + 1}/{args.rounds}", file=sys.stderr)

    print(f"K={args.k}  每配置 {args.rounds} 轮（单进程=Σ7图ms，并发=墙钟ms）")
    for (e, t), v in sorted(res.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        v = sorted(v)
        med = v[len(v) // 2]
        tput = args.k * len(IMGS) / (med / 1000)
        print(f"  {e:4} t={t:2d}   best {v[0]:8.1f}  med {med:8.1f}   {tput:5.2f} img/s")


if __name__ == "__main__":
    main()
