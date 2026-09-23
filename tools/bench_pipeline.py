#!/usr/bin/env python3
"""交错基准：准确率 / 单图速度 / 并发吞吐 / 内存峰值（Rust vs C++）。

方法论（机器波动 ±27%，migration-bench 的结论照搬）：
  - **交错**：每轮 rust、cpp 交替先后（r 奇数 rust 先，偶数 cpp 先）
  - **多轮**：R 轮，报告最好值与中位数
  - warmup：第一轮丢弃（模型加载、冷页）

用法：python tools/bench_pipeline.py [--rounds 5] [--quick]
"""
import argparse
import json
import subprocess
import sys
import time
import re
import threading

sys.stdout.reconfigure(encoding="utf-8")

RUST_EXE = r"D:\qinwh\code\myself\qppocr\target\release\qppocr.exe"
CPP_EXE = r"D:\qinwh\code\myself\ocr-demo\q-lite-ocr-cli.exe"
TESTDATA = r"D:\qinwh\code\myself\ocr-demo\testdata"
CASES = json.load(open(f"{TESTDATA}/cases.json", encoding="utf-8"))

# --tier 映射：Rust 跑上游原件，C++ 跑转换版（C++ 跑不了上游：
# cls 段错误 + 无内嵌字典）；两套模型语义相同（CROSSCHECK.md §4.4）。
TIERS = {
    "tiny": (["--tier", "tiny"],
             ["PP-OCRv6_det_tiny.onnx", "PP-OCRv6_rec_tiny.onnx"]),
    "small": (["--tier", "small"],
              ["PP-OCRv6_det_small.onnx", "PP-OCRv6_rec_small.onnx"]),
}
TIER = "tiny"


def engine_args():
    rust_args, (det, rec) = TIERS[TIER]
    cpp_args = ["--det", rf"D:\qinwh\code\myself\ocr-demo\models\{det}",
                "--rec", rf"D:\qinwh\code\myself\ocr-demo\models\{rec}",
                "--cls", r"D:\qinwh\code\myself\ocr-demo\models\ppocr_cls.onnx"]
    return rust_args, cpp_args


def run_json(exe, extra, img, bench=1):
    """跑一次，返回 (行列表, 单图耗时 ms)。"""
    cmd = [exe, img, "--json", "--quiet", "--bench", str(bench)] + extra
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    raw = r.stdout
    try:
        txt = raw.decode("utf-8")
        # C++ 控制台可能输出 GBK 夹 UTF-8 的 JSON——逐段修复
        if "\ufffd" in txt:
            txt = raw.decode("gbk", errors="replace")
    except UnicodeDecodeError:
        txt = raw.decode("gbk", errors="replace")
    lines = re.findall(r'"text":\s*"((?:[^"\\]|\\.)*)"', txt)
    ms = re.search(r'"total_ms":\s*([0-9.]+)', txt)
    return lines, (float(ms.group(1)) if ms else None)


# ---------------------------------------------------------------- 准确率
def lev(a, b):
    """Levenshtein（字符级，用于 CER）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def accuracy(lines_list):
    """对 cases.json 的 ground truth：exact 行率 + CER。"""
    exact = 0
    total = 0
    err = 0
    ref_len = 0
    for case, lines in zip(CASES, lines_list):
        exp = case["expected_lines"]
        total += len(exp)
        for i, e in enumerate(exp):
            p = lines[i] if i < len(lines) else ""
            if p == e:
                exact += 1
            err += lev(p, e)
            ref_len += len(e)
        # 预测多出的行算进错误
        for p in lines[len(exp):]:
            err += len(p)
    return exact / total, err / ref_len


# ---------------------------------------------------------------- 内存
def peak_ws(exe, extra, img):
    """轮询采样峰值工作集（进程跑 --bench 12 期间）。"""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    psapi = ctypes.WinDLL("psapi")
    kernel32 = ctypes.WinDLL("kernel32")
    cmd = [exe, img, "--quiet", "--bench", "12"] + extra
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = 0
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    while p.poll() is None:
        h = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, p.pid)
        if h:
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            if psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                peak = max(peak, pmc.PeakWorkingSetSize)
            kernel32.CloseHandle(h)
        time.sleep(0.015)
    return peak


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="只跑 3 轮 + 跳过并发")
    ap.add_argument("--tier", choices=list(TIERS), default="tiny")
    args = ap.parse_args()
    global TIER, RUST_ARGS, CPP_ARGS
    TIER = args.tier
    RUST_ARGS, CPP_ARGS = engine_args()
    R = 3 if args.quick else args.rounds

    imgs = [f"{TESTDATA}/{c['file'].split('/')[-1]}" for c in CASES]

    # ---------- 准确率（确定性，各跑一遍；跑两遍确认稳定）
    print("=" * 72)
    print("准确率（cases.json ground truth，7 图）")
    print("=" * 72)
    for name, exe, extra in [("rust", RUST_EXE, RUST_ARGS), ("cpp", CPP_EXE, CPP_ARGS)]:
        lines_list = []
        for img in imgs:
            lines, _ = run_json(exe, extra, img)
            lines_list.append(lines)
        ex, cer = accuracy(lines_list)
        # 稳定性复跑
        lines2 = []
        for img in imgs:
            lines, _ = run_json(exe, extra, img)
            lines2.append(lines)
        ex2, cer2 = accuracy(lines2)
        stab = "稳定" if (ex == ex2 and abs(cer - cer2) < 1e-9) else "!! 两遍不同"
        print(f"  {name:4}  exact {ex:6.1%}   CER {cer:6.2%}   ({stab})")

    # ---------- 单图速度（交错）
    print()
    print("=" * 72)
    print(f"单图速度（交错 {R} 轮 + 1 warmup，每图 bench=3 取最好）")
    print("=" * 72)
    # times[name][img] = [每轮最好 ms]
    times = {"rust": {i: [] for i in imgs}, "cpp": {i: [] for i in imgs}}
    for r in range(R + 1):
        order = ["rust", "cpp"] if r % 2 == 0 else ["cpp", "rust"]
        for name in order:
            exe, extra = (RUST_EXE, RUST_ARGS) if name == "rust" else (CPP_EXE, CPP_ARGS)
            for img in imgs:
                _, ms = run_json(exe, extra, img, bench=3)
                if ms and r > 0:  # r=0 是 warmup
                    times[name][img].append(ms)
    stats = {}
    for name in ["rust", "cpp"]:
        bests, meds = [], []
        per_img = []
        for img in imgs:
            v = sorted(times[name][img])
            bests.append(v[0])
            meds.append(v[len(v) // 2])
            per_img.append((img.split("/")[-1], v[0], v[len(v) // 2]))
        stats[name] = (sum(bests), sum(meds))
        print(f"\n  {name}（全部 {len(imgs)} 图合计）：")
        print(f"    Σ最好 {stats[name][0]:8.1f} ms   Σ中位 {stats[name][1]:8.1f} ms")
        for fn, b, m in per_img:
            print(f"      {fn:28} best {b:7.1f}  med {m:7.1f}")
    rb, rm = stats["rust"]
    cb, cm = stats["cpp"]
    print(f"\n  合计对比：rust/cpp（最好）= {rb/cb:.3f}x   （中位）= {rm/cm:.3f}x")

    if args.quick:
        return

    # ---------- 并发吞吐（多进程）
    print()
    print("=" * 72)
    print("并发吞吐（K 进程同时各跑全部 7 图，交错 3 轮）")
    print("=" * 72)
    for K in (2, 4):
        res = {"rust": [], "cpp": []}
        for r in range(3):
            order = ["rust", "cpp"] if r % 2 == 0 else ["cpp", "rust"]
            for name in order:
                exe, extra = (RUST_EXE, RUST_ARGS) if name == "rust" else (CPP_EXE, CPP_ARGS)
                t0 = time.perf_counter()
                procs = []
                for k in range(K):
                    # 每进程独立跑全部图（简化：单图 --bench 1 顺序）
                    cmd = [exe] + imgs + ["--quiet"] + extra
                    procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                                  stderr=subprocess.DEVNULL))
                for p in procs:
                    p.wait()
                wall = time.perf_counter() - t0
                res[name].append(wall * 1000)
        for name in ["rust", "cpp"]:
            v = sorted(res[name])
            print(f"  K={K}  {name:4}  best {v[0]:8.1f} ms  med {v[1]:8.1f} ms"
                  f"   吞吐 {K * len(imgs) / (v[1] / 1000):5.2f} img/s")
        rb2 = sorted(res["rust"])[1]
        cb2 = sorted(res["cpp"])[1]
        print(f"  K={K}  rust/cpp（中位）= {rb2/cb2:.3f}x")

    # ---------- 内存
    print()
    print("=" * 72)
    print("内存峰值（receipt.png --bench 12，轮询采样 15ms）")
    print("=" * 72)
    for name, exe, extra in [("rust", RUST_EXE, RUST_ARGS), ("cpp", CPP_EXE, CPP_ARGS)]:
        ws = peak_ws(exe, extra, f"{TESTDATA}/receipt.png")
        print(f"  {name:4}  PeakWorkingSet {ws / 1048576:7.1f} MB")
    # 大图再测一张
    for name, exe, extra in [("rust", RUST_EXE, RUST_ARGS), ("cpp", CPP_EXE, CPP_ARGS)]:
        ws = peak_ws(exe, extra, f"{TESTDATA}/big.png")
        print(f"  {name:4}  big.png PeakWS {ws / 1048576:7.1f} MB")


if __name__ == "__main__":
    main()
