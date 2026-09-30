#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整规范对决：竞品 feature/2.0 (vulkan/sharp) vs 我们 (gpu/cpu)。

按对方跑器的协议对齐（Program.cs）：
- 顺序基准：单进程逐图串行、图片预解码（他们）/我们解码单独计时并
  同时报 total 与 total-decode 两个口径；首图只计准确度不计时间；
  准确度用对方 BenchSummary 同算法（score_both.py 复刻）双方同尺。
- 并发多图：对方无单机并发模式（--replica 是 CI 多虚机口径）——
  对称定义为 4 进程 × 25 图（临时目录硬链接，无 metadata=不计分）。
- 内存：外置采样器（ctypes GetProcessMemoryInfo，100ms 轮询）两边同一
  把尺：peak working set + peak private bytes；对方内部 ws 作交叉核对
  （注意其 ws 含预解码语料 ~253MB，见报告注）。
- 交错轮次控机器漂移；每次运行前跑 scaling_probe。

用法（全部工厂默认档）：
  python bench/duel_full.py                 # GPU vs GPU（tiny+small）
  python bench/duel_full.py --cpu           # CPU vs CPU（sharp vs 默认）
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

DS = Path(r"D:\qinwh\code\myself\sku-manager\ocr-tool\bench\simdpaddleocr-dataset-v1\dataset")
OURS = Path(r"D:\qinwh\code\myself\qppocr\target\release\qppocr.exe")
OURS_CWD = Path(r"D:\qinwh\code\myself\qppocr")
THEIR = Path(os.environ.get(
    "AB_SIMD_EXE",
    r"D:\qinwh\code\myself\SimdPaddleOCR-gpu\test\Sdcb.SimdPaddleOCR.Tests\bin\Release\net10.0\Sdcb.SimdPaddleOCR.Tests.exe"))
THEIR_CWD = THEIR.parent
TIERS = ["tiny", "small"]
ROUNDS = 3
CONC_ROUNDS = 2
CONC_PROCS = 4

# ---------------- Windows 进程内存采样（两边同一把尺） ----------------
class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]

_psapi = ctypes.WinDLL("psapi")
_kernel32 = ctypes.WinDLL("kernel32")
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010


def sample_peaks(pids, procs=None):
    """轮询一组 pid 直至全部退出，返回 (peak_ws_MB, peak_private_MB)。

    存活判定用 Popen.poll()——对已退出但句柄未关的进程，
    GetProcessMemoryInfo 依然成功（返回陈旧值），拿它判活会死循环到
    超时（实测踩过）。退出后补读一次峰值（句柄仍有效）再收尾。
    """
    peak_ws = peak_priv = 0
    handles = []
    for pid in pids:
        h = _kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
        if h:
            handles.append(h)

    def read_all():
        nonlocal peak_ws, peak_priv
        for h in handles:
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            if _psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                peak_ws = max(peak_ws, pmc.PeakWorkingSetSize)
                peak_priv = max(peak_priv, pmc.PrivateUsage)

    deadline = time.time() + 3600
    while time.time() < deadline:
        read_all()
        if procs is not None:
            if all(p.poll() is not None for p in procs):
                break
        else:
            # 无 Popen 句柄时用 WaitForSingleObject 兜底
            if all(_kernel32.WaitForSingleObject(h, 0) != 0 for h in handles) and handles:
                break
        time.sleep(0.1)
    read_all()  # 退出后补读（含最后一段的峰值）
    for h in handles:
        _kernel32.CloseHandle(h)
    return peak_ws / 1048576, peak_priv / 1048576


def run_sampled(cmd, cwd, timeout=3600):
    """起进程 + 采样；返回 (stdout, stderr, returncode, peak_ws, peak_priv)。"""
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    box = {}

    def poll():
        box["peaks"] = sample_peaks([p.pid], procs=[p])

    t = threading.Thread(target=poll)
    t.start()
    out, err = p.communicate(timeout=timeout)
    t.join()
    return out, err, p.returncode, box["peaks"][0], box["peaks"][1]


# ---------------- 对方跑器 ----------------
def theirs(tier, engine, tag, count=100, warmup=1, indir=None):
    out_json = Path(tempfile.gettempdir()) / f"duel-{tag}.json"
    cmd = [str(THEIR), "--benchmark", "--engine", engine, "--workers", "8",
           "--model", tier, "--input", str(indir or DS), "--count", str(count),
           "--warmup", str(warmup), "--case-id", tag, "--out", str(out_json)]
    out, err, rc, ws, priv = run_sampled(cmd, THEIR_CWD)
    txt = out.decode("utf-8", "replace")
    if rc != 0:
        raise RuntimeError(f"对方 exit={rc}\n{txt[-500:]}\n{err.decode('utf-8','replace')[-500:]}")
    m = re.search(r"total_ms mean=([\d.]+) median=([\d.]+) p95=([\d.]+)", txt)
    acc = re.search(r"accuracy cls=(\d+)/(\d+) exact_lines=(\d+)/(\d+) .*CER=([\d.]+)% char_acc=([\d.]+)%", txt)
    meta_ws = re.search(r'"working_set_mb_peak":\s*([\d.]+)', out_json.read_text(encoding="utf-8"))
    return {
        "mean": float(m.group(1)), "median": float(m.group(2)), "p95": float(m.group(3)),
        "cls": f"{acc.group(1)}/{acc.group(2)}" if acc else "?",
        "exact": int(acc.group(3)) if acc else 0, "total_lines": int(acc.group(4)) if acc else 0,
        "cer": float(acc.group(5)) if acc else 0, "char_acc": float(acc.group(6)) if acc else 0,
        "ws_ext": ws, "priv_ext": priv, "ws_self": float(meta_ws.group(1)) if meta_ws else None,
    }


# ---------------- 我们 ----------------
def ours(tier, device, imgs, tag):
    cmd = [str(OURS), *[str(p) for p in imgs], "--tier", tier, "--json", "--quiet",
           "--bench", "1", "--workers", "1", "--device", device]
    out, err, rc, ws, priv = run_sampled(cmd, OURS_CWD)
    if rc != 0:
        raise RuntimeError(f"我们 exit={rc}\n{err.decode('utf-8','replace')[-500:]}")
    rows = json.loads(out.decode("utf-8", "replace"))
    tot = [r["timing"]["total_ms"] for r in rows][1:]  # 首图只计准确度
    dec = [r["timing"].get("decode_ms", 0.0) for r in rows][1:]
    tot.sort()
    n = len(tot)
    return {
        "mean": statistics.mean(tot), "median": statistics.median(tot), "p95": tot[int(n * 0.95) - 1],
        "mean_no_decode": statistics.mean([t - d for t, d in zip(tot, dec)]),
        "ws_ext": ws, "priv_ext": priv, "rows": rows,
    }


_SCORE_SRC = Path(r"C:\Users\qin_w\AppData\Local\Temp\claude\score_both.py")


def load_scorer():
    ns = {}
    src = _SCORE_SRC.read_text(encoding="utf-8").split("meta = json.load")[0]
    exec(src, ns)
    return ns


def our_accuracy(rows, scorer):
    preds = {}
    for r in rows:
        preds[Path(r["image"]).name] = [
            (l["text"], l["rotation"], scorer["aabb"](l["box"])) for l in r["lines"]]
    meta = json.load(open(DS / "metadata.json", encoding="utf-8"))
    gt = {im["file"]: [(l["text"], int(round(l["cls_degrees"])), l["bbox"])
                       for l in im["lines"]] for im in meta["images"]}
    return scorer["score"](gt, preds)


# ---------------- 并发 ----------------
def chunk_dirs():
    """4 个临时目录 × 25 图（硬链接，须与数据集同盘——跨盘 os.link 报
    WinError 17）；对方无 metadata 时自动不计分。"""
    tmp_root = OURS_CWD / ".duel-tmp"
    tmp_root.mkdir(exist_ok=True)
    dirs = []
    for c in range(CONC_PROCS):
        d = Path(tempfile.mkdtemp(prefix=f"conc-{c}-", dir=tmp_root))
        for i in range(c * 25 + 1, c * 25 + 26):
            src, dst = DS / f"img-{i:03d}.jpg", d / f"img-{i:03d}.jpg"
            if not dst.exists():
                os.link(src, dst)
        dirs.append(d)
    return dirs


def run_concurrent(cmds_pairs, label):
    """cmds_pairs: [(cmd, cwd), ...] 同时起、采全部 pid、墙钟计时。"""
    procs = [subprocess.Popen(c, cwd=w, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL) for c, w in cmds_pairs]
    t0 = time.time()
    ws, priv = sample_peaks([p.pid for p in procs], procs=procs)
    for p in procs:
        p.wait(timeout=3600)
    wall = time.time() - t0
    rcs = [p.returncode for p in procs]
    return {"wall": wall, "imgs_per_s": 100 / wall, "ws_peak_max": ws,
            "priv_sum_hint": priv, "rcs": rcs, "label": label}


def probe():
    r = subprocess.run(["cargo", "run", "-p", "qppocr-kernels", "--release",
                        "--example", "scaling_probe"], cwd=OURS_CWD,
                       capture_output=True, timeout=600)
    ln = [l for l in r.stdout.decode("utf-8", "replace").splitlines() if l.strip().startswith("16")][-1]
    return ln.strip()


def main():
    cpu_mode = "--cpu" in sys.argv
    conc_only = "--conc-only" in sys.argv
    their_engine = "sharp" if cpu_mode else "vulkan"
    our_device = "cpu" if cpu_mode else "gpu"
    print(f"模式：他们 --engine {their_engine} vs 我们 --device {our_device}（工厂默认档）")
    print(f"探针(前)：{probe()}\n")
    scorer = load_scorer()
    imgs = [DS / f"img-{i:03d}.jpg" for i in range(1, 101)]

    if not conc_only:
      for tier in TIERS:
        print(f"=== {tier} 顺序（{ROUNDS} 轮交错） ===")
        acc_done = False
        for rnd in range(1, ROUNDS + 1):
            th = theirs(tier, their_engine, f"{tier}{their_engine}{rnd}")
            ou = ours(tier, our_device, imgs, f"{tier}{our_device}{rnd}")
            if not acc_done:
                acc = our_accuracy(ou["rows"], scorer)
                print(f"  我们准确度：{acc}")
                acc_done = True
            print(f"  r{rnd}: 对方 {th['mean']:7.1f}(p95 {th['p95']:6.1f}) ws峰 {th['ws_ext']:6.0f}MB priv峰 {th['priv_ext']:6.0f}MB"
                  f"   我们 {ou['mean']:7.1f}（免解码 {ou['mean_no_decode']:6.1f}, p95 {ou['p95']:6.1f}）"
                  f" ws峰 {ou['ws_ext']:6.0f}MB priv峰 {ou['priv_ext']:6.0f}MB   比值 {ou['mean']/th['mean']:.3f}")
            if rnd == 1:
                print(f"      对方准确度：exact {th['exact']}/{th['total_lines']} CER {th['cer']}% cls {th['cls']}"
                      f"   对方内部ws峰 {th['ws_self']}MB")

    print(f"\n探针(顺序后)：{probe()}\n")
    dirs = chunk_dirs()
    try:
        for tier in TIERS:
            print(f"=== {tier} 并发（{CONC_PROCS} 进程 × 25 图 × {CONC_ROUNDS} 轮交错） ===")
            for rnd in range(1, CONC_ROUNDS + 1):
                th_cmds = [([str(THEIR), "--benchmark", "--engine", their_engine, "--workers", "8",
                             "--model", tier, "--input", str(d), "--count", "25", "--warmup", "1",
                             "--case-id", f"conc{rnd}{c}", "--out",
                             str(Path(tempfile.gettempdir()) / f"duel-conc-{tier}-{their_engine}-{rnd}-{c}.json")],
                            str(THEIR_CWD)) for c, d in enumerate(dirs)]
                th = run_concurrent(th_cmds, f"{tier} 对方 r{rnd}")
                ou_cmds = []
                for c, d in enumerate(dirs):
                    imgs_c = sorted(d.glob("img-*.jpg"))
                    ou_cmds.append(([str(OURS), *[str(p) for p in imgs_c], "--tier", tier,
                                     "--json", "--quiet", "--bench", "1", "--workers", "1",
                                     "--device", our_device], str(OURS_CWD)))
                ou = run_concurrent(ou_cmds, f"{tier} 我们 r{rnd}")
                print(f"  r{rnd}: 对方 墙钟 {th['wall']:6.1f}s（{th['imgs_per_s']:5.2f} 张/s）"
                      f"单进程ws峰 {th['ws_peak_max']:6.0f}MB   "
                      f"我们 墙钟 {ou['wall']:6.1f}s（{ou['imgs_per_s']:5.2f} 张/s）"
                      f"单进程ws峰 {ou['ws_peak_max']:6.0f}MB   吞吐比 {th['wall']/ou['wall']:.3f}"
                      f"   exit {th['rcs']}/{ou['rcs']}")
    finally:
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
    print(f"\n探针(后)：{probe()}")


if __name__ == "__main__":
    main()
