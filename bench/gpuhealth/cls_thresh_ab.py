#!/usr/bin/env python3
"""取 `cls_sweep.py` 扫出来的**受影响行**，直接拿真值文本裁决门该定在哪。

决策级的表（`cls_sweep.py`）只能数「翻对/翻错」，数不了两件事：
  ① GT=±90 的行**被排除在表外**，却有 5 行落在同一段 [0.65,0.90) 里 ——
     若竖排裁切本来就被 det 框角度摆正了，降门会把它们翻倒；
  ② 「翻对」只说明方向和真值一致，不说明 rec 因此读出了对的文本。

所以这里对**每一条会被门改动到的行**，在两个门下各跑一次，把两边的输出文本
跟 `metadata.json` 的 `text` 摆在一起，逐行看谁对。

    ./cls_thresh_ab.py --tier small --lo 0.65 --hi 0.90
"""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LINE_RE = re.compile(
    r"\[cls\] 行(\d+) 裁(\d+)x(\d+) \(([\d.]+):1\) p0=([\d.]+) p1=([\d.]+) 判定=(\w+)"
)


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def quad_aabb(q):
    xs, ys = q[0::2], q[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


def _f(c):
    return "  n/a " if c is None else f"{c:.4f}"


def _flip_bad(c_hi, c_lo, verdict):
    """「取置信度高的一边」这条规则，在这一行上会不会选错？"""
    if c_hi is None or c_lo is None or verdict in ("同", "都错"):
        return False
    want_hi = verdict == "hi 对"
    return (c_hi >= c_lo) != want_hi


def run(a, path, thresh, extra=None):
    env = dict(os.environ, QPPOCR_CLS_DEBUG="1", QPPOCR_CLS_THRESH=str(thresh))
    env.update(extra or {})
    r = subprocess.run(
        [a.bin, path, "--tier", a.tier, "--device", a.device,
         "--models", a.models, "--workers", "1", "--quiet", "--json"],
        capture_output=True, text=True, timeout=900, env=env,
    )
    det = json.loads(r.stdout)[0]["lines"]
    dbg = {}
    for ln in r.stderr.splitlines():
        m = LINE_RE.search(ln)
        if m:
            dbg[int(m.group(1))] = (float(m.group(5)), float(m.group(6)),
                                    int(m.group(2)), int(m.group(3)),
                                    m.group(7) == "true")
    return det, dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="small")
    ap.add_argument("--device", default="metal")
    ap.add_argument("--dir", default=os.path.join(HERE, "gen100"))
    ap.add_argument("--bin", default=os.path.join(ROOT, "target/release/qppocr"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--tsv", default=os.path.join(HERE, "logs/cls_sweep/small.tsv"))
    ap.add_argument("--lo", type=float, default=0.65)
    ap.add_argument("--hi", type=float, default=0.90)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--env-a", default=None, action="append",
                    help="A 臂的额外环境变量 K=V（可重复）")
    ap.add_argument("--env-b", default=None, action="append",
                    help="B 臂的额外环境变量 K=V（可重复）")
    ap.add_argument("--all-uncertain", action="store_true",
                    help="不看 p1 落没落在 [lo,hi)，改挑**所有** cls 存疑的行"
                         "（max(p0,p1) < hi）——这是「第二意见」真正会碰到的集合")
    a = ap.parse_args()

    def kv(lst):
        d = {}
        for s in lst or []:
            k, _, v = s.partition("=")
            d[k] = v
        return d

    env_a, env_b = kv(a.env_a), kv(a.env_b)

    # 从扫描表里挑出要复查的行
    want = {}
    for ln in open(a.tsv):
        f, i, w, h, p0, p1, go, v, rot, txt = ln.rstrip("\n").split("\t")
        p0, p1 = float(p0), float(p1)
        if a.all_uncertain:
            if max(p0, p1) < a.hi:
                want.setdefault(f, []).append(int(i))
        elif p1 > p0 and a.lo <= p1 < a.hi:
            want.setdefault(f, []).append(int(i))

    meta = {e["file"]: e for e in json.load(open(os.path.join(a.dir, "metadata.json")))["images"]}
    print(f"== A 臂 {env_a} vs B 臂 {env_b}：会改动 {sum(len(v) for v in want.values())} 行 "
          f"/ {len(want)} 张图 ==")
    print("（GT=±90 的行也在内——扫描表数不到它们；A/B 是脚本里的 arm，"
          "默认 A=高门、B=低门）\n")
    tally = {}
    for f in sorted(want):
        e = meta[f]
        p = os.path.join(a.dir, f)
        det_hi, dbg_hi = run(a, p, a.hi, env_a)
        det_lo, dbg_lo = run(a, p, a.lo, env_b)
        dboxes = [quad_aabb(l["box"]) for l in det_hi]
        used = [False] * len(dboxes)
        gt = {}
        for g in e["lines"]:
            best, bi = 0.0, -1
            for i, db in enumerate(dboxes):
                if used[i]:
                    continue
                v = iou(tuple(g["bbox"]), db)
                if v > best:
                    best, bi = v, i
            if bi >= 0 and best >= a.iou:
                used[bi] = True
                gt[bi] = g
        for i in want[f]:
            g = gt.get(i)
            if g is None:
                continue
            ori = int(g.get("orientation_degrees", 0))
            t_hi = (det_hi[i]["text"] or "")
            t_lo = (det_lo[i]["text"] or "")
            c_hi, c_lo = det_hi[i].get("confidence"), det_lo[i].get("confidence")
            if t_hi == t_lo:
                # 判定变了但文本没变（例如那一行本来就没被门挡住）
                verdict = "同"
            elif t_hi == g["text"]:
                verdict = "hi 对" if t_lo != g["text"] else "both 对"
            elif t_lo == g["text"]:
                verdict = "**lo 对**"
            else:
                verdict = "都错"
            key = (ori, verdict)
            tally[key] = tally.get(key, 0) + 1
            print(f"{f} 行{i:>2} GT方向={ori:>4}  裁{dbg_hi[i][2]}x{dbg_hi[i][3]} "
                  f"p1={dbg_hi[i][1]:.4f}  判定 "
                  f"{'翻' if dbg_hi[i][4] else '不翻'}→{'翻' if dbg_lo[i][4] else '不翻'}")
            print(f"    GT  : {g['text']}")
            print(f"    {a.hi}: [{_f(c_hi)}] {t_hi}")
            print(f"    {a.lo}: [{_f(c_lo)}] {t_lo}   ⇒ {verdict}"
                  + ("   ⚠ 置信度判反了" if _flip_bad(c_hi, c_lo, verdict) else ""))
    print("\n== 汇总（GT方向, 判定）==")
    for (ori, verdict), n in sorted(tally.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"  GT={ori:>4}  {verdict:>8}: {n}")


if __name__ == "__main__":
    main()
