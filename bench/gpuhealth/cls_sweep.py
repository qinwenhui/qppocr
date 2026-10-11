#!/usr/bin/env python3
"""cls 阈值的**离线扫描**：跑一遍，把所有 (p0, p1) 捞出来，之后扫门不花 GPU。

背景（img-028 单图 debug，CPU 档）：

    [cls] 行0 裁885x36 (24.58:1) p0=0.0044 p1=0.9956 判定=true   ← GT 180，翻对了
    [cls] 行3 裁816x34 (24.00:1) p0=0.2925 p1=0.7075 判定=false  ← GT 180，**没翻**
    [cls] 行9 裁847x42 (20.17:1) p0=0.3124 p1=0.6876 判定=false  ← GT 180，**没翻**

分类器**倾向是对的**（p1 > p0 两行都是），只是没跨过 `cls_thresh = 0.9` 的硬门
⇒ 倒置裁切原样喂进 rec，吐出 `'n ono nnd dern n n ne n2'`。这些正是
`acc_head2head.py` 去空格口径下竞品唯一赢的那些行。

**关键**：cls 的输入是每行自己的裁切，决策**不影响**别的行、也不影响 det
⇒ 一次 run（门固定 0.9）捞到的 (p0,p1) 对**任意**门都有效。所以扫门在
Python 里做，一次 GPU run 换一整条扫描曲线。

    ./cls_sweep.py --tier small --device metal --dump logs/cls_sweep/small.tsv
    ./cls_sweep.py --tier small --again logs/cls_sweep/small.tsv   # 只重扫，不跑 GPU
"""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# 逐字对齐 engine.rs 里 QPPOCR_CLS_DEBUG 的 eprintln 格式。
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


def collect(a):
    """跑语料，按图 dump 一张 TSV：file/det_idx/w/h/p0/p1/gt方向/iou/我们的文本。"""
    meta = json.load(open(os.path.join(a.dir, "metadata.json")))
    imgs = meta["images"][: a.limit or None]
    rows = []
    for e in imgs:
        p = os.path.join(a.dir, e["file"])
        env = dict(os.environ, QPPOCR_CLS_DEBUG="1")
        try:
            r = subprocess.run(
                [a.bin, p, "--tier", a.tier, "--device", a.device,
                 "--models", a.models, "--workers", "1", "--quiet", "--json"],
                capture_output=True, text=True, timeout=900, env=env,
            )
            det = json.loads(r.stdout)[0]["lines"]
        except Exception as ex:                                     # noqa: BLE001
            print(f"✗ {e['file']} 跑失败：{ex}", file=sys.stderr)
            continue
        dbg = {}
        for ln in r.stderr.splitlines():
            m = LINE_RE.search(ln)
            if m:
                dbg[int(m.group(1))] = (float(m.group(5)), float(m.group(6)),
                                        int(m.group(2)), int(m.group(3)))
        if len(dbg) != len(det):
            print(f"⚠ {e['file']}：debug {len(dbg)} 行 vs det {len(det)} 行", file=sys.stderr)
        # 拿 det 框跟 GT 框 IoU 贪心配对，取 GT 的 orientation_degrees
        dboxes = [quad_aabb(l["box"]) for l in det]
        used = [False] * len(dboxes)
        gtori = {}
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
                gtori[bi] = (int(g.get("orientation_degrees", 0)), best)
        for i, l in enumerate(det):
            if i not in dbg:
                continue
            p0, p1, w, h = dbg[i]
            go, v = gtori.get(i, (None, 0.0))
            rows.append((e["file"], i, w, h, p0, p1,
                         "" if go is None else str(go), f"{v:.3f}",
                         int(l.get("rotation", 0)), (l["text"] or "").replace("\t", " ")))
    return rows


def sweep(rows):
    """按 GT 方向分档，扫门：翻对了多少 / 翻错了多少。"""
    # 只统计配上了 GT 的行；GT 方向 ∈ {0, 180} 才是二分类的有效样本
    lab = [(r[4], r[5], int(r[6]), r[8]) for r in rows if r[6] != ""]
    print(f"== cls 阈值扫描（{len(rows)} 个 det 行，其中 {len(lab)} 行配上了 GT）==")
    print(f"{'门':>6} {'GT180翻对':>10} {'GT180漏':>8} {'GT0翻错':>8} "
          f"{'GT0正确':>8} {'总对':>7} {'准确率':>8}")
    for t in [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95]:
        # 判定 = p1 > p0 && p1 >= t；“翻”= 判定为 true
        n180 = h180 = n0 = h0 = 0
        for p0, p1, go, _ in lab:
            flip = p1 > p0 and p1 >= t
            if go % 360 == 180:
                n180 += 1
                h180 += 1 if flip else 0
            elif go % 360 == 0:
                n0 += 1
                h0 += 1 if not flip else 0
        tot, hit = n180 + n0, h180 + h0
        print(f"{t:>6.2f} {h180:>10} {n180-h180:>8} {n0-h0:>8} "
              f"{h0:>8} {hit:>7} {hit/max(tot,1):>8.2%}")
    # 边缘带：到底有多少行落在 0.5~0.9 之间，落在哪边
    marg = [(r[0], r[1], r[4], r[5], r[6]) for r in rows
            if r[6] != "" and 0.5 <= r[5] < 0.9]
    print(f"\n边缘带 p1∈[0.5,0.9)：{len(marg)} 行（这些就是「降门」会动的行）")
    for f, i, p0, p1, go in marg[:40]:
        print(f"  {f} 行{i:>2} GT={go:>4} p0={p0:.4f} p1={p1:.4f} "
              f"{'⇒ 降门会翻' if float(p1) > float(p0) else '⇒ 倾向本来就是 0'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="small")
    ap.add_argument("--device", default="metal")
    ap.add_argument("--dir", default=os.path.join(HERE, "gen100"))
    ap.add_argument("--bin", default=os.path.join(ROOT, "target/release/qppocr"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--dump", default=None, help="把 (p0,p1) 表写到这个 TSV")
    ap.add_argument("--again", default=None,
                    help="跳过 GPU，直接读这份 TSV 重扫（换门不花 GPU）")
    a = ap.parse_args()

    if a.again:
        rows = [tuple(ln.rstrip("\n").split("\t")) for ln in open(a.again)]
    else:
        rows = collect(a)
        if a.dump:
            os.makedirs(os.path.dirname(a.dump), exist_ok=True)
            with open(a.dump, "w") as f:
                for r in rows:
                    f.write("\t".join(str(x) for x in r) + "\n")
            print(f"(p0,p1) 表 {len(rows)} 行 -> {a.dump}")
    sweep(rows)


if __name__ == "__main__":
    main()
