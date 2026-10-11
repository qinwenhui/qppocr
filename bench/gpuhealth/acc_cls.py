#!/usr/bin/env python3
"""文字方向分类（cls）的绝对精度：拿生成器真值的 orientation_degrees 当答案。

发现经过：small 档相对竞品"只有它对"的 13 行里，最长的那些（`The quick brown fox…`
`OpenVINO…`）**不是识别错，是 cls 没翻**——把 180° 的裁切原样喂进 rec，于是吐出
`'n ono nnd dern n n ne n2'` 这样的垃圾。裁出 det 框一看，字是倒的。

判据：det 框（8 点四边形取轴对齐外接矩形）与 GT bbox 算 IoU、贪心一对一、阈值 0.5；
配对上的比较我们输出的 `rotation` 与 GT 的 `orientation_degrees`。

    ./acc_cls.py --tier small --device metal
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="small")
    ap.add_argument("--device", default="metal")
    ap.add_argument("--dir", default=os.path.join(HERE, "gen100"))
    ap.add_argument("--bin", default=os.path.join(ROOT, "target/release/qppocr"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--dump", default=None, help="把错的明细写到这个文件")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.dir, "metadata.json")))
    imgs = meta["images"][: a.limit or None]
    # 按真值方向分桶统计
    tot = {}
    detail = []
    for e in imgs:
        p = os.path.join(a.dir, e["file"])
        try:
            out = subprocess.run(
                [a.bin, p, "--tier", a.tier, "--device", a.device,
                 "--models", a.models, "--workers", "1", "--quiet", "--json"],
                capture_output=True, text=True, timeout=600,
            )
            det = json.loads(out.stdout)[0]["lines"]
        except Exception as ex:                                     # noqa: BLE001
            print(f"✗ {e['file']} 跑失败：{ex}", file=sys.stderr)
            continue
        dboxes = [(quad_aabb(l["box"]), int(l.get("rotation", 0)), l["text"]) for l in det]
        used = [False] * len(dboxes)
        for g in e["lines"]:
            ori = int(g.get("orientation_degrees", 0))
            b = tot.setdefault(ori, dict(n=0, hit=0))
            b["n"] += 1
            best, bi = 0.0, -1
            for i, (db, _, _) in enumerate(dboxes):
                if used[i]:
                    continue
                v = iou(tuple(g["bbox"]), db)
                if v > best:
                    best, bi = v, i
            if bi < 0 or best < a.iou:
                detail.append((e["file"], "MISS", str(ori), "?", g["text"]))
                continue
            used[bi] = True
            got = dboxes[bi][1]
            # 180 与 -180 等价
            same = (got % 360) == (ori % 360)
            if same:
                b["hit"] += 1
            else:
                detail.append((e["file"], "WRONG", str(ori), str(got),
                               dboxes[bi][2] or "<空>"))

    print(f"== 方向分类绝对精度：tier={a.tier} device={a.device} 图={len(imgs)} ==")
    tn = th = 0
    for ori in sorted(tot):
        b = tot[ori]
        tn += b["n"]
        th += b["hit"]
        print(f"  GT orientation={ori:>4}°  {b['hit']:>4}/{b['n']:<4} = {b['hit']/max(b['n'],1):.2%}")
    print(f"  合计 {th}/{tn} = {th/max(tn,1):.2%}")
    if a.dump:
        with open(a.dump, "w") as f:
            for r in detail:
                f.write("\t".join(r) + "\n")
        print(f"明细 {len(detail)} 条 -> {a.dump}")


if __name__ == "__main__":
    main()
