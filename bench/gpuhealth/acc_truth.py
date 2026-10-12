#!/usr/bin/env python3
"""绝对精度评分：拿生成器真值（gene100/metadata.json）对 OCR 输出打分。

为什么不用 Metal-vs-CPU：那是**一致性**不是**正确性**——两边一起错看不出来，
而且近形替换（O/0、l/I）两个方向都有，比值没有方向。真值才判得了对错。

指标（逐图，再按行汇总）：
  recall  匹配上框的 GT 行 / GT 行数        —— 漏行直接掉这里
  exact   文本完全正确的框 / 匹配框          —— 主指标
  empty   文本为空的框 / 匹配框              —— **「静默丢行」正是这一列**
  cer     字符错误率（编辑距离 / 真值字符数）

配对判据：检测框（8 点四边形）取轴对齐外接矩形，与 GT bbox 算 IoU，
贪心取 ≥ --iou 的最大者，一对一。

⚠️ **真值的 `bbox` 不是行分割真值。** 生成器会把**一条视觉上的整行**拆成两条
相邻条目（一条 orientation=0、一条 180，x 间隙只有 4~13px）。我们把它认成
一行是**对的**，但按条目配对就会记成两条 MISS。gen100 上 small 档的 6 条 MISS
全是这个来源，`img-056` 那条间隙只有 **4px**。⇒ **看到 MISS 先裁图确认**，
别急着改检测。实测 `merge_line_gap` 0.5 vs 0 在整个语料上结果逐字相同，
那条同行合并逻辑在这份语料上根本没触发。

用法：
  ./acc_truth.py --tier small --device metal
  ./acc_truth.py --tier small --device cpu --limit 20
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def quad_aabb(q):
    xs, ys = q[0::2], q[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


def edit(a, b):
    """Levenshtein，O(len(a)*len(b))；行很短，够用。"""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="small")
    ap.add_argument("--device", default="metal")
    ap.add_argument("--dir", default=os.path.join(HERE, "gen100"))
    ap.add_argument("--meta", default=None, help="默认 <dir>/metadata.json")
    ap.add_argument("--bin", default=os.path.join(ROOT, "target/release/qppocr"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--strip-space", action="store_true",
                    help="两边都去掉半角空格再比（字典里没有空格字符，"
                         "空格是 `rec_space_gap` 按几何补的启发式 ⇒ 报精度"
                         "必须声明算不算，见 [[head2head-accuracy-space-is-the-whole-lead]]）")
    ap.add_argument("--dump", default=None, help="把漏行/空行明细写到这个文件")
    ap.add_argument("--from-json", default=None,
                    help="不跑引擎，直接吃一份 `qppocr *.jpg --json` 的落盘结果"
                         "（按文件名对齐）。交付文档要「一次 run 同时给分阶段"
                         "计时和精度」，而本脚本原来每图起一个进程、只留 lines、"
                         "丢掉 timing ⇒ 不给这个口就得把语料跑两遍")
    a = ap.parse_args()
    meta_path = a.meta or os.path.join(a.dir, "metadata.json")
    meta = json.load(open(meta_path))
    imgs = meta["images"][: a.limit or None]

    env = dict(os.environ)
    tot = dict(gt=0, det=0, m=0, exact=0, empty=0, cer_num=0, cer_den=0)
    bad_imgs = []
    detail = []
    def norm(s):
        return s.replace(" ", "") if a.strip_space else s

    cached = None
    if a.from_json:
        cached = {os.path.basename(o["image"]): o["lines"]
                  for o in json.load(open(a.from_json))}
    for e in imgs:
        p = os.path.join(a.dir, e["file"])
        if cached is not None:
            det = cached.get(e["file"])
            if det is None:
                print(f"✗ {e['file']} 不在 {a.from_json} 里", file=sys.stderr)
                continue
        else:
            try:
                out = subprocess.run(
                    [a.bin, p, "--tier", a.tier, "--device", a.device,
                     "--models", a.models, "--workers", "1", "--quiet", "--json"],
                    capture_output=True, text=True, env=env, timeout=600,
                )
                det = json.loads(out.stdout)[0]["lines"]
            except Exception as ex:                                 # noqa: BLE001
                print(f"✗ {e['file']} 跑失败：{ex}", file=sys.stderr)
                continue

        gt = e["lines"]
        dboxes = [(quad_aabb(l["box"]), l["text"]) for l in det]
        used = [False] * len(dboxes)
        img = dict(gt=len(gt), det=len(det), m=0, exact=0, empty=0, dist=0, gtch=0)
        # 按 orientation 分桶统计：竖排（±90°）和横排分开看，否则一锅粥
        buckets = {}
        for gi, g in enumerate(gt):
            gb = tuple(g["bbox"])
            ori = g.get("orientation_degrees", 0)
            b = buckets.setdefault(ori, dict(n=0, exact=0))
            b["n"] += 1
            best, bi = 0.0, -1
            for i, (db, _) in enumerate(dboxes):
                if used[i]:
                    continue
                v = iou(gb, db)
                if v > best:
                    best, bi = v, i
            if bi < 0 or best < a.iou:
                detail.append((e["file"], "MISS", g["text"], "", str(ori), str(gi)))
                continue
            used[bi] = True
            t = dboxes[bi][1]
            tg, gg = norm(t), norm(g["text"])
            img["m"] += 1
            img["gtch"] += len(gg)
            if tg == gg:
                img["exact"] += 1
                b["exact"] += 1
            else:
                img["dist"] += edit(tg, gg)
                if t == "":
                    img["empty"] += 1
                    detail.append((e["file"], "EMPTY", g["text"], "", str(ori), str(gi)))
                else:
                    detail.append((e["file"], "SUBST", g["text"], t, str(ori), str(gi)))
        for ori, b in buckets.items():
            tb = tot.setdefault(f"ori{ori}", dict(n=0, exact=0))
            tb["n"] += b["n"]
            tb["exact"] += b["exact"]
        for k in ("gt", "det", "m", "exact", "empty"):
            tot[k] += img[k]
        tot["cer_num"] += img["dist"]
        tot["cer_den"] += img["gtch"]
        if img["exact"] < img["m"]:
            bad_imgs.append((img["m"] - img["exact"], e["file"], img))

    cal = "去空格" if a.strip_space else "原样（含启发式补的空格）"
    print(f"== 绝对精度：tier={a.tier} device={a.device} 图={len(imgs)} 口径={cal} ==")
    print(f"GT 行 {tot['gt']} | 检出 {tot['det']} | 匹配 {tot['m']} "
          f"({tot['m']/max(tot['gt'],1):.1%})")
    print(f"文本完全正确 {tot['exact']}/{tot['m']} = {tot['exact']/max(tot['m'],1):.2%}")
    print(f"**空文本 {tot['empty']}/{tot['m']} = {tot['empty']/max(tot['m'],1):.2%}**")
    print(f"CER = {tot['cer_num']}/{tot['cer_den']} = {tot['cer_num']/max(tot['cer_den'],1):.2%}")
    print(f"有错的图 {len(bad_imgs)}/{len(imgs)}")
    for k in sorted(k for k in tot if k.startswith("ori")):
        b = tot[k]
        print(f"  orientation={k[3:]:>4}°  {b['exact']}/{b['n']} = {b['exact']/max(b['n'],1):.2%}")
    if a.dump:
        with open(a.dump, "w") as f:
            for r in detail:
                f.write("\t".join(r) + "\n")
        print(f"明细：{a.dump}")
        empty = [r for r in detail if r[1] == "EMPTY"]
        if empty:
            print("== 空文本行（GT 有字却读出空） ==")
            for r in empty[:40]:
                print(f"  {r[0]}  GT={r[2]!r}")


if __name__ == "__main__":
    main()
