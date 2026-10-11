#!/usr/bin/env python3
"""同真值、双引擎的**精度**对拍（此前只对拍过吞吐）。

竞品的 harness 输出每个图一个 `texts` 数组但**不带框**，所以两边统一按
**文本多重集**打分，而不是 acc_truth.py 的 IoU 配对——这是唯一公平的口径。

两个判据都要看，因为它们回答不同的问题：

  exact     输出里有一条**整串等于**该 GT 行      —— 严口径
  contained 某条输出**包含**该 GT 行为子串        —— 宽口径，给「两条并成一条」计分

两者之差 = 被并进别的行里的 GT 行数。只看 exact 会把「并框」记成丢字，
只看 contained 会把「一整行胡诌里恰好含这几个字」记成命中。

真值：gen100/metadata.json（见 acc_truth.py 里为什么是 gen100 不是 corpus100）。

用法：
  ./acc_head2head.py --comp ~/Projects/SimdPaddleOCR/bench-out/gentruth-small.json \
                     --comp-name 竞品 --tier small --ours metal
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def score(gt_texts, out_texts):
    """返回 (exact命中, contained命中, 输出条数)。一一对应，贪心。"""
    pool = list(out_texts)
    ex = 0
    for g in gt_texts:
        for i, t in enumerate(pool):
            if t == g:
                ex += 1
                pool.pop(i)
                break
    cov = 0
    rest = list(out_texts)
    for g in gt_texts:
        if not g:
            continue
        if any(g in t for t in rest):
            cov += 1
    return ex, cov, len(out_texts)


def ours_texts(binp, models, tier, device, d, files, limit=0):
    out = {}
    for f in (files[:limit] if limit else files):
        p = os.path.join(d, f)
        try:
            r = subprocess.run(
                [binp, p, "--tier", tier, "--device", device, "--models", models,
                 "--workers", "1", "--quiet", "--json"],
                capture_output=True, text=True, timeout=600,
            )
            out[f] = [l["text"] for l in json.loads(r.stdout)[0]["lines"]]
        except Exception as ex:                                     # noqa: BLE001
            print(f"✗ {f}: {ex}", file=sys.stderr)
            out[f] = None
    return out


def report(name, per_image, meta):
    gt = {e["file"]: [l["text"] for l in e["lines"]] for e in meta["images"]}
    ngt = ne = nc = nout = nimg = 0
    worst = []
    for f, g in gt.items():
        o = per_image.get(f)
        if o is None:
            continue
        nimg += 1
        ex, cov, k = score(g, o)
        ngt += len(g); ne += ex; nc += cov; nout += k
        if ex < len(g):
            worst.append((len(g) - ex, f, len(g) - cov))
    print(f"  {name:<8} 图 {nimg}  GT行 {ngt}  输出行 {nout}  "
          f"exact {ne}/{ngt} = {ne/ngt:.2%}   contained {nc}/{ngt} = {nc/ngt:.2%}")
    worst.sort(reverse=True)
    return ne, nc, ngt


def decompose(ours, comp, meta):
    """按 GT 行归属四类，再**按去重后的 GT 字符串**归并——否则同一句话重复几十遍
    会把分辨率伪装成几十个百分点。"""
    gt = {e["file"]: [l["text"] for l in e["lines"]] for e in meta["images"]}
    both = mine = theirs = neither = 0
    by_str = {}
    for f, g in gt.items():
        o, c = ours.get(f), comp.get(f)
        if o is None or c is None:
            continue
        os_, cs = list(o), list(c)
        for t in g:
            mo = t in os_
            mc = t in cs
            if mo:
                os_.remove(t)
            if mc:
                cs.remove(t)
            k = ("两边都对" if mo and mc else
                 "**只有我们对**" if mo else
                 "只有竞品对" if mc else "两边都错")
            d = by_str.setdefault(t, dict(uniq=1, n=0, right=0, comp=0))
            d["n"] += 1
            if mo:
                d["right"] += 1
            if mc:
                d["comp"] += 1
            if mo and mc:
                both += 1
            elif mo:
                mine += 1
            elif mc:
                theirs += 1
            else:
                neither += 1
    print(f"  逐行归属：两边都对 {both} | **只有我们对 {mine}** | "
          f"只有竞品对 {theirs} | 两边都错 {neither}")
    risky = [(v["n"], t, v) for t, v in by_str.items() if v["right"] != v["comp"]]
    risky.sort(reverse=True)
    tot = sum(n for n, _, _ in risky)
    print(f"  有分歧的 GT 字符串 {len(risky)} 种（覆盖 {tot} 行）；"
          f"按**去重字符串**看：")
    for n, t, v in risky[:12]:
        tag = "我们对" if v["right"] > v["comp"] else "竞品对"
        print(f"    {n:>3}行 我们{v['right']:>3}/{n} 竞品{v['comp']:>3}/{n} "
              f"[{tag}] {t[:46]!r}")
    return both, mine, theirs, neither


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="small")
    ap.add_argument("--ours", default="metal", help="我们的 device")
    ap.add_argument("--comp", required=True, help="竞品 harness 的 --out JSON")
    ap.add_argument("--dir", default=os.path.join(HERE, "gen100"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--dump-ours", default=None,
                    help="把我们的逐图文本存成 JSON（存了就无需重跑 GPU 也能做归因）")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.dir, "metadata.json")))
    files = [e["file"] for e in meta["images"]]
    print(f"== 精度对拍 tier={a.tier} 图={len(files)} ==")

    cd = json.load(open(a.comp))
    comp = {r["file"]: (r.get("texts") or []) for r in cd["rows"]}
    report("竞品", comp, meta)

    ours = None
    if a.dump_ours and os.path.exists(a.dump_ours):
        ours = json.load(open(a.dump_ours))
    if ours is None and not a.skip_ours:
        ours = ours_texts(os.path.join(ROOT, "target/release/qppocr"),
                          os.path.join(ROOT, "models"), a.tier, a.ours, a.dir,
                          files, a.limit)
        if a.dump_ours:
            with open(a.dump_ours, "w") as f:
                json.dump(ours, f, ensure_ascii=False)
            print(f"  我们的逐图文本 -> {a.dump_ours}")
    if ours is not None:
        report(f"我们({a.ours})", ours, meta)
        decompose(ours, comp, meta)


if __name__ == "__main__":
    main()
