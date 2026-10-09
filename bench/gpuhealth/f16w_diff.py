#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 bench_f16w 两次输出的 TSV：<基线> <fp16权重版> [标签]。

口径（先定好，避免事后解释）：
  · 行数：逐图相等才算「检测没动」——行数变了是**检测级**事件。
  · 行文本：完全一致 = 该行未受影响。
  · CER：把每图的全部行文本拼成一个串（去空白），对基线算字符编辑距离，
        再除以基线总字符数。这是「整体掉多少字」的口径。
  · conf：同一行（按图+行号对齐）的置信度差。掉得多的地方通常是
        argmax 临界行——正是 fp16 会翻转的位置。
"""
import sys
from collections import defaultdict


def load(p):
    per_img = defaultdict(list)
    for ln in open(p, encoding="utf-8"):
        f = ln.rstrip("\n").split("\t")
        if len(f) == 3 and f[1] == "=":
            continue
        if len(f) < 5:
            continue
        per_img[f[0]].append((int(f[1]), float(f[2]), int(f[3]), f[4]))
    return per_img


def edist(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    base, new = load(sys.argv[1]), load(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else ""

    imgs = sorted(set(base) | set(new))
    n_lines_b = sum(len(v) for v in base.values())
    n_lines_n = sum(len(v) for v in new.values())
    same_lines = diff_lines = 0
    tot_chars = 0
    tot_ed = 0
    linecount_bad = []
    conf_deltas = []
    examples = []

    for img in imgs:
        b, n = base.get(img, []), new.get(img, [])
        if len(b) != len(n):
            linecount_bad.append((img, len(b), len(n)))
        tb = "".join(x[3] for x in b).replace(" ", "")
        tn = "".join(x[3] for x in n).replace(" ", "")
        tot_chars += len(tb)
        tot_ed += edist(tb, tn)
        for (ib, cb, _, xb), (_, cn, _, xn) in zip(b, n):
            if xb == xn:
                same_lines += 1
            else:
                diff_lines += 1
                if len(examples) < 12 and xb != "<空>" and xn != "<空>":
                    examples.append((img, ib, cb, cn, xb, xn))
            conf_deltas.append(cn - cb)

    conf_deltas.sort()
    med = conf_deltas[len(conf_deltas) // 2] if conf_deltas else 0.0
    print(f"=========== {tag} ===========")
    print(f"图数            {len(imgs)}")
    print(f"总行数          {n_lines_b} → {n_lines_n}"
          f"（{'相同' if n_lines_b == n_lines_n else '**不同**'}）")
    if linecount_bad:
        print(f"逐图行数不同    {len(linecount_bad)} 张：{linecount_bad[:6]}")
    else:
        print("逐图行数不同    0 张（检测完全没动）")
    print(f"行文本          {same_lines} 相同 / {diff_lines} 不同"
          f"（不同率 {100.0 * diff_lines / max(same_lines + diff_lines, 1):.2f}%）")
    print(f"CER             {100.0 * tot_ed / max(tot_chars, 1):.3f}%"
          f"（{tot_ed} 字符错 / {tot_chars} 字符）")
    if conf_deltas:
        print(f"置信度差        中位 {med:+.6f}  最小 {conf_deltas[0]:+.6f}"
              f"  最大 {conf_deltas[-1]:+.6f}")
    if examples:
        print("差异样例（图 / 行 / conf 基线→fp16 / 文本）：")
        for img, i, cb, cn, xb, xn in examples:
            print(f"  {img} #{i}  {cb:.4f}→{cn:.4f}")
            print(f"      基线 {xb!r}")
            print(f"      fp16 {xn!r}")


if __name__ == "__main__":
    main()
