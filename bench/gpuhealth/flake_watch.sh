#!/bin/bash
# 逐派发盯窗口：回答「那几十字节在坏轮里到底被哪条派发写过」。
#
# 已钉死的事实（见 logs/flake_2026-10-10.md）：
#   · 同输入、同程序、逐轮输出不同；整个 arena 只有 `[1,3,48,128]` 计划里
#     **两段孤岛**不同 —— 坏轮里它们保持**环境内容**（相邻张量的 24 周期残渣），
#     好轮里是写过的值 ⇒ 坏轮那几十字节**根本没被写过**。
#   · 孤岛 A：激活区相对字 79484..79515（32 字，`gelu.0.0` 尾部 98.5% 处）
#     孤岛 B：170112..170127（16 字，`HardSigmoid.6` 58.8% 处）
#
# `QPPOCR_METAL_WATCH` + `CHUNK=1` 每条派发后把这两个窗口读出来追加到
# `<前缀>.trace`（另有一份 `.pre` = 开跑前的环境内容）。好轮/坏轮各一条，
# **窗口从「环境值」翻成「写过的值」的那条派发**就是本该写它的内核；
# 坏轮里若自始至终都是环境值 ⇒ 那条派发漏写了。
#
# 用法： N=80 IMG=testimgs/img-002.jpg ./flake_watch.sh
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
Q=$ROOT/target/release/qppocr
MODELS=$ROOT/models
N=${N:-80}
IMG=${IMG:-$(cd testimgs && pwd)/img-002.jpg}
WIN=${WIN:-79460:64,170096:48}
OUT=logs/flake_watch_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

[ -x "$Q" ] || { echo "先构建"; exit 1; }
grep -q -a "QPPOCR_METAL_WATCH_WIN" "$Q" || { echo " 探针没编进二进制，先重建"; exit 3; }
./health.sh -q || { echo "健康门禁没过"; exit 2; }

echo "== 蹲翻车（逐派发窗口）：$(basename "$IMG") × ${N}，CHUNK=1 =="
for i in $(seq 1 "$N"); do
    d="$OUT/run$(printf '%03d' "$i")"
    mkdir -p "$d"
    QPPOCR_METAL_CHUNK=1 QPPOCR_METAL_WATCH="$d/w" QPPOCR_METAL_WATCH_WIN="$WIN" \
        "$Q" "$IMG" --tier tiny --device metal --models "$MODELS" \
        --workers 1 --quiet --json >"$d/out.json" 2>"$d/err.txt"
    nu=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[0]["num_unread"])' \
         "$d/out.json" 2>/dev/null || echo "?")
    printf '  run%03d unread=%s\n' "$i" "$nu"
    if [ "$nu" != "0" ] && [ "$nu" != "?" ]; then
        cp "$d/out.json" "$OUT/bad.json"
        echo "$i" >"$OUT/bad_run"
        echo "  ✖ 第 $i 轮翻车 → $d"
        break
    fi
done

BAD=$(cat "$OUT/bad_run" 2>/dev/null || true)
[ -n "$BAD" ] || { echo "  ${N} 轮没蹲到"; exit 1; }
# 找一个「行数与坏轮相同、unread=0、文本与坏轮不同」的好轮当对照。
python3 - "$OUT" "$BAD" <<'PY'
import json, os, sys
out, bad = sys.argv[1], int(sys.argv[2])
b = json.load(open(f"{out}/bad.json"))[0]
bred = [(l["text"], round(l["confidence"], 4)) for l in b["lines"]]
print("\n坏轮 %d 行 unread=%d" % (len(bred), b["num_unread"]))
cands = []
for d in sorted(os.listdir(out)):
    if not d.startswith("run") or int(d[3:]) == bad:
        continue
    try:
        g = json.load(open(f"{out}/{d}/out.json"))[0]
    except Exception:
        continue
    if g["num_unread"] != 0 or len(g["lines"]) != len(bred):
        continue
    gred = [(l["text"], round(l["confidence"], 4)) for l in g["lines"]]
    diff = [k for k in range(len(bred)) if gred[k] != bred[k]]
    cands.append((d, diff, gred))
cands.sort(key=lambda t: (len(t[1]), t[0]))
for d, diff, gred in cands[:3]:
    print("  好轮候选 %s：与坏轮差 %d 行 %s" % (d, len(diff), diff[:6]))
if cands:
    open(f"{out}/good_run", "w").write(cands[0][0])
    print("  → 选 %s 当对照" % cands[0][0])
PY

GOOD=$(cat "$OUT/good_run" 2>/dev/null || true)
[ -n "$GOOD" ] || { echo " 没找到合适的好轮"; exit 1; }

python3 - "$OUT" "$GOOD" "$BAD" <<'PY'
import os, sys
out, good, bad = sys.argv[1], sys.argv[2], sys.argv[3]
gd, bd = f"{out}/{good}", f"{out}/run{bad:03d}"
def parse_line(line):
    """→ (前缀, [窗口1 各字, 窗口2 各字, ...])。行格式：
       `<lo>..<hi> <内核名> w1w1w1 |w2w2w2 |`（窗口以 '|' 分隔，故不能用空格切）。"""
    parts = line.split("|")
    tail = [p for p in parts if p.strip()]
    wins = []
    for i, p in enumerate(tail):
        toks = p.split()
        if i == 0:
            toks = toks[2:]          # 去掉 "lo..hi" 和内核名两个 token
        wins.append(toks)
    prefix = " ".join(line.split("|")[0].split()[:2]) if tail else line
    return prefix, wins
names = sorted(n for n in os.listdir(bd) if n.endswith(".trace"))
print("\n── 逐派发窗口 trace：好=%s vs 坏=%s ──" % (good, bd))
for nm in names:
    gp, bp = f"{gd}/{nm}", f"{bd}/{nm}"
    pre = f"{bd}/{nm[:-6]}.pre"
    if not os.path.exists(gp) or not os.path.exists(pre):
        print("  %-40s 缺文件" % nm); continue
    pw = parse_line(open(pre).read())[1]
    gl = [parse_line(l) for l in open(gp).read().splitlines()]
    bl = [parse_line(l) for l in open(bp).read().splitlines()]
    if not gl or not bl:
        continue
    print("\n  ── %s（好 %d 条 / 坏 %d 条派发；窗口 %d/%d 字）──"
          % (nm, len(gl), len(bl), len(gl[0][1][0]), len(gl[0][1][1])))
    for wi in (0, 1):
        amb = pw[wi]
        line_g = [w[wi] for _, w in gl]
        line_b = [w[wi] for _, w in bl]
        def first_non_ambient(seq):
            for k, v in enumerate(seq):
                if v != amb:
                    return k
            return None
        fg, fb = first_non_ambient(line_g), first_non_ambient(line_b)
        print("     窗口%d：好轮首条内容非环境值 = %s；坏轮 = %s"
              % (wi + 1,
                 ("第 %d 条 %s" % (fg, gl[fg][0])) if fg is not None
                 else "无（全程=环境值 ⇒ 从未被写过）",
                 ("第 %d 条 %s" % (fb, bl[fb][0])) if fb is not None
                 else "无（全程=环境值 ⇒ 从未被写过）"))
        if fg is not None and fb is not None:
            print("        好轮写入时值 %s" % " ".join(line_g[fg][:6]))
            print("        坏轮写入时值 %s" % " ".join(line_b[fb][:6]))
    # 第一条内容不同的派发
    first_diff = None
    for k in range(min(len(gl), len(bl))):
        if gl[k][1] != bl[k][1]:
            first_diff = k; break
    if first_diff is None and len(gl) == len(bl):
        print("     逐条全同")
    else:
        print("     ✖ 首条不同的派发 k=%s：" % first_diff)
        if first_diff is not None:
            print("        好 %s" % gl[first_diff][0])
            print("        坏 %s" % bl[first_diff][0])
            print("        好窗 %s" % " ".join(gl[first_diff][1][0][:10]))
            print("        坏窗 %s" % " ".join(bl[first_diff][1][0][:10]))
PY
echo
echo "  明细：$OUT"
