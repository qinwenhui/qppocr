#!/bin/bash
# 蹲一次翻车：连跑到第一次 `unread>0` 为止，把**好轮**和**坏轮**的完整 JSON
# 都留下，并算出**掉的是哪一行**（好轮有、坏轮没有的那条）。
#
# 为什么要它：drop 是「某一行整条消失」，而 rec 是按**宽度分桶**分批跑的
# （实测 7 个桶：[1,3,48,128] / [4,3,48,256] / [2,3,48,320] / [2,3,48,384]
# / [1,3,48,448] / [1,3,48,1216] / [1,3,48,1344]）。知道掉的是第几行，
# 才能知道该盯哪个桶、哪个计划。
#
# 用法： N=60 IMG=testimgs/img-002.jpg ./flake_catch.sh
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
Q=$ROOT/target/release/qppocr
MODELS=$ROOT/models
N=${N:-60}
IMG=${IMG:-$(cd testimgs && pwd)/img-002.jpg}
OUT=logs/flake_catch_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

[ -x "$Q" ] || { echo "先构建"; exit 1; }
./health.sh -q || { echo "健康门禁没过"; exit 2; }

# 每轮把 rec 各计划的激活区落盘（`QPPOCR_METAL_IO_DUMP`），好轮/坏轮各留一份，
# 这样蹲到之后能立刻比「第一处分歧字节落在哪个计划、哪个字」。
DUMP=${DUMP:-1}

echo "== 蹲翻车：$(basename "$IMG")，最多 ${N} 轮 =="
good=""
for i in $(seq 1 "$N"); do
    d="$OUT/run$(printf '%03d' "$i")"
    if [ "$DUMP" = 1 ]; then mkdir -p "$d"; dump="QPPOCR_METAL_IO_DUMP=$d/a"; else dump=""; fi
    env $dump "$Q" "$IMG" --tier tiny --device metal --models "$MODELS" \
        --workers 1 --quiet --json 2>"$OUT/err$(printf '%03d' "$i").txt" \
        >"$OUT/run$(printf '%03d' "$i").json"
    nu=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[0]["num_unread"])' \
         "$OUT/run$(printf '%03d' "$i").json" 2>/dev/null || echo "?")
    if [ -z "$good" ]; then
        min=$(python3 -c 'import json,sys;print(min(l["confidence"] for l in json.load(open(sys.argv[1]))[0]["lines"]))' \
              "$OUT/run$(printf '%03d' "$i").json" 2>/dev/null || echo 1)
        if [ "$nu" = "0" ] && python3 -c "import sys;sys.exit(0 if $min>0.9 else 1)"; then
            cp "$OUT/run$(printf '%03d' "$i").json" "$OUT/good.json"
            good=$i; rm -f "$OUT"/good_dumps; ln -s "$d" "$OUT/good_dumps"
        fi
    fi
    printf '  run%03d unread=%s%s\n' "$i" "$nu" "$([ -n "$good" ] && [ "$i" = "$good" ] && echo '  ← 存为好轮')"
    if [ "$nu" != "0" ] && [ "$nu" != "?" ]; then
        cp "$OUT/run$(printf '%03d' "$i").json" "$OUT/bad.json"
        rm -f "$OUT/bad_dumps"; ln -s "$d" "$OUT/bad_dumps"
        echo "  ✖ 第 $i 轮翻车，已存 bad.json"
        # 多余的轮次目录删掉，只留好轮/坏轮（每轮 ~几十 MB）
        for j in $(seq 1 "$N"); do
            [ "$j" = "$i" ] && continue
            [ "$j" = "$good" ] && continue
            rm -rf "$OUT/run$(printf '%03d' "$j")"
        done
        break
    fi
    [ -n "$good" ] && [ "$i" != "$good" ] && rm -rf "$d"
done

[ -f "$OUT/bad.json" ] || { echo "  ${N} 轮没蹲到，加大 N"; exit 1; }
[ -f "$OUT/good.json" ] || { echo "  没存到好轮（置信度都偏低？）"; exit 1; }

python3 - "$OUT/good.json" "$OUT/bad.json" <<'PY'
import json, sys, difflib
g = json.load(open(sys.argv[1]))[0]
b = json.load(open(sys.argv[2]))[0]
gt = [l["text"] for l in g["lines"]]
bt = [l["text"] for l in b["lines"]]
print("\n  好轮 %d 行 / 坏轮 %d 行（unread %d → %d）" % (len(gt), len(bt), g["num_unread"], b["num_unread"]))
sm = difflib.SequenceMatcher(None, gt, bt)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        continue
    print("  %-8s 好[%d:%d] %r\n          坏[%d:%d] %r" % (tag, i1, i2, gt[i1:i2], j1, j2, bt[j1:j2]))
    for k in range(i1, i2):
        box = g["lines"][k].get("box") or g["lines"][k].get("points")
        print("           好轮该行 box=%s conf=%.4f" % (box, g["lines"][k]["confidence"]))
    for k in range(j1, j2):
        box = b["lines"][k].get("box") or b["lines"][k].get("points")
        print("           坏轮该行 box=%s conf=%.4f" % (box, b["lines"][k]["confidence"]))
if gt == bt:
    print("  文本全同，差异只在 num_unread → 掉的那行连框都没了（det/后处理侧）")
PY

# 逐计划对拍激活区：好轮 vs 坏轮的 dump 文件同名（`run_seq` 是**进程内**序号，
# 派发序不变 ⇒ 两次进程同名），第一处分歧就落在某个计划、某个字上。
if [ "$DUMP" = 1 ] && [ -e "$OUT/good_dumps" ] && [ -e "$OUT/bad_dumps" ]; then
python3 - "$OUT/good_dumps" "$OUT/bad_dumps" <<'PY'
import sys, os, glob, struct
good, bad = sys.argv[1], sys.argv[2]
base = {os.path.basename(p): p for p in glob.glob(good + "/*.bin")}
badf = {os.path.basename(p): p for p in glob.glob(bad + "/*.bin")}
print("\n  ── 逐计划激活区：好轮 vs 坏轮 ──")
if not base:
    print("   （好轮没有 dump 文件，检查 QPPOCR_METAL_IO_DUMP 是否编进二进制）")
hit = 0
for nm in sorted(set(base) & set(badf)):
    a = open(badf[nm], 'rb').read()
    c = open(base[nm], 'rb').read()
    if a == c:
        print("   %-32s 全同" % nm)
        continue
    hit += 1
    n = min(len(a), len(c))
    d = [i for i in range(0, n, 4) if a[i:i+4] != c[i:i+4]]
    f = d[0] // 4
    fa = struct.unpack('<f', a[d[0]:d[0]+4])[0]
    fc = struct.unpack('<f', c[d[0]:d[0]+4])[0]
    print("   %-32s ✖ 差 %d 字 / %d 字，首差 @字 %d（坏 %.6g | 好 %.6g）"
          % (nm, len(d), n // 4, f, fa, fc))
    # 分歧的字区间 → 推断是哪个张量（按字数聚类）
    runs = []
    s = p = d[0]
    for x in d[1:]:
        if x == p + 4:
            p = x
        else:
            runs.append((s // 4, p // 4)); s = p = x
    runs.append((s // 4, p // 4))
    print("        分歧区间 %d 段，前 6 段（起字, 止字, 长）：%s"
          % (len(runs), [(a_, b_, b_ - a_ + 1) for a_, b_ in runs[:6]]))
for nm in sorted(set(base) - set(badf)):
    print("   %-32s 坏轮没有（计划没跑到？）" % nm)
if not hit:
    print("   所有计划逐位相同 ⇒ 分歧在 det 侧或主机侧，不在 rec 的激活区")
PY
fi
echo "  明细：$OUT"
