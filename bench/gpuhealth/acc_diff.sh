#!/bin/bash
# Metal vs CPU 的**精度**对拍（不是性能）。逐图、两后端各跑一次，
# 按「行索引」对齐（检测框在两种后端上逐位相同，见
# logs/flake_2026-10-10.md 的 det 取证），逐行比文本。
#
# 为什么要按行索引而不是按文本集合：同一张图上，行数/顺序是检测器定的，
# 与 rec 后端无关；文本才是 rec 的产物。索引对齐能把「哪一行变差」直接指出来。
#
# 分类（决定严重性）：
#   LINECOUNT  行数或 unread 不同 —— 最重，说明少了/多了整行
#   SUBST      实质不同（去掉空白后字符串不等）—— 可能是错字
#   SPACE      只差空白 —— 已知的 `rec_space_gap` 启发式类，良性
#
# 用法： TIER=tiny ./acc_diff.sh [图片目录] [前缀]
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
Q=$ROOT/target/release/qppocr
MODELS=$ROOT/models
TIER=${TIER:-tiny}
DIR=${1:-corpus100}
PFX=${2:-img-}

[ -x "$Q" ] || { echo "先构建： cargo build --release -p qppocr-cli --features qppocr/gpu"; exit 1; }
./health.sh -q || { echo "健康门禁没过，先别测"; exit 2; }

OUT=logs/acc_diff_${TIER}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

read -r -d '' PARSE <<'PY' || true
import json,sys
d=json.load(sys.stdin)[0]
print(json.dumps({
  "unread": d["num_unread"],
  "lines": [{"t": l["text"], "c": round(float(l["confidence"]), 4)} for l in d["lines"]],
}, ensure_ascii=False))
PY

echo "== Metal vs CPU 精度对拍：tier=$TIER  目录=$DIR =="
n=0; n_subst=0; n_space=0; n_lc=0
: >"$OUT/summary.tsv"
for f in "$DIR"/$PFX*.jpg "$DIR"/$PFX*.png; do
    [ -e "$f" ] || continue
    img=$(basename "$f"); n=$((n+1))
    "$Q" "$f" --tier "$TIER" --device metal --models "$MODELS" --workers 1 --quiet --json \
        2>/dev/null | python3 -c "$PARSE" >"$OUT/$img.metal" || echo '{}' >"$OUT/$img.metal"
    "$Q" "$f" --tier "$TIER" --device cpu   --models "$MODELS" --workers 1 --quiet --json \
        2>/dev/null | python3 -c "$PARSE" >"$OUT/$img.cpu"   || echo '{}' >"$OUT/$img.cpu"

    verdict=$(python3 - "$OUT/$img.metal" "$OUT/$img.cpu" "$img" "$OUT" <<'PY'
import json,sys,re
mp,cp,img,outd = sys.argv[1:5]
try:
    m=json.load(open(mp)); c=json.load(open(cp))
except Exception:
    print("PARSE\t0\t0"); raise SystemExit
if not m or not c:
    print("PARSE\t0\t0"); raise SystemExit
ml, cl = m["lines"], c["lines"]
if len(ml)!=len(cl) or m["unread"]!=c["unread"]:
    print("LINECOUNT\t%d\t%d" % (len(ml), len(cl))); raise SystemExit
sub=[]; sp=[]
for i,(a,b) in enumerate(zip(ml,cl)):
    if a["t"]==b["t"]: continue
    if re.sub(r"\s+","",a["t"])==re.sub(r"\s+","",b["t"]): sp.append(i)
    else: sub.append((i,a,b))
open(outd+"/"+img+".detail","w").write(json.dumps({"sub":sub,"space":sp},ensure_ascii=False,indent=1))
if sub: print("SUBST\t%d\t%d" % (len(sub), sp and len(sp) or 0))
elif sp: print("SPACE\t0\t%d" % len(sp))
else: print("SAME\t0\t0")
PY
)
    kind=$(echo "$verdict" | cut -f1); printf '%s\t%s\t%s\n' "$kind" "$img" "$verdict" >>"$OUT/summary.tsv"
    case "$kind" in
        LINECOUNT) n_lc=$((n_lc+1));  printf '⚠ %-14s %s\n' "$img" "$verdict" ;;
        SUBST)     n_subst=$((n_subst+1)); printf '⚠ %-14s %s\n' "$img" "$verdict" ;;
        SPACE)     n_space=$((n_space+1)); printf '· %-14s %s\n' "$img" "$verdict" ;;
        PARSE)     printf '✗ %-14s 解析失败\n' "$img" ;;
    esac
done

echo
echo "共 $n 图：行数/unread 不同 $n_lc | 实质不同 $n_subst | 只差空白 $n_space | 相同 $((n-n_lc-n_subst-n_space))"
echo "明细：$OUT"
[ "$n_subst" -gt 0 ] && {
    echo
    echo "== 实质不同的逐行对照 =="
    grep -l . "$OUT"/*.detail 2>/dev/null | while read -r d; do
        python3 -c "
import json,sys
d=json.load(open('$d'))
if not d['sub']: sys.exit()
print('--', '$d'.split('/')[-1].replace('.detail',''))
for i,a,b in d['sub']:
    print('  行%d  metal: %r (%.3f)' % (i, a['t'], a['c']))
    print('        cpu  : %r (%.3f)' % (b['t'], b['c']))
"
    done
}
exit 0
