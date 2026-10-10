#!/bin/bash
# scoping：逐图跑 N 次，看「会翻」是 img-008 的边角 case 还是一批图的共病。
#
# 动机（2026-10-10 笔记 §6）：目前只观察到 img-008 第 10 行会翻，其余行
# 30 轮零次。**若只有它一张**，那是一个边角 case；**若一批图都翻**，那是
# 静默拉低全语料精度、优先级高于所有剩余性能工作。这个数决定后面怎么走。
#
# 判定不看单行、不看 num_unread（翻车有「文本错了但 unread=0」的形态，
# 实测 `0|ANY|0.4534`），而是拿**整份输出**做指纹：行数|unread|各行文本
# 的哈希 + 文本串。逐图统计指纹分布，并标出哪一条与 CPU 臂一致。
#
# 用法： N=20 ./flake_scope.sh [图片目录]
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
Q=$ROOT/target/release/qppocr
MODELS=$ROOT/models
N=${N:-20}
DIR=${1:-testimgs}
OUT=logs/flake_scope_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

[ -x "$Q" ] || { echo "先构建： cargo build --release -p qppocr-cli --features qppocr/gpu"; exit 1; }
./health.sh -q || { echo "健康门禁没过，先别测"; exit 2; }

read -r -d '' PARSE <<'PY' || true
import json,sys,hashlib
d=json.load(sys.stdin)[0]
ts=[l["text"] for l in d["lines"]]
h=hashlib.sha1("\x1f".join(ts).encode()).hexdigest()[:8]
print("%d|%d|%s|%s" % (len(ts), d["num_unread"], h, "⏎".join(ts)))
PY

echo "== 逐图 × ${N} 次（metal）；每图另跑 1 次 cpu 当参照 =="
for f in "$DIR"/*.jpg "$DIR"/*.png; do
    [ -e "$f" ] || continue
    img=$(basename "$f")
    : >"$OUT/$img.metal"
    for _ in $(seq 1 "$N"); do
        "$Q" "$f" --tier tiny --device metal --models "$MODELS" \
            --workers 1 --quiet --json 2>>"$OUT/$img.err" \
            | python3 -c "$PARSE" >>"$OUT/$img.metal" 2>/dev/null \
            || echo "(解析失败)" >>"$OUT/$img.metal"
    done
    cpu=$("$Q" "$f" --tier tiny --device cpu --models "$MODELS" \
            --workers 1 --quiet --json 2>/dev/null | python3 -c "$PARSE" 2>/dev/null)
    printf '%s\n' "$cpu" >"$OUT/$img.cpu"
    n_uniq=$(sort -u "$OUT/$img.metal" | wc -l | tr -d ' ')
    # 指纹（不含文本串）用于判「变没变」
    sig=$(cut -d'|' -f1-3 "$OUT/$img.metal" | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
    same=$(grep -c -F -x "$cpu" "$OUT/$img.metal" || true)
    flag="  "
    [ "$n_uniq" != "1" ] && flag="⚠ "
    printf '%s%-14s 三种以上指纹 %d | 与 CPU 一致 %2d/%s | CPU: %s\n' \
        "$flag" "$img" "$n_uniq" "$same" "$N" "${cpu:0:60}"
    if [ "$n_uniq" != "1" ]; then
        cut -d'|' -f1-3 "$OUT/$img.metal" | sort | uniq -c | sort -rn | head -4 | sed 's/^/       /'
    fi
done
echo
echo "  明细（含全文）：$OUT"
echo "  ⚠️ 指纹 = 行数|unread|文本哈希；与 CPU 不一致但指纹唯一的图 = 系统性差异，不是 flake"
