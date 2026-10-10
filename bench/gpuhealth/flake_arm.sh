#!/bin/bash
# 通用臂跑手：同一张图、同一档位，每臂 N 次，比**输出分布**。
#
# 用法： N=20 ./flake_arm.sh "base" "FILL_DYN0=QPPOCR_METAL_FILL_DYN=0" ...
#   每个参数形如 标签=VAR1=VAL1;VAR2=VAL2   （标签里不许有 '='）
#
# 为什么看分布不看均值：这个 flake **不是二值的**（实测同一行出过
# 15×'No' 0.9522 / 3×'e' 0.4351 / 2×'' 0.0000），均值会把三种形态糊成
# 一个数，把「换了一种错法」误读成「变好了」。
#
# ⚠️ 判定不能只看 num_unread：翻车还有「文本错了但 unread=0」（实测
# `0|ANY|0.4534`）。所以每臂都打出**完整分布**，判据是人眼看分布是否塌成单点。
#
# 用法惯例：N=20，只跑 GPU 一个进程，不与别的 GPU 活并行。
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
Q=$ROOT/target/release/qppocr
MODELS=$ROOT/models
N=${N:-20}
IMG=${IMG:-$(cd testimgs && pwd)/img-008.jpg}
OUT=logs/flake_arm_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

[ -x "$Q" ] || { echo "先构建： cargo build --release -p qppocr-cli --features qppocr/gpu"; exit 1; }
./health.sh -q || { echo "健康门禁没过，先别测"; exit 2; }

PARSE='import json,sys
d=json.load(sys.stdin)[0]
ls=d["lines"]
i=10 if len(ls)>10 else len(ls)-1
print("%d|%s|%.4f" % (d["num_unread"], ls[i]["text"], ls[i]["confidence"]))'

echo "== ${N} 次/臂：$(basename "$IMG") =="
for spec in "$@"; do
    label=${spec%%=*}
    envs=${spec#*=}
    : >"$OUT/$label.txt"
    for _ in $(seq 1 "$N"); do
        if [ -n "$envs" ]; then
            IFS=';' read -r -a kv <<<"$envs"
            env "${kv[@]}" "$Q" "$IMG" --tier tiny --device metal --models "$MODELS" \
                --workers 1 --quiet --json 2>>"$OUT/$label.err" \
                | python3 -c "$PARSE" >>"$OUT/$label.txt" 2>/dev/null \
                || echo "(解析失败)" >>"$OUT/$label.txt"
        else
            "$Q" "$IMG" --tier tiny --device metal --models "$MODELS" \
                --workers 1 --quiet --json 2>>"$OUT/$label.err" \
                | python3 -c "$PARSE" >>"$OUT/$label.txt" 2>/dev/null \
                || echo "(解析失败)" >>"$OUT/$label.txt"
        fi
    done
    ref=$(head -1 "$OUT/$label.txt")
    ok=$(grep -c -F -x "$ref" "$OUT/$label.txt" || true)
    printf '\n  ── %-14s 与首轮一致 %2d/%s ──  基准 %s\n' "$label" "$ok" "$N" "$ref"
    sort "$OUT/$label.txt" | uniq -c | sort -rn | head -6 | sed 's/^/     /'
done
echo
echo "  明细：$OUT"
