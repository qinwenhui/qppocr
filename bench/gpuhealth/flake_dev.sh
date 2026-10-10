#!/bin/bash
# 同一张图、同一档位，metal vs cpu 的**输出分布**对照。
#
# 动机：flake 已量出**不是二值的**——img-008 第 10 行 20 次里
#   15× 'No' 0.9522 / 3× 'e' 0.4351 / 2× '' 0.0000
# 中间态说明是**连续扰动**（不是「读到一个垃圾字节」）。若 CPU 臂也出 'e'，
# 病根在主机侧共用的裁切/缩放，与 Metal 无关；若 CPU 恒对，才轮到 Metal。
#
# ⚠️ 之前「CPU 8/8 干净」对 15% 的速率是**欠功率**的（0.85^8=27%），不足为据。
#
# 用法： N=20 ./flake_dev.sh [cpu|metal] [img路径]
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
Q=$ROOT/target/release/qppocr
MODELS=$ROOT/models
N=${N:-20}
DEV=${1:-cpu}
IMG=${2:-$(cd testimgs && pwd)/img-008.jpg}
OUT=logs/flake_dev_${DEV}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")"

[ -x "$Q" ] || { echo "先构建： cargo build --release -p qppocr-cli --features qppocr/gpu"; exit 1; }
# CPU 臂不吃 GPU，但仍然只跑一个进程，不与别的 GPU 活并行。
[ "$DEV" = metal ] && { ./health.sh -q || { echo "健康门禁没过"; exit 2; }; }

PARSE='import json,sys
d=json.load(sys.stdin)[0]
ls=d["lines"]
i=10 if len(ls)>10 else len(ls)-1
print("unread=%d line%d=%r conf=%.4f" % (d["num_unread"], i, ls[i]["text"], ls[i]["confidence"]))'

echo "== ${DEV} × ${N}：$(basename "$IMG") =="
: >"$OUT.txt"
for _ in $(seq 1 "$N"); do
    "$Q" "$IMG" --tier tiny --device "$DEV" --models "$MODELS" \
        --workers 1 --quiet --json 2>>"$OUT.err" \
        | python3 -c "$PARSE" >>"$OUT.txt" 2>/dev/null \
        || echo "(解析失败)" >>"$OUT.txt"
done
sort "$OUT.txt" | uniq -c | sort -rn | sed 's/^/   /'
echo "   明细：$OUT.txt"
