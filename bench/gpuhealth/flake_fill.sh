#!/bin/bash
# 「静默丢行」是不是「shader 读到了从没被写过的 arena 字节」？
#
# 机理：Metal 的 new_buffer **不保证清零**。全新进程 = 全新 arena，任何
# 没人写过却被 shader 读到的字节 = 驱动残留垃圾 ⇒ **逐次不同**，这是串行
# 程序里唯一剩下的逐轮变量。`QPPOCR_METAL_ARENA_FILL` 把整块 arena 填成
# 统一字节，把那份垃圾换成**确定值**。
#
# 判据（不是看快慢，是看**输出的分布**）：
#   base   若 ~13% 空行
#   FILL=00 全零 ⇒ 若 flake 坍成 0% 或坍成 100%，机理坐实
#   FILL=ff 全 0xFF = f32 NaN ⇒ 若输出变成 NaN/乱码，坐实（并指出读的位置）
#
# 三臂只改 arena 初值，不改任何数值语义。
#
# 用法： N=20 ./flake_fill.sh
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
Q=$ROOT/target/release/qppocr
MODELS=$ROOT/models
N=${N:-20}
IMG=${IMG:-$(cd testimgs && pwd)/img-008.jpg}
OUT=logs/flake_fill_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

[ -x "$Q" ] || { echo "先构建： cargo build --release -p qppocr-cli --features qppocr/gpu"; exit 1; }
./health.sh -q || { echo "健康门禁没过，先别测"; exit 2; }

# 只取会翻的那一行（img-008 是第 10 行，0 起算）的 text/conf，外加 num_unread
PARSE='import json,sys
d=json.load(sys.stdin)[0]
ls=d["lines"]
i=10 if len(ls)>10 else len(ls)-1
print("unread=%d line%d=%r conf=%.4f" % (d["num_unread"], i, ls[i]["text"], ls[i]["confidence"]))'

echo "== arena 初值判别：$N 次/臂 =="
run_arm() {
    local label=$1; shift
    : >"$OUT/$label.txt"
    for _ in $(seq 1 "$N"); do
        env "$@" "$Q" "$IMG" --tier tiny --device metal --models "$MODELS" \
            --workers 1 --quiet --json 2>>"$OUT/$label.err" \
            | python3 -c "$PARSE" >>"$OUT/$label.txt" 2>/dev/null \
            || echo "(解析失败)" >>"$OUT/$label.txt"
    done
    local bad total
    bad=$(grep -c 'unread=1' "$OUT/$label.txt" || true)
    total=$(grep -c . "$OUT/$label.txt" || true)
    printf '\n  ── %-12s 空行 %2d/%s ──\n' "$label" "$bad" "$total"
    sort "$OUT/$label.txt" | uniq -c | sort -rn | head -6 | sed 's/^/     /'
}

run_arm "base"
run_arm "FILL=00" QPPOCR_METAL_ARENA_FILL=00
run_arm "FILL=ff" QPPOCR_METAL_ARENA_FILL=ff

echo
echo "  明细：$OUT"
