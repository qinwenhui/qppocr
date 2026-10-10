#!/bin/bash
# 「静默丢行」是并发/同步问题吗？——三臂判别。
#
#   base         正常
#   CHUNK=1      每条 dispatch 单独 commit+wait（**强制串行**）⇒ 若 flake 消失，是重叠竞态
#   NO_BARRIER=1 拆掉逐条资源屏障（生产不该用）⇒ 若 flake 显著变多，屏障确实在挡东西
#
# 三臂都不改数值语义的只有 CHUNK（只改时间）；NO_BARRIER 会改数值，只用来判「方向」。
#
# 用法： N=30 ./flake_sync.sh
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
Q=$ROOT/target/release/qppocr
MODELS=$ROOT/models
N=${N:-30}
IMG=${IMG:-$(cd testimgs && pwd)/img-008.jpg}
OUT=logs/flake_sync_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

[ -x "$Q" ] || { echo "先构建： cargo build --release -p qppocr-cli --features qppocr/gpu"; exit 1; }
./health.sh -q || { echo "健康门禁没过，先别测"; exit 2; }

COUNT='import json,sys; print(json.load(sys.stdin)[0]["num_unread"])'

echo "== 同步判别：$N 次/臂 =="
run_arm() {
    local label=$1; shift
    : >"$OUT/$label.counts"
    for _ in $(seq 1 "$N"); do
        env "$@" "$Q" "$IMG" --tier tiny --device metal --models "$MODELS" \
            --workers 1 --quiet --json 2>"$OUT/$label.err" \
            | python3 -c "$COUNT" >>"$OUT/$label.counts" 2>/dev/null
    done
    local bad total
    bad=$(awk '$1>0' "$OUT/$label.counts" | wc -l | tr -d ' ')
    total=$(grep -c . "$OUT/$label.counts" | tr -d ' ')
    printf '  %-16s 空行 %2d/%s\n' "$label" "$bad" "$total"
}

run_arm "base"
run_arm "CHUNK=1"     QPPOCR_METAL_CHUNK=1
run_arm "NO_BARRIER=1" QPPOCR_METAL_NO_BARRIER=1

echo "  计数文件：$OUT"
