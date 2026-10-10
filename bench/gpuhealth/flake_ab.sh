#!/bin/bash
# 静默识别错误的**分臂**定位：同一个内核开关组合下，单图跑 N 次，数「出现空行」的次数。
#
# 背景（2026-10-10）：img-008 单图 metal 偶发某一行 `text=""`、`conf=0.0`
# （CTC 全 blank），同图 CPU 路径 8/8 稳定、det 输出逐位不变
# （`num_boxes/num_det_retried/num_flipped` 全同，只有 `num_unread` 从 0 变 1）
# ⇒ 是 **rec 侧**、**Metal 专有**、**全新计划路径**（8 次都是独立进程、无重放）上的 bug。
#
# 四条候选臂都是代码里承诺「关掉时逐位回到改前」的开关（见 session.rs 的文档注释），
# 所以臂间差异只能是那条内核替身，不是别的。
#
# 用法： N=30 ./flake_ab.sh                    # 单图 × 5 臂
#        N=30 IMG=.../img-003.jpg ./flake_ab.sh
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
Q=$ROOT/target/release/qppocr
MODELS=$ROOT/models
N=${N:-30}
IMG=${IMG:-$(cd testimgs && pwd)/img-008.jpg}
OUT=logs/flake_ab_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

[ -x "$Q" ] || { echo "先构建： cargo build --release -p qppocr-cli --features qppocr/gpu"; exit 1; }

# 门禁：先确认机器健康，否则读数无意义（判据是实测带宽，不是 recoveryCount）
./health.sh -q || { echo "健康门禁没过，先别测"; exit 2; }

echo "== 分臂：$N 次/臂，图 $(basename "$IMG") =="

# python 放在这里、单行、单引号包裹，避免跟 bash 的 $ 打架
COUNT='import json,sys; print(json.load(sys.stdin)[0]["num_unread"])'

run_arm() { # $1=标签；其余=要 export 的 VAR=VAL
    local label=$1; shift
    : >"$OUT/$label.counts"
    for _ in $(seq 1 "$N"); do
        env "$@" "$Q" "$IMG" --tier tiny --device metal --models "$MODELS" \
            --workers 1 --quiet --json 2>"$OUT/$label.err" \
            | python3 -c "$COUNT" >>"$OUT/$label.counts" 2>/dev/null
    done
    local bad total empt
    bad=$(awk '$1>0' "$OUT/$label.counts" | wc -l | tr -d ' ')
    total=$(grep -c . "$OUT/$label.counts" | tr -d ' ')
    empt=$((N - total))
    printf '  %-22s 空行 %2d/%d' "$label" "$bad" "$total"
    [ "$empt" -gt 0 ] && printf '（另有 %d 次进程失败）' "$empt"
    echo
}

run_arm "基线（全开）"
run_arm "CONV1X1_MMA=0"  QPPOCR_METAL_CONV1X1_MMA=0
run_arm "CONV_KXK_MMA=0" QPPOCR_METAL_CONV_KXK_MMA=0
run_arm "CONVT_TILE=0"   QPPOCR_METAL_CONVT_TILE=0
run_arm "NO_ARGMAX=1"    QPPOCR_METAL_NO_ARGMAX=1

echo "  计数文件：$OUT"
