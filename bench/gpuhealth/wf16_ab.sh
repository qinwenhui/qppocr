#!/bin/bash
# 生产旋钮 `QPPOCR_METAL_WF16`（fp16 权重存储）的**精度** A/B。
#
# 用法： ./wf16_ab.sh [tier=small] [图片数=100] [轮换序图数=0]
#
# 与 `ab.sh` 的分工：`ab.sh` 量性能（交错轮换序、只要 `base` 那几个数），
# 本脚本量**文本**（逐行 conf/rot/文本 + CER），所以必须整跑一遍流水线并
# 留全量 TSV。两者都用同一个 `bench_f16w`，只是一个截时间、一个截文本。
#
# 三个必须先摆正的前提，否则读数没有意义：
#
# 1. **必须 `QPPOCR_BENCH_DEVICE=metal`**。旋钮是 Metal-only 的，CPU 会话
#    读不到它——两条路会**逐字节相同**，看起来像「零代价」，其实是什么都
#    没测。两条 TSV 完全相同这一点本身就是「跑错了设备」的判据。
# 2. **同模型目录**。与「换模型目录」（`mk_f16w.py` 那条 CPU 上界路）不同，
#    这里两臂读的是**同一份** models/，唯一的自变量是环境变量。
# 3. **同图片序**。TSV 按图名对齐（`f16w_diff.py` 里 sorted(set(...))），
#    顺序其实无关，但保持同序遍历能让 diff 工具之外的人也能直接对读。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$HERE"

TIER=${1:-small}
N=${2:-100}
BIN=$ROOT/target/release/examples/bench_f16w
[ -x "$BIN" ] || {
    echo "先建： cargo build --release -p qppocr --example bench_f16w --features gpu"
    exit 1
}

IMGS=()
while IFS= read -r f; do IMGS[${#IMGS[@]}]="$f"; done < <(ls "$HERE"/corpus100/img-*.jpg | head -"$N")
[ "${#IMGS[@]}" -gt 0 ] || { echo "corpus100 里没图"; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
BASE=logs/wf16_base_${TIER}_${STAMP}.tsv
ON=logs/wf16_on_${TIER}_${STAMP}.tsv

echo "wf16 A/B：tier=$TIER  ${#IMGS[@]} 张图  设备=metal"
echo "  基线（f32 权重）→ $BASE"
QPPOCR_BENCH_DEVICE=metal QPPOCR_METAL_WF16=0 \
    "$BIN" "$ROOT/models" "$TIER" "${IMGS[@]}" >"$BASE" 2>/dev/null || exit 1
echo "  旋钮（fp16 权重）→ $ON"
QPPOCR_BENCH_DEVICE=metal QPPOCR_METAL_WF16=1 \
    "$BIN" "$ROOT/models" "$TIER" "${IMGS[@]}" >"$ON" 2>/dev/null || exit 1

if cmp -s "$BASE" "$ON"; then
    echo
    echo "⚠️  两份 TSV **逐字节相同** —— 旋钮没生效（设备不是 metal？会话走了 CPU？）"
    exit 2
fi

# ⚠️ `${TIER}` 的花括号不能省：紧跟全角括号时 bash 会把多字节字的首字节
# 当变量名的一部分，报 `TIER\xef...: unbound variable`。
python3 "$HERE/f16w_diff.py" "$BASE" "$ON" "Metal fp16 权重 vs f32（tier=${TIER}）"
echo
echo "TSV：$BASE  $ON"
