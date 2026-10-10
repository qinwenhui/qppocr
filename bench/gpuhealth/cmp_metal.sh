#!/bin/bash
# 我们 vs 竞品 SimdPaddleOCR：Metal 后端同机同图配对对拍。
#
# 为什么必须配对交错：本机在**持续 GPU 负载下会爬升**（2026-10-10 实测：
# 一次 medium 100 图的 --rounds 3 内，单轮 30.8s → 41.1s → 46.7s，+52%）。
# 先把一边三档跑完再跑另一边，等于把「先后次序」记成「实现差异」——
# 我第一版就是这么干的，我们的 medium 因此虚高到 410.7 ms/图（真值 ~255）。
# 所以：**每个档位内，两家轮流跑**，只比同档内的相对快慢。
#
# 用法： ./cmp_metal.sh [对数，默认 3] [档位，默认 "tiny small medium"]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PAIRS=${1:-3}
TIERS=${2:-"tiny small medium"}

COMP=$HOME/Projects/SimdPaddleOCR
COMP_BIN=$COMP/test/Sdcb.SimdPaddleOCR.Tests/bin/Release/net10.0/Sdcb.SimdPaddleOCR.Tests
export DOTNET_ROOT="$HOME/.dotnet"
OURS=$ROOT/target/release/qppocr

# 竞品：他们文档的配置（metal-m4.md:3）
#   --workers 4 --benchmark-kind simd --warmup 1，n=99
run_comp() {  # $1=tier -> 打印 median ms/图
    ( cd "$COMP" && "$COMP_BIN" --engine metal --model "$1" --workers 4 \
        --benchmark-kind simd --warmup 1 --input dataset \
        --out "bench-out/cmp-$1.json" >/dev/null 2>&1 )
    python3 -c "import json;print(json.load(open('$COMP/bench-out/cmp-$1.json'))['summary']['total_ms']['median'])"
}

# 我们：同 100 图、同一批模型文件、rounds 3 取中位
run_ours() {  # $1=tier -> 打印 total ms/图
    "$OURS" bench "$HERE"/corpus100/img-*.jpg --models "$ROOT/models" --device metal \
        --tier "$1" --preset balanced --rounds 3 --quiet --workers 1 2>/dev/null \
      | awk '/^base/{print $2}'
}

echo "配对对拍 Metal：竞品（他们 harness，4w，fp16）vs 我们（bench，1w，f32）"
echo "corpus100 = 100 张；每档 $PAIRS 对，档内交错"
echo
printf '%-8s %10s %10s %10s\n' "档位" "竞品中位" "我们中位" "竞品快x"
for t in $TIERS; do
    C=(); O=()
    for r in $(seq 1 "$PAIRS"); do
        # 轮内换序（同 ab.sh 的理由）：本机持续负载下会爬升，固定「竞品先跑」
        # 等于每对都让竞品吃冷机、我们吃热机 —— 静默地偏袒竞品。
        if [ $((r % 2)) -eq 1 ]; then c=$(run_comp "$t"); o=$(run_ours "$t"); ord="竞,我"
        else o=$(run_ours "$t"); c=$(run_comp "$t"); ord="我,竞"; fi
        printf '  %-6s pair%d (%s)  竞品 %8s   我们 %8s\n' "$t" "$r" "$ord" "$c" "$o"
        C[${#C[@]}]="$c"; O[${#O[@]}]="$o"
    done
    med_c=$(printf '%s\n' "${C[@]}" | sort -n | awk '{v[NR]=$1} END{print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
    med_o=$(printf '%s\n' "${O[@]}" | sort -n | awk '{v[NR]=$1} END{print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
    awk -v t="$t" -v c="$med_c" -v o="$med_o" \
        'BEGIN{printf "%-8s %10.1f %10.1f %9.2fx\n", t, c, o, o/c}'
done
