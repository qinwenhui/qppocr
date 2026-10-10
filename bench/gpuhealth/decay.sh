#!/bin/bash
# 衰减曲线实验：回答「到底做了多少 GPU 工作量之后，访存开始变慢？」
#
# 背景：之前的观测是**倾向性**的——同一轮开机里带宽像是随累积工作量单调下滑
# （95.6 → 82.6 → 72.0 → 0.5 GB/s），但当时负载没控制，可能是别的进程挤的。
# 这个脚本把负载变成可控的：固定每轮跑 N 张图，跑完立刻测带宽 + 记录驱动 GPU 内存占用。
#
# **关键设计：一旦带宽掉到 BELOW 以下就立刻停手。** 目的是在"刚开始变慢"的那一刻
# 就抓住它，而不是等它掉到 1/70 —— 那会让用户整个桌面卡到不可用、只能重启。
#
# 用法：
#   bench/gpuhealth/decay.sh [每轮图片数] [最大轮数]
#   默认 2 张 / 20 轮。
#
# 单变量对照（验证静态审计那条嫌疑）：
#   QPPOCR_METAL_NO_ARENA=1 bench/gpuhealth/decay.sh 2 20
#   —— 关掉会话级 arena 复用。如果开着会衰减、关掉就不衰减，凶手就是那条复用路径。
set -uo pipefail
cd "$(dirname "$0")/../.."

PER=${1:-2}
ROUNDS=${2:-20}
BELOW=${BELOW:-70}          # GB/s：低于这个就停手（正常健康是 93~95）
ROUND_TIMEOUT=${ROUND_TIMEOUT:-180}
IMGDIR=${IMGDIR:-/Users/qinwh/Pictures/dataset}
Q=target/release/qppocr
LOG=bench/gpuhealth/logs/decay_$(date +%m%d_%H%M%S).txt

mkdir -p bench/gpuhealth/logs
[ -x "$Q" ] || { echo "先构建： cargo build --release -p qppocr-cli --features qppocr/gpu"; exit 1; }

# 驱动侧 GPU 内存占用——如果泄漏，这个数会随轮次增长
gpu_mem() {
    ioreg -r -c IOAccelerator -l 2>/dev/null \
        | grep -oE '"(Alloc system memory|In use system memory)"=[0-9]+' \
        | awk -F= '{s[$1]+=$2} END {for (k in s) printf "%s=%.2fGB ", k, s[k]/1073741824}'
}

# 单次快速带宽读数：只看 **GPU 那一行**，取最好成绩（前几轮含首次触碰）。
# ⚠️ 别用 `grep '→ .* GB/s'` 抓全部——那会把下面那行 CPU memcpy（约 146 GB/s）也算进来，
# 于是"blit"报的其实是 CPU 的数、永远触发不了停手阈值。这个 bug 我在第一轮真踩过。
bandwidth() {
    bench/gpuhealth/gpuhealth 2>/dev/null \
        | grep 'GPU（' | grep -oE '→ *[0-9.]+' | grep -oE '[0-9.]+$' | sort -rn | head -1
}

{
    echo "════ 衰减曲线  PER=$PER 张/轮  ROUNDS=$ROUNDS  停手阈值 $BELOW GB/s ════"
    echo "环境变量： QPPOCR_METAL_NO_ARENA=${QPPOCR_METAL_NO_ARENA:-<未设>}"
    echo "开始 $(date '+%F %T')"
    echo

    BW0=$(bandwidth)
    echo "轮 00（未跑任何负载）  blit ${BW0:-?} GB/s   $(gpu_mem)"
    if [ -z "${BW0:-}" ] || awk "BEGIN{exit !($BW0 < $BELOW)}"; then
        echo "⛔ 一开始就没到 $BELOW GB/s —— 机器本来就瘸，先重启。本实验不作数。"
        exit 3
    fi

    # ⚠️ 别用 awk 的 BEGIN 块挑图片：BEGIN 在读到输入之前就执行，那里 $1/$2 是空的，
    # 生成出来是个空列表——引擎收不到图，于是打印 usage 并以 rc=0 退出，看起来"跑完了"。
    # 这个坑我踩过两次，两次实验全废。用 bash 数组，并保留下面那道自检。
    IMGS_ARR=("$IMGDIR"/img-*.jpg)
    NTOT=${#IMGS_ARR[@]}
    [ "$NTOT" -ge "$PER" ] || { echo "图不够（只有 $NTOT 张）"; exit 1; }

    for r in $(seq 1 "$ROUNDS"); do
        # 每轮滑动取不同的图（循环使用），避免总跑同样几张
        BATCH=()
        for ((i = 0; i < PER; i++)); do
            BATCH+=("${IMGS_ARR[$(( ((r - 1) * PER + i) % NTOT ))]}")
        done
        # 毫秒精度：2 张图只要 ~128ms，秒级取整会全是 0，看不出趋势
        T0=$(python3 -c 'import time;print(int(time.time()*1000))')
        perl -e "alarm $ROUND_TIMEOUT; exec @ARGV" "$Q" "${BATCH[@]}" \
            --tier tiny --device metal --workers 0 --quiet \
            >> "$LOG.engine" 2>&1
        rc=$?
        # 自检：引擎若没吃到图就会打印 usage。这种情况整轮实验无效，必须立刻停。
        if [ "$r" = "1" ] && grep -q "usage: qppocr" "$LOG.engine" 2>/dev/null; then
            echo "⛔ 引擎没收到图片参数（打印了 usage）—— 本轮实验无效，已停。"
            echo "   检查脚本里拼图片列表的那段。"
            exit 4
        fi
        DT=$(( $(python3 -c 'import time;print(int(time.time()*1000))') - T0 ))

        BW=$(bandwidth)
        printf "轮 %02d  %s张  引擎 %6dms(rc=%d)  blit %6s GB/s  %s\n" \
               "$r" "$PER" "$DT" "$rc" "${BW:-?}" "$(gpu_mem)"

        [ "$rc" = "142" ] && { echo "  ⚠️ 本轮超时 ${ROUND_TIMEOUT}s 被掐断 —— 已经很不正常了，停手。"; break; }
        if [ -z "${BW:-}" ]; then echo "  ⚠️ 读不到带宽，停手。"; break; fi
        awk "BEGIN{exit !($BW < $BELOW)}" && {
            echo
            echo "  ★ 带宽已掉到 ${BW} GB/s（阈值 ${BELOW}）—— **在这个工作量附近开始衰减**。已停手，"
            echo "    机器还有救，别重启，先看下面的结论。"
            break
        }
    done

    echo
    echo "结束 $(date '+%F %T')"
    echo "驱动 GPU 内存最终： $(gpu_mem)"
} 2>&1 | tee "$LOG"

echo
echo "日志 → $LOG   （引擎输出在 $LOG.engine）"
