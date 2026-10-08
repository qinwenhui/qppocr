#!/bin/bash
# 批量大小 → 致瘸阈值扫描（2026-10-08 那条「台阶」的收尾实验）
#
# **已定的事实（别再重跑）：变量是「一次批的图数」，不是并发进程数。**
#   A: 2 张 / 2 进程 每进程 1 张  → 健康 81.62 GB/s
#   B: 10 张 / 2 进程 每进程 5 张 → 打瘸 0.92 GB/s
#   C: 10 张 / 5 进程 每进程 2 张 → 打瘸 0.77 GB/s
# B、C 总图数相同、进程数不同，都瘸；A 只是图少就健康。（我先前据 C 得出「凶手是并发进程数」
# 是错的 —— B 这一臂就是为拆开这个混淆而跑的，它把我否掉了。）
#
# **默认计划：固定 2 进程，升序扫图数，一瘸即停。**
#   3 张 → 5 张 → 8 张      （2 已证健康，10 已证打瘸）
# 目的不是精确阈值，而是**找一个能稳定用的批量**（这批机器上 Metal 后端目前一批 10 张就必瘸，
# 等于不可用）。升序的好处：先拿到小的安全点，再往上推。
#
# 另一个更高价值的单变量实验（若只想跑一次）——判定是不是**我们自己的 arena 复用 bug**：
#   QPPOCR_METAL_NO_ARENA=1 bench/gpuhealth/worker_sweep.sh "10 2"
#   （`session.rs:283-320` 仅按容量复用 arena、静态布局随形状漂移。）
#   不瘸 → 凶手是那条复用路径（且我们可修）；仍瘸 → 复用无罪。
#
# 用法：
#   bench/gpuhealth/worker_sweep.sh              # 默认计划
#   bench/gpuhealth/worker_sweep.sh "5 2" "8 2"  # 自定义，"每步 图数 进程数"
#
# ⚠️ 必须在**刚重启的健康机器**上跑（开头会过门禁，不健康直接退出 3，本 run 不作数）。
# ⚠️ 一步瘸掉 = 一次重启。别随手加步数。
#
# 判据是**实测带宽**，不是 recoveryCount —— 这条已被实测打脸过一次（计数器干净回 0 但只有
# 0.25 GB/s）。低于 HEALTHY_MIN 一律当"已瘸"。
set -uo pipefail
cd "$(dirname "$0")/../.."

Q=target/release/qppocr
IMGDIR=${IMGDIR:-/Users/qinwh/Pictures/dataset}
HEALTHY_MIN=${HEALTHY_MIN:-60}     # GB/s：低于它就当这步已经打瘸了
ROUND_TIMEOUT=${ROUND_TIMEOUT:-120}
LOG=bench/gpuhealth/logs/sweep_$(date +%m%d_%H%M%S).txt
mkdir -p bench/gpuhealth/logs

[ -x "$Q" ] || { echo "先构建： cargo build --release -p qppocr-cli --features qppocr/gpu"; exit 1; }
[ -x bench/gpuhealth/gpuhealth ] || bench/gpuhealth/build.sh >/dev/null 2>&1

# 只看 GPU 那一行！`grep '→ .* GB/s'` 会把下面 CPU memcpy（~146 GB/s）也抓进来，
# 于是永远读不到"已经瘸了"。（decay.sh 里我踩过这个坑，两次实验全废。）
bandwidth() {
    bench/gpuhealth/gpuhealth 2>/dev/null \
        | grep 'GPU（' | grep -oE '→ *[0-9.]+' | grep -oE '[0-9.]+$' | sort -rn | head -1
}
gpu_mem() {
    ioreg -r -c IOAccelerator -l 2>/dev/null \
        | grep -oE '"Alloc system memory"=[0-9]+' \
        | awk -F= '{s+=$2} END {printf "%.2fGB", s/1073741824}'
}
# 收尾：perl 的 alarm 只杀 perl 自己，子进程会活下来占着 GPU（本仓实测过 etime 05:27、
# 0.0% CPU、还持有 2.5 GB）。所以每步后主动清一次。
reap() { pkill -TERM -f "target/release/qppocr" 2>/dev/null; sleep 1; }

# 默认计划："图数 进程数"
STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
    STEPS=("3 2" "5 2" "8 2")
fi

BW0=$(bandwidth)
{
    echo "════ worker 进程数阈值扫描 ════"
    echo "开始 $(date '+%F %T')   健康门槛 ${HEALTHY_MIN} GB/s   每步超时 ${ROUND_TIMEOUT}s"
    echo "计划： ${STEPS[*]}"
    echo
    echo "起始带宽 ${BW0:-?} GB/s   Alloc $(gpu_mem)"
    if [ -z "${BW0:-}" ] || awk "BEGIN{exit !($BW0 < $HEALTHY_MIN)}"; then
        echo "⛔ 机器本来就瘸（<$HEALTHY_MIN GB/s）—— 先重启，本 run 不作数。"
        exit 3
    fi
    echo

    IMGS=("$IMGDIR"/img-*.jpg)
    NTOT=${#IMGS[@]}

    STEP_IDX=0
    for step in "${STEPS[@]}"; do
        set -- $step
        N=$1; W=$2
        [ "$NTOT" -ge "$N" ] || { echo "图不够（只有 $NTOT 张），跳过 $step"; continue; }

        # 滑动取图，避免每步老跑同样几张
        BATCH=()
        for ((i = 0; i < N; i++)); do BATCH+=("${IMGS[$(( (STEP_IDX * N + i) % NTOT ))]}"); done
        STEP_IDX=$(( STEP_IDX + 1 ))

        # ⚠️ 别用 `awk "BEGIN{printf ...}"` 嵌在双引号的 $( ) 里 —— 嵌套引号会让 awk 收到
        # 残缺的程序文本（"BEGIN   10 /2"）。踩过。用 python3 算，稳。
        PERW=$(python3 -c "print(f'{$N/$W:.1f}')" 2>/dev/null || echo "?")
        echo "──── 第 ${STEP_IDX} 步： ${N} 张 / ${W} 进程（每进程 ≈${PERW} 张）────"

        # ⚠️ 不用 awk 的 BEGIN 块拼图片列表：BEGIN 在读到输入前就执行，$1/$2 为空 →
        # 空列表 → 引擎打印 usage 并以 rc=0 退出，看着像"跑完了"。这个坑我踩过两次。
        T0=$(python3 -c 'import time;print(int(time.time()*1000))')
        perl -e "alarm $ROUND_TIMEOUT; exec @ARGV" "$Q" "${BATCH[@]}" \
            --tier tiny --device metal --workers "$W" --quiet \
            >> "$LOG.engine" 2>&1
        rc=$?
        DT=$(( $(python3 -c 'import time;print(int(time.time()*1000))') - T0 ))

        # 自检 1：引擎没吃到图 → 打印 usage。整步无效，必须停。
        if grep -q "usage: qppocr" "$LOG.engine" 2>/dev/null; then
            echo "  ⛔ 引擎没收到图片参数（打印了 usage）—— 脚本拼参数有 bug，整 run 无效。"; reap; exit 4
        fi
        # 自检 2：实际起了几个 worker 进程，必须等于我们要求的 W。
        GOT=$(grep -oE "\[batch\] $N 张图，[0-9]+ 个 worker 进程" "$LOG.engine" 2>/dev/null | tail -1 \
              | grep -oE '[0-9]+ 个' | grep -oE '^[0-9]+')
        if [ -n "$GOT" ] && [ "$GOT" != "$W" ]; then
            echo "  ⚠️ 要求 $W 个进程，实际起了 $GOT 个 —— 这一步的变量没控住，结果别用。"
        fi

        reap
        BW=$(bandwidth)
        printf "  引擎 %6dms(rc=%d)  实际进程 %-3s  复测 blit %6s GB/s  Alloc %s\n" \
               "$DT" "$rc" "${GOT:-?}" "${BW:-?}" "$(gpu_mem)"

        if [ -z "${BW:-}" ]; then echo "  ⚠️ 读不到带宽，停手。"; break; fi
        if [ "$rc" = "142" ]; then
            echo "  → 超时 ${ROUND_TIMEOUT}s 被掐断（本身就是不正常的信号）"
        fi

        # 复查再判死。开机初期系统负载很高（本机实测 load avg 37，Spotlight 等在抢），
        # 单次读数可能被别的东西压到门槛以下，**误判成"已瘸"就要白重启一次**。
        # 瘸态是 ~1 GB/s、健康是 75~95，量级差 70 倍，再测一次就能干净地分开。
        if awk "BEGIN{exit !($BW < $HEALTHY_MIN)}"; then
            printf "  低于门槛（%s < %s），复查一次以排除开机负载干扰…\n" "$BW" "$HEALTHY_MIN"
            sleep 15
            BW2=$(bandwidth)
            printf "  复查 blit %s GB/s\n" "${BW2:-?}"
            [ -n "${BW2:-}" ] && BW=$BW2
        fi

        if awk "BEGIN{exit !($BW < $HEALTHY_MIN)}"; then
            echo
            echo "  ★★ 打瘸了：批大小 **${N} 张**（${W} 进程）就已经扛不住。"
            echo "     已知安全点只有 2 张（实测 81.62 GB/s）。"
            echo "     机器还有救之前已停手 —— **现在重启**，别再喂 GPU。"
            echo "     重启后二选一："
            echo "       a) 更小的批继续夹逼：  bench/gpuhealth/worker_sweep.sh \"4 ${W}\""
            echo "       b) 单变量判定是不是我们自己的 arena 复用 bug（更值得跑）："
            echo "          QPPOCR_METAL_NO_ARENA=1 bench/gpuhealth/worker_sweep.sh \"10 ${W}\""
            break
        fi
        echo "  ✅ 健康"
        sleep 3
    done

    echo
    echo "结束 $(date '+%F %T')   Alloc $(gpu_mem)"
} 2>&1 | tee "$LOG"

echo
echo "日志 → $LOG   （引擎输出在 $LOG.engine）"
