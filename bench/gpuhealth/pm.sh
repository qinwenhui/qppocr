#!/bin/bash
# 采一次「GPU 频率 / 功耗 / 节流」样本——这是唯一需要 root 的探针。
#
# 为什么必须有它：瘸态下量到「算力掉 4.46×」（3083.99 → 690.79 GFLOP/s），
# 但**没有绝对频率**就说不清这是不是「核心被压在最低频档」（1470/330 ≈ 4.45 正好对上）。
# ioreg 只暴露 IOReport 的**图例**（通道 id + 单位标志），不暴露数值；
# 读数值只能靠 root 的 powermetrics。而 `sudo -n` 在非交互环境不可用，必须人来敲。
#
# 用法（需要 sudo，会提示输密码）：
#   sudo bench/gpuhealth/pm.sh <标签> [载荷秒数]
#
#   <标签>      存成 logs/pm_<标签>.txt，建议用 healthy / wedged
#   [载荷秒数]  默认 6 秒。**要读「满频是多少」就必须给载荷**——空闲时 GPU 自然在低频，
#               那是没意义的数字。0 = 不加载荷。
#   [载荷种类]  inbounds（默认，健康态）/ alu（**瘸态用**）/ none
#
#   瘸态样例：  sudo bench/gpuhealth/pm.sh wedged 6 alu
#
# 载荷用 stress inbounds：已实测过它打不瘸机器（后测 94 GB/s），是安全的 GPU 负载。
set -uo pipefail
cd "$(dirname "$0")"

if [ "$(id -u)" -ne 0 ]; then
    echo "需要 root：  sudo bench/gpuhealth/pm.sh ${1:-healthy} ${2:-6}" >&2
    exit 1
fi

TAG=${1:-sample}
LOAD=${2:-6}
LOADKIND=${3:-inbounds}   # inbounds（默认，健康态用）| alu（**瘸态用**，几乎零访存）| none
OUT="logs/pm_${TAG}.txt"
START=$SECONDS
mkdir -p logs

# powermetrics 的输出会带 ANSI 颜色码，去掉便于 grep
run_pm() { powermetrics -n "$1" -i 1000 --samplers gpu_power 2>&1 | sed 's/\x1b\[[0-9;]*m//g'; }

echo "══ powermetrics 样本  label=$TAG  载荷 ${LOAD}s  $(date '+%F %T') ══"

if [ "$LOAD" -gt 0 ] && [ "$LOADKIND" != "none" ]; then
    # 先起采样，再压 GPU —— 让样本落在真正有负载的窗口里
    ( run_pm $((LOAD + 3)) > "$OUT" ) &
    PM=$!
    sleep 1
    case "$LOADKIND" in
      alu)
        # 纯 ALU 载荷：几乎零访存。**瘸态下用它** —— stress inbounds 是 memory-bound 的，
        # 往一条已经瘸掉的访存通路继续灌流量没意义、还可能多拖 WindowServer。
        echo "  （正在跑 $LOAD 秒纯 ALU 载荷：gpualu2 循环）"
        [ -x ./gpualu2 ] || ./build.sh >/dev/null 2>&1
        while [ $SECONDS -lt $((START + LOAD)) ]; do ./gpualu2 >/dev/null 2>&1; done
        ;;
      *)
        echo "  （正在跑 $LOAD 秒访存载荷：stress inbounds）"
        [ -x ./stress ] || ./build.sh >/dev/null 2>&1
        ./stress inbounds "$LOAD" >/dev/null 2>&1
        ;;
    esac
    wait $PM
else
    run_pm 3 > "$OUT"
fi

echo
echo "──── 频率 / 驻留 / 节流（关键行）────"
# 两类字段都要：HW 侧（硬件实际落在哪个档）+ SW 侧（驱动 DVFS 状态机请求了哪个档）。
# 瘸态下这两行的关系是判据：SW 请求高档但 HW 掉低档 → 硬件/固件；SW 自己就请求低档 → 驱动状态机。
grep -iE "freq|resid|throttle|power|SW state|SW requested|GPU Energy" "$OUT" | head -60

echo
echo "──── 若上面是空的，看原始文件找 sampler 名 ────"
echo "    $OUT"
head -20 "$OUT" | sed 's/^/    /'
echo
echo "★ 把 gpu_power 采样器的可用项确认一下： powermetrics --help 2>&1 | grep -A20 samplers"
