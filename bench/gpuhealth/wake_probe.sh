#!/bin/bash
# 睡眠/唤醒能否免重启恢复？—— 这是唯一还没试过的「免重启恢复」候选。
#
# 背景：「楔死之后没有任何免重启的恢复手段」已用对照实验证过——SIGTERM 杀掉客户端、
# 进程干净退出、等 8 秒，blit 仍是 1.40 GB/s。已提交的 GPU 工作没有取消 API。
# 但**睡眠/唤醒会让显卡驱动重走一遍电源状态机**（进入 S 状态时 GPU 掉电、唤醒时重新
# init + 重建地址翻译），这是唯一有机会把坏掉的驱动态清掉、而又不用重启整机的动作。
# 之前几次楔死都没来得及试，机器就被重启了。所以趁楔死时跑它。
#
# 用法：
#   bench/gpuhealth/wake_probe.sh &            # 先架起来
#   sudo pmset sleepnow                        # 然后让它睡（需要你敲，sudo 要密码）
#   <按任意键唤醒>
#   然后看日志里那条时间断层之后的第一行
#
# 判据：
#   - 醒后 blit 回到 ≳60 GB/s → **免重启恢复成功**，这条以后就是标准恢复手段
#   - 醒后仍是 ~1 GB/s       → 睡眠/唤醒清不掉，这条路否掉
#   - boottime 变了          → 唤醒退化成了重启，本次测试作废（不是失败，是无效）
#
# 为什么打 boottime：睡眠不会改 kern.boottime，重启会。没有这条就会把「其实重启了」
# 误读成「唤醒恢复成功」。
set -uo pipefail
cd "$(dirname "$0")/../.."

INTERVAL=${INTERVAL:-20}
ROUNDS=${ROUNDS:-90}          # 90 × ~25s ≈ 37 分钟窗口，足够你慢慢试
LOG=bench/gpuhealth/logs/wake_probe_$(date +%m%d_%H%M%S).txt
mkdir -p bench/gpuhealth/logs

bandwidth() {
    bench/gpuhealth/gpuhealth 2>/dev/null \
        | grep 'GPU（' | grep -oE '→ *[0-9.]+' | grep -oE '[0-9.]+$' | sort -rn | head -1
}
# ⚠️ 不能用 `grep -oE 'sec = [0-9]+'`：`usec = 765411` 里也含 "sec = "，会匹配出两行。
# 取第一个字段（`{ sec = 1791466088, usec = ... }`）。
boottime() { sysctl -n kern.boottime 2>/dev/null | awk -F'[=,]' '{gsub(/ /,"",$2); print $2}'; }
driver_mem() {
    ioreg -r -c IOAccelerator -l 2>/dev/null \
        | grep -oE '"(Alloc system memory)"=[0-9]+' \
        | awk -F= '{s+=$2} END {printf "%.2fGB", s/1073741824}'
}

{
    echo "════ 睡眠/唤醒恢复实验  起始 $(date '+%F %T') ════"
    echo "轮询间隔 ${INTERVAL}s，最多 $ROUNDS 轮"
    echo "操作：本脚本后台跑着 → sudo pmset sleepnow → 按任意键唤醒 → 回来读这个文件"
    echo
    printf "%-10s %-24s %-12s %-8s %-10s %s\n" "时刻" "blit(GB/s)" "boottime" "recovery" "Alloc" "备注"
} | tee "$LOG"

BOOT0=$(boottime)

for r in $(seq 1 "$ROUNDS"); do
    T=$(date '+%T')
    BW=$(bandwidth)
    BOOT=$(boottime)
    RC=$(ioreg -r -c IOAccelerator -l 2>/dev/null | grep -oE '"recoveryCount"=[0-9]+' | head -1 | cut -d= -f2)
    MEM=$(driver_mem)

    NOTE=""
    if [ "$BOOT" != "$BOOT0" ]; then
        NOTE="⚠️ boottime 变了 —— 机器其实重启了，本次测试作废"
    elif [ -n "$BW" ] && awk "BEGIN{exit !($BW >= 60)}"; then
        NOTE="✅ 健康"
    elif [ -n "$BW" ] && awk "BEGIN{exit !($BW < 10)}"; then
        NOTE="🔴 仍瘸"
    fi

    printf "%-10s %-24s %-12s %-8s %-10s %s\n" "$T" "${BW:-?}" "$BOOT" "${RC:-?}" "${MEM:-?}" "$NOTE" \
        | tee -a "$LOG"

    [ "$BOOT" != "$BOOT0" ] && break
    sleep "$INTERVAL"
done

echo | tee -a "$LOG"
echo "结束 $(date '+%F %T')  日志 → $LOG" | tee -a "$LOG"
