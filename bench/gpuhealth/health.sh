#!/bin/bash
# GPU 健康门禁。退出码：0 健康 / 1 存疑 / 2 已瘸。
# 用法： ./health.sh          # 只看结果
#        ./health.sh -q     # 静默模式，只给退出码（给脚本当门禁用）
set -uo pipefail
cd "$(dirname "$0")"
QUIET=${1:-}

[ -x ./gpuhealth ] || ./build.sh >/dev/null 2>&1 || { echo "编译失败"; exit 2; }

out=$(./gpuhealth)
gpu=$(echo "$out" | awk '/^GPU（/{for(i=1;i<=NF;i++) if($i=="GB/s") print $(i-1)}')
rec=$(ioreg -r -c IOAccelerator -l 2>/dev/null | grep -o '"recoveryCount"=[0-9]*' | head -1 | cut -d= -f2)

if [ -z "$gpu" ]; then echo "读不到带宽，探针异常"; exit 2; fi

# ⚠️ 判据是**实测带宽**，`recoveryCount` 只是提示，**不单独否决**。
# 2026-10-08 实测过反例：重启后 recoveryCount 干净回到 0、无任何 gpuEvent，blit 却只有
# 0.25 GB/s；反过来今天 recoveryCount=1（历次 page fault 的残留，**不重启不清零**）
# 而带宽 88.7~91.4，健康。把计数器放在带宽之前判死，会让门禁**恒返回 2**、彻底失效。
verdict=0; msg=""
if   awk "BEGIN{exit !($gpu >= 60)}"; then verdict=0; msg="健康 —— 可以放心测性能"
elif awk "BEGIN{exit !($gpu >= 10)}"; then verdict=1; msg="存疑（10~60 GB/s）—— 复跑一次；别用来出性能数字"
else verdict=2; msg="访存通路已被打瘸 → 立刻停手，重启（没有任何免重启的恢复手段）"
fi

if [ "$QUIET" != "-q" ]; then
    echo "$out"
    echo "  recoveryCount = ${rec:-未知}"
    [ "${rec:-0}" != "0" ] && echo \
      "  ℹ️  本纪元有过 GPU 页故障（该计数器不重启不清零）；本轮带宽 $gpu GB/s 已达标，以带宽为准"
    case $verdict in
        0) echo "  ✅ $msg" ;;
        1) echo "  ⚠️  $msg" ;;
        2) echo "  ❌ $msg" ;;
    esac
fi
exit $verdict
