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

verdict=0; msg=""
if   [ "$rec" != "0" ]; then verdict=2; msg="recoveryCount=$rec（有过 GPU 页故障）→ 重启，重启前的所有数字作废"
elif awk "BEGIN{exit !($gpu >= 60)}"; then verdict=0; msg="健康 —— 可以放心测性能"
elif awk "BEGIN{exit !($gpu >= 10)}"; then verdict=1; msg="存疑（10~60 GB/s）—— 复跑一次；别用来出性能数字"
else verdict=2; msg="访存通路已被打瘸 → 立刻停手，重启（没有任何免重启的恢复手段）"
fi

if [ "$QUIET" != "-q" ]; then
    echo "$out"
    echo "  recoveryCount = ${rec:-未知}"
    case $verdict in
        0) echo "  ✅ $msg" ;;
        1) echo "  ⚠️  $msg" ;;
        2) echo "  ❌ $msg" ;;
    esac
fi
exit $verdict
