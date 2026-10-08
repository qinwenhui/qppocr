#!/bin/bash
# 重启后的第一件事：重建探针 → 体检 → 跑判别实验。
# 每步之间都过一遍健康门禁；一旦瘸了立刻停手（瘸了继续跑只会浪费时间，反正要重启）。
#
#   ./run_after_reboot.sh           # 默认每个模式 30 秒
#   ./run_after_reboot.sh 60        # 每个模式 60 秒
set -uo pipefail
cd "$(dirname "$0")"
SECS=${1:-30}

echo "═══ 1. 重建探针 ═══"
./build.sh || exit 1

echo
echo "═══ 2. 体检（前测：重启后应该是健康的）═══"
./health.sh || { echo "→ 重启后仍不健康，先别跑任何 GPU 负载，回报。"; exit 1; }


echo
echo "═══ 3. 判别实验：idle 对照（证明「什么都不做」不会掉带宽）═══"
./stress idle "$SECS" | tail -6
./health.sh -q || { echo "→ idle 都会掉！凶手不是我们的负载。回报。"; exit 1; }

for mode in inbounds churn; do
    echo
    echo "═══ 4. 判别实验：$mode ═══"
    ./stress "$mode" "$SECS" | tail -6
    if ./health.sh -q; then
        echo "  → $mode 之后仍然健康：这个负载不是凶手。"
    else
        echo "  → ⚠️ $mode 把访存通路打瘸了。这是关键结果 —— 重启，并把上面的输出留下来。"
        echo "     重启后单独复跑 ./stress $mode $SECS 确认可复现。"
        exit 2
    fi
done

echo
echo "═══ 全部跑完 ═══"
echo "inbounds 与 churn 都没打瘸 → 需要引擎里的具体行为（越界 / 特定地址模式）才能复现。"
