#!/bin/bash
# 抓一次「健康基线」，存成可直接对比的文件。
#
# 为什么必须有它：我犯过一个方法错误——判定「算力正常」时用的是**瘸态下测的**
# 690 GFLOP/s，拿它跟一个**假设**的降频倍数（1/68 → 只该有 60）比，而不是跟
# 真实基线比。690 到底是正常还是已经被压过 3~5 倍，没有基线就无从判断。
# 所以每次重启后的第一件事就是跑这个，先把标尺钉下来。
#
# 用法： bench/gpuhealth/baseline.sh [标签]     默认标签 = reboot1
set -uo pipefail
cd "$(dirname "$0")"

TAG=${1:-reboot1}
OUT="logs/baseline_${TAG}.txt"
[ -x ./gpuhealth ] || ./build.sh >/dev/null

{
    echo "════════ GPU 健康基线  label=$TAG  $(date '+%F %T') ════════"
    echo "开机时长： $(uptime | sed 's/.*up/up/;s/,.*load/  load/')"
    echo

    echo "──── 1. 读写带宽（苹果自己的 blit 256 MB，健康应 ≳83 GB/s）────"
    ./gpuhealth

    echo
    echo "──── 2. ★ 纯 ALU 吞吐（算力基线 —— 之前一直缺的就是这个）────"
    echo "   M4 十核 GPU fp32 峰值量级 ~4 TFLOP/s；这个 6 链 FMA 循环正常应到 2~3 TFLOP/s。"
    ./gpualu2

    echo
    echo "──── 3. 带宽 vs 延迟 + 尺寸曲线 ────"
    ./gpulat

    echo
    echo "──── 4. 四条访存路径（DMA/SM × 读+写/纯写，64 MB）────"
    ./gpuaxes

    echo
    echo "════ 完 ════"
} 2>&1 | tee "$OUT"

echo
echo "已存 → $OUT"
echo "★ 以后任何「变慢了」的判断，都拿这份文件当右列，不要凭感觉。"
