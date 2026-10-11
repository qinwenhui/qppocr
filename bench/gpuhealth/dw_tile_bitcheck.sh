#!/bin/bash
# `n_conv_dw_tile` 的**逐位对拍**：det 输出必须一字不差。
#
# 为什么可以要求逐位（而不是"数值等价"）：瓦片版把热循环的数据来源从设备
# 内存换成线程组内存，但
#   1) 累加次序一字未改（都是 `t` 行主序 ⇒ `dy` 外层 / `dx` 内层）；
#   2) 每个在界 tap 的乘数与 `n_conv_dw` 逐字同式；
#   3) 越界 tap 由 `continue` 变成加 `0.0*w`，而 `acc` 从 `+0.0` 起步、IEEE
#      加法只有「两个 -0.0 相加」才得 -0.0 ⇒ `acc` 取值域里没有 -0.0
#      ⇒ `acc + ±0.0 == acc` 恒成立。
# 完整证明在 `crates/qppocr-metal/shaders/n_conv_dw_tile.metal` 头注释。
#
# **det 是确定性的**（不像 rec：同一二进制连跑两遍 10 图里 7 图文本会变，
# 见 memory `metal-rec-output-is-nondeterministic`），所以这里的判据是
# `cmp` 字节相等，不是"行数/框数一致"。
#
# 用法：TIER=medium ./dw_tile_bitcheck.sh [图数]       （默认 30）
set -uo pipefail
cd "$(dirname "$0")"
N=${1:-30}
TIER=${TIER:-medium}
BIN=${BIN:-../../target/release/qppocr}

# 陈旧二进制门禁：开关没编进去时两臂跑同一份代码，必然"逐位相同"——那是
# 假通过。见 memory `ab-null-may-be-stale-binary`。
if [ "$(grep -c -a -- "QPPOCR_METAL_DW_TILE" "$BIN" || true)" -eq 0 ]; then
    echo "❌ $BIN 里没有 QPPOCR_METAL_DW_TILE —— 门禁假通过，先重建。" >&2
    exit 3
fi

echo "# TIER=$TIER  $N 图  逐位对拍 QPPOCR_METAL_DW_TILE=0(基准) vs 1(瓦片)"
# 临时文件放本目录（探针产物一律不许进 /tmp，见仓库纪律）
OFF=$(mktemp ./dwt_off.XXXXXX); ON=$(mktemp ./dwt_on.XXXXXX)
trap 'rm -f "$OFF" "$ON"' EXIT
bad=0
# 去掉 `timing` 块：它装的正是本脚本要量但天然逐次不同（且与对错无关）的
# 东西。第一次跑忘了剥，30/30 全"有差异"而框其实一字不差——**差异全在
# `total_ms`/`det_infer_ms` 上**。剥法是解析 JSON 后就地删键再规范化键序，
# 不用正则（`timing` 里的数字位数会变，字符串替换不可靠）。
STRIP='import sys,json
d=json.load(sys.stdin)
for x in d: x.pop("timing",None)
json.dump(d,sys.stdout,sort_keys=True)'

for f in $(ls corpus100/img-*.jpg | head -"$N"); do
    QPPOCR_METAL_DW_TILE=0 "$BIN" "$f" --models ../../models --device metal \
        --tier "$TIER" --det-only --json --boxes --workers 1 2>/dev/null \
        | python3 -c "$STRIP" >"$OFF"
    QPPOCR_METAL_DW_TILE=1 "$BIN" "$f" --models ../../models --device metal \
        --tier "$TIER" --det-only --json --boxes --workers 1 2>/dev/null \
        | python3 -c "$STRIP" >"$ON"
    if ! cmp -s "$OFF" "$ON"; then
        bad=$((bad + 1))
        echo "  ✗ $f 有差异"
        # 只报第一处差异的行号与前 120 字符，别把整份 JSON 倒出来
        diff <(fold -w 200 "$OFF") <(fold -w 200 "$ON") | head -4
    fi
done
echo
if [ "$bad" -eq 0 ]; then
    echo "✅ $N 图 det 输出逐位相同"
else
    echo "❌ $bad/$N 图有差异 —— 逐位性被破，先查再谈性能"
fi
exit "$bad"
