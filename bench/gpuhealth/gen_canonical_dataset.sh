#!/bin/bash
# 生成竞品 SimdPaddleOCR 的 **canonical 100 张**数据集（他们 docs/metal-m4.md 用的那份）。
#
# 为什么：他们的 `dataset/` 不在仓库里（.gitignore 掉了），导致「图集不同所以绝对值
# 不可比」长期挡着结论。但他们的 `test/Sdcb.SimdPaddleOCR.TestData` 是**确定性**的
# （`Program.cs:25 const int Seed = 20260830;`），尺寸由 rng 前两抽决定、早于任何
# 字体，所以**自己能造出来**：
#
#   2026-10-10 实测：本脚本产出的尺寸与我们的 bench/gpuhealth/corpus100 **逐张
#   100/100 完全相同** ⇒ corpus100 就是这份 canonical 集（只差字体）。
#   据此在 gen100 上跑**他们的代码**，tiny 复现（39.0 vs 文档 39.3），
#   small/medium 复现不了（68.6 vs 54.6；257.6 vs 107.0）。
#   详见 logs/competitor_metal_fp16_tiers_20261010.md §26.8-26.9。
#
# 用法： ./gen_canonical_dataset.sh [输出目录，默认 bench/gpuhealth/gen100]
#
# 字体：他们的候选全是 Windows 路径（C:\Windows\Fonts\...），本机没有，所以要给一个
# CJK 字体目录。macOS 自带 Hiragino Sans GB 够用。**字体只影响像素内容、不影响尺寸**，
# 所以拿它做「尺寸/行数」级复现是合法的；**别拿 gen100 判像素级一致性**。
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-"$HERE/gen100"}
COMP=${SIMDPADDLEOCR_DIR:-"$HOME/Projects/SimdPaddleOCR"}
FONTS="$HERE/fonts_cjk"

[ -d "$COMP" ] || { echo "找不到竞品仓库：$COMP（用 SIMDPADDLEOCR_DIR 覆盖）" >&2; exit 1; }

# macOS 自带的 CJK 字体；换别的也行，只是改像素
if [ ! -e "$FONTS/Hiragino Sans GB.ttc" ]; then
    mkdir -p "$FONTS"
    ln -sf "/System/Library/Fonts/Hiragino Sans GB.ttc" "$FONTS/" 2>/dev/null || true
fi
[ -d "$FONTS" ] && [ -n "$(ls -A "$FONTS" 2>/dev/null)" ] || {
    echo "没有可用的 CJK 字体；装一个 .ttc/.ttf 到 $FONTS" >&2; exit 1; }

# .NET：他们用 net10.0。apphost 会报 "You must install .NET" 除非设 DOTNET_ROOT。
export DOTNET_ROOT="${DOTNET_ROOT:-$HOME/.dotnet}"
DOTNET="$DOTNET_ROOT/dotnet"
[ -x "$DOTNET" ] || DOTNET=dotnet

PROJ="$COMP/test/Sdcb.SimdPaddleOCR.TestData/Sdcb.SimdPaddleOCR.TestData.csproj"
echo "构建生成器…"
"$DOTNET" build "$PROJ" -c Release -v q --nologo

mkdir -p "$OUT"
echo "生成到 $OUT …"
PPOCR_FONTS_DIR="$FONTS" "$DOTNET" \
    "$COMP/test/Sdcb.SimdPaddleOCR.TestData/bin/Release/net10.0/Sdcb.SimdPaddleOCR.TestData.dll" \
    --out "$OUT"

echo
echo "完成：$(ls "$OUT"/img-*.jpg 2>/dev/null | wc -l | tr -d ' ') 张图 → $OUT"
