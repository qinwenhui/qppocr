#!/bin/bash
# 编译全部 GPU 探针。重启后 /tmp 被清空，跑这一句就全回来了（约 3 秒）。
set -euo pipefail
cd "$(dirname "$0")"
CC=${CC:-clang}
FLAGS=(-fobjc-arc -framework Foundation -framework Metal -O2)

for src in gpuhealth gpulat gpuaxes gpualu2 stress membw cpumin; do
    [ -f "$src.m" ] || { echo "跳过 ${src}（源码不在）"; continue; }
    # AppKit 不需要；gpualu2 输出名为 gpualu2，其余同名
    "$CC" "${FLAGS[@]}" -o "$src" "$src.m"
    echo "built  $src"
done
