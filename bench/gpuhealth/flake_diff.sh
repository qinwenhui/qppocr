#!/bin/bash
# 同输入、同程序、逐轮结果不同 —— 找**第一处分歧字节**。
#
# 前提（已实测钉死）：
#   · 裁切/打包是共用纯 CPU 代码，CPU 路径 20/20 恒定正确 ⇒ rec 的输入张量确定
#   · rec 计划每 run 同步提交+等待，`run_deferred` 走默认同步实现（输出当场拷走）
#     ⇒ 没有「延迟收账被下一批冲掉」
#   · 所以逐轮变化只可能来自 arena 里**被读到、却没被本计划写过**的字节
#
# `QPPOCR_METAL_IO_DUMP` 在 run 落地后把 rec 计划激活区 `[静态段末, 计划末)`
# 落盘（输入 + 全部中间激活）。拿一次「好」的当基准，逐文件比对**坏**的，
# 第一处分歧的**字下标**就能按 plan dump 反查到是哪一层的输出。
#
# 只保留 1 份好的 + 全部坏的，磁盘占用有界（每轮 ~65 MB）。
#
# 用法： N=16 ./flake_diff.sh
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
Q=$ROOT/target/release/qppocr
MODELS=$ROOT/models
N=${N:-16}
IMG=${IMG:-$(cd testimgs && pwd)/img-008.jpg}
OUT=logs/flake_diff_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

[ -x "$Q" ] || { echo "先构建： cargo build --release -p qppocr-cli --features qppocr/gpu"; exit 1; }
grep -q -a "QPPOCR_METAL_IO_DUMP" "$Q" || { echo "开关没编进二进制，先重建"; exit 3; }
./health.sh -q || { echo "健康门禁没过，先别测"; exit 2; }

PARSE='import json,sys
d=json.load(sys.stdin)[0]
ls=d["lines"]
i=10 if len(ls)>10 else len(ls)-1
print("%d|%s|%.4f" % (d["num_unread"], ls[i]["text"], ls[i]["confidence"]))'

# ⚠️ 判定**不能只看 num_unread**：翻车还有一种形态是「文本错了但 unread=0」
# （实测 `0|ANY|0.4534`）。所以基准取**首轮的结果字符串**，与它不同即为坏。
REF=""
for i in $(seq 1 "$N"); do
    d="$OUT/run$(printf '%03d' "$i")"
    mkdir -p "$d"
    res=$(QPPOCR_METAL_IO_DUMP="$d/a" "$Q" "$IMG" --tier tiny --device metal \
          --models "$MODELS" --workers 1 --quiet --json 2>"$d/err.txt" \
          | python3 -c "$PARSE" 2>/dev/null)
    printf '%s\n' "$res" >"$d/result.txt"
    [ -z "$REF" ] && REF="$res"
    if [ "$res" = "$REF" ] && [ "$i" != 1 ]; then
        rm -rf "$d"          # 与首轮一致的只留第一份
        printf '  run%03d  %s  (与首轮同，已删)\n' "$i" "$res"
    else
        printf '  run%03d  %s%s\n' "$i" "$res" "$([ "$i" = 1 ] && echo '  ← 基准')"
    fi
done

GOOD=$(ls -d "$OUT"/run* 2>/dev/null | head -1)
echo
echo "== 基准（好）= $GOOD =="
python3 - "$OUT" "$GOOD" <<'PY'
import sys, os, glob
out, good = sys.argv[1], sys.argv[2]
SKIP = {"result.txt", "err.txt"}
base = {
    os.path.basename(p): p
    for p in glob.glob(good + "/*")
    if os.path.basename(p) not in SKIP
}
bad = [d for d in sorted(glob.glob(out + "/run*")) if d != good]
print("   坏轮次：%d 份" % len(bad))
seen = set()
for bd in bad:
    res = open(bd + "/result.txt").read().strip()
    print("\n── %s  结果 %s ──" % (os.path.basename(bd), res))
    for p in sorted(glob.glob(bd + "/*")):
        if os.path.basename(p) in SKIP:
            continue
        nm = os.path.basename(p)
        if nm not in base:
            print("   %-34s 基准里没有同名文件" % nm); continue
        a = open(p, 'rb').read(); c = open(base[nm], 'rb').read()
        if a == c:
            continue
        n = min(len(a), len(c))
        diff = [i for i in range(0, n, 4) if a[i:i+4] != c[i:i+4]]
        first = diff[0]
        fa = int.from_bytes(a[first:first+4], 'little')
        fc = int.from_bytes(c[first:first+4], 'little')
        import struct
        print("   %-34s 差 %6d 字 / 共 %6d 字   首差 @字 %d  (坏=0x%08x %.6g | 好=0x%08x %.6g)"
              % (nm, len(diff), n // 4, first // 4, fa, struct.unpack('<f', a[first:first+4])[0],
                 fc, struct.unpack('<f', c[first:first+4])[0]))
        seen.add(nm)
print()
print("   早于基准全同的文件（无分歧）：%d 份" % len(set(base) - seen))
PY
echo
echo "  明细：$OUT"
echo "  ⚠️ 先记录结论再删——65 MB/轮"
