#!/bin/bash
# 复现/验证「跨计划静态区被冲掉 → 缓存命中的计划读到垃圾参数 → GPU 挂死」这条
# 致瘸路径的修复。
#
# 判定口径（跑之前先定好，避免事后解释）：
#   ① 10 张图塞进**一个进程、一个会话**（--workers 1）——比当初致瘸的
#      arm B（10 图/2 进程 = 5 张一会话）更狠，必须 rc=0 跑完；
#   ② 「缓存命中重放静态区」必须**发生**（>0 次）：图 2 的形状命中图 1 建的
#      计划，修复就是在这一步把被冲掉的参数块写回去；
#   ③ metal **同设备自洽**：把每张图单独跑一遍（计划全新构建，走不到重放路径），
#      输出必须与会话内（命中缓存 + 重放静态区）逐行、逐 conf **完全相同**。
#      这是修复本身的正确性口径。CPU 对拍只作**软**判据（见 ⑤）。
#   ④ 跑前跑后都要过健康门禁。
#   ⑤（软）metal 与 cpu 去掉空白后必须逐行相同。为什么不要求逐字相同：
#      像素空格（rec.rs `add_gap_spaces`）读的是**逐列墨迹剖面** `ink[x] < EMPTY_INK`，
#      CPU/GPU 浮点求和次序不同，临界列上「最长空白 run ≥ min_gap_px」会翻转，
#      表现为同一行少/多一个空格（实测 img-003 首行 `超过320` vs `超过 320`，
#      框完全一致、conf 0.9850 vs 0.9715）。这是既有差异，不是本修复引入的。
#
# 用法： ./fixcheck.sh [图片目录，默认 ./testimgs] [张数，默认 10]
#
# ⚠️ 环境坑（2026-10-08 实测）：/Users/qinwh/Pictures/dataset 在 iCloud Drive
# 里，同步窗口内读文件会间歇性 ENOENT（连 `qppocr 单图` 都一样，CPU 路径也
# 一样）。所以本脚本用 benchmark 目录下的**本地副本**，并且对每步做重试。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)   # 仓库根：模型路径必须以它为基准，否则相对路径按 CWD 解
cd "$HERE"
IMGDIR=${1:-./testimgs}
N=${2:-10}
Q=$ROOT/target/release/qppocr
MODELS=$ROOT/models            # 本脚本 cd 进了 bench 目录，模型必须给绝对路径
WORK=logs/fixcheck_$(date +%Y%m%d_%H%M%S)
mkdir -p "$WORK"

[ -x "$Q" ] || { echo "先构建： cargo build --release -p qppocr-cli --features qppocr/gpu"; exit 1; }
command -v python3 >/dev/null || { echo "需要 python3"; exit 1; }
for f in det.onnx rec.onnx; do
    [ -f "$MODELS/tiny/$f" ] || { echo "缺模型：$MODELS/tiny/${f}（用 --models 指定或补齐 models/）"; exit 1; }
done

IMGS=()
for i in $(seq 1 "$N"); do
    p="$(cd "$IMGDIR" && pwd)/$(printf 'img-%03d.jpg' "$i")"
    [ -f "$p" ] || { echo "缺图：$p"; exit 1; }
    IMGS+=("$p")
done

# 跑一次，环境抖动（io ENOENT）自动重试——抖动时**连解码都进不去**，不是引擎问题。
# ⚠️ 判据必须窄到「读图」那一条：模型路径不存在时也报 "No such file or directory"，
#    曾因此把配置错误当成抖动、白重试 3 次还掩盖了真因（见 README 的坑位记录）。
run_retry() { # $1=输出前缀 ; 其余=参数
    local pre=$1; shift
    for try in 1 2 3; do
        "$Q" "$@" >"$pre.out" 2>"$pre.err"
        local rc=$?
        if [ $rc -eq 0 ]; then echo "$try"; return 0; fi
        if ! grep -q "^error: io: No such file or directory" "$pre.err"; then echo "$try"; return $rc; fi
        sleep 2   # 抖动窗口，等一拍
    done
    echo "$try"; return 1
}

echo "== 修复验证：$N 张图 / 一个进程 / 一个会话 =="
./health.sh -q; pre_h=$?
echo "  跑前健康门禁 rc=$pre_h"

t0=$(date +%s.%N)
try=$(run_retry "$WORK/metal" "${IMGS[@]}" --tier tiny --device metal \
        --models "$MODELS" --workers 1 --quiet --json)
rc=$?
t1=$(date +%s.%N)
rep=$(grep -c '缓存命中重放静态区' "$WORK/metal.err")
echo "  metal rc=${rc}（重试 ${try} 次）墙钟 $(echo "$t1-$t0" | bc) s 重放 ${rep} 次"

try=$(run_retry "$WORK/cpu" "${IMGS[@]}" --tier tiny --device cpu \
        --models "$MODELS" --workers 1 --quiet --json)
rc_c=$?
echo "  cpu   rc=${rc_c}（重试 ${try} 次）"

# ③ 的参照：每张图单独一个进程。计划从零构建 → blobs == recipe → 不触发重放，
# 拿到的是「不复用」的基准输出。
mkdir -p "$WORK/alone"
for i in $(seq 1 "$N"); do
    run_retry "$WORK/alone/$(printf '%03d' "$i")" "${IMGS[$((i - 1))]}" \
        --tier tiny --device metal --models "$MODELS" --workers 1 --quiet --json \
        >/dev/null || true
done

./health.sh -q; post_h=$?
echo "  跑后健康门禁 rc=$post_h"

python3 - "$WORK" "$N" <<'PY'
import json, re, sys, pathlib
work, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])

def by_image(*paths):
    out = {}
    for p in paths:
        for x in json.loads(p.read_text()):
            out[x['image'].split('/')[-1]] = x['lines']
    return out

def norm(t):  # 去空白后比较（见 ⑤ 的说明）
    return re.sub(r'\s+', '', t)

try:
    m = by_image(work / 'metal.out')
    c = by_image(work / 'cpu.out')
    a = by_image(*sorted((work / 'alone').glob('*.out')))
except Exception as e:
    print(f"  ✖ 输出不可解析：{e}"); raise SystemExit(3)

# ③ 自洽：会话内（重放路径）vs 单独跑（新计划）
diffs, maxdc, nline = [], 0.0, 0
for img in sorted(m):
    if img not in a:
        diffs.append((img, '单独跑缺这张')); continue
    for x, y in zip(m[img], a[img]):
        nline += 1
        if x['text'] != y['text'] or x['rotation'] != y['rotation']:
            diffs.append((img, f"{x['text']!r} vs {y['text']!r}"))
        maxdc = max(maxdc, abs(x['confidence'] - y['confidence']))
print(f"  ③ 自洽：{len(m)} 图 / {nline} 行，最大 |Δconf| = {maxdc:.6f}")
print("     ✅ 会话内(重放) == 单独跑(新计划)" if not diffs
      else f"     ✖ {len(diffs)} 行不同，首处：{diffs[0]}")
ok3 = 0 if not diffs and maxdc == 0 else 1

# ⑤ 软：与 CPU 去空白后比较
hard, soft, ncpu = [], [], 0
for img in sorted(m):
    for x, y in zip(m[img], c.get(img, [])):
        ncpu += 1
        if x['text'] != y['text']:
            (soft if norm(x['text']) == norm(y['text']) else hard).append(
                (img, f"{x['text']!r} vs {y['text']!r}"))
print(f"  ⑤ 对拍 CPU：{ncpu} 行；空白差异 {len(soft)} 行 / 实质差异 {len(hard)} 行")
if soft: print(f"     空白差异属既有浮点临界，例：{soft[0][1]}")
if hard: print(f"     ✖ 实质差异首处：{hard[0]}")

(work / 'v3').write_text(str(ok3))
(work / 'v5').write_text(str(0 if not hard else 1))
PY
ok3=$(cat "$WORK/v3" 2>/dev/null || echo 1)
ok5=$(cat "$WORK/v5" 2>/dev/null || echo 1)

echo
echo "== 判定 =="
v=0
[ "$rc" = 0 ]  && echo "  ✅ ① 10 图一会话跑完"        || { echo "  ❌ ① rc=$rc"; v=1; }
[ "$rep" -gt 0 ] && echo "  ✅ ② 触发了缓存命中重放（$rep 次）" || { echo "  ❌ ② 没触发重放——测试没覆盖到那条路径"; v=1; }
[ "$ok3" = 0 ] && echo "  ✅ ③ 会话内(重放) == 单独跑(新计划)" || { echo "  ❌ ③ 重放路径与全新计划不一致"; v=1; }
# ⚠️ `${post_h}` 的花括号是**必须**的：`$post_h）` 紧贴全角右括号时，bash 在非 UTF-8
# locale 下把 `）` 的首字节 0xEF 当成合法变量名字符，于是去找变量 `post_h\xef` →
# `post_h: unbound variable`，脚本在最后一行崩掉（2026-10-10 实测）。凡 `$VAR` 后紧跟
# 中日韩全角字符，一律写成 `${VAR}`。
[ "$pre_h" = 0 ] && [ "$post_h" = 0 ] && echo "  ✅ ④ 跑前跑后都健康" || { echo "  ❌ ④ 健康门禁没过（跑前 ${pre_h} / 跑后 ${post_h}）"; v=1; }
[ "$ok5" = 0 ] && echo "  ✅ ⑤ 与 CPU 去空白后一致"   || { echo "  ❌ ⑤ 与 CPU 有实质差异"; v=1; }
echo "  日志：$WORK"
exit $v
