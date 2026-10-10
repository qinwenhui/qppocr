#!/bin/bash
# 参数块判据蹲翻车：同一张图连跑，只要出现任一异常就停、把证据留下并打印。
#
# 三条判据是同一件事的三个投影，粗细不同：
#   ① `num_unread != 0`：真丢了一行（最粗，需要整条链路都踩中）
#   ② `[verify]`：派发前把计划的**静态清单与 arena 实际内容逐字比对**，
#      出现 `静态 X/Y 块不在位`（X>0）= 账本说谎（早退判据是 `Arc` 指针不是内容）。
#      **这条现在是拒发而不是只打印**（错误参数能让内核跑成巨循环、把 GPU 打瘸），
#      所以坏轮的 out.json 是空的、`num_unread` 读成 `?`——照样算异常。
#   ③ `[exit-param] 应有 != 实有`：出口节点 `n_exit3_argmax` 的参数块
#      （`src_off/out_off/rows/real_cols/cpad/rpb`）与计划自己那一份不符
#      ⇒ 出口会按**别的计划**的 `out_off`/`rows` 写 ⇒ 那几十字根本没写在该在的地方
#
# 为什么必须看内容：坏轮的直证是「相对激活区 79484..79515 那 32 字全程等于环境值」
# （`flake_watch.sh` 的 trace），而命令缓冲是 `wait_until_completed` + 状态 `Completed`、
# 网格/`p_off` 都来自逐位相同的计划 ⇒ 能变的只有参数块里的值。
#
# 用法： N=120 IMG=.../img-002.jpg ./flake_param.sh
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
Q=$ROOT/target/release/qppocr
N=${N:-120}
IMG=${IMG:-$(cd testimgs && pwd)/img-002.jpg}
OUT=logs/flake_param_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

[ -x "$Q" ] || { echo "先构建"; exit 1; }
for v in QPPOCR_METAL_STATIC_VERIFY QPPOCR_METAL_EXIT_PARAM; do
    grep -q -a "$v" "$Q" || { echo "探针 $v 没编进二进制，先重建"; exit 3; }
done
./health.sh -q || { echo "健康门禁没过"; exit 2; }

echo "== 蹲翻车（参数块判据）：$(basename "$IMG") × ${N} =="
found=""
for i in $(seq 1 "$N"); do
    d="$OUT/run$(printf '%03d' "$i")"
    mkdir -p "$d"
    QPPOCR_METAL_STATIC_VERIFY=1 QPPOCR_METAL_EXIT_PARAM=1 QPPOCR_METAL_STATIC_DEBUG=1 \
        "$Q" "$IMG" --tier tiny --device metal --models "$ROOT/models" \
        --workers 1 --quiet --json >"$d/out.json" 2>"$d/err.txt"
    read -r nu bad mismatch < <(python3 - "$d/err.txt" "$d/out.json" <<'PY'
import json, re, sys
err, js = sys.argv[1], sys.argv[2]
try:
    nu = json.load(open(js))[0]["num_unread"]
except Exception:
    nu = "?"
bad = 0
mm = 0
for ln in open(err, errors="replace"):
    if "[verify]" in ln and not re.search(r"静态 0/", ln):
        bad += 1
    m = re.search(r"应有\[([^\]]*)\]实有\[([^\]]*)\]", ln)
    if m and m.group(1).strip() != m.group(2).strip():
        mm += 1
print(nu, bad, mm)
PY
)
    printf 'run%03d unread=%s verify_bad=%s param_bad=%s\n' "$i" "$nu" "$bad" "$mismatch"
    if [ "$nu" != "0" ] || [ "$bad" != "0" ] || [ "$mismatch" != "0" ]; then
        found=$i
        echo "  ✖ 第 $i 轮异常 → 证据留在 $d"
        grep -a -A4 "verify\].*静态 [1-9]" "$d/err.txt" | sed 's/^/     /'
        grep -a "exit-param\]" "$d/err.txt" | sed 's/^/     /' | head -12
        break
    fi
    [ "$i" != 1 ] && rm -rf "$d"   # 留首轮当对照
done

if [ -n "$found" ]; then
    echo
    echo "== 好轮（run001）对照 =="
    grep -a "exit-param\]" "$OUT/run001/err.txt" | sed 's/^/   /'
    echo
    echo "== 坏轮（run$(printf '%03d' "$found")）=="
    grep -a "exit-param\]" "$OUT/run$(printf '%03d' "$found")/err.txt" | sed 's/^/   /'
    python3 - "$OUT/run001/out.json" "$OUT/run$(printf '%03d' "$found")/out.json" <<'PY'
import json, sys, difflib
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None
if load(sys.argv[2]) is None:
    print("\n   坏轮在派发前就被拒了（静态区不在位）⇒ 没有 JSON 可对——"
          "这本身就是判据 ② 成立的最干净形态")
    sys.exit(0)
g = json.load(open(sys.argv[1]))[0]; b = json.load(open(sys.argv[2]))[0]
gt = [l["text"] for l in g["lines"]]; bt = [l["text"] for l in b["lines"]]
print("\n   好轮 %d 行 / 坏轮 %d 行（unread %d → %d）"
      % (len(gt), len(bt), g["num_unread"], b["num_unread"]))
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gt, bt).get_opcodes():
    if tag != "equal":
        print("   %-8s 好[%d:%d] %r / 坏[%d:%d] %r" % (tag, i1, i2, gt[i1:i2], j1, j2, bt[j1:j2]))
PY
else
    echo "  ${N} 轮三条判据全干净（没蹲到）"
fi
echo "  明细：$OUT"
