#!/bin/bash
# N 路**轮转交错** A/B/C…：一次会话里把 K 个变体全量出来，漂移靠轮转抵消。
#
# 为什么不是连跑 K 次 ab.sh：本机整轮漂移极大（2026-10-09 实测一个 8 轮会话里
# det 从 30.5 漂到 48.8，+60%；同一台机另一会话在跑任务时更凶）。两次 ab.sh
# 之间机器状态不可比，于是「T_conv8 = 116」和「T_other = 43」会因为取自不同
# 状态而互相矛盾（116+43=159 ≠ 135 实测的 T_all）。同一个会话里轮转，K 个
# 变体在时间上对称分布，比的是**配对差值**，漂移只在每轮内部影响绝对水平。
#
# 轮转用拉丁方：第 r 轮把变体序列左移 (r-1) mod K。这样每个变体在每个
# 「轮内位置」上出现次数尽量均等 —— ab.sh 里那段「一轮里先跑的更快 6~15%」
# 的次序效应因此在各变体间对称。
#
# 用法： ./abc.sh <轮数> <图片数> <bin1> <bin2> [...] [-- 额外参数…]
# 例：   ./abc.sh 6 20 ./bin/qppocr-cur ./bin/qppocr-c8only ./bin/qppocr-rec8off
#
# 判读（第 1 个 bin 是基准，差值 = 各变体 − 基准，负 = 更快）：
#   · 每个配对的 **det 列**是控制列（这些变体都只动 rec）—— |Δdet|>3 的轮次
#     整轮丢弃，见 memory `ab-variance-must-be-two-sided`；
#   · 只看留存轮次的**中位配对差值**与**离散**，不看均值。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$HERE"

ROUNDS=${1:?用法: ./abc.sh <轮数> <图片数> <bin1> <bin2> [...] [-- 额外参数…]}
N=${2:?}
shift 2
BINS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do BINS[${#BINS[@]}]="$1"; shift; done
[ "${1:-}" = "--" ] && shift
[ "${#BINS[@]}" -ge 2 ] || { echo "至少两个变体"; exit 1; }

MODELS=$ROOT/models
LOG=logs/abc_$(date +%Y%m%d_%H%M%S).log
TSV=$(mktemp)
IMGS=()
while IFS= read -r f; do IMGS[${#IMGS[@]}]="$f"; done < <(ls "$HERE"/corpus100/img-*.jpg | head -"$N")
[ "${#IMGS[@]}" -gt 0 ] || { echo "corpus100 里没图"; exit 1; }

K=${#BINS[@]}
echo "ABC：${#IMGS[@]} 张图  $ROUNDS 轮  K=$K  变体：${BINS[*]}  额外参数：$*"
echo "# abc log bins=${BINS[*]} imgs=${#IMGS[@]} rounds=$ROUNDS extra=$*" >"$LOG"

for r in $(seq 1 "$ROUNDS"); do
    for ((i = 0; i < K; i++)); do
        j=$(( (i + r - 1) % K ))
        bin=${BINS[$j]}
        line=$("$bin" bench "${IMGS[@]}" --models "$MODELS" --device metal \
                   --rounds 3 --quiet "$@" 2>&1 | grep '^base' | head -1)
        printf 'round%-3d %-24s %s\n' "$r" "$(basename "$bin")" "$line" | tee -a "$LOG"
        # 字段：total det line wall imgs ratio …（见 ab.sh 的注释）
        echo "$r	$j	$(basename "$bin")	$(echo "$line" | awk '{print $2, $3, $4, $5}')" >>"$TSV"
    done
done

python3 - "$TSV" "${BINS[@]}" <<'PY'
import sys, statistics as st
tsv, bins = sys.argv[1], [b.split('/')[-1] for b in sys.argv[2:]]
rows=[]
for ln in open(tsv):
    p=ln.rstrip('\n').split('\t')
    if len(p)<4: continue
    r,j,name,v=p[0],int(p[1]),p[2],p[3].split()
    rows.append((int(r),j,name,[float(x) for x in v]))
K=len(bins)
by={}
for r,j,name,v in rows: by[(r,j)]=v
COLS=['total','det','line','wall']
print()
print("── 各轮 (total det line wall) ─────────────────────────────")
for r in sorted({x[0] for x in rows}):
    print(f"  r{r}: " + "  ".join(f"{bins[j]}={by[(r,j)][0]:.1f}/{by[(r,j)][1]:.1f}/{by[(r,j)][2]:.1f}" for j in range(K) if (r,j) in by))
print()
print("── 配对差值（各变体 − 基准，负 = 更快）；控制列 det 判脏 ──")
print(f"{'变体':<22} {'留存/总轮':>9} " + " ".join(f"{c:>9}" for c in COLS))
for j in range(1,K):
    keep={c:[] for c in COLS}
    nkept=0
    for r in sorted({x[0] for x in rows}):
        if (r,0) not in by or (r,j) not in by: continue
        a,b=by[(r,j)],by[(r,0)]          # b 是基准
        if abs(a[1]-b[1])>3.0: continue  # det 控制列动了 ⇒ 整轮丢
        nkept+=1
        for ci,c in enumerate(COLS): keep[c].append(a[ci]-b[ci])
    if not nkept:
        print(f"{bins[j]:<22} {'0/'+str(len({x[0] for x in rows})):>9}   （全轮被判脏）")
        continue
    cells=[]
    for c in COLS:
        d=keep[c]
        spread=max(d)-min(d) if len(d)>1 else 0.0
        cells.append(f"{st.median(d):+.2f}±{spread/2:.1f}")
    print(f"{bins[j]:<22} {str(nkept)+'/'+str(len({x[0] for x in rows})):>9} " + " ".join(f"{c:>9}" for c in cells))
print("\n（每格 = 中位配对差值 ± 半离散；离散和效应同量级时不要下结论）")
PY
echo "日志：$LOG"; rm -f "$TSV"
