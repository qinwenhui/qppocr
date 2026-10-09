#!/bin/bash
# 通用「环境开关」A/B：奇偶轮换序，报中位配对差值。
#
# 用法：./env_ab.sh <图数> <轮数> VAR=VAL [VAR=VAL ...]
#   变体 A = 不带这些变量；变体 B = 带上（`env` 前置，不经 shell 分词）。
#
# 纪律（见 memory：ab-order-effect-swamps-small-gains / ab-variance-must-be-two-sided）：
#   * 必须轮换序，否则「先跑的那个快 6~15%」会吃掉整个效应；
#   * 看**中位配对差**，不看绝对值；
#   * Δ 若单向同号，那是污染不是噪声——用控制列（此处由调用方自己看 det）
#     判断，别用中位数把单向漂移吃进去。
set -uo pipefail
cd "$(dirname "$0")"
N=${1:?图数}; R=${2:?轮数}; shift 2
B=("$@")
BIN=${BIN:-./bin/qppocr-cc}
# 档位：默认 small（与 bench 自己的默认一致），`TIER=medium ./env_ab.sh ...` 可换。
# 曾经的坑：脚本不传 --tier，报出来的数被当成「默认档」，其实是 small。
TIER=${TIER:-small}

# ── 并发门禁：默认 --workers 1（2026-10-10 加的，因为又踩了一次）──────────
#
# 不传 `--workers` 时 bench 走 **auto**，在这台 10 核机上 = 5 个并发 Metal
# 进程。这台机器的**静默致瘸路径**正是「并发 Metal 进程数一多」——见
# memory `metal-unbound-buffer-wedges-gpu`。后果不是慢慢变慢，而是：
#
#   同一条命令、同一批图，4 张 medium 的一轮 A/B 里 total/img 从 259 ms
#   一路爬到 477 ms（1.84×），而**两臂同步漂** ⇒ 配对差全是垃圾：
#   Δdet 控制在 +13.6 / +14.4 ms 晃，符号 9 正 3 负，离散 36 ms。
#   紧接着把 `--workers 1` 单独跑三次同一命令：191.7 / 192.6 / 191.7 ms
#   （轮内离散 ±1 ms）。
#
# 所以：**A/B 一律单进程**。要量「并发吞吐」另开专门的探针，别拿 A/B 当它。
# `WORKERS=0` 可以显式覆盖回 auto（知道自己在干什么时）。
WORKERS=${WORKERS:-1}

# ── 陈旧二进制门禁（2026-10-09 加的，因为踩过一次）────────────────────
#
# `bin/qppocr-cc` 是**手工拷贝的快照**，不会随源码更新——它一度比源码旧 6
# 小时，于是 `QPPOCR_METAL_WF16` 这个开关在它里面**根本不存在**：两臂跑的
# 是同一份 f32 代码，量出来的 Δtotal = −0.05 ms（符号 2 负 2 正、纯噪声），
# 结论是「fp16 端到端零收益」，而真值是 **1.79×**。
#
# **这类失败是静默的，而且恰好长得像「这个优化没用」。** 所以这里对每个
# `VAR=VAL` 断言 `VAR` 这个字符串真的在二进制里——开关压根没编进去时，A/B
# 的结果**恒为零**，必须当场判废而不是当成结论。
for kv in "${B[@]}"; do
    v=${kv%%=*}
    if [ "$(grep -c -a -- "$v" "$BIN" || true)" -eq 0 ]; then
        echo "❌ $BIN 里找不到环境变量名 '$v' —— 这个二进制太旧，开关没编进去。" >&2
        echo "   两臂会跑同一份代码，Δ 恒为 0，那不是结论。" >&2
        echo "   先重建：cargo build --release -p qppocr-cli --features qppocr/gpu" >&2
        echo "   再用：  BIN=\$PWD/../../target/release/qppocr $0 $N $R ${B[*]}" >&2
        exit 3
    fi
done
echo "# BIN=$BIN  (mtime $(stat -f %Sm -t '%m-%d %H:%M' "$BIN" 2>/dev/null || echo '?'))  档位=$TIER  workers=$WORKERS  开关=${B[*]}"

set -- $(ls corpus100/img-*.jpg | head -"$N")
IMGS=("$@")
TSV=$(mktemp)
for r in $(seq 1 "$R"); do
    if [ $((r % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
    for v in $order; do
        if [ "$v" = B ]; then
            out=$(env "${B[@]}" "$BIN" bench "${IMGS[@]}" \
                    --models ../../models --device metal --tier "$TIER" --rounds 3 --quiet \
                    --workers "$WORKERS" 2>/dev/null | grep '^base' | head -1)
        else
            out=$("$BIN" bench "${IMGS[@]}" \
                    --models ../../models --device metal --tier "$TIER" --rounds 3 --quiet \
                    --workers "$WORKERS" 2>/dev/null | grep '^base' | head -1)
        fi
        printf '%s\t%s\t%s\n' "$r" "$v" "$(echo "$out" | awk '{print $2, $3, $4, $5, $6}')" >>"$TSV"
    done
done
python3 - "$TSV" "${B[*]}" <<'PY'
import sys, statistics as st
rows={}
for ln in open(sys.argv[1]):
    p=ln.rstrip('\n').split('\t')
    if len(p)<3 or not p[2].strip(): continue
    rows[(int(p[0]),p[1])]=[float(x) for x in p[2].split()]
print(f"变体 B = env {sys.argv[2]}")
print(f"{'轮':>3} {'A tot':>9} {'B tot':>9} {'Δtot':>8} {'Δdet':>7} {'Δline':>8} {'Δimg/s':>8}")
dt=[];dd=[];dl=[];ds=[]
for r in sorted({k[0] for k in rows}):
    if (r,'A') not in rows or (r,'B') not in rows: continue
    a,b=rows[(r,'A')],rows[(r,'B')]
    if len(a)<5 or len(b)<5: continue
    dt.append(b[0]-a[0]); dd.append(b[1]-a[1]); dl.append(b[2]-a[2]); ds.append(b[4]-a[4])
    print(f"{r:>3} {a[0]:>9.1f} {b[0]:>9.1f} {b[0]-a[0]:>+8.2f} {b[1]-a[1]:>+7.2f} {b[2]-a[2]:>+8.2f} {b[4]-a[4]:>+8.2f}")
if dt:
    m=lambda v: st.median(v)
    print(f"\n中位配对差：Δtotal = {m(dt):+.2f} ms   Δdet = {m(dd):+.2f}   Δline = {m(dl):+.2f}   Δimg/s = {m(ds):+.2f}")
    print(f"符号：Δtotal 负 {sum(1 for x in dt if x<0)}/{len(dt)}，正 {sum(1 for x in dt if x>0)}/{len(dt)}；离散 {max(dt)-min(dt):.1f} ms")
PY
rm -f "$TSV"
