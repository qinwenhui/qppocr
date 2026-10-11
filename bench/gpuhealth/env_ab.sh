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
#   * **漂移门禁**（见下）：A 臂自己一轮内爬升 >1.15× 时，脚本会把漂移轮剔掉
#     再报一次中位数。判据只看 A 臂，不看 Δ——按 Δ 筛轮次等于挑合意的噪声。
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
per=[]
for r in sorted({k[0] for k in rows}):
    if (r,'A') not in rows or (r,'B') not in rows: continue
    a,b=rows[(r,'A')],rows[(r,'B')]
    if len(a)<5 or len(b)<5: continue
    per.append((r,a[0],b[0]-a[0],b[1]-a[1],b[2]-a[2],b[4]-a[4]))
    print(f"{r:>3} {a[0]:>9.1f} {b[0]:>9.1f} {b[0]-a[0]:>+8.2f} {b[1]-a[1]:>+7.2f} {b[2]-a[2]:>+8.2f} {b[4]-a[4]:>+8.2f}")
if per:
    m=st.median
    dt=[x[2] for x in per]; dd=[x[3] for x in per]; dl=[x[4] for x in per]; ds=[x[5] for x in per]
    print(f"\n中位配对差：Δtotal = {m(dt):+.2f} ms   Δdet = {m(dd):+.2f}   Δline = {m(dl):+.2f}   Δimg/s = {m(ds):+.2f}")
    print(f"符号：Δtotal 负 {sum(1 for x in dt if x<0)}/{len(dt)}，正 {sum(1 for x in dt if x>0)}/{len(dt)}；离散 {max(dt)-min(dt):.1f} ms")

    # ── 漂移门禁（2026-10-11 加的，因为又双叒被吃了一次）──────────────────
    #
    # `health.sh` 判的是**带宽 ≥60 GB/s**，而本机在带宽照样达标的同时会整体
    # 爬升：同一天里 small 从 70 爬到 155 ms（+123%）、medium 从 255 到 480。
    # 这是 [[sequential-cross-binary-compare-is-drift]] 记的持续负载爬升，
    # 不是回归、不是致瘸，所以 health.sh 拦不住（它测的是另一件事）。
    #
    # 后果：把漂移轮算进中位数，要么把真效应吃成 0、要么把冷热差记成优化。
    #
    # 处方：**用 A 臂自己的离散给轮次分类，基准取本轮最冷的一轮**（漂移是
    # 单调向上爬的，越早越冷；我们一律以冷态为准，见各 log 的「健康态」）。
    # 只按 A 筛、**绝不按 Δ 筛** —— 按 Δ 筛等于从噪声里挑合意的样本，那是
    # 把 [[ab-variance-must-be-two-sided]] 的教训反过来犯。
    Aall=[x[1] for x in per]
    lo=min(Aall)
    spread=(max(Aall)/lo) if lo>0 else float('inf')
    keep=[x for x in per if x[1] <= lo*1.10]
    print(f"\nA 臂基线：最冷 {lo:.1f} ms、最热 {max(Aall):.1f} ms（{spread:.2f}×）")
    if spread <= 1.15:
        print(f"✅ 漂移门禁：A 臂离散 {spread:.2f}× ≤ 1.15 ⇒ 上面的中位数可用。")
    else:
        print(f"⚠️  漂移门禁：A 臂离散 {spread:.2f}× > 1.15 ⇒ **本轮已漂移，上面的中位数不可用作结论**。")
        if len(keep) >= 2:
            kt=[x[2] for x in keep]; kdd=[x[3] for x in keep]; kl=[x[4] for x in keep]
            print(f"    只留「A ≤ 最冷轮 ×1.10」的 {len(keep)}/{len(per)} 轮（轮号 {[x[0] for x in keep]}）：")
            print(f"    稳定轮 Δtotal 中位 = {m(kt):+.2f} ms   Δdet = {m(kdd):+.2f}   Δline = {m(kl):+.2f}"
                  f"   符号 负{sum(1 for x in kt if x<0)}/正{sum(1 for x in kt if x>0)}"
                  f"   离散 {max(kt)-min(kt):.1f}")
            print(f"    —— 报结论用这一行，别用上面那行。")
        else:
            print(f"    稳定轮只剩 {len(keep)} 个 ⇒ **本轮整个弃用**，等机身凉下来重跑。")
PY
rm -f "$TSV"
