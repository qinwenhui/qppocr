#!/bin/bash
# 用引擎本身找「把 GPU 访存通路打瘸」的触发点。
#
# 背景（2026-10-08 实测）：探针证明「大 dispatch 数」「形状多样性」「随机地址」
# 「大块显存分配/释放搅动」都打不瘸（inbounds 探针压了 25.7 万条 dispatch 仍然健康）。
# 所以触发条件在引擎的**具体行为**里。这个脚本按两条轴升级逼近：
#   图片数（1 → 2 → 10） × 并发形态（串行 workers=1 → 自动并发 workers=0）
#
# 每一步之间都过健康门禁；QPPOCR_METAL_CHUNK_BAIL 让"刚被打瘸"的瞬间就退出，
# 不用等挂死——那不但浪费时间，还会把 WindowServer 一起拖下去（黑屏）。
#
# 用法： bench/gpuhealth/engine_hunt.sh      （必须在重启后的健康机器上跑）
set -uo pipefail
cd "$(dirname "$0")/../.."

Q=target/release/qppocr
IMGDIR=/Users/qinwh/Pictures/dataset
LOG=bench/gpuhealth/logs
mkdir -p "$LOG"

BAIL=${BAIL:-1500}   # ms：任一块等待超这个数就认定「刚被打瘸」
CHUNK=${CHUNK:-8}    # 默认切块大小（单图逐条那步单独用 1）

gate() {
    bench/gpuhealth/health.sh -q && return 0
    echo "  ⛔ GPU 已瘸 —— 停下。按这个脚本的目的，这正是我们想要的信号（凶手在上一份日志里）；"
    echo "     确认完就重启。"
    exit 2
}

# step <名字> <图片数> <workers> <切块>
step() {
    local name=$1 n=$2 workers=$3 chunk=$4
    echo
    echo "═══ $name ═══"
    gate

    local imgs out rc
    imgs=$(ls "$IMGDIR"/img-*.jpg | head -n "$n" | tr '\n' ' ')
    out="$LOG/hunt_${name}.log"
    echo "  图 $n 张 · workers=$workers · CHUNK=$chunk · BAIL=${BAIL}ms"
    echo "  日志 → $out"

    local t0=$SECONDS
    QPPOCR_METAL_CHUNK="$chunk" QPPOCR_METAL_CHUNK_BAIL="$BAIL" \
        perl -e 'alarm 600; exec @ARGV' "$Q" $imgs \
        --tier tiny --device metal --workers "$workers" --quiet >"$out" 2>&1
    rc=$?
    echo "  退出码 $rc · $((SECONDS - t0)) 秒"

    case $rc in
        97)
            echo
            echo "  ★★ 找到凶手区间 ★★"
            grep -a -B1 '✖' "$out" | sed 's/^/     /'
            echo "     完整日志：$out"
            echo "     GPU 现在应该是瘸的 → 重启后**单独复跑这一步**确认可复现。"
            exit 97
            ;;
        142)
            echo "  ✖ 600 秒超时（不是刹车）—— 挂死，但没落在任何一块的等待上。"
            echo "    看 $out 末尾：全部 dispatch 编完却没进 wait，问题在编码侧。"
            exit 142
            ;;
        0)
            echo "  ✓ 跑完且没触发刹车。"
            tail -2 "$out" | sed 's/^/     /'
            ;;
        *)
            echo "  ✖ 非预期退出码，看 $out"
            exit "$rc"
            ;;
    esac
}

echo "════ 引擎触发点搜索（$CHUNK 条/块，刹车 ${BAIL}ms）════"
gate

# 第 0 步：先花 30 秒把「引擎之外」的最后一个未测维度问掉。
# inbounds / churn 都只用 1~2 块 buffer（churn 分了两万次但都是立刻释放，同时活着的始终很少），
# 而引擎的真实形态是**大量不同尺寸的 Shared arena 同时活着**（12 份计划、最大 138 MB）。
# 这一步健康 → 「多块不同尺寸同时活着」不是凶手，再把预算花在引擎上。
echo
echo "═══ 0-manybuf（引擎外最后一个未测维度）═══"
gate
./bench/gpuhealth/stress manybuf 30 2>&1 | sed 's/^/  /' | tee "$LOG/hunt_0-manybuf.log"
grep -q '✅' "$LOG/hunt_0-manybuf.log" || {
    echo "  ⛔ manybuf 就打瘸了 —— 凶手找到了，不用再往下跑引擎步骤。日志：$LOG/hunt_0-manybuf.log"
    exit 3
}

# 单图逐条：先把「一张图内有没有哪条 dispatch 自己就把通路打瘸」问清楚。
# 健康时这一步应该全部 <1ms；若某条突然上到几十/几百 ms，凶手就是它。
step "A-单图逐条" 1 1 1
# 两图串行 vs 两图并发：把「并发」单独拉出来当一个变量。
step "B-两图串行" 2 1 "$CHUNK"
step "C-两图并发" 2 0 "$CHUNK"
# 十图并发：复现 2026-10-08 那次挂死的原始形态。
step "D-十图并发" 10 0 "$CHUNK"

echo
echo "════ manybuf 与四条引擎升级全都没打瘸 ════"
echo "触发条件不在「同时活着的计划集」「图片数 × 并发」这两个面上。下一步换轴："
echo "preset（speed/accuracy）、rec-shards、det-max-side，或不同的图集合（形状分布）。"
