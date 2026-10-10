#!/bin/bash
# 钉住 vs 不钉住的对照：证明「账本假命中」真抓得到，且修法真堵住了它。
#
# 2026-10-11 修的 `Arena::pins`：账本拿 `Arc::as_ptr` 当内容身份，而**指针会被
# 回收复用**——写下账本的计划一被丢掉（`probe()` 探完即弃），那批地址就分给后来
# 的参数块 ⇒ 账本判「已在位」⇒ 那一块根本没写 ⇒ 内核读到别的形状的参数 ⇒
# 静默丢一行（见 logs/flake_2026-10-10.md §8）。修法是账本一并**持有**那些 `Arc`，
# 把地址钉住。
#
# 可是「修好了」和「判据瞎了」在「0 异常」这个读数上长得一模一样。所以两臂：
#   U（`QPPOCR_METAL_LEDGER_UNPIN=1`）：不钉地址，把 bug 放回来 ⇒ **预期很快撞上**
#   P（默认，钉住）：⇒ **预期 N 轮干净**
# U 臂撞上 = 判据没瞎（假命中真的存在，且脚本抓得到）；P 臂干净 = 修法有效。
#
# U 臂不会把 GPU 打瘸：`STATIC_VERIFY` 发现静态区不在位就**拒发**（错参数不进
# 命令缓冲）。互锁条件写在那个开关里——**单开 UNPIN 不生效**。
#
# 用法： N=150 ./flake_pin.sh
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
Q=$ROOT/target/release/qppocr
N=${N:-150}
IMG=${IMG:-$(cd testimgs && pwd)/img-002.jpg}
OUT=logs/flake_pin_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

[ -x "$Q" ] || { echo "先构建"; exit 1; }
for v in QPPOCR_METAL_STATIC_VERIFY QPPOCR_METAL_LEDGER_UNPIN; do
    grep -q -a "$v" "$Q" || { echo "开关 $v 没编进二进制，先重建"; exit 3; }
done
./health.sh -q || { echo "健康门禁没过"; exit 2; }

arm() { # $1=臂名 $2=额外 env（空串=默认臂）
    local name=$1 extra=$2 i d caught=0
    for i in $(seq 1 "$N"); do
        d="$OUT/$name/run$(printf '%03d' "$i")"
        mkdir -p "$d"
        env $extra QPPOCR_METAL_STATIC_VERIFY=1 "$Q" "$IMG" \
            --tier tiny --device metal --models "$ROOT/models" \
            --workers 1 --quiet --json >"$d/out.json" 2>"$d/err.txt"
        # 注意判据：`[verify]` 那行的格式是「静态 0/N 块不在位」——**0 也在里面**，
        # 所以只能匹配「静态 非零/」，不能匹配「块不在位」这个词（第一版就栽在这，
        # 两臂都报「第 1 轮撞上」，撞的却是正常那行）。
        if grep -q -a -E "静态 [1-9][0-9]*/" "$d/err.txt"; then
            caught=$i
            echo "  ${name}：**第 $i 轮撞上**（假命中复现）→ $d"
            grep -a -m1 -E "静态 [1-9][0-9]*/" "$d/err.txt" | sed 's/^/       /'
            break
        fi
        rm -rf "$d"
    done
    [ "$caught" = 0 ] && echo "  ${name}：$N 轮零异常（没撞上）"
    printf '%s catch=%s/%s\n' "$name" "$caught" "$N" >>"$OUT/summary"
}

echo "== 两臂对照（$N 轮封顶）：$(basename "$IMG") =="
echo "── U 臂：不钉住（把 bug 放回来）──"
arm U "QPPOCR_METAL_LEDGER_UNPIN=1"
echo "── P 臂：钉住（当前代码，默认）──"
arm P ""
echo
cat "$OUT/summary"
echo "  明细：$OUT"
