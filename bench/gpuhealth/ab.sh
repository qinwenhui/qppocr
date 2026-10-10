#!/bin/bash
# 两个二进制（或两个后端配置）的**交错 A/B**。
#
# 为什么必须交错：本机实测过整轮漂移——同一份代码连跑两遍，峰值从 3214
# 掉到 2646（均匀慢 ~25%）。「先跑完 A 再跑 B」会把漂移记成 A/B 差异。
# 交错成 new,old,new,old… 则漂移在两列上对称，**只压差值不压绝对值**。
#
# 用法： ./ab.sh <新二进制> <旧二进制> [图片数，默认 40] [额外参数…]
# 例：   ./ab.sh /tmp/qppocr_new /tmp/qppocr_old 40 --tier small
#
# 输出：logs/ab_<时间戳>.log —— 每行 `roundN <变体>  <bench base 行>`。
# 判读看 `line/img`（rec 逐行阶段，多数改动打在这里）与 `total/img`。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$HERE"

NEW=${1:?用法: ./ab.sh <新二进制> <旧二进制> [图片数] [额外参数…]}
OLD=${2:?}
N=${3:-40}
shift 3 2>/dev/null || shift $#
MODELS=$ROOT/models
LOG=logs/ab_$(date +%Y%m%d_%H%M%S).log

# ── 门禁：两臂必须**真的是两个不同的、带 Metal 的二进制** ──────────────
# 踩过的坑（2026-10-10）：按习惯 `cargo build --release -p qppocr-cli` 建两臂，
# **忘了 `--features qppocr/gpu`**（gpu 不是 qppocr 的默认 feature）——建出来的是
# 没有 Metal 后端的二进制，两臂 **md5 完全相同**。`ab.sh` 只比数字不比二进制，
# 于是 Δ 恒为 0，而且中位/符号/离散全都自洽，**静默地长得像「这个优化没用」**。
# 同 [[ab-null-may-be-stale-binary]] 的形态（那边是 env 开关没编进二进制）。
# 正确命令：cargo build --release -p qppocr-cli --features qppocr/gpu
for pair in "NEW=$NEW" "OLD=$OLD"; do
    nm=${pair%%=*}; bin=${pair#*=}
    [ -x "$bin" ] || { echo "门禁：$nm=$bin 不是可执行文件" >&2; exit 3; }
    n=$(grep -c -a 'n_conv_dw' "$bin" 2>/dev/null || echo 0)
    [ "$n" -gt 0 ] || { echo "门禁：$nm=$bin 里没有 Metal 内核标记（n_conv_dw=0）——" >&2
        echo "      大概是漏了 --features qppocr/gpu，这条 A/B 会量出恒 0。" >&2; exit 3; }
done
if [ "$(md5 -q "$NEW")" = "$(md5 -q "$OLD")" ] && [ "${ALLOW_SAME:-0}" != 1 ]; then
    echo "门禁：两臂 md5 相同（$(md5 -q "$NEW")）——量的是噪声，不是差异。" >&2
    echo "      确实要量同一份二进制请设 ALLOW_SAME=1。" >&2; exit 3
fi

# ⚠️ macOS 自带 bash 是 3.2：没有 mapfile；且空数组在 set -u 下取 "${A[@]}"
# 会报 unbound。所以用 while-read 填数组（下面还要判空）。
IMGS=()
while IFS= read -r f; do IMGS[${#IMGS[@]}]="$f"; done < <(ls "$HERE"/corpus100/img-*.jpg | head -"$N")
[ "${#IMGS[@]}" -gt 0 ] || { echo "corpus100 里没图"; exit 1; }
echo "AB：new=$NEW  old=$OLD  ${#IMGS[@]} 张图  轮数 3（+1 warmup）"
echo "# AB  log  new=$NEW old=$OLD imgs=${#IMGS[@]} extra=$*" >"$LOG"

# ⚠️ 每轮**必须换序**。实测（2026-10-08）：固定「先 new 后 old」时，连代码
# 逐字相同的 det（3×3 走的是没动过的 else 分支）都显示 new 快 6~15%——
# 那就是纯粹的**次序效应**（一小时内第二次跑更热/被节流）。它和我们要量的
# 效应同一量级，固定次序的 A/B 等于没量。奇偶轮换序后，次序效应在
# 「本轮 new−old」的符号上对称抵消。
ROUNDS=${ROUNDS:-4}
ROWS=()
for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) -eq 1 ]; then ORDER="new old"; else ORDER="old new"; fi
    for v in $ORDER; do
        bin=$NEW; [ "$v" = old ] && bin=$OLD
        line=$("$bin" bench "${IMGS[@]}" --models "$MODELS" --device metal \
                   --rounds 3 --quiet "$@" 2>&1 | grep '^base' | head -1)
        printf 'round%d %-4s %s\n' "$r" "$v" "$line" | tee -a "$LOG"
        # 该行数值字段：total det line wall imgs ratio
        n=$(echo "$line" | awk '{print $2, $3, $4, $5}')
        ROWS+=("$r $v $n")
    done
done

echo
echo "── 配对差值（new − old，负 = new 更快）───────────────────────────"
printf '%s\n' "${ROWS[@]}" | awk '
{ r[$1"_"$2]=$3" "$4" "$5" "$6; if ($1>maxr) maxr=$1 }
END {
  printf "%-6s %9s %9s %9s\n", "round", "det", "line", "total"
  for (i=1;i<=maxr;i++) {
    split(r[i"_new"], a, " "); split(r[i"_old"], b, " ")
    if (a[1]=="") continue
    printf "%-6d %+9.2f %+9.2f %+9.2f\n", i, a[2]-b[2], a[3]-b[3], a[4]-b[4]
    n++; d[n,1]=a[2]-b[2]; d[n,2]=a[3]-b[3]; d[n,3]=a[4]-b[4]
  }
  for (c=1;c<=3;c++) {
    for (i=1;i<=n;i++) v[i]=d[i,c]
    for (i=1;i<=n;i++) for (j=i+1;j<=n;j++) if (v[i]>v[j]) { t=v[i]; v[i]=v[j]; v[j]=t }
    med[c] = (n%2) ? v[(n+1)/2] : (v[n/2]+v[n/2+1])/2
    s=0; for (i=1;i<=n;i++) s+=v[i]
    mean[c]=s/n
  }
  printf "\n中位配对差值  det %+.2f   line %+.2f   total %+.2f  (n=%d)\n", med[1], med[2], med[3], n
  printf "均值配对差值  det %+.2f   line %+.2f   total %+.2f\n", mean[1], mean[2], mean[3]
}'
echo "日志：$LOG"
