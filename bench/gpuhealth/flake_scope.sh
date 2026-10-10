#!/bin/bash
# 「Metal 路径静默吐空行」的影响面：每张图单独跑 N 次，数每张图翻了几次、翻在哪一行。
#
# 要回答的问题（见 logs/flake_2026-10-10.md §6）：
#   是**只有 img-008 第 10 行**这一种「窄高裁切」会翻（边角 case），
#   还是一批图都会翻（静默拉低全语料精度 ⇒ 优先级高于所有剩余性能工作）？
#
# 用法： N=15 ./flake_scope.sh
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
Q=$ROOT/target/release/qppocr
MODELS=$ROOT/models
N=${N:-15}
INDIR=${INDIR:-testimgs}
OUT=logs/flake_scope_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

[ -x "$Q" ] || { echo "先构建： cargo build --release -p qppocr-cli --features qppocr/gpu"; exit 1; }
./health.sh -q || { echo "健康门禁没过，先别测"; exit 2; }

echo "== 影响面：每张图 $N 次 =="
for f in "$INDIR"/img-*.jpg; do
    b=$(basename "$f" .jpg)
    for i in $(seq 1 "$N"); do
        "$Q" "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" \
            --tier tiny --device metal --models "$MODELS" \
            --workers 1 --quiet --json >"$OUT/${b}_$i.json" 2>/dev/null
    done
done

python3 - "$OUT" "$N" <<'PY'
import json, sys, pathlib, collections
out, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
tot = collections.Counter(); where = collections.defaultdict(collections.Counter)
for p in sorted(out.glob('*.json')):
    t = p.read_text().strip()
    if not t: continue
    img = p.stem.rsplit('_', 1)[0]
    tot[img] += 1
    for i, l in enumerate(json.loads(t)[0]['lines']):
        if not l['text']:
            where[img][i] += 1
print(f"{'图':<10} {'轮数':>5} {'空行次数':>8}   行号")
bad_imgs = 0
for img in sorted(tot):
    c = sum(where[img].values())
    bad_imgs += bool(c)
    lines = ', '.join(f'第{i}行×{n}' for i, n in where[img].most_common()) or '—'
    print(f"{img:<10} {tot[img]:>5} {c:>8}   {lines}")
print(f"\n{len(tot)} 张图里 {bad_imgs} 张出现过空行")
PY
echo "  json 在：$OUT"
