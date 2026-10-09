#!/usr/bin/env python3
"""把「计划转储（有形状）」和「逐内核剖析（有时间）」拼起来，算**每个节点
的 GFLOPS** —— 用来找出「跑得远低于 fp32 天花板」的节点，而不是猜。

用法：
    QPPOCR_METAL_PLAN_DUMP=1 ... --det-only > plan.txt 2>&1
    QPPOCR_GPU_KERNEL_PROF=N ... > prof.txt 2>&1
    flopsmap.py plan.txt prof.txt [--start-line N]

plan.txt 里可能有多份计划（不同形状各一份）；默认取**最后一份**（通常是
真实形状，前面是 64×64 预热）。`--start-line` 可以手工指定真实计划的首行。

输出按「浪费的毫秒」降序 —— 即 `ms × (1 - GFLOPS/PEAK)`，这才是「修它
能省多少」的正确排序，光看 ms 会漏掉那些又慢又不占时间的。
"""
import re
import sys

PEAK_GFLOPS = 3340.0  # 实测 fp32 FMA 峰值（bench/gpuhealth/gpualu2）

PLAN_RE = re.compile(
    r"\[plan\] #(\d+)\s+nid=\S+\s+(\S+)\s+grid=\[(\d+), *(\d+), *(\d+)\]\s+"
    r"out=\[([^\]]*)\]\s+node=(\S+)(.*)$"
)
PROF_RE = re.compile(
    r"\[prof\]\s+·\s+(\S+)\s+([\d.]+) ms\s+node#(\d+)\s+(\S+)\s+grid=\[([^\]]*)\]"
)


def parse_plan(path, start_line):
    """返回 {node_name: dict(fn, flops)}，只取 start_line 之后的那份计划。"""
    nodes = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    body = lines[start_line - 1 :] if start_line else lines
    # 若未指定起点，取最后一段「#0 开头」的连续块
    if not start_line:
        starts = [i for i, l in enumerate(lines) if PLAN_RE.match(l) and PLAN_RE.match(l).group(1) == "0"]
        if starts:
            body = lines[starts[-1] :]
    for l in body:
        m = PLAN_RE.match(l.rstrip("\n"))
        if not m:
            continue
        _, kern, gx, gy, gz, out, node, rest = m.groups()
        flops = conv_flops(kern, out, rest)
        nodes[node] = {"kern": kern, "flops": flops, "out": out.strip(), "args": rest.strip()}
    return nodes


def num(pat, rest):
    m = re.search(pat, rest)
    return int(m.group(1)) if m else None


def conv_flops(kern, out, rest):
    """按形状算 2*MAC 的浮点操作数。只覆盖卷积类；其余算子返回 0。

    `n_convt` 是**上界**：步长 2 的转置卷积按每个输出位置累 kW² 个输入算
    会把实际贡献高估 4×，这里不做修正，读表时记住即可。
    """
    if kern in ("n_conv8", "n_conv8_mma", "n_conv8_mma3", "n_conv8_mma3_n32",
                "n_conv8_edge", "n_conv", "n_convt"):
        cin, co = num(r"cin=(\d+)", rest), num(r"co=(\d+)", rest)
        taps, m = num(r"taps=(\d+)", rest), num(r"m=(\d+)", rest)
        if None in (cin, co, taps, m):
            return 0.0
        return 2.0 * m * co * cin * taps
    if kern == "n_conv_dw":
        c, taps, m = num(r"c=(\d+)", rest), num(r"taps=(\d+)", rest), num(r"m=(\d+)", rest)
        if None in (c, taps, m):
            return 0.0
        return 2.0 * m * c * taps
    return 0.0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    start = None
    for a in sys.argv[1:]:
        if a.startswith("--start-line"):
            start = int(a.split("=", 1)[1]) if "=" in a else None
    for i, a in enumerate(sys.argv[1:]):
        if a == "--start-line":
            start = int(sys.argv[i + 2])
    plan_path, prof_path = args[0], args[1]

    nodes = parse_plan(plan_path, start)

    rows = []
    with open(prof_path, encoding="utf-8", errors="replace") as f:
        for l in f:
            m = PROF_RE.search(l)
            if not m:
                continue
            kern, ms, _idx, node, grid = m.groups()
            ms = float(ms)
            n = nodes.get(node)
            flops = n["flops"] if n else 0.0
            reps = 1
            gflops = flops / (ms * 1e6) if ms > 0 else 0.0
            rows.append((ms, kern, node, flops / 1e9, gflops, n["out"] if n else "?", reps))

    tot_ms = sum(r[0] for r in rows)
    tot_fl = sum(r[3] for r in rows)
    print(f"覆盖节点 {len(rows)} 个，合计 {tot_ms:.2f} ms，{tot_fl:.2f} GFLOP "
          f"⇒ 整体 {tot_fl/(tot_ms/1000):.0f} GFLOPS（峰值 {PEAK_GFLOPS:.0f}）")
    print(f"{'ms':>8} {'GFLOP':>8} {'GFLOPS':>8} {'%峰':>5} {'浪费ms':>7}  kernel           node")
    for ms, kern, node, gfl, gf, out, _ in sorted(rows, key=lambda r: -(r[0] * max(0.0, 1 - r[4] / PEAK_GFLOPS))):
        waste = ms * max(0.0, 1 - gf / PEAK_GFLOPS) if gf > 0 else 0.0
        print(f"{ms:8.3f} {gfl:8.3f} {gf:8.0f} {100*gf/PEAK_GFLOPS:5.1f} {waste:7.3f}  {kern:<16} {node}")


if __name__ == "__main__":
    main()
