#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ONNX 里的 f32 权重就地舍入到 fp16（再回写为 f32）——量「权重 f16」的精度代价。

为什么是**原地字节替换**：f32 → f16 → f32 仍是 4 B/元素，raw_data / float_data
的字节长度**一字不变**，protobuf 的所有长度前缀都不需要动，文件结构零改动。
唯一的风险是定位错字节段，而这有响亮的下游判据——引擎读不出来会直接报错，
读出来数值全飞也会在对拍里显形。

只做一件事：DT_FLOAT 的 initializer，逐元素 `round_to_f16(x)`。

用法：
    python3 bench/gpuhealth/mk_f16w.py --info   models/tiny/det.onnx
    python3 bench/gpuhealth/mk_f16w.py --out DIR models/            # 整目录树
"""
import math
import struct
import sys
from pathlib import Path

DT_FLOAT = 1


# ---- 最小 protobuf 扫描器（只认 ONNX 用到的 wire type）----

def scan(buf, start, end):
    """产出 (field, wire, vstart, vend)；length-delimited 的 v* 是 payload 区间。"""
    i = start
    while i < end:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            _, j = _varint(buf, i)
            yield field, wire, i, j
            i = j
        elif wire == 1:
            yield field, wire, i, i + 8
            i += 8
        elif wire == 2:
            ln, j = _varint(buf, i)
            yield field, wire, j, j + ln
            i = j + ln
        elif wire == 5:
            yield field, wire, i, i + 4
            i += 4
        else:
            raise ValueError(f"wire type {wire} 不支持（offset {i}）")


def _varint(buf, i):
    v = shift = 0
    while True:
        b = buf[i]
        i += 1
        v |= (b & 0x7F) << shift
        if not b & 0x80:
            return v, i
        shift += 7


def walk_graph(buf):
    """返回 [(tensor_name, data_type, kind, vstart, vend)]——kind ∈ {raw, float}。"""
    # ModelProto.graph = field 7
    graph = None
    for field, wire, a, b in scan(buf, 0, len(buf)):
        if field == 7 and wire == 2:
            graph = (a, b)
            break
    if graph is None:
        raise ValueError("找不到 ModelProto.graph(field 7)")
    out = []
    # GraphProto.initializer = field 5（repeated TensorProto）
    for field, wire, a, b in scan(buf, *graph):
        if field != 5 or wire != 2:
            continue
        name, dtype = "", DT_FLOAT
        segs = []
        for f2, w2, a2, b2 in scan(buf, a, b):
            if f2 == 2 and w2 == 0:
                dtype = _varint(buf, a2)[0]
            elif f2 == 8 and w2 == 2:
                name = buf[a2:b2].decode("utf-8", "replace")
            elif f2 == 9 and w2 == 2:
                segs.append(("raw", a2, b2))
            elif f2 == 4 and w2 == 2:
                segs.append(("float", a2, b2))
            elif f2 == 4 and w2 == 5:
                segs.append(("float32one", a2, b2))
        for kind, a2, b2 in segs:
            out.append((name, dtype, kind, a2, b2))
    return out


# ---- fp16 舍入（用 Python 自带的 '<e' = IEEE binary16，就近偶数舍入）----

def round_f16(v):
    if v != v or v in (float("inf"), float("-inf")):
        return v
    try:
        h = struct.pack("<e", v)
    except OverflowError:  # |v| > f16 最大值 → 硬件同样溢出到 inf
        return math.copysign(float("inf"), v)
    return struct.unpack("<e", h)[0]


def patch(path, out):
    buf = bytearray(path.read_bytes())
    n_el = n_t = 0
    changed = 0
    for name, dtype, kind, a, b in walk_graph(bytes(buf)):
        if dtype != DT_FLOAT:
            continue
        if kind == "raw":
            for off in range(a, b, 4):
                v = struct.unpack("<f", buf[off:off + 4])[0]
                r = round_f16(v)
                if r != v:
                    changed += 1
                buf[off:off + 4] = struct.pack("<f", r)
                n_el += 1
            n_t += 1
        elif kind in ("float", "float32one"):
            step = 4 if kind == "float" else 4
            for off in range(a, b, step):
                v = struct.unpack("<f", buf[off:off + 4])[0]
                r = round_f16(v)
                if r != v:
                    changed += 1
                buf[off:off + 4] = struct.pack("<f", r)
                n_el += 1
            n_t += 1
    if len(buf) != path.stat().st_size:
        raise AssertionError("长度变了——不该发生")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buf)
    return n_t, n_el, changed


def main():
    args = sys.argv[1:]
    if "--info" in args:
        for p in [Path(x) for x in args if not x.startswith("--")]:
            segs = walk_graph(p.read_bytes())
            f32 = [s for s in segs if s[1] == DT_FLOAT]
            kinds = {}
            for s in f32:
                kinds[s[2]] = kinds.get(s[2], 0) + 1
            els = sum((s[4] - s[3]) // 4 for s in f32)
            print(f"{p}: initializer 段 {len(segs)}，其中 f32 {len(f32)} 段 / "
                  f"{els:,} 元素，形态 {kinds}")
        return

    if "--out" not in args:
        print(__doc__)
        return
    i = args.index("--out")
    dst = Path(args[i + 1])
    srcs = [Path(x) for x in args[i + 2:]]
    files = []
    for s in srcs:
        files += sorted(s.rglob("*.onnx")) if s.is_dir() else [s]
    for f in files:
        rel = f.relative_to(next(s for s in srcs if s.is_dir() and f.is_relative_to(s))) \
            if any(s.is_dir() and f.is_relative_to(s) for s in srcs) else Path(f.name)
        n_t, n_el, ch = patch(f, dst / rel)
        print(f"{rel}: {n_t} 段 / {n_el:,} 元素，{ch:,} 个值真的变了 "
              f"（{100.0 * ch / max(n_el, 1):.2f}%）")


if __name__ == "__main__":
    main()
