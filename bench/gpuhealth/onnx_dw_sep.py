#!/usr/bin/env python3
"""检查 ONNX 里的深度可分离卷积核是否**可分离**（rank-1）。

动机：`n_conv_dw` 在 medium det 里 17.27 ms（det 的 17.6%），而它的代价
**严格正比于 c·m·taps**、恒定在 ~251 GFLOPS（fp32 峰值的 7.5%）—— 见
`logs/n_conv_dw_dig_20261010.md`：11 个结构改动（循环体/寄存器/并行度）
**无一例外更慢**，所以剩下的唯一杠杆是**减少乘加数**。

9×9 dw 的 81 个 tap 若能拆成 9×1 + 1×9 两级，就是 18 个 tap（4.5×）。
但那只在核**恰好是 rank-1** 时成立 —— 一般学出来的 9×9 核不是。这份脚本
用 SVD 直接量：σ2/σ1 越小越接近可分离；真 rank-1 才谈得上"省 tap 且
数值只差重结合误差（~1e-7）"，否则"分离"算的是**另一个卷积**，输出全错。

用法：onnx_dw_sep.py models/medium/det.onnx
"""
import struct
import sys


def rd_varint(b, i):
    r = 0
    s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        if not (x & 0x80):
            return r, i
        s += 7


def fields(b, start=0, end=None):
    """产出 (field_number, wire_type, value) —— value 视 wire_type 而变：
    0 → int；1 → 8 字节 bytes；2 → bytes；5 → 4 字节 bytes。"""
    i = start
    end = len(b) if end is None else end
    while i < end:
        tag, i = rd_varint(b, i)
        fn, wt = tag >> 3, tag & 7
        if wt == 0:
            v, i = rd_varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            n, i = rd_varint(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"wire_type {wt} @ {i}")
        yield fn, wt, v


def _nfloats(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def parse_tensor(b):
    """TensorProto → dict(dims, dtype, name, floats)。"""
    dims, dtype, name, raw, fdata = [], None, "", b"", []
    for fn, wt, v in fields(b):
        if fn == 1:               # dims
            if wt == 2:
                j = 0
                while j < len(v):
                    d, j = rd_varint(v, j)
                    dims.append(d)
            else:
                dims.append(v)
        elif fn == 2:             # data_type
            dtype = v
        elif fn == 4 and wt == 2:  # float_data（packed）
            fdata += list(struct.unpack("<%df" % (len(v) // 4), v))
        elif fn == 8:
            name = v.decode("utf-8", "replace")
        elif fn == 10 or (fn == 9 and len(v) == _nfloats(dims) * 4):
            # 标准 ONNX 把裸权重放 field 10(raw_data)；本仓这份模型放 field 9。
            # 按「长度 == 元素数×4」认领，避免把 doc_string 当权重。
            raw = v
    if dtype == 1 and not fdata and raw:
        fdata = list(struct.unpack("<%df" % (len(raw) // 4), raw))
    return dims, dtype, name, fdata


def singular_values_sq(mat):
    """WᵀW 的特征值（= σ²）—— Jacobi 旋转，K 只有 9，纯 Python 就够。"""
    k = len(mat)
    a = [[sum(mat[r][i] * mat[r][j] for r in range(k)) for j in range(k)] for i in range(k)]
    for _ in range(100):
        off = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                off += a[i][j] * a[i][j]
        if off < 1e-30:
            break
        for p in range(k):
            for q in range(p + 1, k):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q]
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + (theta * theta + 1.0) ** 0.5)
                c = 1.0 / (t * t + 1.0) ** 0.5
                s = t * c
                for i in range(k):
                    aip, aiq = a[i][p], a[i][q]
                    a[i][p] = c * aip - s * aiq
                    a[i][q] = s * aip + c * aiq
                for i in range(k):
                    api, aqi = a[p][i], a[q][i]
                    a[p][i] = c * api - s * aqi
                    a[q][i] = s * api + c * aqi
    return sorted((max(0.0, a[i][i]) for i in range(k)), reverse=True)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/medium/det.onnx"
    with open(path, "rb") as f:
        blob = f.read()

    tensors = []
    for fn, wt, v in fields(blob):
        if fn != 7:               # GraphProto
            continue
        for gfn, gwt, gv in fields(v):
            if gfn == 5:          # initializer
                tensors.append(parse_tensor(gv))

    dw = [t for t in tensors
          if len(t[0]) == 4 and t[0][1] == 1 and t[0][2] == t[0][3] >= 2]
    print(f"{path}: initializer {len(tensors)} 个，其中 [C,1,K,K] 深度可分离核 {len(dw)} 个")

    rows = []
    for d, dt, n, fl in dw:
        c, k = d[0], d[2]
        if len(fl) != c * k * k:
            print(f"  !! {n} 数据长度 {len(fl)} != {c}*{k}*{k}")
            continue
        worst = 0.0
        for ci in range(c):
            b0 = ci * k * k
            w = [fl[b0 + r * k: b0 + r * k + k] for r in range(k)]
            s2 = singular_values_sq(w)
            if s2[0] > 0:
                worst = max(worst, (s2[1] / s2[0]) ** 0.5)
        rows.append((k, c, worst, n, d))

    # 按「分离度次优比」升序 —— 越接近 0 越可能真的可分离
    print(f"\n{'K':>2} {'C':>4} {'σ2/σ1 最差通道':>15}  名字")
    for k, c, worst, n, d in sorted(rows, key=lambda r: r[2])[:25]:
        flag = "  ← 疑似可分离" if worst < 1e-3 else ""
        print(f"{k:>2} {c:>4} {worst:>15.5f}  {n}{flag}")
    print("\n（σ2/σ1 ≪ 1e-6 才算数值上真的 rank-1；否则「分离」= 换了一个卷积）")


if __name__ == "__main__":
    main()
