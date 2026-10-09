#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫真实 ONNX：列出所有 k×k（k>1）稠密卷积，逐个过 `conv_kxk_mma` 门禁。

**为什么要有这个探针**：端到端跑 medium det 时 `QPPOCR_GPU_KERNEL_PROF=1`
显示 **整张图只有 1 条 conv 点着了 `n_conv8_mma3`**。要决定下一步往哪投
（放宽哪一条判据、还是给别的 N 宽度开一条 MMA 路），必须知道**被哪条判据
刷掉的、刷掉多少条**——而不是猜。

门禁（逐条对应 `plan.rs::PlanBuilder::conv_kxk_mma` 与内核头注释）：

  kh==kw>=2 | strides==1 | pads 四向对称且 = (k-1)/2
  | cpad4(Co)%64==0 | cpad4(Ci)%8==0 | group==1
  | outW-(k-1) >= 64        （内点列至少一个 64 块）
  | outH==inH && outW==inW  （对称 pad + stride 1 蕴含，这里当断言查）

空间尺寸优先取 **value_info 注解**（导出器写了的就有），没有就标 `?` 并
在报告里如实说明——不猜形状。

用法：python3 bench/gpuhealth/kxk_scan.py models/medium/det.onnx [...]
"""
import sys
from pathlib import Path

DT_FLOAT = 1


# ---- 最小 protobuf 扫描器（与 mk_f16w.py 同源）----

def _varint(buf, i):
    r = s = 0
    while True:
        b = buf[i]
        i += 1
        r |= (b & 0x7F) << s
        if not b & 0x80:
            return r, i
        s += 7


def scan(buf, start, end):
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
            raise ValueError(f"wire type {wire}")


def _ints(buf, a, b):
    """repeated int64：packed（wire 2）与非 packed（wire 0）都认。"""
    out = []
    for f, w, x, y in scan(buf, a, b):
        if w == 0:
            out.append(_varint(buf, x)[0])
        elif w == 2:
            i = x
            while i < y:
                v, i = _varint(buf, i)
                out.append(v)
    return out


def graph_of(buf):
    for f, w, a, b in scan(buf, 0, len(buf)):
        if f == 7 and w == 2:
            return (a, b)
    raise ValueError("找不到 ModelProto.graph(field 7)")


def initializers(buf, g):
    """name -> dims（TensorProto: dims=1, name=8）。"""
    out = {}
    for f, w, a, b in scan(buf, *g):
        if f != 5 or w != 2:
            continue
        name, dims = "", []
        for f2, w2, a2, b2 in scan(buf, a, b):
            if f2 == 1:
                dims += _ints(buf, a2, b2) if w2 == 2 else [_varint(buf, a2)[0]]
            elif f2 == 8 and w2 == 2:
                name = buf[a2:b2].decode("utf-8", "replace")
        if name:
            out[name] = dims
    return out


def attr_map(buf, a, b):
    """单个 AttributeProto（**整段** (a,b)）：name=1, i=3, ints=8。"""
    name, i_val, ints = "", None, None
    for f, w, x, y in scan(buf, a, b):
        if f == 1 and w == 2:
            name = buf[x:y].decode("utf-8", "replace")
        elif f == 3 and w == 0:
            i_val = _varint(buf, x)[0]
        elif f == 8:
            # proto2 的 repeated 标量**默认不 packed**（每元素一条记录），
            # 用 `=` 会让 4 个 pads 只剩最后一个——踩过。
            ints = (ints or []) + (_ints(buf, x, y) if w == 2 else [_varint(buf, x)[0]])
    if not name:
        return {}
    if ints is not None:
        return {name: ints}
    if i_val is not None:
        return {name: i_val}
    return {}


def nodes(buf, g):
    """(op, inputs, name, attrs)。NodeProto: input=1, output=2, name=3, op=4, attr=5。"""
    for f, w, a, b in scan(buf, *g):
        if f != 1 or w != 2:
            continue
        ins, name, op, attrs = [], "", "", {}
        for f2, w2, a2, b2 in scan(buf, a, b):
            if f2 == 1 and w2 == 2:
                ins.append(buf[a2:b2].decode("utf-8", "replace"))
            elif f2 == 3 and w2 == 2:
                name = buf[a2:b2].decode("utf-8", "replace")
            elif f2 == 4 and w2 == 2:
                op = buf[a2:b2].decode("utf-8", "replace")
            elif f2 == 5 and w2 == 2:
                attrs.update(attr_map(buf, a2, b2))
        yield op, ins, name, attrs


def value_shapes(buf, g):
    """value_info(13) 与 graph.input(11)/output(12) 上的静态形状注解。"""
    out = {}

    def dims_of(tp_a, tp_b):
        # TypeProto: tensor_type=1 -> TypeProto.Tensor: shape=2 -> TensorShapeProto:
        #   dim=1 -> Dimension: dim_value=1
        for f, w, a, b in scan(buf, tp_a, tp_b):
            if f != 1 or w != 2:
                continue
            for f2, w2, a2, b2 in scan(buf, a, b):
                if f2 != 2 or w2 != 2:
                    continue
                d = []
                for f3, w3, a3, b3 in scan(buf, a2, b2):
                    if f3 != 1 or w3 != 2:
                        continue
                    v = 0
                    for f4, w4, a4, _ in scan(buf, a3, b3):
                        if f4 == 1 and w4 == 0:
                            v = _varint(buf, a4)[0]
                    d.append(v)
                return d
        return None

    for f, w, a, b in scan(buf, *g):
        if f not in (11, 12, 13) or w != 2:
            continue
        name, tp = "", None
        for f2, w2, a2, b2 in scan(buf, a, b):
            if f2 == 1 and w2 == 2:
                name = buf[a2:b2].decode("utf-8", "replace")
            elif f2 == 2 and w2 == 2:
                tp = (a2, b2)
        if name and tp:
            d = dims_of(*tp)
            if d:
                out[name] = d
    return out


def cpad4(v):
    return (v + 3) // 4 * 4


def check(path):
    buf = path.read_bytes()
    g = graph_of(buf)
    inits = initializers(buf, g)
    vs = value_shapes(buf, g)
    rows = []
    for op, ins, name, at in nodes(buf, g):
        if op != "Conv" or len(ins) < 2 or ins[1] not in inits:
            continue
        w = inits[ins[1]]
        if len(w) != 4:
            continue
        co, ci, kh, kw = w
        strides = at.get("strides") or [1, 1]
        pads = at.get("pads") or [0, 0, 0, 0]
        group = at.get("group", 1)
        if len(pads) != 4:
            continue
        xs = vs.get(ins[0])
        ys = vs.get(name if False else None) if False else None
        # 节点输出名 → 形状：节点里没存 output，只有 name（这里 name 属性是节点名，
        # 与输出张量名常不同）；用 value_info 里的输出形状靠 graph 的 output 名对齐
        # 太脆，所以空间尺寸只按输入张量报。
        rows.append(dict(name=name, wname=ins[1], xname=ins[0], co=co, ci=ci,
                         k=(kh, kw), st=strides, pads=pads, group=group, xs=xs))
    return rows


def main():
    for p in sys.argv[1:]:
        path = Path(p)
        rows = check(path)
        print(f"\n=== {path} ===")
        kxk = [r for r in rows
               if r["k"][0] == r["k"][1] and r["k"][0] >= 2
               and r["group"] == 1 and list(r["st"]) == [1, 1]]
        gated = []
        for r in kxk:
            k = r["k"][0]
            padv = r["pads"]
            sym = all(v == (k - 1) // 2 for v in padv) and 2 * ((k - 1) // 2) == k - 1
            ok_co = cpad4(r["co"]) % 64 == 0
            ok_ci = cpad4(r["ci"]) % 8 == 0
            nfb_ok = None
            if r["xs"] and len(r["xs"]) == 4 and r["xs"][3] > 0:
                ow = r["xs"][3]  # 对称 pad + stride1 ⇒ outW==inW
                nfb_ok = ow - (k - 1) >= 64
            r["sym"], r["ok_co"], r["ok_ci"], r["nfb_ok"] = sym, ok_co, ok_ci, nfb_ok
            if sym and ok_co and ok_ci and nfb_ok:
                gated.append(r)
        print(f"conv 总数 {len(rows)}，其中 k×k(k>1) stride1 group1：{len(kxk)}")
        print(f"  ↳ 过 `conv_kxk_mma` 全套门禁：**{len(gated)}** 条")
        miss = [r for r in kxk if r not in gated]
        from collections import Counter
        c = Counter()
        for r in miss:
            rs = []
            if not r["sym"]:
                rs.append("pad不对称")
            if not r["ok_co"]:
                rs.append("Co%64≠0")
            if not r["ok_ci"]:
                rs.append("Ci%8≠0")
            if r["nfb_ok"] is False:
                rs.append("outW-k+1<64")
            c["+".join(rs)] += 1
        print("  被刷掉的原因直方图：")
        for why, n in c.most_common():
            print(f"    {n:>3} 条  {why}")
        # 按 Co 分布看「如果放宽 Co 判据能捞回多少」
        co_hist = Counter(r["co"] for r in kxk)
        print(f"  k×k 的 Co 分布：{dict(sorted(co_hist.items()))}")
        print("  逐条（Co,Ci,k,pads,sym,Co门,Ci门,空间门,in_shape）：")
        for r in kxk:
            print(f"    {r['name'][:34]:<34} Co={r['co']:<4} Ci={r['ci']:<4} "
                  f"k={r['k'][0]} pads={r['pads']} sym={r['sym']} "
                  f"Co64={r['ok_co']} Ci8={r['ok_ci']} sp={r['nfb_ok']} {r['xs']}")


if __name__ == "__main__":
    main()
