//! 图优化（`graph_opt.cpp` ）。
//!
//! ONNX Runtime 必须通用：任意算子、任意拓扑、任意 dtype，融合要运行时决定。
//! 我们只跑 PP-OCR（v4/v6，det/rec/cls），可以认出这些导出器吐的**确切
//! 子图**并换成单个手写内核——「通用框架税」在这里退掉。
//!
//! 全部是**结构匹配**：常数从模型里读、不假设，换个数重导出照样融合；
//! 不匹配的原样保留。

use std::collections::{HashMap, HashSet};

use crate::onnx::model::{Attribute, Graph, Node};
use crate::tensor::DType;

fn act_f(a: Option<&Attribute>, dflt: f32) -> f32 {
    a.filter(|x| x.has_f).map(|x| x.f).unwrap_or(dflt)
}

fn get_i(a: Option<&Attribute>, dflt: i64) -> i64 {
    a.filter(|x| x.has_i).map(|x| x.i).unwrap_or(dflt)
}

/// 单元素 initializer——融合可以当标量用的东西。
fn scalar_const(g: &Graph, name: &str) -> Option<f32> {
    let t = g.initializers.get(name)?;
    if t.dtype != DType::F32 || t.numel() != 1 || t.f32.is_empty() {
        return None;
    }
    Some(t.f32[0])
}

/// tensor 名 -> 产出节点下标（owned：后续要可变借用 g.nodes）。
fn producer_map(g: &Graph) -> HashMap<String, usize> {
    let mut p = HashMap::with_capacity(g.nodes.len() * 2);
    for (i, n) in g.nodes.iter().enumerate() {
        for o in &n.outputs {
            p.insert(o.clone(), i);
        }
    }
    p
}

/// tensor 名 -> 被多少个节点读（owned，同上）。
fn consumer_map(g: &Graph) -> HashMap<String, i64> {
    let mut c: HashMap<String, i64> = HashMap::new();
    for n in &g.nodes {
        for inn in &n.inputs {
            if !inn.is_empty() {
                *c.entry(inn.clone()).or_insert(0) += 1;
            }
        }
    }
    c
}

/// GELU 子图坍缩：`c3·(Mul(X·(Add(Erf(Div(X,c1)),c2))))` 五个节点
///（Div/Erf/Add/Mul/Mul——导出器就这么拆的）换成一个 FusedGelu。
pub fn fuse_gelu(g: &mut Graph) -> i64 {
    let producer = producer_map(g);
    let consumers = consumer_map(g);

    let mut drop: HashSet<usize> = HashSet::new();
    // 替换按节点下标放：融合节点落在原外层 Mul 的位置。追加会破坏拓扑序，
    // 执行器（按序走 + 重试未就绪节点）就要跑跟交错融合数一样多的趟数。
    let mut replace: HashMap<usize, Node> = HashMap::new();

    let nodes = std::sync::Arc::new(std::sync::Mutex::new(()));
    let _ = nodes; // 占位说明：node_at 只读借用 g.nodes，与 replace 阶段无重叠
    let node_at = |nm: &str| -> Option<&Node> { producer.get(nm).map(|&i| &g.nodes[i]) };

    for i in 0..g.nodes.len() {
        let outer = &g.nodes[i];
        if outer.op_type != "Mul" || outer.inputs.len() != 2 || drop.contains(&i) {
            continue;
        }

        // ---- 外层：out = Mul(inner, c3) ----
        let (c3, inner_name) = match (
            scalar_const(g, &outer.inputs[0]),
            scalar_const(g, &outer.inputs[1]),
        ) {
            (Some(v), _) => (v, outer.inputs[1].clone()),
            (None, Some(v)) => (v, outer.inputs[0].clone()),
            (None, None) => continue,
        };

        let inner = match node_at(&inner_name) {
            Some(n) if n.op_type == "Mul" && n.inputs.len() == 2 => n,
            _ => continue,
        };
        if consumers.get(inner_name.as_str()).copied().unwrap_or(0) != 1 {
            continue; // 必须是这个 gelu 私有的
        }

        // ---- 内层：inner = Mul(X, add_out) ----
        let mut addn: Option<&Node> = None;
        let mut x_name = String::new();
        for k in 0..2 {
            let cand = node_at(&inner.inputs[k]);
            if let Some(c) = cand {
                if c.op_type == "Add"
                    && consumers
                        .get(inner.inputs[k].as_str())
                        .copied()
                        .unwrap_or(0)
                        == 1
                {
                    addn = Some(c);
                    x_name = inner.inputs[1 - k].clone();
                    break;
                }
            }
        }
        let addn = match addn {
            Some(a) if !x_name.is_empty() => a,
            _ => continue,
        };

        // ---- add：add_out = Add(erf_out, c2) ----
        let mut erfn: Option<&Node> = None;
        let mut c2 = 0f32;
        for k in 0..2 {
            let cand = node_at(&addn.inputs[k]);
            if let Some(c) = cand {
                if c.op_type == "Erf"
                    && consumers.get(addn.inputs[k].as_str()).copied().unwrap_or(0) == 1
                {
                    match scalar_const(g, &addn.inputs[1 - k]) {
                        Some(v) => {
                            c2 = v;
                            erfn = Some(c);
                        }
                        None => {
                            erfn = None;
                        }
                    }
                    break;
                }
            }
        }
        let erfn = match erfn {
            Some(e) => e,
            None => continue,
        };

        // ---- erf：erf_out = Erf(div_out) ----
        if erfn.inputs.len() != 1 {
            continue;
        }
        let divn = match node_at(&erfn.inputs[0]) {
            Some(d) if d.op_type == "Div" => d,
            _ => continue,
        };
        if consumers.get(erfn.inputs[0].as_str()).copied().unwrap_or(0) != 1 {
            continue;
        }

        // ---- div：div_out = Div(X, c1)——X 必须是同一个张量 ----
        if divn.inputs.len() != 2 {
            continue;
        }
        let c1;
        if divn.inputs[0] == x_name {
            match scalar_const(g, &divn.inputs[1]) {
                Some(v) => c1 = v,
                None => continue,
            }
        } else if divn.inputs[1] == x_name {
            match scalar_const(g, &divn.inputs[0]) {
                Some(v) => c1 = v,
                None => continue,
            }
        } else {
            continue;
        }

        // ---- 匹配 ----
        let mut gn = Node::default();
        gn.op_type = "FusedGelu".into();
        gn.name = format!("{}_fused", outer.name);
        gn.inputs = vec![x_name.clone()];
        gn.outputs = outer.outputs.clone();
        for (nm, v) in [("c1", c1), ("c2", c2), ("c3", c3)] {
            gn.attrs.push(Attribute {
                name: nm.into(),
                f: v,
                has_f: true,
                ..Default::default()
            });
        }
        replace.insert(i, gn);

        drop.insert(i);
        drop.insert(producer[inner_name.as_str()]);
        drop.insert(producer[addn.outputs[0].as_str()]);
        drop.insert(producer[erfn.outputs[0].as_str()]);
        drop.insert(producer[divn.outputs[0].as_str()]);
    }

    if replace.is_empty() {
        return 0;
    }
    let nfused = replace.len() as i64;
    let mut kept: Vec<Node> = Vec::with_capacity(g.nodes.len());
    for (i, n) in std::mem::take(&mut g.nodes).into_iter().enumerate() {
        if let Some(r) = replace.remove(&i) {
            kept.push(r);
            continue;
        }
        if drop.contains(&i) {
            continue;
        }
        kept.push(n);
    }
    g.nodes = kept;
    nfused
}

/// 相互抵消的 Transpose 对：`Transpose(perm)` 紧跟其逆，产生一张 arena 里
/// 已有数据的逐字节拷贝。PP-OCRv6 的 rec 头就吐这个：
///
/// ```text
/// Squeeze    (6,160,1,40) -> (6,160,40)
/// Transpose  (6,160,40)   -> (6,40,160)
/// Transpose  (6,40,160)   -> (6,160,40)     <- 上面那个的逆
/// Unsqueeze  (6,160,40)   -> (6,160,1,40)
/// ```
///
/// 不便宜：transpose 逐元素走 r 维下标、循环里带整数除法，两个各是一整趟
/// GB/s 级的 pass。两个节点（连带分叉）只为被撤销而存在。
///
/// 匹配保守：两个节点都带显式同长 `perm`，且复合为恒等。无 `perm` 的
/// Transpose 是反转轴，这里不知道秩，不动。
pub fn drop_identity_pairs(g: &mut Graph) -> i64 {
    let mut removed: i64 = 0;
    let mut changed = true;
    // 删一对可能暴露另一对（上面的 Squeeze/Unsqueeze 要等中间的 transpose
    // 没了才抵消），跑到不动点。循环有界：每趟至少删一个节点。
    while changed {
        changed = false;
        let prod = producer_map(g);

        let perm_of = |n: &Node| -> Option<Vec<i64>> {
            let a = n.attr("perm")?;
            if a.ints.is_empty() {
                return None; // 默认反转：秩未知
            }
            Some(a.ints.clone())
        };

        // 先按值收好要查的名字（闭包不能借 g.nodes：后面要 remove）
        let consumers_of = |nodes: &[Node], t: &str| -> i64 {
            nodes
                .iter()
                .flat_map(|n| n.inputs.iter())
                .filter(|inn| inn.as_str() == t)
                .count() as i64
        };

        for bi in 0..g.nodes.len() {
            if changed {
                break;
            }
            if g.nodes[bi].op_type != "Transpose"
                || g.nodes[bi].inputs.len() != 1
                || g.nodes[bi].outputs.len() != 1
            {
                continue;
            }
            if g.is_graph_output(&g.nodes[bi].outputs[0]) {
                continue;
            }
            let y1 = g.nodes[bi].inputs[0].clone();
            let ai = match prod.get(y1.as_str()) {
                Some(&i) => i,
                None => continue,
            };
            if ai == bi || g.nodes[ai].op_type != "Transpose" || g.nodes[ai].inputs.len() != 1 {
                continue;
            }

            let p = match perm_of(&g.nodes[ai]) {
                Some(p) => p,
                None => continue,
            };
            let q = match perm_of(&g.nodes[bi]) {
                Some(q) => q,
                None => continue,
            };
            if p.len() != q.len() {
                continue;
            }
            let r = p.len();
            let ident = (0..r).all(|i| {
                let qi = q[i];
                qi >= 0 && qi < r as i64 && p[qi as usize] == i as i64
            });
            if !ident {
                continue;
            }

            // B 的输出是 A 的输入。改写 B 输出的所有读者，删 B。
            let x = g.nodes[ai].inputs[0].clone();
            let y2 = g.nodes[bi].outputs[0].clone();
            for n in g.nodes.iter_mut() {
                for inn in n.inputs.iter_mut() {
                    if *inn == y2 {
                        *inn = x.clone();
                    }
                }
            }
            g.nodes.remove(bi);
            removed += 1;
            // A 现在死了——除非还有别人读 y1（或它是输出）
            let left = consumers_of(&g.nodes, &y1);
            if left == 0 && !g.is_graph_output(&y1) {
                for k in 0..g.nodes.len() {
                    if g.nodes[k].outputs.len() == 1 && g.nodes[k].outputs[0] == y1 {
                        g.nodes.remove(k);
                        removed += 1;
                        break;
                    }
                }
            }
            changed = true;
        }
    }
    removed
}

/// Conv 紧跟激活、conv 输出无其他消费者：
/// `y = Conv(x, w)` + `z = FusedGelu(y)` 变成一个带激活参数的 Conv 节点。
/// 内核自己施加激活，省掉一整个节点：分叉、arena 槽位、逐节点簿记。
///
/// 激活作用在**GEMM 已写出的输出**上、数据还在 cache——不在寄存器 store
/// 阶段里变换累加器。测过：折进 epilogue 也精确，但 det_small 慢 5.3%
///（erf256_ps 把一次除法和一堆常数拖进已经占满 16 个 YMM 的循环）。
/// 写后 pass 不占 GEMM 寄存器。
pub fn fuse_conv_activation(g: &mut Graph) -> i64 {
    let mut consumers: HashMap<String, i64> = HashMap::new();
    let mut consumer_of: HashMap<String, usize> = HashMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        for inn in &n.inputs {
            if inn.is_empty() {
                continue;
            }
            *consumers.entry(inn.clone()).or_insert(0) += 1;
            consumer_of.entry(inn.clone()).or_insert(i);
        }
    }

    let mut fuse: Vec<(usize, usize)> = Vec::new(); // (conv 下标, 激活下标)
    for i in 0..g.nodes.len() {
        let cn = &g.nodes[i];
        if cn.op_type != "Conv" || cn.outputs.len() != 1 {
            continue;
        }
        let co = cn.outputs[0].clone();
        if g.is_graph_output(&co) || consumers.get(&co).copied().unwrap_or(0) != 1 {
            continue;
        }
        let cu = match consumer_of.get(&co) {
            Some(&x) => x,
            None => continue,
        };
        let an = &g.nodes[cu];
        if an.op_type != "FusedGelu" || an.inputs.len() != 1 || an.outputs.len() != 1 {
            continue;
        }
        if an.inputs[0] != co {
            continue;
        }
        // 只融合 group == 1：depthwise 不走 sgemm，一个静默丢掉融合激活的
        // conv2d 比一个从不接手的糟糕得多。
        if get_i(cn.attr("group"), 1) != 1 {
            continue;
        }
        fuse.push((i, cu));
    }
    if fuse.is_empty() {
        return 0;
    }

    let mut drop: HashSet<usize> = HashSet::new();
    for &(ci, ai) in &fuse {
        let an_attrs = g.nodes[ai].attrs.clone();
        let an_out = g.nodes[ai].outputs[0].clone();
        let cn = &mut g.nodes[ci];
        let mut kind = Attribute::default();
        kind.name = "act".into();
        kind.has_i = true;
        kind.i = 1; // 1 = gelu
        cn.attrs.push(kind);
        // 移动 c1/c2/c3 → act_c1/act_c2/act_c3
        for (from, to, dflt) in [
            ("c1", "act_c1", 1.414_213_5f32),
            ("c2", "act_c2", 1.0),
            ("c3", "act_c3", 0.5),
        ] {
            let v = act_f(an_attrs.iter().find(|a| a.name == from), dflt);
            cn.attrs.retain(|x| x.name != to);
            cn.attrs.push(Attribute {
                name: to.into(),
                f: v,
                has_f: true,
                ..Default::default()
            });
        }
        cn.outputs[0] = an_out; // 保留激活的输出名，消费者才能解析
        drop.insert(ai);
    }

    let n = fuse.len() as i64;
    let mut kept: Vec<Node> = Vec::with_capacity(g.nodes.len());
    for (i, node) in std::mem::take(&mut g.nodes).into_iter().enumerate() {
        if drop.contains(&i) {
            continue;
        }
        kept.push(node);
    }
    g.nodes = kept;
    n
}

/// Conv 后紧跟单消费者 Add(bias)：bias 折进 Conv 第三输入。
/// 省掉最常见的一次 Add 的读改写——转换版模型把 bias 拆成独立 Add
///（上游原件折在 Conv 里，这个 pass 对它们是 no-op）。
pub fn fold_conv_bias(g: &mut Graph) -> i64 {
    let consumers = consumer_map(g);
    // tensor -> 唯一读者的下标
    let mut consumer_of: HashMap<String, usize> = HashMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        for inn in &n.inputs {
            if !inn.is_empty() {
                consumer_of.entry(inn.clone()).or_insert(i);
            }
        }
    }

    let mut absorb: Vec<(usize, usize)> = Vec::new(); // (conv 下标, add 下标)
    for i in 0..g.nodes.len() {
        let cn = &g.nodes[i];
        if cn.op_type != "Conv" || cn.inputs.len() != 2 {
            continue; // 已经有 bias
        }
        if cn.outputs.len() != 1 {
            continue;
        }
        let conv_out = cn.outputs[0].clone();
        if g.is_graph_output(&conv_out) {
            continue;
        }
        if consumers.get(&conv_out).copied().unwrap_or(0) != 1 {
            continue; // 必须恰好喂一个消费者
        }
        // producer 映到的是**产出** conv_out 的节点（Conv 自己）；要的 Add
        // 是**读** conv_out 的节点，用 consumer_of 查。
        let cu = match consumer_of.get(&conv_out) {
            Some(&x) => x,
            None => continue,
        };
        let an = &g.nodes[cu];
        if an.op_type != "Add" || an.inputs.len() != 2 || an.outputs.len() != 1 {
            continue;
        }

        // 另一操作数必须是每输出通道一值的常量
        let ww = match g.initializers.get(&cn.inputs[1]) {
            Some(w) => w,
            None => continue,
        };
        if ww.shape.is_empty() {
            continue;
        }
        let m = ww.shape[0];

        let bname = if an.inputs[0] == conv_out {
            an.inputs[1].clone()
        } else if an.inputs[1] == conv_out {
            an.inputs[0].clone()
        } else {
            continue;
        };
        let bb = match g.initializers.get(&bname) {
            Some(b) => b,
            None => continue,
        };
        if bb.dtype != DType::F32 || bb.numel() != m {
            continue;
        }

        absorb.push((i, cu));
    }
    if absorb.is_empty() {
        return 0;
    }

    let mut drop_add: HashSet<usize> = HashSet::new();
    for &(ci, ai) in &absorb {
        let an_in0 = g.nodes[ai].inputs[0].clone();
        let an_in1 = g.nodes[ai].inputs[1].clone();
        let an_out = g.nodes[ai].outputs[0].clone();
        let conv_out = g.nodes[ci].outputs[0].clone();
        let bname = if an_in0 == conv_out { an_in1 } else { an_in0 };
        let cn = &mut g.nodes[ci];
        cn.inputs.push(bname); // conv2d 本来就会施加逐通道 bias
        cn.outputs[0] = an_out; // 保留 Add 的名字，消费者才能解析
        drop_add.insert(ai);
    }

    let n = absorb.len() as i64;
    let mut kept: Vec<Node> = Vec::with_capacity(g.nodes.len());
    for (i, node) in std::mem::take(&mut g.nodes).into_iter().enumerate() {
        if drop_add.contains(&i) {
            continue;
        }
        kept.push(node);
    }
    g.nodes = kept;
    n
}
