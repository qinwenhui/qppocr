//! 打印一张 ONNX 图的**连接关系**：每个节点的算子、输入来自谁、输出给谁，
//! 以及张量的元素个数（据此估算搬运量）。
//!
//! 用途：找可变现的算子融合。逐算子的耗时表看不出拓扑——`Add` 前面是
//! `Resize` 还是 `Conv`，能省的搬运量差一倍以上。
//!
//! cargo run --release -p qppocr-core --example graph_dump -- <model.onnx> [只打印含此子串的]

use std::collections::HashMap;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("用法: graph_dump <model.onnx> [过滤子串]");
    let filter = std::env::args().nth(2);
    let g = qppocr_core::onnx::load_onnx(std::path::Path::new(&path)).unwrap();

    // 张量元素数：从 initializer 或图输入拿；中间张量从产出它的节点的属性推
    let mut elems: HashMap<String, i64> = HashMap::new();
    for (name, t) in &g.initializers {
        elems.insert(name.clone(), t.shape.iter().product::<i64>().max(1));
    }
    let mut producer: HashMap<&str, usize> = HashMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        for o in &n.outputs {
            producer.insert(o.as_str(), i);
        }
    }
    let mut users: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        for inn in &n.inputs {
            if !inn.is_empty() {
                users.entry(inn.as_str()).or_default().push(i);
            }
        }
    }

    let mut total_io = 0i64;
    for (i, n) in g.nodes.iter().enumerate() {
        if let Some(f) = &filter {
            let hit = n.op_type.contains(f.as_str())
                || n.inputs.iter().any(|x| x.contains(f.as_str()))
                || n.outputs.iter().any(|x| x.contains(f.as_str()));
            if !hit {
                continue;
            }
        }
        let ins: Vec<String> = n
            .inputs
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| match producer.get(s.as_str()) {
                Some(p) => format!("{s}←#{p}({})", g.nodes[*p].op_type),
                None => format!("{s}({})", elems.get(s).copied().unwrap_or(0)),
            })
            .collect();
        let outs: Vec<String> = n
            .outputs
            .iter()
            .map(|s| {
                format!(
                    "{s}→[{}]",
                    users.get(s.as_str()).map(|v| v.len()).unwrap_or(0)
                )
            })
            .collect();
        // 搬运量估算：读全部输入 + 写输出（按 known 元素数）
        let rd: i64 = n
            .inputs
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| estimate(&g, &elems, s))
            .sum();
        let wr: i64 = n.outputs.iter().map(|s| estimate(&g, &elems, s)).sum();
        total_io += rd + wr;
        println!(
            "#{i:<4} {:<18} 读{rd:>9} 写{wr:>9}   {:?}  ->  {:?}",
            n.op_type, ins, outs
        );
    }
    println!(
        "\n合计（估算）{} 个 float = {} MB",
        total_io,
        total_io * 4 / 1_000_000
    );
}

/// 元素数估算：initializer 直接查；中间张量按「产出节点的第一个输入的元素数」
/// 近似（对 Conv 是入通道×H×W，误差在倍数以内，够用来排序找机会）。
fn estimate(g: &qppocr_core::onnx::model::Graph, elems: &HashMap<String, i64>, name: &str) -> i64 {
    if let Some(v) = elems.get(name) {
        return *v;
    }
    // 顺着 producer 往上找一步
    for n in &g.nodes {
        if n.outputs.iter().any(|o| o == name) {
            if let Some(first) = n.inputs.first() {
                return elems.get(first).copied().unwrap_or(0);
            }
        }
    }
    0
}
