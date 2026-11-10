//! 打印图中指定下标节点的全部属性与输入输出名。
use qppocr_core::onnx::load_onnx;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = &args[1];
    let idx: usize = args[2].parse().unwrap();
    let g = load_onnx(std::path::Path::new(model)).unwrap();
    let n = &g.nodes[idx];
    println!("node[{idx}] {} {}", n.op_type, n.name);
    println!("  inputs: {:?}", n.inputs);
    println!("  outputs: {:?}", n.outputs);
    for a in &n.attrs {
        println!(
            "  attr {} f={:?} i={} s={:?} ints={:?} floats={:?}",
            a.name,
            if a.has_f { Some(a.f) } else { None },
            a.i,
            if a.has_s { Some(&a.s) } else { None },
            a.ints,
            a.floats
        );
    }
    for inn in &n.inputs {
        if let Some(t) = g.initializers.get(inn) {
            println!("  init {inn}: shape={:?} dtype={:?}", t.shape, t.dtype);
        }
    }
}
