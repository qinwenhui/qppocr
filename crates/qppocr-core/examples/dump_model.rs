//! 对拍驱动：把固定输入喂给模型，QPPOCR_DUMP_DIR 逐节点落盘。
//!
//! 用法：`cargo run --release -p qppocr-core --example dump_model --
//!              <model.onnx> <in.f32> <d0,d1,...> <out.f32>`
//!
//! 输入 .f32 与 C++ `engine_runner` 同格式（裸小端 f32 流）；输出带
//! i32 输出数 + 每输出 (i32 rank + i64×rank + f32 数据) 头——两侧逐字节
//! diff 即阶段 2 判据。

use std::io::Write;

use qppocr_core::executor::Session;
use qppocr_core::tensor::{DType, Tensor};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: dump_model <model.onnx> <in.f32> <d0,d1,...> <out.f32>");
        std::process::exit(2);
    }
    let model = &args[1];
    let in_path = &args[2];
    let dims: Vec<i64> = args[3].split(',').filter_map(|s| s.parse().ok()).collect();
    let out_path = &args[4];

    let t0 = std::time::Instant::now();
    let sess = Session::open(std::path::Path::new(model)).expect("open model");
    println!("[load] {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);

    let raw = std::fs::read(in_path).expect("read input");
    let n: usize = dims.iter().product::<i64>() as usize;
    let mut data = vec![0f32; n];
    for (i, v) in data.iter_mut().enumerate() {
        let b: [u8; 4] = raw[i * 4..i * 4 + 4].try_into().unwrap();
        *v = f32::from_le_bytes(b);
    }
    let in_name = sess
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .unwrap()
        .clone();

    let t0 = std::time::Instant::now();
    let out = sess
        .run(vec![(
            in_name,
            Tensor {
                name: "x".into(),
                shape: dims,
                dtype: DType::F32,
                f32: qppocr_kernels::buf::F32Buf::from_vec(&data),
                i64: Vec::new(),
            },
        )])
        .expect("run");
    println!(
        "[run]  {:.1} ms  ({} outputs)",
        t0.elapsed().as_secs_f64() * 1e3,
        out.len()
    );

    let mut f = std::fs::File::create(out_path).expect("write output");
    f.write_all(&(out.len() as i32).to_le_bytes()).unwrap();
    for o in &out {
        f.write_all(&(o.rank() as i32).to_le_bytes()).unwrap();
        for d in &o.shape {
            f.write_all(&d.to_le_bytes()).unwrap();
        }
        for v in o.f32.as_slice() {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}
