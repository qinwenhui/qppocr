//! 对拍探针：读  侧 dump 的 A/B/bias，跑本 crate 的 sgemm，写 C。
//! 用法：`sgemm_probe M N K`（数据文件在当前目录）。

use qppocr_kernels::activation::Activation;
use qppocr_kernels::gemm::sgemm;

fn read_f32(path: &str) -> Vec<f32> {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args[1].parse().unwrap();
    let n: usize = args[2].parse().unwrap();
    let k: usize = args[3].parse().unwrap();
    let a = read_f32(&format!("sgemm_A_{m}_{n}_{k}.f32"));
    let b = read_f32(&format!("sgemm_B_{m}_{n}_{k}.f32"));
    let bias = read_f32(&format!("sgemm_bias_{m}_{n}_{k}.f32"));
    let mut c = vec![0f32; m * n];
    sgemm(
        &a,
        &b,
        &mut c,
        m,
        n,
        k,
        n,
        Some(&bias),
        &Activation::default(),
    );
    let mut out = Vec::with_capacity(c.len() * 4);
    for v in &c {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(format!("sgemm_rust_{m}_{n}_{k}.f32"), out).unwrap();
    println!("done M={m} N={n} K={k}");
}
