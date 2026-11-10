//! 扫上游 cls 的预处理组合：画布 × 归一化 × 是否开窗。
//! CROPS 指向含倒置行的裁剪目录（rot180 的  裁剪）。
use qppocr_core::executor::Session;
use qppocr_core::pipeline::crop::{cls_view, pack_crop};
use qppocr_core::pipeline::image::{Image, rotate_image};
use qppocr_core::tensor::{DType, Tensor};

fn load_ppm(p: &std::path::Path) -> Image {
    let data = std::fs::read(p).unwrap();
    let mut pos = 0usize;
    let mut toks: Vec<String> = Vec::new();
    while toks.len() < 3 {
        let nl = data[pos..].iter().position(|&b| b == b'\n').unwrap() + pos;
        toks.push(String::from_utf8_lossy(&data[pos..nl]).to_string());
        pos = nl + 1;
    }
    let d: Vec<i32> = toks[1]
        .split_whitespace()
        .flat_map(|s| s.parse().ok())
        .collect();
    Image {
        w: d[0],
        h: d[1],
        c: 3,
        orig_w: d[0],
        orig_h: d[1],
        data: data[pos..].to_vec(),
    }
}

fn main() {
    let crops = std::env::var("CROPS").unwrap();
    let sess = Session::open(std::path::Path::new("models/cls.onnx")).unwrap();
    let in_name = sess.graph.inputs[0].clone();
    for entry in std::fs::read_dir(&crops).unwrap() {
        let p = entry.unwrap().path();
        if p.extension().and_then(|e| e.to_str()) != Some("ppm") {
            continue;
        }
        let orig = load_ppm(&p);
        let upside = rotate_image(&orig, 180);
        for (name, img) in [("as-saved", &orig), ("180", &upside)] {
            for (ch, cw, win) in [
                (48i32, 192i32, true),
                (48, 192, false),
                (80, 160, true),
                (80, 160, false),
                (48, 320, false),
                (64, 256, false),
            ] {
                let view = if win {
                    cls_view(img, cw, ch)
                } else {
                    img.clone()
                };
                let mut buf = vec![0f32; 3 * (ch as usize) * (cw as usize)];
                pack_crop(&view, ch, cw, &mut buf, false);
                let out = sess
                    .run(vec![(
                        in_name.clone(),
                        Tensor {
                            name: String::new(),
                            shape: vec![1, 3, ch as i64, cw as i64],
                            dtype: DType::F32,
                            f32: qppocr_kernels::buf::F32Buf::from_vec(&buf),
                            i64: Vec::new(),
                        },
                    )])
                    .unwrap();
                let o = &out[0].f32;
                println!("{:?}", p.file_name().unwrap().to_string_lossy());
                println!(
                    "  {} {}x{} win={}: [{:.4}, {:.4}] flip={}",
                    name,
                    ch,
                    cw,
                    win,
                    o[0],
                    o[1],
                    o[1] > o[0] && o[1] >= 0.9
                );
            }
        }
    }
}
