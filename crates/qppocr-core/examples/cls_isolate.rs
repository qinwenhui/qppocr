//! 复现 rot90 的裁剪，分别喂两个 cls 模型。
use qppocr_core::executor::Session;
use qppocr_core::pipeline::crop::{cls_view, pack_crop};
use qppocr_core::pipeline::image::Image;
use qppocr_core::tensor::{DType, Tensor};

fn main() {
    // 保存自 C++ 的 rot90 裁剪（见 LEAN_SAVE_CROPS）改用现做：
    // 复刻流水线的裁剪路径太长——直接用 C++ 保存的 crop 文件。
    let crops = std::env::var("CROPS").unwrap();
    let crop_dir = std::path::Path::new(&crops);
    let upstream = Session::open(std::path::Path::new("models/cls.onnx")).unwrap();
    let converted = Session::open(std::path::Path::new(
        "D:/qinwh/code/myself/ocr-demo/models/ppocr_cls.onnx",
    ))
    .unwrap();
    for entry in std::fs::read_dir(crop_dir).unwrap() {
        let p = entry.unwrap().path();
        if p.extension().and_then(|e| e.to_str()) != Some("ppm") {
            continue;
        }
        // 最小 PPM 读：P6 / W H / MAXVAL，第三个换行后是像素
        let data = std::fs::read(&p).unwrap();
        let mut pos = 0usize;
        let mut toks: Vec<String> = Vec::new();
        while toks.len() < 3 {
            let nl = data[pos..].iter().position(|&b| b == b'\n').unwrap() + pos;
            toks.push(String::from_utf8_lossy(&data[pos..nl]).to_string());
            pos = nl + 1;
        }
        let (w, h) = {
            let d: Vec<i32> = toks[1]
                .split_whitespace()
                .flat_map(|s| s.parse().ok())
                .collect();
            (d[0], d[1])
        };
        let img = Image {
            w,
            h,
            c: 3,
            orig_w: w,
            orig_h: h,
            data: data[pos..].to_vec(),
        };
        for (name, sess) in [("upstream", &upstream), ("converted", &converted)] {
            let in_name = sess.graph.inputs[0].clone();
            let view = cls_view(&img, 192, 48);
            let mut buf = vec![0f32; 3 * 48 * 192];
            pack_crop(&view, 48, 192, &mut buf, false);
            let out = sess
                .run(vec![(
                    in_name,
                    Tensor {
                        name: String::new(),
                        shape: vec![1, 3, 48, 192],
                        dtype: DType::F32,
                        f32: qppocr_kernels::buf::F32Buf::from_vec(&buf),
                        i64: Vec::new(),
                    },
                )])
                .unwrap();
            let o = &out[0].f32;
            let flip = o[1] > o[0] && o[1] >= 0.9;
            println!(
                "{} {:?}: [{:.4}, {:.4}] flip={} ({}x{})",
                name,
                p.file_name().unwrap().to_string_lossy(),
                o[0],
                o[1],
                flip,
                img.w,
                img.h
            );
        }
    }
}
