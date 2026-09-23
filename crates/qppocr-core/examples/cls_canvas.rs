//! 经验测定上游 cls 的画布与输出语义：对一张已知方向的图试多种配置。
//! cargo run --release -p qppocr-core --example cls_canvas -- <image>
use qppocr_core::executor::Session;
use qppocr_core::pipeline::crop::pack_crop;
use qppocr_core::pipeline::image::{Image, rotate_image};
use qppocr_core::tensor::{DType, Tensor};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let img = image::open(&args[1]).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let im = Image {
        w: w as i32,
        h: h as i32,
        c: 3,
        orig_w: w as i32,
        orig_h: h as i32,
        data: img.into_raw(),
    };
    let sess = Session::open(std::path::Path::new("models/cls.onnx")).unwrap();
    let in_name = sess.graph.inputs[0].clone();
    for (ch, cw) in [
        (48i32, 192i32),
        (80, 160),
        (48, 80),
        (64, 192),
        (80, 192),
        (48, 160),
    ] {
        for deg in [0i32, 180] {
            let r = rotate_image(&im, deg);
            let mut buf = vec![0f32; 3 * (ch as usize) * (cw as usize)];
            pack_crop(&r, ch, cw, &mut buf, false);
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
            println!(
                "{ch}x{cw} rot{deg}: out=[{:.4}, {:.4}] argmax={}",
                o[0],
                o[1],
                if o[1] > o[0] { 1 } else { 0 }
            );
        }
    }
}
