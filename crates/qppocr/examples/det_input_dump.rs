//! 逐图打印检测输入尺寸与阶段耗时（口径对齐诊断用）。
//! 用法：det_input_dump <img...>
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let eng = qppocr::Engine::new(qppocr::Tier::Tiny, "models").unwrap();
    for p in &args {
        let img = match qppocr::decode_file(p) {
            Ok(i) => i,
            Err(e) => {
                println!("{p}: decode err {e}");
                continue;
            }
        };
        let r = eng.run(&img).unwrap();
        println!(
            "{} orig={}x{} work={}x{} det={}x{} boxes={} det_inf={:.1} rec_inf={:.1} tot={:.1}",
            p.rsplit(['/', '\\']).next().unwrap_or(p),
            img.w,
            img.h,
            r.work_w,
            r.work_h,
            r.det_input_w,
            r.det_input_h,
            r.num_boxes,
            r.timings.det_infer_ms,
            r.timings.rec_infer_ms,
            r.timings.total_ms,
        );
    }
}
