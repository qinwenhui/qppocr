//! 单图阶段分解（诊断用）：bench N 轮，输出每轮九项计时。
fn main() {
    let img_path = std::env::args().nth(1).unwrap();
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let eng = qppocr::Engine::new(qppocr::Tier::Tiny, "models").unwrap();
    let img = qppocr::decode_file(&img_path).unwrap();
    let _ = eng.run(&img); // warm
    for i in 0..n {
        let r = eng.run(&img).unwrap();
        let t = &r.timings;
        println!(
            "run{}: total={:6.1} det_pre={:5.1} det_inf={:6.1} det_post={:5.1} crop={:5.1} cls={:5.1} rec_pre={:5.1} rec_inf={:6.1} rec_post={:5.1} | 未归类={:5.1} boxes={} retried={} flipped={} unread={}",
            i + 1,
            t.total_ms,
            t.det_pre_ms,
            t.det_infer_ms,
            t.det_post_ms,
            t.crop_ms,
            t.cls_ms,
            t.rec_pre_ms,
            t.rec_infer_ms,
            t.rec_post_ms,
            t.total_ms
                - t.det_pre_ms
                - t.det_infer_ms
                - t.det_post_ms
                - t.crop_ms
                - t.cls_ms
                - t.rec_pre_ms
                - t.rec_infer_ms
                - t.rec_post_ms,
            r.num_boxes,
            r.num_det_retried,
            r.num_flipped,
            r.num_unread,
        );
    }
}
