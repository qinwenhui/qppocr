//! 单图阶段分解（诊断用）：bench N 轮，输出每轮九项计时。
fn main() {
    let img_path = std::env::args().nth(1).unwrap();
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    // 档位/设备靠环境变量切（默认 tiny+auto，保持原行为）：
    // `PHASE_TIER=medium PHASE_DEVICE=metal`。诊断工具，改配置不用重编。
    let tier = match std::env::var("PHASE_TIER").as_deref() {
        Ok("small") => qppocr::Tier::Small,
        Ok("medium") => qppocr::Tier::Medium,
        _ => qppocr::Tier::Tiny,
    };
    let device = match std::env::var("PHASE_DEVICE").as_deref() {
        Ok("metal") => qppocr::DeviceChoice::Gpu {
            api: qppocr::GpuApi::Metal,
            index: None,
        },
        _ => qppocr::DeviceChoice::Cpu,
    };
    // 预设跟 CLI 的 `bench` 默认对齐（balanced），否则两边数字不可比。
    let eng = qppocr::Engine::builder()
        .tier(tier)
        .preset(qppocr::Preset::Balanced)
        .device(device)
        .build("models")
        .unwrap();
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
