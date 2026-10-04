//! cls 批大小扫描：cls_batch ∈ {1,2,4,8,16} 在给定图集上的 cls 阶段墙钟。
//!
//! 用法：cls_batch_sweep <img...>
//!
//! 1 = 逐行扇出（每行一次前向）；越大每次前向算的行越多、批间并行度越低。
//! 每张图跑 REP 轮取该图的最小 cls 墙钟，再对图求和——与 bench 口径一致。
//! 同时比对**逐图翻转行数**：批起来只该改前向形状，不该改结论。
const REP: usize = 3;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let imgs: Vec<qppocr::Image> = args
        .iter()
        .map(|p| qppocr::decode_file(p).unwrap())
        .collect();

    let mut base: Option<Vec<usize>> = None;
    for batch in [1usize, 2, 4, 8, 16] {
        let eng = qppocr::Engine::builder()
            .tier(qppocr::Tier::Tiny)
            .advanced(move |a| a.cls_batch = batch)
            .build("models")
            .unwrap();
        for img in &imgs {
            let _ = eng.run(img); // warm
        }
        let mut cls_sum = 0.0f64;
        let mut tot_sum = 0.0f64;
        let mut sig = Vec::with_capacity(imgs.len());
        for img in &imgs {
            let (mut c, mut t) = (f64::MAX, f64::MAX);
            let mut flipped = 0;
            for _ in 0..REP {
                let r = eng.run(img).unwrap();
                c = c.min(r.timings.cls_ms);
                t = t.min(r.timings.total_ms);
                flipped = r.num_flipped;
            }
            cls_sum += c;
            tot_sum += t;
            sig.push(flipped);
        }
        let n = imgs.len();
        println!(
            "cls_batch={batch:<3} cls={:7.2} ms（{:.2}/图）  total={:8.1} ms（{:.1}/图）",
            cls_sum,
            cls_sum / n as f64,
            tot_sum,
            tot_sum / n as f64
        );
        match &base {
            None => base = Some(sig),
            Some(b) => println!(
                "  ↳ 逐图翻转数与 cls_batch=1 一致：{}",
                if *b == sig { "是" } else { "否 ⚠" }
            ),
        }
    }
}
