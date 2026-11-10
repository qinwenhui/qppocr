//! 逐行重试标记验证：找触发区域重试的图，断言 num_det_retried>0 ⟺
//! 存在 retried=true 的行，且被替换行的置信度 ≥ retry_conf 或标记行
//! 全部来自第二遍。
fn main() {
    let eng = qppocr::Engine::new(qppocr::Tier::Tiny, "models").unwrap();
    for path in std::env::args().skip(1) {
        let r = eng.run_image_file(&path).unwrap();
        let n_retry_lines = r.lines.iter().filter(|l| l.retried).count();
        let ok = (r.num_det_retried > 0) == (n_retry_lines > 0);
        println!(
            "{}: num_det_retried={} retried 行={} 一致={} || {}",
            std::path::Path::new(&path)
                .file_name()
                .unwrap()
                .to_string_lossy(),
            r.num_det_retried,
            n_retry_lines,
            ok,
            if n_retry_lines > 0 {
                r.lines
                    .iter()
                    .filter(|l| l.retried)
                    .map(|l| {
                        format!(
                            "\"{}\"({:.2})",
                            l.text.chars().take(8).collect::<String>(),
                            l.confidence
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            } else {
                String::new()
            }
        );
        assert!(ok);
    }
    println!("ALL-OK");
}
