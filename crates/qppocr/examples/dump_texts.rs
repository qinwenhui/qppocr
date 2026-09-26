//! 把整个目录的图逐张跑一遍，输出两列 TSV（文件名 + 行文本，行内以 | 分隔）。
//!
//! 用途：改造前后逐字符对拍（`diff` 两个输出即可），以及给评分脚本喂数据。
//! 运行：cargo run --release -p qppocr --example dump_texts -- <图片目录> <输出文件>
fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let out = std::env::args().nth(2).unwrap();
    let eng = qppocr::Engine::new(qppocr::Tier::Tiny, "models").unwrap();
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "jpg").unwrap_or(false))
        .collect();
    names.sort();
    let mut s = String::new();
    for p in names {
        let r = eng.run_image_file(&p).unwrap();
        s.push_str(&format!(
            "{}\t{}\n",
            p.file_name().unwrap().to_string_lossy(),
            r.lines
                .iter()
                .map(|l| l.text.clone())
                .collect::<Vec<_>>()
                .join("|")
        ));
    }
    std::fs::write(out, s).unwrap();
}
