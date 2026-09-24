//! 上游模型 + cls_window 开/关：旋转测试图的 cls 判定回归检查。
fn main() {
    let imgs = ["rot180", "rot270", "rot90", "rotated"];
    for &win in &[false, true] {
        let eng = qppocr::Engine::builder()
            .tier(qppocr::Tier::Small)
            .advanced(move |a| a.cls_window = win)
            .build("models")
            .unwrap();
        println!("== cls_window={win}");
        for n in imgs {
            let r = eng
                .run_image_file(format!("D:/qinwh/code/myself/ocr-demo/testdata/{n}.png"))
                .unwrap();
            let flip = r.num_flipped;
            let t: Vec<&str> = r.lines.iter().map(|l| l.text.as_str()).collect();
            println!("  {n}: flipped={flip} texts={:?}", &t[..t.len().min(2)]);
        }
    }
}
