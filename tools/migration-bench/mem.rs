// 与 mem.cpp 对等：常驻权重 + 反复申请/释放的中间张量。
use std::time::Instant;

// PROCESS_MEMORY_COUNTERS 的字段顺序：cb(u32) + PageFaultCount(u32)，之后才是 7 个
// SIZE_T。第一版把 PageFaultCount 当成 usize，整个结构读串了一位，峰值读成 0.0 ——
// "读得到数但数是错的" 正是这个项目一直在防的那类东西。
#[repr(C)]
struct Pmc {
    cb: u32,
    page_faults: u32,
    peak_ws: usize,
    ws: usize,
    rest: [usize; 6],
}

#[link(name = "psapi")]
extern "system" {
    fn GetCurrentProcess() -> isize;
    fn GetProcessMemoryInfo(h: isize, p: *mut Pmc, cb: u32) -> i32;
}

fn peak_ws_mb() -> f64 {
    let mut p: Pmc = unsafe { std::mem::zeroed() };
    p.cb = std::mem::size_of::<Pmc>() as u32;
    unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut p, p.cb) };
    p.peak_ws as f64 / 1048576.0
}

fn main() {
    let mut weights: Vec<Vec<f32>> = Vec::new();
    for _ in 0..8 {
        weights.push(vec![0.01f32; 2_500_000]); // ~80 MB 常驻
    }
    let mut touched = 0.0f64;
    let t0 = Instant::now();
    for _ in 0..400 {
        let mut tmp: Vec<Vec<f32>> = Vec::new();
        for j in 0..24usize {
            tmp.push(vec![0.5f32; 20000 + j * 7000]);
        }
        for t in &tmp {
            touched += t[t.len() / 2] as f64;
        }
        for t in &tmp {
            touched += t[0] as f64;
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "rust  wall {:7.1} ms  peakWS {:6.1} MB  touched={:.1}",
        ms,
        peak_ws_mb(),
        touched
    );
}
