//! panic 安全回归：多块并行的拷贝越界（2026-09-25 生产事故）与池毒化。

use qppocr_kernels::par;
use qppocr_kernels::shape::{Payload, PayloadRef, transpose_tensor};

/// ★ 回归：3918×2772 照片 + medium 档触发的 copy_strided dst 越界。
/// 根因：调用方传**相对切片**（从 outer_begin 起），copy_strided 曾用
/// 绝对 o 索引 dst——单块路径（outer_begin=0）恰好掩住错位，大张量并行
/// 多块后 ob>0 即 panic。这里构造大到必然多块（>1 MB 阈值）的转置，
/// 并逐元素对拍朴素参考。
#[test]
fn transpose_multichunk_matches_naive() {
    // 128×96×24 = 294912 元素 > 并行字节阈（1 MB / 4 B = 262144）→ 必多块
    let (n, c, h, w) = (1usize, 128usize, 96usize, 24usize);
    let total = n * c * h * w;
    let x: Vec<f32> = (0..total).map(|i| i as f32 * 0.5 - 1000.0).collect();
    let shape = vec![n as i64, c as i64, h as i64, w as i64];
    let perm = vec![0i64, 2, 3, 1]; // [n,c,h,w] -> [n,h,w,c]

    let (out, out_shape) = transpose_tensor(PayloadRef::F32(&x), &shape, &perm);
    let Payload::F32(y) = out else {
        panic!("dtype")
    };
    assert_eq!(out_shape, vec![n as i64, h as i64, w as i64, c as i64]);

    // 朴素参考：out[n0][h0][w0][c0] = x[n0][c0][h0][w0]
    let nc = (h * w * c) as i64;
    let nh = (w * c) as i64;
    let nw = c as i64;
    for i in 0..total as i64 {
        let (a, b, cc) = (i / nc, (i % nc) / nh, (i % nh) / nw);
        let d = i % nw;
        let src_idx =
            a * (c as i64 * h as i64 * w as i64) + d * (h as i64 * w as i64) + b * w as i64 + cc;
        assert_eq!(y[i as usize], x[src_idx as usize], "idx {i}");
    }
}

/// ★ 回归：池毒化。内核 panic 曾经杀死 worker 线程、参与者屏障永远
/// 凑不齐（同进程重建引擎也救不回）。期望：panic 原样传给调用方
/// （catch_unwind 能接住），且**池继续可用**。
#[test]
fn pool_survives_panic_and_still_works() {
    let n = 64;
    // 让某个必然落在 worker 上的 chunk panic（主线程通常抢 chunk 0）
    let r = std::panic::catch_unwind(|| {
        par::parallel_for(n, 1, |b, _e| {
            if b >= 1 {
                panic!("boom from chunk {b}");
            }
        });
    });
    assert!(r.is_err(), "panic 必须传回调用方");

    // 池必须毫发无损：紧接着的正常并行要能完成且结果正确。按**单元**
    // 计数（不是块调用数——parallel_for 会把 n 个单元合并成
    // min(线程数×8, n) 个块，块数随机器核数变：本机 16 核恰好全 64，
    // CI 的 4 核 runner 是 32 块，曾经的按块断言在 CI 必挂）
    use std::sync::atomic::{AtomicUsize, Ordering};
    let hit = AtomicUsize::new(0);
    par::parallel_for(n, 1, |b, e| {
        for _i in b..e {
            hit.fetch_add(1, Ordering::Relaxed);
        }
    });
    assert_eq!(hit.load(Ordering::Relaxed), n, "panic 后池必须继续正确工作");
}
