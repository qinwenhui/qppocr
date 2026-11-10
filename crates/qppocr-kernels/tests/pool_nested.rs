//! 嵌套 fork 的回归守卫：并行区内再次并行曾是挂死（worker 争 `fork_mu`
//! / 发布方同线程重入），现在是就地串行完成。本测试在修复缺失时会
//! 无限阻塞——它存在即约束。

use qppocr_kernels::par;

/// 外层 4 个 chunk，每个 chunk 内再 fork 4 个：16 个内层单元全部执行、
/// 进程正常返回。外层由主线程与 worker 共同执行，两条嵌套路径（发布方
/// 重入 / worker 内调用）都被覆盖。
#[test]
fn nested_parallel_for_completes() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let hits = Arc::new(AtomicUsize::new(0));
    let outer_hits = Arc::new(AtomicUsize::new(0));
    let h = hits.clone();
    let oh = outer_hits.clone();
    par::parallel_for(4, 1, move |_b, _e| {
        oh.fetch_add(1, Ordering::Relaxed);
        let h2 = h.clone();
        par::parallel_for(4, 1, move |_b2, _e2| {
            h2.fetch_add(1, Ordering::Relaxed);
        });
    });
    assert_eq!(outer_hits.load(Ordering::Relaxed), 4);
    assert_eq!(hits.load(Ordering::Relaxed), 16);
}

/// 三层嵌套同样完成（降级是传递的：内层串行块里再嵌套仍是串行）。
#[test]
fn triple_nested_parallel_for_completes() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let hits = Arc::new(AtomicUsize::new(0));
    let h = hits.clone();
    par::parallel_for(2, 1, move |_, _| {
        let h2 = h.clone();
        par::parallel_for(2, 1, move |_, _| {
            let h3 = h2.clone();
            par::parallel_for(2, 1, move |_, _| {
                h3.fetch_add(1, Ordering::Relaxed);
            });
        });
    });
    assert_eq!(hits.load(Ordering::Relaxed), 8);
}
