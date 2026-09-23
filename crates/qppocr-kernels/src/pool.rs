//! fork-join 线程池：`util.hpp` `ThreadPool`（Windows 信号量路径）的移植。
//!
//! 为什么不用 rayon：det 图是 177 个**串行节点**，每个节点内部 fork-join
//! 一次。rayon 的任务队列（injector + work-stealing）每次 `for_each` 有
//! 固定的入队/唤醒开销，串行节点把它逐节点累加（实测多线程扩展
//! 1.86x vs C++ 池 2.07x）。C++ 的池语义不同：
//!
//! - **一次唤醒全部 worker**（`ReleaseSemaphore(N-1)`），不是一个任务一次
//!   唤醒——这里对应 `Condvar::notify_all`（std 在 Windows 上直接映射
//!   `WakeAllConditionVariable`，没有 libstdc++ condvar 队列的老毛病，
//!   所以 C++ 弃用 condvar 的理由在这里不成立）；
//! - **主线程参与执行**（抢 chunk，干完自己的份额再自旋 join）；
//! - **join 数的是参与者而不是 chunk**：`done >= nthreads` 才返回。数
//!   chunk 的版本（不等迟醒的 worker）在 C++ 实测 `--bench 2` 挂死 ~20%
//!   而被回退（util.hpp `parallel_for` 注释），这里照搬参与者计数。
//!
//! 参与者计数同时买到一个不变量：**epoch 推进被 join 门控**——所有 worker
//! 都为 epoch E bump 过 done，主线程才可能发布 E+1。因此 worker 永远不会
//! 跳过一个 epoch，也永远拿不到已销毁的 job 指针（栈上 `Job` 的生存期由
//! join 保证）。C++ 用 `shared_ptr` 兜底的是迟醒 worker，参与者计数下
//! 迟醒 worker 必须醒来才能放行主线程，指针必然仍有效。
//!
//! 嵌套 `parallel_for` 是死锁（C++ 同款限制）：内核用 `sgemm_serial` /
//! 串行 im2col 避免嵌套。与 C++ 不同的一点：**跨线程并发 `fork_join`
//! 在这里串行化**（`fork_mu`），而不是像 C++ 那样挂死——公开 API 允许
//! 两个引擎在两个线程上各跑推理，C++ 的单调用方假设在库里不成立。
//! worker 永不触碰 `fork_mu`，无递归问题。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// 一次 fork 的协调成本对齐（16 线程实测 ~97us，`tools/fork_bench.cpp`）。
/// 只是文档性常量；实际门控在 [`crate::par::Thresholds`]。
#[allow(dead_code)]
pub(crate) const FORK_US: f64 = 97.0;

/// 把闭包指针的生存期擦成 `'static`（`Job` 存不了借用生存期——它在
/// `static POOL` 的 `Shared` 里）。胖指针布局不变。
///
/// # Safety
///
/// 调用方必须保证指针的所有解引用都发生在被指对象销毁之前——本池的
/// 参与者计数 join 正是这条保证：所有 worker bump done 前，发布方的
/// 栈帧（闭包所在）不可能返回。
unsafe fn erase_lifetime(
    f: *const (dyn Fn(usize, usize) + Sync + '_),
) -> *const (dyn Fn(usize, usize) + Sync + 'static) {
    // SAFETY: 胖指针 (data, vtable) 到同布局胖指针，只改类型。
    unsafe { std::mem::transmute(f) }
}

/// 单次 fork 的工作描述。主线程的栈对象，生存期由参与者计数 join 保证。
struct Job {
    /// 调用方闭包（`par::parallel_for` 的 chunk 划分闭包）。
    f: *const (dyn Fn(usize, usize) + Sync),
    /// chunk 总数（调用方已把 `[0, n)` 合成为 nchunk 个 chunk）。
    nchunk: usize,
    /// 下一个待领 chunk（原子领票，动态负载均衡）。
    next: AtomicUsize,
    /// 已完成的**参与者**数（每线程 bump 一次，不是 chunk 数）。
    done: AtomicUsize,
}

/// `Mutex<Shared>` 跨线程共享需要 `Send`。
// SAFETY: `job` 裸指针的跨线程安全来自 fork_join 的发布-join 协议
// （发布与读取都在锁内；栈上 Job 在 join 完成前不会销毁），不是类型系统能
// 表达的。见模块注释的不变量论证。
unsafe impl Send for Shared {}

struct Shared {
    /// 当前 job；发布后**不置空**（下一个 epoch 覆盖，C++ 同款）。
    job: *const Job,
    /// 发布序号；worker 用 `seen != epoch` 探测新任务。
    epoch: u64,
}

pub(crate) struct Pool {
    shared: Mutex<Shared>,
    /// 唤醒全部等待中的 worker（= `ReleaseSemaphore(N-1)`）。
    wake: Condvar,
    /// 串行化跨线程并发的 fork_join 调用方（见模块注释）。
    fork_mu: Mutex<()>,
    threads: usize,
}

static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();

/// 取全局池（首个调用创建并固定线程数，对齐 C++ `ThreadPool::get`）。
fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        // MXCSR 按线程继承：必须在任何 worker 诞生前开 FTZ/DAZ（C++ 在
        // ThreadPool 构造函数里做同样的事）。
        crate::par::enable_flush_denormals();
        let n = resolve_threads();
        let p = Pool {
            shared: Mutex::new(Shared {
                job: std::ptr::null(),
                epoch: 0,
            }),
            wake: Condvar::new(),
            fork_mu: Mutex::new(()),
            threads: n,
        };
        for _ in 1..n {
            std::thread::Builder::new()
                .name("qppocr-kernel".into())
                .spawn(worker_loop)
                .expect("spawn kernel worker");
        }
        p
    })
}

/// 池的线程数（含主线程）。首次调用会创建池。
pub(crate) fn thread_count() -> usize {
    pool().threads
}

fn resolve_threads() -> usize {
    let n = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(4);
    // C++ 池上限 16：fork 成本随线程数增长（4 线程 21us、16 线程 97us），
    // 自动默认到此为止。
    n.clamp(1, 16)
}

fn worker_loop() {
    let p = pool();
    let mut seen = 0u64;
    loop {
        // 快路径：热 worker 在锁内直接命中新 epoch，不进内核等待
        // （C++ Windows 路径的 peek）。
        let job = {
            let mut g = p.shared.lock().unwrap();
            loop {
                if g.epoch != seen {
                    seen = g.epoch;
                    break g.job;
                }
                // 有界等待 = C++ `WaitForSingleObject(wake_, 50)`：
                // 即使唤醒丢失（机制 bug 的兜底，正常不发生），
                // 50ms 后也会回来重查 epoch。
                let (g2, _timeout) = p.wake.wait_timeout(g, Duration::from_millis(50)).unwrap();
                g = g2;
            }
        };
        // SAFETY: epoch!=seen 时 job 指针必为当前 epoch 的发布值；栈上
        // Job 的生存期由参与者 join 保证——本 worker bump done 之前发布方
        // 不可能返回/销毁它。
        unsafe { run_chunks(&*job) };
    }
}

/// 领票执行 chunk，最后 bump 参与者计数（每线程恰好一次）。
///
/// # Safety
///
/// `j` 必须指向有效且不再变更的 `Job`（发布协议保证）。
unsafe fn run_chunks(j: &Job) {
    loop {
        let i = j.next.fetch_add(1, Ordering::Relaxed);
        if i >= j.nchunk {
            break;
        }
        // SAFETY: f 指向发布方栈上的闭包，生存期同上由 join 保证。
        unsafe { (*j.f)(i, i + 1) };
    }
    j.done.fetch_add(1, Ordering::Release);
}

/// 在池上跑 `f(i, i+1)` for i in `[0, nchunk)`——`nchunk` 已经是 chunk 数
/// （`[0, n)` 的区间合成为 nchunk 个下标，见 [`crate::par::parallel_for`]）。
/// 主线程参与，共 `threads` 个参与者。**禁止嵌套**（死锁）。
pub(crate) fn fork_join(nchunk: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    if nchunk == 0 {
        return;
    }
    let p = pool();
    if p.threads == 1 {
        for i in 0..nchunk {
            f(i, i + 1);
        }
        return;
    }
    let job = Job {
        // SAFETY: join 协议保证闭包在所有解引用完成前存活（见上）。
        f: unsafe { erase_lifetime(f) },
        nchunk,
        next: AtomicUsize::new(0),
        done: AtomicUsize::new(0),
    };
    let _fk = p.fork_mu.lock().unwrap();
    {
        let mut g = p.shared.lock().unwrap();
        g.job = &job;
        g.epoch += 1;
        // Release 语义由锁的 unlock 提供；worker 在锁内读到 epoch 与 job。
    }
    // 锁外唤醒（对齐 C++：publish → ReleaseSemaphore → 主线程干活）
    p.wake.notify_all();
    // 主线程参与（C++ run_job）
    // SAFETY: job 在本栈上，刚初始化。
    unsafe { run_chunks(&job) };
    // 自旋 join：等全部参与者（含自己）各 bump 一次 done。
    // Release/Acquire 对保证：所有 worker 的 chunk 写入对返回后的主线程可见。
    while job.done.load(Ordering::Acquire) < p.threads {
        std::hint::spin_loop();
    }
    // 不清 job 指针：下一次发布覆盖。`_fk` 在此释放，放行下一个调用方。
}
