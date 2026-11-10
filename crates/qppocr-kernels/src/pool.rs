//! fork-join 线程池。
//!
//! 为什么不用 rayon：det 图是 177 个**串行节点**，每个节点内部 fork-join
//! 一次。rayon 的任务队列（injector + work-stealing）每次 `for_each` 有
//! 固定的入队/唤醒开销，串行节点把它逐节点累加（实测多线程扩展
//! 1.86x vs 本池 2.07x）。本池与 rayon 的语义差异：
//!
//! - **一次唤醒全部 worker**（`ReleaseSemaphore(N-1)`），不是一个任务一次
//!   唤醒——这里用 `Condvar::notify_all`（std 在 Windows 上直接映射
//!   `WakeAllConditionVariable`，轻量可靠）；
//! - **主线程参与执行**（抢 chunk，干完自己的份额再自旋 join）；
//! - **join 数的是参与者而不是 chunk**：`done >= nthreads` 才返回。数
//!   chunk 的版本（不等迟醒的 worker）实测 `--bench 2` 挂死 ~20%
//!   而被回退，这里数参与者。
//!
//! 参与者计数同时买到一个不变量：**epoch 推进被 join 门控**——所有 worker
//! 都为 epoch E bump 过 done，主线程才可能发布 E+1。因此 worker 永远不会
//! 跳过一个 epoch，也永远拿不到已销毁的 job 指针（栈上 `Job` 的生存期由
//! join 保证）。 用 `shared_ptr` 兜底的是迟醒 worker，参与者计数下
//! 迟醒 worker 必须醒来才能放行主线程，指针必然仍有效。
//!
//! 嵌套 `parallel_for`（并行区的 chunk 里再次 fork）会争 `fork_mu` 死锁：
//! 外层发布方持锁到 join 完成，卡在锁上的 worker 永远凑不齐参与者计数
//! （等待 condvar 的 50ms 兜底救不了它——它等的不是条件变量）。内核层用
//! `sgemm_serial` / 串行 im2col 避免嵌套；此外 [`fork_join`] 对
//! **worker 线程内的调用**（thread-local 标记）结构性降级为就地串行——
//! 与 [`set_serial_exec`] 同款语义，切分决策不变、逐位一致。于是嵌套
//! 最坏是「多花一点调度判断」，不再是挂死。设计取舍：**跨线程并发
//! `fork_join` 串行化**（`fork_mu`）而不是挂死——公开 API 允许两个引擎
//! 在两个线程上各跑推理，「单调用方」的假设在库里不成立。
//! worker 永不触碰 `fork_mu`，无递归问题。

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// 显式线程数请求（0 = 自动）。进程级全局：池在**首次使用**时按它定容，
/// 之后 [`request_threads`] 是 no-op——对齐  `ThreadPool::get(n)` 的
/// 「最早调用者定容」语义（ 另有 `reconfigure` 在运行边界重建池；
/// 这里不做，多引擎多线程数并存本来就不是可表达的配置）。
static THREADS_REQ: AtomicUsize = AtomicUsize::new(0);

/// 请求池的线程数（含主线程）。`n = 0` 回到自动；必须在第一次并行算子
/// 之前调用才有效。显式请求**不设 16 上限**（ `resolve_threads` 的
/// 「要 24 的人在做实验，悄悄给他 16 会让测量描述一个没人选过的配置」）。
pub(crate) fn request_threads(n: usize) {
    THREADS_REQ.store(n, Ordering::Relaxed);
}

thread_local! {
    /// 本线程的**串行执行**开关：置位后 `fork_join` 不发布任务、不唤醒
    /// worker，就在调用线程里顺序跑完全部 chunk。
    ///
    /// 用途：逐行并行（每行一个线程）时，行内若仍 fork 会争 `fork_mu`
    /// 互相排队。开关只改**执行方式**，不改切分决策——`par::threads()`
    /// 仍报真实池大小，所以 gemm 的分轴、各内核的分块全部与并行路径
    /// 一致，**逐位可验证**。
    static SERIAL_EXEC: Cell<bool> = const { Cell::new(false) };
}

/// 置位/清除本线程的串行执行开关（见 `SERIAL_EXEC`）。
pub(crate) fn set_serial_exec(on: bool) {
    SERIAL_EXEC.with(|c| c.set(on));
}

/// 本线程是否处于串行执行模式。
pub(crate) fn serial_exec() -> bool {
    SERIAL_EXEC.with(|c| c.get())
}

/// 一次 fork 的协调成本（16 线程实测 ~97us）。
/// 只是文档性常量；实际门控在 [`crate::par::Thresholds`]。
#[allow(dead_code)]
pub(crate) const FORK_US: f64 = 97.0;

/// join 纯自旋的轮数上限，之后退化为让出（见 fork_join 的 join 注释）。
/// 一次 PAUSE ~1-2ns 级，2048 轮 ≈ 数微秒——覆盖无争用的常规 fork。
const JOIN_SPIN: u32 = 2048;

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

/// 单次 fork 的工作描述。**`Arc<Job>`**——worker 持克隆直到 bump+notify
/// 完毕，Job 的生存期由引用计数保证（2026-11-04 修正：spin-join 给
/// run_chunks 加了尾部 notify 后，「bump → joiner 放行 → 栈帧销毁 →
/// worker 才 notify」构成 use-after-free——notify 踩垃圾 Condvar，轻则
/// 损坏内部状态（全线程 Wait 的死锁形态），重则野指针解引用（worker
/// 线程 0xC0000005，catch_unwind 抓不住）。栈上 Job + join 计数保证不了
/// 「notify 先于帧销毁」，Arc 可以。开销：每 worker 每作业一次原子 incref。）
struct Job {
    /// 调用方闭包（`par::parallel_for` 的 chunk 划分闭包）。闭包本体仍在
    /// 发布方栈帧——其生存期仍由参与者计数 join 保证（所有 worker bump
    /// done 前，发布方不可能返回销毁闭包）。
    f: *const (dyn Fn(usize, usize) + Sync),
    /// chunk 总数（调用方已把 `[0, n)` 合成为 nchunk 个 chunk）。
    nchunk: usize,
    /// 下一个待领 chunk（原子领票，动态负载均衡）。
    next: AtomicUsize,
    /// 已完成的**参与者**数（每线程 bump 一次，不是 chunk 数）。
    done: AtomicUsize,
    /// join 阻塞等待体（发布方自旋未齐时阻塞于此，worker bump 后放行）。
    done_l: Mutex<()>,
    done_cv: Condvar,
    /// panic 隔离槽：worker/主侧 chunk panic 的原载荷（首个获胜）。
    panic: Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

// SAFETY: `f` 裸指针的跨线程解引用受发布-join 协议保护（闭包在发布方栈帧，
// join 完成前不销毁）；Arc 计数保证 Job 本体活过所有使用者的 bump+notify。
unsafe impl Send for Job {}
// SAFETY: 共享的只是不可变字段（原子计数与内部加锁的 Mutex），f 的解引用
// 论证同上。
unsafe impl Sync for Job {}

/// `Mutex<Shared>` 跨线程共享需要 `Send`。
// SAFETY: `job` 的 Arc 由发布-join 协议在锁内替换，各持有者（发布方、
// worker）各持克隆；不是类型系统能表达的只是「f 指向谁」，见 Job 注释。
unsafe impl Send for Shared {}

struct Shared {
    /// 当前 job；发布后**不置空**（下一个 epoch 覆盖）。
    job: Option<std::sync::Arc<Job>>,
    /// 发布序号；worker 用 `seen != epoch` 探测新任务。
    epoch: u64,
}

pub(crate) struct Pool {
    shared: Mutex<Shared>,
    /// 唤醒全部等待中的 worker（= `ReleaseSemaphore(N-1)` 的广播语义）。
    /// 实测对比过 Windows 信号量的逐计数唤醒（K=2 争用下反而慢 ~7%）：
    /// condvar 的广播在一次 syscall 里完成，信号量的计数累积在 fork
    /// 风暴下留下大量陈旧计数，回港后逐个烧掉。
    wake: Condvar,
    /// 串行化跨线程并发的 fork_join 调用方（见模块注释）。
    fork_mu: Mutex<()>,
    threads: usize,
}

/// 池的**布局**：每个池的线程数（含调用方那一个）。默认只有一个池，
/// 大小 = `resolve_threads()`；调 [`request_pools`] 可以改成多个。
static POOL_SIZES: Mutex<Vec<usize>> = Mutex::new(Vec::new());

static POOLS: std::sync::OnceLock<Vec<&'static Pool>> = std::sync::OnceLock::new();

thread_local! {
    /// 本线程用哪个池（下标进 [`POOLS`]）。
    ///
    /// 多池是为了支持**两级并行**：外层若干线程各自负责一行，每行的算子
    /// 再在本池内 fork。单池做不到——`fork_mu` 会把并发的 fork 串起来，
    /// 外层线程互相排队，等于白给。
    static POOL_SLOT: Cell<usize> = const { Cell::new(0) };
}

thread_local! {
    /// 本线程是否为池 worker：0 = 否，1 = 是（未告警），2 = 是（已告警）。
    /// [`fork_join`] 据此把 worker 内的调用（嵌套 fork）降级为就地串行。
    static WORKER_STATE: Cell<u8> = const { Cell::new(0) };
}

thread_local! {
    /// 本线程是否正处在一次 fork_join 的发布-执行-join 区间内。
    /// 发布方线程自己也参与 chunk 执行且持着 `fork_mu`——它内部再调
    /// [`fork_join`] 是同线程重入锁（std::Mutex 不可重入），同样降级串行。
    static FORK_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// 请求池布局（必须在**第一次 fork 之前**调用；之后请求不再生效）。
///
/// `sizes[0]` 是默认池（det 与所有非分片调用走它），其余是分片池。
/// 全部线程在首次使用时一次性创建，空闲的池只在 condvar 上等，不占 CPU。
pub(crate) fn request_pools(sizes: &[usize]) {
    if std::env::var_os("QPPOCR_POOL_DEBUG").is_some() {
        eprintln!(
            "[pool] request_pools({sizes:?}) 已建池={}",
            POOLS.get().is_some()
        );
    }
    if sizes.is_empty() {
        return;
    }
    let mut g = POOL_SIZES.lock().unwrap();
    if POOLS.get().is_none() {
        *g = sizes.to_vec();
    }
}

/// 本线程当前使用的池下标；超出范围会回落 0。
pub(crate) fn set_pool_slot(i: usize) {
    POOL_SLOT.with(|c| c.set(i));
}

fn pool() -> &'static Pool {
    let pools = POOLS.get_or_init(build_pools);
    let i = POOL_SLOT.with(|c| c.get());
    pools.get(i).unwrap_or(&pools[0])
}

fn build_pools() -> Vec<&'static Pool> {
    {
        // MXCSR 按线程继承：必须在任何 worker 诞生前开 FTZ/DAZ。
        crate::par::enable_flush_denormals();
        let sizes: Vec<usize> = {
            let g = POOL_SIZES.lock().unwrap();
            if g.is_empty() {
                vec![resolve_threads()]
            } else {
                g.clone()
            }
        };
        let mut v: Vec<&'static Pool> = Vec::with_capacity(sizes.len());
        for (i, &n) in sizes.iter().enumerate() {
            let n = n.max(1);
            // leak 而不是存 Box：池活到进程结束（worker 线程持有它的
            // 引用），`&'static` 让 `worker_loop` 不用带生命周期参数。
            let p: &'static Pool = Box::leak(Box::new(Pool {
                shared: Mutex::new(Shared {
                    job: None,
                    epoch: 0,
                }),
                wake: Condvar::new(),
                fork_mu: Mutex::new(()),
                threads: n,
            }));
            for _ in 1..n {
                std::thread::Builder::new()
                    .name(format!("qppocr-kernel{i}"))
                    .spawn(move || worker_loop(p))
                    .expect("spawn kernel worker");
            }
            v.push(p);
        }
        v
    }
}

/// **全局**线程预算（含调用线程）——永远返回 0 号池（默认池）的大小，
/// 不是本线程所在分片池的大小。首次调用会创建池。
///
/// 这一点是**逐位一致的前提**：`par::threads()` 决定 GEMM 的切分轴、
/// 各内核的分块数、各种并行门槛。分片只是把「谁来跑这些块」换了个池，
/// 不能让切分决策跟着变——否则分片前后的输出会在低位上分叉。
/// join 用的是**本地池**的参与者数，那是 `Pool::threads`，与这里无关。
pub(crate) fn thread_count() -> usize {
    let pools = POOLS.get_or_init(build_pools);
    pools[0].threads
}

/// **本线程所在池**的参与者数（含调用线程）。分片池小于默认池。
///
/// 与 [`thread_count`] 的分工：切分轴/分块数这类**进位**的决策用全局
/// （`par::threads()`），"切几个块给几个 worker"这类**只影响负载均衡**的
/// 用这里。用错会让分片池里 2 个参与者去分 128 个块——每块只有原本的
/// 1/64，调度开销盖过收益。
pub(crate) fn pool_thread_count() -> usize {
    pool().threads
}

/// 按当前配置**算**线程数，但**不建池**。给「先请求布局、再首次使用」的
/// 调用方用——用 `thread_count()` 会顺手把池建成默认布局，之后的
/// `request_pools` 就成了 no-op（这个坑真的踩过一次：分片请求永远不生效，
/// 8 个分片线程全挤在 0 号池上被 fork_mu 串行化）。
pub(crate) fn planned_threads() -> usize {
    resolve_threads()
}

fn resolve_threads() -> usize {
    // 显式请求优先：API（request_threads）→ 环境变量（=  QPPOCR_THREADS）。
    // 两者都不设 16 上限——显式数字是用户的选择（对齐  resolve_threads）。
    let req = THREADS_REQ.load(Ordering::Relaxed);
    if req > 0 {
        return req;
    }
    if let Ok(v) = std::env::var("QPPOCR_THREADS")
        && let Ok(n) = v.trim().parse::<usize>()
        && n > 0
    {
        return n;
    }
    let n = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(4);
    //  池上限 16：fork 成本随线程数增长（4 线程 21us、16 线程 97us），
    // 自动默认到此为止。
    n.clamp(1, 16)
}

fn worker_loop(p: &'static Pool) {
    WORKER_STATE.with(|c| c.set(1));
    let mut seen = 0u64;
    loop {
        // 快路径：热 worker 在锁内直接命中新 epoch，不进内核等待
        // （ Windows 路径的 peek）。Arc 克隆让 Job 活过本 worker 的
        // bump+notify——notify 不再与 joiner 的帧销毁竞速（UAF 修正）。
        let job = {
            let mut g = p.shared.lock().unwrap();
            loop {
                if g.epoch != seen {
                    seen = g.epoch;
                    break std::sync::Arc::clone(g.job.as_ref().expect("发布的 job 必在"));
                }
                // 有界等待 =  `WaitForSingleObject(wake_, 50)`：
                // 即使唤醒丢失（机制 bug 的兜底，正常不发生），
                // 50ms 后也会回来重查 epoch。
                let (g2, _timeout) = p.wake.wait_timeout(g, Duration::from_millis(50)).unwrap();
                g = g2;
            }
        };
        //
        // ★ panic 隔离（生产事故 2026-09-25：内核 panic 杀死 worker 线程，
        //   参与者屏障永远凑不齐，进程级池从此毒化——同进程重建引擎也
        //   救不回）。接住 panic：线程活着回等待循环、done 照常 bump
        //   （屏障放行）、原载荷暂存——主线程 join 后 resume_unwind 还给
        //   调用方。部分写入的输出不会被当作有效结果消费（调用方拿到
        //   panic，不是 Ok）。
        if let Err(payload) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_chunks(&job)))
        {
            {
                let mut slot = job.panic.lock().unwrap();
                if slot.is_none() {
                    *slot = Some(payload);
                }
                // 显式放锁再 bump：joiner 的 Acquire 读到本 bump 才会放行
                // 去消费 panic 槽（into_inner 不加锁）——放锁必须先行于
                // Release，否则存在「锁还活着、joiner 已进 into_inner」
                // 的窗口。
                drop(slot);
            }
            job.done.fetch_add(1, Ordering::Release);
            job.done_cv.notify_all();
        }
        // job（Arc 克隆）在此 drop——若发布方已 join 完也持有自己的克隆，
        // 最后一个 drop 负责销毁；notify 永远踩不到死帧。
    }
}

/// 领票执行 chunk，最后 bump 参与者计数并广播 join（每线程恰好一次）。
/// `notify` 的安全性由 `Arc<Job>` 保证——见 `Job` 的文档。
fn run_chunks(j: &Job) {
    loop {
        let i = j.next.fetch_add(1, Ordering::Relaxed);
        if i >= j.nchunk {
            break;
        }
        // SAFETY: f 指向发布方栈上的闭包，生存期由参与者计数 join 保证
        // （所有 worker bump done 前，发布方不可能返回销毁闭包）。
        unsafe { (*j.f)(i, i + 1) };
    }
    j.done.fetch_add(1, Ordering::Release);
    j.done_cv.notify_all(); // 无锁通知：漏唤醒由 join 的 1ms 轮询兜底
}

/// 在池上跑 `f(i, i+1)` for i in `[0, nchunk)`——`nchunk` 已经是 chunk 数
/// （`[0, n)` 的区间合成为 nchunk 个下标，见 [`crate::par::parallel_for`]）。
/// 主线程参与，共 `threads` 个参与者。
///
/// **嵌套调用（并行区内再次并行）降级为就地串行**：worker 线程内是争
/// `fork_mu`（外层发布方持锁到 join 完成），发布方线程内是同线程重入锁
/// ——两种都会挂死（见模块注释）。降级只改执行方式、不改切分决策，与
/// 并行路径逐位一致；每线程提示一次。
pub(crate) fn fork_join(nchunk: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    if nchunk == 0 {
        return;
    }
    let p = pool();
    let nested = WORKER_STATE.with(|c| c.get()) != 0
        || FORK_ACTIVE.with(|c| c.get())
        // squad worker 内不再发布到默认池：内核叶子路径本就该走 *_serial
        // 入口；万一漏网，降级串行而不是从 squad 线程里掀一轮 fork 风暴。
        || SQUAD_WORKER.with(|c| c.get()) != 0;
    if nested && p.threads > 1 && !serial_exec() {
        WORKER_STATE.with(|c| {
            if c.get() == 1 {
                c.set(2);
                eprintln!(
                    "[pool] 嵌套 fork_join：并行区内再次并行，已降级为就地串行 \
                     （内层应改用 *_serial 入口）；本次运行后续不再提示"
                );
            }
        });
    }
    if p.threads == 1 || serial_exec() || nested {
        for i in 0..nchunk {
            f(i, i + 1);
        }
        return;
    }
    FORK_ACTIVE.with(|c| c.set(true));
    let job = std::sync::Arc::new(Job {
        // SAFETY: join 协议保证闭包在所有解引用完成前存活（见上）。
        f: unsafe { erase_lifetime(f) },
        nchunk,
        next: AtomicUsize::new(0),
        done: AtomicUsize::new(0),
        done_l: Mutex::new(()),
        done_cv: Condvar::new(),
        panic: Mutex::new(None),
    });
    let _fk = p.fork_mu.lock().unwrap();
    {
        let mut g = p.shared.lock().unwrap();
        g.job = Some(std::sync::Arc::clone(&job));
        g.epoch += 1;
        // Release 语义由锁的 unlock 提供；worker 在锁内读到 epoch 与 job。
    }
    // 锁外唤醒（对齐 ：publish → ReleaseSemaphore(N-1) → 主线程干活）
    p.wake.notify_all();
    // 主线程参与（run_job）。同样包 catch_unwind：主侧 panic 若直接
    // 展开，参与屏障凑不齐。接住后屏障照常完成，再统一把首个 panic
    // 还给调用方。
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_chunks(&job))).is_err() {
        // 主线程也是参与者：panic 路径同样要 bump done + notify，否则
        // 屏障少一人、下面的自旋 join 永远等不齐（毒化测试实测挂死点）。
        let mut slot = job.panic.lock().unwrap();
        if slot.is_none() {
            *slot = Some(Box::new("panic in fork_join main chunks".to_string()));
        }
        drop(slot);
        job.done.fetch_add(1, Ordering::Release);
        job.done_cv.notify_all();
    }
    // 自旋 join：等全部参与者（含自己）各 bump 一次 done。
    // Release/Acquire 对保证：所有 worker 的 chunk 写入对返回后的主线程可见。
    //
    // 纯自旋在核被占满时（多进程并发）是病态的：迟醒的
    // worker 已 READY 却等不到核，而自旋的主线程恰好占着一个核不放。
    // 先自旋一段（无争用时零成本命中），之后阻塞 done_cv——worker 的
    // notify 踩不到死帧（Arc 保证），漏唤醒由 1ms 轮询兜底。
    let mut spin = 0u32;
    while job.done.load(Ordering::Acquire) < p.threads {
        spin += 1;
        if spin < JOIN_SPIN {
            std::hint::spin_loop();
            continue;
        }
        let g = job.done_l.lock().unwrap();
        if job.done.load(Ordering::Acquire) >= p.threads {
            break;
        }
        let (_g2, _to) = job
            .done_cv
            .wait_timeout(g, std::time::Duration::from_millis(1))
            .unwrap();
    }
    // Shared 里的 Arc 在下一次发布时被替换（此处不清：worker 持有克隆，
    // 引用计数兜底）。
    //
    // ★ 屏障完成后：如有 panic，在本线程原样重抛——调用方（executor →
    // Engine::run）看到原 panic（应用层可 catch_unwind 转提示），而池
    // 毫发无损地服务下一次调用。
    //
    // 先取载荷、释放 fork_mu，再重抛：带着 MutexGuard 展开会把锁标毒
    // （毒化测试实测：下一次 fork_join 的 lock().unwrap() 直接
    // PoisonError——这就是「panic 一次、整个进程池报废」的第二种形态）。
    let payload = job.panic.lock().unwrap().take();
    FORK_ACTIVE.with(|c| c.set(false));
    drop(_fk);
    drop(job);
    if let Some(p) = payload {
        std::panic::resume_unwind(p);
    }
}

// ============================================================================
// 常驻队伍：rec 单线全机的**静态轮转**执行后端。
//
// fork_join 是动态队列（原子领票 + 每次发布唤醒 + join 协调），对 0.5-1ms
// 级的 GEMM 切片这笔账不小——「行内并行」在 100 图交错实测里输给行间并行
// 1.06-1.43x（engine.rs 的切换注释）正是它。但引擎级对决给了第二个结论：
// medium/small 的输面随**模型体积**单调恶化（tiny
// 0.82x 赢 / small 1.50x / medium 1.9x）——分片并行让 4 条行同时流权重，
// 权重超出 L3 后是 DRAM 风暴；「单线全机静态轮转」同一时刻只流一份
// 权重，对工作集大小不敏感。
//
// 常驻队伍复刻后者的调度形态，与被证伪的「QPPOCR_REC_1X1_PAR 复刻」不是
// 同一款（那次走的是 fork_join 动态队列；conv.rs 的门注释记录在案）：
//
// - **线程常驻**：worker 空闲时 condvar 停车（50ms 兜底轮询），不占 CPU、
//   不与 det 抢核——det 用默认池，rec 相位 squad 才动；
// - **一次发布**：每次 fork 只发布一次作业；参与者按 `chunk = w, w+N,
//   w+2N…` **静态循环 stride** 消费——全程无原子领票、无 chunk 粒度锁
//   （.NET Parallel.For 静态切分同款，`for tile = worker; tile += workers`）;
// - **提交者即 0 号参与者**，自己先跑自己的轮转份额；
// - join 与 fork_join 同款：参与者计数 + 短自旋 → 阻塞 `done_cv`
//   （worker 无锁裸通知，1ms 轮询兜底——带锁通知版本死锁的教训见
//   fork_join 的 join 注释）；
// - panic 隔离/生存期擦除/参与者屏障：与 fork_join 完全同构，同一套
//   不变量论证（所有 worker bump done 前，发布方栈帧不可能返回）。
//
// 切分决策不变（`parallel_for` 的 nchunk 公式照旧），变的只是「谁跑哪块」
// ——从领票改成轮转，逐位可验证。rec 相位由 `par::set_squad_scope(true)`
// 置位（thread-local），`parallel_for` 据此分流；det/cls 与一切未置位
// 线程照走默认池。
// ============================================================================

/// 常驻队伍的实验开关与容量。`QPPOCR_REC_SQUAD=1` 开；`QPPOCR_SQUAD_THREADS`
/// 覆盖队伍人数（含提交者；0/未设 = 跟池大小）。OnceLock：首用定容。
fn squad_conf() -> (bool, usize) {
    static CONF: std::sync::OnceLock<(bool, usize)> = std::sync::OnceLock::new();
    *CONF.get_or_init(|| {
        let on = std::env::var("QPPOCR_REC_SQUAD").ok().as_deref() == Some("1");
        let n = std::env::var("QPPOCR_SQUAD_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(0);
        (on, n)
    })
}

/// 常驻队伍开关（实验旋钮，见模块注释）。
pub(crate) fn squad_enabled() -> bool {
    squad_conf().0
}

thread_local! {
    /// 本线程的 rec 相位标志：置位后 `parallel_for` 分流到 [`squad_join`]。
    /// 只影响当前线程——det/cls 相位不置位，走默认池。
    static SQUAD_SCOPE: Cell<bool> = const { Cell::new(false) };
    /// 本线程是否为 squad worker：0 = 否，1 = 是（与 `WORKER_STATE` 同款
    /// 嵌套防线——worker 内再提交 squad 是同线程重入 `squad.mu`，会挂死）。
    static SQUAD_WORKER: Cell<u8> = const { Cell::new(0) };
    /// 本线程是否处在一次 squad 发布-执行-join 区间内（提交者重入防线）。
    static SQUAD_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// 置位/清除本线程的 rec 相位标志。
pub(crate) fn set_squad_scope(on: bool) {
    SQUAD_SCOPE.with(|c| c.set(on));
}

/// 本线程的 rec 相位标志是否置位（`parallel_for` 的分流判据）。
pub(crate) fn squad_scope_active() -> bool {
    SQUAD_SCOPE.with(|c| c.get())
}

/// 一次 squad 作业。与 [`Job`] 的差异：没有 `next` 领票计数——chunk 归属
/// 由静态轮转公式决定，不需要原子协调。**`Arc` 生存期论证同 [`Job`]**
/// （bump+notify 不得与发布方帧销毁竞速——UAF 修正同源）。
struct SquadJob {
    f: *const (dyn Fn(usize, usize) + Sync),
    nchunk: usize,
    done: AtomicUsize,
    done_l: Mutex<()>,
    done_cv: Condvar,
    panic: Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

// SAFETY: 同 [`Job`]——f 由发布-join 协议保护，Arc 计数保护 Job 本体。
unsafe impl Send for SquadJob {}
// SAFETY: 共享的只是不可变字段（原子计数与内部加锁的 Mutex），f 的解引用
// 论证同上。
unsafe impl Sync for SquadJob {}

/// `Mutex<SquadShared>` 跨线程共享需要 `Send`（论证同 [`Shared`]）。
// SAFETY: `job` 的 Arc 由发布-join 协议在锁内替换，各持有者各持克隆。
unsafe impl Send for SquadShared {}

struct SquadShared {
    job: Option<std::sync::Arc<SquadJob>>,
    epoch: u64,
}

struct Squad {
    shared: Mutex<SquadShared>,
    /// 唤醒全部停车 worker（广播语义，同 [`Pool::wake`]）。
    wake: Condvar,
    /// 串行化跨线程并发的 squad 提交方（`fork_mu` 同款）。
    mu: Mutex<()>,
    threads: usize,
}

static SQUAD: std::sync::OnceLock<&'static Squad> = std::sync::OnceLock::new();

fn squad() -> &'static Squad {
    SQUAD.get_or_init(build_squad)
}

fn build_squad() -> &'static Squad {
    // MXCSR 按线程继承：先于任何 squad worker 诞生开 FTZ/DAZ（与
    // build_pools 同款双保险——主线程在引擎构造时已设过一次）。
    crate::par::enable_flush_denormals();
    let n = squad_conf().1;
    let n = if n > 0 { n } else { resolve_threads() }.max(1);
    let s: &'static Squad = Box::leak(Box::new(Squad {
        shared: Mutex::new(SquadShared {
            job: None,
            epoch: 0,
        }),
        wake: Condvar::new(),
        mu: Mutex::new(()),
        threads: n,
    }));
    for w in 1..n {
        std::thread::Builder::new()
            .name(format!("qppocr-squad{w}"))
            .spawn(move || squad_worker_loop(s, w))
            .expect("spawn squad worker");
    }
    s
}

fn squad_debug() -> bool {
    static DBG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DBG.get_or_init(|| std::env::var_os("QPPOCR_SQUAD_DEBUG").is_some())
}

fn squad_worker_loop(s: &'static Squad, w: usize) {
    SQUAD_WORKER.with(|c| c.set(1));
    let dbg = squad_debug();
    let mut seen = 0u64;
    loop {
        // Arc 克隆让 SquadJob 活过本 worker 的 bump+notify（UAF 修正同源）。
        let job = {
            let mut g = s.shared.lock().unwrap();
            loop {
                if g.epoch != seen {
                    seen = g.epoch;
                    break std::sync::Arc::clone(g.job.as_ref().expect("发布的 job 必在"));
                }
                let (g2, _timeout) = s.wake.wait_timeout(g, Duration::from_millis(50)).unwrap();
                g = g2;
            }
        };
        if dbg {
            eprintln!("[squad] w{w} take epoch={seen}");
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_stride(&job, s.threads, w)
        }));
        if dbg {
            eprintln!("[squad] w{w} bump epoch={seen} err={}", r.is_err());
        }
        if let Err(payload) = r {
            {
                let mut slot = job.panic.lock().unwrap();
                if slot.is_none() {
                    *slot = Some(payload);
                }
                // 显式放锁再 bump（理由同 fork_join 的 worker 侧）。
                drop(slot);
            }
            job.done.fetch_add(1, Ordering::Release);
            job.done_cv.notify_all();
        }
        // job（Arc 克隆）在此 drop——最后一个 drop 负责销毁。
    }
}

/// 静态轮转：参与者 `w` 领 chunk `{w, w+N, w+2N…}`，最后 bump 参与者计数
/// 并广播 join（无锁通知；漏唤醒由 join 的 1ms 轮询兜底）。
///
/// # Safety（文档性）
///
/// `j.f` 指向发布方栈上的闭包，生存期由参与者计数 join 保证。
fn run_stride(j: &SquadJob, n: usize, w: usize) {
    let mut i = w;
    while i < j.nchunk {
        // SAFETY: f 指向发布方栈上的闭包，生存期同上由 join 保证。
        unsafe { (*j.f)(i, i + 1) };
        i += n;
    }
    j.done.fetch_add(1, Ordering::Release);
    j.done_cv.notify_all();
}

/// 在常驻队伍上静态轮转跑 `f(i, i+1)` for i in `[0, nchunk)`。提交者自己
/// 是 0 号参与者（先跑自己的份额再 join）。
///
/// 嵌套防线与 [`fork_join`] 同款：squad worker 内、提交区间内的再提交都
/// 降级为就地串行（提交者重入是同线程重入 `squad.mu`；worker 重入是等
/// 自己 bump done——两种都挂死）。降级只改执行方式，不改切分决策。
pub(crate) fn squad_join(nchunk: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    if nchunk == 0 {
        return;
    }
    let s = squad();
    let nested = SQUAD_WORKER.with(|c| c.get()) != 0 || SQUAD_ACTIVE.with(|c| c.get());
    if s.threads == 1 || serial_exec() || nested {
        for i in 0..nchunk {
            f(i, i + 1);
        }
        return;
    }
    SQUAD_ACTIVE.with(|c| c.set(true));
    let job = std::sync::Arc::new(SquadJob {
        // SAFETY: join 协议保证闭包在所有解引用完成前存活（同 fork_join）。
        f: unsafe { erase_lifetime(f) },
        nchunk,
        done: AtomicUsize::new(0),
        done_l: Mutex::new(()),
        done_cv: Condvar::new(),
        panic: Mutex::new(None),
    });
    let _sm = s.mu.lock().unwrap();
    let ep = {
        let mut g = s.shared.lock().unwrap();
        g.job = Some(std::sync::Arc::clone(&job));
        g.epoch += 1;
        g.epoch
    };
    if squad_debug() {
        eprintln!("[squad] publish epoch={ep} nchunk={nchunk}");
    }
    s.wake.notify_all();
    // 提交者 = 0 号参与者。catch_unwind：主侧 panic 若直接展开，参与屏障
    // 凑不齐（fork_join 同款处理）。
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_stride(&job, s.threads, 0)
    }))
    .is_err()
    {
        let mut slot = job.panic.lock().unwrap();
        if slot.is_none() {
            *slot = Some(Box::new("panic in squad main stride".to_string()));
        }
        drop(slot);
        job.done.fetch_add(1, Ordering::Release);
        job.done_cv.notify_all();
    }
    // join：短自旋 → 阻塞 done_cv（fork_join 同款；worker 无锁通知）。
    let mut spin = 0u32;
    while job.done.load(Ordering::Acquire) < s.threads {
        spin += 1;
        if spin < JOIN_SPIN {
            std::hint::spin_loop();
            continue;
        }
        let g = job.done_l.lock().unwrap();
        if job.done.load(Ordering::Acquire) >= s.threads {
            break;
        }
        let (_g2, _to) = job
            .done_cv
            .wait_timeout(g, std::time::Duration::from_millis(1))
            .unwrap();
    }
    // 先取载荷、释放 mu，再重抛（带 MutexGuard 展开会标毒锁，fork_join
    // 的毒化教训）。
    if squad_debug() {
        eprintln!(
            "[squad] join epoch={ep} done={}",
            job.done.load(Ordering::Acquire)
        );
    }
    let payload = job.panic.lock().unwrap().take();
    SQUAD_ACTIVE.with(|c| c.set(false));
    drop(_sm);
    drop(job);
    if let Some(p) = payload {
        std::panic::resume_unwind(p);
    }
}
