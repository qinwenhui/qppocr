//! 性能微基准：把「一次 run 的墙钟」拆成三个可独立测的固定开销。
//!
//! 动机：`QPPOCR_GPU_PROF` 显示 rec 每个宽度批次的墙钟在 **4.2~17 ms**
//! 之间，而 [8,3,48,192] 这种小批的纯算力需求只有零点几毫秒。差值说明
//! 墙钟被**固定开销**支配。本文件的测试就是逐个量出这些固定开销，让
//! 「优化哪里」由数字决定而不是猜。
//!
//! 三个待测量：
//! 1. **命令缓冲往返**（空 CB 的 commit + wait）——每 run 一次；
//! 2. **Shared 缓冲分配**（按尺寸）——每建一个新形状计划一次；
//! 3. **单 dispatch 编码成本**（同一 CB 内 N 条 dispatch 的边际成本）
//!    ——决定「算子数 × 每条成本」是不是墙钟主项。
//!
//! 全部只在有 Metal 设备的机器上跑；无设备自动跳过（CI 非 mac）。
//! 数字用 eprintln 打印，配 `--nocapture` 读。

use std::time::Instant;

use metal::MTLSize;

use super::device::submit_compute;
use super::pipeline::{dispatch_with_tg, KernelSet};
use super::MetalContext;
use super::conv_probe::{mm_sg_groups, MM_PROBE_SHADERS, MM_TG_PROBE_SHADERS, ProbeSet};

/// 有 Metal 设备就打开，否则跳过（CI 非 mac）。与 `kernel_tests` 同款。
fn open_or_skip() -> Option<MetalContext> {
    match MetalContext::open(None) {
        Ok(c) => {
            eprintln!("[mb] 设备: {}", c.inner.name);
            Some(c)
        }
        Err(_) => None,
    }
}

/// 迭代取中位：首轮含冷态（首次分配、管线预热），丢弃。
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// 分配一块共享缓冲并清零。**申请不到存储时 `contents()` 是空指针**，直接写就是
/// SIGSEGV，所以这里判空并返回 `None`，让调用方跳过那一路而不是把测试进程带走。
/// （本文件曾因此在 `dispatch_floor_attribution` 里段错误，见那里的长注释。）
fn shared_zeroed(dev: &metal::Device, bytes: usize) -> Option<metal::Buffer> {
    let b = dev.new_buffer(
        bytes as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let p = b.contents() as *mut u8;
    if p.is_null() {
        return None;
    }
    // SAFETY: 缓冲 `bytes` 长、写 `bytes` 字节在界内；Shared 存储 CPU 侧可写。
    unsafe { std::ptr::write_bytes(p, 0, bytes) };
    Some(b)
}

fn bench<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    // warmup
    for _ in 0..3 {
        f();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    median(samples)
}

#[test]
fn cmd_buffer_roundtrip() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ms = bench(50, || {
        submit_compute(&ctx.inner.queue, |_enc| {}).expect("空提交");
    });
    eprintln!("[mb] 空命令缓冲 commit+wait 往返: {ms:.3} ms");
}

#[test]
fn buffer_alloc_cost() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    for mb in [1u64, 8, 32, 64, 128, 320] {
        let bytes = mb << 20;
        let ms = bench(15, || {
            let b = ctx
                .inner
                .device
                .new_buffer(bytes, metal::MTLResourceOptions::StorageModeShared);
            // 触碰首页，逼出真实页映射（只分配不触碰会高估惰性分配的好处）
            if b.contents() as *mut u8 as usize != 0 {
                // SAFETY: 刚分配的 `bytes` 字节 Shared 缓冲，基址非空；只写
                // 首、尾各 1 字节，偏移 add(bytes-1) 落在最后一字节内。
                unsafe {
                    std::ptr::write_volatile(b.contents() as *mut u8, 1u8);
                    std::ptr::write_volatile(
                        (b.contents() as *mut u8).add(bytes as usize - 1),
                        1u8,
                    );
                }
            }
            std::hint::black_box(&b);
        });
        eprintln!("[mb] Shared 缓冲分配 {mb:>4} MB: {ms:.3} ms");
    }
}

#[test]
fn dispatch_marginal_cost() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");
    // **必须显式清零**，不能指望新缓冲「恰好是干净的」：`n_entry` 会从参数区读
    // `nb/h/w` 并据此决定内层循环，读到垃圾尺寸就可能变成巨循环 → 把 GPU 挂死
    // （正是 `plan.rs` 记的那条故障链）。`new_buffer` 不清零，所以这里自己来；
    // 拿不到可寻址存储就直接跳过（见 `shared_zeroed` 的注释）。
    let Some(buf) = shared_zeroed(&ctx.inner.device, 1 << 20) else {
        eprintln!("[mb] 1 MB 缓冲分配失败（Metal 未给出可寻址存储）——跳过本探针");
        return;
    };

    for n in [1usize, 16, 64, 128, 256] {
        let ms = bench(20, || {
            submit_compute(&ctx.inner.queue, |enc| {
                for _ in 0..n {
                    ks.dispatch(enc, "n_entry", &buf, 0, [1, 1, 1]).expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [&buf];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
        });
        eprintln!(
            "[mb] {n:>4} 条 dispatch（含逐条 barrier）: {ms:.3} ms  \
             （边际 {:.3} ms/条）",
            ms / n as f64
        );
    }

    // 对照：不带 barrier 的纯编码+执行成本
    for n in [64usize, 256] {
        let ms = bench(20, || {
            submit_compute(&ctx.inner.queue, |enc| {
                for _ in 0..n {
                    ks.dispatch(enc, "n_entry", &buf, 0, [1, 1, 1]).expect("dispatch");
                }
            })
            .expect("提交");
        });
        eprintln!("[mb] {n:>4} 条 dispatch（无 barrier）: {ms:.3} ms（边际 {:.3} ms/条）", ms / n as f64);
    }
}

/// 真实负载的算子规模：把 rec 的 104 条 dispatch 按计划里的网格重放
/// 一遍，量「真算力」与「固定开销」的占比。用计划构建器拿真实 dispatch
/// 列表，但只执行（不建会话、不走引擎）。
#[test]
fn real_plan_replay_cost() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");
    // 用一个中等尺寸的空 buffer 近似 arena：这里只量 dispatch 机制的
    // 成本，kernel 里的越界守卫会让它立即返回（算力≈0），因此测到的
    // 数字是**开销下界**——真实推理时间必然不小于它。
    let buf = ctx
        .inner
        .device
        .new_buffer(64 << 20, metal::MTLResourceOptions::StorageModeShared);
    // 把 params[0] 之外全零：内核读到的尺寸多为 0 → 早退。
    // SAFETY: 缓冲是 64 MB 的 Shared 存储、基址非空；清零范围 64 KiB
    // （1 << 16 字节）远小于缓冲长度，不越界。
    unsafe {
        std::ptr::write_bytes(buf.contents() as *mut u8, 0, 1 << 16);
    }

    for n in [104usize, 173] {
        let names: Vec<&str> = (0..n)
            .map(|i| {
                const K: [&str; 6] = [
                    "n_conv", "n_elem", "n_conv_dw", "n_pool", "n_channel", "n_resize",
                ];
                K[i % K.len()]
            })
            .collect();
        let ms = bench(20, || {
            submit_compute(&ctx.inner.queue, |enc| {
                for nm in &names {
                    ks.dispatch(enc, nm, &buf, 0, [4, 2, 1]).expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [&buf];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
        });
        eprintln!("[mb] 仿 {n} 条真实名 dispatch（守卫早退，纯开销）: {ms:.3} ms");
    }
}

/// 每 dispatch 固定开销的**归因**——当前第一优先级。
///
/// `dispatch_marginal_cost` 把地板量到 ~1 ms/条（内核早退、零算力），而
/// 空命令缓冲往返只有 **0.016 ms**：所以这 1 ms 是按 dispatch 计、落在
/// **设备侧**的，与内核真正算什么无关。它是 rec（173 条/forward，每图
/// 11 个 forward）与 det（154 条）共同的乘数，必须钉死由什么决定。
///
/// 每格只改一个变量：
/// - **网格大小** 1 / 64 / 1024 / 16384 个线程组（生产 det 最大 ~3600 组）
/// - **绑定个数** 1 个 vs 3 个（生产里同一 arena 绑 0/1/2 三个别名视图）
/// - **arena 尺寸** 1 MB vs 256 MB（生产的整块 222 MB）
/// - **存储模式** Shared vs Private（Shared 的脏数据可能要回写到一致性点）
///
/// 零算力内核（`n_entry` + **全零 params** → `gid >= nb*hw` 立即早退）
/// 保证量到的是地板本身。params 必须全零，否则 `nb/h/w` 是垃圾值。
#[test]
fn dispatch_floor_attribution() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");
    let dev = &ctx.inner.device;
    let queue = &ctx.inner.queue;
    let pipe = ks.get("n_entry").expect("n_entry");
    const COUNT: usize = 16;

    // 一块零页：共享/私有两条路都拿它填，保证 params 全零 → 内核零算力。
    let zeros = vec![0u8; 1 << 18];

    // **arena 的三个别名视图（0/1/2）必须全部绑定，一个都不能省。**
    //
    // `n_entry` 的签名是 0=data、1=params、2=d，它读参数走的是 **buffer 1**。
    // 曾经这里带过一个「少绑几个是不是更便宜」的对照（只绑 buffer 0，
    // 注释还写着「内核只用 buffer 0/1」），那是错的：Metal 里**未绑定的
    // buffer 解析到地址 0**，内核读 `params[P+0]` 就是从地址 0 取数。
    //
    // 后果不是「慢一点」，而是整块 GPU 被打瘸：
    //   /Library/Logs/DiagnosticReports/gpuEvent-qppocr_metal-2ed-*.ips
    //   → "restart_reason_desc":"BIF0 page fault"
    //     {"bif0_fault":{"is_read":true,"address":0,...},"guilty_dm":3}
    //   两次（09:53:28 / 09:53:30）把 IOAccelerator 的 recoveryCount 推到 2。
    // 故障之后**所有**走 GPU 的访存都掉到 ~1 GB/s——连苹果自己的
    // `MTLBlitCommandEncoder` 都只剩 1.3 GB/s（CPU 同尺寸 memcpy 136 GB/s），
    // 而 GPU 仍报 1470 MHz 满频、100% 驻留、6.2 W。这个状态直到重启才恢复，
    // 期间任何性能结论都是垃圾。**排查 GPU 性能前先确认 recoveryCount==0。**
    let run = |arena: &metal::BufferRef, tiles: u64, label: &str| {
        let ms = bench(15, || {
            submit_compute(queue, |enc| {
                for _ in 0..COUNT {
                    enc.set_compute_pipeline_state(pipe);
                    for b in 0..3u64 {
                        enc.set_buffer(b, Some(arena), 0);
                    }
                    let off = 0u32;
                    // SAFETY: `off` 是本作用域内的 u32，4 字节读在界内；
                    // set_bytes 在本调用期间同步拷贝。
                    enc.set_bytes(3, 4, &off as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(
                        MTLSize::new(tiles * 256, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                }
            })
            .expect("提交");
        });
        eprintln!("[mb] {label}: {:.3} ms/条", ms / COUNT as f64);
    };

    // 共享 256 KB（`dispatch_marginal_cost` 的同一构型，做基准）
    //
    // **`new_buffer` 不清零**（`dispatch_marginal_cost` 靠 1 MB 里恰好
    // 全零才没踩坑）。不显式清零的话 `nb/h/w` 是垃圾值：`gid >= nb*hw`
    // 守卫失效，内核就真的按垃圾尺寸搬数据——结论会变成「每组 1.8 µs
    // 的启动成本」，而实际是「组数越多、干的活越多」。
    let Some(shared1) = shared_zeroed(dev, zeros.len()) else {
        eprintln!("[mb] 256 KB 共享缓冲分配失败（Metal 未给出可寻址存储）——跳过本探针");
        return;
    };
    run(&shared1, 1, "网格 1 组");

    // 网格大小：真实生产里最大的组数（n_concat_c 是 7200）
    //
    // 这一组才是这条探针的本意——每组摊到的开销随网格怎么变。原先还有一条
    // 「只绑 buffer 0」的对照，正是它把 GPU 打瘸的（见 `run` 上方注释），已删：
    // 内核要读 buffer 1，少绑一个量到的不是「绑定更便宜」而是「GPU 故障」。
    for tiles in [64u64, 1024, 7200] {
        run(&shared1, tiles, &format!("网格 {tiles} 组"));
    }

    // arena 尺寸：生产整块 222 MB
    //
    // 只清零**头 64 KB**（内核守卫读的参数区）：整块 256 MB 用 CPU 清零会把数据
    // 刷进缓存，热度与真实场景不同，会把这次测量污染掉。其余部分读到什么都没关系。
    let shared256 = dev.new_buffer(
        256 << 20,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let p256 = shared256.contents() as *mut u8;
    if p256.is_null() {
        eprintln!("[mb] 256 MB 共享缓冲分配失败——跳过（生产整块 222 MB，值得警惕）");
    } else {
        // SAFETY: 缓冲 256 MB，写头 64 KB 在界内。
        unsafe { std::ptr::write_bytes(p256, 0, 1 << 16) };
        run(&shared256, 1, "shared 256 MB 网格 1 组");
    }

    // 存储模式：同尺寸成对——Shared 的脏数据可能要回写到一致性点，Private 不用。
    //
    // ⚠️ **这台 M4 上 `Private` 缓冲的 `contents()` 是空指针**（实测：`new_buffer`
    // 配 `StorageModePrivate` 返回的 `contents()` 判空即中）。所以：
    // ① 绝不能对 Private 缓冲解引用 `contents()`——包括**隐式**的那种：
    //    `new_buffer_with_data(..., Private)` 内部就是一次 memmove 到空指针，
    //    会把整个测试进程 SIGSEGV 带走（崩溃报告：`_platform_memmove`
    //    ← `-[AGXG16GFamilyDevice newBufferWithBytes:length:options:]`
    //    ← `microbench::dispatch_floor_attribution`，KERN_INVALID_ADDRESS at 0x0）。
    //    这条**与下面 `buffer_creation_path_bandwidth` 文档里记的是同一件事**，
    //    那里已把「Private + 拷入」这一路去掉——本处原先漏改，现已一并对齐。
    // ② `bench/gpuhealth/*.m` 的探针用 Private 是安全的，因为它们**只 blit**
    //    （`fillBuffer` / `MTLBlitCommandEncoder`），从不碰 `contents()`。
    // ③ 生产路径（arena / 计划缓冲）全部走 `StorageModeShared`，不受影响。
    let priv1 = dev.new_buffer(
        zeros.len() as u64,
        metal::MTLResourceOptions::StorageModePrivate,
    );
    let p = priv1.contents() as *mut u8;
    if p.is_null() {
        eprintln!("[mb] priv   256 KB：Private 无 CPU 可寻址存储——跳过这一路（见上方注释）");
    } else {
        // SAFETY: 缓冲 256 KB，写 256 KB 在界内。
        unsafe { std::ptr::write_bytes(p, 0, zeros.len()) };
        run(&priv1, 1, "priv   256 KB 网格 1 组");
    }
}

/// 64×64 瓦片的 simdgroup_matrix GEMM 探针 —— 竞品结构的理想布局，测 fp16 MMA 能跑多快。
///
/// 竞品 `mm_sg_dd` 的核心：A 是 [M,K] 行主序、W 是 [N,K] n-主序、K 步 8、
/// 2×4 acc 分片分布在 8 个 simdgroup 上，64×64 瓦片摊给 1 个线程组。
/// 量的是**这个结构能达到的上限**：如果它在我们的生产 1×1 形状上能跑
/// 2.5 TFLOPS 以上，那值得重写取数路径；否则不用在布局上浪费时间。
#[test]
fn mm_sg_scan() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ps = ProbeSet::from_list(&ctx.inner.device, MM_PROBE_SHADERS).expect("编译 MM 探针");

    // 生产里的 1×1 卷积层形状（det/rec 各几个关键 C→O）。
    let cases = [
        // det medium: 960x960, c512→1024 k1 → M=960*960=921600, N=1024, K=512
        ("det960_c512o1024", 960u32 * 960, 1024u32, 512u32),
        // det medium: c1024→512 k1
        ("det960_c1024o512", 960u32 * 960, 512u32, 1024u32),
        // rec: 6x208, c192→384 k1 → M=6*208=1248, N=384, K=192
        ("rec6x208_c192o384", 6u32 * 208, 384u32, 192u32),
        // rec: 3x208, c384→768 k1 → M=3*208=624, N=768, K=384
        ("rec3x208_c384o768", 3u32 * 208, 768u32, 384u32),
        // rec CTC head: 104x1, c120→18710 k1 → M=104, N=18710, K=120
        ("ctc104_c120o18710", 104u32, 18710u32, 120u32),
    ];

    // 分配 fp16 缓冲：A [M,K], W [N,K], C [M,N]，三段**互不重叠**。
    //
    // 这里原来是「把各 case 的 m/n/k 各取独立最大值再相乘」：
    // `m_max * n_max` = 921600 × 18710 ⇒ **36.4 GB**，比本机 24 GB 物理内存
    // 还大，而且 A/W/C **全绑在偏移 0 上互相覆写**。fp16 只碰 1.9 GB 时侥幸
    // 过关，同日同款算法的 f32 版本（碰 3.8 GB）在 `mm_tg_scan` 里直接打出了
    // **GPU 页故障**（`recoveryCount` 0→1）。改成「各 case 自己的精确尺寸取
    // 最大值」后只有 2.83 GB，且不再有同址读写。
    let per_case = |m: u32, n: u32, k: u32| {
        let (mp, np, kp) = (m.div_ceil(64) * 64, n.div_ceil(64) * 64, k.div_ceil(8) * 8);
        mp as u64 * kp as u64 + np as u64 * kp as u64 + mp as u64 * np as u64
    };
    let max_elems = cases.iter().map(|(_, m, n, k)| per_case(*m, *n, *k)).max().unwrap();
    let buf = ctx.inner.device.new_buffer(max_elems * 2, metal::MTLResourceOptions::StorageModeShared);
    unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, (max_elems * 2) as usize) };

    for variant in &[("p_mm_sg", "pure GEMM, ideal 64x64 tile"), ("p_mm_sg_conv8", "conv8 adapted tile")] {
        let name = variant.0;
        let pipe = ps.pipe(name).expect(&format!("取 {} 管线", name));
        eprintln!("--- Variant: {} ---", name);
        for (label, m, n, k) in &cases {
            let m = *m;
            let n = *n;
            let k = *k;
            // 必须是 64 的倍数（探针要求），且 K 是 8 的倍数。
            let m_pad = m.div_ceil(64) * 64;
            let n_pad = n.div_ceil(64) * 64;
            let k_pad = k.div_ceil(8) * 8;
            let flops = (m as u64) * (n as u64) * (k as u64) * 2; // FMA = MAC ×2
            let bytes = (m_pad * k_pad * 2 + n_pad * k_pad * 2 + m_pad * n_pad * 2) as f64;
            // 本 case 的三段基址（元素 → 字节），两臂共用同一布局：
            // A 在 0、W 在 M*K、C 在 M*K + N*K。
            let off_w = (m_pad as u64 * k_pad as u64) * 2;
            let off_c = (m_pad as u64 * k_pad as u64 + n_pad as u64 * k_pad as u64) * 2;

            // 参数：M,N,K（对齐后的值）。
            let params = [m_pad as u32, n_pad as u32, k_pad as u32];
            let params_bytes = unsafe { std::slice::from_raw_parts(params.as_ptr() as *const u8, 12) };

            // 网格与线程组：与 mm_sg_groups 一致，tg=256 线程/组。
            let groups = [m_pad.div_ceil(64), n_pad.div_ceil(64), 1];
            let threads = [groups[0] * 256, groups[1], groups[2]];

            // 预热 3 轮，然后 best-of-5 取 median。
            let ms = bench(15, || {
                submit_compute(&ctx.inner.queue, |enc| {
                    enc.set_compute_pipeline_state(pipe);
                    // probe 的 A/W/C 都按 half 解释，与 conv8 同 abi；三段错开。
                    enc.set_buffer(0, Some(&buf), 0);
                    enc.set_buffer(1, Some(&buf), off_w);
                    enc.set_buffer(2, Some(&buf), off_c);
                    enc.set_bytes(3, params_bytes.len() as u64, params_bytes.as_ptr() as *const std::ffi::c_void);
                    enc.dispatch_threads(
                        MTLSize::new(threads[0] as u64, threads[1] as u64, threads[2] as u64),
                        MTLSize::new(256, 1, 1),
                    );
                }).expect("提交");
            });
            // FLOPs ÷ 秒 ÷ 1e9 = GFLOPS。（曾错写 /1e6——打印值大 1000 倍，
            // 「44236%」其实是 44.2%。读数自洽但单位错了，和陈旧二进制同款
            // 的静默误导，2026-10-10 修。）
            let gflops = flops as f64 / (ms / 1000.0) / 1e9;
            let roof = 3.34 * 1000.0; // FP32 峰值 GFLOPS（竞品 doc：MMA≈fp32）
            eprintln!(
                "[mb/mm_sg] {:20} {:20} M={:6} N={:6} K={:6}  grid={:5}x{:3}  time={:6.2} ms  gflops={:7.1}  util={:4.1}%  B/FLOP={:.3} B/FLOP",
                name, label, m, n, k, groups[0], groups[1], ms, gflops, gflops / roof * 100.0, bytes / flops as f64
            );
        }
    }
}

/// **MMA 装载路径**探针 —— 分片从 device 直读 vs 经 threadgroup 中转。
///
/// 动机（见 `probe_mm_tg.metal` 头注释）：全图 36~66% 的天花板 + fp16 探针
/// 反而只有 f32 的 0.62×，两个反常指向**同一个**共享病根。全仓四个 MMA 内核
/// 的分片都是 `simdgroup_load` 从 device 直读，而 Apple 的标准快路是
/// 协作拷贝 → threadgroup → barrier → 从 threadgroup 取分片。
///
/// **对照是 `p_mm_sg` 本身**（同一份 run、同一块 buffer、同一套网格），
/// 不是历史读数。
#[test]
fn mm_tg_scan() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };

    // 表 1 出 `p_mm_sg`（fp16 device 直读，对照），表 2 出四条新臂。
    let ctl = ProbeSet::from_list(&ctx.inner.device, MM_PROBE_SHADERS).expect("编译对照探针");
    let ps = ProbeSet::from_list(&ctx.inner.device, MM_TG_PROBE_SHADERS).expect("编译 TG 探针");

    // ⚠️ det 那两个形状的 M **缩到 1/8**（960×120 而不是 960×960）。
    // 原因不是性能，是**内存安全**：M=921600 时 C 是 921600×1024 f32 = 3.8 GB，
    // 叠上 A 就 5.7 GB。这个探针的效果是**每线程组局部**的（协作拷贝 → 屏障 →
    // 分片），与 M 只通过占用率相关：960×120 仍然是 1800×16 = 28800 个线程组，
    // 对本机 ~40 个 GPU 核是 **90× 超订**，占用率早已饱和。
    // 判据：`p_mm_sg` 在小形状上应当仍落在 ~1500（历史大形状读数 1500.5）。
    let cases = [
        ("det960_c512o1024", 960u32 * 120, 1024u32, 512u32),
        ("det960_c1024o512", 960u32 * 120, 512u32, 1024u32),
        ("rec6x208_c192o384", 6u32 * 208, 384u32, 192u32),
        ("rec3x208_c384o768", 3u32 * 208, 768u32, 384u32),
        ("ctc104_c120o18710", 104u32, 18710u32, 120u32),
    ];

    // (臂名, 元素字节数, 说明)。`p_mm_sg` 在对照表里，其余在 TG 表里。
    let arms: [(&str, u64, &str); 6] = [
        ("p_mm_sg", 2, "fp16  device 直读（对照）"),
        ("p_mm_tg", 2, "fp16  threadgroup 单缓冲"),
        ("p_mm_tg2", 2, "fp16  threadgroup 双缓冲"),
        ("p_mm_f32_dev", 4, "f32   device 直读（对照）"),
        ("p_mm_f32_tg", 4, "f32   threadgroup 单缓冲"),
        ("p_mm_f32_tg2", 4, "f32   threadgroup 双缓冲"),
    ];

    for (label, m, n, k) in &cases {
        let (m, n, k) = (*m, *n, *k);
        let m_pad = m.div_ceil(64) * 64;
        let n_pad = n.div_ceil(64) * 64;
        let k_pad = k.div_ceil(8) * 8;
        let flops = (m as u64) * (n as u64) * (k as u64) * 2;
        let groups = [m_pad.div_ceil(64), n_pad.div_ceil(64), 1];
        let threads = [groups[0] * 256, groups[1], groups[2]];
        let params = [m_pad, n_pad, k_pad];
        let params_bytes =
            unsafe { std::slice::from_raw_parts(params.as_ptr() as *const u8, 12) };

        // **每个 case 按自己的精确尺寸分配，A/W/C 三段互不重叠**。
        //
        // 这里原来是照抄 `mm_sg_scan` 的算法：把各 case 的 m/n/k 各取独立最大值
        // 再乘起来 —— `m_max * n_max` = 921600 × 18710 ⇒ **34.4 GB**，比本机
        // 24 GB 物理内存还大，而且 A/W/C 三者**全绑在同一个偏移 0 上互相覆写**。
        // 2026-10-10 实测：fp16 臂（只碰 1.9 GB）三次侥幸过关，加上 f32 臂
        // （要碰 3.8 GB）后 `p_mm_f32_dev` 直接触发 **GPU 页故障**
        // （`recoveryCount` 0→1）。见 `bench/gpuhealth/health.sh`。
        let a_len = m_pad as u64 * k_pad as u64;
        let w_len = n_pad as u64 * k_pad as u64;
        let c_len = m_pad as u64 * n_pad as u64;
        let buf = ctx.inner.device
            .new_buffer((a_len + w_len + c_len) * 4, metal::MTLResourceOptions::StorageModeShared);
        unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, ((a_len + w_len + c_len) * 4) as usize) };

        eprintln!(
            "--- {} M={} N={} K={} grid={}x{}  buf={:.0} MB ---",
            label, m, n, k, groups[0], groups[1],
            (a_len + w_len + c_len) as f64 * 4.0 / 1e6
        );
        for (name, esz, desc) in &arms {
            let pipe = match *name == "p_mm_sg" {
                true => ctl.pipe(name),
                false => ps.pipe(name),
            }
            .expect(&format!("取 {name} 管线"));
            // 三段基址按本臂的元素宽度换算成字节偏移。
            let off_w = a_len * esz;
            let off_c = (a_len + w_len) * esz;
            let ms = bench(15, || {
                submit_compute(&ctx.inner.queue, |enc| {
                    enc.set_compute_pipeline_state(pipe);
                    enc.set_buffer(0, Some(&buf), 0);
                    enc.set_buffer(1, Some(&buf), off_w);
                    enc.set_buffer(2, Some(&buf), off_c);
                    enc.set_bytes(3, params_bytes.len() as u64, params_bytes.as_ptr() as *const std::ffi::c_void);
                    enc.dispatch_threads(
                        MTLSize::new(threads[0] as u64, threads[1] as u64, threads[2] as u64),
                        MTLSize::new(256, 1, 1),
                    );
                }).expect("提交");
            });
            let gflops = flops as f64 / (ms / 1000.0) / 1e9;
            eprintln!(
                "[mb/mm_tg] {:16} {:20} time={:8.2} ms  gflops={:7.1}  util={:4.1}%   {}",
                name, label, ms, gflops, gflops / 3340.0 * 100.0, desc
            );
        }
    }
}

// ---------------------------------------------------------------------------
// conv 吞吐探针：拿**真实层形状**直接测 n_conv8/n_conv 的达成 FLOPS。
//
// 动机：端到端剖析只能告诉我们「n_conv8 占了 det 的 49%」，不能回答
// 「它离硬件峰值还有多远」。这里把层的 (H,W,Cin,Cout,k,s,p) 摊开单独
// dispatch，用**真实 FLOPs ÷ 实测时间**算达成率——是 5% 还是 70%，
// 决定了接下来该改内核还是该换算法。
//
// 参数块布局与 `qppocr_gpu::plan` 发出的 `n_conv8` 逐字段一致（见
// plan.rs 的 ParamBlock 构造）；先跑一个 8×8×8 的小例与朴素 CPU 卷积
// 对拍，确认参数没搬错——不然测出来的时间是假的。
// ---------------------------------------------------------------------------

/// 一个卷积层的形状（NCHW 语义，内部按 NHWC 走）。
struct ConvCase {
    label: &'static str,
    h: u32,
    w: u32,
    cin: u32,
    cout: u32,
    kh: u32,
    kw: u32,
}

/// 步长与 pad。**与 `ConvCase` 分开**是因为后者有 59 处字面量，多 6 个字段
/// 就是 59 处机械改动；而绝大多数用例走的是同一套（stride 1 + 对称 pad），
/// 用 [`ConvGeom::sym`] 生成即可，只有 stride / 非对称 pad 的新用例需要显式写。
#[derive(Clone, Copy)]
struct ConvGeom {
    sh: u32,
    sw: u32,
    pt: u32,
    pl: u32,
    pb: u32,
    pr: u32,
}

impl ConvGeom {
    /// 生产里 k×k 那一族的老约定：stride 1、四边 pad 都是 (k-1)/2。
    fn sym(c: &ConvCase) -> Self {
        Self {
            sh: 1,
            sw: 1,
            pt: c.kh / 2,
            pl: c.kw / 2,
            pb: (c.kh - 1) - c.kh / 2,
            pr: (c.kw - 1) - c.kw / 2,
        }
    }

    /// 输出空间尺寸：`floor((in + padT + padB - k)/s) + 1`。
    fn out_hw(self, c: &ConvCase) -> (u32, u32) {
        let h = (c.h + self.pt + self.pb - c.kh) / self.sh + 1;
        let w = (c.w + self.pl + self.pr - c.kw) / self.sw + 1;
        (h, w)
    }
}

/// 把 NCHW 索引写成 NHWC 下标，做一次朴素卷积（对拍基准）。
/// stride 1 + 对称 pad 的简写——见 [`cpu_conv_nhwc_geom`]。
fn cpu_conv_nhwc(
    input: &[f32],
    weight: &[f32],
    c: &ConvCase,
    out_h: u32,
    out_w: u32,
) -> Vec<f32> {
    cpu_conv_nhwc_geom(input, weight, c, ConvGeom::sym(c), out_h, out_w)
}

/// 同 [`cpu_conv_nhwc`]，但步长与 pad 由 `g` 给。**这是「值」的唯一真值源**
/// ——MMA / edge 两 pass 的数值断言都比它。
fn cpu_conv_nhwc_geom(
    input: &[f32],
    weight: &[f32],
    c: &ConvCase,
    g: ConvGeom,
    out_h: u32,
    out_w: u32,
) -> Vec<f32> {
    let (cin, cout, kh, kw) = (c.cin as usize, c.cout as usize, c.kh as usize, c.kw as usize);
    let (sh, sw) = (g.sh as i64, g.sw as i64);
    let (pt, pl) = (g.pt as i64, g.pl as i64);
    let (ih_max, iw_max) = (c.h as i64, c.w as i64);
    // weight 输入是 [cout][cin][kh][kw]（ONNX 布局）
    let mut out = vec![0f32; out_h as usize * out_w as usize * cout];
    // 1×1/无 pad 快径：逐字等价于下面的通用循环（ih=oh、iw=ow 恒在界内，
    // 内层只剩 ic 的点积），但去掉两层边界判断与除法。**rec 的 CTC 头
    // 分类层**（216×1 c120→18712，485 M MAC）走的是这里——通用循环跑
    // 一遍要几十秒，变体扫描里每个变体都要对拍一次，根本扫不动。
    if kh == 1 && kw == 1 && pt == 0 && pl == 0 && sh == 1 && sw == 1 && cin == c.cin as usize {
        for oh in 0..out_h as usize {
            for ow in 0..out_w as usize {
                let irow = &input[(oh * out_w as usize + ow) * cin..][..cin];
                let orow = &mut out[(oh * out_w as usize + ow) * cout..][..cout];
                for (oc, slot) in orow.iter_mut().enumerate() {
                    let wrow = &weight[oc * cin..][..cin];
                    let mut acc = 0f32;
                    for ic in 0..cin {
                        acc += irow[ic] * wrow[ic];
                    }
                    *slot = acc;
                }
            }
        }
        return out;
    }
    for oh in 0..out_h as usize {
        for ow in 0..out_w as usize {
            for oc in 0..cout {
                let mut acc = 0f32;
                for ic in 0..cin {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let ih = oh as i64 * sh + ky as i64 - pt;
                            let iw = ow as i64 * sw + kx as i64 - pl;
                            if ih < 0 || iw < 0 || ih >= ih_max || iw >= iw_max {
                                continue;
                            }
                            let a = input[((ih as usize * c.w as usize + iw as usize) * cin) + ic];
                            let b = weight[((oc * cin + ic) * kh + ky) * kw + kx];
                            acc += a * b;
                        }
                    }
                }
                out[(oh * out_w as usize + ow) * cout + oc] = acc;
            }
        }
    }
    out
}

/// 一个已组好的 conv 用例：arena、参数块偏移、网格、FLOPs 与对拍素材。
struct ConvProbe {
    buf: metal::Buffer,
    p_off: u32,
    groups: [u32; 3],
    /// 该层的有用 FLOPs（macs × 2）。
    flops: f64,
    /// 下界流量（字节）：读输入 + 读权重 + 写输出。
    bytes: f64,
    /// 输出区在 arena 里的 word 偏移（回读用）。
    out_off_w: u32,
    out_h: u32,
    out_w: u32,
    /// 输入空间尺寸（覆盖几何的列上界要用 `inW`，不是 `outW`——stride≠1
    /// 时两者不同）。
    in_h: u32,
    in_w: u32,
    co4: u32,
    /// 本用例的步长与 pad（写进了参数块，也决定覆盖几何）。
    geom: ConvGeom,
    /// 对拍素材（NHWC 输入、ONNX 布局权重）。
    input: Vec<f32>,
    weight: Vec<f32>,
}

/// 组一个 conv 用例的 arena 与参数块。
fn build_conv_case(device: &metal::DeviceRef, c: &ConvCase) -> ConvProbe {
    build_conv_case_batch(device, c, 1, false)
}

/// 同 `build_conv_case`，但**权重段存 fp16**（`p_whalf` 用）。
///
/// **单变量控制**：arena 的总长、`in_off`/`w_off`/`b_off`/`out_off` 与
/// `build_conv_case` 逐位相同，输入/输出/bias 的内容也逐位相同——**只有
/// 权重段前一半被写成 fp16**（后一半清零闲置）。于是「足迹/line 数」是
/// 两版之间唯一的自变量，与 `p_wsmall` 改地址范围的手法同型。
fn build_conv_case_half(device: &metal::DeviceRef, c: &ConvCase) -> ConvProbe {
    build_conv_case_batch(device, c, 1, true)
}

/// f32 → f16 位型，**就近偶数**舍入。
///
/// 实现在 `qppocr_gpu::nhwc::f32_to_f16_bits`——**只有一份**：生产侧的
/// fp16 权重打包（`repack_conv_w_f16`）与这里的探针必须用同一个函数，
/// 否则「探针量到的收益能不能兑现成生产收益」就多了一个假变量。那边
/// 自带 CPython 参考值对拍测试，这里不再抄一遍测试。
use qppocr_gpu::nhwc::f32_to_f16_bits;

/// 同 `build_conv_case`，但输入/输出段按 `batch` 份铺开、网格 y = `batch`。
/// 内核里 `b = wg.y`，权重各份**共用**（只读一份），输入与输出各占一份。
///
/// ## 为什么需要这个旋钮（2026-10-08）
///
/// `p_fmaonly`（访存全砍、只留 FMA）能在**与 `p_base` 相同的网格**上跑到
/// 78% 峰值 —— 说明线程数对「算力」不是瓶颈。可真内核里 ~66% 的时间是
/// 访存停顿，而停顿只能靠「多几条 warp 可切」来藏。
///
/// batch 是唯一能在**不改每线程工作量、不改访存形状、不改指令序列**的
/// 前提下把常驻 warp 数翻倍的旋钮。于是它把两个还活着的假设劈开：
///
///   · 每份的时间随 batch 上升而**下降** → 停顿没被藏住，是**占用率/隐藏
///     能力**不足 → 处方是提高并行度（合批、缩小瓦片）；
///   · 每份的时间**持平** → 访存吞吐已经到顶，加 warp 没用 → 只能改访存
///     结构（暂存、改布局）。
///
/// 注意这是**唯一**不靠「砍掉某样东西」来做的实验：`p_wconst`/`p_wsmall`/
/// `p_fmaonly` 全都同时改了延迟与字节，只能证伪不能定位。
fn build_conv_case_batch(device: &metal::DeviceRef, c: &ConvCase, batch: u32, half_w: bool) -> ConvProbe {
    build_conv_case_geom(device, c, ConvGeom::sym(c), batch, half_w)
}

/// 同 [`build_conv_case`]，但在**输出段之前**插入 `gap` 个空 word。
///
/// 唯一变化的自由度是地址：段长、段间次序、每个张量的内容、参数块的全部
/// 字段（含 `out_off` 本身要改的那个值——它记录的是真实偏移，内核按它寻址）
/// 都逐位对齐。于是「吞吐随时间随地址摆动」这件事被单独拎出来量。
fn build_conv_case_gap(device: &metal::DeviceRef, c: &ConvCase, gap: u32) -> ConvProbe {
    build_conv_case_geom_w(device, c, ConvGeom::sym(c), 1, false, false, LayoutK::tight(gap, false))
}

/// 段布局旋钮。**只在探针里用**：生产布局由 `qppocr-gpu/src/plan.rs` 的
/// 生命周期分配器决定，这里是为了把它的两个可变量单独拎出来当自变量。
#[derive(Clone, Copy)]
struct LayoutK {
    /// 输出段前插多少空 word。
    gap: u32,
    /// 是否写 `r_off`（写出 `out = conv + r` 的残差段，紧接 bias 段之后）。
    with_res: bool,
    /// 显式段偏移 `[in, w, b, r, out]`（word）与**缓冲总长**。`None` = 紧排。
    ///
    /// 紧排是「段挨着段」的紧凑布局（总长 ~17 MB）；给定时按生产的**绝对**
    /// 地址摆段、缓冲也开到生产那么大（~115 MB）——于是「段被摆在收割 140 MB
    /// arena 的哪几处」这件事本身成了自变量。见 [`conv8_mma_prod_geom`]。
    at: Option<([u32; 5], u32)>,
}

impl LayoutK {
    /// 紧排布局（历史的默认）。
    fn tight(gap: u32, with_res: bool) -> Self {
        LayoutK { gap, with_res, at: None }
    }
}

/// 带残差段的探针：`r` 段紧接 bias、`out` 段在 `r` 之后隔 `gap` 个 word。
///
/// **为什么需要它**（2026-10-11）：生产里 `Conv.55/58` 参数块逐字节相同、
/// 网格相同，只差地址，却稳定差 1.31×（先后对调、前置臂/自然位两次都复现）。
/// 而把 `out` 段在 33 MB 范围内平移的 [`conv8_mma_layout_gap`] 是**平的**
/// （离散 2%）⇒ 单纯「输出段在哪」不是原因。剩下没被控过的变量只剩残差段：
/// 生产里 `Conv.55` 的 `r` 与 `out` 紧邻（`out − r = m·co`），`Conv.58` 的
/// 隔开 2.67 倍。本探针把这两档搭出来。
fn build_conv_case_res(device: &metal::DeviceRef, c: &ConvCase, gap: u32, with_res: bool) -> ConvProbe {
    build_conv_case_geom_w(device, c, ConvGeom::sym(c), 1, false, false, LayoutK::tight(gap, with_res))
}

/// 按**显式绝对偏移**摆段的探针（布局见 [`LayoutK::at`]）。
///
/// 用途见 [`conv8_mma_prod_geom`]：把生产 `[1,3,48,1344]` 计划里 `Conv.55` 与
/// `Conv.58` 的真实地址几何原样搬进来。两者段长相同、内容同族、参数块除偏移
/// 外逐位相同，生产却差 1.29×；本探针回答「这个差是几何本身，还是探针看不见
/// 的东西（arena 总长、别处的内容、别的派发的余波）」。
fn build_conv_case_at(
    device: &metal::DeviceRef,
    c: &ConvCase,
    at: [u32; 5],
    total_w: u32,
    with_res: bool,
) -> ConvProbe {
    build_conv_case_geom_w(
        device,
        c,
        ConvGeom::sym(c),
        1,
        false,
        false,
        LayoutK { gap: 0, with_res, at: Some((at, total_w)) },
    )
}

/// 权重按**分片分块（fragment-blocked）**铺的探针（`p_mma_wfrag` 用）。
///
/// 与 `build_conv_case` **逐字节同布局**：arena 总长、`in_off`/`w_off`/`b_off`/
/// `out_off`、输入/bias/输出的内容全都一样，**只有权重段里那 64 个 float 的
/// 排列顺序**不同 —— 于是「B 侧 `simdgroup_load` 的行距」是两臂之间唯一的
/// 自变量。
///
/// 为什么值得单独一个布局：`p_mma_base` 的 B 装载行距是 `nDim`（几百~几千个
/// float），一条 8×8 分片被拆成 8 段各自的 32 B；分片连续之后同一条装载是
/// 一段 256 B。权重是静态量，重排是一次性的（与既有 `repack_conv_w` 同族），
/// 所以这条处方**不需要动热循环之外任何东西**。
///
/// 前置：`ci4 % 8 == 0 && co4 % 8 == 0`（生产门禁 `cpad4 % 8` / `% 64` 已保证）。
fn build_conv_case_wfrag(device: &metal::DeviceRef, c: &ConvCase) -> ConvProbe {
    build_conv_case_geom_w(device, c, ConvGeom::sym(c), 1, false, true, LayoutK::tight(0, false))
}

/// 同 [`build_conv_case_batch`]，但步长与 pad 由 `g` 给（写进参数块的
/// P+12..P+15 四词，与 `plan.rs` 的 `ParamBlock` 逐字段同义）。
fn build_conv_case_geom(
    device: &metal::DeviceRef,
    c: &ConvCase,
    g: ConvGeom,
    batch: u32,
    half_w: bool,
) -> ConvProbe {
    build_conv_case_geom_w(device, c, g, batch, half_w, false, LayoutK::tight(0, false))
}

/// [`build_conv_case_geom`] 的本体。`w_frag` 是**唯一**多出来的自由度：为真时
/// 权重段按分片分块铺（见 [`build_conv_case_wfrag`]），其余一切逐位相同。
fn build_conv_case_geom_w(
    device: &metal::DeviceRef,
    c: &ConvCase,
    g: ConvGeom,
    batch: u32,
    half_w: bool,
    w_frag: bool,
    k: LayoutK,
) -> ConvProbe {
    assert!(batch >= 1, "batch 至少 1");
    let (pt, pl) = (g.pt, g.pl);
    let (out_h, out_w) = g.out_hw(c);
    let m_dim = out_h * out_w;
    let co4 = c.cout.div_ceil(4) * 4; // nDim（Co 补齐到 4）
    let ci4 = c.cin.div_ceil(4) * 4; // Cin 补齐到 4
    let taps = c.kh * c.kw;

    // 输入/权重：确定性的小值（避开 0/denormal）
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rnd = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 0.125
    };
    // 输入按 NHWC 存（平面顺序 = 层内的自然顺序）。
    // batch 份各自独立（同一形状的不同「图」）——共用同一份会让 L2 命中
    // 率失真，把「多 warp 能不能藏停顿」与「输入是不是已经全在 L2 里」
    // 两个变量混在一起。CPU 对拍只比 b=0 那一份（见 `conv_error`）。
    let input: Vec<f32> = (0..(c.h * c.w * ci4 * batch) as usize).map(|_| rnd()).collect();
    // 权重 ONNX 布局 [cout][cin][kh][kw]，用于 CPU 对拍
    let weight: Vec<f32> = (0..(c.cout * c.cin * taps) as usize).map(|_| rnd()).collect();

    // 重排成内核要的 [tap][ci_pad4][co4]（float4 粒度）
    let mut w_repacked = vec![0f32; (taps * ci4 * co4) as usize];
    for t in 0..taps {
        let (ky, kx) = (t / c.kw, t % c.kw);
        for ic in 0..c.cin {
            for oc in 0..c.cout {
                let src = ((oc * c.cin + ic) * c.kh + ky) * c.kw + kx;
                let dst = ((t * ci4 + ic) * co4 + oc) as usize;
                w_repacked[dst] = weight[src as usize];
            }
        }
    }

    // 权重按 8×8 分片分块重排（`p_mma_wfrag`）：同一批 64 个 float 换个次序，
    // 元素总数与段长一字不变（`taps * ci4 * co4`）。下标约定与
    // `probe_mma_split.metal` 里 `p_mma_wfrag` 的寻址逐字对应：
    //   wf[((tap*(ci4/8) + ci4blk) * (co4/8) + co4blk) * 64 + (ci&7)*8 + (co&7)]
    let w_store: Vec<f32> = if !w_frag {
        w_repacked
    } else {
        assert!(
            ci4 % 8 == 0 && co4 % 8 == 0,
            "分片分块布局要 ci4/co4 都是 8 的倍数，本形状 {}->{}（cin%8 或 co%8 不为 0）",
            c.cin,
            c.cout
        );
        let ci8 = (ci4 / 8) as usize;
        let co8 = (co4 / 8) as usize;
        let mut v = vec![0f32; w_repacked.len()];
        for t in 0..taps as usize {
            for cb in 0..ci8 {
                for ob in 0..co8 {
                    let blk = (t * ci8 + cb) * co8 + ob;
                    for r in 0..8usize {
                        for s in 0..8usize {
                            let src = ((t * ci4 as usize + cb * 8 + r) * co4 as usize) + ob * 8 + s;
                            v[blk * 64 + r * 8 + s] = w_repacked[src];
                        }
                    }
                }
            }
        }
        v
    };

    let n_in = c.h * c.w * ci4 * batch;
    let n_w_data = taps * ci4 * co4; // 权重**不随 batch 增长**（各份共用一份）
    // 与生产侧 `plan.rs` 同款尾垫（`nhwc::MMA_W_TAIL`）：`n_conv8_mma*` 在
    // 列块不满时会读到权重行尾之后。`k.at` 给定时段偏移是显式写死的、后面
    // 紧跟的 bias 段恒为定值，越读落在那儿同样安全，故只在紧排布局里垫。
    let n_w = n_w_data + if k.at.is_none() { qppocr_gpu::nhwc::MMA_W_TAIL as u32 } else { 0 };
    let n_out = m_dim * co4 * batch;
    // 参数块占 word 64..64+21；张量必须让开，否则输入第一段就把
    // taps/cin4v 覆盖成浮点位型——内核会按天文数字循环，直接挂死。
    let p_words = 64u32;
    // `k.at` 给定 = 按生产计划的**绝对**偏移摆段、缓冲开到生产那么大；
    // 否则紧排（`k.gap` 只在输出段前插空白，`k.with_res` 在 bias 之后插一段
    // 长度同输出的残差段）。三种都只改地址，段长与内容一字不改。
    let (in_off_w, w_off_w, b_off_w, r_off_w, out_off_w, total) = match k.at {
        Some(([i, wt, b, r, o], tot)) => {
            assert!(p_words + 21 <= i, "显式布局里参数块会与 in 段交叠");
            assert!(tot as u64 >= o as u64 + n_out as u64, "显式布局的缓冲总长装不下 out 段");
            (i, wt, b, r, o, tot as u64)
        }
        None => {
            let in_off_w = 256u32;
            let w_off_w = in_off_w + n_in;
            let b_off_w = w_off_w + n_w;
            let r_off_w = b_off_w + co4;
            let res_words = if k.with_res { n_out } else { 0 };
            let out_off_w = r_off_w + res_words + k.gap;
            (in_off_w, w_off_w, b_off_w, r_off_w, out_off_w, (out_off_w + n_out + 64) as u64)
        }
    };
    let buf = device.new_buffer(total * 4, metal::MTLResourceOptions::StorageModeShared);

    let base = buf.contents() as *mut u32;
    // SAFETY: `base` 是 Shared 存储的非空基址，缓冲按 `total` 个 word 分配。
    // 紧排时段按 word 依次相接（params[64..85] → in[256..] → w → b → [r] → out，
    // 每段起点是上一段加其长度，故下标 < total；参数块与输入段之间留的空隙是
    // 刻意的，见 `in_off_w` 处注释）；`k.at` 给定时段偏移由调用方保证落在
    // `total` 内（`at` 的 assert 已把参数块与 `in`/`out` 的重叠挡掉）。
    unsafe {
        // 参数块（word 视图）
        let p = base.add(p_words as usize);
        // params[0]（绝对 word 0）= real_n，0 = 不限
        *base = 0;
        // 偏移一律是 **word（f32 元素）下标**——内核内部 `>>2` 折算成
        // float4 下标（1 个 float4 = 4 word）。传字节会整个写飞。
        let words: [u32; 21] = [
            in_off_w,  // in_off（word）
            w_off_w,   // w_off
            b_off_w,   // b_off
            out_off_w, // out_off
            m_dim,         // mDim
            co4,           // nDim
            out_w,         // outW
            c.w,           // inW
            c.h,           // inH
            ci4 / 4,       // cin4v
            taps,          // taps
            c.kw,          // kW
            g.sh,          // sH
            g.sw,          // sW
            pt,            // padT
            pl,            // padL
            0,             // act
            0,             // c1
            0,             // c2
            0,             // c3
            if k.with_res { r_off_w } else { 0xFFFF_FFFF }, // r_off
        ];
        for (i, v) in words.iter().enumerate() {
            *p.add(i) = *v;
        }
        // 张量
        std::ptr::copy_nonoverlapping(input.as_ptr(), base.add(in_off_w as usize) as *mut f32, n_in as usize);
        if half_w {
            // 权重存 fp16：同一逻辑序（`[t][ci4][co4]`）、同一元素下标，
            // 只是每个值占 2 B。整段先清零（`new_buffer` 不清零，尾部的
            // 闲置区必须确定，否则 NaN 会经未定义路径影响计时），再把
            // `n_w_data` 个 half 写进段的前半。
            let w16: Vec<u16> = w_store.iter().map(|&v| f32_to_f16_bits(v)).collect();
            let wp = base.add(w_off_w as usize) as *mut u16;
            std::ptr::write_bytes(wp, 0, n_w as usize);
            std::ptr::copy_nonoverlapping(w16.as_ptr(), wp, n_w_data as usize);
        } else {
            let wdst = base.add(w_off_w as usize) as *mut f32;
            std::ptr::copy_nonoverlapping(w_store.as_ptr(), wdst, n_w_data as usize);
            // 尾垫必须显式清零：`new_buffer` 不清零，越读到任意位型
            // （含 NaN）虽然只落在被掩码的列上，但测试要的是确定性。
            std::ptr::write_bytes(
                wdst.add(n_w_data as usize),
                0,
                (n_w - n_w_data) as usize,
            );
        }
        // bias 清零（b_off 有效但全 0）
        std::ptr::write_bytes(base.add(b_off_w as usize) as *mut f32, 0, co4 as usize);
        // 输出区铺哨兵 7.0：跑完还是 7.0 = 内核根本没写（早退/没派发），
        // 是排障时一眼分辨「没跑」和「跑了但累加器为空」的关键。
        let op = base.add(out_off_w as usize) as *mut f32;
        for i in 0..n_out as usize {
            *op.add(i) = 7.0;
        }
        // 残差段（`k.with_res`）：长度同输出、紧接 bias 之后，寻址与输出逐字段
        // 一致（`b*mDim*nDim + m*nDim + n`，见 `n_conv8_mma.metal` 尾声）。
        // 内容必须显式写好：`new_buffer` 不清零，未写的 float 可能是 NaN 位型，
        // 会把「残差加法的成本」与「NaN 走的路」混成一个自变量。
        if k.with_res {
            let rp = base.add(r_off_w as usize) as *mut f32;
            for i in 0..n_out as usize {
                *rp.add(i) = rnd();
            }
        }
    }

    let nv = co4 / 4;
    // 网格仍是「每份一份线程」，只是 y 方向铺 batch 份 —— 这与 `n_conv8`
    // 里 `b = wg.y` 的生产约定一致（`groups_for` 只算 x，恒 y=1）。
    let groups = [(m_dim.div_ceil(4) * (nv / 2)).div_ceil(256), batch, 1];
    let flops = m_dim as f64 * c.cout as f64 * c.cin as f64 * taps as f64 * 2.0 * batch as f64;
    // 下界流量：权重只算一份（真实网络里它在 L2 里被各份共用）。
    // `half_w` 时权重段每元素 2 B——这个差额就是 `p_whalf` 要验的那件事，
    // 分母必须如实反映，否则达成带宽会虚高。
    let w_bytes = if half_w { n_w as f64 * 2.0 } else { n_w as f64 * 4.0 };
    let bytes = (n_in + n_out) as f64 * 4.0 + w_bytes;
    ConvProbe {
        buf,
        p_off: p_words,
        groups,
        flops,
        bytes,
        out_off_w,
        out_h,
        out_w,
        in_h: c.h,
        in_w: c.w,
        co4,
        geom: g,
        input,
        weight,
    }
}

/// `n_conv8_edge` 的**紧凑网格**：只铺真正含边框像素的行组。
///
/// 生产在 `qppocr-gpu/src/plan.rs` 的 k×k 派发支，内核在
/// `shaders/n_conv8.metal` 的 EDGE 枚举段。**这里是第三份独立拷贝，故意
/// 的**：三份不一致时测试就该红，而不是让探针量到一个和生产不同的网格、
/// 把「内核慢」和「网格算错」混成一个数。
///
/// 契约（`k` = 方块边长，对称 pad ⇒ `padT == padL == (k-1)/2`）：
///
/// ```text
///   gh     = outW / 4              每行行组数（行组 = 一行内连续 4 个 x）
///   gL     = ceil(padL / 4)        中带每行头/尾各铺几个行组
///   tiles  = (2*(padT*gh) + (outH-2*padT)*2*gL) * nh      nh = nDim/8
///   groups = ceil(tiles / 256)     与 `n_conv8` 同线程组尺寸、无 grid-stride
/// ```
///
/// 两条门禁与 plan.rs、内核头注释逐字相同，这里用 `assert!` 顶住：`outW%4`
/// 保证行组不跨行，`2*gL <= gh` 保证中带头尾不重叠（重叠 = 同一行组枚举
/// 两次 = **写两次**，静默错）。
fn edge_groups(pr: &ConvProbe, k: u32) -> [u32; 3] {
    edge_groups_geom(pr, k, k)
}

/// mma3 的线程组网格——与内核里 `xblk`/`nfb` 的解码区间逐字一致。
fn cv_groups(cv: &KxkTestCover, nch: u32) -> [u32; 3] {
    [(cv.nfb + u32::from(cv.has_tail)) * nch, cv.ny, 1]
}

/// 覆盖几何的**第三份独立拷贝**（另两份在 `qppocr-gpu/src/plan.rs` 的
/// `KxkCover::new`、`shaders/n_conv8_mma3.metal` 的 `mma3_body` 与
/// `shaders/n_conv8.metal` 的 `conv8_body<EDGE>`）。**故意的**：三份不一致
/// 时测试就该红，而不是让探针量到一个和生产不同的网格、把「内核慢」和
/// 「网格算错」混成一个数。
///
/// 语义见 `KxkCover` 的文档：内点 ⟺ 乘上步长之后仍落在
/// `[pad, in-1-(k-1)+pad]` 里。
struct KxkTestCover {
    /// 内点行闭区间。
    y_lo: u32,
    y_hi: u32,
    /// 单点安全列闭区间。
    x_lo: u32,
    x_hi: u32,
    /// 整块数与尾块标志。
    nfb: u32,
    has_tail: bool,
    /// 内点行数 / 每行行组数 / 中带头尾行组数。
    ny: u32,
    gh: u32,
    gl: u32,
    gr: u32,
}

fn kxk_cover_test(pr: &ConvProbe, kh: u32, kw: u32) -> KxkTestCover {
    let g = pr.geom;
    assert!(pr.out_w % 4 == 0, "紧凑网格门禁 outW%4（outW={}）", pr.out_w);
    let y_lo = g.pt.div_ceil(g.sh);
    let y_hi = (pr.in_h - 1 - (kh - 1) + g.pt) / g.sh;
    let x_lo = g.pl.div_ceil(g.sw);
    let x_hi = (pr.in_w - 1 - (kw - 1) + g.pl) / g.sw;
    assert!(y_hi >= y_lo && y_hi <= pr.out_h - 1, "行内点区间非法");
    assert!(x_hi >= x_lo + 63, "列上铺不出一个 64 整块（x_lo={x_lo} x_hi={x_hi}）");
    let nfb = (x_hi - 63 - x_lo) / 64 + 1;
    let has_tail = x_lo + 64 * nfb - 1 < x_hi;
    let gh = pr.out_w / 4;
    let gl = x_lo.div_ceil(4);
    let gr = (pr.out_w - 1 - x_hi).div_ceil(4);
    assert!(gl + gr <= gh, "紧凑网格门禁 gL+gR<=gh（gL={gl} gR={gr} gh={gh}）");
    KxkTestCover {
        y_lo,
        y_hi,
        x_lo,
        x_hi,
        nfb,
        has_tail,
        ny: y_hi - y_lo + 1,
        gh,
        gl,
        gr,
    }
}

fn edge_groups_geom(pr: &ConvProbe, kh: u32, kw: u32) -> [u32; 3] {
    let cv = kxk_cover_test(pr, kh, kw);
    let nh = (pr.co4 / 4) / 2;
    let n_bot = pr.out_h - 1 - cv.y_hi;
    let tiles = ((cv.y_lo + n_bot) * cv.gh + cv.ny * (cv.gl + cv.gr)) * nh;
    // 上界对拍：紧凑网格必须是**全平面网格**的子集再被筛掉一些。`pr.groups`
    // 是 `build_conv_case` 按旧的 `ceil(m/4)*nh` 算的，公式来源不同 ⇒ 这条
    // 断言是有内容的（枚举式写错量级会当场红）。
    let all = (pr.out_h * pr.out_w).div_ceil(4) * nh;
    assert!(tiles <= all, "紧凑网格比全平面还大：{tiles} > {all}");
    [tiles.div_ceil(256), 1, 1]
}

// ---------------------------------------------------------------------------
// depthwise（`n_conv_dw`）的用例构造
// ---------------------------------------------------------------------------

/// 一个 depthwise 层的形状。语义与 `plan.rs` 的 `xs[1..]` 一致：weight
/// 是 `[C,1,kh,kw]`，输入是 NHWC `[h][w][c]`，输出通道 = 输入通道 = `c`。
struct DwCase {
    label: &'static str,
    /// 输入高/宽（**不是**输出 —— 与 `ConvCase` 的约定相反，那边给的是输出）。
    h: u32,
    w: u32,
    c: u32,
    kh: u32,
    kw: u32,
    stride: u32,
}

/// depthwise 的 `(输出高, 输出宽, 上/左 padding)`。
///
/// `no_pad` 那份是 `p_dw_nobc` 专用的：pad=0 且输出只覆盖「每个 tap 都在
/// 界内」的位置，于是内核里删掉边界判断**不会**越界。它只有 `p_dw_base`
/// 在**同一 no_pad 配置**上的读数可比 —— 换配置后 m 变了，绝对值不可比。
fn dw_dims(c: &DwCase, no_pad: bool) -> (u32, u32, u32) {
    let pad = if no_pad { 0 } else { (c.kh - 1) / 2 };
    (((c.h + 2 * pad - c.kh) / c.stride) + 1, ((c.w + 2 * pad - c.kw) / c.stride) + 1, pad)
}

/// 朴素 depthwise 卷积（NHWC 输入、ONNX `[C,1,kh,kw]` 权重），对拍基准。
fn cpu_dw_ref(input: &[f32], weight: &[f32], c: &DwCase, no_pad: bool) -> Vec<f32> {
    let (out_h, out_w, pad) = dw_dims(c, no_pad);
    let (ch, kh, kw, s) = (c.c as usize, c.kh as usize, c.kw as usize, c.stride as usize);
    let (ih_max, iw_max) = (c.h as i64, c.w as i64);
    let mut out = vec![0f32; out_h as usize * out_w as usize * ch];
    for oh in 0..out_h as usize {
        for ow in 0..out_w as usize {
            for ic in 0..ch {
                let mut acc = 0f32;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let ih = (oh * s + ky) as i64 - pad as i64;
                        let iw = (ow * s + kx) as i64 - pad as i64;
                        if ih < 0 || iw < 0 || ih >= ih_max || iw >= iw_max {
                            continue;
                        }
                        let a = input[(ih as usize * c.w as usize + iw as usize) * ch + ic];
                        acc += a * weight[ic * kh * kw + ky * kw + kx];
                    }
                }
                out[(oh * out_w as usize + ow) * ch + ic] = acc;
            }
        }
    }
    out
}

/// 组一个 depthwise 用例的 arena 与参数块（字段含义同 `ConvProbe`，
/// `co4` 这里装的是补齐后的通道数 `cDim`）。
///
/// 参数块**逐字段抄 `plan.rs`**（那里是 `n_conv_dw` 的唯一出口），偏移一律
/// 是 word 下标 —— 内核内部 `>>2` 折成 float4 下标，传字节会整个写飞。
/// `ingap`/`wgap`：输入段、权重段前各插多少空 word。**只为量相位**。
///
/// `n_conv_dw` 的权重装载和 `n_conv8_mma` 一样吃「起点字节偏移 mod 64」，
/// 但幅度只有 1.09~1.20×，**解释不了**生产里 dw 家族 24.31 → 17.37 ms
/// （1.40×，同一二进制重跑离散只有 4%）。dw 的另一个操作数是 NHWC 输入，
/// 行距 `c_dim*4`（c=256 时 1024 B）是 64 的整数倍 ⇒ **每一行同相**，
/// 所以输入段起点的相位同样值得单独扫——dw 的读形态（一个 float4 = 同一
/// 像素的 4 个通道）与 MMA 的激活读完全不同。
fn build_dw_case(
    device: &metal::DeviceRef,
    c: &DwCase,
    no_pad: bool,
    ingap: u32,
    wgap: u32,
) -> ConvProbe {
    let (out_h, out_w, pad) = dw_dims(c, no_pad);
    let m_dim = out_h * out_w;
    let c_dim = c.c.div_ceil(4) * 4; // cpad4(Co)
    let cv4 = c_dim / 4;
    let taps = c.kh * c.kw;

    let mut seed = 0x0bad_c0de_dead_beefu64;
    let mut rnd = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 0.125
    };
    // 输入 NHWC；权重 ONNX 布局 [c][1][kh][kw]（对拍用），再重排成
    // 内核要的 `[t][c4]`（一个 float4 = 4 个通道该 tap 的权重）。
    let input: Vec<f32> = (0..(c.h * c.w * c.c) as usize).map(|_| rnd()).collect();
    let weight: Vec<f32> = (0..(c.c * taps) as usize).map(|_| rnd()).collect();
    let mut w_repacked = vec![0f32; (taps * cv4 * 4) as usize];
    for t in 0..taps {
        let (ky, kx) = (t / c.kw, t % c.kw);
        for c4 in 0..cv4 {
            for l in 0..4u32 {
                let ch = c4 * 4 + l;
                w_repacked[(t * cv4 + c4) as usize * 4 + l as usize] =
                    if ch < c.c { weight[(ch * c.kh * c.kw + ky * c.kw + kx) as usize] } else { 0.0 };
            }
        }
    }

    let n_in = c.h * c.w * c_dim;
    let n_w = taps * c_dim;
    let n_out = m_dim * c_dim;
    let p_words = 64u32;
    let in_off_w = 256u32 + ingap;
    let w_off_w = in_off_w + n_in + wgap;
    let b_off_w = w_off_w + n_w;
    let out_off_w = b_off_w + c_dim;
    let total = (out_off_w + n_out + 64) as u64;
    let buf = device.new_buffer(total * 4, metal::MTLResourceOptions::StorageModeShared);

    let base = buf.contents() as *mut u32;
    // SAFETY: 同 `build_conv_case_batch` —— Shared 存储的非空基址，缓冲按
    // `total` 个 word 分配，各段起点依次相接，故所有下标 < total。
    unsafe {
        let p = base.add(p_words as usize);
        *base = 0; // real_n = 0（不限批）
        let words: [u32; 20] = [
            in_off_w,
            w_off_w,
            b_off_w,
            out_off_w,
            m_dim,
            c_dim,
            out_w,
            c.w,  // inW
            c.h,  // inH
            taps,
            c.kw, // kW
            c.stride,
            c.stride,
            pad,  // padT
            pad,  // padL
            0,    // act = 0（探针只量卷积本体；act 由尾部一次施加，81 tap 下可忽略）
            0,    // c1
            0,    // c2
            0,    // c3
            0xFFFF_FFFF, // r_off = OFF_NONE
        ];
        for (i, v) in words.iter().enumerate() {
            *p.add(i) = *v;
        }
        std::ptr::copy_nonoverlapping(input.as_ptr(), base.add(in_off_w as usize) as *mut f32, n_in as usize);
        std::ptr::copy_nonoverlapping(w_repacked.as_ptr(), base.add(w_off_w as usize) as *mut f32, n_w as usize);
        std::ptr::write_bytes(base.add(b_off_w as usize) as *mut f32, 0, c_dim as usize);
        let op = base.add(out_off_w as usize) as *mut f32;
        for i in 0..n_out as usize {
            *op.add(i) = 7.0; // 哨兵：跑完还是 7.0 = 内核没写
        }
    }

    let groups = super::conv_probe::dw_groups_for(m_dim, c_dim);
    let flops = m_dim as f64 * c.c as f64 * taps as f64 * 2.0;
    let bytes = ((c.h * c.w * c.c) + n_out) as f64 * 4.0 + (n_w * 4) as f64;
    ConvProbe {
        buf,
        p_off: p_words,
        groups,
        flops,
        bytes,
        out_off_w,
        out_h,
        out_w,
        in_h: c.h,
        in_w: c.w,
        co4: c_dim,
        // depthwise 的 pad 是对称的（`pad` 同时给了 padT/padL，参数块里也只
        // 有这两词），步长取 DwCase 自己的。这里只是为了填满 `ConvProbe`
        // ——dw 用例不走 k×k 覆盖几何那条路。
        geom: ConvGeom {
            sh: c.stride,
            sw: c.stride,
            pt: pad,
            pl: pad,
            pb: pad,
            pr: pad,
        },
        input,
        weight,
    }
}

/// GPU 测试串行闸。
///
/// `cargo test` 默认多线程并行跑测试，两个 GPU 基准同时跑会互相抢设备。
/// 实测同一内核同一形状能因此相差 **7 倍**（156 vs 1066 GFLOPS）——那种
/// 数字上的一切归因都是假的。所有 GPU 基准进门先取这把锁。
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// 取串行闸。中毒（别的测试 panic）时继续用——**不掩盖**，只是不级联。
fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// 一次计时：单次时间（最好的一轮）与轮间离散度（诊断用）。
struct Timing {
    ms: f64,
    /// 最慢轮 / 最快轮。≈1 说明这一轮测量可信；≥1.5 说明设备在浮动，
    /// 数字只能当量级看。
    spread: f64,
}

/// 单次执行时间：《一条命令缓冲内跑 `r` 次，总时 / r》，多轮取**最小**。
///
/// ## 两个踩过的坑
///
/// - **固定小 reps 会骗人**：同一个 256 通道 1×1 两次跑出 0.061 ms 与
///   0.000 ms（882 与 5×10⁷ GFLOPS）——单次真实执行远低于命令缓冲固定
///   开销，差分全被抖动吃掉。所以 `r` 自适应涨到总时盖过地板。
/// - **取斜率也会骗人**：`(T(2r) − T(r))/r` 假设 T 对 r 线性。实测
///   `T(8)/8 ≈ 2.5 ms` 而 `T(64)/64 ≈ 0.33 ms`——前几次 dispatch 在
///   把 arena 拉进 L2（几 MB 的工作集），冷启动那一段被除到斜率里，
///   把 3×3 conv 的算力低报了近一个数量级。**换成总时 / r，让冷启动
///   被 r 摊薄。**
///
/// 取最小而非平均：M4 的 GPU 频率随负载浮动，同一份代码两次能差 1.6 倍
/// （实测 p_base 与 n_conv8 的高低方向都能翻转）。最小值逼近「没被降频
/// 打断」的那一轮。
///
/// 代价：这是 **L2 热** 口径。小特征图（rec 的 24×272×64）在真实网络里
/// 上一层的输出本就在 L2，热口径对得上；det 的 26 MB 全分辨率层超出
/// L2，真实情况更接近冷口径。**跨变体比较用它，绝对值别当端到端。**
fn time_dispatch(run: &dyn Fn(u32) -> f64) -> Timing {
    const ROUNDS: usize = 3;
    let _ = run(1); // 预热（也把管线状态、页映射逼出来）
    let r = calibrate(run);
    let mut best = f64::INFINITY;
    let mut worst = 0f64;
    for _ in 0..ROUNDS {
        let t = run(r) / r as f64;
        best = best.min(t);
        worst = worst.max(t);
    }
    Timing { ms: best.max(1e-6), spread: worst / best.max(1e-6) }
}

/// 校准重复次数 `r`：让一轮（**一条命令缓冲里跑 `r` 次**）的总时长同时
/// 盖过两样东西——命令缓冲的固定开销，以及头几次 dispatch 把 arena 拉进
/// L2 的冷启动。`总时 / r` 只有在这两样都被摊薄后才等于单次执行时间。
///
/// 起点就是 `REP_MIN = 16`：从 1 开始翻倍的话，慢内核（det 全分辨率层
/// 单次 13 ms）在 `r = 2` 就撞过地板停住，固定开销除以 2 还剩几百微秒，
/// 正好是它单次时间的几个百分点——那点偏置足以让变体排名翻转。
fn calibrate(run: &dyn Fn(u32) -> f64) -> u32 {
    const REP_MIN: u32 = 16;
    const FLOOR_MS: f64 = 60.0;
    const REP_CAP: u32 = 1024;
    let mut r = REP_MIN;
    while run(r) < FLOOR_MS && r < REP_CAP {
        r *= 2;
    }
    r
}

/// 读回输出段，与朴素 CPU 卷积比最大绝对误差。
///
/// **每个「结果正确」的变体都必须过这一关**——微基准里出现过「快得可疑」
/// 的变体，查下去是内核压根没写输出（哨兵 7.0 还在）。只量时间的扫描
/// 会把它当成大赢家。
fn conv_error(pr: &ConvProbe, c: &ConvCase) -> f32 {
    // SAFETY: out_off_w + out_h*out_w*co4 ≤ total（见 build_conv_case 的段
    // 布局），缓冲是 Shared 存储、调用方已 wait_until_completed，长度
    // out_h*out_w*co4 的 f32 视图完全落在输出段内；`want` 同长。
    let got = unsafe {
        std::slice::from_raw_parts(
            (pr.buf.contents() as *const f32).add(pr.out_off_w as usize),
            (pr.out_h * pr.out_w * pr.co4) as usize,
        )
    };
    let want = cpu_conv_nhwc(&pr.input, &pr.weight, c, pr.out_h, pr.out_w);
    got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max)
}

/// 参数搬运对拍：小卷积跑一遍，与朴素 CPU 卷积比数值。
#[test]
fn conv8_param_layout_matches_cpu() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");
    let c = ConvCase { label: "sanity", h: 8, w: 8, cin: 8, cout: 8, kh: 3, kw: 3 };
    let pr = build_conv_case(&ctx.inner.device, &c);

    submit_compute(&ctx.inner.queue, |enc| {
        ks.dispatch(enc, "n_conv8", &pr.buf, pr.p_off, pr.groups).expect("dispatch");
    })
    .expect("提交");

    // n_conv8 的输出是 NHWC；对拍用 cout==co4 的形状，通道维无补齐差
    let worst = conv_error(&pr, &c);
    eprintln!(
        "[mb] conv8 参数对拍：{} 个输出，最大绝对误差 {worst:.3e}",
        pr.out_h * pr.out_w * pr.co4
    );
    assert!(worst < 1e-4, "n_conv8 与 CPU 卷积不符（最大误差 {worst:.3e}）——参数块搬错了");
}

/// `n_conv8_mma` 与朴素 CPU 卷积对拍（1×1，f32 MMA）。
///
/// ## 为什么必须先过这一关
///
/// 这个内核里有两处**从未被数值验证过**的假设，任一处错了都会得到
/// 「跑得飞快但结果是垃圾」——而只量时间的扫描会把它当大赢家：
///
/// 1. **`simdgroup_load` 的行 stride 语义**：`simdgroup_load(m, base, s)`
///    里 `s` 是源矩阵的**行**元素数（行主序）。我们的 A（NHWC，行 stride
///    `kf`）与 B（`[k][n]` k-主序，行 stride `nDim`）都押在这条上。
/// 2. **`simdgroup_multiply_accumulate` 的 f32×f32→f32 组合**：
///    编译过了只说明重载存在，不说明算的是对的（MMA 的 lane 分发与
///    转置约定错了照样编译）。
///
/// 三个形状覆盖三种边界：`mDim` 是 64 的整数倍（满块）、`mDim < 64`
/// （**单块尾掩码**）、多块（`mDim=256` 跨 4 个 m 块）。`nDim`/`kf` 都由
/// 内核契约要求 64/8 对齐，故不测非对齐（计划层不会派发）。
#[test]
fn conv8_mma_matches_cpu() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    // (h, w, cin, cout, 说明)：m_dim = h*w。
    //
    // **两条实例化都要过**（NCOLS=64 走 `n_conv8_mma`，NCOLS=32 走
    // `n_conv8_mma_n32`）：Co=96（3×32 列块）是 rec 的 Conv.4/9/12/15/47
    // 与 det 的 CSP 支线的真实形状，`96%64=32` ⇒ 只能进 32 那一档。
    // 这一档的 NSG=4/128 线程是个独立耦合面（`threads_per_group` 少写一条
    // 就会让线程认领到不存在的 simdgroup、越界写 `ost`），所以必须单列用例。
    let cases: [(u32, u32, u32, u32, &str); 11] = [
        (8, 8, 64, 64, "满块 64 行 × 64 列"),
        (4, 8, 64, 64, "尾掩码 32 行（<64）"),
        (16, 16, 128, 128, "多块 256 行 × 2 列块"),
        (5, 7, 32, 64, "怪形状 35 行 × 64 列"),
        (8, 8, 32, 96, "**N32** Co=96=3×32 列块"),
        (4, 32, 192, 96, "**N32** Co=96 尾掩码 128 行"),
        (16, 144, 48, 96, "**N32** Co=96 多行块（rec 的真实形状）"),
        // 以下四条 `cpad4(Co) % 32 != 0`：走「尾块」那条路（列块不满 ⇒
        // 尾声 `n < nDim` 掩码 + `nhwc::MMA_W_TAIL` 尾垫）。**这组才是
        // 判据从 `%32` 放到 `%8` 之后真正新覆盖的形状**，Co=24 尤其常见
        // （det 的 CSP 支线），MatMul.12 则是剩余 `n_conv8` 里最大的一条。
        (8, 8, 32, 24, "**尾块** Co=24 = 3 个 8 列分片"),
        (16, 144, 96, 48, "**尾块** Co=48 多行块"),
        (8, 17, 120, 360, "**尾块** Co=360（120→360 真实形状）"),
        (2, 68, 120, 18712, "**尾块** MatMul.12 真实形状（120→18712）"),
    ];

    for (h, w, cin, cout, note) in cases {
        let c = ConvCase { label: "mma", h, w, cin, cout, kh: 1, kw: 1 };
        let mut pr = build_conv_case(&ctx.inner.device, &c);
        // MMA 网格与 n_conv8 不同：x = 64 行块，y = NCOLS 列块，z = 批。
        // 通道块宽与内核名成对选（与 `plan.rs` 那一支同一个判据）。
        let ncols = if pr.co4 % 64 == 0 { 64 } else { 32 };
        let kname = if ncols == 64 { "n_conv8_mma" } else { "n_conv8_mma_n32" };
        let groups = [
            (h * w).div_ceil(64),
            pr.co4.div_ceil(ncols),
            1,
        ];
        submit_compute(&ctx.inner.queue, |enc| {
            ks.dispatch(enc, kname, &pr.buf, pr.p_off, groups)
                .expect("dispatch");
        })
        .expect("提交");
        let worst = conv_error(&pr, &c);
        eprintln!(
            "[mb] {kname:<16} 对拍 {note:<28} ({h}×{w} c{cin}->{cout} k1): 最大误差 {worst:.3e}"
        );
        assert!(
            worst < 1e-4,
            "{kname} 与 CPU 卷积不符（{note}）：最大误差 {worst:.3e}——\
             simdgroup_load/MMA 的布局假设错了，或 NCOLS/NSG 与线程组尺寸脱钩"
        );
        std::hint::black_box(&mut pr);
    }
}

/// `n_conv8` vs `n_conv8_mma3 + n_conv8_edge` 的孤立吞吐对比（k×k 稠密层）。
///
/// 形状取 `conv8_throughput` 名单里的 k3 项（那批是 det 的 FLOP 前几名与
/// rec 的前段）——**"变体探针有用" 与 "这条形状上有用" 是两件事**，后者
/// 必须按真实名单量。
///
/// 三列读数：base（`n_conv8` 一条）、mma3（只派 MMA 那支）、edge（只派补算
/// 那支）。生产的 B 臂是 `mma3 + edge` 两条 dispatch，所以**看的是 mma3+edge
/// 的和**，单列出来是为了让「边框补算到底值多少」可见——它是这套设计唯一的
/// 固定成本：edge 在整图上空转，只做边框那不到 1% 的像素。
///
/// ## 为什么给 edge 单列一栏（而不是合起来量一个数）
///
/// 若合起来只报一个和，两种完全不同的失败会长得一样慢：MMA 慢（处方在内核）
/// 与 edge 空转贵（处方在「缩小 edge 网格」）。分栏才能一眼分开。
#[test]
fn conv8_mma3_throughput() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    let cases = [
        // det medium 的 FLOP 首名（9.6 GFLOP）
        ConvCase { label: "det 136x240 c256->64 k3", h: 136, w: 240, cin: 256, cout: 64, kh: 3, kw: 3 },
        ConvCase { label: "det 136x240 c128->64 k3", h: 136, w: 240, cin: 128, cout: 64, kh: 3, kw: 3 },
        ConvCase { label: "det 240x216 c64->64 k3", h: 240, w: 216, cin: 64, cout: 64, kh: 3, kw: 3 },
        // 这两条曾是**门外**（`co4=32 < 64`，64 宽通道瓦片放不下），逼出了
        // `n_conv8_mma3_n32`。det 480x432 c32->32 k3 正是 `conv8_variant_scan`
        // 量 p_base 的地方（1157 GFLOPS），现在可以直接对着那条 2.38× 的
        // 处方空间量 N32 兑现了多少。
        ConvCase { label: "det 480x432 c32->32 k3 (N32)", h: 480, w: 432, cin: 32, cout: 32, kh: 3, kw: 3 },
        ConvCase { label: "rec 48x544 c32->32 k3 (N32)", h: 48, w: 544, cin: 32, cout: 32, kh: 3, kw: 3 },
        // rec 前段（3×3 占 rec 设备时间的 16.2%）
        ConvCase { label: "rec 24x272 c64->64 k3", h: 24, w: 272, cin: 64, cout: 64, kh: 3, kw: 3 },
        ConvCase { label: "rec 12x136 c128->128 k3", h: 12, w: 136, cin: 128, cout: 128, kh: 3, kw: 3 },
    ];

    for c in &cases {
        let pr = build_conv_case(&ctx.inner.device, c);
        let k = c.kh;
        let nfb = (pr.out_w - (k - 1)) / 64;
        let nrows = pr.out_h - (k - 1);
        // 通道块宽与 plan.rs 同一判据（第 N 份拷贝，测试里就是要独立一份）。
        let ncols = if pr.co4 % 64 == 0 { 64 } else { 32 };
        let nch = pr.co4.div_ceil(ncols); // 上取整（通道尾块，见 plan.rs 同支）
        let kname = if ncols == 64 { "n_conv8_mma3" } else { "n_conv8_mma3_n32" };
        // 契约检查：不满足就是**门外**（计划层不会派发这种形状），打印后跳过。
        // 后两条是边框紧凑网格的门禁（plan.rs 逐字同款）——把它们也算进契约，
        // 否则这里会量到一种生产根本不派发的配置。
        let gl = (k / 2 + 3) / 4;
        if nfb < 1 || c.cin % 8 != 0 || pr.co4 % 8 != 0 || pr.out_w % 4 != 0 || 2 * gl > pr.out_w / 4 {
            eprintln!(
                "[mb] {:<26} 门外：nfb={nfb} cin={} co4={} outW={} 2gL={}（计划层会回落 n_conv8）",
                c.label, c.cin, pr.co4, pr.out_w, 2 * gl
            );
            continue;
        }
        let inner_w = pr.out_w - (k - 1);
        let tail = inner_w - nfb * 64;
        let nx = nfb + if tail > 0 { 1 } else { 0 };
        let mma_groups = [nx * nch, nrows, 1];
        let eg = edge_groups(&pr, k);

        let run = |names: &[&str], groups: [u32; 3], reps: u32| -> f64 {
            let t = Instant::now();
            submit_compute(&ctx.inner.queue, |enc| {
                for _ in 0..reps {
                    for nm in names {
                        ks.dispatch(enc, nm, &pr.buf, pr.p_off, groups).expect("dispatch");
                        let res: [&metal::ResourceRef; 1] = [&pr.buf];
                        enc.memory_barrier_with_resources(&res);
                    }
                }
            })
            .expect("提交");
            t.elapsed().as_secs_f64() * 1000.0
        };
        // 奇偶轮换序（本机先跑的那条快 6~15%）
        let (mut a, mut b, mut e) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        for i in 0..3 {
            let one = |nm: &str, g: [u32; 3]| {
                time_dispatch(&|r| run(&[nm], g, r))
            };
            let (x, y, z) = if i % 2 == 0 {
                (one("n_conv8", pr.groups), one(kname, mma_groups), one("n_conv8_edge", eg))
            } else {
                let z = one("n_conv8_edge", eg);
                let y = one(kname, mma_groups);
                let x = one("n_conv8", pr.groups);
                (x, y, z)
            };
            a = a.min(x.ms);
            b = b.min(y.ms);
            e = e.min(z.ms);
        }
        // 生产 B 臂 = 两条 dispatch 的和（不单独量一遍：那要再多轮，而这里的
        // 每支读数已经各自用「总时/r」的口径摊掉了命令缓冲固定开销）。
        let gf = |ms: f64| pr.flops / (ms * 1e6);
        eprintln!(
            "[mb] {:<26} base {:>7.3} ms ({:>6.1} GF) → mma{:<2} {:>7.3} ({:>6.1} GF) + edge {:>6.3} \
             = {:>7.3} ms = **{:.2}×**  [grid {:?} → {:?} + edge {:?}, 尾 {tail} 列, 走慢路 {:.1}%]",
            c.label,
            a,
            gf(a),
            ncols,
            b,
            gf(b),
            e,
            b + e,
            a / (b + e),
            pr.groups,
            mma_groups,
            eg,
            100.0 * (1.0 - (nrows * inner_w) as f64 / (pr.out_h * pr.out_w) as f64)
        );
    }
}

/// `n_conv8_mma3` 家族的**四臂诊断**（NCOLS=64 与 32 各一套）+ 尾块定价。
///
/// ## 靶子与两个待解释的数
///
/// 这族端到端 **23.98 ms / 11 条 = det 设备端的 20%**。把每条的形状从
/// `QPPOCR_METAL_PLAN_DUMP` 取出来、按 `2·m·N·K` 算 GFLOPS 之后：
///
/// | 内核 | 形状 | 有用 GFLOPS | **纯 MMA** GFLOPS |
/// |---|---|---|---|
/// | Conv.121 | 240x216 c256→64 k3 | 1929 | 2307 |
/// | Conv.3   | 240x216 c128→64 k3 | 1944 | 2315 |
/// | Conv.2   | 480x432 c32→64 k2  | 1947 | 2024 |
/// | Conv.66  | 120x108 c64→64 k3  | 2033 | 2455 |
/// | Conv.111 | 240x216 c32→32 k7  | 1494 | 1821 |
/// | Conv.114 | 240x216 c32→32 k5  | 1245 | 1503 |
/// | Conv.1   | 480x432 c64→32 k2  | 1649 | 1714 |
///
/// 「纯 MMA」= 有用 × 尾块冗余（算出 256 列只需 214 列 ⇒ 1.196×）。两条读法：
///
/// 1. **64 宽那族已经到顶**：2024~2455 落在家族天花板（2044~2444，见
///    `conv8_mma3_throughput` 的历史读数）里，没余量。它的 14~21% 冗余是
///    **尾块**造成的——不是热循环问题，用四臂量不出来（所以本测试单列
///    `tail`/`notail` 两臂直接给这笔账定价）。
/// 2. **n32 明显低于天花板**：1503~1821。而 n32 每条 MMA 的装载数比 64 宽
///    高 33%（8 组 = 4 行组 × 2 列组；每组 16×16 ⇒ 2A+2B 喂 4 条 MMA；
///    64 宽是 16×32 ⇒ 2A+4B 喂 8 条 = 0.75）。**本测试就是要判 n32 是不是
///    慢在这 0.33 上**——若是，处方是把每 simdgroup 的瓦片从 16×16 抬到
///    512 格（n32: 32×16），比值回到 0.75，而且**网格/尾块/数值一字不改**。
///
/// ## 判读（`probe_m3.metal` 头注释有完整版）
///
///   base ≈ hoist                ⇒ 不是装载发射槽 ⇒ n32 的 0.33 不是主因，别改映射；
///   base ≫ hoist（≈ small）     ⇒ 卡在**装载发射槽** ⇒ 抬瓦片值得做；
///   base ≫ small，hoist ≈ base  ⇒ 卡在**流量/延迟** ⇒ 该动 L2 局部性。
///
/// 除 `base` 外**数值全是错的**（诊断件）。`notail` 臂只少发 wg.x 的最后一档，
/// 尾部列不会被写。
#[test]
fn conv8_mma3_load_attribution() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");
    let probes = super::conv_probe::ProbeSet::new(&ctx.inner.device).expect("编译探针");

    // medium det 的真实形状（`QPPOCR_METAL_PLAN_DUMP`），按端到端时间排序。
    // 全是 stride1 + 对称 pad 的内点几何 ⇒ `build_conv_case` 直接支持。
    let cases = [
        ConvCase { label: "Conv.121 240x216 c256->64 k3", h: 240, w: 216, cin: 256, cout: 64, kh: 3, kw: 3 },
        ConvCase { label: "Conv.3   240x216 c128->64 k3", h: 240, w: 216, cin: 128, cout: 64, kh: 3, kw: 3 },
        ConvCase { label: "Conv.111 240x216 c32->32 k7 ", h: 240, w: 216, cin: 32, cout: 32, kh: 7, kw: 7 },
        ConvCase { label: "Conv.114 240x216 c32->32 k5 ", h: 240, w: 216, cin: 32, cout: 32, kh: 5, kw: 5 },
    ];

    for c in &cases {
        let pr = build_conv_case(&ctx.inner.device, c);
        let k = c.kh;
        let ncols = if pr.co4 % 64 == 0 { 64 } else { 32 };
        // 本组的形状全是 co4 ∈ {32,64}，整除/上取整同值；照抄 plan.rs 的式子。
        let nch = pr.co4.div_ceil(ncols);
        let kname = if ncols == 64 { "n_conv8_mma3" } else { "n_conv8_mma3_n32" };
        let inner_w = pr.out_w - (k - 1);
        let nfb = inner_w / 64;
        let tail = inner_w - nfb * 64;
        let nx = nfb + u32::from(tail > 0);
        let nrows = pr.out_h - (k - 1);
        let g_tail = [nx * nch, nrows, 1];
        let g_notail = [nfb * nch, nrows, 1];

        let run = |arm: &str, groups: [u32; 3], reps: u32| -> f64 {
            let pipe = probes.pipe(arm).expect("取管线");
            // `*_half` 是 128 线程/组的（4 个 simdgroup），其余都是 256。
            let tg = if arm.ends_with("half") { 128u64 } else { 256u64 };
            let t = Instant::now();
            submit_compute(&ctx.inner.queue, |enc| {
                for _ in 0..reps {
                    dispatch_with_tg(enc, pipe, &pr.buf, pr.p_off, groups, tg).expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [&pr.buf];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
            t.elapsed().as_secs_f64() * 1000.0
        };

        // 生产内核作参照（同网格、同 arena）。
        let run_prod = |reps: u32| -> f64 {
            let t = Instant::now();
            submit_compute(&ctx.inner.queue, |enc| {
                for _ in 0..reps {
                    ks.dispatch(enc, kname, &pr.buf, pr.p_off, g_tail).expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [&pr.buf];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
            t.elapsed().as_secs_f64() * 1000.0
        };

        // 变体名（NCOLS 前缀 p_m3_ / p_m3n_）× 网格档（带尾 / 不带尾）。
        let pre = if ncols == 64 { "p_m3_" } else { "p_m3n_" };
        let mut arms: Vec<(&str, [u32; 3])> = vec![
            ("base", g_tail),
            ("base", g_notail),
            ("small", g_tail),
            ("hoist", g_tail),
            ("noepi", g_tail),
            ("noepi", g_notail),
        ];
        // 候选：抬每 simdgroup 的瓦片（n32: 16×16→32×16，比值 1.00→0.75；
        // 64 宽: 16×32→32×32，比值 0.75→0.50），代价都是累加器分片翻倍 +
        // 线程组减半（128 线程）。
        arms.push(("half", g_tail));

        let na = arms.len();
        let mut best = vec![f64::INFINITY; na];
        let mut best_prod = f64::INFINITY;
        // 奇偶轮换序（本机先跑的那条快 6~15%）。
        for round in 0..3 {
            let mut order: Vec<usize> = (0..na).collect();
            if round % 2 == 1 {
                order.reverse();
            }
            for a in order {
                let name = format!("{pre}{}", arms[a].0);
                let ms = time_dispatch(&|r| run(&name, arms[a].1, r)).ms;
                best[a] = best[a].min(ms);
            }
            best_prod = best_prod.min(time_dispatch(&|r| run_prod(r)).ms);
        }

        let b_tail = best[0];
        let b_notail = best[1];
        let small = best[2];
        let hoist = best[3];
        let noepi = best[4];
        let gf = |ms: f64| pr.flops / (ms * 1e6);
        // 尾块「算完就丢」的纯度：尾块共 64 列，只有 `inner_w - nfb*64` 列
        // 是有效输出，其余纯冗余。
        let tail_valid = inner_w - nfb * 64;
        let tail_waste = (b_tail - b_notail) * (64 - tail_valid) as f64 / 64.0;
        eprintln!(
            "[mb/m3] {:<28} NCOLS={ncols} 网格 {:?}  生产 {kname} {best_prod:>7.3} ms（{:>6.1} GF）",
            c.label,
            g_tail,
            gf(best_prod)
        );
        eprintln!(
            "[mb/m3]   base {b_tail:>7.3}（{:>6.1} GF）  small {small:>7.3}（{:.2}×）  hoist {hoist:>7.3}（{:.2}×）  noepi {noepi:>7.3}（{:.2}×）",
            gf(b_tail),
            b_tail / small,
            b_tail / hoist,
            b_tail / noepi
        );
        eprintln!(
            "[mb/m3]   尾块：整块 {:.3} ms（带尾 {b_tail:.3} − 不带尾 {b_notail:.3}）；其中有效 {tail_valid}/64 列 ⇒ **纯冗余 {tail_waste:.3} ms（{:.1}%）**",
            b_tail - b_notail,
            100.0 * tail_waste / b_tail
        );
        eprintln!(
            "[mb/m3]   纯 MMA 口径：有用 {:.0} → 除尾块冗余（{nx}×64/{inner_w} 列 = {:.3}×）**{:.0} GFLOPS**",
            gf(b_tail),
            (nx * 64) as f64 / inner_w as f64,
            gf(b_tail) * (nx * 64) as f64 / inner_w as f64
        );
        let half = best[na - 1];
        let half_tile = if ncols == 64 { "32×32" } else { "32×16" };
        eprintln!(
            "[mb/m3]   **候选 half（4 组/128 线程，每 sg {half_tile}）** {half:>7.3} ms（{:>6.1} GF）⇒ **{:.3}×**（vs base）",
            gf(half),
            b_tail / half
        );
    }
}

/// `n_conv8_edge` 的**地板**：同一形状上的三条读数。
///
/// ## 为什么必须单独量这个（2026-10-10，已结案）
///
/// 边框补算那 2.3% 的像素，按 `n_conv8` 的每像素速率只值 ~0.2 ms，而
/// `conv8_mma3_throughput` 量到的 edge 是 **1.43 ms**——模型与读数差 7 倍。
/// 这种分歧只有两种可能，处方完全相反：真边框那 2.3% 比我想的贵得多
/// （处方：把边框也搬进 MMA），或者「第二跳 dispatch + 前导 + 早退」本身
/// 就要这么多（处方：别无脑加 dispatch，或者把前导做便宜）。
///
/// 判据是把「全平面都被覆盖」这一档**造出来**：`padT=padL=0, taps=kW=1`
/// 时覆盖矩形 = 整个输出平面 ⇒ 每个线程组都在早退那一行返回、一次 tap
/// 循环都不进（那时的网格还铺满整张平面）。读数 1.43 ms 里绝大部分落在
/// 这一档 ⇒ **空转本身就是成本**，不是边框像素贵。于是处方是紧凑网格
/// （只铺真正含边框的行组，见 `n_conv8.metal` 文件头），B 读数随之掉到
/// 地板上方一点点。
///
/// 紧凑网格之后这个「全早退」档的**构造方式变了**：内核现在自己按 pads
/// 算 `e_tiles`，把 pads 归零就等于 `tiles = 0` ⇒ 连一个 tile 都不解码，
/// 量到的是**纯启动地板**（网格照铺，循环不进体）。三读数仍然是：
///
///   C = 纯启动地板；  B − C = 真边框 2.3% 的代价；
///   B + MMA vs A = 这一跳到底值不值。
///
/// 顺带打印紧凑网格相对全平面的**压缩比**——这个比值就是当初那 1.43 ms
/// 的旋钮：tile 少掉多少，前导就少掉多少。
#[test]
fn conv8_edge_floor_diag() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    let cases = [
        ConvCase { label: "det 136x240 c256->64 k3", h: 136, w: 240, cin: 256, cout: 64, kh: 3, kw: 3 },
        ConvCase { label: "rec 12x136 c128->128 k3", h: 12, w: 136, cin: 128, cout: 128, kh: 3, kw: 3 },
    ];
    for c in &cases {
        let pr = build_conv_case(&ctx.inner.device, c);
        let run = |name: &str, groups: [u32; 3], reps: u32| -> f64 {
            let t = Instant::now();
            submit_compute(&ctx.inner.queue, |enc| {
                for _ in 0..reps {
                    ks.dispatch(enc, name, &pr.buf, pr.p_off, groups).expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [&pr.buf];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
            t.elapsed().as_secs_f64() * 1000.0
        };
        let base = time_dispatch(&|r| run("n_conv8", pr.groups, r)).ms;

        // B：正常参数的 edge，**紧凑网格**（生产配置）
        let eg = edge_groups(&pr, c.kh);
        let edge = time_dispatch(&|r| run("n_conv8_edge", eg, r)).ms;

        // C：把 padT/padL 归零（只动 2 个参数词）⇒ 内核自算的 e_tiles = 0
        // ⇒ 一个 tile 都不进体。网格仍铺全平面（与旧配置线程数一致），
        // 所以量到的就是「启动 + 空网格」的地板。
        let (all_exit, tiles_c, tiles_all) = {
            // SAFETY: 参数块在 word 64 起（21 词），落在输入段（word 256 起）之前。
            unsafe {
                let p = (pr.buf.contents() as *mut u32).add(pr.p_off as usize);
                *p.add(14) = 0; // padT
                *p.add(15) = 0; // padL
            }
            let t = time_dispatch(&|r| run("n_conv8_edge", pr.groups, r)).ms;
            // C 档下内核自算的 e_tiles：（2*(0*gh) + (outH-0)*2*ceil(0/4)）*nh = 0
            (t, 0u64, (pr.out_h * pr.out_w).div_ceil(4) as u64 * (pr.co4 / 8) as u64)
        };
        // B 档的 tile 数（与 edge_groups 同源，这里只是回读一遍用于打印）
        let gh = (pr.out_w / 4) as u64;
        let gl = ((c.kh / 2 + 3) / 4) as u64;
        let tiles_b = (2 * ((c.kh as u64 / 2) * gh)
            + (pr.out_h - c.kh + 1) as u64 * 2 * gl)
            * (pr.co4 / 8) as u64;
        eprintln!(
            "[mb] edge 地板 {:<24} base {:>7.3} ms | edge 紧凑 {:>7.3}（网格 {:?}）| 空网格 {:>7.3} \
             ⇒ 边框本体 {:>7.3} ms、启动地板 {:>7.3} ms；tile {tiles_b} vs 全平面 {tiles_all} = {:.1}%",
            c.label,
            base,
            edge,
            eg,
            all_exit,
            edge - all_exit,
            all_exit,
            100.0 * tiles_b as f64 / tiles_all as f64,
        );
        let _ = tiles_c;
    }
}

/// 读回输出段（NHWC，`co4` 通道/像素）。
fn read_out(pr: &ConvProbe) -> Vec<f32> {
    // SAFETY: 见 conv_error——输出段完全在 arena 内，调用方已同步。
    unsafe {
        std::slice::from_raw_parts(
            (pr.buf.contents() as *const f32).add(pr.out_off_w as usize),
            (pr.out_h * pr.out_w * pr.co4) as usize,
        )
    }
    .to_vec()
}

/// 把输出段重铺哨兵 7.0（每次「只派一支」的对拍都要先复位）。
fn refill_out_sentinel(pr: &ConvProbe) {
    // SAFETY: 同 read_out；写的是同一段长度。
    unsafe {
        let n = (pr.out_h * pr.out_w * pr.co4) as usize;
        let p = (pr.buf.contents() as *mut f32).add(pr.out_off_w as usize);
        for i in 0..n {
            *p.add(i) = 7.0;
        }
    }
}

/// `n_conv8_mma3` + `n_conv8_edge` 对拍（k×k，f32 MMA）。
///
/// ## 为什么要「一支一支跑」而不是合起来比一次
///
/// 生产里这两支是**同一次 conv 的两半**：MMA 只吃「全内点块」（每块 = 一行
/// 内 64 个连续输出 x，9 个 tap 的读数学上全落在平面内），edge 吃补集。于是
/// 有两类失败都表现为「数值对不上」，但处方完全不同：
///
///   · **一半算错** —— 内核里的 stride/layout 假设错了（`simdgroup_load`
///     的行 stride 语义、A 瓦片的 64 行是不是真的连续、tap→(ky,kx) 解对没）；
///   · **两半重叠** —— 同一个输出像素被写两次，谁赢取决于 dispatch 顺序。
///     这是**静默错**：不报错、不越界、只在特定形状上偶发。
///
/// 只比一次整幅图分不出这两类。所以每次**只派一支**、把另一支的像素要求
/// 「必须还是哨兵 7.0」：pass A 证明「MMA 算得对 + 没碰 border」，pass B 证明
/// 「edge 算得对 + 没碰内点」。两 pass 合起来 = 每个像素恰好被写一次。
///
/// ## 覆盖矩形的定义（与计划层必须逐字一致）
///
///   y_lo = padT,                       y_hi = padT + (outH-(kh-1)) - 1
///   x_lo = padL,                       x_hi = padL + 64*nfb - 1
///   nfb  = (outW-(kw-1)) / 64          （**向下取整，不进位**）
///
/// 计划层那份拷贝在 `plan.rs` 的 conv 分支里；两边对不上就是「一圈没被算」
/// 或「算两遍」。这条耦合是本设计**唯一的**新耦合面，所以这里把公式硬编码
/// 一遍再做交叉验证，而不是从生产代码里 import（那会让错的那份掩盖对的那份）。
#[test]
fn conv8_mma3_matches_cpu() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    // 用例分三族，**同一个断言体跑**：
    //
    //   · **对称 pad + stride1**（前 9 例，历史用例）：走的是旧式覆盖矩形，
    //     作用是钉住「泛化是逐位等价的」。**其中后四例专打 32 通道块那一档**
    //     （`n_conv8_mma3_n32`）：det 的 CSP 三连 Conv.111/114/117 就是
    //     Co=32 Ci=32 落在 240×216 上。
    //   · **stride≠1 / 非对称 pad**（中 5 例）：det 头部三条的真实
    //     形状——`Conv.1/2` 是 2×2 SAME（pads `(0,1,0,1)` ⇒ `gL=0 gR=1`）、
    //     `Conv.3` 是 3×3 stride2。这三条此前被「步长 1 + 对称 pad」两条
    //     判据挡在 MMA 之外，合计 13.3 ms ≈ `n_conv8` 总时的 61%。
    //   · **通道尾块**（`cpad4(Co) % 32 != 0`，中 5 例）：判据从 `%32`
    //     放宽到 `%8` 后新收的 Co=24/40/48 那几档，打的是 B 分片越过权重
    //     行尾之后的落点 + `n < nDim` 掩码。
    //   · **非方形 k**（`kh != kw`，后 6 例，最新）：`kh==kw` 这条旧判据
    //     拆掉后新收的 det 六对（1×7/7×1/5×1/1×5/3×1/1×3）。打的是
    //     `ky = t/kW` 的解码，以及 `kH=1`/`kW=1` 两种**边框退化**——
    //     见用例表里那一段的说明。
    //
    // **全部 outW ≡ 0 (mod 4)**：紧凑网格门禁（行组 = 连续 4 个 x，不跨行
    // 是枚举与平面映射不错位的前提）。对称 pad + stride1 下 `outW≡0 mod 4`
    // 蕴含 `tail = inner_w mod 64` 必为偶数，最小可达尾是 2。
    let mk = |h, w, cin, cout, k, note| {
        let c = ConvCase { label: "mma3", h, w, cin, cout, kh: k, kw: k };
        let g = ConvGeom::sym(&c);
        (c, g, note)
    };
    let strided = |h, w, cin, cout, k, g, note| {
        let c = ConvCase { label: "mma3", h, w, cin, cout, kh: k, kw: k };
        (c, g, note)
    };
    // 非方形：`kh`/`kw` 分开给，pad 由 [`ConvGeom::sym`] 各算一份
    // （`pH = kh/2`、`pW = kw/2`）——正是 det 那六对 1×7/7×1/5×1/1×5/3×1/1×3
    // 的形态（ONNX SAME：kH=1 那对 `padT=padB=0`，kW=1 那对 `padL=padR=0`）。
    let rect = |h, w, cin, cout, kh, kw, note| {
        let c = ConvCase { label: "mma3", h, w, cin, cout, kh, kw };
        let g = ConvGeom::sym(&c);
        (c, g, note)
    };
    let cases: Vec<(ConvCase, ConvGeom, &str)> = vec![
        // ── 对称 pad + stride1（回归）──
        mk(16, 132, 32, 64, 3, "nfb=2 尾2"),
        mk(32, 96, 64, 128, 3, "nfb=1 双通道块 尾30"),
        mk(9, 200, 32, 64, 3, "nfb=3 奇行数 尾6"),
        mk(16, 68, 32, 64, 3, "nfb=1 **最小尾 2 列**"),
        mk(16, 132, 32, 64, 5, "5×5 **无尾块**"),
        mk(16, 132, 32, 32, 3, "**N32** Co=32 尾2"),
        mk(32, 96, 32, 96, 3, "**N32** Co=96=3×32 尾30"),
        mk(16, 68, 32, 32, 3, "**N32** Co=32 最小尾"),
        mk(24, 200, 32, 32, 7, "**N32** Co=32 7×7 尾2"),
        // ── stride≠1 / 非对称 pad ──
        strided(
            64, 132, 64, 64, 2,
            ConvGeom { sh: 1, sw: 1, pt: 0, pl: 1, pb: 1, pr: 0 },
            "**2×2 SAME** pads(0,1,0,1) gL=0 gR=1",
        ),
        strided(
            64, 136, 64, 64, 3,
            ConvGeom { sh: 2, sw: 2, pt: 1, pl: 1, pb: 1, pr: 1 },
            "**3×3 stride2** 对称 pad1 gR=0",
        ),
        strided(
            64, 136, 32, 32, 3,
            ConvGeom { sh: 2, sw: 2, pt: 0, pl: 1, pb: 1, pr: 0 },
            "**N32 3×3 stride2** pads(0,1,1,0)",
        ),
        strided(
            65, 129, 32, 64, 2,
            ConvGeom { sh: 1, sw: 1, pt: 0, pl: 0, pb: 0, pr: 0 },
            "**2×2 无 pad ⇒ 边框空集**",
        ),
        strided(
            67, 143, 32, 64, 4,
            ConvGeom { sh: 2, sw: 2, pt: 1, pl: 1, pb: 2, pr: 2 },
            "**4×4 stride2** 非对称 pads(1,1,2,2)",
        ),
        // ── 通道尾块（`cpad4(Co) % 32 != 0`）——
        //
        // 判据从 `%32` 放宽到 `%8` 后新收进来的形状（det 的 Conv. 里有
        // Co=24/48 那几档）。**这一族打的是 B 分片越过权重行尾之后的落点**：
        // 越读到的是别的 k 的权重或段尾的 `nhwc::MMA_W_TAIL` 个 0，产物只
        // 落进 `n >= nDim` 的列——本族任何一例的 `worst_in` 飘起来，都说明
        // 那条掩码或那段尾垫没兜住。
        //
        // 第一例的第二块只剩 8 列（`40 = 32 + 8`）：本行 B 分片的列偏移最大
        // 24（`sn=16` + `j*8=8`），加 8 列宽 ⇒ 越过行尾 `64 - nDim = 24` 个
        // float，是全族最接近尾垫上限 32 的一档（一般式的上界是 31，见内核
        // 头注释「尾块（N 方向）」）。
        mk(16, 68, 32, 40, 3, "**Co=40=32+8** 第二块只 8 列、越读最深"),
        mk(16, 132, 32, 48, 3, "**Co=48=32+16** 尾块半满"),
        mk(16, 132, 32, 24, 3, "**Co=24** 单块 32 列只填 24"),
        mk(24, 200, 32, 8, 7, "**Co=8** 单块只填 8（最极端不满）"),
        strided(
            64, 136, 32, 48, 3,
            ConvGeom { sh: 2, sw: 2, pt: 1, pl: 1, pb: 1, pr: 1 },
            "**Co=48 + 3×3 stride2** 尾块×步长",
        ),
        // ── 非方形 k（`kh != kw`）——det 那六对的形态 ──
        //
        // 这一族打的三件事，每一件都能单独把结果打歪：
        //   · `ky = t/kW`、`kx = t-ky*kW` 的解码（权重表是 `[tap][k][n]`，
        //     tap 的最快变化维是 **kx 不是 ky**；弄反了错值仍有限、不越界，
        //     只能靠数值对上否掉）；
        //   · **kH=1 那两例**（`padT=padB=0` ⇒ `y_lo=0`、`y_hi=outH-1`）：
        //     内点行铺满整幅图 ⇒ 边框集合退化成「只有左右两条竖带」，
        //     上/下台行数为 0（这是方形族**永远走不到**的一支，方形 k 的
        //     `padT>=1` 让上台恒非空）；
        //   · **kW=1 那两例**（`padL=padR=0` ⇒ `x_lo=0`、`x_hi=outW-1`）：
        //     反向退化——边框只剩上/下两条横带，`gL=gR=0`（中带瓦片数 = 0）。
        //   两例都要求 `outW % 4 == 0` 且 `outW ≥ 64`（列内点至少一个 64 块）。
        rect(64, 132, 32, 64, 1, 7, "**1×7** kH=1 padT=0 上台空、边框只剩左右"),
        rect(64, 132, 32, 64, 7, 1, "**7×1** kW=1 padL=0 中带空、边框只剩上下"),
        rect(64, 136, 32, 64, 3, 1, "**3×1** kW=1 窄列"),
        rect(64, 136, 32, 48, 1, 3, "**1×3 + Co=48** 非方形×通道尾块"),
        rect(64, 200, 32, 32, 5, 1, "**N32 5×1** 非方形×32 列块"),
        rect(64, 200, 32, 32, 1, 5, "**N32 1×5** 非方形×32 列块"),
    ];

    for (c, geom, note) in cases {
        let pr = build_conv_case_geom(&ctx.inner.device, &c, geom, 1, false);
        let (out_h, out_w, co4) = (pr.out_h, pr.out_w, pr.co4);
        // `kh`/`kw` **分开传**（非方形族）：`taps`、覆盖几何、边框网格三处
        // 都只认这一对，内核里 `ky = t/kW`、`kx = t-ky*kW` 也是分开的。
        let (ci4, taps) = (c.cin.div_ceil(4) * 4, c.kh * c.kw);
        // 覆盖几何——**第三份独立拷贝**（见 `kxk_cover_test`）。
        let cv = kxk_cover_test(&pr, c.kh, c.kw);
        // 通道块宽 64/32 二选一——与 plan.rs 那支**同一判据**；`nch` 上取整
        // （`co4 % ncols != 0` 时最后一块不满，见内核头注释「尾块（N 方向）」）。
        let ncols = if co4 % 64 == 0 { 64 } else { 32 };
        let nch = co4.div_ceil(ncols);
        let kname = if ncols == 64 { "n_conv8_mma3" } else { "n_conv8_mma3_n32" };
        assert!(co4 % 8 == 0 && (ci4 % 8) == 0, "用例不满足通道门：co4={co4} ci4={ci4}");

        // 段布局重算（与 build_conv_case_geom 同式）+ 自检：公式漂了就先炸在
        // 这里，而不是让下面读到别的段上。
        let n_in = c.h * c.w * ci4;
        // 权重段 = 数据 + `MMA_W_TAIL` 尾垫（`build_conv_case_geom_w` 里
        // 紧排布局那一支同式）——尾垫是生产侧也有的，漏掉这条自检就漂 32。
        let n_w = taps * ci4 * co4 + qppocr_gpu::nhwc::MMA_W_TAIL as u32;
        let out_off_w = 256u32 + n_in + n_w + co4; // 参数块 64 起、输入 256 起
        assert_eq!(out_off_w, pr.out_off_w, "段布局重算与构造函数不一致");

        let (y_lo, y_hi) = (cv.y_lo, cv.y_hi);
        let (x_lo, x_hi) = (cv.x_lo, cv.x_hi);

        let want = cpu_conv_nhwc_geom(&pr.input, &pr.weight, &c, geom, out_h, out_w);

        // 两 pass 都用 `build_conv_case_geom` 写好的**标准参数块**——本内核与
        // `n_conv8_edge` 共用一套布局（见 n_conv8_mma3.metal 文件头），所以
        // 这里一个参数词都不用碰。这一步本身就是那条共享契约的验证。

        // ── pass A：只派 MMA（含尾块），要求「内点全对 + 边框原封不动」──
        // 尾块真正写出去的列数 = 覆盖端之后再有多少列；无尾块时为 0。
        let tail = x_hi.saturating_sub(x_lo + 64 * cv.nfb - 1);
        let groups = cv_groups(&cv, nch);
        submit_compute(&ctx.inner.queue, |enc| {
            ks.dispatch(enc, kname, &pr.buf, pr.p_off, groups).expect("dispatch");
        })
        .expect("提交");
        let got = read_out(&pr);
        let (mut worst_in, mut stray_out, mut covered) = (0f32, 0u32, 0u32);
        for y in 0..out_h {
            for x in 0..out_w {
                let inside = y >= y_lo && y <= y_hi && x >= x_lo && x <= x_hi;
                if inside {
                    covered += 1;
                }
                for n in 0..c.cout {
                    let g = got[((y * out_w + x) * co4 + n) as usize];
                    let wn = want[((y * out_w + x) * c.cout + n) as usize];
                    if inside {
                        worst_in = worst_in.max((g - wn).abs());
                    } else if g != 7.0 {
                        stray_out += 1;
                    }
                }
            }
        }
        assert!(
            stray_out == 0,
            "n_conv8_mma3 写了覆盖矩形**之外**的 {stray_out} 个元素（{note}）——\
             与 n_conv8_edge 重叠了"
        );
        assert!(
            worst_in < 1e-4,
            "n_conv8_mma3 内点数值不符（{note}）：最大误差 {worst_in:.3e}——\
             simdgroup_load 的 A 瓦片/权重行 stride 假设错了"
        );

        // ── pass B：只派 edge（**紧凑网格**），要求「边框全对 + 内点原封不动」──
        // 边框是**空集**时生产不发这条 dispatch（空网格没定义，见 plan.rs 那一支），
        // 这里也必须跳过——否则测的就不是生产行为。
        let eg = edge_groups_geom(&pr, c.kh, c.kw);
        refill_out_sentinel(&pr);
        if eg[0] != 0 {
            submit_compute(&ctx.inner.queue, |enc| {
                ks.dispatch(enc, "n_conv8_edge", &pr.buf, pr.p_off, eg).expect("dispatch");
            })
            .expect("提交");
        }
        let got = read_out(&pr);
        let (mut worst_out, mut stray_in) = (0f32, 0u32);
        for y in 0..out_h {
            for x in 0..out_w {
                let inside = y >= y_lo && y <= y_hi && x >= x_lo && x <= x_hi;
                for n in 0..c.cout {
                    let g = got[((y * out_w + x) * co4 + n) as usize];
                    let wn = want[((y * out_w + x) * c.cout + n) as usize];
                    if inside {
                        if g != 7.0 {
                            stray_in += 1;
                        }
                    } else {
                        worst_out = worst_out.max((g - wn).abs());
                    }
                }
            }
        }
        assert!(
            stray_in == 0,
            "n_conv8_edge 写了覆盖矩形**之内**的 {stray_in} 个元素（{note}）——\
             MMA 的覆盖矩形与 edge 的跳过谓词不自洽"
        );
        assert!(
            worst_out < 1e-4,
            "n_conv8_edge 边框数值不符（{note}）：最大误差 {worst_out:.3e}"
        );

        let border = out_h * out_w - covered;
        eprintln!(
            "[mb] conv8_mma3 对拍 {note:<34} ({out_h}×{out_w} c{cin}->{cout} k{kh}×{kw} s{sh} 尾 {tail}): \
             内点 {covered} px 误差 {worst_in:.3e}、边框 {border} px 误差 {worst_out:.3e}、\
             越界写 0/0｜edge 网格 {} → {} 组（紧凑/全平面）",
            pr.groups[0], eg[0],
            cin = c.cin,
            cout = c.cout,
            kh = c.kh,
            kw = c.kw,
            sh = geom.sh,
        );
    }
}

/// 频率爬坡探针：同一条内核连跑不同次数，看**单次时间**随累计时长的变化。
///
/// ## 为什么怀疑它
///
/// `fma_peak_reference` 报的「峰值」只有 430–760 GFLOPS——比 M4 十大核
/// GPU 常见的标称值（~4 TFLOPS）低一个数量级，而且**随网格变大而升高**
/// （13 组 461 → 2560 组 762）。算力不该跟网格大小挂钩，除非我们量到的
/// 是别的量。
///
/// 一个自洽的解释：Apple GPU 空闲时降频，跑起来才逐步提频。我们的每个
/// dispatch 只有 0.1–1 ms——如果提频需要几十毫秒，那**所有真实内核都跑
/// 在爬坡段**，量到的「算力」是提频曲线的下半段，而不是稳态算力。
///
/// 判据：如果 `单次时间` 随 `reps` 增大而**显著下降**，就是爬坡；
/// 稳态值才是真算力。如果平，那 700 GFLOPS 就是这块芯片的实价，
/// 后面所有优化目标都得按它重算。
#[test]
fn clock_ramp_probe() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let probes = super::conv_probe::ProbeSet::new(&ctx.inner.device).expect("编译探针");
    let c = ConvCase {
        label: "rec 48x544 c32->32 k3",
        h: 48,
        w: 544,
        cin: 32,
        cout: 32,
        kh: 3,
        kw: 3,
    };
    let pr = build_conv_case(&ctx.inner.device, &c);
    let pipe = probes.pipe("p_base").expect("管线");
    let run = |reps: u32| -> f64 {
        let t = Instant::now();
        submit_compute(&ctx.inner.queue, |enc| {
            for _ in 0..reps {
                super::pipeline::dispatch_with(enc, pipe, &pr.buf, pr.p_off, pr.groups)
                    .expect("dispatch");
                let res: [&metal::ResourceRef; 1] = [&pr.buf];
                enc.memory_barrier_with_resources(&res);
            }
        })
        .expect("提交");
        t.elapsed().as_secs_f64() * 1000.0
    };
    let _ = run(16);
    eprintln!("[mb] ── 频率爬坡：p_base on {} ──", c.label);
    let mut prev = f64::NAN;
    for reps in [16u32, 64, 256, 1024, 4096] {
        let t = run(reps) / reps as f64;
        let gflops = pr.flops / (t * 1e6);
        let delta = if prev.is_nan() { 0.0 } else { (t / prev - 1.0) * 100.0 };
        eprintln!(
            "[mb]   reps={reps:<5} 单次 {t:>7.3} ms  {gflops:>7.1} GFLOPS  \
             （较上一档 {delta:+.1}%）"
        );
        prev = t;
    }
}

/// 算力标尺：这块 M4 在**当前 dispatch 形状下**真实的 fp32 峰值（GFLOPS）。
///
/// 内核里没有任何访存/分支/索引，只有互相独立的 `fma`（4 或 16 个
/// float4 累加器）。它给出分母——没有它，「160 GFLOPS」既可能是内核烂，
/// 也可能是标称的 4 TFLOPS 根本不适用。
///
/// **必须是同一轮里的分母**：这台机器的 GPU 与 WindowServer 共享，
/// 环境负载能让同一内核差 7 倍。绝对 GFLOPS 因此不可跨轮比较——把峰值
/// 和被测对象放在同一次运行里量，两边一起被环境缩放，比值才站得住。
fn measure_fma_peak(
    ctx: &super::MetalContext,
    probes: &super::conv_probe::ProbeSet,
    groups: u32,
    nacc: u64,
) -> f64 {
    let buf = ctx
        .inner
        .device
        .new_buffer(4096, metal::MTLResourceOptions::StorageModeShared);
    const P_OFF: u32 = 64;
    let name = if nacc == 4 { "p_peak4" } else { "p_peak16" };
    let pipe = probes.pipe(name).expect("峰值管线");
    // 一条命令缓冲里跑 reps 次，每次 iters 轮 FMA
    let run = |iters: u32, reps: u32| -> f64 {
        // SAFETY: 缓冲 4096 字节、Shared 存储、基址非空；只写 word 64 一个 u32。
        unsafe {
            *(buf.contents() as *mut u32).add(P_OFF as usize) = iters;
        }
        let t = Instant::now();
        submit_compute(&ctx.inner.queue, |enc| {
            for _ in 0..reps {
                super::pipeline::dispatch_with(enc, pipe, &buf, P_OFF, [groups, 1, 1])
                    .expect("dispatch");
                let res: [&metal::ResourceRef; 1] = [&buf];
                enc.memory_barrier_with_resources(&res);
            }
        })
        .expect("提交");
        t.elapsed().as_secs_f64() * 1000.0
    };
    // 粗调 iters 到 ≥30 ms，再取多轮最小（稳态，避开提频段）
    let mut iters: u32 = 256;
    loop {
        let t = run(iters, 1);
        if t >= 30.0 || iters >= (1 << 22) {
            break;
        }
        iters = ((iters as f64) * (30.0 / t.max(1e-3)).clamp(1.5, 8.0)) as u32 + 1;
    }
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        best = best.min(run(iters, 4) / 4.0);
    }
    let flops = groups as f64 * 256.0 * iters as f64 * nacc as f64 * 4.0 * 2.0;
    flops / (best * 1e6)
}

/// 打印这块 M4 的 fp32 峰值随网格/累加器数的变化。
#[test]
fn fma_peak_reference() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let probes = super::conv_probe::ProbeSet::new(&ctx.inner.device).expect("编译探针");
    for (nacc, tag) in [(4u64, "4 个 float4 累加器（16 寄存器）"), (16, "16 个（64 寄存器）")] {
        for groups in [13u32, 102, 810, 2560] {
            let gflops = measure_fma_peak(&ctx, &probes, groups, nacc);
            eprintln!("[mb] 峰值 {tag:<32} grid={groups:<5} {gflops:>8.1} GFLOPS");
        }
    }
}

/// 变体二分：同一形状、同一 arena，只换内核，量差异。
///
/// ## 为什么是**轮转交错**而不是「测完一个再测下一个」
///
/// 这台机器上 GPU 是和 WindowServer 共享的（实测 Finder 能飙到 115% CPU，
/// load average 5.6），环境噪声让**同一个内核两次测量差 4 倍以上**。
/// 串行测变体等于把环境漂移当成变体差异——`p_base` 与真正的 `n_conv8`
/// 就出现过方向都翻转的对比。轮转让同一轮的三个变体分摊同一段环境，
/// 每轮取最小再抹掉突刺。
///
/// 输出：时间、GFLOPS、占**同轮峰值**的百分比、与 CPU 的误差。
///
/// `p_nocoal`/`p_fmaonly`/`p_wconst` 三个诊断变体故意算错，误差必然很大；
/// 其余（`p_base`/`p_r8`/`p_m8n4`）必须 < 1e-4，否则「变快了」是假象。
#[test]
fn conv8_variant_scan() {
    let Some(ctx) = open_or_skip() else { return };
    let probes = super::conv_probe::ProbeSet::new(&ctx.inner.device).expect("编译探针");
    let cases = [
        // rec 的 **CTC 分类层**（`MatMul.12`）：整图 prof 里单条 27–41% 的
        // 那个 dispatch。形状按 [8,27,120]→[8,27,18710] 折批后取：
        // m=8×27=216、cin=120、cout 补到 4 的倍数（18710→18712）、k=1。
        // 权重 9 MB 而输出 16 MB —— 是**权重流**型，不是算力型。
        ConvCase { label: "rec ctc head 216x1 c120->18710 k1", h: 216, w: 1, cin: 120, cout: 18712, kh: 1, kw: 1 },
        ConvCase { label: "rec 48x544 c32->32 k3 (full res)", h: 48, w: 544, cin: 32, cout: 32, kh: 3, kw: 3 },
        ConvCase { label: "rec 24x272 c64->64 k3", h: 24, w: 272, cin: 64, cout: 64, kh: 3, kw: 3 },
        ConvCase { label: "det 480x432 c32->32 k3", h: 480, w: 432, cin: 32, cout: 32, kh: 3, kw: 3 },
        ConvCase { label: "rec 6x68 c256->256 k1", h: 6, w: 68, cin: 256, cout: 256, kh: 1, kw: 1 },
        // ── 补齐「生产形状的 1×1」这一档（2026-10-08）──────────────────
        //
        // 上面 5 个 case 里**没有一个是 det 尺寸的 1×1**，而 1×1 占了 det 的
        // n_conv8 dispatch 的 98%（702 条 taps=1 对 31 条 taps=9），rec 头
        // 81% 的设备时间也全在 1×1 上。没有这档，扫描出的结论对大头无效。
        //
        // 形状**逐字取自 `QPPOCR_METAL_PLAN_DUMP`**（img-004，small 模型），
        // 不是编的：以下分别是 det / rec 头里 m 与 cin×co 最大的那几层。
        // 实测达成率：det 侧 1×1 约 180–800 GFLOPS，也就是峰值的 5~23%。
        ConvCase { label: "det 34x60 c192->384 k1 (×4 层)", h: 34, w: 60, cin: 192, cout: 384, kh: 1, kw: 1 },
        ConvCase { label: "det 34x60 c384->192 k1 (×4 层)", h: 34, w: 60, cin: 384, cout: 192, kh: 1, kw: 1 },
        ConvCase { label: "det 17x30 c384->768 k1", h: 17, w: 30, cin: 384, cout: 768, kh: 1, kw: 1 },
        ConvCase { label: "rec 6x208 c192->384 k1 (热)", h: 6, w: 208, cin: 192, cout: 384, kh: 1, kw: 1 },
        ConvCase { label: "rec 3x208 c384->768 k1 (热)", h: 3, w: 208, cin: 384, cout: 768, kh: 1, kw: 1 },
        ConvCase { label: "rec 3x208 c768->384 k1 (热)", h: 3, w: 208, cin: 768, cout: 384, kh: 1, kw: 1 },
        // 真正的 CTC 分类层是 m=104（上面那条 216 是折批后的），cin 也小：
        // 它是**权重大、算力小**的极端，和卷积的瓶颈未必同一个。
        ConvCase { label: "rec ctc head 104x1 c120->18710 k1", h: 104, w: 1, cin: 120, cout: 18712, kh: 1, kw: 1 },
        // ── 权重尺寸阶梯（2026-10-08）────────────────────────────────
        //
        // 生产里 taps==1 的权重段从 1 KB 到 8771 KB，而**窗口序的门**
        // （`cin4v*nv >= 4096`，即权重 ≥ 256 KB）把 288 KB 及以上全塞进
        // 窗口序。实测 288 KB 上窗口序只值 5%、列主序值 25%；8771 KB 上
        // 反过来。**中间那几档一条都没测过**，门槛该划在哪儿无从谈起。
        //
        // 这条 576 KB（384×384）是补中间档——生产里 384x384 有 14 条、
        // 576 KB 那档 3 条、640 KB 那档 3 条。有了它才能判 `p_colmajw`
        // （窗口序+列主序叠加）是不是**两端都不劣**，从而免掉阈值。
        ConvCase { label: "rec 6x208 c384->384 k1 (576K 中档)", h: 6, w: 208, cin: 384, cout: 384, kh: 1, kw: 1 },
        // ── 1–4 MB 权重档（2026-10-09）：**门下那段空白** ────────────────
        //
        // 窗口序的门（`cin4v*nv >= 65536`，即权重 ≥ 4 MB）当初划在 4 MB，
        // 理由是「生产里 1152 KB 与 8771 KB 之间没有别的权重档」。那句话
        // 对 tiny/small 成立，**对 medium 是错的**：`QPPOCR_METAL_PLAN_DUMP`
        // 数出来 medium det 有 **37 条**落在 1–4 MB，合计 **21.65 GFLOP =
        // 全 det 的 25.2%**——它们现在**全走列主序，从没进过窗口序**。
        //
        // 而它们恰恰是 f32 下最慢的一档（`conv8_throughput` 同轮实测 516
        // GFLOPS，而 3×3 那批 1148）。形状逐字取自计划：
        // `out=[1,512,34,60]`（m=2040，grid=[128,1,1]）。
        //
        // **判读（2026-10-09 实测，假设被推翻）**：`p_colmajw` 在这三条上
        // 分别 1.964 / 4.188 / 3.668，而 `p_colmaj` 1.935 / 4.070 / 3.576
        // ——**窗口序一致慢 1.5~2.9%**，`p_window` 更差（4.221）。所以门
        // 虽然理由（那句话）是错的，**它给出的决定是对的，不该动**。
        //
        // 真正有价值的是同轮的诊断列：把权重地址掩进 1 KB（`p_wsmall`，
        // 指令条数不变、流量归零）只快 **1.27~1.32×**，而 `p_fmaonly`
        // （去掉全部 load）快 **3.2~3.4×**。⇒ 这一档**不是权重带宽瓶颈，
        // 是 load 指令发射瓶颈**；带宽类手段（窗口序、LDS 中转、fp16 的
        // 字节）在这里全都撞同一堵墙——fp16 的 1.6× 里有近一半来自指令数
        // 以外的东西，这也解释了它为何显著超过权重的字节收益。
        // 下一根真正的杠杆是**减少 load 指令条数**（更宽的 load / 输入复用），
        // 不是提高局部性。
        ConvCase { label: "det 34x60 c512->512 k1 (1.05M 门下)", h: 34, w: 60, cin: 512, cout: 512, kh: 1, kw: 1 },
        ConvCase { label: "det 34x60 c512->1024 k1 (2.10M 门下)", h: 34, w: 60, cin: 512, cout: 1024, kh: 1, kw: 1 },
        ConvCase { label: "det 34x60 c1024->512 k1 (2.10M 门下)", h: 34, w: 60, cin: 1024, cout: 512, kh: 1, kw: 1 },
    ];
    const ROUNDS: usize = 3;
    for c in &cases {
        let pr = build_conv_case(&ctx.inner.device, c);
        // `p_whalf` 用自己的 arena：**只有权重段存 fp16**，其余段（输入/
        // bias/输出/参数块）与 `pr` 逐位相同、偏移也相同。于是两版之间
        // 唯一的自变量就是权重足迹，见 `build_conv_case_half`。
        let prh = build_conv_case_half(&ctx.inner.device, c);
        let arenas = [&pr, &prh];
        let m_dim = pr.out_h * pr.out_w;

        // 同一轮、同一网格下量分母。峰值随网格大小变化（小网格铺不满
        // 核），所以必须用**本 case 的网格**量——拿全局最大值当分母会把
        // 小层的达成率算低。
        let peak = measure_fma_peak(&ctx, &probes, pr.groups[0], 16);

        // 组各变体的 (名, 行数, 管线, 网格)
        let mut plans = Vec::new();
        for v in super::conv_probe::VARIANTS {
            // 按形状跳过不适用（只对 1×1 成立的瓦片）：照跑会早退成
            // 0.002 ms，看着像三千倍加速——错的方向比没数还糟。
            if v.k1_only && !(c.kh == 1 && c.kw == 1) {
                continue;
            }
            let groups = super::conv_probe::groups_for(m_dim, pr.co4, v.rows, v.nblk);
            // 控制关：p_base 的网格必须与 n_conv8 计划给的逐位相同。
            // 这是「量法与生产对齐」的唯一证据——对不上，后面所有
            // 变体差异都无从归因。（p_r8 行数不同，网格本就不同。）
            if v.is_control {
                assert_eq!(
                    groups, pr.groups,
                    "控制组网格与 n_conv8 不一致（{:?} vs {:?}）——线程覆盖不同，比不了",
                    groups, pr.groups
                );
            }
            // `p_whalf` 读自己的 arena（fp16 权重），其余全走 `pr`。
            let ai = usize::from(v.name == "p_whalf");
            plans.push((v.name, v.tile, v.rows, probes.pipe(v.name).expect("管线"), groups, ai));
        }

        // 一条命令缓冲里跑 reps 次，返回总墙钟（毫秒）
        let run = |arena: &metal::Buffer,
                   p_off: u32,
                   pipe: &metal::ComputePipelineState,
                   groups: [u32; 3],
                   reps: u32|
         -> f64 {
            let t = Instant::now();
            submit_compute(&ctx.inner.queue, |enc| {
                for _ in 0..reps {
                    super::pipeline::dispatch_with(enc, pipe, arena, p_off, groups)
                        .expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [arena];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
            t.elapsed().as_secs_f64() * 1000.0
        };

        // 预热 + 逐变体校准重复次数。误差也在这一步就地取——变体一个接
        // 一个写同一块输出，跑到最后再读只会读到最后一个变体的结果
        // （踩过：三行都印同一个 `p_nocoal` 的误差，把「结果正确」的证据
        // 一起弄丢了）。
        let mut reps = Vec::with_capacity(plans.len());
        let mut errs = Vec::with_capacity(plans.len());
        for (_, _, _, pipe, groups, ai) in &plans {
            let a = arenas[*ai];
            let _ = run(&a.buf, a.p_off, pipe, *groups, 1);
            errs.push(conv_error(a, c));
            reps.push(calibrate(&|r| run(&a.buf, a.p_off, pipe, *groups, r)));
        }

        // 轮转交错取最小
        let mut best = vec![f64::INFINITY; plans.len()];
        for _ in 0..ROUNDS {
            for (i, (_, _, _, pipe, groups, ai)) in plans.iter().enumerate() {
                let a = arenas[*ai];
                let t = run(&a.buf, a.p_off, pipe, *groups, reps[i]) / reps[i] as f64;
                best[i] = best[i].min(t);
            }
        }

        eprintln!("[mb] ── {} ──（同轮峰值 {peak:.0} GFLOPS）", c.label);
        for (i, (name, tile, rows, _, groups, ai)) in plans.iter().enumerate() {
            let ms = best[i].max(1e-6);
            // 分母一律取**该变体自己那份 arena** 的计数：`p_whalf` 的权重
            // 只占一半字节，拿 `pr.bytes` 会把它算成「白赚一倍带宽」。
            let base = arenas[*ai];
            let gflops = base.flops / (ms * 1e6);
            // 达成带宽（下界流量 / 时间）——权重流型的层（如 CTC 分类）
            // 光看 GFLOPS 会误判成「算力差 20 倍」，其实卡在 DRAM 上。
            let gbs = base.bytes / (ms * 1e3);
            eprintln!(
                "[mb]   {name:<9} {tile:<6} rows(grid)={rows} grid={:<10} {ms:>7.3} ms  \
                 {gflops:>7.1} GFLOPS  = {:.0}% 峰值  {gbs:>6.1} GB/s(下界)  误差 {:.2e}",
                format!("{:?}", groups),
                gflops / peak * 100.0,
                errs[i]
            );
        }
    }
}

/// 真实层形状的达成 FLOPS，**f32 与 fp16 两条生产入口同轮对照**。
///
/// 两条入口（`n_conv8` / `n_conv8h`）读同一个逻辑权重、同一个网格、同一份
/// 输入输出；`build_conv_case_half` 只把权重段写成 fp16。所以比值就是
/// 「0.5 → 0.25 B/FLOP」在这一形状上的兑现程度。
///
/// 判读：若某形状 fp16 只值 ~1.2×，说明它本来就不是被权重带宽卡的
/// （权重早已 L2 常驻 / 算术受限）；值接近 2× 的形状才是真正吃这一刀的。
/// 这两类在后端里要分开对待，不能拿平均比值去外推。
#[test]
fn conv8_throughput() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");
    // 形状取自 rec(48×W) 与 det(960×864) 实际出现的层。
    let cases = [
        // rec 前段：全分辨率、3×3，通道少
        ConvCase { label: "rec 48x544 c32->32 k3 (full res)", h: 48, w: 544, cin: 32, cout: 32, kh: 3, kw: 3 },
        ConvCase { label: "rec 24x272 c64->64 k3", h: 24, w: 272, cin: 64, cout: 64, kh: 3, kw: 3 },
        ConvCase { label: "rec 24x272 c64->64 k1", h: 24, w: 272, cin: 64, cout: 64, kh: 1, kw: 1 },
        ConvCase { label: "rec 12x136 c128->128 k3", h: 12, w: 136, cin: 128, cout: 128, kh: 3, kw: 3 },
        // det 前段
        ConvCase { label: "det 480x432 c32->32 k3", h: 480, w: 432, cin: 32, cout: 32, kh: 3, kw: 3 },
        ConvCase { label: "det 240x216 c64->64 k3", h: 240, w: 216, cin: 64, cout: 64, kh: 3, kw: 3 },
        ConvCase { label: "det 120x108 c128->128 k3", h: 120, w: 108, cin: 128, cout: 128, kh: 3, kw: 3 },
        // 大通道 1×1（rec 头常见）
        ConvCase { label: "rec 6x68 c256->256 k1", h: 6, w: 68, cin: 256, cout: 256, kh: 1, kw: 1 },
        // ── medium det 的**重头层**（`QPPOCR_METAL_PLAN_DUMP` 实测的 FLOP
        // 前几名，2026-10-09）。medium 与小档的差距（我们 4.9× vs 竞品 2.0×）
        // 全在这批层的效率上，而它们**一个都没进过上面的名单**。
        //
        // 形状从 dump 的 `out=[1,co,H,W]` 抄，**不是**输入图分辨率：
        // 960×864 的图在 det 骨干里是 `m = 136*240 = 32640`（PaddleOCR 的
        // 下采样把 H/W 各砍到 1/4），照图像尺寸写会把 FLOP 放大 16 倍。
        ConvCase { label: "det m32640 c256->64  k3 (9.6G 首名)", h: 136, w: 240, cin: 256, cout: 64, kh: 3, kw: 3 },
        ConvCase { label: "det m32640 c128->64  k3 (4.8G)", h: 136, w: 240, cin: 128, cout: 64, kh: 3, kw: 3 },
        ConvCase { label: "det m32640 c32->32   k7 (4.3G)", h: 136, w: 240, cin: 32, cout: 32, kh: 7, kw: 7 },
        ConvCase { label: "det m32640 c128->256 k1 (2.1G)", h: 136, w: 240, cin: 128, cout: 256, kh: 1, kw: 1 },
        ConvCase { label: "det m130560 c64->32  k2 (2.1G)", h: 272, w: 480, cin: 64, cout: 32, kh: 2, kw: 2 },
        ConvCase { label: "det 34x60  c1024->512 k1 (1.05G 门下)", h: 34, w: 60, cin: 1024, cout: 512, kh: 1, kw: 1 },
        ConvCase { label: "det 34x60  c512->1024 k1 (1.05G 门下)", h: 34, w: 60, cin: 512, cout: 1024, kh: 1, kw: 1 },
    ];
    for c in &cases {
        let pr = build_conv_case(&ctx.inner.device, c);
        let prh = build_conv_case_half(&ctx.inner.device, c);
        assert_eq!(pr.groups, prh.groups, "两臂网格不一致（{}）", c.label);
        // 重复外推：同一条内核跑 R 次，斜率 = 单次时间（消掉命令缓冲固定开销）
        let run = |pr: &ConvProbe, name: &str, reps: u32| -> f64 {
            let t = Instant::now();
            submit_compute(&ctx.inner.queue, |enc| {
                for _ in 0..reps {
                    ks.dispatch(enc, name, &pr.buf, pr.p_off, pr.groups)
                        .expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [&pr.buf];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
            t.elapsed().as_secs_f64() * 1000.0
        };
        // 奇偶轮换序（本机先跑的那条快 6~15%，固定序的 A/B 等于没量）
        let (mut a, mut b) = (f64::INFINITY, f64::INFINITY);
        for i in 0..3 {
            let (x, y) = if i % 2 == 0 {
                (time_dispatch(&|r| run(&pr, "n_conv8", r)), time_dispatch(&|r| run(&prh, "n_conv8h", r)))
            } else {
                let y = time_dispatch(&|r| run(&prh, "n_conv8h", r));
                let x = time_dispatch(&|r| run(&pr, "n_conv8", r));
                (x, y)
            };
            a = a.min(x.ms);
            b = b.min(y.ms);
        }
        let gf = |ms: f64| pr.flops / (ms * 1e6);
        eprintln!(
            "[mb] {:<38} grid={:<12} f32 {:>7.3} ms ({:>7.1} GFLOPS) → \
             fp16 {:>7.3} ms ({:>7.1} GFLOPS) = {:.2}×",
            c.label,
            format!("{:?}", pr.groups),
            a,
            gf(a),
            b,
            gf(b),
            a / b
        );
    }
}

/// `n_conv8` vs `n_conv8_mma` 的孤立吞吐对比（**只跑 MMA 启用条件下的 1×1 层**）。
///
/// 形状取自 `QPPOCR_METAL_PLAN_DUMP` 实测的 det/rec 重头 1×1 层（见
/// `conv8_throughput` 名单里的 k1 项）。两臂用奇偶轮换序 + 取最小轮，
/// 抵消本机「先跑的那条快 6~15%」的次序效应。
#[test]
fn conv8_mma_throughput() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    let cases = [
        // det medium 重头 1×1（2.1 GFLOP，门内）
        ConvCase { label: "det m32640 c128->256 k1", h: 136, w: 240, cin: 128, cout: 256, kh: 1, kw: 1 },
        // det 34x60 门下（1.05 GFLOP 各）
        ConvCase { label: "det 34x60 c1024->512 k1", h: 34, w: 60, cin: 1024, cout: 512, kh: 1, kw: 1 },
        ConvCase { label: "det 34x60 c512->1024 k1", h: 34, w: 60, cin: 512, cout: 1024, kh: 1, kw: 1 },
        // det 全分辨率 k1（m=32640，通道小）
        ConvCase { label: "det m32640 c32->64 k1", h: 136, w: 240, cin: 32, cout: 64, kh: 1, kw: 1 },
        // rec 1×1（行批次，m 小）
        ConvCase { label: "rec 6x68 c256->256 k1", h: 6, w: 68, cin: 256, cout: 256, kh: 1, kw: 1 },
        ConvCase { label: "rec 6x208 c192->384 k1", h: 6, w: 208, cin: 192, cout: 384, kh: 1, kw: 1 },
        // rec 的 CTC 分类头（`MatMul.12`，out=[1,147,18710]，cin=192 co=18712）：
        // rec 设备端 `n_conv8` 的 87%（剖析值 41.5/47.8 ms），单条最大 1.886 ms。
        // **真实 co=18712 掉在 1×1 MMA 门外**（18712 % 64 = 24）⇒ 落回 `n_conv8`。
        // 这里把 co 补到 18816（多 0.55% 的活）让两臂同台 —— 量的正是
        // 「这一个节点搬进 MMA 能拿回多少」，也是放宽门禁前的定价。
        ConvCase { label: "rec head m147 c192->18712", h: 147, w: 1, cin: 192, cout: 18816, kh: 1, kw: 1 },
        // ── `Conv.54/55/57/58` 的形状对（2026-10-11 加）───────────────────
        //
        // 生产里这两对是**互为镜像**的 1×1 层：同样 `2·m·cin·co` FLOP，
        // 只是把 K 与 N 的角色对调。实测（img-002 medium 那份 185 条计划，
        // 逐条剖析，`QPPOCR_GPU_KERNEL_PROF_ALL`）：
        //   Conv.54 c768→1536 m1008 0.959 ms → 2480 GFLOPS
        //   Conv.55 c1536→768 m1008 1.429 ms → 1664
        //   Conv.57 c768→1536 m1008 1.054 ms → 2256
        //   Conv.58 c1536→768 m1008 1.881 ms → 1264
        // 而两条的**参数块逐字节相同**（只差偏移），所以这不是尾声、不是形状
        // 之外的东西——本对探针就是问：这个 1.5~1.9× 在孤立环境里复现吗。
        // 复现 ⇒ 是这条形状本身的性质，与谁挨着谁无关。
        ConvCase { label: "pair c768->1536 m1008 k1", h: 21, w: 48, cin: 768, cout: 1536, kh: 1, kw: 1 },
        ConvCase { label: "pair c1536->768 m1008 k1", h: 21, w: 48, cin: 1536, cout: 768, kh: 1, kw: 1 },
    ];

    for c in &cases {
        let pr = build_conv_case(&ctx.inner.device, c);
        // MMA 网格与 n_conv8 完全不同（见内核头注释）。
        let mma_groups = mm_sg_groups(c.h * c.w, pr.co4);
        // 契约检查：不满足就该跳过（计划层不会派发这种形状）。
        assert!(c.cin % 8 == 0 && pr.co4 % 64 == 0, "{} 不满足 MMA 启用条件", c.label);

        let run = |pr: &ConvProbe, name: &str, groups: [u32; 3], reps: u32| -> f64 {
            let t = Instant::now();
            submit_compute(&ctx.inner.queue, |enc| {
                for _ in 0..reps {
                    ks.dispatch(enc, name, &pr.buf, pr.p_off, groups)
                        .expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [&pr.buf];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
            t.elapsed().as_secs_f64() * 1000.0
        };
        let (mut a, mut b) = (f64::INFINITY, f64::INFINITY);
        for i in 0..3 {
            let (x, y) = if i % 2 == 0 {
                (
                    time_dispatch(&|r| run(&pr, "n_conv8", pr.groups, r)),
                    time_dispatch(&|r| run(&pr, "n_conv8_mma", mma_groups, r)),
                )
            } else {
                let y = time_dispatch(&|r| run(&pr, "n_conv8_mma", mma_groups, r));
                let x = time_dispatch(&|r| run(&pr, "n_conv8", pr.groups, r));
                (x, y)
            };
            a = a.min(x.ms);
            b = b.min(y.ms);
        }
        let gf = |ms: f64| pr.flops / (ms * 1e6);
        eprintln!(
            "[mb] {:<26} base {:>7.3} ms ({:>6.1} GFLOPS) → mma {:>7.3} ms ({:>6.1} GFLOPS) = {:.2}×  [grid {:?} → {:?}]",
            c.label, a, gf(a), b, gf(b), a / b, pr.groups, mma_groups
        );
    }
}

/// `n_conv8_mma` 的**纯地址扫描**：形状、张量内容、参数块字段全不动，只把
/// 输出段在缓冲里往后挪。
///
/// ## 要回答的问题
///
/// 生产里 `Conv.55` 与 `Conv.58` 的形状/网格/**参数块逐字节相同**（`in/w/r/out`
/// 四个偏移词以外全部相等，已用 `QPPOCR_METAL_PLAN_DUMP_FULL` 逐字对过），
/// 却稳定差 1.3×（img-002 medium 那份 185 条计划，逐条剖析；两天、两趟复现）。
/// 两者**唯一**的结构差别是地址排布：13 份计划里一条不差地
///   `Conv.55`：`out_off − r_off` = `m·co`（残差与输出紧邻）
///   `Conv.58`：`out_off − r_off` = `2.67·m·co`（隔开）
/// 若地址真能摆动吞吐，本探针就该复现同样的量级；若摆不动，那条 1.3× 就
/// 只能归给测量上下文（重放序、arena 状态），与内核和其参数无关。
///
/// ## 读法
///
/// 各档**交错**跑（偶数轮升序、奇数轮降序），取每档最小——本机「一轮里先跑
/// 的那条快 6~15%」（见 `ab-order-effect-swamps-small-gains`），沿一个方向扫
/// 会把漂移记进斜率。离散度本身就是读数：各档若一致，说明地址不是自变量。
///
/// **每一档都重开一个 buffer**（偏移变了，缓冲总长随之变，基址由驱动给），
/// 所以这里量的是「段落在这块缓冲的哪里」，不是绝对物理地址——后者探针控
/// 不了，别把结论说过头。
#[test]
fn conv8_mma_layout_gap() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    // 与生产那两条同形：`pair c1536->768 m1008`。间隔按 word 给。
    let slow = ConvCase { label: "c1536->768", h: 21, w: 48, cin: 1536, cout: 768, kh: 1, kw: 1 };
    let mirror = ConvCase { label: "c768->1536", h: 21, w: 48, cin: 768, cout: 1536, kh: 1, kw: 1 };
    // 786432 word = 3 MB = 本形状的 `m·co`；`Conv.55/58` 的差正好是 1.67 个它。
    let gaps: [u32; 6] = [0, 262144, 786432, 2097152, 4194304, 8388608];

    let run = |pr: &ConvProbe, reps: u32| -> f64 {
        let t = Instant::now();
        submit_compute(&ctx.inner.queue, |enc| {
            for _ in 0..reps {
                ks.dispatch(enc, "n_conv8_mma", &pr.buf, pr.p_off, mm_sg_groups(21 * 48, pr.co4))
                    .expect("dispatch");
                let res: [&metal::ResourceRef; 1] = [&pr.buf];
                enc.memory_barrier_with_resources(&res);
            }
        })
        .expect("提交");
        t.elapsed().as_secs_f64() * 1000.0
    };

    let probes: Vec<ConvProbe> = gaps
        .iter()
        .map(|&gp| build_conv_case_gap(&ctx.inner.device, &slow, gp))
        .collect();
    let mut best = vec![f64::INFINITY; probes.len()];
    for round in 0..3 {
        let asc: Vec<usize> = (0..probes.len()).collect();
        let order: Vec<usize> = if round % 2 == 0 {
            asc
        } else {
            asc.into_iter().rev().collect()
        };
        for i in order {
            let pr = &probes[i];
            let ms = time_dispatch(&|r| run(pr, r)).ms;
            best[i] = best[i].min(ms);
        }
    }
    eprintln!("[mb/gap] c1536->768 m1008 k1（与生产 Conv.58/55 同形），只动 out 段偏移");
    let base = best[0];
    for (i, &gp) in gaps.iter().enumerate() {
        let ms = best[i];
        let gf = probes[i].flops / (ms * 1e6);
        eprintln!(
            "[mb/gap]   gap={:>8} word ({:>5.1} MB) {:>7.3} ms {:>7.1} GFLOPS  相对 gap=0 {:.3}×",
            gp,
            gp as f64 * 4.0 / 1e6,
            ms,
            gf,
            ms / base
        );
    }
    // 镜像形状做同一把尺子：它是生产里「没残差的那一半」，各档应当同样平。
    let mpr = build_conv_case(&ctx.inner.device, &mirror);
    let mms = time_dispatch(&|r| run(&mpr, r)).ms;
    eprintln!(
        "[mb/gap] 镜像 c768->1536（gap 由构造给定）{:>7.3} ms {:>7.1} GFLOPS",
        mms,
        mpr.flops / (mms * 1e6)
    );
}

/// `n_conv8_mma` 的**残差段**对照臂：同形状、同内容、只差「有没有残差段」与
/// 「残差段离输出多远」。
///
/// ## 要回答的问题（任务 #39）
///
/// 生产里 `Conv.55` 与 `Conv.58` 的参数块 21 词里只有 4 个偏移不同、网格
/// 逐字相同，却在逐内核剖析里稳定差 1.31×——先后对调（`PROF_ONLY="85,88"`
/// vs `"88,85"`）、以及前置臂（干净 arena）与自然位两次都复现。两条候选已
/// 被证伪：
/// - **尾声 act/bias/模块**：两条参数块里 `act/c1/c2/c3` 逐位相同；
/// - **输出段地址**：[`conv8_mma_layout_gap`] 把 `out` 平移 33 MB，全档离散 2%。
///
/// 剩下没被控过的自由度只有**残差段**：两条都有 `r_off`，但 `Conv.55` 的
/// `r` 与 `out` 紧邻（`out − r` = 1.00 × `m·co`），`Conv.58` 隔开
/// 2.67 × `m·co`（这个比例在全部 14 份计划里逐份成立）。本探针把这两档搭出来。
///
/// **纪律**：探针只能否证、不能拍板。无论正负，结论只用来排除或聚焦；真要
/// 落地必须回生产按 `ab.sh` 交错对拍。
#[test]
fn conv8_mma_residual() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    // 与生产那两条同形：`Conv.55/58`（1536->768，21×48）。
    let slow = ConvCase { label: "c1536->768", h: 21, w: 48, cin: 1536, cout: 768, kh: 1, kw: 1 };
    let mirror = ConvCase { label: "c768->1536", h: 21, w: 48, cin: 768, cout: 1536, kh: 1, kw: 1 };
    // `m·co` = 1008 × 768 = 774144 word。生产两档：紧邻（+0）与 +5/3·m·co
    // （即 `out − r` = 2.67·m·co）。5/3 是先除后乘，避免 `mco*5` 溢出前的取整差。
    let mco = 1008u32 * 768;
    let far_gap = mco / 3 * 5;

    let run = |pr: &ConvProbe, reps: u32| -> f64 {
        let t = Instant::now();
        submit_compute(&ctx.inner.queue, |enc| {
            for _ in 0..reps {
                ks.dispatch(enc, "n_conv8_mma", &pr.buf, pr.p_off, mm_sg_groups(21 * 48, pr.co4))
                    .expect("dispatch");
                let res: [&metal::ResourceRef; 1] = [&pr.buf];
                enc.memory_barrier_with_resources(&res);
            }
        })
        .expect("提交");
        t.elapsed().as_secs_f64() * 1000.0
    };

    let arms: [(&str, ConvProbe); 3] = [
        ("无残差      ", build_conv_case_res(&ctx.inner.device, &slow, 0, false)),
        ("残差紧邻 1.00×", build_conv_case_res(&ctx.inner.device, &slow, 0, true)),
        ("残差远隔 2.67×", build_conv_case_res(&ctx.inner.device, &slow, far_gap, true)),
    ];
    let mut best = vec![f64::INFINITY; arms.len()];
    for round in 0..3 {
        let asc: Vec<usize> = (0..arms.len()).collect();
        let order: Vec<usize> = if round % 2 == 0 {
            asc
        } else {
            asc.into_iter().rev().collect()
        };
        for i in order {
            let pr = &arms[i].1;
            let ms = time_dispatch(&|r| run(pr, r)).ms;
            best[i] = best[i].min(ms);
        }
    }
    let base = best[0];
    eprintln!("[mb/res] c1536->768 m1008 k1（与生产 Conv.55/58 同形），唯一自变量=残差段");
    for (i, (label, pr)) in arms.iter().enumerate() {
        let ms = best[i];
        eprintln!(
            "[mb/res]   {label} {:>7.3} ms {:>7.1} GFLOPS  相对无残差 {:.3}×",
            ms,
            pr.flops / (ms * 1e6),
            ms / base
        );
    }
    // 尺子：生产里 `Conv.54/57` 是「同形状、无残差」的另一半，用镜像形状再
    // 钉一次「形状本身」的贡献（isolated 已知 c768->1536 ≈ 0.99 ms）。
    let mpr = build_conv_case(&ctx.inner.device, &mirror);
    let mms = time_dispatch(&|r| run(&mpr, r)).ms;
    eprintln!(
        "[mb/res] 镜像 c768->1536 无残差 {:>7.3} ms {:>7.1} GFLOPS",
        mms,
        mpr.flops / (mms * 1e6)
    );
}

/// 把生产计划 `[1,3,48,1344]`（185 条）里**四条邻接派发的真实地址几何**
/// 原样搬进探针。
///
/// ## 要回答的问题
///
/// 生产逐内核剖析（今天复现）：`Conv.54` 0.967 / `Conv.55` 1.462 /
/// `Conv.57` 1.050 / `Conv.58` 1.892 ms，四条的 `m/cin/co` 只有两档
/// （768→1536 与 1536→768），网格逐字相同，参数块除 4 个偏移外逐位相同。
/// 已知（隔离探针，`conv8_mma_throughput`）同形状同参数的基线是
/// **1536→768 1.165 / 768→1536 0.985 ms** ⇒ `Conv.54/57` ≈ 基线（1.0×），
/// 而 `Conv.55` 1.25×、`Conv.58` 1.62×。已经证伪的候选：
/// 尾声 act/bias（参数逐位同）、`out` 段地址（[`conv8_mma_layout_gap`] 平移
/// 33 MB 全平）、残差段与输出的距离（[`conv8_mma_residual`] 两档只差 0.5%）。
///
/// **没被控过的**只剩「段被摆在 115 MB arena 的哪几处」与「arena 本身多大」
/// ——紧排探针的 arena 只有 17 MB。本测试把生产的绝对偏移与总长照搬：
///
/// | 派发 | m | in | w | r | out | 生产实测 |
/// |---|---|---|---|---|---|---|
/// | Conv.54 | 1008 | 25246848 | 9610832 | — | 27311232 | 0.967 |
/// | Conv.55 | 1008 | 27311232 | 10792040 | 25246848 | 26020992 | 1.462 |
/// | Conv.57 | 1008 | 21118080 | 11980180 | — | 27311232 | 1.050 |
/// | Conv.58 | 1008 | 27311232 | 13161388 | 21118080 | 23182464 | 1.892 |
///
/// 四条的总长都是 `28859520` word（115.4 MB）。**复现出来 ⇒ 几何（或 arena
/// 尺寸）就是原因，下一步二分；复现不出来 ⇒ 原因在探针看不见的地方**（别的
/// 派发留下的缓存状态、计划里其余 181 条的余波），这条线索该收摊。
///
/// ⚠️ **下面那些绝对偏移取自 `MMA_W_TAIL` 之前的一次计划 dump**：现在每个
/// conv 权重块尾巴多垫 128 B（见 [`qppocr_gpu::nhwc::MMA_W_TAIL`]），生产
/// 静态区的落点整体后移。探针仍然自洽（同一份偏移喂给所有臂），但**不要再
/// 拿这里的数字去对当前生产的 `偏: w=...`**——要重新 dump。
#[test]
fn conv8_mma_prod_geom() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    // m = 3×336 = 1008 = 21×48（段长逐字节相同，只是空间形状不同）。
    let wide = ConvCase { label: "c768->1536", h: 21, w: 48, cin: 768, cout: 1536, kh: 1, kw: 1 };
    let bott = ConvCase { label: "c1536->768", h: 21, w: 48, cin: 1536, cout: 768, kh: 1, kw: 1 };
    const TOTAL: u32 = 28859520; // 生产计划里这四条的总长（word）

    let run = |pr: &ConvProbe, reps: u32| -> f64 {
        let t = Instant::now();
        submit_compute(&ctx.inner.queue, |enc| {
            for _ in 0..reps {
                ks.dispatch(enc, "n_conv8_mma", &pr.buf, pr.p_off, mm_sg_groups(21 * 48, pr.co4))
                    .expect("dispatch");
                let res: [&metal::ResourceRef; 1] = [&pr.buf];
                enc.memory_barrier_with_resources(&res);
            }
        })
        .expect("提交");
        t.elapsed().as_secs_f64() * 1000.0
    };

    let arms: [(&str, ConvProbe); 4] = [
        (
            "Conv.54 生产几何",
            build_conv_case_at(&ctx.inner.device, &wide, [25246848, 9610832, 10790480, 0, 27311232], TOTAL, false),
        ),
        (
            "Conv.55 生产几何",
            build_conv_case_at(&ctx.inner.device, &bott, [27311232, 10792040, 11971688, 25246848, 26020992], TOTAL, true),
        ),
        (
            "Conv.57 生产几何",
            build_conv_case_at(&ctx.inner.device, &wide, [21118080, 11980180, 13159828, 0, 27311232], TOTAL, false),
        ),
        (
            "Conv.58 生产几何",
            build_conv_case_at(&ctx.inner.device, &bott, [27311232, 13161388, 14341036, 21118080, 23182464], TOTAL, true),
        ),
    ];
    let mut best = vec![f64::INFINITY; arms.len()];
    for round in 0..3 {
        let asc: Vec<usize> = (0..arms.len()).collect();
        let order: Vec<usize> = if round % 2 == 0 {
            asc
        } else {
            asc.into_iter().rev().collect()
        };
        for i in order {
            let pr = &arms[i].1;
            let ms = time_dispatch(&|r| run(pr, r)).ms;
            best[i] = best[i].min(ms);
        }
    }
    eprintln!("[mb/geo] 生产四条的无残差谱系；紧排基线 c1536->768 1.165 / c768->1536 0.985 ms");
    for (i, (label, pr)) in arms.iter().enumerate() {
        let ms = best[i];
        eprintln!(
            "[mb/geo]   {label} {:>7.3} ms {:>7.1} GFLOPS",
            ms,
            pr.flops / (ms * 1e6)
        );
    }
}

/// [`conv8_mma_prod_geom`] 复现成功后的**单字段二分**：形状固定
/// （1536→768, m=1008, res=true），从 `Conv.58` 的几何出发，每次只把一个
/// 字段换成 `Conv.55` 的值，看哪一格吃掉那 0.455 ms。
///
/// 生产两档的字段（word）：
///
/// | 字段 | Conv.55（1.435） | Conv.58（1.890） |
/// |---|---|---|
/// | in | 27311232 | 27311232（相同） |
/// | w | 10792040 | 13161388 |
/// | r | 25246848 | 21118080 |
/// | out | 26020992 | 23182464 |
///
/// 另加两臂把 `in` 也换掉（`Conv.54` 的 25246848 / `Conv.57` 的 21118080），
/// 因为「宽形状」那一对（0.977 vs 1.045）只差 `in`/`w` 的落点。
/// 权重段在生产里是**静态区**（跨计划固定偏移），所以这几档 `w` 的差异对
/// 整个模型是常量——找到「落在哪就慢」的规律才有可能去改分配器。
#[test]
fn conv8_mma_prod_bisect() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    let bott = ConvCase { label: "c1536->768", h: 21, w: 48, cin: 1536, cout: 768, kh: 1, kw: 1 };
    const TOTAL: u32 = 28859520;
    // `b` 恒取 `w + n_w`（生产的静态块就是「权重 + bias + 参数块」紧排）。
    const N_W: u32 = 1536 * 768;
    let b_of = |w: u32| w + N_W;

    // (标签, in, w, r, out)
    let geoms: [(&str, [u32; 4]); 7] = [
        ("A55 全 55      ", [27311232, 10792040, 25246848, 26020992]),
        ("B58 全 58      ", [27311232, 13161388, 21118080, 23182464]),
        ("w → 55 的      ", [27311232, 10792040, 21118080, 23182464]),
        ("r → 55 的      ", [27311232, 13161388, 25246848, 23182464]),
        ("out → 55 的    ", [27311232, 13161388, 21118080, 26020992]),
        ("in → 25246848（54）", [25246848, 13161388, 21118080, 23182464]),
        ("in → 21118080（57）", [21118080, 13161388, 21118080, 23182464]),
    ];

    let run = |pr: &ConvProbe, reps: u32| -> f64 {
        let t = Instant::now();
        submit_compute(&ctx.inner.queue, |enc| {
            for _ in 0..reps {
                ks.dispatch(enc, "n_conv8_mma", &pr.buf, pr.p_off, mm_sg_groups(21 * 48, pr.co4))
                    .expect("dispatch");
                let res: [&metal::ResourceRef; 1] = [&pr.buf];
                enc.memory_barrier_with_resources(&res);
            }
        })
        .expect("提交");
        t.elapsed().as_secs_f64() * 1000.0
    };

    let probes: Vec<(&str, ConvProbe)> = geoms
        .iter()
        .map(|&(label, [i, w, r, o])| {
            (label, build_conv_case_at(&ctx.inner.device, &bott, [i, w, b_of(w), r, o], TOTAL, true))
        })
        .collect();
    let mut best = vec![f64::INFINITY; probes.len()];
    for round in 0..3 {
        let asc: Vec<usize> = (0..probes.len()).collect();
        let order: Vec<usize> = if round % 2 == 0 {
            asc
        } else {
            asc.into_iter().rev().collect()
        };
        for i in order {
            let ms = time_dispatch(&|r| run(&probes[i].1, r)).ms;
            best[i] = best[i].min(ms);
        }
    }
    let base = best[0]; // A55
    eprintln!("[mb/bis] c1536->768 m1008 res=true，同 arena（115.4 MB），每次只换一个字段");
    for (i, (label, pr)) in probes.iter().enumerate() {
        eprintln!(
            "[mb/bis]   {label} {:>7.3} ms {:>7.1} GFLOPS  相对 A55 {:.3}×",
            best[i],
            pr.flops / (best[i] * 1e6),
            best[i] / base
        );
    }
}

/// [`conv8_mma_prod_bisect`] 把 1.31× 钉在**权重段地址**上之后的刻画：扫 `w`
/// 的落点。
///
/// 四个激活段全部钉在 `Conv.55` 的几何（in=27311232, r=25246848,
/// out=26020992），只把 `w` 从 `10792040`（快，1.43 ms）往 `13161388`
/// （慢，1.87 ms）方向按 4 KB 步长挪 —— 两档相差 2369348 word（9.04 MB），
/// 且两处的 **64 KB 页内偏移不同**（45472 vs 20144 字节），所以先按 4 KB
/// 扫一个 64 KB 周期，看「页内偏移」是不是自变量。每臂同时打印页内偏移。
///
/// 若确实成规律，处方在计划分配器侧：静态权重的落点是可以整体平移的
/// （权重是离线重排出来的，与激活段无关），把每个权重块对齐到「好格」即可。
#[test]
fn conv8_mma_w_sweep() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    let bott = ConvCase { label: "c1536->768", h: 21, w: 48, cin: 1536, cout: 768, kh: 1, kw: 1 };
    const TOTAL: u32 = 28859520;
    const N_W: u32 = 1536 * 768;
    // 激活段钉死在 Conv.55 的几何。
    let (in_w, r_w, out_w) = (27311232u32, 25246848u32, 26020992u32);
    let w0 = 10792040u32;

    let run = |pr: &ConvProbe, reps: u32| -> f64 {
        let t = Instant::now();
        submit_compute(&ctx.inner.queue, |enc| {
            for _ in 0..reps {
                ks.dispatch(enc, "n_conv8_mma", &pr.buf, pr.p_off, mm_sg_groups(21 * 48, pr.co4))
                    .expect("dispatch");
                let res: [&metal::ResourceRef; 1] = [&pr.buf];
                enc.memory_barrier_with_resources(&res);
            }
        })
        .expect("提交");
        t.elapsed().as_secs_f64() * 1000.0
    };

    // **三遍**，靠 `QPPOCR_MB_WMODE` 切：
    //  - `fine`：4 KB 步长扫满一个 64 KB 周期（结论：平，±1.2%）⇒ 缓存组号
    //    不用 bit12..15；
    //  - `coarse`：1 MB 步长扫 −4..+11 MB（覆盖 `Conv.58` 的 13161388）
    //    （结论：**也平**，1.418~1.448；只有恰好 13161388 是 1.90）；
    //  - `dup`：把几个地址各放两份并打印**缓冲基址** —— 每臂各自
    //    `new_buffer(115 MB)`，基址若不同，则「同偏移两臂」的差就是**分配
    //    运气**而不是地址几何；基址相同才说明前一版的单字段二分成立。
    let mode = std::env::var("QPPOCR_MB_WMODE").unwrap_or_else(|_| "coarse".into());
    // `dup` 已证**确定可复现**（同地址两臂分别 1.441/1.454 与 1.898/1.903，
    // 基址逐臂 +115.4 MB 连续），且 `w=13162384`（只差 3984 B）掉到 1.229。
    // `near` 因此在 13161388 附近做 512 B 步长的刻画，并把 **bias 段单独掰开**
    // （之前每臂 `b = w + n_w` 恒同步，分不清是 w 还是 b）：
    // 每臂 `(w, b)` 显式给出。
    let ws: Vec<u32> = match mode.as_str() {
        "fine" => (0..=16u32).map(|j| w0 + j * 1024).chain([13161388, 13261388]).collect(),
        "dup" => vec![w0, 13161388, w0, 13161388, 13151336, 13151336, 10793064, 13162384],
        _ => (-4i32..=11)
            .map(|j| (w0 as i64 + j as i64 * 262144) as u32)
            .chain([13161388])
            .collect(),
    };
    // `near` 专用：`(w, b)` 对。上一轮已用三组解耦臂证明 **`b` 完全无关**
    // （w 钉死 13161388、b 换到 14330984 / 14342032，读数仍是 1.905~1.907），
    // 且 `w` 在 **13161900（1.883）与 13162384（1.253，三臂一致）之间有离散跳变**。
    // 这里按 512 B 步长把跳变扫出来：`w = W58 + j*128`，`j = 0..=20`。
    const B58: u32 = 14341036; // Conv.58 的 b（= w + n_w）
    const W58: u32 = 13161388;
    let near: Vec<(u32, u32)> = (0..=20u32).map(|j| (W58 + j * 128, B58 + j * 128)).collect();
    // `align`：把已测的所有点按 `w mod 16`（word，= 64 B）归类后浮现的规律 ——
    // ≡12（48 B）全是最慢 1.85~1.91，≡8（32 B）全是 1.44，唯一个 ≡0（64 B）
    // 的是 1.25。4 KB 与 1 MB 两个扫描之所以全平，正是因为两个步长都是 16 的
    // 倍数、从未换过相位。这里把 16 个相位一次测全（b 钉死，已证无关）。
    let align: Vec<(u32, u32)> = (0..16u32).map(|j| (W58 + j, B58)).collect();
    // `alignw`：把同一条相位规律换到**另一个形状**上验证。行距由 `co4` 决定
    // （1536 → 6144 B，仍是 128 B 整数倍），激活段用 `Conv.57` 的
    // `in=21118080 / out=27311232`、无残差。只对 `co4=768` 成立就不算规律。
    let wide = ConvCase { label: "c768->1536", h: 21, w: 48, cin: 768, cout: 1536, kh: 1, kw: 1 };
    let alignw: Vec<(u32, u32)> =
        (0..16u32).map(|j| (11980180 + j, 11980180 + j + 768 * 1536)).collect();
    // `aligna`：把 16 个相位换到**激活段**上。`co4=1536` 那一档里 B 的行距是
    // 6144 B、A 的是 3072 B —— 而 `co4=768` 那档恰好反过来（B 3072 / A 6144），
    // 且大摆幅只出现在「3072 B 行距」的那个操作数上。所以「敏感的是 3072 B
    // 行距的操作数」这一假说预测：换 `in` 的相位会看到大摆幅（而不是 7%）。
    // 若成立，处方不止要动静态布局，还要把激活段的分配抬到 64 B 对齐。
    let (case, with_res, varies_in, geoms): (&ConvCase, bool, bool, Vec<[u32; 5]>) =
        match mode.as_str() {
            "near" => (
                &bott,
                true,
                false,
                near.iter().map(|&(w, b)| [in_w, w, b, r_w, out_w]).collect(),
            ),
            "align" => (
                &bott,
                true,
                false,
                align.iter().map(|&(w, b)| [in_w, w, b, r_w, out_w]).collect(),
            ),
            "alignw" => (
                &wide,
                false,
                false,
                (0..16u32)
                    .map(|j| {
                        let w = 11980180 + j;
                        [21118080, w, w + 768 * 1536, 0, 27311232]
                    })
                    .collect(),
            ),
            "aligna" => (
                &wide,
                false,
                true,
                (0..16u32)
                    .map(|j| [21118080 + j, 11980192, 11980192 + 768 * 1536, 0, 27311232])
                    .collect(),
            ),
            _ => (
                &bott,
                true,
                false,
                ws.iter().map(|&w| [in_w, w, w + N_W, r_w, out_w]).collect(),
            ),
        };
    let probes: Vec<ConvProbe> = geoms
        .iter()
        .map(|&g| build_conv_case_at(&ctx.inner.device, case, g, TOTAL, with_res))
        .collect();
    let mut best = vec![f64::INFINITY; probes.len()];
    for round in 0..3 {
        let asc: Vec<usize> = (0..probes.len()).collect();
        let order: Vec<usize> = if round % 2 == 0 {
            asc
        } else {
            asc.into_iter().rev().collect()
        };
        for i in order {
            let ms = time_dispatch(&|r| run(&probes[i], r)).ms;
            best[i] = best[i].min(ms);
        }
    }
    eprintln!(
        "[mb/ws] {} （mode={mode}，{}）",
        case.label,
        if varies_in {
            "只挪激活段，权重/bias 钉死"
        } else {
            "激活段钉死，只挪权重/bias 段"
        }
    );
    for (i, g) in geoms.iter().enumerate() {
        let pr = &probes[i];
        let (i_off, w_off, b_off) = (g[0], g[1], g[2]);
        eprintln!(
            "[mb/ws]   in={:>9}(+{:>5} B) w={:>9}(+{:>5} B) b={:>9} 基址={:>12} {:>7.3} ms {:>7.1} GFLOPS",
            i_off,
            (i_off as u64 * 4) % 64,
            w_off,
            (w_off as u64 * 4) % 64,
            b_off,
            pr.buf.contents() as usize,
            best[i],
            pr.flops / (best[i] * 1e6)
        );
    }
}

/// `n_conv_dw` 的**权重段相位扫描**（`conv8_mma_w_sweep` 的 `align` 臂的 dw 版）。
///
/// ## 为什么要做
///
/// 修「静态块对齐到 64 B」那条时，顺带看到 det 的 dw 家族 **24.31 → 17.33 ms**
/// （`Conv.64` 单条 14.19 → 8.58，1.65×），比要修的那两条 MMA 涨得还多。
/// 但 64 B 是**顺手**摆的相位：MMA 那边量过 16 个相位里 ≡0 最快，dw 的
/// 权重装载形态完全不同（`[tap][c4]`，一个 float4 = 4 个通道该 tap 的权重），
/// **没有理由同相**。若 dw 的最优相位不是 ≡0，那么在生产里给 dw 那一类
/// 权重块单独补一小段填充，就能再拿一次。
///
/// ## 怎么读
///
/// 16 个相位逐臂跑同一内核、同一网格、同一形状，只有权重段起点差 1 word。
/// 取每臂三轮最小。**只有出现一个幅度明显（>10%）且可复现的谷底**才值得
/// 动生产布局——相位噪声本身也就几个百分点。
///
/// 形状取 `QPPOCR_METAL_PLAN_DUMP` 里 medium det 计划的首恶 `Conv.64`
/// （9x9 c256 136x240，单条最大的 dw）。
#[test]
fn conv_dw_w_sweep() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    // **两个形状都要扫**：MMA 那边的相位幅度是**随形状变**的（`co4=768` 是
    // 1.51×、`co4=1536` 只有 1.07×），所以在一个形状上量到「只有 1.09×」
    // 不足以否掉「生产 dw 掉 1.4× 是这个相位」——生产的 `Conv.64` 是
    // m=51840（grid 12960，对应 216x240 输入），下面第一例才是它。
    let cases = [
        DwCase { label: "dw 9x9 c256 216x240 (Conv.64 生产网格 grid=12960)", h: 216, w: 240, c: 256, kh: 9, kw: 9, stride: 1 },
        DwCase { label: "dw 9x9 c256 136x240 (Conv.64 小档)", h: 136, w: 240, c: 256, kh: 9, kw: 9, stride: 1 },
    ];
    // `QPPOCR_MB_DWMODE=w`（默认，扫权重段）/ `in`（扫输入段，权重钉死）。
    let mode = std::env::var("QPPOCR_MB_DWMODE").unwrap_or_else(|_| "w".into());
    for c in &cases {
        // 16 个相位各一臂（`*gap = j` ⇒ 该段起点字节偏移 +4j）。
        let probes: Vec<ConvProbe> = (0..16u32)
            .map(|j| {
                if mode == "in" {
                    build_dw_case(&ctx.inner.device, c, false, j, 0)
                } else {
                    build_dw_case(&ctx.inner.device, c, false, 0, j)
                }
            })
            .collect();
        // 网格必须与生产逐位一致，否则绝对值和相位效应都无从与生产对齐。
        for pr in &probes {
            assert_eq!(pr.groups, super::conv_probe::dw_groups_for(pr.out_h * pr.out_w, pr.co4));
        }

        let run = |pr: &ConvProbe, reps: u32| -> f64 {
            let t = Instant::now();
            submit_compute(&ctx.inner.queue, |enc| {
                for _ in 0..reps {
                    ks.dispatch(enc, "n_conv_dw", &pr.buf, pr.p_off, pr.groups).expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [&pr.buf];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
            t.elapsed().as_secs_f64() * 1000.0
        };

        let mut best = vec![f64::INFINITY; probes.len()];
        for round in 0..3 {
            let asc: Vec<usize> = (0..probes.len()).collect();
            let order: Vec<usize> = if round % 2 == 0 { asc } else { asc.into_iter().rev().collect() };
            for i in order {
                let ms = time_dispatch(&|r| run(&probes[i], r)).ms;
                best[i] = best[i].min(ms);
            }
        }
        // 被扫那一段的起点：`in` 恒从 256 起（+ingap），`w` 在 `in` 段之后（+wgap）。
        let base = if mode == "in" {
            256u32
        } else {
            256 + c.h * c.w * c.c.div_ceil(4) * 4
        };
        let fastest = best.iter().cloned().fold(f64::INFINITY, f64::min);
        let slowest = best.iter().cloned().fold(0.0f64, f64::max);
        eprintln!(
            "[mb/dws] {} （mode={mode}，扫的是{}段、基线 {base}，grid={:?}；幅度 {:.3}×）",
            c.label,
            if mode == "in" { "输入" } else { "权重" },
            probes[0].groups,
            slowest / fastest
        );
        for (j, &ms) in best.iter().enumerate() {
            let o = base + j as u32;
            eprintln!(
                "[mb/dws]   off={:>9}（+{:>3} B）{:>7.3} ms  {:>6.1} GFLOPS  {:.3}×",
                o,
                (o as u64 * 4) % 64,
                ms,
                probes[j].flops / (ms * 1e6),
                ms / fastest
            );
        }
    }
}

/// `n_conv8_mma` 的**受控 K 扫描**：固定 `m` 与 `co`，只动 `cin`。
///
/// ## 要回答的问题
///
/// `conv8_mma_throughput` 里最短的一条（`det m32640 c32->64`，K=32）只有
/// **1017 GFLOPS**，而同族长 K 的到 **2455**（73% 峰）。两处差 2.4×。
/// 但那条的形状与长 K 的**同时**变了 `co`（64 vs 256）⇒ 网格 y、线程组数、
/// 权重足迹全跟着变，分不清是「K 太短」还是「co=64 天生慢」。
///
/// 本探针把自变量收敛成一个：`h/w/co` 全钉死，`cin` 取 32…1024。
/// 于是每个线程组的**瓦片（64×64）、尾声（16 KB 线程组暂存 + 屏障 +
/// 4096 次逐元素 bias/act/残差/写回）、网格形状**逐字相同，唯一变化是
/// 热循环的 K 步数（`cin/8` = 4…128 步）。
///
/// ## 怎么读
///
/// 若时间 ≈ `E + c·(cin/8)`，则 `E` 就是**每组固定成本**（尾声为主）：
///   * `E` 相对总时**小** ⇒ 短 K 的亏损不是尾声，回去查热循环；
///   * `E` 相对总时**大** ⇒ 尾声是那一档的主项，值得动（去暂存/去屏障/
///     把逐元素尾声改成 fragment 直出）。
///
/// 同一批数据也能读「长 K 饱和在多少」——那是这条内核自己的天花板，
/// **拿它当尺子**（不许拿 3340 的理论峰算余量，见
/// `roofline-must-use-kernels-own-ceiling`）。
///
/// 序：各 `cin` 交错跑，偶数轮升序、奇数轮降序，取每例最小——本机
/// 「一轮里先跑的那条快 6~15%」，沿一个方向扫会把漂移记进斜率。
#[test]
fn conv8_mma_ksweep() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    // m = 136*240 = 32640（det 全分辨率那一档），co = 64 ⇒ 网格 [510, 1, 1]，
    // 510 个线程组足够喂满机器，且 64 列刚好一个通道块（无浪费）。
    const H: u32 = 136;
    const W: u32 = 240;
    const CO: u32 = 64;
    let cins: [u32; 6] = [32, 64, 128, 256, 512, 1024];

    let cases: Vec<ConvCase> = cins
        .iter()
        .map(|&cin| ConvCase {
            label: "ksweep",
            h: H,
            w: W,
            cin,
            cout: CO,
            kh: 1,
            kw: 1,
        })
        .collect();
    let probes: Vec<ConvProbe> = cases
        .iter()
        .map(|c| build_conv_case(&ctx.inner.device, c))
        .collect();

    let run = |pr: &ConvProbe, reps: u32| -> f64 {
        let t = Instant::now();
        let g = mm_sg_groups(H * W, pr.co4);
        submit_compute(&ctx.inner.queue, |enc| {
            for _ in 0..reps {
                ks.dispatch(enc, "n_conv8_mma", &pr.buf, pr.p_off, g)
                    .expect("dispatch");
                let res: [&metal::ResourceRef; 1] = [&pr.buf];
                enc.memory_barrier_with_resources(&res);
            }
        })
        .expect("提交");
        t.elapsed().as_secs_f64() * 1000.0
    };

    let mut best = vec![f64::INFINITY; probes.len()];
    for round in 0..3 {
        let asc: Vec<usize> = (0..probes.len()).collect();
        let order: Vec<usize> = if round % 2 == 0 {
            asc
        } else {
            asc.into_iter().rev().collect()
        };
        for i in order {
            let pr = &probes[i];
            assert_eq!(
                mm_sg_groups(H * W, pr.co4),
                [510, 1, 1],
                "cin={} 的网格变了（co 被补齐成 {}）——自变量不唯一，结论作废",
                cins[i],
                pr.co4
            );
            let ms = time_dispatch(&|r| run(pr, r)).ms;
            best[i] = best[i].min(ms);
        }
    }

    let ntg = 510.0f64;
    eprintln!("[mb/ksweep] m={} co={CO} 网格 [510,1,1]，只动 cin", H * W);
    for (i, &cin) in cins.iter().enumerate() {
        let pr = &probes[i];
        let ms = best[i];
        let gf = pr.flops / (ms * 1e6);
        eprintln!(
            "[mb/ksweep] cin={:<5} K步={:<4} {:>7.3} ms  {:>7.1} GFLOPS  {:>4.1}%峰  每组 {:>6.3} µs",
            cin,
            cin / 8,
            ms,
            gf,
            gf / 3340.0 * 100.0,
            ms * 1000.0 / ntg
        );
    }
}

/// `n_conv8_mma` 热循环那 25% 缺口的**归因**：装载指令 还是 流量/延迟。
///
/// ## 背景
///
/// `conv8_mma_ksweep` 量出边际斜率 0.0262 µs/K步 ⇒ 热循环只到 **75% 峰**。
/// 本测试用 `probe_mma_split.metal` 的三臂把它劈开：
///
/// | 臂 | 变了什么 | 回答 |
/// |---|---|---|
/// | `p_mma_base` | 与生产逐字节相同 | 对照 |
/// | `p_mma_small` | A+B 基址+行距收进 1 KB（条数不变） | **流量/延迟**值多少 |
/// | `p_mma_hoist` | 6 条 load 挪出 K 循环（MMA 不变） | **发射槽**值多少 |
/// | `p_mma_asmall` | **只**收 A（2 条） | 1.22× 里 A 侧占多少 |
/// | `p_mma_bsmall` | **只**收 B（4 条） | 1.22× 里 B 侧占多少 |
/// | `p_mma_noepi` | 删掉 16 KB 暂存 + 尾声 + 输出写 | **动尾声的整体上界** |
///
/// 前两臂的分侧是给 `p_mma_small` 收尾用的：`p_mma_wfrag`（权重离线重排成分片
/// 连续，**结果正确**）在 nDim=64 上量到 **1.00×** ⇒ B 侧不是行数/合并问题，
/// 必须把 1.22× 劈到某一侧才知道处方往哪开。
///
/// 末臂同时去掉尾声算术、输出写流量、16 KB 线程组内存三样，所以它是**上界**
/// 而不是可落地值——`p_mma_hoist` 给热循环的上界（1.44×），它给尾声的上界。
///
/// 判读见该 .metal 的头注释。**注意除 `p_mma_base` 外数值都是错的**——
/// 诊断件，不是能落地的内核。
///
/// 序：五臂 × 6 个 cin 全交错，偶数轮升序、奇数轮降序，取每例最小。
#[test]
fn conv8_mma_load_attribution() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let probes = super::conv_probe::ProbeSet::new(&ctx.inner.device).expect("编译探针");

    const H: u32 = 136;
    const W: u32 = 240;
    const CO: u32 = 64;
    let cins: [u32; 6] = [32, 64, 128, 256, 512, 1024];
    let arms = [
        "p_mma_base",
        "p_mma_small",
        "p_mma_hoist",
        "p_mma_asmall",
        "p_mma_bsmall",
        "p_mma_ost16",
        "p_mma_ost2",
    ];

    let cases: Vec<ConvCase> = cins
        .iter()
        .map(|&cin| ConvCase {
            label: "attr",
            h: H,
            w: W,
            cin,
            cout: CO,
            kh: 1,
            kw: 1,
        })
        .collect();
    let built: Vec<ConvProbe> = cases
        .iter()
        .map(|c| build_conv_case(&ctx.inner.device, c))
        .collect();

    let run = |pr: &ConvProbe, arm: &str, reps: u32| -> f64 {
        let pipe = probes.pipe(arm).expect("取管线");
        let g = mm_sg_groups(H * W, pr.co4);
        let t = Instant::now();
        submit_compute(&ctx.inner.queue, |enc| {
            for _ in 0..reps {
                super::pipeline::dispatch_with(enc, pipe, &pr.buf, pr.p_off, g)
                    .expect("dispatch");
                let res: [&metal::ResourceRef; 1] = [&pr.buf];
                enc.memory_barrier_with_resources(&res);
            }
        })
        .expect("提交");
        t.elapsed().as_secs_f64() * 1000.0
    };

    // best[cin][arm]
    let na = arms.len();
    let mut best = vec![vec![f64::INFINITY; na]; built.len()];
    for round in 0..3 {
        let mut order: Vec<(usize, usize)> = Vec::new();
        for i in 0..built.len() {
            for a in 0..na {
                order.push((i, a));
            }
        }
        if round % 2 == 1 {
            order.reverse();
        }
        for (i, a) in order {
            let ms = time_dispatch(&|r| run(&built[i], arms[a], r)).ms;
            best[i][a] = best[i][a].min(ms);
        }
    }

    eprintln!(
        "[mb/attr] m={} co={CO} 网格 [510,1,1]；base=生产逐字节同款，small=A+B 收 1KB，hoist=load 挪出循环，asmall/bsmall=只收一侧",
        H * W
    );
    eprintln!(
        "[mb/attr] ost16/ost2 = 热循环与尾声尾部逐字节相同，**只有线程组数组声明的大小不同**（16 KB vs 2 KB）"
    );
    eprintln!(
        "[mb/attr] {:<6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}   {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "cin", "base ms", "small ms", "hoist ms", "asmall ms", "bsmall ms", "ost16 ms", "ost2 ms",
        "b/small", "b/hoist", "b/asmall", "b/bsmall", "b/ost16", "ost16/ost2"
    );
    for (i, &cin) in cins.iter().enumerate() {
        let b = &best[i];
        eprintln!(
            "[mb/attr] {:<6} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3}   {:>6.2}× {:>6.2}× {:>6.2}× {:>6.2}× {:>6.2}× {:>6.2}×",
            cin,
            b[0],
            b[1],
            b[2],
            b[3],
            b[4],
            b[5],
            b[6],
            b[0] / b[1],
            b[0] / b[2],
            b[0] / b[3],
            b[0] / b[4],
            b[0] / b[5],
            b[5] / b[6]
        );
    }
    // 长 K 那一档的绝对达成率（用 base 臂，K=1024 时尾声只占 4%）
    let last = cins.len() - 1;
    let gfa = |a: usize| built[last].flops / (best[last][a] * 1e6);
    eprintln!(
        "[mb/attr] cin=1024（尾声占比≈4%，≈纯热循环）: base {:.0} GFLOPS ({:.0}%峰) | small {:.0} ({:.0}%) | hoist {:.0} ({:.0}%) | asmall {:.0} ({:.0}%) | bsmall {:.0} ({:.0}%)",
        gfa(0),
        gfa(0) / 3340.0 * 100.0,
        gfa(1),
        gfa(1) / 3340.0 * 100.0,
        gfa(2),
        gfa(2) / 3340.0 * 100.0,
        gfa(3),
        gfa(3) / 3340.0 * 100.0,
        gfa(4),
        gfa(4) / 3340.0 * 100.0
    );
}

/// 权重**分片分块**布局的收益（`p_mma_wfrag`）——把 B 侧那 4/6 条装载
/// 从「8 段各自的 32 B、相距 `nDim*4` 字节」变成「一段连续的 256 B」。
///
/// ## 为什么只动权重这一侧
///
/// `conv8_mma_load_attribution` 量出 `p_mma_small`（A+B 的足迹与行距**全**
/// 收进 1 KB）值 1.22×，但那条臂同时动了 A 与 B、且把流量整个抹掉，**没有
/// 可落地性**。权重是**静态**量（离线重排一次，与 `repack_conv_w` 同族成本），
/// 输入是激活、不能离线重排 ⇒ 唯一能落地的切法是**只动 B**。
/// 6 条装载里 B 占 4 条，所以这条臂是 1.22× 的**部分版**，
/// 上界猜法是 `1.22^(4/6) ≈ 1.14×`——**只当上限，真值看这张表**。
///
/// ## 单变量
///
/// `build_conv_case` 与 `build_conv_case_wfrag` 的 arena 长度、段偏移、
/// 输入/bias/输出内容**逐位相同**，只有权重段那 64 个 float 的次序不同。
/// 网格、线程组大小、dispatch 条数全同。
///
/// ## 对拍
///
/// 先在生产 `n_conv8_mma`（跑 `build_conv_case` 的 buffer）与 `p_mma_wfrag`
/// （跑 wfrag buffer）之间比输出段——重排写错了这里就红，省得把一条算错的
/// 内核量成一个「收益」。
///
/// 序：两臂 × 6 个 cin 全交错，偶数轮升序、奇数轮降序，取每例最小。
#[test]
fn conv8_mma_wfrag() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");
    let probes = super::conv_probe::ProbeSet::new(&ctx.inner.device).expect("编译探针");

    const H: u32 = 136;
    const W: u32 = 240;
    const CO: u32 = 64;
    let cins: [u32; 6] = [32, 64, 128, 256, 512, 1024];

    let cases: Vec<ConvCase> = cins
        .iter()
        .map(|&cin| ConvCase {
            label: "wfrag",
            h: H,
            w: W,
            cin,
            cout: CO,
            kh: 1,
            kw: 1,
        })
        .collect();
    let base: Vec<ConvProbe> = cases
        .iter()
        .map(|c| build_conv_case(&ctx.inner.device, c))
        .collect();
    let wfrag: Vec<ConvProbe> = cases
        .iter()
        .map(|c| build_conv_case_wfrag(&ctx.inner.device, c))
        .collect();

    // ── 对拍：先证明重排写对了，再谈快慢 ────────────────────────────────
    let wfrag_pipe = probes.pipe("p_mma_wfrag").expect("取管线");
    for (i, c) in cases.iter().enumerate() {
        // 注意 `ConvProbe::groups` 是**标量** `n_conv8` 的旧网格，MMA 两支都不用
        // 它——网格由 `mm_sg_groups` 现算（与 `plan.rs` 的 MMA 支同式）。
        let g = mm_sg_groups(H * W, base[i].co4);
        assert_eq!(g, [510, 1, 1], "cin={} 的网格变了，自变量不唯一", c.cin);
        refill_out_sentinel(&base[i]);
        refill_out_sentinel(&wfrag[i]);
        submit_compute(&ctx.inner.queue, |enc| {
            ks.dispatch(enc, "n_conv8_mma", &base[i].buf, base[i].p_off, g)
                .expect("dispatch");
            super::pipeline::dispatch_with(enc, wfrag_pipe, &wfrag[i].buf, wfrag[i].p_off, g)
                .expect("dispatch");
        })
        .expect("提交");
        let a = read_out(&base[i]);
        let b = read_out(&wfrag[i]);
        let (mut mx, mut scale) = (0f32, 0f32);
        for (x, y) in a.iter().zip(b.iter()) {
            mx = mx.max((x - y).abs());
            scale = scale.max(x.abs());
        }
        assert!(
            mx <= 1e-4 * scale.max(1.0),
            "cin={} 分片分块布局与生产输出对不上：最大差 {mx}，量级 {scale}",
            c.cin
        );
    }
    eprintln!("[mb/wfrag] 对拍通过：6 个 cin 上 p_mma_wfrag 与 n_conv8_mma 输出逐位同量级");

    let run = |pr: &ConvProbe, arm: &str, reps: u32| -> f64 {
        let pipe = probes.pipe(arm).expect("取管线");
        let g = mm_sg_groups(H * W, pr.co4);
        let t = Instant::now();
        submit_compute(&ctx.inner.queue, |enc| {
            for _ in 0..reps {
                super::pipeline::dispatch_with(enc, pipe, &pr.buf, pr.p_off, g)
                    .expect("dispatch");
                let res: [&metal::ResourceRef; 1] = [&pr.buf];
                enc.memory_barrier_with_resources(&res);
            }
        })
        .expect("提交");
        t.elapsed().as_secs_f64() * 1000.0
    };

    // best[cin][arm]：0 = p_mma_base（跑普通布局），1 = p_mma_wfrag（跑重排布局）
    let mut best = vec![[f64::INFINITY; 2]; cases.len()];
    for round in 0..3 {
        let mut order: Vec<(usize, usize)> = Vec::new();
        for i in 0..cases.len() {
            for a in 0..2 {
                order.push((i, a));
            }
        }
        if round % 2 == 1 {
            order.reverse();
        }
        for (i, a) in order {
            let (buf, pr) = if a == 0 {
                ("p_mma_base", &base[i])
            } else {
                ("p_mma_wfrag", &wfrag[i])
            };
            let ms = time_dispatch(&|r| run(pr, buf, r)).ms;
            best[i][a] = best[i][a].min(ms);
        }
    }

    eprintln!(
        "[mb/wfrag] m={} co={CO} 网格 [510,1,1]；base=行距 nDim，wfrag=权重分片连续（行距 8）",
        H * W
    );
    eprintln!(
        "[mb/wfrag] {:<6} {:>9} {:>9}   {:>8}   {:>10} {:>10}",
        "cin", "base ms", "wfrag ms", "base/wfrag", "base GF", "wfrag GF"
    );
    for (i, &cin) in cins.iter().enumerate() {
        let (b, w) = (best[i][0], best[i][1]);
        let gf = |ms: f64| base[i].flops / (ms * 1e6);
        eprintln!(
            "[mb/wfrag] {:<6} {:>9.3} {:>9.3}   {:>7.2}×   {:>10.0} {:>10.0}",
            cin,
            b,
            w,
            b / w,
            gf(b),
            gf(w)
        );
    }
}

/// `n_conv8_mma` 的 **A 侧再读是不是 L2 命中**——决定「加宽每线程组的列块」
/// 这条处方值不值。
///
/// ## 背景
///
/// A 的字节数恒等于 `FLOPs/32`（`m*cin*4 = m*cin*co4*2/2/co4...`，见下），
/// 而 **y 网格 = `co4/64`** ⇒ 同一块 A 瓦片被 `co4/64` 个线程组各读一遍。
/// co4=512 就是 **8 遍**。这 8 遍是 L2 命中还是 DRAM 未命中，决定了两条
/// 完全不同的处方：
///
///   · **L2 命中** ⇒ 宽 co 的形状不吃 DRAM ⇒ 「加宽列块」只赚装载条数
///     （有限），真正卡的是 L1/LSU 与 MMA 流水；
///   · **DRAM 未命中** ⇒ 「把一个线程组覆盖的 co 从 64 加宽到 128/256」
///     直接把 A 的 DRAM 流量按比例砍掉 ⇒ 大处方。
///
/// ## 怎么做到单变量
///
/// `A 字节 = m*cin*4*(co4/64)`，`out 字节 = m*co4*4`，`FLOPs = m*cin*co4*2`，
/// `线程组数 = (m/64)*(co4/64)`。取 **`m*co4` 恒定**：
///
/// | 臂 | m | cin | co4 | gy | FLOPs | out | 线程组 | A 若按 y 冗余算 |
/// |---|---|---|---|---|---|---|---|---|
/// | 窄 | 25920 | 1024 | 64 | 1 | 1.70 G | 6.6 MB | 405 | 106 MB |
/// | 宽 | 3240 | 1024 | 512 | 8 | 1.70 G | 6.6 MB | 408 | 106 MB |
///
/// 两臂的 FLOPs、输出字节、线程组数、**A 的「按 y 冗余计」流量**全都相同；
/// 唯一不同的是**宽臂那 8 遍再读能不能被 L2 接住**。
///
/// ## 判读
///
///   * 宽臂 ≈ 窄臂 ⇒ 再读是 DRAM 未命中（L2 没接住）⇒ **加宽列块有肉**；
///   * 宽臂 ≪ 窄臂 ⇒ L2 接住了 ⇒ 宽 co 的形状本来就不吃 A 的 DRAM，
///     别在这条路上花力气。
///
/// 序：两臂交错，偶数轮升序、奇数轮降序，取每例最小。
#[test]
fn conv8_mma_arearead() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    // m*co4 恒定 1658880；cin 恒定 1024
    let cases = [
        ConvCase {
            label: "窄 gy=1 co4=64",
            h: 108,
            w: 240,
            cin: 1024,
            cout: 64,
            kh: 1,
            kw: 1,
        },
        ConvCase {
            label: "宽 gy=8 co4=512",
            h: 54,
            w: 60,
            cin: 1024,
            cout: 512,
            kh: 1,
            kw: 1,
        },
    ];
    let built: Vec<(ConvProbe, [u32; 3])> = cases
        .iter()
        .map(|c| {
            let pr = build_conv_case(&ctx.inner.device, c);
            let g = mm_sg_groups(c.h * c.w, pr.co4);
            (pr, g)
        })
        .collect();

    // 单变量自检：两臂的 FLOPs / 线程组数 / 输出字节 / A 按冗余计的字节必须相等
    let (f0, f1) = (built[0].0.flops, built[1].0.flops);
    let (t0, t1) = (
        (built[0].1[0] * built[0].1[1]) as f64,
        (built[1].1[0] * built[1].1[1]) as f64,
    );
    assert!((f0 - f1).abs() < f0 * 1e-9, "两臂 FLOPs 不等：{f0} vs {f1}");
    assert!((t0 - t1).abs() < 4.0, "两臂线程组数不等：{t0} vs {t1}");
    // A 字节（按 y 冗余计）= m*cin*4*gy；输出 = m*co4*4
    let ab = |i: usize| {
        let (pr, g) = &built[i];
        let m = pr.out_h as f64 * pr.out_w as f64;
        (
            m * cases[i].cin as f64 * 4.0 * g[1] as f64 / 1e6,
            m * pr.co4 as f64 * 4.0 / 1e6,
        )
    };
    let (a0, o0) = ab(0);
    let (a1, o1) = ab(1);
    assert!((a0 - a1).abs() < a0 * 1e-6, "两臂 A 流量不等：{a0} vs {a1}");
    assert!((o0 - o1).abs() < o0 * 1e-6, "两臂输出流量不等：{o0} vs {o1}");

    let run = |pr: &ConvProbe, grid: [u32; 3], reps: u32| -> f64 {
        let t = Instant::now();
        submit_compute(&ctx.inner.queue, |enc| {
            for _ in 0..reps {
                ks.dispatch(enc, "n_conv8_mma", &pr.buf, pr.p_off, grid)
                    .expect("dispatch");
                let res: [&metal::ResourceRef; 1] = [&pr.buf];
                enc.memory_barrier_with_resources(&res);
            }
        })
        .expect("提交");
        t.elapsed().as_secs_f64() * 1000.0
    };

    let mut best = [f64::INFINITY; 2];
    for round in 0..3 {
        let order: [usize; 2] = if round % 2 == 0 { [0, 1] } else { [1, 0] };
        for i in order {
            let (pr, g) = &built[i];
            let ms = time_dispatch(&|r| run(pr, *g, r)).ms;
            best[i] = best[i].min(ms);
        }
    }

    eprintln!(
        "[mb/arearead] cin=1024，两组 FLOPs/线程组数/输出字节/按 y 冗余计的 A 字节都相同，只有 gy 不同"
    );
    eprintln!(
        "[mb/arearead] A(按冗余) {a0:.1} MB | 输出 {o0:.1} MB | FLOPs {:.2} G | 线程组 {t0}",
        f0 / 1e9
    );
    for (i, c) in cases.iter().enumerate() {
        let gf = built[i].0.flops / (best[i] * 1e6);
        eprintln!(
            "[mb/arearead] {:<14} grid=[{}, {}, 1] {:>8.3} ms  {:>7.1} GFLOPS  {:>4.1}%峰",
            c.label,
            built[i].1[0],
            built[i].1[1],
            best[i],
            gf,
            gf / 3340.0 * 100.0
        );
    }
    eprintln!(
        "[mb/arearead] 比值 窄/宽 = {:.2}×；≈1.0 ⇒ 再读是 DRAM 未命中（加宽列块有肉），≫1 ⇒ L2 接住了",
        best[0] / best[1]
    );
}

/// `n_conv8_mma` 的 **batch 标度**：A 侧那 1.09~1.27× 是「停顿没藏住」还是
/// 「访存通路已经到顶」。
///
/// ## 为什么是这一条
///
/// 五臂归因（`conv8_mma_load_attribution`）已经钉死：那 1.22× 的余量**全在
/// A 侧（激活）**，B 侧（权重）单独收窄是 **0.99×**（与 `p_mma_wfrag` 的
/// 1.00× 相互印证）。于是只剩两种解释，处方完全不同：
///
///   · **DRAM 延迟没被藏住**（占用率 / 访存并行度不足）
///     ⇒ 处方是**提高并行度**（合批、拆 K、缩小线程组），或给 A 做软件预取；
///   · **A 的访存通路已经饱和**
///     ⇒ 处方只能是**改访存结构**（协作暂存 / 换布局），加并行度白搭。
///
/// `batch` 是唯一能在**不改每线程工作量、不改访存形状、不改指令序列**的
/// 前提下把常驻线程组数翻倍的旋钮（`build_conv_case_batch` 的头注释）。
/// 判据：
///
///   · **每份时间随 batch 下降** ⇒ 前者，停顿没藏住；
///   · **每份时间持平**         ⇒ 后者，通路到顶。
///
/// 网格按 MMA 的约定在 z 上铺 batch（`b = wg.z`），与标量 `n_conv8` 在 y 上
/// 铺 batch 不同——`mm_sg_groups` 只给 x/y，z 由这里补。
///
/// 序：全部 (cin, batch) 组合交错，偶数轮升序、奇数轮降序，取每例最小。
#[test]
fn conv8_mma_batch_scaling() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    const H: u32 = 136;
    const W: u32 = 240;
    const CO: u32 = 64;
    let cins: [u32; 2] = [256, 512];
    let batches: [u32; 4] = [1, 2, 4, 8];

    let cases: Vec<ConvCase> = cins
        .iter()
        .map(|&cin| ConvCase {
            label: "batch",
            h: H,
            w: W,
            cin,
            cout: CO,
            kh: 1,
            kw: 1,
        })
        .collect();

    // combos[(ci, bi)] → (probe, grid)
    let mut combos: Vec<(usize, usize, ConvProbe, [u32; 3])> = Vec::new();
    for (ci, c) in cases.iter().enumerate() {
        for (bi, &b) in batches.iter().enumerate() {
            let pr = build_conv_case_batch(&ctx.inner.device, c, b, false);
            let g = mm_sg_groups(H * W, pr.co4);
            assert_eq!(g[0], 510, "cin={} 的 x 网格变了", c.cin);
            assert_eq!(g[1], 1, "co=64 时 y 网格应当是 1");
            combos.push((ci, bi, pr, [g[0], g[1], b]));
        }
    }

    let run = |pr: &ConvProbe, grid: [u32; 3], reps: u32| -> f64 {
        let t = Instant::now();
        submit_compute(&ctx.inner.queue, |enc| {
            for _ in 0..reps {
                ks.dispatch(enc, "n_conv8_mma", &pr.buf, pr.p_off, grid)
                    .expect("dispatch");
                let res: [&metal::ResourceRef; 1] = [&pr.buf];
                enc.memory_barrier_with_resources(&res);
            }
        })
        .expect("提交");
        t.elapsed().as_secs_f64() * 1000.0
    };

    let mut best = vec![f64::INFINITY; combos.len()];
    for round in 0..3 {
        let mut order: Vec<usize> = (0..combos.len()).collect();
        if round % 2 == 1 {
            order.reverse();
        }
        for i in order {
            let (_, _, pr, grid) = &combos[i];
            let ms = time_dispatch(&|r| run(pr, *grid, r)).ms;
            best[i] = best[i].min(ms);
        }
    }

    eprintln!(
        "[mb/batch] m={} co={CO}；网格 x=510，batch 铺在 z 上（b = wg.z）",
        H * W
    );
    eprintln!(
        "[mb/batch] {:<6} {:>6} {:>10} {:>11} {:>10}   {:>9}",
        "cin", "batch", "总 ms", "每份 ms", "每份 µs/组", "GFLOPS"
    );
    for (ci, &cin) in cins.iter().enumerate() {
        // 以 batch=1 的每份时间为基，后面各档的相对值才是要读的数
        let base_pi = {
            let i = combos.iter().position(|(a, b, ..)| *a == ci && *b == 0).unwrap();
            best[i]
        };
        for (bi, &b) in batches.iter().enumerate() {
            let i = combos.iter().position(|(a, bb, ..)| *a == ci && *bb == bi).unwrap();
            let ms = best[i];
            let per = ms / b as f64;
            let flops = cases[ci].h as f64 * cases[ci].w as f64 * CO as f64 * cin as f64 * 2.0;
            eprintln!(
                "[mb/batch] {:<6} {:>6} {:>10.3} {:>11.3} {:>10.3}   {:>9.0}{}",
                cin,
                b,
                ms,
                per,
                per * 1000.0 / 510.0,
                flops / (per * 1e6),
                if bi == 0 {
                    String::new()
                } else {
                    format!("   = {:.2}× 基", base_pi / per)
                }
            );
        }
    }
}

/// 两个**同形状**用例输出段的 `(最大绝对差, 参考臂量级 max|a|)`。
///
/// 差值一律相对量级判：这个形状的输出是 216×18712 个未归一化的
/// logits，绝对值大小取决于权重初值，写死一个绝对阈值必然要么太松
/// （垃圾差也能过）要么太紧（fp16 的舍入差被当成错）。
fn out_diff(a: &ConvProbe, b: &ConvProbe) -> (f32, f32) {
    let n = (a.out_h * a.out_w * a.co4) as usize;
    assert_eq!(n, (b.out_h * b.out_w * b.co4) as usize, "两臂输出长度不一致");
    // SAFETY: 两臂的段布局由同一段代码生成，输出段长 out_h*out_w*co4；
    // 缓冲是 Shared 存储，调用方已 wait_until_completed。
    let (sa, sb) = unsafe {
        (
            std::slice::from_raw_parts((a.buf.contents() as *const f32).add(a.out_off_w as usize), n),
            std::slice::from_raw_parts((b.buf.contents() as *const f32).add(b.out_off_w as usize), n),
        )
    };
    let scale = sa.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let dmax = sa.iter().zip(sb).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    (dmax, scale)
}

/// **生产内核** f32 ── fp16：rec 的 CTC 分类头。
///
/// ## 为什么这条要单独量（而不是看 `conv8_variant_scan`）
///
/// 变体扫描量的是 `p_*` 探针内核，它的 `p_whalf` 用的是 `p_base` 的映射。
/// 而生产在 CTC 头上走的是**窗口序**（门 `taps==1 && cin4v*nv >= 65536`，
/// 本层 30×4678 = 140340）。于是「探针给出的 1.7~2.6× 能不能在最大那条
/// dispatch 上兑现」这个问题里，混进了一个**映射**假变量。这条测试把它
/// 钉死：只 dispatch 生产注册表里真实存在的两个入口，参数块、网格、
/// arena 布局全部出自同一段构造代码，两臂唯一的自变量是
/// `build_conv_case_half` 把权重段写成 fp16。
///
/// ## 顺带验一件事：half4 的地址折算
///
/// `n_conv8h` 用 `w_off >> 1` 把「4 B 元素偏移」折成 half4 下标，而
/// `w_off` 是**同一个 word 偏移**——两个视图落在同一字节地址上，差别只在
/// 步距。这套折算若有一处错，读到的就是**整块错位的垃圾**，误差是
/// 量级差而不是 1e-3 量级。所以下面的对拍断言顺手把它证了。
#[test]
fn conv8_ctc_head_prod_f32_vs_f16() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");
    // 形状逐字取自 rec 计划（`[8,27,120]→[8,27,18710]` 折批），与
    // `conv8_variant_scan` 里那条同形状。
    let c = ConvCase {
        label: "rec ctc head 216x1 c120->18710 k1",
        h: 216,
        w: 1,
        cin: 120,
        cout: 18712,
        kh: 1,
        kw: 1,
    };
    let pr = build_conv_case(&ctx.inner.device, &c);
    let prh = build_conv_case_half(&ctx.inner.device, &c);
    // 两臂除了权重段内容之外必须逐位同源。网格不同 = 「同一条内核同一个
    // 形状」这个前提就破了，后面所有比较都无从归因。
    assert_eq!(pr.groups, prh.groups, "两臂网格不一致");
    assert_eq!(pr.p_off, prh.p_off, "两臂参数块偏移不一致");
    assert_eq!(pr.out_off_w, prh.out_off_w, "两臂输出段偏移不一致");

    let run = |pr: &ConvProbe, name: &str, reps: u32| -> f64 {
        let t = Instant::now();
        submit_compute(&ctx.inner.queue, |enc| {
            for _ in 0..reps {
                ks.dispatch(enc, name, &pr.buf, pr.p_off, pr.groups)
                    .expect("dispatch");
                let res: [&metal::ResourceRef; 1] = [&pr.buf];
                enc.memory_barrier_with_resources(&res);
            }
        })
        .expect("提交");
        t.elapsed().as_secs_f64() * 1000.0
    };

    // 正确性：各跑一次再读回两臂的输出段。
    run(&pr, "n_conv8", 1);
    run(&prh, "n_conv8h", 1);
    let (dmax, scale) = out_diff(&pr, &prh);
    eprintln!(
        "[mb] CTC 头 fp16 ── f32 输出：最大绝对差 {dmax:.3e}（量级 {scale:.3e}，\
         相对 {:.2e}）",
        dmax / scale.max(1e-30)
    );
    assert!(scale > 0.0, "参考臂输出全 0 —— 内核没写（哨兵 7.0 也没了？）");
    assert!(
        dmax < scale * 1e-2,
        "fp16 与 f32 输出差 {dmax:.3e}（量级 {scale:.3e}）——这不是舍入，\
         是权重段的解释错了（w_off 折算 / half4 步距）"
    );

    // 计时：两臂各按自己的速度校准重复次数，取较大者共用——重复次数
    // 相同才能让「每轮两臂的固定开销摊薄程度」一致。
    let reps = calibrate(&|r| run(&pr, "n_conv8", r)).max(calibrate(&|r| run(&prh, "n_conv8h", r)));

    // 奇偶轮换序：本机「一轮里先跑的那个」快 6~15%（见 bench/gpuhealth
    // 的 ab.sh 注释），固定次序的 A/B 等于没量。
    let (mut ta, mut tb) = (f64::INFINITY, f64::INFINITY);
    let mut per = Vec::new();
    for i in 0..4 {
        let (a, b) = if i % 2 == 0 {
            let a = run(&pr, "n_conv8", reps) / reps as f64;
            let b = run(&prh, "n_conv8h", reps) / reps as f64;
            (a, b)
        } else {
            let b = run(&prh, "n_conv8h", reps) / reps as f64;
            let a = run(&pr, "n_conv8", reps) / reps as f64;
            (a, b)
        };
        ta = ta.min(a);
        tb = tb.min(b);
        per.push(format!("{a:.3}/{b:.3}"));
    }
    eprintln!(
        "[mb] CTC 头生产内核（reps={reps}，逐轮 f32/fp16: {}）：\
         f32 {ta:.3} ms → fp16 {tb:.3} ms = {:.2}×",
        per.join(" "),
        ta / tb
    );
    assert!(tb < ta, "fp16 臂没有比 f32 快（{tb:.3} vs {ta:.3}）");
}

/// **batch 扫描**：把常驻 warp 数翻倍，看**单份**时间掉不掉。
///
/// 见 `build_conv_case_batch` 的头注释——这是目前唯一不靠「砍掉某样东西」
/// 来做的实验：`p_wconst`/`p_wsmall`/`p_fmaonly` 都同时改了延迟与字节，
/// 只能证伪「是带宽」或「是延迟」，**不能定位**是哪一个。
///
/// 判读：
///   · 每份时间随 batch **下降**（总时涨得比 batch 慢）→ 停顿没被藏住，
///     GPU 在等内存而周围没有别的 warp 可切 ⇒ 处方是提高并行度；
///   · 每份时间**持平**（总时 ∝ batch）→ 访存吞吐已打满，加 warp 无用
///     ⇒ 只能改访存结构（暂存 / 改布局）。
///
/// 用**生产内核** `n_conv8`（不是 `p_base`）：1×1 上它可能走窗口序
/// （门 `taps==1 && cin4v*nv>=4096`），那才是生产里真跑的代码。
///
/// ⚠️ batch 一大，工作集就超出 L2（M4 的 SLC 十几 MB），DRAM 流量随之
/// 上升。所以判读看的是**趋势**（每份时间单调下降 = 有得藏），不是把
/// batch 推到最大就最好——生产上还有个「合批就得补宽度」的代价。
/// **真实 rec 形状的网格伸缩扫描**——回答「13× 里有多少是网格太小」。
///
/// 动机：`fma_peak_reference` 测出峰值要 `grid >= 810`（`grid=13` 只有
/// 53% 峰值、`grid=102` 才 91%）。而 `conv8_batch_sweep` 上面那三个 case
/// 的 b=1 网格已经是 58 起——**从来没扫过低网格档**，而生产里 rec 的
/// conv8 dispatch 网格是 **9 / 17 / 74**（`QPPOCR_METAL_PLAN_DUMP`）。
/// 如果网格依赖在真实形状上也成立，那连一个完美的内核都上不去，
/// 「重写内核」这件事就得先排队。
///
/// 形状逐字取自 `img-004` 的 rec 计划（`[1,3,48,232]`，173 条），
/// 乘 y 铺 batch 会把网格线性放大而**每线程工作量、访存形状、指令序列
/// 一个字节都不改**，权重各份共用（`n_w` 不随 batch 增长）⇒ 权重保持
/// L2 热，变量只剩「常驻 warp 数」。这是唯一能不动内核就测网格的做法。
///
/// 判读：每份时间随 batch **下降** ⇒ 网格是瓶颈（处方是提并行度）；
/// **持平** ⇒ 网格无罪，13× 全在访存结构上（与 §4 定案一致）。
#[test]
fn conv8_grid_scaling_real_shapes() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");
    // (h, w, cin, cout) 取自 `QPPOCR_METAL_PLAN_DUMP`；b=1 网格见 label。
    let cases = [
        ConvCase { label: "Conv.42  3x58 c384->192 (grid= 9)", h: 6, w: 58, cin: 384, cout: 192, kh: 1, kw: 1 },
        ConvCase { label: "Conv.49  3x58 c384->768 (grid=17)", h: 3, w: 58, cin: 384, cout: 768, kh: 1, kw: 1 },
        ConvCase { label: "Conv.41  6x58 c192->384 (grid=17)", h: 6, w: 58, cin: 192, cout: 384, kh: 1, kw: 1 },
        ConvCase { label: "MatMul.12  1x29 c120->18712 (grid=74)", h: 1, w: 29, cin: 120, cout: 18712, kh: 1, kw: 1 },
    ];
    // 要盖到 grid>=810：grid=9 那档得 b=90。取 96 一档给足余量。
    const BATCHES: [u32; 7] = [1, 3, 12, 24, 48, 96, 192];

    for c in &cases {
        eprintln!("[mb] ── {} ──", c.label);
        let mut base_ms = f64::NAN;
        for batch in BATCHES {
            let pr = build_conv_case_batch(&ctx.inner.device, c, batch, false);
            let run = |reps: u32| -> f64 {
                let t = Instant::now();
                submit_compute(&ctx.inner.queue, |enc| {
                    for _ in 0..reps {
                        ks.dispatch(enc, "n_conv8", &pr.buf, pr.p_off, pr.groups)
                            .expect("dispatch");
                        let res: [&metal::ResourceRef; 1] = [&pr.buf];
                        enc.memory_barrier_with_resources(&res);
                    }
                })
                .expect("提交");
                t.elapsed().as_secs_f64() * 1000.0
            };
            let t = time_dispatch(&run);
            let per = t.ms / batch as f64;
            if batch == 1 {
                base_ms = per;
            }
            eprintln!(
                "[mb]   b={batch:>3}  grid={:<13} 单份 {per:>7.3} ms  {:.2}×b=1  {:>7.1} GFLOPS",
                format!("{:?}", pr.groups),
                per / base_ms,
                pr.flops / (t.ms * 1e6),
            );
        }
    }
}

#[test]
fn conv8_batch_sweep() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");
    let cases = [
        ConvCase { label: "rec 6x208 c192->384 k1 (热)", h: 6, w: 208, cin: 192, cout: 384, kh: 1, kw: 1 },
        ConvCase { label: "rec 3x208 c384->768 k1 (热)", h: 3, w: 208, cin: 384, cout: 768, kh: 1, kw: 1 },
        ConvCase { label: "det 34x60 c192->384 k1", h: 34, w: 60, cin: 192, cout: 384, kh: 1, kw: 1 },
    ];
    const BATCHES: [u32; 5] = [1, 2, 4, 8, 16];

    for c in &cases {
        eprintln!("[mb] ── {} ──", c.label);
        let mut base_ms = f64::NAN;
        for batch in BATCHES {
            let pr = build_conv_case_batch(&ctx.inner.device, c, batch, false);
            let run = |reps: u32| -> f64 {
                let t = Instant::now();
                submit_compute(&ctx.inner.queue, |enc| {
                    for _ in 0..reps {
                        ks.dispatch(enc, "n_conv8", &pr.buf, pr.p_off, pr.groups)
                            .expect("dispatch");
                        let res: [&metal::ResourceRef; 1] = [&pr.buf];
                        enc.memory_barrier_with_resources(&res);
                    }
                })
                .expect("提交");
                t.elapsed().as_secs_f64() * 1000.0
            };
            let t = time_dispatch(&run);
            let per = t.ms / batch as f64; // 单份（单张图）的时间
            if batch == 1 {
                base_ms = per;
            }
            // 「相对 b=1 的单份时间」——这就是判据本身。1.00 = 加 warp 没用；
            // 明显 < 1 = 停顿本来就没藏住。
            let ratio = per / base_ms;
            let total_ms = t.ms;
            eprintln!(
                "[mb]   batch={batch:>2}  grid={:<12} 总 {total_ms:>8.3} ms  单份 {per:>7.3} ms  \
                 {ratio:.2}×b=1  {:>7.1} GFLOPS  离散×{:.2}",
                format!("{:?}", pr.groups),
                pr.flops / (t.ms * 1e6),
                t.spread
            );
        }
    }
}

/// 流式带宽标尺：用**生产内核** `n_elem`（op=0 relu，原地读写）扫一组
/// 尺寸，量「这块 GPU 上一个纯流式 float4 内核能跑多少 GB/s」。
///
/// 动机：真实 rec 计划的剖析里 `n_channel` 在 `[8,125,18710]`（75 MB，
/// 读+写 150 MB）上量到 78.9 ms ≈ **1.9 GB/s**——比 M4 的纸面带宽低
/// 约两个数量级。这张表把两种解释分开：
/// - 标尺也低 ⇒ 问题在设备/缓冲属性/测量口径，不在某个内核；
/// - 标尺很高 ⇒ 问题在那个内核的访存模式（那样才值得去改内核）。
///
/// 口径：`time_dispatch`（总时 / r，r 校准到盖过命令缓冲固定开销），
/// L2 热。GB/s 按**读 + 写**总字节算。
#[test]
fn mem_bandwidth_sweep() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");
    let device = &ctx.inner.device;
    let queue = &ctx.inner.queue;

    eprintln!("[mb] 流式带宽标尺（n_elem op=0 原地，字节 = 读 + 写）");
    for mb in [1usize, 4, 16, 64, 256] {
        let n4 = mb * 1024 * 1024 / 16; // float4 个数
        let words = (n4 * 4) as u32;
        let p_off = 64u32;
        let data_off = 256u32; // word 偏移（避开参数块，与 conv 用例同款）
        let total = (data_off as u64 + words as u64 + 64) * 4;
        let buf = device.new_buffer(total, metal::MTLResourceOptions::StorageModeShared);
        // SAFETY: `buf` 按 total 字节分配；参数块写在 word 64..71、数据段
        // 写在 word 256.. —— 两者都在 total 之内（words ≥ 64K 时恒成立）。
        unsafe {
            let base = buf.contents() as *mut u32;
            for i in 0..(total as usize / 4) {
                *base.add(i) = 0x3f00_0000; // 0.5（避开 0 与 denormal）
            }
            let p = base.add(p_off as usize);
            *p = 0; // op = relu
            *p.add(1) = data_off; // in_off（word 下标）
            *p.add(2) = data_off; // in2_off（单目不用）
            *p.add(3) = data_off; // out_off（原地）
            *p.add(4) = words; // n（元素数）
            *p.add(5) = 0;
            *p.add(6) = 0;
        }
        let groups = [n4.div_ceil(256) as u32, 1, 1];
        let run = |r: u32| -> f64 {
            let t = Instant::now();
            submit_compute(queue, |enc| {
                for _ in 0..r {
                    ks.dispatch(enc, "n_elem", &buf, p_off, groups).expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [&buf];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
            t.elapsed().as_secs_f64() * 1000.0
        };
        // 口径自查：把 r 与「单次（r=1）」一并打出来——`总时 / r` 与
        // 单次时间如果差一个量级，说明固定开销还没被摊掉，GB/s 就是假的。
        let r = calibrate(&run);
        let t = time_dispatch(&run);
        let one = run(1);
        let bytes = words as f64 * 4.0 * 2.0;
        eprintln!(
            "[mb] {:>4} MB  grid={:<12} r={:<5} {:>8.3} ms  {:>7.1} GB/s  \
             （r=1 单次 {:.3} ms）  离散×{:.2}",
            mb,
            format!("{:?}", groups),
            r,
            t.ms,
            bytes / (t.ms * 1e6),
            one,
            t.spread
        );
    }
}

/// 线程组尺寸扫描：同一份字节量、同一内核，只改 `threadsPerThreadgroup`。
///
/// 动机：`mem_bandwidth_sweep` 量到「每线程组 ≈ 4.1 µs，与组大小无关」，
/// 换算出来只相当于「同时只有一个线程组在飞」——这对 10 核 GPU 是不可
/// 思议的。两种解释要用这张表劈开：
/// - **每组固定成本主导**（组被串行化/每组启动开销 ≈4 µs）⇒ 加大组
///   能在同样 4 µs 里干更多活，吞吐随组大小**线性增长**；
/// - **每线程/带宽主导** ⇒ 改组大小几乎不动吞吐。
///
/// 只测一个 64 MB 缓冲区（超出 L2，是真 DRAM 口径），总线程数恒定
/// （= n/4），所以唯一变量就是组大小。
#[test]
fn threads_per_group_sweep() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");
    let device = &ctx.inner.device;
    let queue = &ctx.inner.queue;

    let mb = 64usize;
    let n4 = mb * 1024 * 1024 / 16;
    let words = (n4 * 4) as u32;
    let p_off = 64u32;
    let data_off = 256u32;
    let total = (data_off as u64 + words as u64 + 64) * 4;
    let buf = device.new_buffer(total, metal::MTLResourceOptions::StorageModeShared);
    // SAFETY: 同 `mem_bandwidth_sweep`——参数块在 word 64..71、数据在 256..。
    unsafe {
        let base = buf.contents() as *mut u32;
        for i in 0..(total as usize / 4) {
            *base.add(i) = 0x3f00_0000;
        }
        let p = base.add(p_off as usize);
        *p = 0;
        *p.add(1) = data_off;
        *p.add(2) = data_off;
        *p.add(3) = data_off;
        *p.add(4) = words;
        *p.add(5) = 0;
        *p.add(6) = 0;
    }
    let pipe = ks.get("n_elem").expect("n_elem 管线");
    eprintln!("[mb] 线程组尺寸扫描（{mb} MB，总线程数恒定 = {n4}）");
    for tg in [64u64, 128, 256, 512, 1024] {
        let groups = (n4 as u64).div_ceil(tg);
        let run = |r: u32| -> f64 {
            let t = Instant::now();
            submit_compute(queue, |enc| {
                for _ in 0..r {
                    // 走生产编码路径（只换线程组尺寸）：每线程仍是 1 个
                    // float4，总线程数恒定，唯一变量是组大小。
                    dispatch_with_tg(
                        enc,
                        pipe,
                        &buf,
                        p_off,
                        [groups as u32, 1, 1],
                        tg,
                    )
                    .expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [&buf];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
            t.elapsed().as_secs_f64() * 1000.0
        };
        let tt = time_dispatch(&run);
        let bytes = words as f64 * 4.0 * 2.0;
        eprintln!(
            "[mb]   tg={:<5} groups={:<7} {:>8.3} ms  {:>6.1} GB/s  \
             （{:.3} µs/组，离散×{:.2}）",
            tg,
            groups,
            tt.ms,
            bytes / (tt.ms * 1e6),
            tt.ms * 1e3 / groups as f64,
            tt.spread
        );
    }
}

/// 造一块 head 探针用的 arena（布局与 `plan.rs` 的 head_fuse 逐字对应）。
///
/// 返回 `(buf, p_off, out_off)`。`in_off = 4`（word 0 是 real_n 槽，0 =
/// 不限），数据在 `in_off..in_off+rows*cpad`，输出对 `out_off..`。
fn head_arena(
    device: &metal::DeviceRef,
    rows: usize,
    cols: usize,
    cpad: usize,
    seed: u64,
) -> (metal::Buffer, u32, u32) {
    let in_off = 4usize;
    let out_off = in_off + rows * cpad;
    let pbase = out_off + rows * 2 + 8;
    let words = pbase + 32;
    let buf = device.new_buffer(
        (words * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    // SAFETY: 缓冲独占、长度就是 words；映射持久，测试内无别名并发。
    unsafe {
        let base = buf.contents() as *mut u32;
        for i in 0..words {
            *base.add(i) = 0;
        }
        // 与 `kernel_tests::lcg` 同款（那边是私有 fn，兄弟模块取不到）：
        // 确定性 ±20 的 logits。取值本身不影响时间——exp1 与 argmax 都无
        // 数据相关分支。
        let mut s = seed;
        for i in 0..rows * cpad {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((s >> 33) as f32 / (1u64 << 31) as f32) * 40.0 - 20.0;
            *base.add(in_off + i) = v.to_bits();
        }
        let p = base.add(pbase);
        *p = in_off as u32;
        *p.add(1) = out_off as u32;
        *p.add(2) = rows as u32;
        *p.add(3) = cols as u32;
        *p.add(4) = cpad as u32;
        *p.add(5) = rows as u32; // rpb：real_n=0 时无意义
    }
    (buf, pbase as u32, out_off as u32)
}

/// `n_head_argmax` 的**线程组大小**扫描（`probe_head.metal`，见文件头注释）。
///
/// 真形状：small 档 rec 的四份计划，`QPPOCR_GPU_KERNEL_PROF` 的
/// `grid=[rows,1,1]` = **160 / 168 / 112 / 256**，`cols = 18710`
/// （`plan.rs` 的 `[T×18710]`；与同计划 `n_channel Add.204 grid=[2924,1,1]`
/// 互证：160×18712/4/256 = 2924 ✓）。plan1 的 `rows=160` ⇒ **整卡只有 160
/// 个线程组**在搬 18712 列 × 4 趟。
///
/// 变体只改线程组大小，**位级安全**：max 与 argmax 的结合律/平局规则与划分
/// 无关（见 `probe_head.metal` 头注释），phase 3 的 8 条 lane 链一字不动。
/// 所以先拿 `p_h_t256` 的输出当基准断言其余变体逐位相等，再量时间。
#[test]
fn head_argmax_tg_sweep() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let device = &ctx.inner.device;
    let queue = &ctx.inner.queue;
    let probes = super::conv_probe::ProbeSet::from_list(
        device,
        super::conv_probe::HEAD_PROBE_SHADERS,
    )
    .expect("编译 head 探针");
    // 断言位级相同的变体：换线程组大小（归约序变但 max/argmax 与序无关）
    // 与链展开（加法次序一字不改）。`p_h_s*` / `p_h_noexp` 故意不等，只量时间。
    let names = [
        "p_h_t256",
        "p_h_t512",
        "p_h_t1024",
        "p_h_u4",
        "p_h_u8",
        "p_h_u16",
        "p_h_u32",
        "p_h_u64",
    ];
    let cols = 18710usize;
    let cpad = cols.div_ceil(4) * 4;
    let shapes = [160usize, 168, 112, 256];

    // ── 位级：所有变体应与 256 组输出逐位相同 ──
    let mut nbad = 0usize;
    for rows in shapes {
        let mut want: Option<Vec<u32>> = None;
        for (vi, name) in names.iter().enumerate() {
            let (buf, p_off, out_off) = head_arena(device, rows, cols, cpad, 0xabcd_1234);
            submit_compute(queue, |enc| {
                dispatch_with_tg(
                    enc,
                    probes.pipe(name).expect("取管线"),
                    &buf,
                    p_off,
                    [rows as u32, 1, 1],
                    super::conv_probe::head_probe_tg(name),
                )
                .expect("dispatch");
            })
            .expect("提交");
            // SAFETY: out_off + rows*2 在分配内（见 head_arena 的 pbase 计算）。
            let got: Vec<u32> = unsafe {
                let base = buf.contents() as *const u32;
                (0..rows * 2).map(|i| *base.add(out_off as usize + i)).collect()
            };
            match &want {
                None => want = Some(got),
                Some(w) => {
                    let d = w.iter().zip(&got).filter(|(a, b)| a != b).count();
                    if d > 0 {
                        eprintln!("[head] ⚠️ rows={rows} {name} 与 t256 有 {d} 处位级分歧");
                    }
                    nbad += d;
                    assert_eq!(d, 0, "rows={rows} {name} 线程组大小扫描不是位级不变");
                }
            }
            let _ = vi;
        }
    }
    eprintln!(
        "[head] 位级：{} 个变体 × {} 个形状全部与 t256 逐位一致（分歧 {nbad}）",
        names.len(),
        shapes.len()
    );

    // ── 时间 ──
    // 列 = 变体。`s1/s2/s3` 是**前缀**（只跑前 k 个阶段），所以列间差
    // 就是那一阶段的成本；`noexp` 只换 exp1 多项式。
    let timed = [
        "p_h_t256",
        "p_h_t512",
        "p_h_t1024",
        "p_h_s1",
        "p_h_s2",
        "p_h_s3",
        "p_h_noexp",
        "p_h_u4",
        "p_h_u8",
        "p_h_u16",
        "p_h_u32",
        "p_h_u64",
    ];
    eprintln!("[head] n_head_argmax 扫描（cols={cols}，cpad={cpad}）");
    eprint!("[head] {:>5}", "rows");
    for name in timed {
        eprint!(" {:>9}", &name[4..]);
    }
    eprintln!();
    for rows in shapes {
        eprint!("[head] {:>5}", rows);
        for name in timed {
            let (buf, p_off, _out_off) = head_arena(device, rows, cols, cpad, 0xabcd_1234);
            let pipe = probes.pipe(name).expect("取管线");
            let tg = super::conv_probe::head_probe_tg(name);
            let run = |r: u32| -> f64 {
                let t = Instant::now();
                submit_compute(queue, |enc| {
                    for _ in 0..r {
                        dispatch_with_tg(enc, pipe, &buf, p_off, [rows as u32, 1, 1], tg)
                            .expect("dispatch");
                        let res: [&metal::ResourceRef; 1] = [&buf];
                        enc.memory_barrier_with_resources(&res);
                    }
                })
                .expect("提交");
                t.elapsed().as_secs_f64() * 1000.0
            };
            eprint!(" {:>9.3}", time_dispatch(&run).ms);
        }
        eprintln!();
    }
}

/// 把「墙钟 = 内核」与「墙钟 = 内核 + 逐条 barrier」劈开。
///
/// `mem_bandwidth_sweep` 量到 trivial relu 只有 ~2 GB/s，但同一个 GPU 上
/// 纯 FMA 探针能到 ~47% 峰值——读写路径与算力路径差 30 倍，这不可能同时
/// 是「DRAM 慢」（算力探针也走 L1/L2/DRAM）。本测试在同一内核、同一缓冲区
/// 上只改一件事：**循环里有没有 barrier**。
///
/// - 有 barrier（生产路径）：每 rep 都要等前一 rep 的写全局可见；
/// - 无 barrier：同 CB 内 r 条 dispatch 背靠背，写写无依赖，GPU 可重叠。
///
/// 写法上的注意：无 barrier 时 r 条 dispatch 全打在**同一地址**，硬件会
/// 合并/丢写，但走的仍是完整的 load→FMA→store 流水，作为带宽口径成立。
#[test]
fn mem_bandwidth_barrier_cost() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");
    let device = &ctx.inner.device;
    let queue = &ctx.inner.queue;

    let mb = 64usize;
    let n4 = mb * 1024 * 1024 / 16;
    let words = (n4 * 4) as u32;
    let p_off = 64u32;
    let data_off = 256u32;
    let total = (data_off as u64 + words as u64 + 64) * 4;
    let buf = device.new_buffer(total, metal::MTLResourceOptions::StorageModeShared);
    // SAFETY: 同 `mem_bandwidth_sweep`——参数块 word 64..71，数据 256..。
    unsafe {
        let base = buf.contents() as *mut u32;
        for i in 0..(total as usize / 4) {
            *base.add(i) = 0x3f00_0000;
        }
        let p = base.add(p_off as usize);
        *p = 0;
        *p.add(1) = data_off;
        *p.add(2) = data_off;
        *p.add(3) = data_off;
        *p.add(4) = words;
        *p.add(5) = 0;
        *p.add(6) = 0;
    }
    let pipe = ks.get("n_elem").expect("n_elem 管线");
    let tg = 256u64;
    let groups = (n4 as u64).div_ceil(tg) as u32;
    let bytes = words as f64 * 4.0 * 2.0;

    // 固定 r（不用 `calibrate`）：这里要的正是「每 rep 时间随 r 怎么变」，
    // 自动加 rep 会把自变量改掉。三轮取最小。
    let run = |r: u32, with: bool| -> f64 {
        let t = Instant::now();
        submit_compute(queue, |enc| {
            for _ in 0..r {
                dispatch_with_tg(enc, pipe, &buf, p_off, [groups, 1, 1], tg)
                    .expect("dispatch");
                if with {
                    let res: [&metal::ResourceRef; 1] = [&buf];
                    enc.memory_barrier_with_resources(&res);
                }
            }
            if !with {
                let res: [&metal::ResourceRef; 1] = [&buf];
                enc.memory_barrier_with_resources(&res);
            }
        })
        .expect("提交");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        ms / r as f64
    };
    eprintln!("[mb] barrier 成本（{mb} MB，tg={tg}，每格 = 单 rep ms）");
    for r in [1u32, 4, 16] {
        let mut row = String::new();
        for with in [true, false] {
            let per = (0..3).map(|_| run(r, with)).fold(f64::INFINITY, f64::min);
            row += &format!(
                "  barrier={:<1}: {:>7.3} ms/rep（{:>5.1} GB/s）",
                if with { "有" } else { "无" },
                per,
                bytes / (per * 1e6)
            );
        }
        eprintln!("[mb]   r={r:<3}{row}");
    }
}

/// **Shared vs Private**：同一内核、同样字节、同一网格，只换存储模式。
///
/// 上一格（`mem_bandwidth_barrier_cost`）已经排掉 barrier：trivial relu 在
/// 64 MB 上就是 66 ms/rep（128 MB 流量 ⇒ 2 GB/s），而这个 GPU 上纯 FMA
/// 探针能到 2154 GFLOPS（≈47% 峰值）。**算力走得动、内存走不动**，两者
/// 只能有一个解释：要么「Shared 的统一内存映射」本身是慢路径，要么整机
/// 内存带宽被别的客户端占了。
///
/// Private 是 GPU 独占显存（Apple 侧仍是同一块物理内存，但**不走** CPU
/// 可见映射、不做缓存维护），所以这一格是判决性的：
/// - Private 也 2 GB/s ⇒ 整机内存带宽被占/降频，与我方代码无关；
/// - Private 突然 100+ GB/s ⇒ 病根在 Shared 路径，处方是把 arena 改成
///   Private + 小号 staging 缓冲上传（这是能改的）。
///
/// 两个模式都用 `new_buffer_with_bytes` 建（连填充方式都一致），只差
/// `MTLResourceOptions`。
/// **缓冲建立路径 vs 达成带宽**：同一内核、同一字节数、同一网格、同一
/// 参数，只换缓冲是**怎么建出来的**。
///
/// 这一格是被一个刺眼的反差逼出来的：`mem_bandwidth_sweep`（`new_buffer`
/// + `contents()` 逐字写）量到 trivial relu 只有 2 GB/s，而刚写的
/// `storage_mode_bandwidth`（`new_buffer_with_data`）在同一台机器同一尺寸
/// 上量到 **99 GB/s**——差 48 倍。两者参数块、网格、线程组、内核全同，
/// 唯一剩下的变量就是分配路径。三条路径同场对比：
///
/// - **A 手写**：`new_buffer` + 逐 u32 写 `contents()`（生产 arena 走的那条）；
/// - **B 清零**：`new_buffer` + `write_bytes` 全量（同样是 CPU 脏页，但
///   不经逐字循环）；
/// - **C 拷入**：`new_buffer_with_data`（内核之外的一次 64 MB memcpy）。
///
/// （原计划的第四条「Private + 拷入」已去掉：`newBufferWithBytes:` 配
/// `StorageModePrivate` 在 M4 上直接 SIGSEGV，非受支持的组合。）
///
/// 若 A/B 慢而 C 快，则病根是「CPU 写过的 Shared 缓冲」，处方在生产侧是
/// **把 arena 的写入改成「建缓冲时就带初值」或干脆 staging→blit**；这类
/// 改动能一次性提速所有 memory-bound 内核，是当前最高价值的候选。
#[test]
fn buffer_creation_path_bandwidth() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");
    let device = &ctx.inner.device;
    let queue = &ctx.inner.queue;

    let mb = 64usize;
    let n4 = mb * 1024 * 1024 / 16;
    let words = (n4 * 4) as u32;
    let (p_off, data_off) = (64u32, 256u32);
    let total_words = data_off as usize + words as usize + 64;
    let total_bytes = (total_words * 4) as u64;
    let fill = |base: *mut u32| {
        // SAFETY: 调用方保证 `base` 指向 `total_words` 个 u32 的 Shared 映射。
        unsafe {
            for i in 0..total_words {
                *base.add(i) = 0x3f00_0000;
            }
            let p = base.add(p_off as usize);
            *p = 0;
            *p.add(1) = data_off;
            *p.add(2) = data_off;
            *p.add(3) = data_off;
            *p.add(4) = words;
            *p.add(5) = 0;
            *p.add(6) = 0;
        }
    };

    let shared = || metal::MTLResourceOptions::StorageModeShared;
    let a = device.new_buffer(total_bytes, shared());
    fill(a.contents() as *mut u32);
    let b = device.new_buffer(total_bytes, shared());
    // SAFETY: 刚建的 `total_bytes` Shared 缓冲，整段清零在界内。
    unsafe { std::ptr::write_bytes(b.contents() as *mut u8, 0, total_bytes as usize) };
    fill(b.contents() as *mut u32);
    let mut host = vec![0x3f00_0000u32; total_words];
    {
        let p = &mut host[p_off as usize..];
        p[0] = 0;
        p[1] = data_off;
        p[2] = data_off;
        p[3] = data_off;
        p[4] = words;
        p[5] = 0;
        p[6] = 0;
    }
    let c = device.new_buffer_with_data(
        host.as_ptr() as *const std::ffi::c_void,
        total_bytes,
        shared(),
    );

    let pipe = ks.get("n_elem").expect("n_elem 管线");
    let tg = 256u64;
    let groups = (n4 as u64).div_ceil(tg) as u32;
    let bytes = words as f64 * 4.0 * 2.0;

    for (label, buf) in [
        ("A new_buffer + 逐字写   ", &a),
        ("B new_buffer + 清零     ", &b),
        ("C new_buffer_with_data  ", &c),
    ] {
        let run = |r: u32| -> f64 {
            let t = Instant::now();
            submit_compute(queue, |enc| {
                for _ in 0..r {
                    dispatch_with_tg(enc, pipe, buf, p_off, [groups, 1, 1], tg)
                        .expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [buf];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
            t.elapsed().as_secs_f64() * 1000.0 / r as f64
        };
        let per = (0..3).map(|_| run(16)).fold(f64::INFINITY, f64::min);
        eprintln!(
            "[mb] {label}: {:>8.3} ms/rep  {:>6.1} GB/s（{mb} MB，tg={tg}）",
            per,
            bytes / (per * 1e6)
        );
    }
}

/// 读回 depthwise 输出段，与朴素 CPU depthwise 比最大绝对误差。
///
/// **只对 `p_dw_base` / `p_dw_nodiv` / `p_dw_nobc` 有意义**：`p_dw_now`
/// 与 `p_dw_fma` 是**故意算错**的诊断变体（前者把权重换成常数、后者不
/// 读输入），它们的输出不参与对拍。
fn dw_error(pr: &ConvProbe, c: &DwCase, no_pad: bool) -> f32 {
    let n = (pr.out_h * pr.out_w * c.c) as usize;
    // SAFETY: Shared 存储，`out_off_w` 起 `out_h*out_w*co4` 个 word 有效；
    // 调用方已 wait_until_completed。前 `out_w*c` 列是真实通道（尾部补齐）。
    let got = unsafe {
        std::slice::from_raw_parts((pr.buf.contents() as *const f32).add(pr.out_off_w as usize), n)
    };
    let want = cpu_dw_ref(&pr.input, &pr.weight, c, no_pad);
    got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)
}

/// `n_conv_dw` 热循环拆解：medium det 里它只做 **3.2% 的 FLOP**，却占
/// **15% 的设备时间**（12.6 / 82.4 ms），最大那条单条 6.54 ms 只有
/// **207 GFLOPS**（`n_conv8h` 同机 1400+）。拆法见 `probe_dw.metal`。
///
/// 判读：`p_dw_fma` 与 `p_dw_base` 的差 = 这一档**全部**可动的量；
/// 其中 `p_dw_nodiv` 吃掉的部分归**整数除法**、`p_dw_now` 吃掉的部分归
/// **权重装载**，剩下的归地址数学与边界分支（后者由同轮 no_pad 那组
/// 的 `p_dw_base` vs `p_dw_nobc` 单独钉死）。
#[test]
fn conv_dw_variant_scan() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    // 形状逐字取自 `QPPOCR_METAL_PLAN_DUMP` 的 medium det 计划块。dw 的
    // `h/w` 是**输入**分辨率（stride 1 时 = 输出）。
    let cases = [
        // 首恶：单条 6.54 ms、1.354 GFLOP。输入由 Resize.2/Add.158 给到 136x240。
        DwCase { label: "dw 9x9 c256 136x240 (Conv.64 首恶)", h: 136, w: 240, c: 256, kh: 9, kw: 9, stride: 1 },
        DwCase { label: "dw 9x9 c256  68x120 (Conv.62)", h: 68, w: 120, c: 256, kh: 9, kw: 9, stride: 1 },
        DwCase { label: "dw 9x9 c64  136x240 (Conv.69)", h: 136, w: 240, c: 64, kh: 9, kw: 9, stride: 1 },
        // 3x3 那批条数多（Conv.5/10/13/…），单条小但铺满整个骨干。
        DwCase { label: "dw 3x3 c128 136x240 (Conv.5)", h: 136, w: 240, c: 128, kh: 3, kw: 3, stride: 1 },
        DwCase { label: "dw 3x3 c512  34x60 ", h: 34, w: 60, c: 512, kh: 3, kw: 3, stride: 1 },
        // 5x5 那档：判据 `5..=9` 把它收进来了，但探针从没量过 —— det 里真有
        // 两条（Conv.21 c64 / Conv.27 c128），形状照抄 PLAN_DUMP。
        //   ⚠️ `DwCase::h/w` 是**输入**分辨率，且 k=5,p=2,s=1 时
        //   out = (in + 4 - 5) + 1 = in —— 想凑出 out=4 就要填 in=4，
        //   按 3x3 的直觉去「补两圈 pad」会得到另一个形状（已踩过）。
        DwCase { label: "dw 5x5 c64   4x32  (Conv.21)", h: 4, w: 32, c: 64, kh: 5, kw: 5, stride: 1 },
        DwCase { label: "dw 5x5 c128  2x32  (Conv.27)", h: 2, w: 32, c: 128, kh: 5, kw: 5, stride: 1 },
    ];
    let probes = super::conv_probe::ProbeSet::from_list(
        &ctx.inner.device,
        super::conv_probe::DW_PROBE_SHADERS,
    )
    .expect("编译 dw 探针");
    // 带 padding 的配置上只能跑这些；`p_dw_nobc` 会越界，见最后一个代码块。
    // 除 `p_dw_wide4` 外全部**同网格**；`wide4` 的线程数是它们的 1/4，网格
    // 由 `dw_groups_for_wide` 单独给 —— 它量的是「摊薄地址数学」，本就要
    // 换一个线程划分。
    // (变体, 结果是否应与 CPU 一致)。`false` 的是**故意算错**的诊断变体
    // ——各自删掉或改写了某样东西，它们的输出不参与对拍。
    let all: &[(&str, bool)] = &[
        ("p_dw_base", true),
        ("p_dw_nodiv", true),
        ("p_dw_wide4", true),
        ("p_dw_u9", true),
        ("p_dw_u3", true),
        ("p_dw_interior", true),
        ("p_dw_incr", true),
        ("p_dw_intun", true),
        ("p_dw_intinc", true),
        ("p_dw_now", false),
        ("p_dw_fma", false),
        ("p_dw_noin", false),
        ("p_dw_win", false),
        ("p_dw_tap1o3", false),
        ("p_dw_tap2o3", false),
        // 第三轮：线程组瓦片暂存（**网格不同**，见 `dw_groups_for_tile`）。
        // 结果应与 CPU 一致 —— 它只换了热循环的取数来源，没改累加次序。
        ("p_dw_tile", true),
        ("p_dw_tile9", true),
        ("p_dw_tilew", true),
    ];

    // ── `p_dw_interior` 的**派发前**安全证明 ────────────────────────
    //
    // 那个内核按「首尾两个 tap 都在界内」判内点，然后走一条**没有 compare**
    // 的循环体。谓词判宽了 ⇒ `uint` 回绕成天文数字地址 ⇒ 直接读飞 ⇒ GPU
    // 打瘸（要重启）。所以先把同一算式在 CPU 上对**每个输出位置**暴力跑
    // 一遍，确认「内点 ⟹ base 的每个 tap 都不会 `continue`」。不成立就
    // panic，一次 dispatch 都不发。
    for c in &cases {
        let (out_h, out_w, pad) = dw_dims(c, false);
        let (k_h, k_w, s) = (c.kh as i64, c.kw as i64, c.stride as i64);
        let (inh, inw) = (c.h as i64, c.w as i64);
        let mut n_inner = 0usize;
        for oh in 0..out_h as i64 {
            for ow in 0..out_w as i64 {
                let ih0 = oh * s - pad as i64;
                let iw0 = ow * s - pad as i64;
                // 与 MSL 逐字同形：只判首尾。
                let inner = ih0 >= 0 && ih0 + k_h - 1 < inh && iw0 >= 0 && iw0 + k_w - 1 < inw;
                if !inner {
                    continue;
                }
                n_inner += 1;
                // 暴力复算：base 的逐 tap 判断一次都不能触发。
                for ky in 0..k_h {
                    let ih = oh * s + ky - pad as i64;
                    assert!(
                        (0..inh).contains(&ih),
                        "{}：内点判宽了 —— oh={oh} ky={ky} → ih={ih} 越界（{}×{}，pad={pad}）。\
                         `p_dw_interior` 会算飞地址，禁止派发。",
                        c.label,
                        out_h,
                        out_w
                    );
                }
                for kx in 0..k_w {
                    let iw = ow * s + kx - pad as i64;
                    assert!(
                        (0..inw).contains(&iw),
                        "{}：内点判宽了 —— ow={ow} kx={kx} → iw={iw} 越界（{}×{}，pad={pad}）。\
                         `p_dw_interior` 会算飞地址，禁止派发。",
                        c.label,
                        out_h,
                        out_w
                    );
                }
            }
        }
        eprintln!(
            "[mb] ── {} ── 内点证明通过：{}/{} 个位置走无判断路径（{:.1}%）",
            c.label,
            n_inner,
            out_h as usize * out_w as usize,
            100.0 * n_inner as f64 / (out_h as f64 * out_w as f64)
        );
    }

    for c in &cases {
        let pr = build_dw_case(&ctx.inner.device, c, false, 0, 0);
        let m_dim = pr.out_h * pr.out_w;
        // 控制关：网格必须与 `plan.rs` 的 `div256(m_dim * cp / 4)` 逐位一致，
        // 否则线程覆盖不同，绝对值无从与生产对齐。
        assert_eq!(
            pr.groups,
            super::conv_probe::dw_groups_for(m_dim, pr.co4),
            "网格与 plan.rs 公式不一致（{}）",
            c.label
        );
        assert_eq!(pr.co4 % 16, 0, "wide4 要求 cv4 能被 4 整除（{}）", c.label);
        let wide = super::conv_probe::dw_groups_for_wide(m_dim, pr.co4);
        // 特化变体只在与模板常量相符的形状上有意义，其余整条剔除
        // （不剔除会拿常量算出的错地址乱读）。
        let pad0 = (c.kh - 1) / 2;
        let variants: Vec<(&str, bool)> = all
            .iter()
            .copied()
            .filter(|(n, _)| super::conv_probe::dw_spec_ok(n, c.kh, c.kw, pad0, c.stride))
            .collect();
        let tile = super::conv_probe::dw_groups_for_tile(m_dim, pr.out_w, pr.co4);
        let grid_of = |n: &str| match n {
            "p_dw_wide4" => wide,
            "p_dw_tile" | "p_dw_tile9" | "p_dw_tilew" => tile,
            _ => pr.groups,
        };

        let run = |pipe: &metal::ComputePipelineState, grid: [u32; 3], reps: u32| -> f64 {
            let t = Instant::now();
            submit_compute(&ctx.inner.queue, |enc| {
                for _ in 0..reps {
                    super::pipeline::dispatch_with(enc, pipe, &pr.buf, pr.p_off, grid)
                        .expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [&pr.buf];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
            t.elapsed().as_secs_f64() * 1000.0
        };

        let pipes: Vec<(&str, bool, &metal::ComputePipelineState, [u32; 3])> = variants
            .iter()
            .map(|(n, ok)| (*n, *ok, probes.pipe(n).expect("管线"), grid_of(n)))
            .collect();
        // 预热 + 就地取误差（变体逐个覆盖同一块输出，必须**跑完就立刻读**，
        // 否则读到的是最后一个变体的结果 —— 这条在 conv8 扫描里踩过）+ 校准。
        let mut errs = Vec::new();
        let mut reps = Vec::new();
        for (_, _, p, g) in &pipes {
            let _ = run(p, *g, 1);
            errs.push(dw_error(&pr, c, false));
            reps.push(calibrate(&|r| run(p, *g, r)));
        }
        for (i, (n, ok, _, _)) in pipes.iter().enumerate() {
            if *ok {
                assert!(
                    errs[i] < 1e-2,
                    "{} 的 {} 与 CPU 对拍失败：最大绝对差 {}",
                    c.label,
                    n,
                    errs[i]
                );
            }
        }

        // 轮转交错取最小（本机先跑的那条快 6~15%）
        let mut best = vec![f64::INFINITY; pipes.len()];
        for _ in 0..3 {
            for (i, (_, _, p, g)) in pipes.iter().enumerate() {
                let t = run(p, *g, reps[i]) / reps[i] as f64;
                best[i] = best[i].min(t);
            }
        }

        let base = best[0].max(1e-6);
        eprintln!(
            "[mb] ── {} ── grid={:?}  {} GFLOP  下界 {:.2} ms（{:.0} GB/s）",
            c.label,
            pr.groups,
            pr.flops / 1e9,
            pr.bytes / 93.0e6,
            pr.bytes / (base * 1e6)
        );
        for (i, (n, ok, _, _)) in pipes.iter().enumerate() {
            let ms = best[i].max(1e-6);
            eprintln!(
                "[mb]    {:<11} {:>8.3} ms  {:>7.1} GFLOPS  {:.2}× base{}",
                n,
                ms,
                pr.flops / (ms * 1e6),
                base / ms,
                if *ok { "" } else { "   （故意算错）" }
            );
        }
    }

    // ── 边界分支的代价（只能在无 padding 的配置上量）─────────────────
    //
    // `p_dw_nobc` 删掉了 `ih/iw` 的界内判断。带 padding 时内点之外的 tap
    // 会算出负的 `ih`，`uint` 回绕成天文数字 —— 直接读飞。所以换一份
    // pad=0、输出只覆盖「每个 tap 恒在界内」的配置：此时判断永真，
    // 删掉它**不改变结果**，两臂都能过 CPU 对拍。
    //
    // 注意这两条的绝对值**不与上面那块可比**（m 从 32640 变 29696），
    // 只有它俩之间的比值有意义。
    {
        let c0 = &cases[0];
        let prn = build_dw_case(&ctx.inner.device, c0, true, 0, 0);
        let groups = prn.groups;
        let names = ["p_dw_base", "p_dw_nobc"];
        let run = |pipe: &metal::ComputePipelineState, reps: u32| -> f64 {
            let t = Instant::now();
            submit_compute(&ctx.inner.queue, |enc| {
                for _ in 0..reps {
                    super::pipeline::dispatch_with(enc, pipe, &prn.buf, prn.p_off, groups)
                        .expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [&prn.buf];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
            t.elapsed().as_secs_f64() * 1000.0
        };
        let pipes: Vec<&metal::ComputePipelineState> =
            names.iter().map(|n| probes.pipe(n).expect("管线")).collect();
        let mut reps = Vec::new();
        let mut errs = Vec::new();
        for p in &pipes {
            let _ = run(p, 1);
            errs.push(dw_error(&prn, c0, true));
            reps.push(calibrate(&|r| run(p, r)));
        }
        for i in 0..2 {
            assert!(
                errs[i] < 1e-2,
                "no_pad 配置上 {} 与 CPU 对拍失败：{}（{}）",
                names[i],
                errs[i],
                c0.label
            );
        }
        let mut best = [f64::INFINITY; 2];
        for _ in 0..3 {
            for (i, p) in pipes.iter().enumerate() {
                best[i] = best[i].min(run(p, reps[i]) / reps[i] as f64);
            }
        }
        eprintln!(
            "[mb] ── {} no_pad（pad=0，输出 {}x{}）── 边界分支的代价",
            c0.label, prn.out_h, prn.out_w
        );
        for (i, n) in names.iter().enumerate() {
            eprintln!(
                "[mb]    {:<11} {:>8.3} ms  {:>7.1} GFLOPS  {:.2}× base",
                n,
                best[i],
                prn.flops / (best[i] * 1e6),
                best[0] / best[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// n_convt 探针的驱动（内核见 `probe_convt.metal`，变体表见其头注释）
// ---------------------------------------------------------------------------

/// 一个已组好的 ConvTranspose 用例的 arena。
///
/// **单开一份**而不复用 `build_conv_case_geom_w`：参数块语义不同——conv8 是
/// `[…,cin4v,taps,kW,sH,sW,padT,padL,act,…]`（21 词），ConvTranspose 是
/// `[…,cin(已 pad),ci4v,sH,sW,kW,act,c1,c2,c3]`（18 词），第 9/10/11 个位置
/// 就会对错（conv8 的 `cin4v` 在 P+9，convt 的 `cin` 在 P+9）。混用 = 内核按
/// 错参数跑，量出来的数是假的。
struct CtProbe {
    buf: metal::Buffer,
    p_off: u32,
    out_off_w: u32,
    n_out: u32,
    m_dim: u32,
    in_h: u32,
    in_w: u32,
    cin: u32,
    cout: u32,
    co4: u32,
    ci4: u32,
    /// NHWC（含 cpad4）输入，CPU 对拍用。
    input: Vec<f32>,
    /// ONNX 布局 `[Ci][Co][2][2]` 权重，CPU 对拍用。
    weight: Vec<f32>,
    flops: f64,
    bytes: f64,
}

/// 组一个 2×2 stride-2、无 pad 的 ConvTranspose 用例（`n_convt` 在 det 里
/// 的两条都是这个几何）。字段顺序与 `plan.rs` 的 ConvTranspose 分支逐字一致。
fn build_ct_case(device: &metal::DeviceRef, cin: u32, cout: u32, in_h: u32, in_w: u32) -> CtProbe {
    assert!(in_w % 4 == 0, "p_ct_tap4w4 要求 inW%4==0");
    let ci4 = cin.div_ceil(4) * 4;
    let co4 = cout.div_ceil(4) * 4;
    let (out_h, out_w) = (in_h * 2, in_w * 2);
    let m_dim = out_h * out_w;
    let taps = 4u32;
    let nv = co4 / 4;

    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 0.25
    };
    let input: Vec<f32> = (0..(in_h * in_w * ci4) as usize).map(|_| rnd()).collect();
    let weight: Vec<f32> = (0..(cin * cout * 4) as usize).map(|_| rnd()).collect();

    // 重排成 [tap][ci][nv][4]（float 序）——与 `qppocr_gpu::nhwc::repack_convt_w`
    // 逐字同布局：元素 (t, ci, oc) 在 `((t*ci4 + ci)*nv + oc/4)*4 + oc%4`。
    let mut w_repacked = vec![0f32; (taps * ci4 * nv * 4) as usize];
    for t in 0..taps {
        let (ky, kx) = (t / 2, t % 2);
        for ci_i in 0..ci4 {
            for oc in 0..co4 {
                let dst = (((t * ci4 + ci_i) * nv) + oc / 4) * 4 + oc % 4;
                w_repacked[dst as usize] = if ci_i < cin && oc < cout {
                    weight[((ci_i * cout + oc) * 4 + ky * 2 + kx) as usize]
                } else {
                    0.0
                };
            }
        }
    }

    let n_in = in_h * in_w * ci4;
    let n_w = taps * ci4 * co4;
    let n_out = m_dim * co4;
    // 参数块占 word 64..82，张量必须让开（同 `build_conv_case_geom_w`：
    // 输入段从 word 256 起，否则第一段就把参数覆盖成浮点位型 → 巨循环）。
    let p_words = 64u32;
    let in_off_w = 256u32;
    let w_off_w = in_off_w + n_in;
    let b_off_w = w_off_w + n_w;
    let out_off_w = b_off_w + co4;
    let total = (out_off_w + n_out + 64) as u64;
    let buf = device.new_buffer(total * 4, metal::MTLResourceOptions::StorageModeShared);

    let base = buf.contents() as *mut u32;
    // SAFETY: `base` 是 Shared 存储的非空基址，缓冲按 `total` 个 word 分配；
    // 段按 word 依次相接（params[64..82] → in[256..] → w → b → out），每个
    // 下标都 < total。参数块（≤82）与输入段（≥256）之间的空隙是刻意的。
    unsafe {
        let p = base.add(p_words as usize);
        *base = 0;
        // 偏移一律是 **word（f32 元素）下标**——内核内部 `>>2` 折算成 float4。
        let words: [u32; 18] = [
            in_off_w, w_off_w, b_off_w, out_off_w, // 0..3
            m_dim, co4, out_w, in_w, in_h,         // 4..8
            ci4, ci4 / 4, 2, 2, 2, 0,              // 9..14 cin,ci4v,sH,sW,kW,act
            0, 0, 0,                               // 15..17 c1,c2,c3（act=0 用不到）
        ];
        for (i, v) in words.iter().enumerate() {
            *p.add(i) = *v;
        }
        std::ptr::copy_nonoverlapping(input.as_ptr(), base.add(in_off_w as usize) as *mut f32, n_in as usize);
        std::ptr::copy_nonoverlapping(w_repacked.as_ptr(), base.add(w_off_w as usize) as *mut f32, n_w as usize);
        // bias 段存在但全 0：把 `b_off != OFF_NONE` 那条分支也照到（生产里
        // det 的 ConvTranspose 有 bias）。每线程多一条广播装载，可忽略。
        std::ptr::write_bytes(base.add(b_off_w as usize) as *mut f32, 0, co4 as usize);
        // 哨兵 7.0：跑完还是 7.0 = 内核没写（早退/网格算错），一眼可辨。
        let op = base.add(out_off_w as usize) as *mut f32;
        for i in 0..n_out as usize {
            *op.add(i) = 7.0;
        }
    }

    let flops = m_dim as f64 * co4 as f64 * ci4 as f64 * 2.0;
    let bytes = (n_in + n_out) as f64 * 4.0 + n_w as f64 * 4.0;
    CtProbe {
        buf, p_off: p_words, out_off_w, n_out, m_dim, in_h, in_w,
        cin, cout, co4, ci4, input, weight, flops, bytes,
    }
}

/// 朴素 ConvTranspose（2×2、stride 2、无 pad）：NHWC in、ONNX `[Ci][Co][2][2]` 权重。
fn cpu_convt_ref(pr: &CtProbe) -> Vec<f32> {
    let (ih_n, iw_n) = (pr.in_h as usize, pr.in_w as usize);
    let (oh_n, ow_n) = (ih_n * 2, iw_n * 2);
    let (cin, cout) = (pr.cin as usize, pr.cout as usize);
    let (ci4, co4) = (pr.ci4 as usize, pr.co4 as usize);
    let mut out = vec![0f32; oh_n * ow_n * co4];
    for ih in 0..ih_n {
        for iw in 0..iw_n {
            for ic in 0..cin {
                let xv = pr.input[(ih * iw_n + iw) * ci4 + ic];
                if xv == 0.0 {
                    continue;
                }
                for ky in 0..2usize {
                    for kx in 0..2usize {
                        let orow = (ih * 2 + ky) * ow_n + iw * 2 + kx;
                        for oc in 0..cout {
                            let w = pr.weight[((ic * cout + oc) * 2 + ky) * 2 + kx];
                            out[orow * co4 + oc] += xv * w;
                        }
                    }
                }
            }
        }
    }
    out
}

/// 回读输出段。
fn ct_read_out(pr: &CtProbe) -> Vec<f32> {
    // SAFETY: out_off_w + n_out ≤ total（见 build_ct_case 的段布局）；Shared 存储，
    // 调用方已 wait_until_completed。
    unsafe {
        std::slice::from_raw_parts(
            (pr.buf.contents() as *const f32).add(pr.out_off_w as usize),
            pr.n_out as usize,
        )
        .to_vec()
    }
}

/// `n_convt` 的余量归因：三条流（输入/权重/输出）的**字节**与**装载条数**分开量，
/// 再试两个可落地的候选（4-tap 空间分块、叠空间复用）。见 `probe_convt.metal`。
#[test]
fn convt_probe_attribution() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let probes = ProbeSet::from_list(&ctx.inner.device, super::conv_probe::CT_PROBE_SHADERS)
        .expect("编译 n_convt 探针");

    // det 里真实的两条（plan dump：grid 分别是 [12960,1,1] 与 [3240,1,1]）。
    let shapes: [(&str, u32, u32, u32, u32); 2] = [
        ("ConvTranspose.0", 64, 64, 240, 216),
        ("ConvTranspose.2", 64, 4, 480, 432),
    ];
    let arms = [
        "p_ct_base", "p_ct_ci4", "p_ct_tap4", "p_ct_tap4w4", // 结果正确
        "p_ct_noin", "p_ct_now", "p_ct_fmaonly",             // 诊断：砍流
        "p_ct_ismall", "p_ct_wsmall", "p_ct_osmall",         // 诊断：留条数砍字节
    ];

    for (label, cin, cout, h, w) in shapes {
        let pr = build_ct_case(&ctx.inner.device, cin, cout, h, w);
        let base_g = super::conv_probe::ct_groups_for("p_ct_base", pr.m_dim, h, w, pr.co4);

        let run = |arm: &str, reps: u32| -> f64 {
            let pipe = probes.pipe(arm).expect("取管线");
            let g = super::conv_probe::ct_groups_for(arm, pr.m_dim, h, w, pr.co4);
            let t = Instant::now();
            submit_compute(&ctx.inner.queue, |enc| {
                for _ in 0..reps {
                    super::pipeline::dispatch_with(enc, pipe, &pr.buf, pr.p_off, g)
                        .expect("dispatch");
                    let res: [&metal::ResourceRef; 1] = [&pr.buf];
                    enc.memory_barrier_with_resources(&res);
                }
            })
            .expect("提交");
            t.elapsed().as_secs_f64() * 1000.0
        };

        // ---- 正确性：p_ct_base 对 CPU，另三个「结果正确」的臂对 p_ct_base
        let want = cpu_convt_ref(&pr);
        let _ = run("p_ct_base", 1);
        let got = ct_read_out(&pr);
        let err_base = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
        let base_out = got.clone();
        let mut errs = vec![("p_ct_base", err_base)];
        for arm in ["p_ct_ci4", "p_ct_tap4", "p_ct_tap4w4"] {
            let _ = run(arm, 1);
                let g = ct_read_out(&pr);
            // 与 p_ct_base 逐位比：这三个臂的累加顺序与 base 相同，应当**位级一致**。
            let d = g.iter().zip(&base_out).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            errs.push((arm, d));
        }

        eprintln!(
            "[ct] {label}: cin={cin} cout={cout} in={h}×{w} out={}×{} mDim={} nv={} 网格 base=[{},1,1] tap4=[{},1,1] tap4w4=[{},1,1]",
            h * 2, w * 2, pr.m_dim, pr.co4 / 4,
            base_g[0],
            super::conv_probe::ct_groups_for("p_ct_tap4", pr.m_dim, h, w, pr.co4)[0],
            super::conv_probe::ct_groups_for("p_ct_tap4w4", pr.m_dim, h, w, pr.co4)[0],
        );
        eprintln!(
            "[ct] {label}: FLOPs(真)={:.4} GF（flopsmap 记 4×={:.4}）  下界流量={:.1} MB ⇒ DRAM 下界 {:.2} ms @77GB/s",
            pr.flops / 1e9,
            pr.flops * 4.0 / 1e9,
            pr.bytes / 1e6,
            pr.bytes / 77e9 * 1000.0,
        );
        eprintln!(
            "[ct] {label}: 最大绝对误差 vs CPU / vs base: {}",
            errs.iter().map(|(n, e)| format!("{n}={e:.3e}")).collect::<Vec<_>>().join("  ")
        );

        // ---- 计时：三轮，奇偶轮换序（顺序效应在本机是 6~15%，见 ab.sh 那条）
        let na = arms.len();
        let mut best = vec![f64::INFINITY; na];
        for round in 0..3 {
            let mut order: Vec<usize> = (0..na).collect();
            if round % 2 == 1 {
                order.reverse();
            }
            for a in order {
                let ms = time_dispatch(&|r| run(arms[a], r)).ms;
                best[a] = best[a].min(ms);
            }
        }
        let b0 = best[0];
        eprintln!(
            "[ct] {label}  {:>9} {:>9} {:>9} {:>9} | {:>9} {:>9} {:>9} | {:>9} {:>9} {:>9}",
            "base", "ci4", "tap4", "tap4w4", "noin", "now", "fmaonly", "ismall", "wsmall", "osmall"
        );
        eprintln!(
            "[ct] {label}  {:>9.3} {:>9.3} {:>9.3} {:>9.3} | {:>9.3} {:>9.3} {:>9.3} | {:>9.3} {:>9.3} {:>9.3}",
            best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[7], best[8], best[9]
        );
        eprintln!(
            "[ct] {label}  {:>9} {:>8.2}× {:>8.2}× {:>8.2}× | {:>8.2}× {:>8.2}× {:>8.2}× | {:>8.2}× {:>8.2}× {:>8.2}×",
            "base/→",
            b0 / best[1], b0 / best[2], b0 / best[3],
            b0 / best[4], b0 / best[5], b0 / best[6],
            b0 / best[7], b0 / best[8], b0 / best[9]
        );
        let gf = pr.flops / 1e9;
        eprintln!(
            "[ct] {label}: base {:.0} GFLOPS（峰值 6.3% 量级）/ tap4 {:.0} / tap4w4 {:.0}",
            gf / best[0] * 1000.0, gf / best[2] * 1000.0, gf / best[3] * 1000.0
        );
    }
}

/// **生产内核** `n_convt` / `n_convt4w` 与 CPU 的逐点对拍（同一 arena、同一
/// 参数块，只有内核名与网格不同）。
///
/// 为什么必须是生产内核而不是探针：`probe_convt.metal` 里的副本**探针自己
/// 编译**，与 `KernelSet` 编译的那份是两个二进制。两者同源只能靠这一条测试
/// 钉住——`conv8_mma3_matches_cpu` 用同一手法抓过 `n_conv8_mma3`。
///
/// 三条断言：
/// 1. `n_convt` 对 CPU：只查量法本身（算子语义、权重排布、参数块）；
/// 2. `n_convt4w` 对 CPU：新内核自己算对；
/// 3. `n_convt4w` 对 `n_convt`：**位级相等**——两个内核的累加次序逐 ci 相同，
///    所以差异必须恰好是 0.0。任何非零都说明「换个加法顺序」之外还动了别的东西。
#[test]
fn convt_prod_matches_cpu() {
    let _g = gpu_guard();
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    // 两个真实 det 形状 + 一个「nv=4 的边界档」（门禁 nv>=4 的下沿）+
    // 一个 Batch 形状（网格 y 维）。
    struct Case(&'static str, u32, u32, u32, u32);
    let cases = [
        Case("ConvTranspose.0 (nv=16)", 64, 64, 240, 216),
        Case("ConvTranspose.2 (nv=1)", 64, 4, 480, 432),
        Case("nv=4 下沿", 64, 16, 40, 36),
        Case("小形状", 8, 8, 8, 8),
    ];

    for Case(label, cin, cout, h, w) in cases {
        let pr = build_ct_case(&ctx.inner.device, cin, cout, h, w);
        let want = cpu_convt_ref(&pr);
        let g4 = super::conv_probe::ct_groups_for("p_ct_tap4w4", pr.m_dim, h, w, pr.co4);
        let g1 = super::conv_probe::ct_groups_for("p_ct_base", pr.m_dim, h, w, pr.co4);

        let run = |name: &str, g: [u32; 3]| {
            submit_compute(&ctx.inner.queue, |enc| {
                ks.dispatch(enc, name, &pr.buf, pr.p_off, g).expect("dispatch");
            })
            .expect("提交");
        };

        // 哨兵先复位，否则「没写」会被读成上一臂的结果。
        let op = pr.buf.contents() as *mut f32;
        let reset = || unsafe {
            for i in 0..(pr.n_out as usize) {
                *op.add(pr.out_off_w as usize + i) = 7.0;
            }
        };
        let maxdiff = |a: &[f32], b: &[f32]| {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
        };

        reset();
        run("n_convt", g1);
        let o1 = ct_read_out(&pr);
        let e1 = maxdiff(&o1, &want);

        reset();
        run("n_convt4w", g4);
        let o4 = ct_read_out(&pr);
        let e4 = maxdiff(&o4, &want);

        let e14 = maxdiff(&o1, &o4);
        // 哨兵：内核没写的话输出全 7.0，maxdiff 会是大数——上面两条会先炸。
        assert!(e1 < 1e-4, "{label}: n_convt 对 CPU 误差 {e1:.3e}");
        assert!(e4 < 1e-4, "{label}: n_convt4w 对 CPU 误差 {e4:.3e}");
        assert!(e14 == 0.0, "{label}: n_convt4w 与 n_convt 不是位级相等（{e14:.3e}）");
        eprintln!(
            "[ct/prod] {label}: cin={cin} cout={cout} in={h}×{w} 网格 n_convt=[{},1,1] n_convt4w=[{},1,1]  vs CPU {e1:.2e} / {e4:.2e}  vs 彼此 {e14:.1e} ✅",
            g1[0], g4[0]
        );
    }
}
