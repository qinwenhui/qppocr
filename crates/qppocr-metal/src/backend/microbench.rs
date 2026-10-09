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

/// 把 NCHW 索引写成 NHWC 下标，做一次朴素卷积（对拍基准）。
fn cpu_conv_nhwc(
    input: &[f32],
    weight: &[f32],
    c: &ConvCase,
    out_h: u32,
    out_w: u32,
) -> Vec<f32> {
    let (cin, cout, kh, kw) = (c.cin as usize, c.cout as usize, c.kh as usize, c.kw as usize);
    let (pt, pl) = ((kh / 2) as i64, (kw / 2) as i64);
    let (ih_max, iw_max) = (c.h as i64, c.w as i64);
    // weight 输入是 [cout][cin][kh][kw]（ONNX 布局）
    let mut out = vec![0f32; out_h as usize * out_w as usize * cout];
    // 1×1/无 pad 快径：逐字等价于下面的通用循环（ih=oh、iw=ow 恒在界内，
    // 内层只剩 ic 的点积），但去掉两层边界判断与除法。**rec 的 CTC 头
    // 分类层**（216×1 c120→18712，485 M MAC）走的是这里——通用循环跑
    // 一遍要几十秒，变体扫描里每个变体都要对拍一次，根本扫不动。
    if kh == 1 && kw == 1 && pt == 0 && pl == 0 && cin == c.cin as usize {
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
                            let ih = oh as i64 + ky as i64 - pt;
                            let iw = ow as i64 + kx as i64 - pl;
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
    co4: u32,
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
    assert!(batch >= 1, "batch 至少 1");
    let (pt, pl) = (c.kh / 2, c.kw / 2);
    let out_h = (c.h + 2 * pt - c.kh) + 1;
    let out_w = (c.w + 2 * pl - c.kw) + 1;
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

    let n_in = c.h * c.w * ci4 * batch;
    let n_w = taps * ci4 * co4; // 权重**不随 batch 增长**（各份共用一份）
    let n_out = m_dim * co4 * batch;
    // 参数块占 word 64..64+21；张量必须让开，否则输入第一段就把
    // taps/cin4v 覆盖成浮点位型——内核会按天文数字循环，直接挂死。
    let p_words = 64u32;
    let in_off_w = 256u32;
    let w_off_w = in_off_w + n_in;
    let b_off_w = w_off_w + n_w;
    let out_off_w = b_off_w + co4;
    let total = (out_off_w + n_out + 64) as u64;
    let buf = device.new_buffer(total * 4, metal::MTLResourceOptions::StorageModeShared);

    let base = buf.contents() as *mut u32;
    // SAFETY: `base` 是 Shared 存储的非空基址，缓冲按 `total` 个 word 分配。
    // 段布局按 word 依次相接：params[64..85] → in[256..] → w → b → out，
    // 每段起点都是上一段起点加其长度，故所有下标 < total；参数块（≤85）
    // 与输入段（≥256）之间留的空隙是刻意的，见 `in_off_w` 处注释。
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
            1,             // sH
            1,             // sW
            pt,            // padT
            pl,            // padL
            0,             // act
            0,             // c1
            0,             // c2
            0,             // c3
            0xFFFF_FFFF,   // r_off = OFF_NONE
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
            // `n_w` 个 half 写进段的前半。
            let w16: Vec<u16> = w_repacked.iter().map(|&v| f32_to_f16_bits(v)).collect();
            let wp = base.add(w_off_w as usize) as *mut u16;
            std::ptr::write_bytes(wp, 0, n_w as usize);
            std::ptr::copy_nonoverlapping(w16.as_ptr(), wp, n_w as usize);
        } else {
            std::ptr::copy_nonoverlapping(w_repacked.as_ptr(), base.add(w_off_w as usize) as *mut f32, n_w as usize);
        }
        // bias 清零（b_off 有效但全 0）
        std::ptr::write_bytes(base.add(b_off_w as usize) as *mut f32, 0, co4 as usize);
        // 输出区铺哨兵 7.0：跑完还是 7.0 = 内核根本没写（早退/没派发），
        // 是排障时一眼分辨「没跑」和「跑了但累加器为空」的关键。
        let op = base.add(out_off_w as usize) as *mut f32;
        for i in 0..n_out as usize {
            *op.add(i) = 7.0;
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
        co4,
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
