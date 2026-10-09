#![allow(clippy::undocumented_unsafe_blocks)] // 裸内核取证：持久映射指针操作，逐块 SAFETY 注释无信息量
//! P1 内核位级冒烟：`n_entry`（NCHW→NHWC 入口转换，纯数据搬移——位级
//! 断言）与 `n_elem`（逐元素族——IEEE 基本运算逐位断言；exp 类与
//! 乘加链见用例表说明）。
//!
//! 纪律与 vulkan 侧 `n_kernel_tests` 同源：**每内核一次独立提交**、
//! LCG 确定性数据、位级比较。无 Metal 设备的环境（CI 非 mac job）
//! 自动跳过；CI 的 macos job / M4 真机是实跑环境。

use super::MetalContext;
use super::device::submit_compute;
use super::pipeline::KernelSet;
use metal::MTLResourceOptions;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut v = ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
    if v == 0.0 {
        v = 0.25; // ±0.0 的 max/min 语义是平台自由的——数据侧直接避开
    }
    v
}

fn open_or_skip() -> Option<MetalContext> {
    match MetalContext::open(None) {
        Ok(c) => {
            eprintln!("[metal] 设备: {}", c.inner.name);
            Some(c)
        }
        Err(e) => {
            eprintln!("[metal] 跳过（无 Metal 设备）: {e}");
            None
        }
    }
}

/// Shared arena 的 word 视图（统一内存：CPU/GPU 同址，写即设备可见）。
struct Words {
    ptr: *mut u32,
    len: usize,
}

impl Words {
    /// 分配 Shared 缓冲并取 word 视图。
    fn new(device: &metal::DeviceRef, words: usize) -> (metal::Buffer, Self) {
        let buf = device.new_buffer((words * 4) as u64, MTLResourceOptions::StorageModeShared);
        // SAFETY: buf 独占且在下述视图使用期间存活；contents() 持久
        // 有效至释放，长度由分配保证。
        let view = unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut u32, words) };
        (
            buf,
            Self {
                ptr: view.as_mut_ptr(),
                len: words,
            },
        )
    }
}

impl std::ops::Index<usize> for Words {
    type Output = u32;
    fn index(&self, i: usize) -> &u32 {
        assert!(i < self.len, "word {i} 越界（共 {} words）", self.len);
        // SAFETY: 上界由 assert 保证；ptr 来自 Shared 缓冲的持久映射。
        unsafe { &*self.ptr.add(i) }
    }
}

impl std::ops::IndexMut<usize> for Words {
    fn index_mut(&mut self, i: usize) -> &mut u32 {
        assert!(i < self.len, "word {i} 越界（共 {} words）", self.len);
        // SAFETY: 同上；测试内无别名并发访问。
        unsafe { &mut *self.ptr.add(i) }
    }
}

/// `n_entry` 的 CPU 参考（与 .metal 逐行对应）：NCHW → NHWC(Cpad4)。
fn entry_ref(x: &[f32], nb: usize, c: usize, h: usize, w: usize, cpad: usize) -> Vec<f32> {
    let hw = h * w;
    let mut out = vec![0f32; nb * hw * cpad];
    for n in 0..nb {
        for pos in 0..hw {
            for ci in 0..c {
                out[(n * hw + pos) * cpad + ci] = x[n * c * hw + ci * hw + pos];
            }
        }
    }
    out
}

/// `n_elem` 的 CPU 参考（与 .metal 逐分支对应；float4 展开成标量循环）。
/// `b` 按**字（word）偏移**给值：双目 `b[i]`、标量 `b[0]`、行向量
/// `b[row*4]`（[rows,4] 存储的 lane0）。
fn elem_ref(op: u32, a: &[f32], b: &[f32], p1: f32, p2: f32, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let c4 = p1.to_bits() as usize; // 仅 op 8/11 有意义
    for i in 0..n {
        let x = a[i];
        out[i] = match op {
            0 => x.max(0.0),
            1 => 1.0f32 / (1.0 + (-x).exp()),
            2 => (x * p1 + p2).clamp(0.0, 1.0),
            3 => x.clamp(p1, p2),
            4 => x + b[i],
            5 => x * b[i],
            6 => x + b[0],
            7 => x * b[0],
            8 => x - b[(i / 4) / c4],
            9 => x * x,
            10 => x.sqrt(),
            11 => x / b[(i / 4) / c4],
            12 => x - b[i],
            13 => x / b[i],
            _ => unreachable!("op {op} 不在 n_elem 语义表内"),
        };
    }
    out
}

#[test]
fn entry_nhwc_bitexact() {
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    let (nb, c, h, w) = (2usize, 3usize, 4usize, 5usize);
    let hw = h * w;
    let cpad = c.div_ceil(4) * 4; // 4
    let n_in = nb * c * hw;
    let n_out = nb * hw * cpad;
    let seg = |n: usize| n.div_ceil(4) * 4;
    let in_off = 4usize; // word 0 是 real_n 全局槽
    let out_off = in_off + seg(n_in);
    let pbase = out_off + seg(n_out);
    let (buf, mut m) = Words::new(&ctx.inner.device, pbase + 32);

    let mut seed = 0x1234_5678_9abc_def0u64;
    let x: Vec<f32> = (0..n_in).map(|_| lcg(&mut seed)).collect();
    m[0] = 0; // real_n = 0（不限）
    for (i, v) in x.iter().enumerate() {
        m[in_off + i] = v.to_bits();
    }
    let p = pbase;
    m[p] = in_off as u32; // float 元素偏移（f32 段 word==元素）
    m[p + 1] = out_off as u32; // word 偏移
    m[p + 2] = nb as u32;
    m[p + 3] = c as u32;
    m[p + 4] = h as u32;
    m[p + 5] = w as u32;
    m[p + 6] = cpad as u32;

    submit_compute(&ctx.inner.queue, |enc| {
        ks.dispatch(
            enc,
            "n_entry",
            &buf,
            p as u32,
            [(nb * hw).div_ceil(256) as u32, 1, 1],
        )
        .expect("dispatch n_entry");
    })
    .expect("submit");

    // 纯数据搬移——位级断言（捕捉偏移/寻址错位，这是 GPU 后端最高频
    // 的错误类别）
    let got: Vec<u32> = (0..n_out).map(|i| m[out_off + i]).collect();
    let want: Vec<u32> = entry_ref(&x, nb, c, h, w, cpad)
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(got, want, "n_entry NCHW→NHWC 位级不符");
    eprintln!("[metal] n_entry 位级对拍 OK（{n_out} words）");
}

#[test]
fn entry_real_n_early_exit() {
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    let (nb, c, h, w) = (2usize, 3usize, 2usize, 3usize);
    let hw = h * w;
    let cpad = 4;
    let n_in = nb * c * hw;
    let n_out = nb * hw * cpad;
    let seg = |n: usize| n.div_ceil(4) * 4;
    let in_off = 4usize;
    let out_off = in_off + seg(n_in);
    let pbase = out_off + seg(n_out);
    let (buf, mut m) = Words::new(&ctx.inner.device, pbase + 32);

    let mut seed = 0x0fed_cba9_8765_4321u64;
    let x: Vec<f32> = (0..n_in).map(|_| lcg(&mut seed)).collect();
    m[0] = 1; // real_n = 1：第 1 行（n≥1）早退不写
    for (i, v) in x.iter().enumerate() {
        m[in_off + i] = v.to_bits();
    }
    const SENTINEL: u32 = 0x5A5A_A5A5;
    for i in 0..n_out {
        m[out_off + i] = SENTINEL;
    }
    let p = pbase;
    m[p] = in_off as u32;
    m[p + 1] = out_off as u32;
    m[p + 2] = nb as u32;
    m[p + 3] = c as u32;
    m[p + 4] = h as u32;
    m[p + 5] = w as u32;
    m[p + 6] = cpad as u32;

    submit_compute(&ctx.inner.queue, |enc| {
        ks.dispatch(
            enc,
            "n_entry",
            &buf,
            p as u32,
            [(nb * hw).div_ceil(256) as u32, 1, 1],
        )
        .expect("dispatch n_entry");
    })
    .expect("submit");

    // 行 0 转换正确；行 1 保持哨兵（早退 = 不写，不是写零）
    let want_row0 = entry_ref(&x, nb, c, h, w, cpad);
    for pos in 0..hw {
        for ci in 0..cpad {
            let i = pos * cpad + ci;
            assert_eq!(m[out_off + i], want_row0[i].to_bits(), "行 0 位级不符");
        }
    }
    for i in hw * cpad..n_out {
        assert_eq!(m[out_off + i], SENTINEL, "real_n 早退后的行必须原样不动");
    }
    eprintln!("[metal] n_entry real_n 早退 OK（行 1 未触碰）");
}

#[test]
fn elem_family_bitexact() {
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    // 数据面：16 个 float4（n = 64 元素）
    let n = 64usize;
    let nv = n / 4;
    // 行向量广播（op 8/11）：c4 = 4 → 4 行
    let c4 = 4usize;
    let a_off = 4usize; // word 0 是 real_n 槽
    let apos_off = a_off + n; // 非负数据段（sqrt 的定义域）
    let b_off = apos_off + n;
    let out_off = b_off + n;
    let rowv_off = out_off + n;
    let scalar_off = rowv_off + c4 * 4;
    let pbase = scalar_off + 4;
    let stride = 32usize; // 参数块步长（7 字用）
    let (buf, mut m) = Words::new(&ctx.inner.device, pbase + stride * 16);

    let mut seed = 0x2468_ac0e_1357_9bdfu64;
    let a: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
    // 非负数据（sqrt 用）：值域 [0.06, 1.06)
    let a_pos: Vec<f32> = (0..n)
        .map(|_| (lcg(&mut seed) + 1.0) * 0.5 + 0.06)
        .collect();
    // 分母安全带：b 侧值域 (0.25, 1.25]，远离零（div 族不产生非规格数）
    let b_elem: Vec<f32> = (0..n)
        .map(|_| (lcg(&mut seed) + 1.0) * 0.5 + 0.25)
        .collect();
    let rowv: Vec<f32> = (0..c4)
        .map(|_| (lcg(&mut seed) + 1.0) * 0.5 + 0.25)
        .collect();
    let scalar = (lcg(&mut seed) + 1.0) * 0.5 + 0.25;
    m[0] = 0;
    for i in 0..n {
        m[a_off + i] = a[i].to_bits();
        m[apos_off + i] = a_pos[i].to_bits();
        m[b_off + i] = b_elem[i].to_bits();
    }
    for r in 0..c4 {
        m[rowv_off + r * 4] = rowv[r].to_bits(); // lane0
    }
    m[scalar_off] = scalar.to_bits(); // lane0

    // (op, in_off, in2_off, p1, p2, 位级?)。op1 的 exp 与 op2 的乘加链
    // 是实现定义精度（Metal 的 exp 非 IEEE 正确舍入、a*p1+p2 可能收缩
    // 成 fma），不逐位断言；其余全是 IEEE 基本运算，逐位必符。这两个
    // op 的全量数值判据在 P2 的 det 端到端（对照 CPU executor 全图输出）。
    let cases: Vec<(u32, usize, usize, f32, f32, bool)> = vec![
        (0, a_off, b_off, 0.0, 0.0, true),                           // relu
        (1, a_off, b_off, 0.0, 0.0, false),                          // sigmoid（exp）
        (2, a_off, b_off, 1.0 / 6.0, 1.0 / 2.0, false),              // hardsigmoid（乘加链）
        (3, a_off, b_off, -0.5, 0.5, true),                          // clip
        (4, a_off, b_off, 0.0, 0.0, true),                           // add
        (5, a_off, b_off, 0.0, 0.0, true),                           // mul
        (6, a_off, scalar_off, 0.0, 0.0, true),                      // 标量 add
        (7, a_off, scalar_off, 0.0, 0.0, true),                      // 标量 mul
        (8, a_off, rowv_off, f32::from_bits(c4 as u32), 0.0, true),  // 减行向量
        (9, a_off, b_off, 0.0, 0.0, true),                           // square
        (10, apos_off, b_off, 0.0, 0.0, true), // sqrt（IEEE 正确舍入；输入非负段）
        (11, a_off, rowv_off, f32::from_bits(c4 as u32), 0.0, true), // 除行向量
        (12, a_off, b_off, 0.0, 0.0, true),    // sub
        (13, a_off, b_off, 0.0, 0.0, true),    // div
    ];
    for (step, (op, in_off, in2, p1, p2, bitexact)) in cases.into_iter().enumerate() {
        let p = pbase + step * stride;
        m[p] = op;
        m[p + 1] = in_off as u32;
        m[p + 2] = in2 as u32;
        m[p + 3] = out_off as u32;
        m[p + 4] = n as u32;
        m[p + 5] = p1.to_bits();
        m[p + 6] = p2.to_bits();
        // 每内核一次独立提交：出事时最后一条日志即肇事者
        submit_compute(&ctx.inner.queue, |enc| {
            ks.dispatch(
                enc,
                "n_elem",
                &buf,
                p as u32,
                [(nv.div_ceil(256)) as u32, 1, 1],
            )
            .expect("dispatch n_elem");
        })
        .expect("submit");

        let src = if op == 10 { &a_pos } else { &a };
        let got: Vec<f32> = (0..n).map(|i| f32::from_bits(m[out_off + i])).collect();
        // op 6/7 的 in2 是 scalar_off、op 8/11 是 rowv_off——两者都单独存在
        // arena 另一段，不在 b_elem 里。以前这里一律传 &b_elem，等于让参考
        // 实现去 b_elem 里捞标量/行向量：参考值本身是错的，却指着内核喊
        // 「位级不符」（实测 op6 两侧 1.31 vs 0.64）。按内核语义把真正的
        // 第二操作数交给参考实现。
        let b_ref = match op {
            6 | 7 => {
                let mut v = b_elem.clone();
                v[0] = scalar;
                v
            }
            8 | 11 => rowv.clone(),
            _ => b_elem.clone(),
        };
        let want = elem_ref(op, src, &b_ref, p1, p2, n);
        if bitexact {
            let gb: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let wb: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
            assert_eq!(gb, wb, "n_elem op={op} 位级不符");
        } else {
            // exp 类：实现定义精度——逐元素相对误差 < 1e-6（有据的容差，
            // 不是遮羞布：依据见用例表注释）
            for (g, w) in got.iter().zip(&want) {
                let denom = g.abs().max(w.abs()).max(1e-30);
                assert!(
                    ((g - w) / denom).abs() < 1e-6,
                    "n_elem op={op} 相对误差超限: gpu={g} cpu={w}"
                );
            }
        }
        eprintln!(
            "[metal] n_elem op={op:>2} OK（{}）",
            if bitexact { "位级" } else { "1e-6" }
        );
    }
}

/// 端到端：真实 tiny det，Metal 整图计划 vs CPU executor，同一输入的
/// 概率图容差对拍（判据口径与 vulkan 侧 det_session_end_to_end 一致：
/// mean 反映整体数值健康，max 放行 sigmoid 斜坡处的极值放大）。
#[test]
fn det_session_end_to_end() {
    let Some(ctx) = open_or_skip() else { return };
    let p = std::path::Path::new("../../models/tiny/det.onnx");
    if !p.is_file() {
        eprintln!("[metal] 无模型，跳过");
        return;
    }
    let bytes = std::fs::read(p).unwrap();
    use qppocr_core::executor::Session;
    use qppocr_core::tensor::{DType, Tensor};
    use qppocr_kernels::buf::F32Buf;

    let (h, w) = (960i64, 864i64);
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    // 同一份输入克隆给 CPU/GPU（各生成一份曾是「全图分歧」假阳性源）
    let input_once: Vec<f32> = (0..3 * h * w).map(|_| lcg(&mut seed)).collect();
    let mk = |nm: &str| -> Tensor {
        let mut buf = F32Buf::with_zeroed(input_once.len());
        buf.as_mut_slice().copy_from_slice(&input_once);
        Tensor {
            name: nm.into(),
            shape: vec![1, 3, h, w],
            dtype: DType::F32,
            f32: buf,
            i64: Vec::new(),
        }
    };

    // CPU 参考
    let cpu = Session::from_memory(&bytes, "metal.det.cpu").unwrap();
    let in_name = cpu
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let t0 = std::time::Instant::now();
    let out_cpu = cpu.run(vec![(in_name.clone(), mk(&in_name))]).unwrap();
    eprintln!(
        "[metal] CPU 前向 {:.1} ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Metal 会话（同图同权重）
    let (graph, init) = {
        let s = Session::from_memory(&bytes, "metal.det.gpu").unwrap();
        s.into_parts()
    };
    let gpu = super::session::MetalSession::new(ctx.inner.clone(), graph, init).unwrap();
    let t1 = std::time::Instant::now();
    let out_gpu =
        qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name.clone(), mk(&in_name))])
            .unwrap();
    eprintln!(
        "[metal] GPU 首跑（含建计划）{:.1} ms",
        t1.elapsed().as_secs_f64() * 1000.0
    );
    let t2 = std::time::Instant::now();
    let _ = qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name.clone(), mk(&in_name))])
        .unwrap();
    eprintln!(
        "[metal] GPU 二跑（缓存）{:.1} ms",
        t2.elapsed().as_secs_f64() * 1000.0
    );

    assert_eq!(out_cpu.len(), out_gpu.len());
    let a = &out_gpu[0].f32;
    let b = &out_cpu[0].f32;
    assert_eq!(a.len(), b.len(), "输出元素数不一致");
    assert_eq!(out_gpu[0].shape, out_cpu[0].shape, "输出形状不一致");
    let max_abs = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let mean_abs = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32;
    eprintln!("[metal] det 概率图：max|diff| = {max_abs:.3e}，mean|diff| = {mean_abs:.3e}");
    assert!(mean_abs < 1e-3, "概率图均值偏差超容差: {mean_abs}");
    assert!(max_abs < 5e-2, "概率图极值偏差超容差: {max_abs}");
}

/// 同图同输入，`weight_f16` **关 / 开**两条 Metal 路各跑一遍，返回两份
/// 输出（关 = f32 参考）。
///
/// 交付**原始输出**而不是预先算好的误差统计：判「这条路径对不对」的
/// 尺子取决于下游怎么用——det 是阈值二值化、rec 是 argmax，两者都不是
/// 「和 f32 差多少」。把统计留在各测试里，才能对着**消费端**判。
///
/// 两条路各自从 `bytes` 装配（权重是 move 语义，各需一份）。
fn f16_vs_f32_outputs(
    ctx: &MetalContext,
    bytes: &[u8],
    shape: &[i64],
    seed0: u64,
) -> (Vec<f32>, Vec<f32>, Vec<i64>) {
    use qppocr_core::executor::Session;
    use qppocr_core::tensor::{DType, Tensor};
    use qppocr_kernels::buf::F32Buf;

    let n: usize = shape.iter().product::<i64>() as usize;
    let mut seed = seed0;
    let mut buf = F32Buf::with_zeroed(n);
    for v in buf.as_mut_slice().iter_mut() {
        *v = lcg(&mut seed);
    }
    let mk = |nm: &str| -> Tensor {
        let mut b = F32Buf::with_zeroed(n);
        b.as_mut_slice().copy_from_slice(buf.as_slice());
        Tensor {
            name: nm.into(),
            shape: shape.to_vec(),
            dtype: DType::F32,
            f32: b,
            i64: Vec::new(),
        }
    };
    let run = |tag: &str, f16: bool| -> (Vec<f32>, Vec<i64>) {
        let s = Session::from_memory(bytes, tag).unwrap();
        let in_name = s
            .graph
            .inputs
            .iter()
            .find(|s| !s.is_empty())
            .cloned()
            .unwrap();
        let (graph, init) = s.into_parts();
        let mut gpu = super::session::MetalSession::new(ctx.inner.clone(), graph, init).unwrap();
        // 计划是惰性构建的：构造后、首次 run 前改它仍然生效。
        gpu.weight_f16 = f16;
        gpu.batch_grain = shape[0] as i32;
        let out =
            qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name.clone(), mk(&in_name))])
                .unwrap();
        (out[0].f32.as_slice().to_vec(), out[0].shape.clone())
    };
    let (a, shape_a) = run("metal.f16.off", false);
    let (b, shape_b) = run("metal.f16.on", true);
    assert_eq!(shape_a, shape_b, "两条路输出形状不一致");
    (a, b, shape_a)
}

/// 两条路的逐元素差统计：`(max|diff|, mean|diff|, 非零差个数)`。
fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32, usize) {
    assert_eq!(a.len(), b.len(), "两条路输出元素数不一致");
    let mut max_abs = 0.0f32;
    let mut sum = 0.0f64;
    let mut ndiff = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(x.is_finite() && y.is_finite(), "出现 NaN/inf —— 多半是权重索引错位");
        let d = (x - y).abs();
        if *x != *y {
            ndiff += 1;
        }
        max_abs = max_abs.max(d);
        sum += d as f64;
    }
    (max_abs, (sum / a.len() as f64) as f32, ndiff)
}

/// 端到端：**fp16 权重**（`n_conv8h`）与同一张图的 f32 路对拍。
///
/// 三件事，缺一条这条路径就不可信：
///
/// 1. **MSL 模板真的编出了 `n_conv8h`**——`KernelSet` 里没这个名字的话，
///    派发时会硬报错（`dispatch` 查表失败），不会静默降级。
/// 2. **计划真的换了内核**——判据是 `ndiff > 0`。若 `weight_f16` 没被
///    计划层读到，两条路会**逐位相同**；「有非零差」正是「换过去了」的
///    证明，比任何计数器都直接。
/// 3. **换过去之后下游判得一样**——见下。
///
/// ## 判据是「二值化后有没有变」，不是「和 f32 差多少」
///
/// 本路径唯一的**结构性**风险是内核与打包配对错：`n_conv8h` 与
/// `n_conv8` 解释同一个 `w_off`（half4 vs float4），配错了内核照样跑完，
/// 只是读到**别人的字节**。这类错误落在输出上是什么样？——读错通道的
/// 权重等于换了个网络，概率图会被推过下游阈值。
///
/// 所以尺子取消费端：det 的出口是 `det_thresh`（默认 0.2）二值化，
/// 那就数**翻转的像素占比**。它同时是「不是垃圾」和「下游无感」两个
/// 主张的判据，且不像「误差 ÷ 量程」那样在下游非线性（sigmoid 饱和区
/// 把误差压扁、二值化再放大）时失真——随机输入下 det 输出几乎是常数
/// 图，量程只有 2.8e-3，用绝对阈值或量程比都只会量到「图有多平」。
#[test]
fn det_session_f16_weights_end_to_end() {
    let Some(ctx) = open_or_skip() else { return };
    let p = std::path::Path::new("../../models/tiny/det.onnx");
    if !p.is_file() {
        eprintln!("[metal] 无模型，跳过");
        return;
    }
    let bytes = std::fs::read(p).unwrap();
    let (a, b, shape) = f16_vs_f32_outputs(&ctx, &bytes, &[1, 3, 960, 864], 0x5f16_0001);
    eprintln!("[metal] det 输出形状 {shape:?}");
    let (max_abs, mean_abs, ndiff) = diff_stats(&a, &b);

    // 消费端：det 的二值化门限（`Config::det_thresh` 默认 0.2）。
    const DET_THRESH: f32 = 0.2;
    let flipped = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| (**x > DET_THRESH) != (**y > DET_THRESH))
        .count();
    let flip_frac = flipped as f64 / a.len() as f64;
    eprintln!(
        "[metal] det fp16 权重 vs f32：max|diff| = {max_abs:.3e}，mean|diff| = {mean_abs:.3e}，\
         非零差 {ndiff}/{}，二值化翻转 {flipped}（{:.3e}）",
        a.len(),
        flip_frac
    );
    assert!(ndiff > 0, "fp16 权重没有生效：两条路逐位相同（weight_f16 没被计划读到？）");
    // 千分之一是一个**很松**的界：真要读错通道的权重，翻转率是 O(1)
    // （整块前景/背景判反），不是 O(1e-3)。
    assert!(
        flip_frac <= 1e-3,
        "二值化翻转 {flip_frac:.3e}（>1e-3）——不是舍入级，查内核与打包是否配对"
    );
}

/// 同上，rec（small）——**CTC 分类头**（`MatMul.12` → `n_conv8h`）是
/// fp16 收益最大的一条（权重 9 MB、探针实测 2.59×），也是数值上最该
/// 盯的一条：18710 类 argmax，权重一窄，临界行的翻转最先在这里出现。
///
/// 判据同样是消费端：rec 的出口是**逐帧 argmax**（GPU 侧
/// `n_exit3_argmax` 直接吐 (val, idx) 对）。所以数 **argmax 不同的帧
/// 占比**——它逐字对应 CER 的驱动量，而且不像 float 差值那样需要挑阈值：
/// 要么同一类，要么不是。
///
/// **门是零容忍**（`disagree == 0`），依据是两层：读错通道的话 argmax 会
/// 退化成近似随机，18710 类里命中率约 1/18710，瞬间就炸；而本机本模型
/// 实测就是 0/384 帧（同一批权重下两条路的 argmax 逐帧一致）。零容忍
/// 同时也是一条**回归**门：将来任何改动只要让两个精度在某一帧上分道，
/// 这里就会红——那正是需要有人看一眼的时刻，而不是默默放过的噪声。
/// （CPU 侧全量转 fp16 的 CER 增量 tiny 0.199% / small 0.072% /
/// medium 0.165% 说明真实语料上的翻转确实存在，但那是**全部**权重都转
/// 的代价；这里只转走 `n_conv8` 的那些层，且这条测试的输入是随机张量，
/// 与语料分布无关。）
#[test]
fn rec_session_f16_weights_end_to_end() {
    let Some(ctx) = open_or_skip() else { return };
    let p = std::path::Path::new("../../models/small/rec.onnx");
    if !p.is_file() {
        eprintln!("[metal] 无 small rec 模型，跳过");
        return;
    }
    let bytes = std::fs::read(p).unwrap();
    let (a, c, shape) = f16_vs_f32_outputs(&ctx, &bytes, &[8, 3, 48, 384], 0x5f16_0002);
    eprintln!("[metal] rec 输出形状 {shape:?}");
    let (max_abs, mean_abs, ndiff) = diff_stats(&a, &c);
    // 形状是 [B, T, V]，V（类别数）是最后一维——**从张量读**，不硬编码：
    // 硬编码错了会把「帧」切在某行的中间，argmax 比的就是错位的数据。
    let v = *shape.last().expect("输出无维度") as usize;
    assert_eq!(a.len() % v, 0, "输出元素数不是 V 的整数倍，形状解读错了");
    let nframes = a.len() / v;

    let argmax = |x: &[f32], frame: usize| -> usize {
        let s = &x[frame * v..(frame + 1) * v];
        let mut best = 0usize;
        for (i, val) in s.iter().enumerate() {
            if *val > s[best] {
                best = i;
            }
        }
        best
    };
    let disagree = (0..nframes).filter(|f| argmax(&a, *f) != argmax(&c, *f)).count();
    eprintln!(
        "[metal] rec fp16 权重 vs f32：max|diff| = {max_abs:.3e}，mean|diff| = {mean_abs:.3e}，\
         非零差 {ndiff}/{}，argmax 不一致 {disagree}/{nframes} 帧",
        a.len()
    );
    assert!(ndiff > 0, "fp16 权重没有生效：两条路逐位相同");
    assert_eq!(disagree, 0, "有 {disagree} 帧 argmax 变了——查内核与打包是否配对");
}

/// 端到端：真实 tiny rec，Metal 整图 vs CPU executor——概率图容差 +
/// softmax 行和 + 每时间步 top-1 对齐（判据口径与 vulkan 侧
/// rec_session_end_to_end 同源；直接构造会话 → 全量概率输出）。
/// 零 GPU 风险的复用复现：连续 build 一串形状（**不派发任何 dispatch**），
/// 然后逐份核对「该计划的静态区（权重 + 参数块）此刻还在不在位」。
///
/// 为什么这条判据是根因级的：会话级 arena 被所有计划共用，而静态区**每次
/// build 都从偏移 0 重新分配**（`Layout::alloc_static` 与 `alloc` 共用一个
/// cursor），于是 build j 的激活区会压在 build i 的参数块上。`plan_for` 命中
/// 缓存时**一个字节都不重传**，被压掉的参数块就这么被拿去跑 → 垃圾参数 →
/// 巨循环 → GPU 挂死设备丢失（`plan.rs:1804` 记的形态）。本测试给出**无派发**
/// 的证据：哪些计划的静态区被这串 build 冲掉了；再走 `ensure_statics` 复核
/// 修复后「派发前必然齐全」。
///
/// 形状/模型可由环境变量驱动，rec / cls 的序列不必重编译。**默认那两个 det 形状是弱例**
/// （它们静态区恰好重合、0 份被冲），要跑**能真正冲掉**的强例用 rec：
///
///   QPPOCR_METAL_REPLAY_MODEL=../../models/tiny/rec.onnx \
///   QPPOCR_METAL_REPLAY_SHAPES="1x3x48x384,1x3x48x1024,4x3x48x256,8x3x48x192,3x3x48x512,1x3x48x1088" \
///   cargo test --release -p qppocr-metal arena_reuse_replay -- --nocapture
///
/// 2026-10-08 修后实测（重启后的健康机器，与修前一致）：`建完 6 份计划：4 份静态区被冲掉`，
/// 最坏的 `[1,3,48,1024]` 只剩 **13/154 块**在位（9.08/13.47 MB）；逐份走 `ensure_statics`
/// 之后 **6/6 全部在位**。默认形状则是 2026-10-08 22:22 那次「2 张图 / 1 进程」致瘸日志里的
/// 两个 det 形状（`[1,3,960,864]` = 图 1，`[1,3,800,960]` = 图 2）。
#[test]
fn arena_reuse_replay_det() {
    let Some(ctx) = open_or_skip() else { return };
    let model = std::env::var("QPPOCR_METAL_REPLAY_MODEL")
        .unwrap_or_else(|_| "../../models/tiny/det.onnx".into());
    let p = std::path::Path::new(&model);
    if !p.is_file() {
        eprintln!("[metal] 无模型 {model}，跳过");
        return;
    }
    let shapes: Vec<Vec<i64>> = match std::env::var("QPPOCR_METAL_REPLAY_SHAPES") {
        Ok(s) => s
            .split(',')
            .filter(|t| !t.trim().is_empty())
            .map(|t| {
                t.trim()
                    .split('x')
                    .map(|v| v.trim().parse::<i64>().expect("形状形如 1x3x48x384"))
                    .collect()
            })
            .collect(),
        Err(_) => vec![
            vec![1i64, 3, 960, 864], // 图 1 的 det
            vec![1i64, 3, 800, 960], // 图 2 的 det
        ],
    };
    let bytes = std::fs::read(p).unwrap();
    use qppocr_core::executor::Session;
    let (graph, init) = Session::from_memory(&bytes, "metal.replay")
        .unwrap()
        .into_parts();
    let gpu = super::session::MetalSession::new(ctx.inner.clone(), graph, init).unwrap();
    gpu.debug_replay_builds(&shapes).unwrap();
}

#[test]
fn rec_session_end_to_end() {
    let Some(ctx) = open_or_skip() else { return };
    let p = std::path::Path::new("../../models/tiny/rec.onnx");
    if !p.is_file() {
        eprintln!("[metal] 无 rec 模型，跳过");
        return;
    }
    let bytes = std::fs::read(p).unwrap();
    use qppocr_core::executor::Session;
    use qppocr_core::tensor::{DType, Tensor};
    use qppocr_kernels::buf::F32Buf;

    let (b, hh, ww) = (1usize, 48usize, 185usize);
    let mut seed = 0x8899_aabb_ccdd_eeff_u64;
    let mut buf = F32Buf::with_zeroed(b * 3 * hh * ww);
    for v in buf.as_mut_slice().iter_mut() {
        *v = lcg(&mut seed);
    }
    let t0 = Tensor {
        name: String::new(),
        shape: vec![b as i64, 3, hh as i64, ww as i64],
        dtype: DType::F32,
        f32: buf,
        i64: Vec::new(),
    };
    let cpu = Session::from_memory(&bytes, "metal.rec.cpu").unwrap();
    let in_name = cpu
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let out_cpu = cpu.run(vec![(in_name.clone(), t0.clone())]).unwrap();
    let (graph, init) = {
        let s = Session::from_memory(&bytes, "metal.rec.gpu").unwrap();
        s.into_parts()
    };
    let gpu = super::session::MetalSession::new(ctx.inner.clone(), graph, init).unwrap();
    let out_gpu =
        qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name.clone(), t0)]).unwrap();

    let a = &out_gpu[0].f32;
    let bw = &out_cpu[0].f32;
    assert_eq!(a.len(), bw.len(), "输出元素数不一致");
    assert_eq!(out_gpu[0].shape, out_cpu[0].shape, "输出形状不一致");
    let max_abs = a
        .iter()
        .zip(bw.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let mean_abs = a
        .iter()
        .zip(bw.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32;
    eprintln!("[metal-rec] max|diff|={max_abs:.3e} mean|diff|={mean_abs:.3e}");
    assert!(mean_abs < 1e-6, "概率均值偏差超容差: {mean_abs}");
    // 极值容差按实测定格（曾放 5e-2——那是个能把整块注意力算错都盖住的
    // 数字，见 `attn_mm_raw` 的 203/210：真正的错误在这里被放行了）
    assert!(max_abs < 1e-4, "概率极值偏差超容差: {max_abs}");
    // softmax 行和 ≈ 1（全链数值健康的独立判据）
    let rows = a.len() / *out_cpu[0].shape.last().unwrap() as usize;
    let vocab = *out_cpu[0].shape.last().unwrap() as usize;
    let bad_sums = (0..rows)
        .filter(|r| {
            let s: f32 = a[r * vocab..(r + 1) * vocab].iter().sum();
            (s - 1.0).abs() > 1e-3
        })
        .count();
    eprintln!("[metal-rec] softmax 行和偏差 >1e-3：{bad_sums}/{rows}");
    assert_eq!(bad_sums, 0, "softmax 行和异常");
    // 每时间步 top-1（CTC 路径）：报告精确计数，宽松断言（M4 首跑后
    // 按实测定格收紧）
    let pick = |sl: &[f32]| -> usize {
        sl.iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    let top_mis = (0..rows)
        .filter(|r| pick(&a[r * vocab..(r + 1) * vocab]) != pick(&bw[r * vocab..(r + 1) * vocab]))
        .count();
    eprintln!("[metal-rec] top-1 不一致 {top_mis}/{rows}");
    // 逐位要求：CTC 的每步 top-1 就是最终字符，差一个就是**错字**。
    // 这里曾写 `top_mis <= rows/3`——一个允许 1/3 时间步选错字的断言，
    // 等于把「精度与 CPU 一致」这条硬约束换成了「大致像」。
    assert_eq!(top_mis, 0, "top-1 不一致 {top_mis}/{rows}");
}

/// 端到端：**small 档 rec（transformer 主干）**的逐元素对拍。
///
/// `rec_session_end_to_end` 只覆盖 `models/tiny`（纯卷积 rec，63 条
/// dispatch），而 small 档的 rec 是 173 条 dispatch 的 transformer
/// （LayerNorm + 多头注意力 + CTC 头）——窗口、掩码、softmax 语义都不同。
/// 实测端到端 12 图的**逐字符**对拍里 small 档 rec 有 33% 的行与 CPU 不同
/// （tiny 档与 cls/det 均逐位一致），所以这里把它单拉出来量。
///
/// 形状取真实语料桶宽（48×216），批维 1（引擎 exact_width 下的常态）。
#[test]
fn rec_small_end_to_end() {
    let Some(ctx) = open_or_skip() else { return };
    if !rec_small_bytes().is_some() {
        return;
    }
    let (max_abs, mean_abs, shape) = rec_small_case(&ctx, 1, 216, 0x1357_9bdf_0246_8ace);
    eprintln!(
        "[metal-rec-small] 输出0 {shape:?} max|diff|={max_abs:.3e} mean|diff|={mean_abs:.3e}"
    );
    // 判据：注意力块算错时这里是 4.6e-1（见 `attn_mm_raw`），修好后
    // 是 2.5e-6（FMA 收缩的余量）。1e-4 两边都夹得住——不允许退回。
    assert!(max_abs < 1e-4, "极值偏差超容差: {max_abs:.3e}");
    assert!(mean_abs < 1e-6, "均值偏差超容差: {mean_abs:.3e}");
}

/// small 档 rec 的**形状扫描**：单点对拍（上面那个用例）只证明了一个
/// 宽度，而引擎真实的桶宽是一整套（grain=8 下 96…1008 都可能出现）、
/// 批维也从 1 到 8。先前「单点绿、整图红」就是栽在这个缝里：注意力块
/// 一错，216 恰好因为某种巧合过关（或反之）。
///
/// 所以这里把「一算子一算子地测」落到形状维上：逐形状打印偏差，任何
/// 一个形状炸掉都直接点名。
#[test]
fn rec_small_shape_sweep() {
    let Some(ctx) = open_or_skip() else { return };
    if !rec_small_bytes().is_some() {
        return;
    }
    // 宽度取语料里真实出现过的桶宽（grain 8）+ 极端窄/宽
    let widths = [
        96usize, 128, 160, 192, 216, 240, 264, 288, 336, 408, 480, 552, 720, 1008,
    ];
    let mut worst = (0f32, 0usize);
    for (i, &w) in widths.iter().enumerate() {
        let (max_abs, mean_abs, _) =
            rec_small_case(&ctx, 1, w, 0x51ed_2701_dead_0001 + i as u64 * 0x9e37);
        eprintln!("[sweep] B=1 W={w:<5} max|diff|={max_abs:.3e} mean|diff|={mean_abs:.3e}");
        if max_abs > worst.0 {
            worst = (max_abs, w);
        }
        assert!(max_abs < 1e-4, "B=1 W={w} 极值偏差超容差: {max_abs:.3e}");
    }
    // 批维：引擎 per_batch=8，实跑多为 3/5/8 行一批（其余行零填充，
    // real_n 早退）。批维一错，整批行文本一起错。
    for &b in &[2usize, 3, 5, 8] {
        let (max_abs, mean_abs, _) = rec_small_case(&ctx, b, 216, 0x0b_a7_0003 + b as u64);
        eprintln!("[sweep] B={b} W=216  max|diff|={max_abs:.3e} mean|diff|={mean_abs:.3e}");
        assert!(max_abs < 1e-4, "B={b} W=216 极值偏差超容差: {max_abs:.3e}");
    }
    eprintln!("[sweep] 最差形状 W={}", worst.1);
}

fn rec_small_bytes() -> Option<Vec<u8>> {
    let p = std::path::Path::new("../../models/small/rec.onnx");
    if !p.is_file() {
        eprintln!("[metal] 无 small rec 模型，跳过");
        return None;
    }
    Some(std::fs::read(p).unwrap())
}

/// 跑一组 small rec 对拍：CPU `Session` vs Metal 会话（批维按引擎语义
/// 补齐到 8），返回 (max|diff|, mean|diff|, 输出形状)。
///
/// `seed` 必须按形状给不同的值：同一份随机输入喂给不同宽度，会掩盖
/// 「某些宽度才触发的寻址错误」。
fn rec_small_case(
    ctx: &MetalContext,
    b: usize,
    ww: usize,
    seed0: u64,
) -> (f32, f32, Vec<i64>) {
    let bytes = rec_small_bytes().expect("已确认模型存在");
    use qppocr_core::executor::Session;
    use qppocr_core::tensor::{DType, Tensor};
    use qppocr_kernels::buf::F32Buf;

    let hh = 48usize;
    let mut seed = seed0;
    let mut buf = F32Buf::with_zeroed(b * 3 * hh * ww);
    for v in buf.as_mut_slice().iter_mut() {
        *v = lcg(&mut seed);
    }
    let t0 = Tensor {
        name: String::new(),
        shape: vec![b as i64, 3, hh as i64, ww as i64],
        dtype: DType::F32,
        f32: buf,
        i64: Vec::new(),
    };
    let cpu = Session::from_memory(&bytes, "metal.rec_small.cpu").unwrap();
    let in_name = cpu
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let out_cpu = cpu.run(vec![(in_name.clone(), t0.clone())]).unwrap();
    let (graph, init) = {
        let s = Session::from_memory(&bytes, "metal.rec_small.gpu").unwrap();
        s.into_parts()
    };
    let mut gpu = super::session::MetalSession::new(ctx.inner.clone(), graph, init).unwrap();
    gpu.batch_grain = 8; // 引擎合批语义（create_session 的 rec 配置）
    let out_gpu =
        qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name.clone(), t0)]).unwrap();

    assert_eq!(out_gpu.len(), out_cpu.len(), "输出个数不一致");
    let (a, c) = (&out_gpu[0].f32, &out_cpu[0].f32);
    assert_eq!(out_gpu[0].shape, out_cpu[0].shape, "输出形状不一致");
    assert_eq!(a.len(), c.len(), "输出元素数不一致");
    let max_abs = a
        .iter()
        .zip(c.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let mean_abs = a.iter().zip(c.iter()).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    (max_abs, mean_abs, out_gpu[0].shape.clone())
}

/// 端到端：真实 cls（B=17 批维——历史 bug 高发形态），argmax 对齐 +
/// 行和（判据口径与 vulkan 侧 cls_session_end_to_end 同源）。
#[test]
fn cls_session_end_to_end() {
    let Some(ctx) = open_or_skip() else { return };
    let p = std::path::Path::new("../../models/cls.onnx");
    if !p.is_file() {
        eprintln!("[metal] 无 cls 模型，跳过");
        return;
    }
    let bytes = std::fs::read(p).unwrap();
    use qppocr_core::executor::Session;
    use qppocr_core::tensor::{DType, Tensor};
    use qppocr_kernels::buf::F32Buf;

    let (b, hh, ww) = (17usize, 80usize, 160usize);
    let mut seed = 0x1234_5678_abcd_ef01_u64;
    let mut buf = F32Buf::with_zeroed(b * 3 * hh * ww);
    for v in buf.as_mut_slice().iter_mut() {
        *v = lcg(&mut seed);
    }
    let t0 = Tensor {
        name: String::new(),
        shape: vec![b as i64, 3, hh as i64, ww as i64],
        dtype: DType::F32,
        f32: buf,
        i64: Vec::new(),
    };
    let cpu = Session::from_memory(&bytes, "metal.cls.cpu").unwrap();
    let in_name = cpu
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let out_cpu = cpu.run(vec![(in_name.clone(), t0.clone())]).unwrap();
    let (graph, init) = {
        let s = Session::from_memory(&bytes, "metal.cls.gpu").unwrap();
        s.into_parts()
    };
    let gpu = super::session::MetalSession::new(ctx.inner.clone(), graph, init).unwrap();
    let out_gpu =
        qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name.clone(), t0)]).unwrap();

    let a = &out_gpu[0].f32;
    let bw = &out_cpu[0].f32;
    assert_eq!(a.len(), bw.len());
    let cls_dim = *out_cpu[0].shape.last().unwrap() as usize;
    let rows = a.len() / cls_dim;
    let mut top_mis = 0usize;
    let mut sum_bad = 0usize;
    for r in 0..rows {
        let pick = |sl: &[f32]| -> usize {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, v) in sl.iter().enumerate() {
                if v.is_finite() && *v > bv {
                    bv = *v;
                    best = i;
                }
            }
            best
        };
        if pick(&a[r * cls_dim..(r + 1) * cls_dim]) != pick(&bw[r * cls_dim..(r + 1) * cls_dim]) {
            top_mis += 1;
        }
        let s: f32 = a[r * cls_dim..(r + 1) * cls_dim].iter().sum();
        if (s - 1.0).abs() > 1e-3 {
            sum_bad += 1;
        }
    }
    eprintln!("[metal-cls] B={b} argmax 不一致 {top_mis}/{rows}，行和异常 {sum_bad}/{rows}");
    assert!(
        top_mis <= rows / 3,
        "cls argmax 不一致超 1/3: {top_mis}/{rows}"
    );
    assert_eq!(sum_bad, 0, "softmax 行和异常");
}

/// 批维补齐（rec 合批语义）：B=3 补到 grain=8——计划按 (8,W) 建、
/// real_n=3 早退空行、读回按真实行缩回。对照 CPU 的 B=3 前向。
#[test]
fn rec_batch_padding() {
    let Some(ctx) = open_or_skip() else { return };
    let p = std::path::Path::new("../../models/tiny/rec.onnx");
    if !p.is_file() {
        eprintln!("[metal] 无 rec 模型，跳过");
        return;
    }
    let bytes = std::fs::read(p).unwrap();
    use qppocr_core::executor::Session;
    use qppocr_core::tensor::{DType, Tensor};
    use qppocr_kernels::buf::F32Buf;

    let (b, hh, ww) = (3usize, 48usize, 96usize);
    let mut seed = 0x55aa_1234_5678_9abc_u64;
    let mut buf = F32Buf::with_zeroed(b * 3 * hh * ww);
    for v in buf.as_mut_slice().iter_mut() {
        *v = lcg(&mut seed);
    }
    let t0 = Tensor {
        name: String::new(),
        shape: vec![b as i64, 3, hh as i64, ww as i64],
        dtype: DType::F32,
        f32: buf,
        i64: Vec::new(),
    };
    let cpu = Session::from_memory(&bytes, "metal.rec_b3.cpu").unwrap();
    let in_name = cpu
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let out_cpu = cpu.run(vec![(in_name.clone(), t0.clone())]).unwrap();
    let (graph, init) = {
        let s = Session::from_memory(&bytes, "metal.rec_b3.gpu").unwrap();
        s.into_parts()
    };
    let mut gpu = super::session::MetalSession::new(ctx.inner.clone(), graph, init).unwrap();
    gpu.batch_grain = 8; // 引擎合批语义（create_session 的 rec 配置）
    let out_gpu =
        qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name.clone(), t0)]).unwrap();

    assert_eq!(
        out_gpu[0].shape, out_cpu[0].shape,
        "补齐读回形状应缩回真实批"
    );
    let a = &out_gpu[0].f32;
    let bw = &out_cpu[0].f32;
    assert_eq!(a.len(), bw.len());
    let mean_abs = a
        .iter()
        .zip(bw.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32;
    eprintln!("[metal-rec-b3] B=3→pad8 mean|diff|={mean_abs:.3e}");
    assert!(mean_abs < 1e-3, "补齐批均值偏差超容差: {mean_abs}");
}

// ─────────────── 裸内核对拍：注意力块三件套 ───────────────
//
// det / cls / tiny-rec 三条端点用例**都不经过** `n_attn_mm` /
// `n_softmax` / `n_transpose_nd`——它们是本后端唯一没有端点覆盖的内核。
// small 档 rec（173 dispatch 的 transformer）与 tiny 档（纯卷积 63
// dispatch）的差异恰好只有这一段，所以 `rec_small_end_to_end` 报出的
// max|diff| 4.6e-1 的嫌疑全部落在这里。
//
// 三个用例从 vulkan 侧 `n_kernel_tests.rs` 的同名用例移植：同样的形状、
// 同样的 LCG 种子、同样的 CPU 参考算法——一侧红一侧绿就直接定位到后端。

/// `n_softmax` 的 CPU 参考。**必须复刻内核的运算顺序**：exp1 多项式的
/// 展开序、以及 x86 `softmax_row_vec` 的 8-lane 顺序和（树归约序差
/// ~1e-6 就足以翻 CTC 单字——这是 vulkan 侧用整段注释换来的教训）。
/// 只写「数学上等价的 softmax」会让参考自己在 1e-6 量级漂。
fn softmax_ref(src: &[f32], rows: usize, cols: usize, cpad: usize) -> Vec<f32> {
    fn exp1(x: f32) -> f32 {
        const LN2_HI: f32 = 0.6931472;
        const LN2_LO: f32 = -2.9802322e-8;
        const INV_LN2: f32 = 1.442695;
        let x = x.clamp(-88.0, 88.0);
        let kf = (x * INV_LN2).round_ties_even();
        let mut r = (-kf).mul_add(LN2_HI, x);
        r = (-kf).mul_add(LN2_LO, r);
        let mut p = 1.0f32 / 720.0;
        p = p.mul_add(r, 1.0 / 120.0);
        p = p.mul_add(r, 1.0 / 24.0);
        p = p.mul_add(r, 1.0 / 6.0);
        p = p.mul_add(r, 0.5);
        p = p.mul_add(r, 1.0);
        p = p.mul_add(r, 1.0);
        let ki = kf as i32 + 127;
        f32::from_bits((ki as u32) << 23) * p
    }
    let mut out = vec![0f32; rows * cpad];
    for row in 0..rows {
        let base = &src[row * cpad..row * cpad + cols];
        // 内核起手 -3.4e38（不是 -inf）——照抄，别「顺手改对」
        let m = base.iter().copied().fold(-3.4e38f32, f32::max);
        let e: Vec<f32> = base.iter().map(|v| exp1(v - m)).collect();
        // 8-lane 顺序和 + ((l0+l4)+(l1+l5))+((l2+l6)+(l3+l7))
        let mut l = [0f32; 8];
        let q = cols >> 3;
        for t in 0..q {
            for k in 0..8 {
                l[k] += e[t * 8 + k];
            }
        }
        let mut s = ((l[0] + l[4]) + (l[1] + l[5])) + ((l[2] + l[6]) + (l[3] + l[7]));
        for j in q * 8..cols {
            s += e[j];
        }
        let inv = 1.0 / s;
        for j in 0..cols {
            out[row * cpad + j] = e[j] * inv;
        }
    }
    out
}

/// `n_transpose_nd` 裸内核：rank-5 `[2,0,3,1,4]`（small rec 的
/// `Transpose.3` 同构）对拍 CPU 参考，逐元素位级。
#[test]
fn transpose_nd_raw() {
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    let dims: [u32; 5] = [4, 3, 2, 8, 15];
    let perm: [u32; 5] = [2, 0, 3, 1, 4];
    let total: usize = dims.iter().map(|&d| d as usize).product();
    let mut seed = 0x1234_0000_5555_aaaa_u64;
    let src: Vec<f32> = (0..total).map(|_| lcg(&mut seed)).collect();

    // CPU 参考：输出线性下标 i → 按输出维（dims 按 perm 重排）分解 →
    // 经 perm 映射回输入坐标 → 行主序 Horner
    let odims: Vec<usize> = (0..5).map(|d| dims[perm[d] as usize] as usize).collect();
    let want: Vec<f32> = (0..total)
        .map(|i| {
            let mut oc = [0usize; 5];
            let mut rem = i;
            for d in (0..5).rev() {
                oc[d] = rem % odims[d];
                rem /= odims[d];
            }
            let mut ic = [0usize; 5];
            for d in 0..5 {
                ic[perm[d] as usize] = oc[d];
            }
            let mut s = 0usize;
            for k in 0..5 {
                s = s * dims[k] as usize + ic[k];
            }
            src[s]
        })
        .collect();

    let seg = |n: usize| n.div_ceil(4) * 4;
    let in_off = 4usize; // word 0 是 real_n 槽
    let out_off = in_off + seg(total);
    let pbase = out_off + seg(total);
    let (buf, mut m) = Words::new(&ctx.inner.device, pbase + 32);

    m[0] = 0;
    for (i, v) in src.iter().enumerate() {
        m[in_off + i] = v.to_bits();
    }
    let p = pbase;
    m[p] = in_off as u32;
    m[p + 1] = out_off as u32;
    m[p + 2] = total as u32;
    m[p + 3] = 5;
    for d in 0..5 {
        m[p + 4 + d] = dims[d];
        m[p + 9 + d] = perm[d];
    }

    submit_compute(&ctx.inner.queue, |enc| {
        ks.dispatch(
            enc,
            "n_transpose_nd",
            &buf,
            p as u32,
            [total.div_ceil(256) as u32, 1, 1],
        )
        .expect("dispatch n_transpose_nd");
    })
    .expect("submit");

    let got: Vec<u32> = (0..total).map(|i| m[out_off + i]).collect();
    let wantb: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
    let bad = got.iter().zip(&wantb).filter(|(a, b)| a != b).count();
    eprintln!("[metal] n_transpose_nd dims={dims:?} perm={perm:?} 分歧 {bad}/{total}");
    assert_eq!(bad, 0, "n_transpose_nd rank-5 位级对拍失败");
}

/// `n_attn_mm` 裸内核：batched `[b,M,K]×[b,K,N]`（QK^T 与 scores×V
/// 同构）对拍 CPU 参考。
///
/// 形状刻意与 vulkan 侧一致（b=6 行进 `real_n` 早退分支、m=5 不整齐），
/// 因为这条内核的**行索引取自在网格的 y 维**（grid = [1, batch*m, 1]），
/// 而 GLSL 的 `gl_WorkGroupID.y` 到 Metal 的映射正是本用例要钉住的。
#[test]
fn attn_mm_raw() {
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    let (batch, heads, m_, k, n) = (6usize, 3usize, 5usize, 4usize, 7usize);
    let mut seed = 0xabcd_ef01_2345_6789_u64;
    let a: Vec<f32> = (0..batch * m_ * k).map(|_| lcg(&mut seed)).collect();
    let b: Vec<f32> = (0..batch * k * n).map(|_| lcg(&mut seed)).collect();
    // 累加序 = k 升序标量（与内核/CPU MatMul 一致——换了序就是另一个
    // 数，参考值自己先漂，判据就废了）
    let mut want = vec![0f32; batch * m_ * n];
    for bi in 0..batch {
        for mi in 0..m_ {
            for j in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    acc += a[(bi * m_ + mi) * k + ki] * b[(bi * k) * n + ki * n + j];
                }
                want[(bi * m_ + mi) * n + j] = acc;
            }
        }
    }

    let seg = |x: usize| x.div_ceil(4) * 4;
    let (a_w, b_w, o_w) = (
        seg(batch * m_ * k),
        seg(batch * k * n),
        seg(batch * m_ * n),
    );
    let a_off = 4usize;
    let b_off = a_off + a_w;
    let o_off = b_off + b_w;
    let pbase = o_off + o_w;
    let (buf, mut m) = Words::new(&ctx.inner.device, pbase + 32);

    m[0] = 0; // real_n = 0：不走早退，全批真算
    for (i, v) in a.iter().enumerate() {
        m[a_off + i] = v.to_bits();
    }
    for (i, v) in b.iter().enumerate() {
        m[b_off + i] = v.to_bits();
    }
    let p = pbase;
    m[p] = a_off as u32;
    m[p + 1] = b_off as u32;
    m[p + 2] = o_off as u32;
    m[p + 3] = m_ as u32;
    m[p + 4] = k as u32;
    m[p + 5] = n as u32;
    m[p + 6] = batch as u32;
    m[p + 7] = heads as u32;

    submit_compute(&ctx.inner.queue, |enc| {
        ks.dispatch(enc, "n_attn_mm", &buf, p as u32, [1, (batch * m_) as u32, 1])
            .expect("dispatch n_attn_mm");
    })
    .expect("submit");

    let got: Vec<f32> = (0..want.len())
        .map(|i| f32::from_bits(m[o_off + i]))
        .collect();
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let bad = got
        .iter()
        .zip(&want)
        .filter(|(x, y)| (**x - **y).abs() > 1e-5)
        .count();
    eprintln!("[metal] n_attn_mm 分歧 {bad}/{} max|diff|={max_abs:.3e}", want.len());
    assert_eq!(bad, 0, "n_attn_mm 对拍失败");
}

/// `n_softmax` 裸内核对拍 CPU 参考。两档形状：一档**小而行多**
/// （`[64, 8]`——注意力分数矩阵，cols 不足 256 线程，走尾部单轮），
/// 一档**宽而列多**（`[5, 603]`——CTC 头形态，cols > 256 走多轮、
/// 且 603 不是 8 的倍数以触到标量尾段）。
#[test]
fn softmax_raw() {
    let Some(ctx) = open_or_skip() else { return };
    let ks = KernelSet::new(&ctx.inner.device).expect("编译内核");

    for (rows, cols) in [(64usize, 8usize), (5usize, 603usize)] {
        let cpad = cols.div_ceil(4) * 4;
        let mut seed = 0x0bad_c0de_dead_beef_u64 ^ (rows as u64 * 31 + cols as u64);
        // 值域拉开 [-10, 10)：exp 的多项式分支、max 平移都要真走
        let src: Vec<f32> = (0..rows * cpad)
            .map(|_| lcg(&mut seed) * 20.0 - 10.0)
            .collect();
        let want = softmax_ref(&src, rows, cols, cpad);

        let in_off = 4usize;
        let out_off = in_off + rows * cpad;
        let pbase = out_off + rows * cpad;
        let (buf, mut m) = Words::new(&ctx.inner.device, pbase + 32);

        m[0] = 0; // real_n 不限
        for (i, v) in src.iter().enumerate() {
            m[in_off + i] = v.to_bits();
        }
        let p = pbase;
        m[p] = in_off as u32;
        m[p + 1] = out_off as u32;
        m[p + 2] = rows as u32;
        m[p + 3] = cols as u32;
        m[p + 4] = cpad as u32;
        m[p + 5] = 8; // rpb：rows/批（real_n=0 时无意义，填真实值）

        submit_compute(&ctx.inner.queue, |enc| {
            ks.dispatch(enc, "n_softmax", &buf, p as u32, [rows as u32, 1, 1])
                .expect("dispatch n_softmax");
        })
        .expect("submit");

        let got: Vec<f32> = (0..rows * cpad)
            .map(|i| f32::from_bits(m[out_off + i]))
            .collect();
        let (mut max_abs, mut at) = (0f32, 0usize);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let d = (g - w).abs();
            if d > max_abs {
                max_abs = d;
                at = i;
            }
        }
        let bad = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| (**g - **w).abs() > 1e-6)
            .count();
        // 行和：softmax 的**语义**判据（不依赖参考实现）——某一行整体
        // 错时位级比较会报「一堆小差」，行和会直接报「这行不是分布」
        let mut sum_bad = 0usize;
        for r in 0..rows {
            let s: f32 = got[r * cpad..r * cpad + cols].iter().sum();
            if (s - 1.0).abs() > 1e-4 {
                sum_bad += 1;
            }
        }
        eprintln!(
            "[metal] n_softmax rows={rows} cols={cols} 分歧 {bad}/{} \
             max|diff|={max_abs:.3e}(@{at} gpu={:.6e} cpu={:.6e}) 行和异常 {sum_bad}/{rows}",
            rows * cpad,
            got[at],
            want[at]
        );
        assert_eq!(bad, 0, "n_softmax [{rows},{cols}] 对拍失败");
    }
}
