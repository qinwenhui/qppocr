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
            8 => x - b[(i / 4) / c4 * 4],
            9 => x * x,
            10 => x.sqrt(),
            11 => x / b[(i / 4) / c4 * 4],
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
        let want = elem_ref(op, src, &b_elem, p1, p2, n);
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

/// 端到端：真实 tiny rec，Metal 整图 vs CPU executor——概率图容差 +
/// softmax 行和 + 每时间步 top-1 对齐（判据口径与 vulkan 侧
/// rec_session_end_to_end 同源；直接构造会话 → 全量概率输出）。
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
    assert!(mean_abs < 1e-3, "概率均值偏差超容差: {mean_abs}");
    assert!(max_abs < 5e-2, "概率极值偏差超容差: {max_abs}");
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
    assert!(top_mis <= rows / 3, "top-1 不一致超 1/3: {top_mis}/{rows}");
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
