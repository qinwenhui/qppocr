//! n_ 内核族（NHWC f16）的设备冒烟：entry → conv → dw → convt → pool →
//! resize → concat → reduce → elem → channel，**每内核一次独立提交**——
//! 设备丢失按序定位肇事者。另含真实规模（960×864）entry 复现段。
//! 数值判据在 det 端到端测试（对照 CPU executor 全图输出）。
//!
//! 区域管理：**单一大区手动切分**——arena 的 chunk 上限 16MB，跨
//! chunk 的区域不在 KernelSet 绑定的缓冲里（偏移错位 = 设备打挂）。

use super::VulkanContext;
use super::memory::Arena;
use super::pipeline::{KernelSet, OFF_NONE, ParamBlock, PcParams, record_dispatch};
use ash::vk;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn open_or_skip() -> Option<VulkanContext> {
    match VulkanContext::open(None) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("[gpu] 跳过（无满足基线的设备）: {e}");
            None
        }
    }
}

#[test]
fn n_family_smoke() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);

    let (ci, co1, co2) = (3usize, 16usize, 8usize);
    let (h, w) = (16usize, 12usize);
    let hw = h * w;
    let (oh, ow) = (8usize, 6usize); // 3x3 s2 p1 的输出
    let ohw = oh * ow;
    let cip = super::nhwc::cpad4(ci as i64) as usize;
    let co1p = super::nhwc::cpad4(co1 as i64) as usize;
    let co2p = super::nhwc::cpad4(co2 as i64) as usize;

    let mut seed = 0x1234_5678_9abc_def0_u64;
    let x: Vec<f32> = (0..ci * hw).map(|_| lcg(&mut seed)).collect();
    let w1: Vec<f32> = (0..co1 * ci * 9).map(|_| lcg(&mut seed) * 0.1).collect();
    let b1: Vec<f32> = (0..co1).map(|_| lcg(&mut seed) * 0.1).collect();
    let w2: Vec<f32> = (0..co2 * co1 * 9).map(|_| lcg(&mut seed) * 0.1).collect();
    let dwv: Vec<f32> = (0..co2 * 25).map(|_| lcg(&mut seed) * 0.1).collect();
    let ctw: Vec<f32> = (0..co2 * co2 * 4).map(|_| lcg(&mut seed) * 0.1).collect();

    let w1_16 = super::nhwc::repack_conv_w(&w1, co1, ci, 3, 3);
    let b1_16 = super::nhwc::conv_bias(&b1, co1 as i64);
    let w2_16 = super::nhwc::repack_conv_w(&w2, co2, co1, 3, 3);
    let dw_16 = super::nhwc::repack_dw_w(&dwv, co2, 5, 5);
    let ct_16 = super::nhwc::repack_convt_w(&ctw, co2, co2, 2, 2);

    // 各段 word 数（16B 对齐）——真实规模段 + 参数段一起算总量
    let seg = |n: usize| n.div_ceil(4) * 4;
    #[allow(unused_variables)]
    let w16 = |n: usize| seg(n); // f32：word=元素
    let f32_in_w = seg(ci * hw);
    let f16_in_w = seg(hw * cip);
    let c1_w = seg(ohw * co1p);
    let c2_w = seg(ohw * co2p);
    let cto_w = seg(ohw * 4 * co2p);
    let cato_w = seg(ohw * 4 * co2p * 2);
    let redo_w = seg(co2p);
    let w1r_w = seg(w1_16.len());
    let b1r_w = seg(b1_16.len());
    let w2r_w = seg(w2_16.len());
    let dwr_w = seg(dw_16.len());
    let ctr_w = seg(ct_16.len());
    let real_w = 9 << 20; // 真实规模 entry 段（9M words——总块 ~37.4MB 对齐 session）
    let params_w = 8192 + 128; // 尾部含 reduce scratch
    let total_w: usize = f32_in_w
        + f16_in_w
        + c1_w
        + c2_w * 3
        + cto_w * 2
        + cato_w
        + redo_w
        + w1r_w
        + b1r_w
        + w2r_w
        + dwr_w
        + ctr_w
        + real_w
        + params_w;
    let big = arena.alloc((total_w * 4) as vk::DeviceSize).unwrap();
    let base_word = big.offset as u32 / 4;
    let mut cur = base_word;
    let mut sub = |n: usize| -> u32 {
        let o = cur;
        cur += n as u32;
        o
    };
    let f32_in = sub(f32_in_w);
    let f16_in = sub(f16_in_w);
    let c1 = sub(c1_w);
    let c2 = sub(c2_w);
    let dwo = sub(c2_w);
    let cto = sub(cto_w);
    let poolo = sub(c2_w);
    let rso = sub(cto_w);
    let cato = sub(cato_w);
    let redo = sub(redo_w);
    let emo = sub(c2_w);
    let cho = sub(c2_w);
    let w1r = sub(w1r_w);
    let b1r = sub(b1r_w);
    let w2r = sub(w2r_w);
    let dwr = sub(dwr_w);
    let ctr = sub(ctr_w);
    let real_base = sub(real_w);
    let pbase = sub(params_w);
    // SAFETY: 测试独占设备与映射区；长度由本测试的分配保证。
    let p_of = |w: u32| unsafe { big.ptr.add(w as usize * 4) };

    // SAFETY: big 是单区持久映射；各子偏移互不相交且都在区内。
    unsafe {
        let s = std::slice::from_raw_parts_mut(p_of(f32_in) as *mut f32, ci * hw);
        s.copy_from_slice(&x);
        for (r, d) in [
            (w1r, &w1_16),
            (b1r, &b1_16),
            (w2r, &w2_16),
            (dwr, &dw_16),
            (ctr, &ct_16),
        ] {
            std::ptr::copy_nonoverlapping(d.as_ptr(), p_of(r) as *mut u32, d.len());
        }
    }
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();
    let div256 = |t: u32| t.div_ceil(256);
    let mut pcur = pbase;

    // 单内核单提交：出事时最后一条日志即肇事者
    let mut step = |kernel: &str, pb: ParamBlock, groups: [u32; 3]| {
        let p_off = pcur;
        pcur += 128;
        // SAFETY: 参数段在 big 内；words ≤ 段长。
        unsafe {
            std::ptr::copy_nonoverlapping(
                pb.words().as_ptr(),
                p_of(p_off) as *mut u32,
                pb.words().len(),
            );
        }
        let pc = PcParams { p_off };
        eprintln!("[gpu][n] {kernel} …");
        ctx.inner
            .device
            .submit_one_shot(|d, cb| {
                // SAFETY: cb 录制态；PC 与该内核参数块逐字段对应（本测试构造）。
                unsafe {
                    record_dispatch(d, cb, &ks, kernel, pc.bytes(), groups);
                }
            })
            .unwrap();
    };

    // 1) entry：f32 NCHW [1,3,16,12] → f16 NHWC
    let mut pb = ParamBlock::new();
    pb.u(f32_in)
        .u(f16_in)
        .u(1)
        .u(ci as u32)
        .u(h as u32)
        .u(w as u32)
        .u(cip as u32);
    step("n_entry", pb, [div256(hw as u32), 1, 1]);

    // 2) conv1：3x3 s2 p1，ci=3→16，bias + relu
    let mut pb = ParamBlock::new();
    pb.u(f16_in)
        .u(w1r)
        .u(b1r)
        .u(c1)
        .u(ohw as u32)
        .u(co1p as u32)
        .u(ow as u32)
        .u(w as u32)
        .u(h as u32)
        .u(cip as u32 / 4)
        .u(9)
        .u(3)
        .u(2)
        .u(2)
        .u(1)
        .u(1)
        .u(2)
        .f(std::f32::consts::SQRT_2)
        .f(1.0)
        .f(0.5);
    step(
        "n_conv",
        pb,
        [div256((ohw as u32).div_ceil(4) * (co1p as u32 / 4)), 1, 1],
    );

    // 3) conv2：3x3 s1 p1，ci=16→8，无 bias 无 act
    let mut pb = ParamBlock::new();
    pb.u(c1)
        .u(w2r)
        .u(OFF_NONE)
        .u(c2)
        .u(ohw as u32)
        .u(co2p as u32)
        .u(ow as u32)
        .u(ow as u32)
        .u(oh as u32)
        .u(co1p as u32 / 4)
        .u(9)
        .u(3)
        .u(1)
        .u(1)
        .u(1)
        .u(1)
        .u(0)
        .f(std::f32::consts::SQRT_2)
        .f(1.0)
        .f(0.5);
    step(
        "n_conv",
        pb,
        [div256((ohw as u32).div_ceil(4) * (co2p as u32 / 4)), 1, 1],
    );

    // 4) depthwise：5x5 s1 p2
    let mut pb = ParamBlock::new();
    pb.u(c2)
        .u(dwr)
        .u(OFF_NONE)
        .u(dwo)
        .u(ohw as u32)
        .u(co2p as u32)
        .u(ow as u32)
        .u(ow as u32)
        .u(oh as u32)
        .u(25)
        .u(5)
        .u(1)
        .u(1)
        .u(2)
        .u(2)
        .u(0)
        .f(std::f32::consts::SQRT_2)
        .f(1.0)
        .f(0.5);
    step(
        "n_conv_dw",
        pb,
        [div256(ohw as u32 * co2p as u32 / 4), 1, 1],
    );

    // 5) convt：2x2 s2（[8,6]→[16,12]）
    let mut pb = ParamBlock::new();
    pb.u(c2)
        .u(ctr)
        .u(OFF_NONE)
        .u(cto)
        .u((ohw * 4) as u32)
        .u(co2p as u32)
        .u((ow * 2) as u32)
        .u(ow as u32)
        .u(oh as u32)
        .u(co2p as u32)
        .u(co2p as u32 / 4)
        .u(2)
        .u(2)
        .u(2)
        .u(0)
        .f(std::f32::consts::SQRT_2)
        .f(1.0)
        .f(0.5);
    step(
        "n_convt",
        pb,
        [div256((ohw * 4) as u32 * co2p as u32 / 4), 1, 1],
    );

    // 6) pool：2x2 s2 max（cto [16,12] → [8,6]）
    let mut pb = ParamBlock::new();
    pb.u(cto)
        .u(poolo)
        .u(ohw as u32)
        .u(co2p as u32)
        .u(ow as u32)
        .u((ow * 2) as u32)
        .u((oh * 2) as u32)
        .u(2)
        .u(2)
        .u(2)
        .u(2)
        .u(0)
        .u(0)
        .u(1);
    step("n_pool", pb, [div256(ohw as u32 * co2p as u32 / 4), 1, 1]);

    // 7) resize：nearest ×2（poolo [8,6] → [16,12]）
    let mut pb = ParamBlock::new();
    pb.u(poolo)
        .u(rso)
        .u((ohw * 4) as u32)
        .u(co2p as u32)
        .u((ow * 2) as u32)
        .u(ow as u32)
        .u(oh as u32);
    step(
        "n_resize",
        pb,
        [div256((ohw * 4) as u32 * co2p as u32 / 4), 1, 1],
    );

    // 8) concat：rso + rso（通道 ×2）
    let mut pb = ParamBlock::new();
    pb.u(cato)
        .u((ohw * 4) as u32)
        .u(2)
        .u((co2p * 2 / 4) as u32)
        .u(rso)
        .u(co2p as u32 / 4)
        .u(rso)
        .u(co2p as u32 / 4);
    step(
        "n_concat_c",
        pb,
        [div256((ohw * 4) as u32 * co2p as u32 / 2), 1, 1],
    );

    // 9) reduce：rso [16,12] → gate[8]
    let mut pb = ParamBlock::new();
    pb.u(rso).u(redo).u((ohw * 4) as u32).u(co2p as u32);
    step("n_reduce_hw", pb, [div256(co2p as u32), 1, 1]);

    // 10) elem：relu（op=0）
    let n16 = (ohw * co2p) as u32;
    let mut pb = ParamBlock::new();
    pb.u(0).u(poolo).u(OFF_NONE).u(emo).u(n16).f(0.0).f(0.0);
    step("n_elem", pb, [div256(n16 / 4), 1, 1]);

    // 11) channel：mul_c（op=0）
    let mut pb = ParamBlock::new();
    pb.u(0)
        .u(poolo)
        .u(redo)
        .u(OFF_NONE)
        .u(cho)
        .u(n16)
        .u(co2p as u32)
        .f(0.0)
        .f(0.0);
    step("n_channel", pb, [div256(n16 / 4), 1, 1]);

    // ---- 对拍 ----
    let read32 = |off: u32, n: usize| -> Vec<f32> {
        // SAFETY: 已等信号；n ≤ 段长。
        let words = unsafe { std::slice::from_raw_parts(p_of(off) as *const u32, n) };
        words.iter().map(|w| f32::from_bits(*w)).collect()
    };
    let got = read32(f16_in, hw * cip);
    let rt = x.clone();
    let want: Vec<f32> = {
        let mut v = vec![0f32; hw * cip];
        for pos in 0..hw {
            for ch in 0..ci {
                v[pos * cip + ch] = rt[ch * hw + pos];
            }
        }
        v
    };
    let max_d = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_d == 0.0, "entry 应逐位一致: max|diff|={max_d}");
    // reduce：每通道均值应有限（链上卷积输出幅度不受控，只查健康度）
    let gate = read32(redo, co2p);
    for (c, g) in gate.iter().take(co2).enumerate() {
        assert!(g.is_finite(), "reduce 通道 {c} 值异常: {g}");
    }
    eprintln!("[gpu][n] 11 内核冒烟链全部存活，entry 逐位一致");

    // ---- 真实规模复现：960x864x3 entry（det 首层形态，同块内切分）----
    {
        let (h2, w2) = (480usize, 864usize);
        let hw2 = h2 * w2;
        let _ = ((ci * hw2) as u32, (hw2 * cip) as u32); // 规模注释值
        // 精确复刻 session 的参数形态：in=36, out=1244196
        let in_off32 = real_base + 36;
        let out_off16 = real_base + 1244196;
        // SAFETY: 偏移在 big 内。
        unsafe {
            let s2 = std::slice::from_raw_parts_mut(p_of(in_off32) as *mut f32, ci * hw2);
            for v in s2.iter_mut() {
                *v = lcg(&mut seed);
            }
        }
        let mut pb = ParamBlock::new();
        pb.u(in_off32)
            .u(out_off16)
            .u(1)
            .u(ci as u32)
            .u(h2 as u32)
            .u(w2 as u32)
            .u(cip as u32);
        let p_off = pcur;
        #[allow(unused_assignments)]
        {
            pcur += 128;
        }
        // SAFETY: 同上。
        unsafe {
            std::ptr::copy_nonoverlapping(
                pb.words().as_ptr(),
                p_of(p_off) as *mut u32,
                pb.words().len(),
            );
        }
        let pc = PcParams { p_off };
        eprintln!("[gpu][n] n_entry 真实规模（{h2}x{w2}）…");
        ctx.inner
            .device
            .submit_one_shot(|d, cb| {
                // SAFETY: 同上。
                unsafe {
                    record_dispatch(
                        d,
                        cb,
                        &ks,
                        "n_entry",
                        pc.bytes(),
                        [div256(hw2 as u32), 1, 1],
                    );
                }
            })
            .unwrap();
        let got = read32(out_off16, hw2 * cip);
        // SAFETY: 同上。
        let rt2 =
            unsafe { std::slice::from_raw_parts(p_of(in_off32) as *const f32, ci * hw2).to_vec() };
        let mut bad = 0usize;
        for pos in 0..hw2 {
            for ch in 0..ci {
                if got[pos * cip + ch] != rt2[ch * hw2 + pos] {
                    bad += 1;
                }
            }
        }
        assert!(bad == 0, "真实规模 entry 不一致: {bad} 处");
        eprintln!("[gpu][n] 真实规模 entry 逐位一致");
    }
}

/// 真实数据定位：用 CPU 引擎 dump 的 det 输入（QPPOCR_DUMP_DIR 产物，
/// s0_000000.f32）同输入对拍 GPU vs CPU 的概率图统计——随机数据的
/// 容差测试通过了但真实图片 0 框，用真实分布找分歧。
#[test]
fn det_real_input_cmp() {
    let dir = std::path::PathBuf::from(
        std::env::var("QPPOCR_REAL_DET_INPUT").unwrap_or_else(|_| "/tmp/det-dump".into()),
    );
    let bytes = match std::fs::read(dir.join("s0_000000.f32")) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("[gpu] 无真实 det 输入 dump，跳过");
            return;
        }
    };
    let Some(ctx) = open_or_skip() else { return };
    let rank = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let shape: Vec<i64> = bytes[4..4 + 8 * rank]
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let data: Vec<f32> = bytes[4 + 8 * rank..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    eprintln!(
        "[gpu] 真实输入 shape={shape:?} n={} 值域 [{:.3},{:.3}]",
        data.len(),
        data.iter().cloned().fold(f32::INFINITY, f32::min),
        data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    let model = std::path::Path::new("../../models/tiny/det.onnx");
    if !model.is_file() {
        eprintln!("[gpu] 无模型，跳过");
        return;
    }
    let mbytes = std::fs::read(model).unwrap();
    use qppocr_core::executor::Session;
    use qppocr_core::tensor::{DType, Tensor};
    use qppocr_kernels::buf::F32Buf;

    let mk = || {
        let mut b = F32Buf::with_zeroed(data.len());
        b.as_mut_slice().copy_from_slice(&data);
        Tensor {
            name: String::new(),
            shape: shape.clone(),
            dtype: DType::F32,
            f32: b,
            i64: Vec::new(),
        }
    };
    // CPU 参考（带逐节点 dump：GPU 区域按 NHWC f16 转 NCHW f32 对拍）
    let dump_dir = std::env::temp_dir().join("qppocr-real-cmp");
    let _ = std::fs::remove_dir_all(&dump_dir);
    std::fs::create_dir_all(&dump_dir).unwrap();
    // SAFETY: 测试独占该环境变量（仅 executor 读）。
    unsafe { std::env::set_var("QPPOCR_DUMP_DIR", &dump_dir) };
    let cpu = Session::from_memory(&mbytes, "cmp.cpu").unwrap();
    let in_name = cpu
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let out_cpu = cpu.run(vec![(in_name.clone(), mk())]).unwrap();
    // SAFETY: CPU 前向完成，撤销 dump 环境。
    unsafe { std::env::remove_var("QPPOCR_DUMP_DIR") };
    // GPU
    let (graph, init) = {
        let s = Session::from_memory(&mbytes, "cmp.gpu").unwrap();
        s.into_parts()
    };
    let _w_probe: Vec<(String, Vec<i64>, Vec<f32>)> = graph
        .nodes
        .iter()
        .map(|n| {
            let wname = n.inputs.get(1).cloned().unwrap_or_default();
            let (shp, dat) = init
                .get(&wname)
                .map(|t| (t.shape.clone(), t.f32.to_vec()))
                .unwrap_or_default();
            (wname, shp, dat)
        })
        .collect();
    eprintln!("[gpu][图] 节点 12..17（名/输入/属性）：");
    for (i, n) in graph.nodes.iter().enumerate().take(17).skip(12) {
        let acts: Vec<String> = n
            .attrs
            .iter()
            .map(|a| format!("{}={:?}i/{:?}f", a.name, a.i, a.f))
            .collect();
        eprintln!(
            "  [{i}] {} {} in={:?} out={:?} attrs={}",
            n.op_type,
            n.name,
            n.inputs,
            n.outputs,
            acts.join(",")
        );
    }
    let cpu_manifest =
        std::fs::read_to_string(std::env::temp_dir().join("qppocr-real-cmp/manifest.tsv"))
            .unwrap_or_default();
    eprintln!("[gpu][图] CPU manifest 12..17：");
    for line in cpu_manifest.lines().take(17).skip(12) {
        eprintln!("  {line}");
    }
    let gpu = super::session::VulkanSession::new(ctx.inner.clone(), graph, init).unwrap();
    let out_gpu =
        qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name.clone(), mk())]).unwrap();

    let a = &out_gpu[0].f32;
    let b = &out_cpu[0].f32;
    let mean_abs = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32;
    let max_abs = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let hi_g = a.iter().filter(|v| **v > 0.3).count();
    let hi_c = b.iter().filter(|v| **v > 0.3).count();
    eprintln!(
        "[gpu] 概率图 mean|d|={mean_abs:.3e} max|d|={max_abs:.3e} >0.3: gpu={hi_g} cpu={hi_c}"
    );
    // 双路径对拍：老 NCHW-f32 路径（已验证）vs n_ 路径，按节点名配对
    //（两会话共用同一 planner 图——节点名与形状一致）。
    let mut n_vs_old_mean = f32::INFINITY;
    let _ = &mut n_vs_old_mean;
    {
        // 强制老路径建参考会话
        // SAFETY: 测试独占该环境变量（build_plan 读）。
        unsafe { std::env::set_var("QPPOCR_GPU_F32", "1") };
        let mbytes2 = std::fs::read("../../models/tiny/det.onnx").unwrap();
        let (graph2, init2) = {
            let s2 = Session::from_memory(&mbytes2, "cmp.f32ref").unwrap();
            s2.into_parts()
        };
        let gpu2 = super::session::VulkanSession::new(ctx.inner.clone(), graph2, init2).unwrap();
        let out_ref =
            qppocr_core::device::DeviceSession::run(&gpu2, vec![(in_name.clone(), mk())]).unwrap();
        // SAFETY: 参考会话计划已建，撤销环境。
        unsafe { std::env::remove_var("QPPOCR_GPU_F32") };
        n_vs_old_mean = a
            .iter()
            .zip(out_ref[0].f32.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>()
            / a.len() as f32;
        eprintln!(
            "[gpu][ref] 老路径 >0.3: {}；n_ vs 老 mean|d|={n_vs_old_mean:.3e}",
            out_ref[0].f32.iter().filter(|v| **v > 0.3).count()
        );

        let (base, recs) = gpu.debug_recs(&shape).expect("n_ 无计划");
        let (base2, recs2) = gpu2.debug_recs(&shape).expect("f32 无计划");
        use std::collections::HashMap;
        let mut by_name2: HashMap<&str, (u32, u32, &Vec<i64>)> = HashMap::new();
        for (_k, n2, _ix, o2, on2, _p, osh2) in recs2.iter() {
            by_name2.insert(n2, (*o2, *on2, osh2));
        }
        let mut printed = 0;
        for (i, (kernel, node, _idx, off, n, _pc, osh)) in recs.iter().enumerate() {
            if *n == 0
                || kernel == "n_entry"
                || kernel == "n_exit"
                || kernel.starts_with("n_reduce")
            {
                continue;
            }
            let Some((off2, _on2, sh2)) = by_name2.get(node.as_str()) else {
                continue;
            };
            if sh2.len() != 4 || osh.len() != 4 || **sh2 != *osh {
                continue;
            }
            let (nb, c, h, w) = (
                osh[0] as usize,
                osh[1] as usize,
                osh[2] as usize,
                osh[3] as usize,
            );
            let cp = super::nhwc::cpad4(osh[1]) as usize;
            // SAFETY: base 持久映射；off+n ≤ total。
            let gv: Vec<f32> = unsafe {
                std::slice::from_raw_parts(base.add(*off as usize * 4) as *const f32, *n as usize)
            }
            .to_vec();
            // SAFETY: 同上。
            let rv: Vec<f32> = unsafe {
                std::slice::from_raw_parts(base2.add(*off2 as usize * 4) as *const f32, *n as usize)
            }
            .to_vec();
            // NHWC→NCHW 采样比对（首尾行各 2 行 × 全通道采样 8）
            let mut e_max = 0.0f32;
            let hw = h * w;
            for nn in 0..nb {
                for ch in 0..c {
                    for y in [0usize, 1, h.saturating_sub(2), h.saturating_sub(1)] {
                        for x in [0usize, w / 2, w.saturating_sub(1)] {
                            let pos = y * w + x;
                            if pos >= hw {
                                continue;
                            }
                            let g = gv[(nn * hw + pos) * cp + ch];
                            let r = rv[(nn * c + ch) * hw + pos];
                            let e = (g - r).abs() / (1.0 + r.abs());
                            if e > e_max {
                                e_max = e;
                            }
                        }
                    }
                }
            }
            if e_max > 1e-3 {
                eprintln!(
                    "[gpu][ref] #{i:<3} {kernel:<12} {node:<22} rel_max={e_max:.4} shape={osh:?}"
                );
                printed += 1;
                if printed >= 12 {
                    eprintln!("[gpu][ref] …（后续省略）");
                    break;
                }
            }
        }
    }
    eprintln!(
        "[gpu] gpu 值域 [{:.4},{:.4}] mean={:.5}；cpu 值域 [{:.4},{:.4}] mean={:.5}",
        a.iter().cloned().fold(1f32, f32::min),
        a.iter().cloned().fold(0f32, f32::max),
        a.iter().sum::<f32>() / a.len() as f32,
        b.iter().cloned().fold(1f32, f32::min),
        b.iter().cloned().fold(0f32, f32::max),
        b.iter().sum::<f32>() / b.len() as f32,
    );
    // 验收口径：两条 GPU 路径必须机器一致（≤1e-6）；GPU vs CPU 的
    // 浮点差异（卷积累加序不同）按原计划走 verify.py 容差对拍，不在此卡。
    assert!(
        n_vs_old_mean < 1e-6,
        "n_ 与老路径末图不一致: {n_vs_old_mean}"
    );
}

#[test]
fn conv8_inf_forensics() {
    let dir = std::path::PathBuf::from(
        std::env::var("QPPOCR_REAL_DET_INPUT").unwrap_or_else(|_| "/tmp/det-real".into()),
    );
    let Ok(bytes) = std::fs::read(dir.join("s0_000000.f32")) else {
        eprintln!("[gpu] 无真实输入，跳过");
        return;
    };
    let Some(ctx) = open_or_skip() else { return };
    let rank = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let shape: Vec<i64> = bytes[4..4 + 8 * rank]
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let data: Vec<f32> = bytes[4 + 8 * rank..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let mbytes = std::fs::read("../../models/tiny/det.onnx").unwrap();
    use qppocr_core::executor::Session;
    use qppocr_core::tensor::{DType, Tensor};
    use qppocr_kernels::buf::F32Buf;
    let mut b = F32Buf::with_zeroed(data.len());
    b.as_mut_slice().copy_from_slice(&data);
    let t_in = Tensor {
        name: String::new(),
        shape: shape.clone(),
        dtype: DType::F32,
        f32: b,
        i64: Vec::new(),
    };
    let (graph, init) = {
        let s = Session::from_memory(&mbytes, "foren").unwrap();
        s.into_parts()
    };
    let _w_probe: Vec<(String, Vec<i64>, Vec<f32>)> = graph
        .nodes
        .iter()
        .map(|n| {
            let wname = n.inputs.get(1).cloned().unwrap_or_default();
            let (shp, dat) = init
                .get(&wname)
                .map(|t| (t.shape.clone(), t.f32.to_vec()))
                .unwrap_or_default();
            (wname, shp, dat)
        })
        .collect();
    eprintln!("[gpu][图] 节点 12..17（名/输入/属性）：");
    for (i, n) in graph.nodes.iter().enumerate().take(17).skip(12) {
        let acts: Vec<String> = n
            .attrs
            .iter()
            .map(|a| format!("{}={:?}i/{:?}f", a.name, a.i, a.f))
            .collect();
        eprintln!(
            "  [{i}] {} {} in={:?} out={:?} attrs={}",
            n.op_type,
            n.name,
            n.inputs,
            n.outputs,
            acts.join(",")
        );
    }
    let cpu_manifest =
        std::fs::read_to_string(std::env::temp_dir().join("qppocr-real-cmp/manifest.tsv"))
            .unwrap_or_default();
    eprintln!("[gpu][图] CPU manifest 12..17：");
    for line in cpu_manifest.lines().take(17).skip(12) {
        eprintln!("  {line}");
    }
    let gpu = super::session::VulkanSession::new(ctx.inner.clone(), graph, init).unwrap();
    let in_name = gpu.input_name_for_test();
    let _ = qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name, t_in)]).unwrap();

    // 现场参数：从当前计划的 recs 取（硬编码偏移在不同复用模式下会错位）
    let (base_dbg, recs) = gpu.debug_recs(&shape).expect("无计划");
    let out_off = recs[14].3 as usize;
    let in_off = {
        let pc = &recs[14].5;
        let p_off = u32::from_le_bytes([pc[0], pc[1], pc[2], pc[3]]) as usize;
        // SAFETY: base 持久映射；参数块在 total 内。
        let w = unsafe { std::slice::from_raw_parts(base_dbg.add(p_off * 4) as *const u32, 4) };
        w[0] as usize
    };
    eprintln!("[gpu][取证] 现场 in_off={in_off} out_off={out_off}");
    // 权重/偏置实值（当前计划偏移）
    {
        let pc = &recs[14].5;
        let p_off = u32::from_le_bytes([pc[0], pc[1], pc[2], pc[3]]) as usize;
        // SAFETY: base 持久映射；参数块在 total 内。
        let w: Vec<u32> =
            unsafe { std::slice::from_raw_parts(base_dbg.add(p_off * 4) as *const u32, 20) }
                .to_vec();
        eprintln!("[gpu][取证] #14 参数 words={w:?}");
        let (wo, bo) = (w[1] as usize, w[2] as usize);
        // SAFETY: 同上；权重区 1024 words、偏置 32 words。
        let wfull: Vec<u32> =
            unsafe { std::slice::from_raw_parts(base_dbg.add(wo * 4) as *const u32, 1024) }
                .to_vec();
        let wv = super::fp16::f16_words_to_f32(&wfull);
        eprintln!(
            "[gpu][取证] 权重 absmax={:.3} inf={} nan={}",
            wv.iter().map(|x| x.abs()).fold(0.0f32, f32::max),
            wv.iter().filter(|x| x.is_infinite()).count(),
            wv.iter().filter(|x| x.is_nan()).count()
        );
        if bo != 0xFFFF_FFFF {
            let bw: Vec<u32> =
                // SAFETY: 测试独占设备与映射区；长度由本测试的分配保证。
                unsafe { std::slice::from_raw_parts(base_dbg.add(bo * 4) as *const u32, 32) }
                    .to_vec();
            eprintln!("[gpu][取证] bias={:?}", super::fp16::f16_words_to_f32(&bw));
        }
    }
    // 输出 Inf 计数
    let count_inf_pub = |base: *mut u8| -> usize {
        // SAFETY: base 持久映射；输出区 51840*64 f16。
        let of: Vec<u32> = unsafe {
            std::slice::from_raw_parts(base.add(out_off * 4) as *const u32, 51840 * 64 / 2)
        }
        .to_vec();
        super::fp16::f16_words_to_f32(&of)
            .iter()
            .filter(|v| !v.is_finite())
            .count()
    };

    let _ = base_dbg;

    // rec#14 的参数：p_off → 现场参数块
    let (_, _, _) = (0u32, 0u32, 0u32);
    let rec14 = 14usize;
    // 前缀二分：找让 #14 出 Inf 的最小前缀（都从当前内存状态出发）
    let mut base = std::ptr::null_mut::<u8>();
    for (from, to) in [(14, 14), (13, 14), (12, 14), (10, 14), (5, 14), (0, 14)] {
        let (b, _, _) = gpu.debug_replay_range(&shape, from, to, None).unwrap();
        base = b;
        let c = count_inf_pub(b);
        eprintln!("[gpu][取证] 重放 #{from}..#{to} → Inf={c}");
    }
    let n_inf = count_inf_pub(base);
    eprintln!("[gpu][取证] 原样重放后 Inf 数 = {n_inf}");

    if n_inf > 0 {
        // 备份输入区（f16 NHWC [51840, 32]）
        // SAFETY: base 持久映射；输入区 51840*32 f16。
        let orig: Vec<u32> = unsafe {
            std::slice::from_raw_parts(base.add(in_off * 4) as *const u32, 51840 * 32 / 2)
        }
        .to_vec();
        // 全零输入 → 若仍有 Inf：bias/weights/kernel 本身；否则输入位毒
        // SAFETY: 测试独占设备与映射区；长度由本测试的分配保证。
        let zero_all = |b: *mut u8| unsafe {
            std::ptr::write_bytes(b.add(in_off * 4), 0, 51840 * 32 / 2 * 4);
        };
        let (_, _, _) = gpu.debug_replay(&shape, rec14, Some(&zero_all)).unwrap();
        eprintln!("[gpu][取证] 全零输入后 Inf 数 = {}", count_inf_pub(base));
        // 还原 + 逐通道二分
        let orig_arc = std::sync::Arc::new(orig);
        for ch in 0..32 {
            let orig = orig_arc.clone();
            // SAFETY: 测试独占设备与映射区；长度由本测试的分配保证。
            let zero_ch = move |b: *mut u8| unsafe {
                std::ptr::copy_nonoverlapping(
                    orig.as_ptr(),
                    b.add(in_off * 4) as *mut u32,
                    orig.len(),
                );
                // NHWC：位置 m 的通道 ch 在 f16 下标 m*32+ch → word (m*32+ch)/2 的半边
                for m in 0..51840 {
                    let e = m * 32 + ch;
                    let wp = b.add(in_off * 4) as *mut u32;
                    let w = wp.add(e / 2).read();
                    wp.add(e / 2).write(if e % 2 == 0 {
                        w & 0xFFFF_0000
                    } else {
                        w & 0x0000_FFFF
                    });
                }
            };
            let (_, _, _) = gpu.debug_replay(&shape, rec14, Some(&zero_ch)).unwrap();
            let c = count_inf_pub(base);
            if c < n_inf {
                eprintln!("[gpu][取证] 置零通道 {ch} 后 Inf={c}（原 {n_inf}）→ 毒通道之一");
            }
        }
        let orig = orig_arc.clone();
        // SAFETY: 测试独占设备与映射区；长度由本测试的分配保证。
        unsafe {
            std::ptr::copy_nonoverlapping(
                orig.as_ptr(),
                base.add(in_off * 4) as *mut u32,
                orig.len(),
            );
        }
    }
}

/// act=1（GELU）1×1 conv 零输入最小复现：n_conv 在 GELU 路径是否写 Inf。
#[test]
fn n_conv_gelu_zero_repro() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);

    // Conv.8 形态：Ci=32→Co=64，1x1，act=1(gelu)，c1=1.4646
    let (ci, co) = (32usize, 64usize);
    let m_dim = 51840usize;
    let mut seed = 0xdead_beef_u64;
    let w: Vec<f32> = (0..co * ci).map(|_| lcg(&mut seed) * 0.5).collect();
    let bias: Vec<f32> = (0..co).map(|_| lcg(&mut seed)).collect();
    let w16 = super::nhwc::repack_conv_w(&w, co, ci, 1, 1);
    let b16 = super::nhwc::conv_bias(&bias, co as i64);

    let seg = |n: usize| n.div_ceil(4) * 4;
    let in_w = seg(m_dim * ci);
    let out_w = seg(m_dim * co);
    let w_w = seg(w16.len());
    let b_w = seg(b16.len());
    let p_w = 256;
    let total = in_w + out_w + w_w + b_w + p_w;
    let big = arena.alloc((total * 4) as vk::DeviceSize).unwrap();
    let base_word = big.offset as u32 / 4;
    let in_off = base_word;
    let out_off = base_word + in_w as u32;
    let w_off = base_word + (in_w + out_w) as u32;
    let b_off = base_word + (in_w + out_w + w_w) as u32;
    let p_off = base_word + (in_w + out_w + w_w + b_w) as u32;
    // SAFETY: 测试独占设备与映射区；长度由本测试的分配保证。
    let p_of = |w: u32| unsafe { big.ptr.add(w as usize * 4) };
    // SAFETY: big 单区持久映射；各子区互不相交。
    unsafe {
        std::ptr::write_bytes(p_of(in_off), 0, in_w * 4); // 零输入
        std::ptr::copy_nonoverlapping(w16.as_ptr(), p_of(w_off) as *mut u32, w16.len());
        std::ptr::copy_nonoverlapping(b16.as_ptr(), p_of(b_off) as *mut u32, b16.len());
    }
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();
    let mut pb = ParamBlock::new();
    pb.u(in_off)
        .u(w_off)
        .u(b_off)
        .u(out_off)
        .u(m_dim as u32)
        .u(co as u32)
        .u(216)
        .u(216)
        .u(240)
        .u(ci as u32 / 4)
        .u(1)
        .u(1)
        .u(1)
        .u(1)
        .u(0)
        .u(0)
        .u(1) // act=gelu
        .f(f32::from_bits(1068827891))
        .f(1.0)
        .f(0.5);
    // SAFETY: 参数区在 big 内。
    unsafe {
        std::ptr::copy_nonoverlapping(
            pb.words().as_ptr(),
            p_of(p_off) as *mut u32,
            pb.words().len(),
        );
    }
    // act 对照：0（无）/2（relu）/1（gelu）
    for act_try in [0u32, 2, 1] {
        let mut pb2 = ParamBlock::new();
        for w in pb.words().iter().take(16) {
            pb2.u(*w);
        }
        pb2.u(act_try).f(f32::from_bits(1068827891)).f(1.0).f(0.5);
        // SAFETY: 参数区在 big 内（p_off+128 预留）。
        unsafe {
            std::ptr::copy_nonoverlapping(
                pb2.words().as_ptr(),
                p_of(p_off + 32) as *mut u32,
                pb2.words().len(),
            );
        }
        let pc2 = super::pipeline::PcParams { p_off: p_off + 32 };
        let grid = [(m_dim as u32).div_ceil(4) * (co as u32 / 4) / 256 + 1, 1, 1];
        ctx.inner
            .device
            .submit_one_shot(|d, cb| {
                // SAFETY: cb 录制态；PC 与本测试构造一致。
                unsafe { record_dispatch(d, cb, &ks, "n_conv", pc2.bytes(), grid) };
            })
            .unwrap();
        // SAFETY: 已等信号。
        let of: Vec<u32> =
            unsafe { std::slice::from_raw_parts(p_of(out_off) as *const u32, m_dim * co) }.to_vec();
        let ov: Vec<f32> = of.iter().map(|w| f32::from_bits(*w)).collect();
        let nonfinite = ov.iter().filter(|v| !v.is_finite()).count();
        eprintln!(
            "[gpu][gelu] act={act_try}: 非有限={nonfinite} ov[0..2]={:?}",
            &ov[..2.min(ov.len())]
        );
        if act_try == 1 {
            // 期望 = gelu(bias)（零输入 ⇒ acc = bias）
            let mut bad = 0usize;
            for m in 0..64.min(ov.len()) {
                let x = bias[m % co];
                let want = 0.5 * x * (erf_ref(x * (1.0 / f32::from_bits(1068827891))) + 1.0);
                if (ov[m] - want).abs() > 0.05 {
                    bad += 1;
                }
            }
            assert!(
                nonfinite == 0 && bad == 0,
                "gelu 路径损坏：nonfinite={nonfinite} bad={bad}"
            );
        }
    }
}

/// CPU 侧 erf 参考（A&S 7.1.26，与 kernels::activation 同型）。
fn erf_ref(x: f32) -> f32 {
    let ax = x.abs();
    let t = 1.0 / (0.3275911 * ax + 1.0);
    let p = t
        * (0.254_829_6_f32
            + t * (-0.284_496_7 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    let e = (-ax * ax).exp();
    (1.0 - p * e) * if x < 0.0 { -1.0 } else { 1.0 }
}

/// 图确定性回归：同一字节流两次 load 的节点序列必须完全一致
/// （名/类型/输入）。曾在两次 from_memory 间观察到命名错位
/// （"Mul.0" vs "Mul.1"）——优化器若有 HashMap 顺序依赖，两个会话
/// 拿到不同结构的图，一切跨会话对拍都失真。
#[test]
fn graph_load_determinism() {
    let m = std::path::Path::new("../../models/tiny/det.onnx");
    if !m.is_file() {
        eprintln!("[gpu] 无模型，跳过");
        return;
    }
    let bytes = std::fs::read(m).unwrap();
    use qppocr_core::executor::Session;
    let a = Session::from_memory(&bytes, "det.a").unwrap();
    for (i, n) in a.graph.nodes.iter().enumerate().take(6) {
        eprintln!("[det] [{i}] {} {} in={:?}", n.op_type, n.name, n.inputs);
    }
    let b = Session::from_memory(&bytes, "det.b").unwrap();
    assert_eq!(a.graph.nodes.len(), b.graph.nodes.len(), "节点数不同");
    for (i, (x, y)) in a.graph.nodes.iter().zip(b.graph.nodes.iter()).enumerate() {
        assert_eq!(x.op_type, y.op_type, "节点 {i} 类型不同");
        assert_eq!(x.name, y.name, "节点 {i} 名不同: {} vs {}", x.name, y.name);
        assert_eq!(x.inputs, y.inputs, "节点 {i} 输入不同");
    }
    eprintln!("[det] {} 节点两次 load 完全一致", a.graph.nodes.len());
}
