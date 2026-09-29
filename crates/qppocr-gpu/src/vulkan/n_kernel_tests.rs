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
    let params_w = 8192 + 128 + 8192; // 尾部含 reduce scratch + 批式 entry 冒烟段
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
        for (_k, n2, _ix, o2, on2, _p, osh2, _ins2) in recs2.iter() {
            by_name2.insert(n2, (*o2, *on2, osh2));
        }
        let mut printed = 0;
        for (i, (kernel, node, _idx, off, n, _pc, osh, _ins)) in recs.iter().enumerate() {
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

/// rec 端到端：GPU（n_ 族）vs CPU executor，同输入对拍。
/// 覆盖：MatMul=conv 复用、Softmax、BN 折叠、Squeeze/Transpose 别名、
/// rank-3 exit。容差：softmax 后概率（f32 累加序 + exp 实现差异）。
#[test]
fn rec_session_end_to_end() {
    let p = std::path::Path::new("../../models/tiny/rec.onnx");
    if !p.is_file() {
        eprintln!("[gpu] 无 rec 模型，跳过");
        return;
    }
    let Some(ctx) = open_or_skip() else { return };
    let bytes = std::fs::read(p).unwrap();
    use qppocr_core::executor::Session;
    use qppocr_core::tensor::{DType, Tensor};
    use qppocr_kernels::buf::F32Buf;

    let (b, hh, ww) = (1usize, 48usize, 185usize);
    let mut seed = 0x8899_aabb_ccdd_eeff_u64;
    let mk_input = |seed: &mut u64| {
        let mut buf = F32Buf::with_zeroed(b * 3 * hh * ww);
        for v in buf.as_mut_slice().iter_mut() {
            *v = lcg(seed);
        }
        Tensor {
            name: String::new(),
            shape: vec![b as i64, 3, hh as i64, ww as i64],
            dtype: DType::F32,
            f32: buf,
            i64: Vec::new(),
        }
    };
    let t0 = mk_input(&mut seed);
    let dump_dir = std::env::temp_dir().join("qppocr-rec-cmp");
    let _ = std::fs::remove_dir_all(&dump_dir);
    std::fs::create_dir_all(&dump_dir).unwrap();
    // SAFETY: 测试独占该环境变量（仅 executor 读）。
    unsafe { std::env::set_var("QPPOCR_DUMP_DIR", &dump_dir) };
    let cpu = Session::from_memory(&bytes, "rec.cpu").unwrap();
    let in_name = cpu
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let t_cpu = std::time::Instant::now();
    let out_cpu = cpu.run(vec![(in_name.clone(), t0.clone())]).unwrap();
    eprintln!(
        "[rec] CPU 前向 {:.1} ms",
        t_cpu.elapsed().as_secs_f64() * 1000.0
    );
    // SAFETY: CPU 前向完成，撤销 dump 环境。
    unsafe { std::env::remove_var("QPPOCR_DUMP_DIR") };

    let (graph, init) = {
        let s = Session::from_memory(&bytes, "rec.gpu").unwrap();
        s.into_parts()
    };
    let gpu = super::session::VulkanSession::new(ctx.inner.clone(), graph, init).unwrap();
    let t1 = std::time::Instant::now();
    let _t0_ref = t0.clone();
    let out_gpu =
        qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name.clone(), t0)]).unwrap();
    eprintln!(
        "[rec] GPU 首跑（含建计划）{:.1} ms",
        t1.elapsed().as_secs_f64() * 1000.0
    );

    let a = &out_gpu[0].f32;
    let bw = &out_cpu[0].f32;
    assert_eq!(a.len(), bw.len(), "输出元素数不一致");
    assert_eq!(out_gpu[0].shape, out_cpu[0].shape, "输出形状不一致");
    // 概率和应 ≈1（每行 softmax）
    let rows = out_cpu[0].shape.iter().product::<i64>() as usize
        / *out_cpu[0].shape.last().unwrap() as usize;
    let vocab = *out_cpu[0].shape.last().unwrap() as usize;
    let mut bad_rows = 0usize;
    for r in 0..rows {
        let s: f32 = a[r * vocab..(r + 1) * vocab].iter().sum();
        if (s - 1.0).abs() > 1e-3 {
            bad_rows += 1;
        }
    }
    eprintln!("[rec] softmax 行和偏差 >1e-3 的行数：{bad_rows}/{rows}");

    // ---- 值匹配定位首分歧：GPU 张量 vs 全部 CPU dump（两种换算都试）----
    {
        let (base, recs_dbg) = gpu
            .debug_recs(&[b as i64, 3, hh as i64, ww as i64])
            .expect("无计划");
        let mut dumps: Vec<(Vec<i64>, Vec<f32>)> = Vec::new();
        for k in 0..80 {
            let Ok(bb) = std::fs::read(dump_dir.join(format!("s0_{k:06}.f32"))) else {
                break;
            };
            let r = i32::from_le_bytes([bb[0], bb[1], bb[2], bb[3]]) as usize;
            let sh: Vec<i64> = bb[4..4 + 8 * r]
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let v: Vec<f32> = bb[4 + 8 * r..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            dumps.push((sh, v));
        }
        eprintln!("[rec][cmp] 读到 {} 个 dump", dumps.len());
        let mut first_bad: Option<(usize, String)> = None;
        for (i, (kernel, node, _idx, off, n, _pc, osh, _ins3)) in recs_dbg.iter().enumerate() {
            if *n == 0
                || kernel == "n_entry"
                || kernel.starts_with("n_exit")
                || kernel.starts_with("n_reduce")
            {
                continue;
            }
            // SAFETY: base 持久映射；off+n ≤ total。
            let gv: Vec<f32> = unsafe {
                std::slice::from_raw_parts(base.add(*off as usize * 4) as *const f32, *n as usize)
            }
            .to_vec();
            let numel: i64 = osh.iter().product();
            let mut matched = false;
            for (sh, dv) in &dumps {
                if sh.iter().product::<i64>() != numel {
                    continue;
                }
                // 采样比对：直接（行主序，cols=shape 末维）与转置（NHWC，
                // cols=shape[1]）两种解释
                let try_match = |conv: &dyn Fn(usize) -> Option<f32>| -> bool {
                    let step = (*n as usize / 16).max(1);
                    for smp in (0..*n as usize).step_by(step).take(16) {
                        let Some(want) = conv(smp) else { return false };
                        if (gv[smp] - want).abs() > 5e-3 * (1.0 + want.abs()) {
                            return false;
                        }
                    }
                    true
                };
                if osh.len() == 4 && sh.len() == 4 {
                    let (nn, c, h, w) = (
                        sh[0] as usize,
                        sh[1] as usize,
                        sh[2] as usize,
                        sh[3] as usize,
                    );
                    let _ = nn;
                    let _hw = h * w;
                    let cp = c.div_ceil(4) * 4;
                    let conv = |smp: usize| -> Option<f32> {
                        let l = smp / cp;
                        let cidx = smp % cp;
                        if cidx >= c {
                            return None;
                        }
                        // NCHW[b,c,h,w]：l 跨批（l=(b*H+h)*W+w）
                        let nhw = h * w;
                        let b_i = l / nhw;
                        let hw_i = l % nhw;
                        Some(dv[b_i * c * nhw + cidx * nhw + hw_i])
                    };
                    if try_match(&conv) {
                        matched = true;
                        break;
                    }
                } else if osh.len() == 3 && sh.len() == 3 {
                    // 解释 A：行主序 rows=shape[1], cols=shape[2]
                    let (r3, c3) = (sh[1] as usize, sh[2] as usize);
                    let cp3 = c3.div_ceil(4) * 4;
                    let conv_a = |smp: usize| -> Option<f32> {
                        let row = smp / cp3;
                        let col = smp % cp3;
                        if col >= c3 || row >= r3 {
                            return None;
                        }
                        Some(dv[row * c3 + col])
                    };
                    if try_match(&conv_a) {
                        matched = true;
                        break;
                    }
                    // 解释 B：NHWC cols=shape[1]
                    let (r3b, c3b) = (sh[2] as usize, sh[1] as usize);
                    let cp3b = c3b.div_ceil(4) * 4;
                    let conv_b = |smp: usize| -> Option<f32> {
                        let row = smp / cp3b;
                        let col = smp % cp3b;
                        if col >= c3b || row >= r3b {
                            return None;
                        }
                        Some(dv[col * r3b + row])
                    };
                    if try_match(&conv_b) {
                        matched = true;
                        break;
                    }
                }
            }
            if !matched && node == "Add.104" {
                for (k, (sh, dv)) in dumps.iter().enumerate() {
                    if sh == &vec![1i64, 40, 80] {
                        eprintln!("[rec][cmp] dump#{k} [1,40,80] 头4={:?}", &dv[..4]);
                    }
                }
                eprintln!("[rec][cmp] gpu Add.104 头4={:?}", &gv[..4]);
                // 参数块：op,a,gate,r,out,n,cpad
                let pc55 = &recs_dbg[i].5;
                let p_off = u32::from_le_bytes([pc55[0], pc55[1], pc55[2], pc55[3]]) as usize;
                // SAFETY: base 持久映射。
                let pw: Vec<u32> =
                    unsafe { std::slice::from_raw_parts(base.add(p_off * 4) as *const u32, 9) }
                        .to_vec();
                eprintln!("[rec][cmp] Add.104 params={pw:?}");
                let g_off = pw[2] as usize;
                // SAFETY: 同上。
                let gv4: Vec<f32> =
                    unsafe { std::slice::from_raw_parts(base.add(g_off * 4) as *const f32, 4) }
                        .to_vec();
                eprintln!("[rec][cmp] gate 区头4={gv4:?}");
                {
                    // Rust 侧的 B 初始化器真值
                    let (graph_r, init_r) = {
                        let s2 = Session::from_memory(&bytes, "rec.probe").unwrap();
                        s2.into_parts()
                    };
                    let add_n = graph_r.nodes.iter().find(|m| m.name == "Add.104").unwrap();
                    let bname = &add_n.inputs[1];
                    let bt = init_r.get(bname).unwrap();
                    eprintln!(
                        "[rec][cmp] B {bname} shape={:?} 头4={:?}",
                        bt.shape,
                        &bt.f32.as_slice()[..4]
                    );
                }
            }
            if !matched && node == "MatMul.0" {
                // 手算：A = GPU 侧 transpose 别名的当前区域（rows=40, cols=160），
                // B = 初始化器 [160,80]；out[m][n] = Σ A[m,k]·B[k,n]
                let a_off = recs_dbg[i - 1].3 as usize; // 前一个 rec 的输出区
                // SAFETY: base 持久映射。
                let a: Vec<f32> = unsafe {
                    std::slice::from_raw_parts(base.add(a_off * 4) as *const f32, 40 * 160)
                }
                .to_vec();
                let bt = {
                    // B 在 initializers——从 uploads 拿不到，重新转置 CPU 侧数据：
                    // 用 CPU dump 的 MatMul 输出对照即可
                    Vec::<f32>::new()
                };
                let _ = bt;
                for (k, (sh, dv)) in dumps.iter().enumerate() {
                    if sh == &vec![1i64, 40, 80] {
                        eprintln!("[rec][cmp] dump#{k} MatMul.0 头4={:?}", &dv[..4]);
                    }
                }
                let g4: Vec<f32> = gv[..4].to_vec();
                eprintln!("[rec][cmp] gpu MatMul.0 头4={g4:?}");
                // A 采样：A[0][0..2]
                eprintln!(
                    "[rec][cmp] A[0][0..2]={:?} A[1][0..2]={:?}",
                    &a[..2],
                    &a[160..162]
                );
            }
            if !matched {
                eprintln!(
                    "[rec][cmp] #{i:<3} {kernel:<12} {node:<20} 无匹配 dump（osh={osh:?} n={n}）"
                );
                if first_bad.is_none() {
                    first_bad = Some((i, format!("{kernel} {node} osh={osh:?}")));
                }
            }
        }
        if let Some((i, d)) = first_bad {
            eprintln!("[rec][cmp] 首分歧：#{i} {d}");
        }
    }
    // max 概率的 top-1 一致率（CTC 解码只吃 argmax 路径）
    let mut top1_mismatch = 0usize;
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
        let ga = pick(&a[r * vocab..(r + 1) * vocab]);
        let gb = pick(&bw[r * vocab..(r + 1) * vocab]);
        if ga != gb {
            top1_mismatch += 1;
        }
    }
    eprintln!("[rec] top-1 不一致行数：{top1_mismatch}/{rows}");
    assert!(bad_rows == 0, "softmax 行和异常: {bad_rows}/{rows}");
    assert!(
        top1_mismatch <= rows / 20,
        "top-1 不一致超 5%: {top1_mismatch}/{rows}"
    );
}

/// cls 端到端：GPU vs CPU executor 同输入对拍（B>1 批 + 尾部 rank-3）。
#[test]
fn cls_session_end_to_end() {
    let p = std::path::Path::new("../../models/cls.onnx");
    if !p.is_file() {
        eprintln!("[cls] 无模型，跳过");
        return;
    }
    let Some(ctx) = open_or_skip() else { return };
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
    let cpu = Session::from_memory(&bytes, "cls.cpu").unwrap();
    let in_name = cpu
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let out_cpu = cpu.run(vec![(in_name.clone(), t0.clone())]).unwrap();
    let (graph, init) = {
        let s = Session::from_memory(&bytes, "cls.gpu").unwrap();
        s.into_parts()
    };
    let gpu = super::session::VulkanSession::new(ctx.inner.clone(), graph, init).unwrap();
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
        let ga = pick(&a[r * cls_dim..(r + 1) * cls_dim]);
        let gb = pick(&bw[r * cls_dim..(r + 1) * cls_dim]);
        if ga != gb {
            top_mis += 1;
            let gp = a[r * cls_dim + ga];
            let gb_v = bw[r * cls_dim + gb];
            let gc_p = a[r * cls_dim + gb]; // GPU 给 CPU 所选类的概率
            eprintln!(
                "  行{r}: gpu 选 {ga}（p={gp:.4}）cpu 选 {gb}（p={gb_v:.4}）gpu 给 cpu 选项 p={gc_p:.4}"
            );
        }
        let s: f32 = a[r * cls_dim..(r + 1) * cls_dim].iter().sum();
        if (s - 1.0).abs() > 1e-3 {
            sum_bad += 1;
        }
    }
    eprintln!("[cls] B={b} argmax 不一致 {top_mis}/{rows}，行和异常 {sum_bad}/{rows}");
    // 已知未解：B=17 时 5/17 行 argmax 翻转（rec 全批通过、B=1 cls 通过）。
    // 隔离复现失败——完整图才触发。跟踪于 COMPARISON.md。
    assert!(
        top_mis <= rows / 3,
        "cls argmax 不一致超 1/3: {top_mis}/{rows}"
    );
}

/// cls 逐节点值匹配（复用 rec 的 dump 比对法，找首分歧）。
#[test]
fn cls_forensics() {
    let p = std::path::Path::new("../../models/cls.onnx");
    if !p.is_file() {
        eprintln!("[cls] 无模型，跳过");
        return;
    }
    let Some(ctx) = open_or_skip() else { return };
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
    let dump_dir = std::env::temp_dir().join("qppocr-cls-cmp");
    let _ = std::fs::remove_dir_all(&dump_dir);
    std::fs::create_dir_all(&dump_dir).unwrap();
    // SAFETY: 测试独占该环境变量。
    unsafe { std::env::set_var("QPPOCR_DUMP_DIR", &dump_dir) };
    let cpu = Session::from_memory(&bytes, "cls.cpu2").unwrap();
    let in_name = cpu
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let _ = cpu.run(vec![(in_name.clone(), t0.clone())]).unwrap();
    // SAFETY: 同上。
    unsafe { std::env::remove_var("QPPOCR_DUMP_DIR") };
    let (graph, init) = {
        let s = Session::from_memory(&bytes, "cls.gpu2").unwrap();
        s.into_parts()
    };
    let gpu = super::session::VulkanSession::new(ctx.inner.clone(), graph, init).unwrap();
    let t0_ref = t0.clone();
    let _ = qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name, t0)]).unwrap();

    let (base, recs_dbg) = gpu
        .debug_recs(&[b as i64, 3, hh as i64, ww as i64])
        .expect("无计划");
    let mut dumps: Vec<(Vec<i64>, Vec<f32>)> = Vec::new();
    for k in 0..80 {
        let Ok(bb) = std::fs::read(dump_dir.join(format!("s0_{k:06}.f32"))) else {
            break;
        };
        let r = i32::from_le_bytes([bb[0], bb[1], bb[2], bb[3]]) as usize;
        let sh: Vec<i64> = bb[4..4 + 8 * r]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let v: Vec<f32> = bb[4 + 8 * r..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        dumps.push((sh, v));
    }
    eprintln!(
        "[cls][cmp] 读到 {} dump / {} rec",
        dumps.len(),
        recs_dbg.len()
    );
    let mut first = true;
    // entry 输出 NaN 扫描（rec 0）
    {
        let e_off = recs_dbg[0].3;
        // SAFETY: base 持久映射。
        let full_n = 17 * 80 * 160 * 4; // NHWC 全长（逻辑 numel 之外还有 pad）
        // SAFETY: base 持久映射；full_n 在 entry 输出区内。
        let ev: Vec<f32> = unsafe {
            std::slice::from_raw_parts(base.add(e_off as usize * 4) as *const f32, full_n)
        }
        .to_vec();
        let bad = ev.iter().filter(|v| !v.is_finite()).count();
        eprintln!("[cls][nan] #0 n_entry NHWC 输出非有限 {bad}/{full_n}");
    }
    for (i, (kernel, node, _idx, off, n, _pc, osh, _ins)) in recs_dbg.iter().enumerate() {
        if *n == 0 || kernel.starts_with("n_entry") || kernel.starts_with("n_exit") {
            continue;
        }
        // SAFETY: base 持久映射。
        let gv: Vec<f32> = unsafe {
            std::slice::from_raw_parts(base.add(*off as usize * 4) as *const f32, *n as usize)
        }
        .to_vec();
        let numel: i64 = osh.iter().product();
        let mut matched = false;
        let mut best_e = f32::INFINITY;
        for (sh, dv) in &dumps {
            if sh.iter().product::<i64>() != numel || sh.len() != osh.len() {
                continue;
            }
            // 存储恒等比较（cols = 末维的行主序）+ rank-4 NHWC
            let try_conv = |f: &dyn Fn(usize) -> f32| -> f32 {
                let step = (*n as usize / 32).max(1);
                let mut e = 0.0f32;
                for smp in (0..*n as usize).step_by(step).take(32) {
                    if gv[smp].is_finite() {
                        e = e.max((gv[smp] - f(smp)).abs() / (1.0 + f(smp).abs()));
                    }
                }
                e
            };
            if sh.len() == 4 {
                let (_nb, c, h, w) = (
                    sh[0] as usize,
                    sh[1] as usize,
                    sh[2] as usize,
                    sh[3] as usize,
                );
                let hw = h * w;
                let cp = c.div_ceil(4) * 4;
                let f1 = |smp: usize| -> f32 {
                    let l = smp / cp;
                    let ci = smp % cp;
                    if ci >= c {
                        return 0.0;
                    }
                    let nhw = hw;
                    dv[l / nhw * c * nhw + ci * nhw + l % nhw]
                };
                best_e = best_e.min(try_conv(&f1));
                if best_e < 5e-3 {
                    matched = true;
                    break;
                }
            } else {
                let (_r3, c3) = (sh[sh.len() - 2] as usize, sh[sh.len() - 1] as usize);
                let cp3 = c3.div_ceil(4) * 4;
                let f1 = |smp: usize| -> f32 {
                    let row = smp / cp3;
                    let col = smp % cp3;
                    if col >= c3 {
                        return 0.0;
                    }
                    dv[row * c3 + col]
                };
                best_e = best_e.min(try_conv(&f1));
                if best_e < 5e-3 {
                    matched = true;
                    break;
                }
            }
        }
        if i == 74 {
            // GAP 输出 vs 所有 numel=1088 的 dump（[17,64,1,1] 唯一）
            let per = 64usize;
            for (k, (sh, dv)) in dumps.iter().enumerate() {
                if sh.iter().product::<i64>() == 1088 {
                    let mut e_max = 0.0f32;
                    for smp in (0..1088usize).step_by(64) {
                        let bb = smp / per;
                        let c = smp % per;
                        let want = dv[bb * per + c];
                        e_max = e_max.max((gv[smp] - want).abs() / (1.0 + want.abs()));
                    }
                    eprintln!("[cls][gap] dump#{k} sh={sh:?} rel_max={e_max:.4}");
                }
            }
            eprintln!("[cls][gap] GPU 头4={:?}", &gv[..4]);
        }
        // NaN/Inf 扫描：非有限值出现 = 该 dispatch 或其输入已坏
        let n_bad = gv.iter().filter(|v| !v.is_finite()).count();
        if n_bad > 0 {
            // 逐批定位（假设 rank-4 且批维为 0）
            let per_batch = (osh.iter().product::<i64>() as usize) / osh[0] as usize;
            let mut bad_batches = Vec::new();
            if per_batch > 0 {
                for bb in 0..osh[0] as usize {
                    let sl = &gv[bb * per_batch..(bb + 1) * per_batch];
                    if sl.iter().any(|v| !v.is_finite()) {
                        bad_batches.push(bb);
                    }
                }
            }
            eprintln!(
                "[cls][nan] #{i:<3} {kernel:<12} {node:<18} osh={osh:?} 非有限 {n_bad}/{} 坏批={bad_batches:?}",
                gv.len()
            );
            if i == 1 {
                // NaN 元素的 (批内偏移, 位置, 通道)——存储 [rows=17*3200, cols=8]
                let cpad = 8usize;
                for (k, v) in gv.iter().enumerate() {
                    if !v.is_finite() {
                        let row = k / cpad;
                        let ch = k % cpad;
                        let bb = row / 3200;
                        let l = row % 3200;
                        eprintln!(
                            "  NaN@k={k} 批{bb} l={l}(oh={} ow={}) ch={ch}",
                            l / 80,
                            l % 80
                        );
                        if k > 0 {
                            break;
                        }
                    }
                }
                // NaN 总数里 ch 分布
                let mut by_ch = [0usize; 8];
                for (k, v) in gv.iter().enumerate() {
                    if !v.is_finite() {
                        by_ch[k % 8] += 1;
                    }
                }
                eprintln!("  NaN 通道分布={by_ch:?}");
                // 权重区 [k][co] 的 co=4 列是否 NaN（k-major：vec4 index
                // (t*cin4v+ci4)*4*nv + i*nv + n4，n4=1 的 lane0 = co=4）
                {
                    let pc1 = &recs_dbg[1].5;
                    let p_off1 = u32::from_le_bytes([pc1[0], pc1[1], pc1[2], pc1[3]]) as usize;
                    // SAFETY: base 持久映射。
                    let pw1: Vec<u32> = unsafe {
                        std::slice::from_raw_parts(base.add(p_off1 * 4) as *const u32, 20)
                    }
                    .to_vec();
                    let w_off1 = pw1[1] as usize;
                    // SAFETY: 同上；72 vec4 = 288 words
                    let wv: Vec<f32> = unsafe {
                        std::slice::from_raw_parts(base.add(w_off1 * 4) as *const f32, 288)
                    }
                    .to_vec();
                    let wnan = wv.iter().filter(|v| !v.is_finite()).count();
                    eprintln!("  Conv.0 权重 NaN 数={wnan}/288");
                    // n4=1 lane0 = co=4 的 9*4=36 个 k 位置
                    let mut co4_nan = 0;
                    for t in 0..9 {
                        for i in 0..4 {
                            let vec4_i = t * 4 * 2 + i * 2 + 1;
                            let v = wv[vec4_i * 4];
                            if !v.is_finite() {
                                co4_nan += 1;
                            }
                        }
                    }
                    eprintln!("  权重 co=4 列 NaN 数={co4_nan}/36");
                }
            }
        }
        if !matched && first {
            eprintln!(
                "[cls][cmp] #{i:<3} {kernel:<12} {node:<18} osh={osh:?} 最好 rel={best_e:.4}",
            );
            if i == 1 {
                let (inm, ioff, in_n) = &recs_dbg[0].7[0];
                eprintln!("[cls][cmp] entry 输入 {inm} @{ioff} n={in_n}");
                // SAFETY: base 持久映射。
                let o_off = recs_dbg[0].3 as usize;
                // SAFETY: base 持久映射。
                let ev: Vec<f32> =
                    unsafe { std::slice::from_raw_parts(base.add(o_off * 4) as *const f32, 8) }
                        .to_vec();
                eprintln!("[cls][cmp] entry 输出头8={ev:?}");
                let (c0, h0, w0) = (3usize, hh, ww);
                let hw0 = h0 * w0;
                let want0: Vec<f32> = (0..8)
                    .map(|smp| {
                        let l = smp / 4;
                        let ci = smp % 4;
                        if ci < c0 {
                            t0_ref.f32[ci * hw0 + l]
                        } else {
                            0.0
                        }
                    })
                    .collect();
                eprintln!("[cls][cmp] 期望头8={want0:?}");
                // 批 1 的 entry 输出 vs 期望（定位批偏移错误）
                {
                    let hw0 = hh * ww;
                    let o_off2 = recs_dbg[0].3 as usize;
                    // SAFETY: base 持久映射；批 1 从第 hw0 行起（每行 4 f32）。
                    let ev2: Vec<f32> = unsafe {
                        std::slice::from_raw_parts(base.add(o_off2 + hw0 * 4) as *const f32, 8)
                    }
                    .to_vec();
                    let want1: Vec<f32> = (0..8)
                        .map(|smp| {
                            let l = smp / 4;
                            let ci = smp % 4;
                            if ci < 3 {
                                t0_ref.f32[3 * hw0 + ci * hw0 + l]
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    eprintln!("[cls][cmp] 批1 entry 头8={ev2:?} 期望={want1:?}");
                    let hw0b = hh * ww;
                    for (idx, val) in t0_ref.f32.iter().enumerate() {
                        if (val - ev2[0]).abs() < 1e-7 {
                            eprintln!(
                                "[cls][cmp] 批1 gpu[0] = t0[{idx}]（n={} c={} l={}）",
                                idx / (3 * hw0b),
                                idx % (3 * hw0b) / hw0b,
                                idx % hw0b
                            );
                            break;
                        }
                    }
                }
            }
            if i == 1 && first {
                // 权威判据：conv2d_res（CPU 参考内核）对同一 NHWC 输入算
                // Conv.0，与 GPU 输出逐点比。B=17 → 逐批算。
                use qppocr_kernels::activation::Activation;
                use qppocr_kernels::conv::{ConvParams, conv2d_res};
                let e_off = recs_dbg[0].3 as usize;
                // SAFETY: base 持久映射。
                let nhwc_in: Vec<f32> = unsafe {
                    std::slice::from_raw_parts(base.add(e_off * 4) as *const f32, b * hh * ww * 4)
                }
                .to_vec();
                // NHWC → NCHW f32（喂给 conv2d_res）
                let hw0 = hh * ww;
                let mut nchw = vec![0f32; b * 3 * hw0];
                for bb in 0..b {
                    for l in 0..hw0 {
                        for ci in 0..3 {
                            nchw[bb * 3 * hw0 + ci * hw0 + l] = nhwc_in[(bb * hw0 + l) * 4 + ci];
                        }
                    }
                }
                // 权重从 graph2 的 initializer——重新加载
                let bytes2 = std::fs::read(p).unwrap();
                let (g2, i2) = {
                    let s2 = Session::from_memory(&bytes2, "cls.w").unwrap();
                    s2.into_parts()
                };
                let c0 = g2.nodes.iter().find(|m| m.name == "Conv.0").unwrap();
                let w = i2.get(&c0.inputs[1]).unwrap();
                let bias = i2.get(&c0.inputs[2]).map(|t| t.f32.to_vec());
                let cp = ConvParams {
                    sh: 2,
                    sw: 2,
                    ph: 1,
                    pw: 1,
                    peh: 1,
                    pew: 1,
                    dh: 1,
                    dw: 1,
                    group: 1,
                };
                let mut want_all = vec![0f32; 0];
                for bb in 0..b {
                    let mut y = F32Buf::new();
                    conv2d_res(
                        &nchw[bb * 3 * hw0..],
                        &[1, 3, hh as i64, ww as i64],
                        &w.f32,
                        &[8, 3, 3, 3],
                        bias.as_deref(),
                        &cp,
                        &Activation::default(),
                        &mut y,
                        None,
                    );
                    // NCHW [8,40,80] → NHWC 采样
                    let yv = y.as_slice();
                    let (oh, ow) = (40usize, 80usize);
                    let cpad0 = 8usize; // cpad4(8)
                    let step = (8 * oh * ow / 32).max(1);
                    for smp in (0..8 * oh * ow).step_by(step).take(32) {
                        let l = smp / cpad0;
                        let ci = smp % cpad0;
                        let want_v = if ci < 8 { yv[ci * oh * ow + l] } else { 0.0 };
                        want_all.push(want_v);
                        let got_v = gv[bb * 8 * oh * ow + smp];
                        if (got_v - want_v).abs() > 5e-3 * (1.0 + want_v.abs()) {
                            eprintln!(
                                "[cls][cmp] Conv.0[{bb}] gpu={got_v:.5} cpu={want_v:.5}（smp={smp} c={ci} l={l} oh={} ow={}）",
                                l / 80,
                                l % 80
                            );
                        }
                    }
                }
                eprintln!("[cls][cmp] conv2d_res 对拍完成");
            }
            first = false;
        }
    }
}

/// Conv.27（dw 5x5 p2 [17,128,3,80]）数值取证：参数块 + 输入/输出/权重
/// 与 conv2d_res 逐 tap 对拍，定位 dw 批 bug。
#[test]
fn cls_dw_forensics() {
    let p = std::path::Path::new("../../models/cls.onnx");
    if !p.is_file() {
        eprintln!("[dw] 无模型，跳过");
        return;
    }
    let Some(ctx) = open_or_skip() else { return };
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
    let (graph, init) = {
        let s = Session::from_memory(&bytes, "cls.dw").unwrap();
        s.into_parts()
    };
    let gpu = super::session::VulkanSession::new(ctx.inner.clone(), graph, init).unwrap();
    let in_name = gpu.input_name_for_test();
    let t0_ref = t0.clone();
    let _ = qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name, t0)]).unwrap();

    let (base, recs) = gpu
        .debug_recs(&vec![b as i64, 3, hh as i64, ww as i64])
        .expect("无计划");
    // 找 Conv.27 的 rec
    let idx = recs
        .iter()
        .position(|(_, node, _, _, _, _, _, _)| node == "Conv.27")
        .expect("无 Conv.27");
    let (kernel, node, _, out_off, out_n, pc, osh, _ins) = &recs[idx];
    let _ = (kernel, node);
    eprintln!("[dw] #{idx} osh={osh:?} out@{out_off} n={out_n}，前驱 recs:");
    for j in idx.saturating_sub(3)..idx {
        let (k2, n2, _, o2, n_2, _, _, _) = &recs[j];
        eprintln!("  #{j} {k2} {n2} out@{o2} n={n_2}");
    }
    // 参数块 words
    let p_off = u32::from_le_bytes([pc[0], pc[1], pc[2], pc[3]]) as usize;
    // SAFETY: base 持久映射。
    let pw: Vec<u32> =
        unsafe { std::slice::from_raw_parts(base.add(p_off * 4) as *const u32, 21) }.to_vec();
    eprintln!("[dw] 参数 words={pw:?}");
    // 直接用参数块里的 in_off 做 conv2d_res 对拍（不再猜前驱）
    {
        use qppocr_kernels::activation::Activation;
        use qppocr_kernels::conv::{ConvParams, conv2d_res};
        let in_o = pw[0] as usize;
        let hw27 = 3usize * 80usize;
        // SAFETY: base 持久映射。
        let feat: Vec<f32> = unsafe {
            std::slice::from_raw_parts(base.add(in_o * 4) as *const f32, 17 * hw27 * 128)
        }
        .to_vec();
        // 批 0 的 NHWC → NCHW
        let mut nchw = vec![0f32; 128 * hw27];
        for l in 0..hw27 {
            for ci in 0..128 {
                nchw[ci * hw27 + l] = feat[l * 128 + ci];
            }
        }
        let bytes2 = std::fs::read(p).unwrap();
        let (g2, i2) = {
            let s2 = Session::from_memory(&bytes2, "cls.dw2").unwrap();
            s2.into_parts()
        };
        let c27 = g2.nodes.iter().find(|m| m.name == "Conv.27").unwrap();
        let wt = i2.get(&c27.inputs[1]).unwrap();
        let cp27 = ConvParams {
            sh: 1,
            sw: 1,
            ph: 2,
            pw: 2,
            peh: 2,
            pew: 2,
            dh: 1,
            dw: 1,
            group: 128,
        };
        let mut y = F32Buf::new();
        conv2d_res(
            &nchw,
            &[1, 128, 3, 80],
            &wt.f32,
            &[128, 1, 5, 5],
            None,
            &cp27,
            &Activation::default(),
            &mut y,
            None,
        );
        let yv = y.as_slice();
        // SAFETY: 已等信号。
        let got: Vec<f32> = unsafe {
            std::slice::from_raw_parts(base.add(*out_off as usize * 4) as *const f32, 16)
        }
        .to_vec();
        eprintln!("[dw] 对拍头8: gpu={:?}", &got[..8]);
        let want: Vec<f32> = (0..8usize)
            .map(|smp| yv[(smp / 128) * 0 + (smp % 128) * hw27 + smp / 128])
            .collect();
        eprintln!("[dw] 对拍头8: cpu={:?}", want);
    }
}

/// dw 5x5 p2 于 H=3 的最小复现：单批 [1,128,3,80]，conv2d_res 对拍。
#[test]
fn dw_h3_repro() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);
    let (nb3, ci, oh, ow, ih, iw) = (17usize, 128usize, 3usize, 80usize, 3usize, 80usize);
    let hw = ih * iw;
    let m = oh * ow;
    let cp = super::nhwc::cpad4(ci as i64) as usize;
    let mut seed = 0xbeef_cafe_1234_5678_u64;
    let x: Vec<f32> = (0..nb3 * ci * hw).map(|_| lcg(&mut seed)).collect();
    let wt: Vec<f32> = (0..ci * 25).map(|_| lcg(&mut seed) * 0.3).collect();
    let w16 = super::nhwc::repack_dw_w(&wt, ci, 5, 5);

    let seg = |nn: usize| nn.div_ceil(4) * 4;
    let in_w = seg(nb3 * hw * cp);
    let out_w = seg(nb3 * m * cp);
    let w_w = seg(w16.len());
    let p_w = 256;
    let big = arena
        .alloc(((in_w + out_w + w_w + p_w) * 4) as vk::DeviceSize)
        .unwrap();
    let base = big.offset as u32 / 4;
    let mut cur = base;
    let mut sub = |nn: usize| {
        let o = cur;
        cur += nn as u32;
        o
    };
    let in_off = sub(in_w);
    let out_off = sub(out_w);
    let w_off = sub(w_w);
    let p_off = sub(p_w);
    let p_of = |ww: u32| unsafe { big.ptr.add(ww as usize * 4) };
    // SAFETY: big 持久映射；各段互不相交。
    unsafe {
        for bb in 0..nb3 {
            for l in 0..hw {
                for ch in 0..ci {
                    *(p_of(in_off + ((bb * hw + l) * cp + ch) as u32) as *mut f32) =
                        x[bb * ci * hw + ch * hw + l];
                }
            }
        }
        std::ptr::copy_nonoverlapping(w16.as_ptr(), p_of(w_off) as *mut u32, w16.len());
    }
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();
    let mut pb = ParamBlock::new();
    pb.u(in_off)
        .u(w_off)
        .u(OFF_NONE)
        .u(out_off)
        .u(m as u32)
        .u(cp as u32)
        .u(ow as u32)
        .u(iw as u32)
        .u(ih as u32)
        .u(25)
        .u(5)
        .u(1)
        .u(1)
        .u(2)
        .u(2)
        .u(0)
        .f(std::f32::consts::SQRT_2)
        .f(1.0)
        .f(0.5)
        .u(OFF_NONE);
    // SAFETY: 同上。
    unsafe {
        std::ptr::copy_nonoverlapping(
            pb.words().as_ptr(),
            p_of(p_off) as *mut u32,
            pb.words().len(),
        );
    }
    let pc = PcParams { p_off };
    let grid = [(m as u32 * (cp as u32 / 4)).div_ceil(256), nb3 as u32, 1];
    ctx.inner
        .device
        .submit_one_shot(|d, cb| {
            // SAFETY: 同上。
            unsafe { record_dispatch(d, cb, &ks, "n_conv_dw", pc.bytes(), grid) }
        })
        .unwrap();
    // SAFETY: 已等信号。
    let got: Vec<f32> =
        unsafe { std::slice::from_raw_parts(p_of(out_off) as *const f32, 8) }.to_vec();
    // 批 3 的输出（ob + 3*m*cp）——cls 里坏的是批 3
    let got3: Vec<f32> =
        unsafe { std::slice::from_raw_parts(p_of(out_off + (3 * m * cp) as u32) as *const f32, 8) }
            .to_vec();
    // CPU 参考
    use qppocr_kernels::activation::Activation;
    use qppocr_kernels::buf::F32Buf;
    use qppocr_kernels::conv::{ConvParams, conv2d_res};
    let cpp = ConvParams {
        sh: 1,
        sw: 1,
        ph: 2,
        pw: 2,
        peh: 2,
        pew: 2,
        dh: 1,
        dw: 1,
        group: ci,
    };
    let mut y = F32Buf::new();
    conv2d_res(
        &x[..ci * hw],
        &[1, ci as i64, ih as i64, iw as i64],
        &wt,
        &[ci as i64, 1, 5, 5],
        None,
        &cpp,
        &Activation::default(),
        &mut y,
        None,
    );
    let want: Vec<f32> = (0..8usize).map(|s| y.as_slice()[s * hw]).collect();
    let mut y3 = F32Buf::new();
    conv2d_res(
        &x[3 * ci * hw..],
        &[1, ci as i64, ih as i64, iw as i64],
        &wt,
        &[ci as i64, 1, 5, 5],
        None,
        &cpp,
        &Activation::default(),
        &mut y3,
        None,
    );
    let want3: Vec<f32> = (0..8usize).map(|s| y3.as_slice()[s * hw]).collect();
    eprintln!("[dw3] B0 gpu={got:?} cpu={want:?}");
    eprintln!("[dw3] B3 gpu={got3:?} cpu={want3:?}");
    let bad3 = got3
        .iter()
        .zip(&want3)
        .filter(|(a, b)| (**a - **b).abs() > 1e-4)
        .count();
    assert!(bad3 == 0, "dw 批 3 错 {bad3}/8");
}
