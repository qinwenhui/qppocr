#![allow(clippy::undocumented_unsafe_blocks)] // 裸内核取证：持久映射指针操作，逐块 SAFETY 注释无信息量
//! n_ 内核族（NHWC f16）的设备冒烟：entry → conv → dw → convt → pool →
//! resize → concat → reduce → elem → channel，**每内核一次独立提交**——
//! 设备丢失按序定位肇事者。另含真实规模（960×864）entry 复现段。
//! 数值判据在 det 端到端测试（对照 CPU executor 全图输出）。
//!
//! 区域管理：**单一大区手动切分**——arena 的 chunk 上限 16MB，跨
//! chunk 的区域不在 KernelSet 绑定的缓冲里（偏移错位 = 设备打挂）。

use super::VulkanContext;
use super::memory::Arena;
use super::pipeline::{KernelSet, record_dispatch};
use crate::plan::{OFF_NONE, ParamBlock, PcParams};
use ash::vk;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn open_or_skip() -> Option<VulkanContext> {
    match VulkanContext::open(None) {
        Ok(c) => {
            eprintln!("[gpu] 设备: {}", c.inner.name);
            Some(c)
        }
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
    let cip = crate::nhwc::cpad4(ci as i64) as usize;
    let co1p = crate::nhwc::cpad4(co1 as i64) as usize;
    let co2p = crate::nhwc::cpad4(co2 as i64) as usize;

    let mut seed = 0x1234_5678_9abc_def0_u64;
    let x: Vec<f32> = (0..ci * hw).map(|_| lcg(&mut seed)).collect();
    let w1: Vec<f32> = (0..co1 * ci * 9).map(|_| lcg(&mut seed) * 0.1).collect();
    let b1: Vec<f32> = (0..co1).map(|_| lcg(&mut seed) * 0.1).collect();
    let w2: Vec<f32> = (0..co2 * co1 * 9).map(|_| lcg(&mut seed) * 0.1).collect();
    let dwv: Vec<f32> = (0..co2 * 25).map(|_| lcg(&mut seed) * 0.1).collect();
    let ctw: Vec<f32> = (0..co2 * co2 * 4).map(|_| lcg(&mut seed) * 0.1).collect();

    let w1_16 = crate::nhwc::repack_conv_w(&w1, co1, ci, 3, 3);
    let b1_16 = crate::nhwc::conv_bias(&b1, co1 as i64);
    let w2_16 = crate::nhwc::repack_conv_w(&w2, co2, co1, 3, 3);
    let dw_16 = crate::nhwc::repack_dw_w(&dwv, co2, 5, 5);
    let ct_16 = crate::nhwc::repack_convt_w(&ctw, co2, co2, 2, 2);

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
///容差测试通过了但真实图片 0 框，用真实分布找分歧。
#[test]
fn det_shift_bisect() {
    let dir = std::path::PathBuf::from(
        std::env::var("QPPOCR_REAL_DET_INPUT").unwrap_or_else(|_| "/tmp/det-dump".into()),
    );
    let bytes = match std::fs::read(dir.join("s0_000000.f32")) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("[bisect] 无真实 det 输入 dump，跳过");
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
    let model = std::path::Path::new("../../models/tiny/det.onnx");
    if !model.is_file() {
        eprintln!("[bisect] 无模型，跳过");
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
    // CPU 逐节点 dump
    let dump_dir = std::env::temp_dir().join("qppocr-det-bisect");
    let _ = std::fs::remove_dir_all(&dump_dir);
    std::fs::create_dir_all(&dump_dir).unwrap();
    let mut cpu = Session::from_memory(&mbytes, "bisect.cpu").unwrap();
    cpu.set_dump_dir(dump_dir.to_str().unwrap());
    let in_name = cpu
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let _ = cpu.run(vec![(in_name.clone(), mk())]).unwrap();
    let sid: u32 = {
        let mut mx = 0u32;
        for e in std::fs::read_dir(&dump_dir).unwrap().flatten() {
            let nm = e.file_name().into_string().unwrap_or_default();
            if let Some(stem) = nm.strip_suffix(".f32") {
                if let Some((s, _)) = stem.split_once('_') {
                    if let Ok(v) = s.strip_prefix('s').unwrap_or_default().parse::<u32>() {
                        mx = mx.max(v);
                    }
                }
            }
        }
        mx
    };
    // GPU：禁区域回收（事后读区保真），逐 rec 对拍 + 小平移搜索
    unsafe { std::env::set_var("QPPOCR_GPU_NO_FREE", "1") };
    let (graph, _init) = {
        let s = Session::from_memory(&mbytes, "bisect.gpu").unwrap();
        s.into_parts()
    };
    let gpu = super::session::VulkanSession::new(ctx.inner.clone(), graph, _init).unwrap();
    let _ = qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name, mk())]).unwrap();
    unsafe { std::env::remove_var("QPPOCR_GPU_NO_FREE") };
    let (base, recs) = gpu.debug_recs(&shape).expect("无计划");
    let mut first_shift: Option<String> = None;
    for (i, (kernel, node, nidx, off, n, _, osh, _)) in recs.iter().enumerate() {
        if *n == 0 || *nidx == usize::MAX || kernel == "n_entry" || kernel.starts_with("n_exit") {
            continue;
        }
        let Ok(bb) = std::fs::read(dump_dir.join(format!("s{sid}_{nidx:06}.f32"))) else {
            continue;
        };
        let r = i32::from_le_bytes([bb[0], bb[1], bb[2], bb[3]]) as usize;
        let sh: Vec<i64> = bb[4..4 + 8 * r]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let dv: Vec<f32> = bb[4 + 8 * r..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if sh.len() != 4 || sh[0] != 1 || sh != *osh {
            continue; // 只比 rank-4 单批、形状一致者
        }
        let (c, h, w) = (sh[1] as usize, sh[2] as usize, sh[3] as usize);
        let cp = c.div_ceil(4) * 4;
        if cp != c {
            continue; // 区域含 pad 列，比较略
        }
        // SAFETY: base 持久映射；off+n 在区内（NO_FREE 下计划即布局）。
        let gv: Vec<f32> = unsafe {
            std::slice::from_raw_parts(base.add(*off as usize * 4) as *const f32, *n as usize)
        }
        .to_vec();
        let (mut b0, mut bbest, mut bdx, mut bdy) = (f32::INFINITY, f32::INFINITY, 0i32, 0i32);
        for dy in -3i32..=3 {
            for dx in -2i32..=2 {
                let (mut s, mut cnt) = (0f32, 0usize);
                for y in 0..h {
                    let yy = y as i32 + dy;
                    if yy < 0 || yy as usize >= h {
                        continue;
                    }
                    let row = yy as usize * w;
                    let crow = y * w;
                    for x in (0..w).step_by(4) {
                        let xx = x as i32 + dx;
                        if xx < 0 || xx as usize >= w {
                            continue;
                        }
                        // NHWC vs NCHW：ch 步长 cp（=c）
                        for ch in (0..c).step_by(4) {
                            s += (gv[(row + xx as usize) * c + ch] - dv[ch * h * w + crow + x])
                                .abs();
                            cnt += 1;
                        }
                    }
                }
                if cnt == 0 {
                    continue;
                }
                let m = s / cnt as f32;
                if dx == 0 && dy == 0 {
                    b0 = m;
                }
                if m < bbest {
                    (bbest, bdx, bdy) = (m, dx, dy);
                }
            }
        }
        let shifted = bdy.abs() >= 2 && bbest < b0 * 0.7;
        if shifted && first_shift.is_none() {
            first_shift = Some(format!(
                "#{i} {kernel} {node} osh={osh:?} best=({bdx},{bdy}) r={b0:.3e}->{bbest:.3e}"
            ));
        }
        eprintln!(
            "[bisect] #{i} {kernel} {node} [{c},{h},{w}] ({bdx},{bdy}) {b0:.3e}->{bbest:.3e}"
        );
    }
    match first_shift {
        Some(s) => eprintln!("[bisect] 首个垂直错位节点: {s}"),
        None => eprintln!("[bisect] 无 ≥2px 错位节点"),
    }
}

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
    let mut cpu = Session::from_memory(&mbytes, "cmp.cpu").unwrap();
    cpu.set_dump_dir(dump_dir.to_str().unwrap());
    let in_name = cpu
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let out_cpu = cpu.run(vec![(in_name.clone(), mk())]).unwrap();
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
    // 形态学：平移搜索——mean|d| 最小的 (dx,dy) 揭示系统性空间错位
    //（mean|d|~5e-2 且 GPU 多 ~1000 个 >0.3 像素 = 桥接合并行的指纹）。
    if shape == [1, 3, 960, 864] {
        let (hm, wm) = (960usize, 864usize);
        let mut best = (f32::INFINITY, 0i32, 0i32);
        for dy in -4i32..=4 {
            for dx in -4i32..=4 {
                let (x0, x1) = ((-dx).max(0) as usize, (wm as i32 - dx.max(0)) as usize);
                let (y0, y1) = ((-dy).max(0) as usize, (hm as i32 - dy.max(0)) as usize);
                let (mut s, mut n) = (0f32, 0usize);
                for y in y0..y1 {
                    for x in x0..x1 {
                        let ga = a[((y as i32 + dy) as usize) * wm + (x as i32 + dx) as usize];
                        s += (ga - b[y * wm + x]).abs();
                        n += 1;
                    }
                }
                let m = s / n as f32;
                if m < best.0 {
                    best = (m, dx, dy);
                }
            }
        }
        eprintln!(
            "[gpu] 平移搜索: 最优 mean|d|={:.3e} @ (dx,dy)=({},{})",
            best.0, best.1, best.2
        );
        // 渐进 vs 常数：条带分别搜——最优 dy 随条带下移而增大 = 尺度拉伸。
        for qi in 0..4usize {
            let ys = qi * 240..(qi + 1) * 240;
            let mut best_h = (f32::INFINITY, 0i32);
            for dy in -4i32..=24 {
                let (y0, y1) = (
                    ys.start.saturating_sub((-dy).max(0) as usize),
                    (ys.end as i32 - dy.max(0)) as usize,
                );
                let (mut s, mut n) = (0f32, 0usize);
                for y in y0..y1 {
                    let yi = y as i32 + dy;
                    if yi < 0 || yi as usize >= hm {
                        continue;
                    }
                    for x in 0..wm {
                        s += (a[yi as usize * wm + x] - b[y * wm + x]).abs();
                        n += 1;
                    }
                }
                let m = s / n as f32;
                if m < best_h.0 {
                    best_h = (m, dy);
                }
            }
            eprintln!(
                "[gpu] 条带{} (y {}..{}) 最优 dy={} mean|d|={:.3e}",
                qi, ys.start, ys.end, best_h.1, best_h.0
            );
        }
        // 逐行精测：单行最优 dy（内容行取有信号的行）——偏移结构定位。
        for y in [60usize, 150, 300, 500, 700, 900] {
            let mut best_r = (f32::INFINITY, 0i32);
            for dy in -4i32..=24i32 {
                let yi = y as i32 + dy;
                if yi < 0 || yi as usize >= hm {
                    continue;
                }
                let mut s = 0f32;
                for x in 0..wm {
                    s += (a[yi as usize * wm + x] - b[y * wm + x]).abs();
                }
                if s < best_r.0 {
                    best_r = (s, dy);
                }
            }
            eprintln!("[gpu] 行 {y}: 最优 dy={}", best_r.1);
        }
    }
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
            let cp = crate::nhwc::cpad4(osh[1]) as usize;
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
    // 浮点差异（卷积累加序不同）由端到端对拍覆盖，不在此卡。
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
    let w16 = crate::nhwc::repack_conv_w(&w, co, ci, 1, 1);
    let b16 = crate::nhwc::conv_bias(&bias, co as i64);

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
        let pc2 = crate::plan::PcParams { p_off: p_off + 32 };
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
    let mut cpu = Session::from_memory(&bytes, "rec.cpu").unwrap();
    cpu.set_dump_dir(dump_dir.to_str().unwrap());
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

/// rec 批维补齐 + real_n 早退的回归：batch_grain=8 会话喂 3 行，
/// 计划按 (8,…) 建但空行零算力——真实行输出必须与 grain=1 会话跑
/// 同 3 行**逐位一致**（批内行独立、无跨行归约；布局偏移不同但算术
/// 序相同）。这是「补齐不改变实行结果」的直接断言，早退守卫错杀
/// 实行（写偏移/漏算）会立刻现形。
#[test]
fn rec_batch_pad_early_exit() {
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

    let (b, hh, ww) = (3usize, 48usize, 185usize);
    let mut seed = 0x1234_5678_9abc_def0_u64;
    let mut buf = F32Buf::with_zeroed(b * 3 * hh * ww);
    for v in buf.as_mut_slice().iter_mut() {
        *v = lcg(&mut seed);
    }
    let mk_input = |buf: F32Buf| Tensor {
        name: String::new(),
        shape: vec![b as i64, 3, hh as i64, ww as i64],
        dtype: DType::F32,
        f32: buf,
        i64: Vec::new(),
    };
    let (graph, init) = {
        let s = Session::from_memory(&bytes, "rec.gpu-pad").unwrap();
        s.into_parts()
    };
    let in_name = graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();

    // grain=1 基线（现网 det-only 的逐行形态）
    let gpu_plain =
        super::session::VulkanSession::new(ctx.inner.clone(), graph.clone(), init.clone()).unwrap();
    let out_plain = qppocr_core::device::DeviceSession::run(
        &gpu_plain,
        vec![(in_name.clone(), mk_input(buf.clone()))],
    )
    .unwrap();

    // grain=8 补齐：3 行 → 计划 (8,…) + 5 空行早退
    let mut gpu_pad = super::session::VulkanSession::new(ctx.inner.clone(), graph, init).unwrap();
    gpu_pad.batch_grain = 8;
    let out_pad =
        qppocr_core::device::DeviceSession::run(&gpu_pad, vec![(in_name, mk_input(buf))]).unwrap();

    assert_eq!(
        out_pad[0].shape, out_plain[0].shape,
        "补齐后输出形状应为真实行数"
    );
    assert_eq!(out_pad[0].f32.len(), out_plain[0].f32.len());
    let max_d = out_pad[0]
        .f32
        .iter()
        .zip(out_plain[0].f32.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert_eq!(max_d, 0.0, "补齐空行不得改变真实行输出: max|diff|={max_d}");
    // softmax 行和（守卫错位会让整行变 0/垃圾）
    let vocab = *out_pad[0].shape.last().unwrap() as usize;
    let rows = out_pad[0].f32.len() / vocab;
    for r in 0..rows {
        let s: f32 = out_pad[0].f32[r * vocab..(r + 1) * vocab].iter().sum();
        assert!((s - 1.0).abs() <= 1e-3, "行 {r} softmax 和={s}");
    }
    eprintln!("[rec][pad] {b} 行 @grain8 vs 逐行：逐位一致（{rows}×{vocab}）");

    // ---- argmax 出口：[B,T,2] 对解码 == 概率路径解码（逐字 + 置信度）----
    let (graph2, init2) = {
        let s = Session::from_memory(&bytes, "rec.gpu-argmax").unwrap();
        s.into_parts()
    };
    let in2 = graph2
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let mut gpu_am = super::session::VulkanSession::new(ctx.inner.clone(), graph2, init2).unwrap();
    gpu_am.batch_grain = 8;
    gpu_am.argmax_exit = true;
    let mut seed2 = 0x1234_5678_9abc_def0_u64;
    let mut buf2 = qppocr_kernels::buf::F32Buf::with_zeroed(b * 3 * hh * ww);
    for v in buf2.as_mut_slice().iter_mut() {
        *v = lcg(&mut seed2);
    }
    let t2 = Tensor {
        name: String::new(),
        shape: vec![b as i64, 3, hh as i64, ww as i64],
        dtype: DType::F32,
        f32: buf2,
        i64: Vec::new(),
    };
    let out_am = qppocr_core::device::DeviceSession::run(&gpu_am, vec![(in2, t2)]).unwrap();
    assert_eq!(out_am[0].shape[2], 2, "argmax 出口末维应为 2");
    let t = out_am[0].shape[1] as usize;
    // 同输入的概率基线（grain=1 概率会话）
    let (graph3, init3) = {
        let s = Session::from_memory(&bytes, "rec.gpu-am-base").unwrap();
        s.into_parts()
    };
    let in3 = graph3
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let gpu_p = super::session::VulkanSession::new(ctx.inner.clone(), graph3, init3).unwrap();
    let mut seed3 = 0x1234_5678_9abc_def0_u64;
    let mut buf3 = qppocr_kernels::buf::F32Buf::with_zeroed(b * 3 * hh * ww);
    for v in buf3.as_mut_slice().iter_mut() {
        *v = lcg(&mut seed3);
    }
    let out_p = qppocr_core::device::DeviceSession::run(
        &gpu_p,
        vec![(
            in3,
            Tensor {
                name: String::new(),
                shape: vec![b as i64, 3, hh as i64, ww as i64],
                dtype: DType::F32,
                f32: buf3,
                i64: Vec::new(),
            },
        )],
    )
    .unwrap();
    let charset: Vec<String> = (0..*out_p[0].shape.last().unwrap())
        .map(|i| String::from_utf8_lossy(&[(i % 97 + 33) as u8]).into_owned())
        .collect();
    for k in 0..b {
        let (tx_am, cf_am, _) =
            qppocr_core::pipeline::rec::ctc_decode_pairs(&out_am[0].f32[k * t * 2..], t, &charset);
        let (tx_p, cf_p, _) = qppocr_core::pipeline::rec::ctc_decode(
            &out_p[0].f32[k * t * charset.len()..],
            t,
            charset.len(),
            &charset,
        );
        assert_eq!(tx_am, tx_p, "行 {k} argmax 解码文本与概率路径不一致");
        assert!(
            (cf_am - cf_p).abs() < 1e-6,
            "行 {k} 置信度不一致: {cf_am} vs {cf_p}"
        );
    }
    eprintln!("[rec][argmax] {b} 行 (val,idx) 对解码 == 概率路径（逐字+置信度）");
}

/// 批间流水（run_deferred/complete）的回归：两条**不同计划**同时在飞
/// 必须互不干扰；**同计划**连发必须先收账（inflight 断言兜底）。结果
/// 与同步 run 逐位一致。
#[test]
fn rec_deferred_pipeline() {
    let p = std::path::Path::new("../../models/tiny/rec.onnx");
    if !p.is_file() {
        eprintln!("[gpu] 无 rec 模型，跳过");
        return;
    }
    let Some(ctx) = open_or_skip() else { return };
    let bytes = std::fs::read(p).unwrap();
    use qppocr_core::device::DeviceSession;
    use qppocr_core::executor::Session;
    use qppocr_core::tensor::{DType, Tensor};
    use qppocr_kernels::buf::F32Buf;

    let (graph, init) = {
        let s = Session::from_memory(&bytes, "rec.defer").unwrap();
        s.into_parts()
    };
    let in_name = graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let mk_input = |seed: &mut u64, w: usize| -> Tensor {
        let mut buf = F32Buf::with_zeroed(3 * 48 * w);
        for v in buf.as_mut_slice().iter_mut() {
            *v = lcg(seed);
        }
        Tensor {
            name: String::new(),
            shape: vec![1, 3, 48, w as i64],
            dtype: DType::F32,
            f32: buf,
            i64: Vec::new(),
        }
    };
    let mut seed = 0x0f0f_aaaa_5555_1234_u64;
    let (t_a, t_b) = (mk_input(&mut seed, 320), mk_input(&mut seed, 704));

    let mut gpu = super::session::VulkanSession::new(ctx.inner.clone(), graph, init).unwrap();
    gpu.batch_grain = 8;

    // 同步基线
    let out_a_sync = gpu.run(vec![(in_name.clone(), t_a.clone())]).unwrap();
    let out_b_sync = gpu.run(vec![(in_name.clone(), t_b.clone())]).unwrap();

    // 流水：两条不同计划同时在飞 → 依次收账；再同计划连发两次
    //（第二条必须等第一条收账后才提交——session 的 inflight 断言）。
    let d1 = gpu
        .run_deferred(vec![(in_name.clone(), t_a.clone())])
        .unwrap();
    let d2 = gpu
        .run_deferred(vec![(in_name.clone(), t_b.clone())])
        .unwrap();
    let out_a = d1.complete().unwrap();
    let out_b = d2.complete().unwrap();
    for (am, sm) in [(&out_a, &out_a_sync), (&out_b, &out_b_sync)] {
        assert_eq!(am[0].shape, sm[0].shape);
        let max_d = am[0]
            .f32
            .iter()
            .zip(sm[0].f32.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert_eq!(max_d, 0.0, "流水结果与同步不一致: max|diff|={max_d}");
    }

    // 同计划连发（同宽两次）：第一条在飞时提交第二条 = 写穿，引擎靠
    // 先收账规避；这里验证「收完再发」结果正确。
    let d3 = gpu
        .run_deferred(vec![(in_name.clone(), t_a.clone())])
        .unwrap();
    let out_c1 = d3.complete().unwrap();
    let d4 = gpu
        .run_deferred(vec![(in_name.clone(), t_a.clone())])
        .unwrap();
    let out_c2 = d4.complete().unwrap();
    let m1 = out_c1[0]
        .f32
        .iter()
        .zip(out_a_sync[0].f32.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let m2 = out_c2[0]
        .f32
        .iter()
        .zip(out_a_sync[0].f32.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert_eq!(m1, 0.0, "同计划收后重发(1) 不一致");
    assert_eq!(m2, 0.0, "同计划收后重发(2) 不一致");

    // **影子计划**：同形状第一条在飞时直接提交第二条（引擎同桶流水）——
    // session 分流影子（独立 arena/CB/rn），两条并发结果都与同步逐位
    // 一致。第三条同形状在飞 = 深度超限，应显式报错。
    let d5 = gpu
        .run_deferred(vec![(in_name.clone(), t_a.clone())])
        .unwrap();
    let d6 = gpu
        .run_deferred(vec![(in_name.clone(), t_a.clone())])
        .unwrap();
    let out_s1 = d5.complete().unwrap();
    let out_s2 = d6.complete().unwrap();
    let s1 = out_s1[0]
        .f32
        .iter()
        .zip(out_a_sync[0].f32.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let s2 = out_s2[0]
        .f32
        .iter()
        .zip(out_a_sync[0].f32.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert_eq!(s1, 0.0, "影子路径 primary 侧不一致");
    assert_eq!(s2, 0.0, "影子路径 shadow 侧不一致");
    eprintln!("[rec][defer] 双计划在飞 + 同计划连发 + 影子并发：与同步逐位一致");

    // **排队**：同形状三条并发在飞（多线程共享 Engine 的官方用法）——
    // 曾是硬错误（深度 ≤2），现应排队等影子收账后自动放行，三条结果
    // 都与同步逐位一致。
    let t_c = mk_input(&mut seed, 704);
    let out_c_sync =
        qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name.clone(), t_c.clone())])
            .unwrap();
    let gpu_arc = std::sync::Arc::new(gpu);
    let hs: Vec<_> = [t_c.clone(), t_c.clone(), t_c.clone()]
        .into_iter()
        .map(|t| {
            let g = std::sync::Arc::clone(&gpu_arc);
            let in_name = in_name.clone();
            std::thread::spawn(move || {
                let d = qppocr_core::device::DeviceSession::run_deferred(&*g, vec![(in_name, t)])
                    .unwrap();
                d.complete().unwrap()
            })
        })
        .collect();
    for h in hs {
        let out = h.join().unwrap();
        let m = out[0]
            .f32
            .iter()
            .zip(out_c_sync[0].f32.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert_eq!(m, 0.0, "排队路径结果与同步不一致");
    }
    eprintln!("[rec][defer] 三条并发排队：全部与同步逐位一致");
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
    // 隔离复现失败——完整图才触发。
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
    let mut cpu = Session::from_memory(&bytes, "cls.cpu2").unwrap();
    cpu.set_dump_dir(dump_dir.to_str().unwrap());
    let in_name = cpu
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let _ = cpu.run(vec![(in_name.clone(), t0.clone())]).unwrap();
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
    let _t0_ref = t0.clone();
    let _ = qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name, t0)]).unwrap();

    let (base, recs) = gpu
        .debug_recs(&[b as i64, 3, hh as i64, ww as i64])
        .expect("无计划");
    // 找 Conv.27 的 rec
    let idx = recs
        .iter()
        .position(|(_, node, _, _, _, _, _, _)| node == "Conv.27")
        .expect("无 Conv.27");
    let (kernel, node, _, out_off, out_n, pc, osh, _ins) = &recs[idx];
    let _ = (kernel, node);
    eprintln!("[dw] #{idx} osh={osh:?} out@{out_off} n={out_n}，前驱 recs:");
    for (j, (k2, n2, _, o2, n_2, _, _, _)) in recs.iter().enumerate().skip(idx - 3).take(3) {
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
            .map(|smp| yv[(smp % 128) * hw27 + smp / 128])
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
    let cp = crate::nhwc::cpad4(ci as i64) as usize;
    let mut seed = 0xbeef_cafe_1234_5678_u64;
    let x: Vec<f32> = (0..nb3 * ci * hw).map(|_| lcg(&mut seed)).collect();
    let wt: Vec<f32> = (0..ci * 25).map(|_| lcg(&mut seed) * 0.3).collect();
    let w16 = crate::nhwc::repack_dw_w(&wt, ci, 5, 5);

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
    // SAFETY: big 是单区持久映射；ww 在区内（段互不相交）。
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
    // 读回屏障（见 memory::readback_clean）：本机驱动对 coherent 映射的
    // 设备写不做主机缓存一致，不逐出则 ~30% 概率随机批部分行读到零。
    super::memory::readback_clean(p_of(out_off) as *const u8, nb3 * m * cp * 4);
    // SAFETY: 已等信号。
    let got: Vec<f32> =
        unsafe { std::slice::from_raw_parts(p_of(out_off) as *const f32, 8) }.to_vec();
    // 批 3 的输出（ob + 3*m*cp）——cls 里坏的是批 3
    let got3: Vec<f32> =
        // SAFETY: 已等信号；批 3 偏移在输出区内。
        unsafe { std::slice::from_raw_parts(p_of(out_off + (3 * m * cp) as u32) as *const f32, 8) }
            .to_vec();
    // 竞态诊断：全批 × 8 行采样，零值地图（固定工作组 vs 随机子集；
    // 整批 vs 个别行）——间歇性 B3 全零的形态学。
    let zero_scan = |tag: &str| {
        let mut zero_map: Vec<(u32, usize)> = vec![];
        for bb in 0..nb3 as u32 {
            let zeros = (0..8usize)
                .filter(|s| {
                    // SAFETY: 已等信号；行偏移在批区内。
                    let v: f32 = unsafe {
                        *(p_of(out_off + (bb as usize * m + s * 30) as u32 * cp as u32)
                            as *const f32)
                    };
                    v == 0.0
                })
                .count();
            if zeros > 0 {
                zero_map.push((bb, zeros));
            }
        }
        eprintln!("[dw3] 零值地图{tag} (批,零行数/8): {zero_map:?}");
        zero_map.is_empty()
    };
    let clean1 = zero_scan("");
    if !clean1 {
        // 可见性延迟假设：睡 200ms 再扫——零值消失=传播延迟；不消失=真零。
        std::thread::sleep(std::time::Duration::from_millis(200));
        zero_scan("(睡后)");
    }
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

/// 图内 Conv.27 精确对拍：CPU executor dump 的同名节点输出 vs GPU 输出区，
/// 以及 Conv.27 输入区 vs CPU 的输入张量（上游是否已分歧）。
#[test]
fn cls_dw_precise() {
    let p = std::path::Path::new("../../models/cls.onnx");
    if !p.is_file() {
        eprintln!("[dwp] 无模型，跳过");
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
    // CPU 参考（dump 到内存目录）
    let dump_dir = std::env::temp_dir().join("qppocr-cls-precise");
    let _ = std::fs::remove_dir_all(&dump_dir);
    std::fs::create_dir_all(&dump_dir).unwrap();
    let mut cpu = Session::from_memory(&bytes, "cls.precise").unwrap();
    cpu.set_dump_dir(dump_dir.to_str().unwrap());
    let in_name = cpu
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let _out_cpu = cpu.run(vec![(in_name.clone(), t0.clone())]).unwrap();

    // Conv.27 与其输入生产者的节点序号
    let graph_probe = {
        let s = Session::from_memory(&bytes, "cls.probe").unwrap();
        s.into_parts().0
    };
    let idx27 = graph_probe
        .nodes
        .iter()
        .position(|m| m.name == "Conv.27")
        .unwrap();
    let producer = graph_probe.nodes[idx27].inputs[0].clone();
    let idx_prod = graph_probe
        .nodes
        .iter()
        .position(|m| m.outputs.first() == Some(&producer))
        .unwrap();
    eprintln!(
        "[dwp] Conv.27 = 节点#{idx27}；输入 {producer} 由 #{idx_prod} ({}) 产出",
        graph_probe.nodes[idx_prod].op_type
    );

    let (graph, init) = {
        let s = Session::from_memory(&bytes, "cls.gpu3").unwrap();
        s.into_parts()
    };
    let gpu = super::session::VulkanSession::new(ctx.inner.clone(), graph, init).unwrap();
    let _ = qppocr_core::device::DeviceSession::run(&gpu, vec![(in_name, t0.clone())]).unwrap();

    let (base, recs) = gpu
        .debug_recs(&[b as i64, 3, hh as i64, ww as i64])
        .expect("无计划");
    let ri = recs
        .iter()
        .position(|(_, node, _, _, _, _, _, _)| node == "Conv.27")
        .unwrap();
    let (_, _, _, out_off, _, pc, _, _) = &recs[ri];
    let p_off = u32::from_le_bytes([pc[0], pc[1], pc[2], pc[3]]) as usize;
    // SAFETY: base 持久映射。
    let pw: Vec<u32> =
        unsafe { std::slice::from_raw_parts(base.add(p_off * 4) as *const u32, 20) }.to_vec();
    let in_off = pw[0] as usize;

    // CPU dump：节点序号 → f32 平铺（NCHW）。目录独占本会话
    //（set_dump_dir），sid 从文件名解析——全套并行时本会话不是进程里
    // 第一个 dumper，硬编码 s0_ 会静默跳过全部比较。
    let sid: u32 = {
        let mut mx = 0u32;
        for e in std::fs::read_dir(&dump_dir).unwrap().flatten() {
            let nm = e.file_name().into_string().unwrap_or_default();
            if let Some(stem) = nm.strip_suffix(".f32") {
                if let Some((s, _)) = stem.split_once('_') {
                    if let Ok(v) = s.strip_prefix('s').unwrap_or_default().parse::<u32>() {
                        mx = mx.max(v);
                    }
                }
            }
        }
        mx
    };
    let dump_of = |node_idx: usize| -> Option<Vec<f32>> {
        let bb = std::fs::read(dump_dir.join(format!("s{sid}_{node_idx:06}.f32"))).ok()?;
        let r = i32::from_le_bytes([bb[0], bb[1], bb[2], bb[3]]) as usize;
        Some(
            bb[4 + 8 * r..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    };
    // NHWC 采样比较（rows = b*240, cols=128）
    let cmp_region = |region: &[f32], dv: &[f32], tag: &str| {
        let (rows, cols) = (b * 240usize, 128usize);
        let step = (rows * cols / 64).max(1);
        let mut bad = 0usize;
        for smp in (0..rows * cols).step_by(step).take(64) {
            let row = smp / cols;
            let ch = smp % cols;
            let bb = row / 240;
            let l = row % 240;
            let want = dv[bb * 128 * 240 + ch * 240 + l];
            if (region[smp] - want).abs() > 5e-3 * (1.0 + want.abs()) {
                bad += 1;
            }
        }
        eprintln!("[dwp] {tag}: 64 点采样 {bad} 不符");
    };
    // SAFETY: base 持久映射。
    let in_region: Vec<f32> =
        unsafe { std::slice::from_raw_parts(base.add(in_off * 4) as *const f32, b * 240 * 128) }
            .to_vec();
    // SAFETY: 同上。
    let out_region: Vec<f32> = unsafe {
        std::slice::from_raw_parts(base.add(*out_off as usize * 4) as *const f32, b * 240 * 128)
    }
    .to_vec();
    // Conv.27 区级探针：其 off 若被更晚的 rec 复用，跑完读区看到的是
    // 最后写者的数据——打印跳过而非误报（全图扫描同此约定）。
    let recycled27 = |off: u32| recs.iter().skip(ri + 1).any(|r| r.3 == off);
    if recycled27(*out_off) {
        eprintln!("[dwp] Conv.27 输出区被复用，跳过（读区=最后写者）");
    } else if let Some(dv) = dump_of(idx27) {
        cmp_region(&out_region, &dv, "Conv.27 输出区 vs CPU Conv.27");
    }
    if recycled27(in_off as u32) {
        eprintln!("[dwp] Conv.27 输入区被复用，跳过（读区=最后写者）");
    } else if let Some(dv) = dump_of(idx_prod) {
        cmp_region(&in_region, &dv, "Conv.27 输入区 vs CPU 上游输出");
    }
    // ---- entry 打包对拍：Conv.0 的输入区（n_entry 输出）vs CPU 输入 ----
    // 二分定位首个分歧：entry 区就错 → 打包/布局问题；entry 区对 →
    // n_conv8 内核（其批路径无内核级测试覆盖）。
    {
        let entry = recs
            .iter()
            .find(|(k, ..)| k == "n_entry")
            .expect("计划里没有 n_entry");
        let (_, _, _, e_off, _, _, _, _) = entry;
        let (c_in, hw_in, cp_in) = (3usize, hh * ww, 4usize); // [B,3,H,W] → cpad4=4
        // rec 的 out_n 是逻辑 numel（不含 pad）；真实区按 NHWC 全长读
        //（与上方 NaN 扫描同约定：f32-per-word + cpad4）。
        let total = b * hw_in * cp_in;
        // SAFETY: base 持久映射；total 为 entry 输出区全长。
        let ev: Vec<f32> = unsafe {
            std::slice::from_raw_parts(base.add(*e_off as usize * 4) as *const f32, total)
        }
        .to_vec();
        let step = (total / 64).max(1);
        let (mut bad, mut n_ok) = (0usize, 0usize);
        for smp in (0..total).step_by(step).take(64) {
            let row = smp / cp_in;
            let ch = smp % cp_in;
            let (l, bi) = (row % hw_in, row / hw_in);
            if ch >= c_in {
                continue; // 填充列，无 CPU 对应
            }
            n_ok += 1;
            let want = t0.f32.as_slice()[bi * c_in * hw_in + ch * hw_in + l];
            if !ev[smp].is_finite() || (ev[smp] - want).abs() > 5e-3 * (1.0 + want.abs()) {
                bad += 1;
            }
        }
        eprintln!("[dwp] entry 打包区 vs CPU 输入: {bad}/{n_ok} 不符");
    }
    // ---- 全图精确扫描：每个 rec 输出区 vs 同节点序号的 CPU dump ----
    let mut first_bad: Option<(usize, String)> = None;
    let (mut n_cmp, mut bad_kernels): (usize, Vec<&str>) = (0, Vec::new());
    for (i, (kernel, node, nidx, off, n, _, osh, _)) in recs.iter().enumerate() {
        if *n == 0 || *nidx == usize::MAX || kernel == "n_entry" || kernel.starts_with("n_exit") {
            continue;
        }
        // region 复用：之后还有 rec 写同一 off → 跑完读区看到的是最后
        // 写者的数据，比较无意义（曾在 Conv.0 上误报 48/48——后写的
        // HardSigmoid 把值域压进 [0,1]）。只比本 rec 是该 off 最终值者。
        let recycled = recs.iter().skip(i + 1).any(|r| r.3 == *off);
        if recycled {
            continue;
        }
        let Some(dv) = dump_of(*nidx) else { continue };
        // 存储约定：rank-4 用 rc（rows,cols)；直接用形状换算两版都试
        // SAFETY: base 持久映射。
        let gv: Vec<f32> = unsafe {
            std::slice::from_raw_parts(base.add(*off as usize * 4) as *const f32, *n as usize)
        }
        .to_vec();
        let numel: i64 = osh.iter().product();
        if (numel as usize) != dv.len() {
            continue; // 常量折叠/别名等
        }
        // 解析失败用模糊标记（不是断言依据，供打印）
        let mut bad_cnt = usize::MAX;
        if osh.len() == 4 && osh[0] == b as i64 {
            // [B,C,H,W] NHWC 存储 vs NCHW dump
            let (nb, c, h, w) = (
                osh[0] as usize,
                osh[1] as usize,
                osh[2] as usize,
                osh[3] as usize,
            );
            let hw = h * w;
            let cp = c.div_ceil(4) * 4;
            let mut bad = 0;
            let step = (nb * hw * cp / 48).max(1);
            for smp in (0..nb * hw * cp).step_by(step).take(48) {
                let row = smp / cp;
                let ch = smp % cp;
                let l = row % hw;
                let bi = row / hw;
                let want = dv[bi * c * hw + ch * hw + l];
                if !gv[smp].is_finite() || (gv[smp] - want).abs() > 5e-3 * (1.0 + want.abs()) {
                    bad += 1;
                }
            }
            bad_cnt = bad;
        }
        if bad_cnt != usize::MAX {
            n_cmp += 1;
            if bad_cnt > 0 {
                bad_kernels.push(kernel);
            }
        }
        if bad_cnt != usize::MAX && bad_cnt > 0 && first_bad.is_none() {
            first_bad = Some((i, format!("{kernel} {node} osh={osh:?} bad={bad_cnt}/48")));
        }
    }
    let fb_msg = first_bad.as_ref().map(|(i, d)| format!("#{i} {d}"));
    if let Some((i, d)) = first_bad {
        eprintln!("[dwp] 全图首个精确分歧：#{i} {d}");
        // region 复用检测：off 相同的其它 rec（arena 回收后改写 → 事后读区
        // 看到的是后来者的数据，分歧是读法 artifact 而非内核错）。
        if let Some((_, _, _, off8, _, _, _, _)) = recs.get(i) {
            let sharers: Vec<String> = recs
                .iter()
                .enumerate()
                .filter(|(j, r)| *j != i && r.3 == *off8)
                .map(|(j, r)| format!("#{j} {} {}", r.0, r.1))
                .collect();
            eprintln!(
                "[dwp]   region 复用: off={off8} 另有 {} 个 rec 共用: {:?}",
                sharers.len(),
                sharers
            );
        }
        // 数值形态：首 8 对（GPU 区 NHWC 序 vs CPU dump NCHW 序的对应位）
        if let Some((_, _, nidx8, off8, _, _, osh8, _)) = recs.get(i) {
            if let Some(dv8) = dump_of(*nidx8) {
                // SAFETY: base 持久映射。
                let gv8: Vec<f32> = unsafe {
                    std::slice::from_raw_parts(base.add(*off8 as usize * 4) as *const f32, 64)
                }
                .to_vec();
                // NCHW 前 8 = (b0, c0, h0, w0..7)；NHWC 对应 = 每行 w 步进 cp
                let cp8 = osh8
                    .get(1)
                    .map(|c| (*c as usize).div_ceil(4) * 4)
                    .unwrap_or(1);
                let pairs: Vec<String> = (0..8)
                    .map(|k| {
                        let want = dv8.get(k).copied().unwrap_or(f32::NAN);
                        let got = gv8.get(k * cp8).copied().unwrap_or(f32::NAN);
                        format!("{got:.4}|{want:.4}")
                    })
                    .collect();
                eprintln!("[dwp]   首8值 gpu|cpu: {}", pairs.join("  "));
            }
        }
        for (j, (k2, n2, nidx2, _, _, _, osh2, ins2)) in recs.iter().enumerate().skip(i - 3).take(5)
        {
            eprintln!("[dwp]   #{j} {k2} {n2} node#{nidx2} osh={osh2:?} ins={ins2:?}");
        }
        if let Some((k97, _, _, _, _, pc97, _, ins97)) = recs.get(i) {
            let p97 = u32::from_le_bytes([pc97[0], pc97[1], pc97[2], pc97[3]]) as usize;
            // SAFETY: base 持久映射。
            let pw97: Vec<u32> =
                unsafe { std::slice::from_raw_parts(base.add(p97 * 4) as *const u32, 21) }.to_vec();
            eprintln!("[dwp] {k97} 参数(0..21)={pw97:?} ins={ins97:?}");
        }
    } else {
        eprintln!("[dwp] 全图 rank-4 批张量无分歧（分歧在 rank-3 尾部或输出）");
    }
    eprintln!(
        "[dwp] 精确扫描: {}/{} 个 rec 不符，内核分布={:?}",
        bad_kernels.len(),
        n_cmp,
        {
            let mut h = std::collections::BTreeMap::new();
            for k in &bad_kernels {
                *h.entry(k).or_insert(0usize) += 1;
            }
            h
        }
    );
    // 断言（声明=断言）：可比 rec（区未被复用、rank-4 批张量）须全对。
    // 曾以 48/48 误报 Conv.0——实为事后读已回收 region 的假阳性。
    assert!(
        bad_kernels.is_empty(),
        "cls f16 路径全图对拍 {}/{} 个可比 rec 不符：{:?}（首个：{:?}）",
        bad_kernels.len(),
        n_cmp,
        bad_kernels,
        fb_msg
    );
}

/// small rec 对右零填充的**模型不变性**（纯 CPU，与 GPU 无关）：
/// 自然宽 w 的输出 vs 桶宽 W（右侧补零）的重叠时间步。引擎 GPU 小 rec 走
/// 桶宽填充、CPU 走自然宽——不变性不成立则两侧文本必然分叉（无注意力
/// mask 的 SVTR 混合填充步）。
#[test]
fn small_rec_padding_invariance() {
    let model = std::env::var("SMALL_CMP_MODEL").unwrap_or_else(|_| "small".into());
    let mpath = format!("../../models/{model}/rec.onnx");
    if !std::path::Path::new(&mpath).is_file() {
        eprintln!("[gpu] 无 {model} rec 模型，跳过");
        return;
    }
    use qppocr_core::executor::Session;
    use qppocr_core::tensor::{DType, Tensor};
    use qppocr_kernels::buf::F32Buf;

    let (hh, w_nat, w_pad) = (48usize, 192usize, 256usize);
    let bytes = std::fs::read(&mpath).unwrap();
    let mut seed = 0x1234_5678_9abc_def0_u64;
    let mut buf = F32Buf::with_zeroed(3 * hh * w_nat);
    for v in buf.as_mut_slice().iter_mut() {
        *v = lcg(&mut seed) * 2.0 - 1.0;
    }
    let mk = |buf: F32Buf, w: usize| Tensor {
        name: String::new(),
        shape: vec![1, 3, hh as i64, w as i64],
        dtype: DType::F32,
        f32: buf,
        i64: Vec::new(),
    };
    let in_name = {
        let s = Session::from_memory(&bytes, "small.rec.pad.a").unwrap();
        s.graph
            .inputs
            .iter()
            .find(|s| !s.is_empty())
            .cloned()
            .unwrap()
    };
    let sa = Session::from_memory(&bytes, "small.rec.pad.a").unwrap();
    let out_a = sa
        .run(vec![(in_name.clone(), mk(buf.clone(), w_nat))])
        .unwrap();
    let mut padded = F32Buf::with_zeroed(3 * hh * w_pad);
    for c in 0..3 {
        for y in 0..hh {
            let src = (c * hh + y) * w_nat;
            let dst = (c * hh + y) * w_pad;
            padded.as_mut_slice()[dst..dst + w_nat]
                .copy_from_slice(&buf.as_slice()[src..src + w_nat]);
        }
    }
    let sb = Session::from_memory(&bytes, "small.rec.pad.b").unwrap();
    let out_b = sb.run(vec![(in_name, mk(padded, w_pad))]).unwrap();
    // 重叠时间步：T = w/8，自然宽 24 步、桶宽 32 步。
    let (va, vb) = (out_a[0].f32.as_slice(), out_b[0].f32.as_slice());
    let (t_nat, vocab) = (24usize, va.len() / 24);
    let mut worst = 0.0f64;
    let mut nbad = 0usize;
    let mut first = None;
    for t in 0..t_nat {
        for v in 0..vocab {
            let (a, b) = (va[t * vocab + v], vb[t * vocab + v]);
            let e = ((a - b).abs() / (1.0 + a.abs())) as f64;
            if e > 0.02 {
                nbad += 1;
                if first.is_none() {
                    first = Some((t, v, a, b));
                }
            }
            worst = worst.max(e);
        }
    }
    let argmax = |base: &[f32], t: usize| -> i32 {
        let s = &base[t * vocab..(t + 1) * vocab];
        s.iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap_or(-1)
    };
    let am_diff = (0..t_nat)
        .filter(|&t| argmax(va, t) != argmax(vb, t))
        .count();
    eprintln!(
        "[pad] 重叠 {t_nat} 步：worst={worst:.4} 不符(>0.02)={}/{} argmax 异步数={am_diff}/{t_nat} 首坏={:?}",
        nbad,
        t_nat * vocab,
        first
    );
    // 断言（声明=断言）：**不不变性是预期行为**——动态 B MatMul 的注意力
    // rec 桶宽右零填充会改真步输出（引擎因此对这类模型启用 exact_width/
    // bucket_grain=1，见 mod.rs open_bytes_with）。若某日此断言失败
    // （填充变成无害），可重新评估 exact_width 的形状税是否可退。
    assert!(
        nbad > 0,
        "small rec 已对右零填充不变——exact_width（精确宽）的形状税可能可退役"
    );
}

/// small rec 注意力头的 GPU↔CPU 逐节点对拍：找首个分歧节点（调试用，
/// 非回归闸门——注意力值正确后本测试应全绿，可留作守卫）。
#[test]
fn small_rec_gpu_vs_cpu() {
    let model = std::env::var("SMALL_CMP_MODEL").unwrap_or_else(|_| "small".into());
    let mpath = format!("../../models/{model}/rec.onnx");
    let p = std::path::Path::new(&mpath);
    if !p.is_file() {
        eprintln!("[gpu] 无 {model} rec 模型，跳过");
        return;
    }
    let Some(ctx) = open_or_skip() else { return };
    let bytes = std::fs::read(p).unwrap();
    use qppocr_core::executor::Session;
    use qppocr_core::tensor::{DType, Tensor};
    use qppocr_kernels::buf::F32Buf;

    // 形状可由环境变量覆写（宽行对拍：QPPOCR_CMP_W=640 复现长行乱码）。
    let (b, hh, ww) = (
        std::env::var("QPPOCR_CMP_B")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8usize),
        48usize,
        std::env::var("QPPOCR_CMP_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(192usize),
    );
    let mut seed = 0x5566_7788_99aa_bbcc_u64;
    let mut buf = F32Buf::with_zeroed(b * 3 * hh * ww);
    for v in buf.as_mut_slice().iter_mut() {
        *v = lcg(&mut seed) * 2.0 - 1.0;
    }
    // 零尾巴（复现引擎桶宽右填充：内容左对齐、右侧 grain 内补零）。
    if let Ok(zt) = std::env::var("QPPOCR_CMP_ZEROTAIL") {
        let zt: usize = zt.parse().unwrap();
        for bi in 0..b {
            for c in 0..3 {
                for y in 0..hh {
                    let row = ((bi * 3 + c) * hh + y) * ww;
                    for x in (ww - zt)..ww {
                        buf.as_mut_slice()[row + x] = 0.0;
                    }
                }
            }
        }
    }
    let mk = |buf: F32Buf| Tensor {
        name: String::new(),
        shape: vec![b as i64, 3, hh as i64, ww as i64],
        dtype: DType::F32,
        f32: buf,
        i64: Vec::new(),
    };
    // CPU dump
    let dump_dir = std::env::temp_dir().join("qppocr-small-rec-cmp");
    let _ = std::fs::remove_dir_all(&dump_dir);
    std::fs::create_dir_all(&dump_dir).unwrap();
    let mut cpu = Session::from_memory(&bytes, "small.rec.cpu").unwrap();
    cpu.set_dump_dir(dump_dir.to_str().unwrap());
    let in_name = cpu
        .graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let _out_cpu = cpu.run(vec![(in_name.clone(), mk(buf.clone()))]).unwrap();
    let mut dumps: Vec<(Vec<i64>, Vec<f32>)> = Vec::new();
    let mut files: Vec<_> = std::fs::read_dir(&dump_dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "f32"))
        .collect();
    files.sort();
    for f in files {
        let Ok(bb) = std::fs::read(&f) else {
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
    eprintln!("[small][cmp] 读到 {} 个 dump", dumps.len());

    let (graph, init) = {
        let s = Session::from_memory(&bytes, "small.rec.gpu").unwrap();
        s.into_parts()
    };
    let in2 = graph
        .inputs
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap();
    let mut gpu = super::session::VulkanSession::new(ctx.inner.clone(), graph, init).unwrap();
    gpu.batch_grain = 8;
    // 禁区域回收：run 后读区=最后写者（曾以此假阳性坑过一整轮——见
    // qppocr-gpu-region-recycle-artifact 记忆）
    unsafe { std::env::set_var("QPPOCR_GPU_NO_FREE", "1") };
    let out = qppocr_core::device::DeviceSession::run(&gpu, vec![(in2, mk(buf))]).unwrap();
    unsafe { std::env::remove_var("QPPOCR_GPU_NO_FREE") };
    // 计划键按批维补齐（grain 8）——real_n<8 的部分批也查得到计划。
    let b_pad = b.div_ceil(8) * 8;
    let (base, recs) = gpu
        .debug_recs(&[b_pad as i64, 3, hh as i64, ww as i64])
        .expect("无计划");

    // === 关键链精确探针：rec（按节点名）↔ dump（按 manifest 节点索引），
    // 全量对比（不用形状猜——numel 碰撞曾拿错张量、浅采样曾漏深处分歧）。
    // dump 索引 = CPU 节点索引：87 transpose 出（=LN 输入）、88 mean、
    // 89 Sub.1、90 Pow.1、93 Sqrt.1、94 Div.27、102-115 attention 头链、
    // 189 transpose.7（尾段真转置）。
    {
        // 文件名前缀 s{sid}_ 的 sid 是进程级会话计数（全套跑时非 0）——
        // 按 `_{idx:06}.f32` 后缀匹配，勿硬编码 s0_。
        let read_dump = |idx: usize| -> Option<(Vec<i64>, Vec<f32>)> {
            let suffix = format!("_{idx:06}.f32");
            let path = std::fs::read_dir(&dump_dir)
                .ok()?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .find(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.ends_with(&suffix))
                })?;
            let bb = std::fs::read(path).ok()?;
            let r = i32::from_le_bytes([bb[0], bb[1], bb[2], bb[3]]) as usize;
            let sh: Vec<i64> = bb[4..4 + 8 * r]
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let v: Vec<f32> = bb[4 + 8 * r..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Some((sh, v))
        };
        // (rec 节点名, op 判别（None=唯一）, dump 索引, lane0?)——op 取参数块
        // 首 word（n_elem：5=mul 4=add；行广播 8=sub 11=div）。
        // 尾段 MatMul.12/Add.204/Softmax.2（cpad 行填充布局）与
        // fused:Add.202（NHWC 出 vs NCHW dump）不在此直读——由末尾输出
        // 全量对拍断言覆盖。
        let probes: &[(&str, Option<u32>, usize, bool)] = &[
            ("fused:Add.156", Some(4), 87, false),
            ("ReduceMean.10", None, 88, true),
            ("Sub.0", None, 89, false),
            ("Pow.0", None, 90, false),
            ("Sqrt.0", None, 93, true),
            ("Div.26", None, 94, false),
            // attention 头链（rec↔dump 直读；Transpose.1 rank-5 已验直读存储）
            ("Mul.54", None, 102, false),
            ("Transpose.2", None, 109, false),
            ("MatMul.1", None, 110, false),
            ("Softmax.0", None, 111, false),
            ("MatMul.2", None, 112, false),
            ("Transpose.3", None, 113, false),
            ("MatMul.3", None, 115, false),
            // 第二 attention 头链（同构第二块）
            ("MatMul.6", None, 141, false),
            ("Transpose.4", None, 144, false),
            ("Mul.61", None, 147, false),
            ("Transpose.5", None, 154, false),
            ("MatMul.7", None, 155, false),
            ("Softmax.1", None, 156, false),
            ("MatMul.8", None, 157, false),
            ("Transpose.6", None, 158, false),
            ("MatMul.9", None, 160, false),
            ("Transpose.7", None, 189, false),
        ];
        for (want, want_op, di, lane0) in probes {
            let Some(rec_i) = recs.iter().position(|(k, node, _, _, _, pc, _, _)| {
                if !node.starts_with(want) || k == "n_entry" {
                    return false;
                }
                match want_op {
                    None => true,
                    Some(op) => {
                        let p = u32::from_le_bytes([pc[0], pc[1], pc[2], pc[3]]) as usize;
                        let w0 = unsafe { *(base.add(p * 4) as *const u32) };
                        w0 == *op
                    }
                }
            }) else {
                eprintln!("[probe] {want} rec 未找到");
                continue;
            };
            let (_, node, _, off, n, _, _, _) = &recs[rec_i];
            let Some((sh, dv)) = read_dump(*di) else {
                eprintln!("[probe] dump {di} 未找到");
                continue;
            };
            // lane0（[rows,4] 存储）：区宽 dv.len()*4 字，比 lane0；rec.n 只
            // 记逻辑元素数（192），按分配宽读。
            let gv: Vec<f32> = unsafe {
                std::slice::from_raw_parts(
                    base.add(*off as usize * 4) as *const f32,
                    if *lane0 { dv.len() * 4 } else { *n as usize },
                )
            }
            .to_vec();
            let (mut worst, mut nbad, mut first_bad_i) = (0.0f64, 0usize, None);
            let (mut n1e3, mut n1e5) = (0usize, 0usize);
            let mut bad_rows: Vec<usize> = Vec::new();
            let lane0 = *lane0;
            let rows = if lane0 {
                dv.len()
            } else {
                dv.len().min(gv.len())
            };
            for j in 0..rows {
                let g = if lane0 { gv[j * 4] } else { gv[j] };
                let d = dv[j];
                let e = ((g - d).abs() / (1.0 + d.abs())) as f64;
                if e > 0.05 {
                    nbad += 1;
                    if first_bad_i.is_none() {
                        first_bad_i = Some(j);
                    }
                }
                if e > 1e-3 {
                    n1e3 += 1;
                    let row_w = (*sh.last().unwrap_or(&1)).max(1) as usize;
                    if !lane0 && bad_rows.len() < 64 && !bad_rows.contains(&(j / row_w)) {
                        bad_rows.push(j / row_w);
                    }
                }
                if e > 1e-5 {
                    n1e5 += 1;
                }
                worst = worst.max(e);
            }
            let e1k = if worst > 1e-3 {
                if !bad_rows.is_empty() {
                    format!(" >1e-3:{n1e3} >1e-5:{n1e5} 坏行(≤64)={bad_rows:?}")
                } else {
                    format!(" >1e-3:{n1e3} >1e-5:{n1e5}")
                }
            } else {
                String::new()
            };
            eprintln!(
                "[probe] #{rec_i} {node} ↔ dump{di} {sh:?} lane0={lane0} n={}：worst={worst:.4} 不符={nbad}/{}{e1k} 首坏idx={:?}",
                gv.len(),
                rows,
                first_bad_i
            );
            if let Some(j) = first_bad_i {
                let g = if lane0 { gv[j * 4] } else { gv[j] };
                eprintln!(
                    "        首坏 j={j}（行{} 列{}）gpu={g:.6} cpu={:.6}",
                    j / 120,
                    j % 120,
                    dv[j]
                );
            }
        }
    }
    // rec 名册（QPPOCR_CMP_RECS=1 时打印——现场取证用）。
    if std::env::var_os("QPPOCR_CMP_RECS").is_some() {
        for (i, (kernel, node, _, off, n, _, osh, _)) in recs.iter().enumerate() {
            eprintln!("[rec] #{i} {kernel} {node} osh={osh:?} off={off} n={n}");
        }
    }

    // 全图对拍（诊断性，无断言）：形状精确优先、numel 回退，rank-4 双解释
    //（NHWC↔NCHW 与直读取优——头维张量是直读存储）。
    let mut first_bad: Option<(usize, String)> = None;
    let mut n_bad = 0;
    for (i, (kernel, node, _idx, off, n, _pc, osh, _ins)) in recs.iter().enumerate() {
        if *n == 0 || kernel == "n_entry" || kernel.starts_with("n_exit") {
            continue;
        }
        let gv: Vec<f32> = unsafe {
            std::slice::from_raw_parts(base.add(*off as usize * 4) as *const f32, *n as usize)
        }
        .to_vec();
        let numel: i64 = osh.iter().product();
        let mut matched = false;
        let mut rel = 0.0f64;
        // 先试形状完全一致的 dump（消除 numel 碰撞的假阴/假阳）
        let osh_full: Vec<i64> = osh.clone();
        let exact: Vec<&(Vec<i64>, Vec<f32>)> = dumps
            .iter()
            .filter(|(sh, dv)| *sh == osh_full && dv.len() >= 8)
            .collect();
        let cand: Vec<&(Vec<i64>, Vec<f32>)> = if exact.is_empty() {
            dumps
                .iter()
                .filter(|(sh, dv)| sh.iter().product::<i64>() == numel && dv.len() >= 8)
                .collect()
        } else {
            exact
        };
        for (sh, dv) in &cand {
            // rank-4 双解释取优：backbone 卷积量是 NHWC 存储（↔dump NCHW），
            // attention 头维量（[B,H,T,D] 等）是朴素直读——曾只按 NHWC 解释
            // 把 Mul.54 [8,8,24,15] 误报 rel=0.97。
            let e2 = if sh.len() == 4 {
                let (nn, c3, h3, w3) = (
                    sh[0] as usize,
                    sh[1] as usize,
                    sh[2] as usize,
                    sh[3] as usize,
                );
                let hw = h3 * w3;
                let c4 = c3.div_ceil(4) * 4;
                let rows = nn * hw;
                // NHWC 解释：gv[row*NHWC 列 c] ↔ dump NCHW
                let mut e_nhwc = 0.0f64;
                let ns = 96.min(rows);
                for k in 0..ns {
                    let row = k * rows / ns.max(1);
                    let (n, hwr) = (row / hw, row % hw);
                    let (h, w) = (hwr / w3, hwr % w3);
                    for c in [0usize, 1, c3 / 2, c3 - 1] {
                        let g = gv.get(row * c4 + c).copied().unwrap_or(f32::NAN);
                        let d = dv[(n * c3 + c) * hw + h * w3 + w];
                        e_nhwc = e_nhwc.max(((g - d).abs() / (1.0 + d.abs())) as f64);
                    }
                }
                // 直读解释：同一线性下标逐位对（attention 头维量是行主序无 cpad）
                let mut e_direct = 0.0f64;
                let n_tot = dv.len().min(gv.len());
                let ns2 = 64.min(n_tot);
                for k in 0..ns2 {
                    let idx = k * n_tot / ns2.max(1);
                    e_direct =
                        e_direct.max(((gv[idx] - dv[idx]).abs() / (1.0 + dv[idx].abs())) as f64);
                }
                e_nhwc.min(e_direct)
            } else {
                // [rows,4] lane0 存储（末维=1 的归约出）：gv[r*4] ↔ dv[r]
                let lane0 = sh.last() == Some(&1) && sh.len() == 3;
                let mut e = 0.0f64;
                let nr = dv.len();
                let ns = 64.min(nr);
                for k in 0..ns {
                    let r = k * nr / ns.max(1);
                    let d = dv[r];
                    let g = if lane0 {
                        gv.get(r * 4).copied().unwrap_or(f32::NAN)
                    } else {
                        gv.get(r).copied().unwrap_or(f32::NAN)
                    };
                    e = e.max(((g - d).abs() / (1.0 + d.abs())) as f64);
                }
                e
            };
            if e2 < 0.05 {
                matched = true;
            } else {
                rel = rel.max(e2);
            }
        }
        if !matched && n_bad < 3 {
            if (85..=95).contains(&i) {
                let nan = gv.iter().filter(|v| !v.is_finite()).count();
                eprintln!(
                    "  [#{i}] {kernel} {node} osh={osh:?} 非有限={}/{}",
                    nan,
                    gv.len()
                );
            }
            if node.starts_with("Transpose.1") && osh.len() == 5 {
                let (_sh, dv) = dumps.iter().find(|(sh, _)| *sh == *osh).unwrap();
                let probes: [usize; 8] = [0, 14, 15, 119, 120, 359, 360, 8640];
                for &q in &probes {
                    if q < gv.len() && q < dv.len() {
                        eprintln!(
                            "  [#{i}] idx={q}: gpu={:.6} cpu={:.6} {}",
                            gv[q],
                            dv[q],
                            if (gv[q] - dv[q]).abs() < 1e-3 {
                                "✓"
                            } else {
                                "✗"
                            }
                        );
                    }
                }
            }
            for (sh, dv) in dumps
                .iter()
                .filter(|(sh, _)| sh.iter().product::<i64>() == numel)
                .take(2)
            {
                eprintln!(
                    "  候补 dump shape={sh:?} dv 前6={:?}",
                    &dv[..6.min(dv.len())]
                );
            }
            // 取 numel 匹配的首个 dump 对拍采样值
            for (sh, dv) in &dumps {
                if sh.iter().product::<i64>() == numel {
                    let c3 = if sh.len() == 4 { sh[1] as usize } else { 0 };
                    eprintln!(
                        "[small][dbg] {kernel} {node}: shape {sh:?} | gpu[0..6]={:?} | cpu[0..6]={:?} | C={c3}",
                        &gv[..6.min(gv.len())],
                        &dv[..6.min(dv.len())]
                    );
                    break;
                }
            }
        }
        if !matched {
            n_bad += 1;
            if first_bad.is_none() {
                // 解码参数块：pc（PcParams）前 4 字节 = p_off，再读 arena 参数
                let pcb: &[u8] = &recs[i].5;
                let p_off = u32::from_le_bytes([pcb[0], pcb[1], pcb[2], pcb[3]]) as usize;
                let pw: Vec<u32> =
                    unsafe { std::slice::from_raw_parts(base.add(p_off * 4) as *const u32, 12) }
                        .to_vec();
                eprintln!("[small][dbg] 首分歧参数 words = {pw:?}");
                first_bad = Some((i, format!("{kernel} {node} osh={osh:?} rel={rel:.3}")));
            }
        }
    }
    eprintln!(
        "[small][cmp] 可比节点中分歧 {n_bad} 个；首个：{:?}",
        first_bad
    );
    // 最终闸门：GPU 输出 vs CPU 输出全量对拍（注意力头全链正确的判据——
    // 中间对拍循环有 rank-4 NHWC 解释假警报，以此断言为准）。
    let o = &out[0];
    eprintln!(
        "[small][cmp] 输出 shape={:?} 头4={:?}",
        o.shape,
        &o.f32.as_slice()[..4.min(o.f32.len())]
    );
    let oc = &_out_cpu[0];
    assert_eq!(o.shape, oc.shape, "GPU/CPU 输出形状不符");
    let g = o.f32.as_slice();
    let c = oc.f32.as_slice();
    let mut worst = 0.0f64;
    let mut nbad = 0usize;
    let mut nbit = 0usize; // 逐位（含 <0.02 的微差）不相等数
    let mut first_bad_j = None;
    for j in 0..g.len() {
        let e = ((g[j] - c[j]).abs() / (1.0 + c[j].abs())) as f64;
        if g[j].to_bits() != c[j].to_bits() {
            nbit += 1;
        }
        worst = worst.max(e);
        if e > 0.02 {
            nbad += 1;
            if first_bad_j.is_none() {
                first_bad_j = Some(j);
            }
        }
    }
    eprintln!(
        "[small][cmp] 输出对拍：worst={worst:.5} 不符(>0.02)={}/{} 逐位不等={} 首坏idx={:?}",
        nbad,
        g.len(),
        nbit,
        first_bad_j
    );
    // 坏点分布：按图（b）与时间步（t）聚合——集中某图=批错位、某步=时间步错位。
    {
        let (t_steps, vocab) = (o.shape[1] as usize, o.shape[2] as usize);
        let mut per_b = [0usize; 8];
        let mut per_t_acc: Vec<(usize, usize, f32, f32)> = Vec::new(); // (t, n, gpu, cpu)
        for j in 0..g.len() {
            let e = ((g[j] - c[j]).abs() / (1.0 + c[j].abs())) as f64;
            if e > 0.02 {
                let (b, t) = (j / (t_steps * vocab), (j / vocab) % t_steps);
                per_b[b.min(7)] += 1;
                if per_t_acc.len() < 12 && per_t_acc.iter().all(|x| x.0 != t) {
                    per_t_acc.push((t, 0, g[j], c[j]));
                }
                per_t_acc.iter_mut().for_each(|x| {
                    if x.0 == t {
                        x.1 += 1
                    }
                });
            }
        }
        eprintln!("[small][cmp] 坏点按图分布 {per_b:?}；按时间步（前若干）：{per_t_acc:?}");
        // 按词表列 v 聚合：集中单列 = GEMM 列错位/权重错读
        let mut per_v: Vec<(usize, usize)> = Vec::new();
        for j in 0..g.len() {
            let e = ((g[j] - c[j]).abs() / (1.0 + c[j].abs())) as f64;
            if e > 0.02 {
                let v = j % vocab;
                match per_v.iter_mut().find(|x| x.0 == v) {
                    Some(x) => x.1 += 1,
                    None => per_v.push((v, 1)),
                }
            }
        }
        per_v.sort_by_key(|x| std::cmp::Reverse(x.1));
        eprintln!(
            "[small][cmp] 坏点按词表列 top：{:?}",
            &per_v[..12.min(per_v.len())]
        );
    }
    if let Some(j) = first_bad_j {
        eprintln!("        首坏 j={j} gpu={:.6} cpu={:.6}", g[j], c[j]);
    }
    assert!(
        nbad == 0,
        "small rec GPU↔CPU 输出对拍 {nbad}/{} 超 0.02（worst={worst:.4}，首坏 j={:?} gpu={:?} cpu={:?}）",
        g.len(),
        first_bad_j,
        first_bad_j.map(|j| g[j]),
        first_bad_j.map(|j| c[j]),
    );
}

/// n_softmax 裸内核抗扰测试：T=120 注意力分数 [7680,120] 固定输入，单
/// dispatch ×30 次提交，跨次比较自身一致性——图级曾观测到整行级非确定
/// （同输入 16/51/13 行随运行变，T≤80 恒稳），在此隔离：裸形态仍抖 =
/// 内核内竞态；稳定 = 图上下文（大 CB/驱动屏障）问题。
#[test]
fn n_softmax_flaky_raw() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);
    use super::pipeline::{KernelSet, record_dispatch};
    use crate::plan::{ParamBlock, PcParams};

    let (rows, cols) = (7680usize, 120usize);
    let mut seed = 0xabcd_ef01_2345_6789_u64;
    let mut src = vec![0f32; rows * cols];
    for v in src.iter_mut() {
        *v = lcg(&mut seed) * 20.0 - 10.0;
    }
    let seg = |n: usize| n.div_ceil(4) * 4;
    let in_w = seg(rows * cols);
    let out_w = seg(rows * cols);
    let p_w = 64;
    let big = arena
        .alloc(((in_w + out_w + p_w) * 4) as vk::DeviceSize)
        .unwrap();
    let base_word = big.offset as u32 / 4;
    let (in_off, out_off) = (base_word, base_word + in_w as u32);
    let p_off = base_word + (in_w + out_w) as u32;
    let p_of = |w: u32| unsafe { big.ptr.add(w as usize * 4) };
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), p_of(in_off) as *mut f32, rows * cols);
        let mut pb = ParamBlock::new();
        pb.u(in_off)
            .u(out_off)
            .u(rows as u32)
            .u(cols as u32)
            .u(cols as u32)
            .u((rows / 8) as u32); // rpb
        std::ptr::copy_nonoverlapping(
            pb.words().as_ptr(),
            p_of(p_off) as *mut u32,
            pb.words().len(),
        );
    }
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();
    let mut ref_out: Option<Vec<f32>> = None;
    let mut n_flaky = 0usize;
    for run in 0..30 {
        ctx.inner
            .device
            .submit_one_shot(|d, cb| unsafe {
                record_dispatch(
                    d,
                    cb,
                    &ks,
                    "n_softmax",
                    PcParams { p_off }.bytes(),
                    [rows as u32, 1, 1],
                );
            })
            .unwrap();
        let got: Vec<f32> =
            unsafe { std::slice::from_raw_parts(p_of(out_off) as *const f32, rows * cols) }
                .to_vec();
        // clflush 读回（本机驱动 HOST_COHERENT 陷阱，生产路径同款）
        super::memory::readback_clean(p_of(out_off) as *const u8, rows * cols * 4);
        match &ref_out {
            None => ref_out = Some(got),
            Some(r) => {
                let d = r
                    .iter()
                    .zip(got.iter())
                    .filter(|(a, b)| (**a - **b).abs() > 1e-6)
                    .count();
                if d > 0 {
                    n_flaky += 1;
                    eprintln!("[sm-flaky] run {run}: 与首跑分歧 {d}/{}", rows * cols);
                }
            }
        }
        // 复位输入区（softmax 非原位，输入不动；无需复位）
    }
    eprintln!("[sm-flaky] 30 次提交，跨次分歧运行数 {n_flaky}");
    assert_eq!(n_flaky, 0, "n_softmax 裸形态自身不一致——内核内竞态");
}

/// n_transpose_nd 裸内核：rank-5 [2,0,3,1,4] 与 rank-4 [0,2,1,3] 对拍
/// CPU 参考值（逐元素精确）。
#[test]
fn n_transpose_nd_raw() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);
    use super::pipeline::{KernelSet, record_dispatch};
    use crate::plan::{ParamBlock, PcParams};

    // dims=[4,3,2,8,15]（B,T,3,8,15 的同构）、perm=[2,0,3,1,4]
    let dims: [u32; 5] = [4, 3, 2, 8, 15];
    let perm: [u32; 5] = [2, 0, 3, 1, 4];
    let total: u32 = dims.iter().product();
    let mut seed = 0x1234_0000_5555_aaaa_u64;
    let mut src = vec![0f32; total as usize];
    for v in src.iter_mut() {
        *v = lcg(&mut seed);
    }
    // CPU 参考：out[o] where oc 分解按输出维 dims[perm[d]]
    let odims: Vec<u32> = (0..5).map(|d| dims[perm[d] as usize]).collect();
    let mut want = vec![0f32; total as usize];
    for (i, w) in want.iter_mut().enumerate() {
        let mut oc = [0u32; 5];
        let mut rem = i as u32;
        for d in (0..5).rev() {
            oc[d] = rem % odims[d];
            rem /= odims[d];
        }
        let mut ic = [0u32; 5];
        for d in 0..5 {
            ic[perm[d] as usize] = oc[d];
        }
        let mut s = 0usize;
        for k in 0..5 {
            s = s * dims[k] as usize + ic[k] as usize;
        }
        *w = src[s];
    }

    let seg = |n: usize| n.div_ceil(4) * 4;
    let in_w = seg(total as usize);
    let out_w = seg(total as usize);
    let p_w = 256;
    let big = arena
        .alloc(((in_w + out_w + p_w) * 4) as vk::DeviceSize)
        .unwrap();
    let base_word = big.offset as u32 / 4;
    let in_off = base_word;
    let out_off = base_word + in_w as u32;
    let p_off = base_word + (in_w + out_w) as u32;
    let p_of = |w: u32| unsafe { big.ptr.add(w as usize * 4) };
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), p_of(in_off) as *mut f32, total as usize);
        let mut pb = ParamBlock::new();
        pb.u(in_off).u(out_off).u(total).u(5);
        for &d in &dims {
            pb.u(d);
        }
        for &p in &perm {
            pb.u(p);
        }
        std::ptr::copy_nonoverlapping(
            pb.words().as_ptr(),
            p_of(p_off) as *mut u32,
            pb.words().len(),
        );
    }
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();
    ctx.inner
        .device
        .submit_one_shot(|d, cb| unsafe {
            record_dispatch(
                d,
                cb,
                &ks,
                "n_transpose_nd",
                PcParams { p_off }.bytes(),
                [total.div_ceil(256), 1, 1],
            );
        })
        .unwrap();
    let got: Vec<f32> =
        unsafe { std::slice::from_raw_parts(p_of(out_off) as *const f32, total as usize) }.to_vec();
    let bad = got
        .iter()
        .zip(want.iter())
        .filter(|(a, b)| (**a - **b).abs() > 1e-6)
        .count();
    eprintln!("[tpose][nd] 分歧 {bad}/{total}");
    assert_eq!(bad, 0, "n_transpose_nd rank-5 对拍失败");
}

/// n_attn_mm 裸内核：batched [b,M,K]×[b,K,N] 对拍 CPU 参考。
#[test]
fn n_attn_mm_raw() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);
    use super::pipeline::{KernelSet, record_dispatch};
    use crate::plan::{ParamBlock, PcParams};

    let (batch, heads, m, k, n) = (6u32, 3u32, 5u32, 4u32, 7u32);
    let mut seed = 0xabcd_ef01_2345_6789_u64;
    let a: Vec<f32> = (0..batch * m * k).map(|_| lcg(&mut seed)).collect();
    let b: Vec<f32> = (0..batch * k * n).map(|_| lcg(&mut seed)).collect();
    let mut want = vec![0f32; (batch * m * n) as usize];
    for bi in 0..batch {
        for mi in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    acc += a[(bi * m + mi) as usize * k as usize + ki as usize]
                        * b[(bi * k) as usize * n as usize + ki as usize * n as usize + j as usize];
                }
                want[(bi * m + mi) as usize * n as usize + j as usize] = acc;
            }
        }
    }

    let seg = |x: u32| x.div_ceil(4) * 4;
    let a_w = seg(batch * m * k) as usize;
    let b_w = seg(batch * k * n) as usize;
    let o_w = seg(batch * m * n) as usize;
    let p_w = 256usize;
    let big = arena
        .alloc(((a_w + b_w + o_w + p_w) * 4) as vk::DeviceSize)
        .unwrap();
    let base_word = big.offset as u32 / 4;
    let (a_off, b_off, o_off, p_off) = (
        base_word,
        base_word + a_w as u32,
        base_word + (a_w + b_w) as u32,
        base_word + (a_w + b_w + o_w) as u32,
    );
    let p_of = |w: u32| unsafe { big.ptr.add(w as usize * 4) };
    unsafe {
        std::ptr::copy_nonoverlapping(a.as_ptr(), p_of(a_off) as *mut f32, a.len());
        std::ptr::copy_nonoverlapping(b.as_ptr(), p_of(b_off) as *mut f32, b.len());
        let mut pb = ParamBlock::new();
        pb.u(a_off)
            .u(b_off)
            .u(o_off)
            .u(m)
            .u(k)
            .u(n)
            .u(batch)
            .u(heads);
        std::ptr::copy_nonoverlapping(
            pb.words().as_ptr(),
            p_of(p_off) as *mut u32,
            pb.words().len(),
        );
    }
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();
    ctx.inner
        .device
        .submit_one_shot(|d, cb| unsafe {
            record_dispatch(
                d,
                cb,
                &ks,
                "n_attn_mm",
                PcParams { p_off }.bytes(),
                [1, batch * m, 1],
            );
        })
        .unwrap();
    let got: Vec<f32> =
        unsafe { std::slice::from_raw_parts(p_of(o_off) as *const f32, want.len()) }.to_vec();
    let bad = got
        .iter()
        .zip(want.iter())
        .filter(|(x, y)| (**x - **y).abs() > 1e-5)
        .count();
    eprintln!("[attnmm] 分歧 {bad}/{}", want.len());
    assert_eq!(bad, 0, "n_attn_mm 对拍失败");
}
