//! G1 内核的 CPU 判据对拍（conv/convT/pool/reduce/resize/concat）。
//!
//! 容差口径预先标注（与 elementwise 测试同一纪律）：
//! - conv：累加序不同（CPU 按 ch→ky→kx 的 FMA 链，GPU 标量乘加可能被
//!   编译器收缩）——rel 误差 ~1e-5 级，断言 1e-3；
//! - convT：每输出 Σci 项（ci ≤ 32）——1e-4；
//! - pool max / resize / concat：纯比较/搬运——应**逐位一致**；
//! - pool avg / reduce_hw：Σ 项数 ~1e2——1e-4。

use super::VulkanContext;
use super::memory::{Arena, Region};
use super::pipeline::{KernelSet, OFF_NONE, ParamBlock, PcUnaryF, record_dispatch};
use ash::vk;
use qppocr_kernels::activation::Activation;
use qppocr_kernels::buf::F32Buf;
use qppocr_kernels::conv::{ConvParams, conv2d_res, convtranspose2d};
use qppocr_kernels::pool2d::{global_avg_pool, pool2d};
use qppocr_kernels::resize::resize_nearest;
use qppocr_kernels::shape::{Payload, PayloadRef, concat_any};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 // [-1, 1)
}

/// 相对误差（分母 max(1,|b|)），打印与断言共用。
fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
        .fold(0.0f32, f32::max)
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

/// 主机写一段确定性数据。
fn fill(reg: &Region, n: usize, seed: &mut u64) {
    // SAFETY: 持久映射可写；长度 = 分配长度（按 f32 计）。
    unsafe {
        let s = std::slice::from_raw_parts_mut(reg.ptr as *mut f32, n);
        for v in s.iter_mut() {
            *v = lcg(seed);
        }
    }
}

/// 读回（提交并等到信号之后）。
fn read(reg: &Region, n: usize) -> Vec<f32> {
    // SAFETY: 已等信号；长度 = 分配长度。
    unsafe { std::slice::from_raw_parts(reg.ptr as *const f32, n) }.to_vec()
}

fn el(reg: &Region) -> u32 {
    reg.offset as u32 / 4
}

/// 统一收尾：录完 → 提交 → 逐项读回对拍。
struct Check {
    label: String,
    out: Region,
    n: usize,
    want: Vec<f32>,
    tol: f32,
}

fn finish(ctx: &VulkanContext, cb: vk::CommandBuffer, checks: Vec<Check>) {
    ctx.inner.device.end_reusable_cb(cb).unwrap();
    ctx.inner.device.submit_wait_cb(cb).unwrap();
    for c in checks {
        let got = read(&c.out, c.n);
        let e = rel_err(&got, &c.want);
        eprintln!("[gpu] {:<24} rel|diff| = {e:.3e}", c.label);
        assert!(
            e <= c.tol,
            "{} 超容差: {e} > {}（首元素 gpu={} cpu={}）",
            c.label,
            c.tol,
            got.first().map(|v| v.to_string()).unwrap_or_default(),
            c.want.first().map(|v| v.to_string()).unwrap_or_default(),
        );
    }
}

#[test]
fn conv_vs_cpu() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);

    struct Case {
        m: u32,
        ci: u32,
        h: u32,
        w: u32,
        k: u32,
        s: u32,
        pad: u32,
        group: u32,
        act: u32,
        residual: bool,
    }
    let cases = [
        Case {
            m: 24,
            ci: 16,
            h: 16,
            w: 16,
            k: 1,
            s: 1,
            pad: 0,
            group: 1,
            act: 2,
            residual: false,
        },
        Case {
            m: 32,
            ci: 24,
            h: 14,
            w: 14,
            k: 3,
            s: 1,
            pad: 1,
            group: 1,
            act: 1,
            residual: false,
        },
        Case {
            m: 32,
            ci: 32,
            h: 16,
            w: 16,
            k: 3,
            s: 1,
            pad: 1,
            group: 32,
            act: 0,
            residual: false,
        },
        Case {
            m: 20,
            ci: 16,
            h: 17,
            w: 19,
            k: 5,
            s: 2,
            pad: 2,
            group: 1,
            act: 0,
            residual: true,
        },
    ];

    let mut seed = 0x243f_6a88_85a3_08d3_u64;
    // 先按总量开一块（KernelSet 单缓冲绑定），再建管线
    let mut probe = 0usize;
    for c in &cases {
        let oh = (c.h + 2 * c.pad - c.k) / c.s + 1;
        let ow = (c.w + 2 * c.pad - c.k) / c.s + 1;
        probe += (c.ci * c.h * c.w) as usize * 2
            + (c.m * (c.ci / c.group) * c.k * c.k) as usize
            + c.m as usize
            + (c.m * oh * ow) as usize * 2
            + 32;
    }
    arena.alloc(probe as vk::DeviceSize * 4).unwrap();
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();

    let cb = ctx.inner.device.alloc_reusable_cb().unwrap();
    let mut checks: Vec<Check> = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let oh = (c.h + 2 * c.pad - c.k) / c.s + 1;
        let ow = (c.w + 2 * c.pad - c.k) / c.s + 1;
        let x_n = (c.ci * c.h * c.w) as usize;
        let w_n = (c.m * (c.ci / c.group) * c.k * c.k) as usize;
        let out_n = (c.m * oh * ow) as usize;

        let x = arena.alloc(x_n as vk::DeviceSize * 4).unwrap();
        let w = arena.alloc(w_n as vk::DeviceSize * 4).unwrap();
        let b = arena.alloc(c.m as vk::DeviceSize * 4).unwrap();
        let out = arena.alloc(out_n as vk::DeviceSize * 4).unwrap();
        let r = if c.residual {
            Some(arena.alloc(out_n as vk::DeviceSize * 4).unwrap())
        } else {
            None
        };
        let pp = arena.alloc(128).unwrap();

        fill(&x, x_n, &mut seed);
        fill(&w, w_n, &mut seed);
        fill(&b, c.m as usize, &mut seed);
        if let Some(r) = &r {
            fill(r, out_n, &mut seed);
        }
        // conv.comp 参数序：in,w,b,r,out,n,ci,h,w,m,kh,kw,sh,sw,ph,pw,
        // group,act,c1,c2,c3,ohw,ow
        let mut pb = ParamBlock::new();
        pb.u(el(&x))
            .u(el(&w))
            .u(el(&b))
            .u(r.as_ref().map_or(OFF_NONE, el))
            .u(el(&out))
            .u(1)
            .u(c.ci)
            .u(c.h)
            .u(c.w)
            .u(c.m)
            .u(c.k)
            .u(c.k)
            .u(c.s)
            .u(c.s)
            .u(c.pad)
            .u(c.pad)
            .u(c.group)
            .u(c.act)
            .f(std::f32::consts::SQRT_2)
            .f(1.0)
            .f(0.5)
            .u(oh * ow)
            .u(ow);
        let pc = pb.finish(&pp);
        // SAFETY: cb 录制态；PC 与 conv.comp 参数块逐字段对应。
        unsafe {
            record_dispatch(
                &dev,
                cb,
                &ks,
                "conv",
                pc.bytes(),
                [ow / 8 + 1, oh / 8 + 1, c.m],
            );
        }

        // CPU 判据（conv2d_res 内部同序：acc+bias → act → +residual）
        let xv = read(&x, x_n);
        let wv = read(&w, w_n);
        let bv = read(&b, c.m as usize);
        let rv = r.as_ref().map(|r| read(r, out_n));
        let act = match c.act {
            1 => Activation::gelu(std::f32::consts::SQRT_2, 1.0, 0.5),
            2 => Activation::relu(),
            _ => Activation::default(),
        };
        let mut y = F32Buf::new();
        let os = conv2d_res(
            &xv,
            &[1, c.ci as i64, c.h as i64, c.w as i64],
            &wv,
            &[c.m as i64, (c.ci / c.group) as i64, c.k as i64, c.k as i64],
            Some(&bv),
            &ConvParams {
                sh: c.s as usize,
                sw: c.s as usize,
                ph: c.pad as usize,
                pw: c.pad as usize,
                peh: c.pad as usize,
                pew: c.pad as usize,
                dh: 1,
                dw: 1,
                group: c.group as usize,
            },
            &act,
            &mut y,
            rv.as_deref(),
        );
        assert_eq!(os[2], oh as i64, "case {i}");
        assert_eq!(os[3], ow as i64, "case {i}");
        checks.push(Check {
            label: format!(
                "conv#{i}(m{} ci{} k{} s{} g{})",
                c.m, c.ci, c.k, c.s, c.group
            ),
            out,
            n: out_n,
            want: y.as_slice().to_vec(),
            tol: 1e-3,
        });
    }
    finish(&ctx, cb, checks);
}

#[test]
fn convtranspose_vs_cpu() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);
    // in [1,ci,h,w]，权重 [ci,m,2,2]，out [1,m,2h,2w]
    let (ci, m, h, w) = (16u32, 24u32, 8u32, 6u32);
    let x_n = (ci * h * w) as usize;
    let w_n = (ci * m * 4) as usize;
    let out_n = (m * 4 * h * w) as usize;
    arena
        .alloc(((x_n + w_n + out_n) * 4 + 64) as vk::DeviceSize)
        .unwrap();
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();

    let mut seed = 0x1319_8a2e_0370_7344_u64;
    let x = arena.alloc(x_n as vk::DeviceSize * 4).unwrap();
    let wgt = arena.alloc(w_n as vk::DeviceSize * 4).unwrap();
    let out = arena.alloc(out_n as vk::DeviceSize * 4).unwrap();
    let pp = arena.alloc(64).unwrap();
    fill(&x, x_n, &mut seed);
    fill(&wgt, w_n, &mut seed);

    let cb = ctx.inner.device.alloc_reusable_cb().unwrap();
    // convtranspose.comp 参数序：in,w,out,n,ci,h,w,m
    let mut pb = ParamBlock::new();
    pb.u(el(&x))
        .u(el(&wgt))
        .u(el(&out))
        .u(1)
        .u(ci)
        .u(h)
        .u(w)
        .u(m);
    let pc = pb.finish(&pp);
    // 输出网格 [2w, 2h, n*m]（新内核是输出线程累加版）
    // SAFETY: 同 conv。
    unsafe {
        record_dispatch(
            &dev,
            cb,
            &ks,
            "convtranspose",
            pc.bytes(),
            [2 * w / 8 + 1, 2 * h / 8 + 1, m],
        );
    }

    let mut y = F32Buf::new();
    convtranspose2d(
        &read(&x, x_n),
        &[1, ci as i64, h as i64, w as i64],
        &read(&wgt, w_n),
        &[ci as i64, m as i64, 2, 2],
        2,
        2,
        0,
        0,
        &mut y,
    );
    finish(
        &ctx,
        cb,
        vec![Check {
            label: format!("convT(ci{ci} m{m})"),
            out,
            n: out_n,
            want: y.as_slice().to_vec(),
            tol: 1e-4,
        }],
    );
}

#[test]
fn pool_vs_cpu() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);
    // 两个 case：max 3x3 s2 pad1；avg 2x2 s2
    let (n, c, h, w) = (1u32, 12u32, 16u32, 16u32);
    let in_n = (n * c * h * w) as usize;
    let oh_max = (h + 2 - 3) / 2 + 1;
    let ow_max = oh_max;
    let out_max_n = (n * c * oh_max * ow_max) as usize;
    let oh_avg = (h - 2) / 2 + 1;
    let out_avg_n = (n * c * oh_avg * oh_avg) as usize;
    arena
        .alloc(((in_n + out_max_n + out_avg_n) * 4 + 128) as vk::DeviceSize)
        .unwrap();
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();

    let mut seed = 0xa409_3822_299f_31d0_u64;
    let x = arena.alloc(in_n as vk::DeviceSize * 4).unwrap();
    fill(&x, in_n, &mut seed);
    let out_max = arena.alloc(out_max_n as vk::DeviceSize * 4).unwrap();
    let out_avg = arena.alloc(out_avg_n as vk::DeviceSize * 4).unwrap();
    let pp1 = arena.alloc(64).unwrap();
    let pp2 = arena.alloc(64).unwrap();

    let cb = ctx.inner.device.alloc_reusable_cb().unwrap();
    // pool.comp 参数序：in,out,n,c,h,w,kh,kw,sh,sw,ph,pw,is_max,oh,ow
    let mut pb = ParamBlock::new();
    pb.u(el(&x))
        .u(el(&out_max))
        .u(n)
        .u(c)
        .u(h)
        .u(w)
        .u(3)
        .u(3)
        .u(2)
        .u(2)
        .u(1)
        .u(1)
        .u(1)
        .u(oh_max)
        .u(ow_max);
    let pc1 = pb.finish(&pp1);
    // SAFETY: 同 conv。
    unsafe {
        record_dispatch(
            &dev,
            cb,
            &ks,
            "pool",
            pc1.bytes(),
            [ow_max / 8 + 1, oh_max / 8 + 1, n * c],
        );
    }
    let mut pb = ParamBlock::new();
    pb.u(el(&x))
        .u(el(&out_avg))
        .u(n)
        .u(c)
        .u(h)
        .u(w)
        .u(2)
        .u(2)
        .u(2)
        .u(2)
        .u(0)
        .u(0)
        .u(0)
        .u(oh_avg)
        .u(oh_avg);
    let pc2 = pb.finish(&pp2);
    // SAFETY: 同上。
    unsafe {
        record_dispatch(
            &dev,
            cb,
            &ks,
            "pool",
            pc2.bytes(),
            [oh_avg / 8 + 1, oh_avg / 8 + 1, n * c],
        );
    }

    let xv = read(&x, in_n);
    let mut y1 = F32Buf::new();
    let (o1h, o1w) = pool2d(
        &xv, n as usize, c as usize, h as usize, w as usize, 3, 3, 2, 2, 1, 1, 1, 1, true, &mut y1,
    );
    let mut y2 = F32Buf::new();
    let (o2h, o2w) = pool2d(
        &xv, n as usize, c as usize, h as usize, w as usize, 2, 2, 2, 2, 0, 0, 0, 0, false, &mut y2,
    );
    assert_eq!((o1h as u32, o1w as u32), (oh_max, ow_max));
    assert_eq!((o2h as u32, o2w as u32), (oh_avg, oh_avg));
    finish(
        &ctx,
        cb,
        vec![
            Check {
                label: "pool max3x3s2p1".into(),
                out: out_max,
                n: out_max_n,
                want: y1.as_slice().to_vec(),
                tol: 0.0, // 纯比较：逐位一致
            },
            Check {
                label: "pool avg2x2s2".into(),
                out: out_avg,
                n: out_avg_n,
                want: y2.as_slice().to_vec(),
                tol: 1e-4,
            },
        ],
    );
}

#[test]
fn reduce_resize_concat_vs_cpu() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);
    let (n, c, h, w) = (1u32, 8u32, 12u32, 10u32);
    let in_n = (n * c * h * w) as usize;
    let (oh, ow) = (h * 2, w * 2);
    let up_n = (n * c * oh * ow) as usize;
    // concat：两个输入 [1,4,h,w] + [1,6,h,w] → [1,10,h,w]
    let c1n = (4 * h * w) as usize;
    let c2n = (6 * h * w) as usize;
    let cat_n = (10 * h * w) as usize;
    arena
        .alloc(
            ((in_n + up_n + c1n + c2n + cat_n + n as usize * c as usize) * 4 + 256)
                as vk::DeviceSize,
        )
        .unwrap();
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();

    let mut seed = 0x082e_fa98_ec4e_6c89_u64;
    let x = arena.alloc(in_n as vk::DeviceSize * 4).unwrap();
    fill(&x, in_n, &mut seed);
    let red = arena
        .alloc(n as vk::DeviceSize * c as usize as vk::DeviceSize * 4)
        .unwrap();
    let up = arena.alloc(up_n as vk::DeviceSize * 4).unwrap();
    let ca = arena.alloc(c1n as vk::DeviceSize * 4).unwrap();
    let cb_in = arena.alloc(c2n as vk::DeviceSize * 4).unwrap();
    fill(&ca, c1n, &mut seed);
    fill(&cb_in, c2n, &mut seed);
    let cat = arena.alloc(cat_n as vk::DeviceSize * 4).unwrap();
    let p1 = arena.alloc(64).unwrap();
    let p2 = arena.alloc(64).unwrap();
    let p3 = arena.alloc(64).unwrap();

    let cb = ctx.inner.device.alloc_reusable_cb().unwrap();
    // reduce_hw.comp：in,out,n,c,h,w
    let mut pb = ParamBlock::new();
    pb.u(el(&x)).u(el(&red)).u(n).u(c).u(h).u(w);
    let pc1 = pb.finish(&p1);
    // SAFETY: 同 conv。
    unsafe {
        record_dispatch(
            &dev,
            cb,
            &ks,
            "reduce_hw",
            pc1.bytes(),
            [(n * c).div_ceil(256), 1, 1],
        );
    }
    // resize_nearest.comp：in,out,n,c,h,w,oh,ow
    let mut pb = ParamBlock::new();
    pb.u(el(&x)).u(el(&up)).u(n).u(c).u(h).u(w).u(oh).u(ow);
    let pc2 = pb.finish(&p2);
    // SAFETY: 同上。
    unsafe {
        record_dispatch(
            &dev,
            cb,
            &ks,
            "resize_nearest",
            pc2.bytes(),
            [ow / 8 + 1, oh / 8 + 1, n * c],
        );
    }
    // concat_c.comp：out,n,h,w,n_in,c_total,0,0, 然后 {src,c_len}×2
    let mut pb = ParamBlock::new();
    pb.u(el(&cat))
        .u(n)
        .u(h)
        .u(w)
        .u(2)
        .u(10)
        .u(0)
        .u(0)
        .u(el(&ca))
        .u(4)
        .u(el(&cb_in))
        .u(6);
    let pc3 = pb.finish(&p3);
    // SAFETY: 同上。
    unsafe {
        record_dispatch(
            &dev,
            cb,
            &ks,
            "concat_c",
            pc3.bytes(),
            [w / 8 + 1, h / 8 + 1, n * 10],
        );
    }

    let xv = read(&x, in_n);
    let mut y_red = F32Buf::new();
    global_avg_pool(&xv, n as usize, c as usize, &mut y_red);
    let mut y_up = F32Buf::new();
    resize_nearest(
        &xv,
        n as usize,
        c as usize,
        h as usize,
        w as usize,
        oh as usize,
        ow as usize,
        &mut y_up,
    );
    let mut y_cat = Payload::F32(F32Buf::new());
    let shapes = [[1i64, 4, h as i64, w as i64], [1, 6, h as i64, w as i64]];
    let cat_shape = concat_any(
        &[
            PayloadRef::F32(&read(&ca, c1n)),
            PayloadRef::F32(&read(&cb_in, c2n)),
        ],
        &[&shapes[0], &shapes[1]],
        1,
        &mut y_cat,
    );
    assert_eq!(cat_shape, vec![1, 10, h as i64, w as i64]);
    let y_cat = match y_cat {
        Payload::F32(v) => v,
        Payload::I64(_) => unreachable!(),
    };
    finish(
        &ctx,
        cb,
        vec![
            Check {
                label: "reduce_hw".into(),
                out: red,
                n: (n * c) as usize,
                want: y_red.as_slice().to_vec(),
                tol: 1e-4,
            },
            Check {
                label: "resize_nearest".into(),
                out: up,
                n: up_n,
                want: y_up.as_slice().to_vec(),
                tol: 0.0, // 纯索引：逐位一致
            },
            Check {
                label: "concat_c".into(),
                out: cat,
                n: cat_n,
                want: y_cat.as_slice().to_vec(),
                tol: 0.0, // 纯搬运：逐位一致
            },
        ],
    );
}

#[test]
fn hardsigmoid_then_conv_min_repro() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);
    // 小张量：hardsigmoid 32 元素 + 一个 1x1 conv（镜像 Conv.8 的形态）
    let (ci, m, h, w) = (32u32, 64u32, 8u32, 8u32);
    let x_n = (ci * h * w) as usize;
    let w_n = (m * ci) as usize;
    let out_n = (m * h * w) as usize;
    arena
        .alloc(((x_n + w_n + m as usize + out_n + 32 + 64) * 4) as vk::DeviceSize)
        .unwrap();
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();

    let mut seed = 0xdead_beef_cafe_1234_u64;
    let x = arena.alloc(x_n as vk::DeviceSize * 4).unwrap();
    let wgt = arena.alloc(w_n as vk::DeviceSize * 4).unwrap();
    let bias = arena.alloc(m as vk::DeviceSize * 4).unwrap();
    let out = arena.alloc(out_n as vk::DeviceSize * 4).unwrap();
    let gate = arena.alloc(128).unwrap();
    let pp = arena.alloc(128).unwrap();
    fill(&x, x_n, &mut seed);
    fill(&wgt, w_n, &mut seed);
    fill(&bias, m as usize, &mut seed);
    fill(&gate, 32, &mut seed);

    let cb = ctx.inner.device.alloc_reusable_cb().unwrap();
    // 1) hardsigmoid（PcUnaryF，20 字节 PC——和会话里 #11 同款）
    let pc1 = PcUnaryF {
        in_off: el(&gate),
        out_off: el(&gate),
        n: 32,
        p1: 0.2,
        p2: 0.5,
    };
    // SAFETY: 同 elementwise 测试。
    unsafe {
        record_dispatch(&dev, cb, &ks, "hardsigmoid", pc1.bytes(), [1, 1, 1]);
    }
    // 2) conv（PcParams 参数块——和会话里 #13 同款，act=1 gelu）
    let mut pb = ParamBlock::new();
    pb.u(el(&x))
        .u(el(&wgt))
        .u(el(&bias))
        .u(OFF_NONE)
        .u(el(&out))
        .u(1)
        .u(ci)
        .u(h)
        .u(w)
        .u(m)
        .u(1)
        .u(1)
        .u(1)
        .u(1)
        .u(0)
        .u(0)
        .u(1)
        .u(1)
        .f(std::f32::consts::SQRT_2)
        .f(1.0)
        .f(0.5)
        .u(h * w)
        .u(w);
    let pc2 = pb.finish(&pp);
    // SAFETY: 同上。
    unsafe {
        record_dispatch(
            &dev,
            cb,
            &ks,
            "conv",
            pc2.bytes(),
            [w / 8 + 1, h / 8 + 1, m],
        );
    }
    ctx.inner.device.end_reusable_cb(cb).unwrap();
    ctx.inner.device.submit_wait_cb(cb).unwrap();
    eprintln!("[gpu] hardsigmoid→conv 序列通过（该形态未复现设备丢失）");
}

#[test]
fn conv_gemm_vs_cpu() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);

    // 1x1 conv：[1,ci,h,w] × [m,ci,1,1] → [1,m,h,w]
    let (ci, m, h, w) = (32u32, 24u32, 16u32, 12u32);
    let x_n = (ci * h * w) as usize;
    let w_n = (m * ci) as usize;
    let out_n = (m * h * w) as usize;
    arena
        .alloc(((x_n + w_n + m as usize + out_n) * 4 + 128) as vk::DeviceSize)
        .unwrap();
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();

    let mut seed = 0x4528_21e6_38d0_1377_u64;
    let x = arena.alloc(x_n as vk::DeviceSize * 4).unwrap();
    let wgt = arena.alloc(w_n as vk::DeviceSize * 4).unwrap();
    let bias = arena.alloc(m as vk::DeviceSize * 4).unwrap();
    let out = arena.alloc(out_n as vk::DeviceSize * 4).unwrap();
    let pp = arena.alloc(128).unwrap();
    fill(&x, x_n, &mut seed);
    fill(&wgt, w_n, &mut seed);
    fill(&bias, m as usize, &mut seed);

    let cb = ctx.inner.device.alloc_reusable_cb().unwrap();
    // conv_gemm 参数序（与 conv.comp 相同布局，1x1 专用的字段含义）：
    // in,w,b,r,out, n,ci,h,w, m,kh,kw,sh,sw,ph,pw, group,act, c1,c2,c3, hw,ow
    let mut pb = ParamBlock::new();
    pb.u(el(&x))
        .u(el(&wgt))
        .u(el(&bias))
        .u(OFF_NONE)
        .u(el(&out))
        .u(1) // n（占位）
        .u(ci) // ci = K
        .u(h)
        .u(w) // h, w（占位）
        .u(m) // m = M
        .u(1)
        .u(1) // kh, kw
        .u(1)
        .u(1) // sh, sw
        .u(0)
        .u(0) // ph, pw
        .u(1) // group
        .u(0) // act = none
        .f(std::f32::consts::SQRT_2)
        .f(1.0)
        .f(0.5)
        .u(h * w) // hw = N 维
        .u(0); // ow（不用）
    let pc = pb.finish(&pp);
    // SAFETY: cb 录制态；PC 与 conv_gemm.comp 参数块对应。
    unsafe {
        record_dispatch(
            &dev,
            cb,
            &ks,
            "conv_gemm",
            pc.bytes(),
            [h * w / 16 + 1, m / 16 + 1, 1],
        );
    }
    ctx.inner.device.end_reusable_cb(cb).unwrap();
    ctx.inner.device.submit_wait_cb(cb).unwrap();

    // CPU 判据
    let xv = read(&x, x_n);
    let wv = read(&wgt, w_n);
    let bv = read(&bias, m as usize);
    let mut want = vec![0f32; out_n];
    for mi in 0..m as usize {
        for hw in 0..(h * w) as usize {
            let mut acc = 0f32;
            for c in 0..ci as usize {
                acc += xv[c * (h * w) as usize + hw] * wv[mi * ci as usize + c];
            }
            want[mi * (h * w) as usize + hw] = acc + bv[mi];
        }
    }
    let got = read(&out, out_n);
    let e = rel_err(&got, &want);
    eprintln!("[gpu] conv_gemm(ci{ci} m{m} {h}x{w}) rel|diff| = {e:.3e}");
    eprintln!("[gpu]   got[0..4]  = {:?}", &got[..4.min(got.len())]);
    eprintln!("[gpu]   want[0..4] = {:?}", &want[..4.min(want.len())]);
    // 找最差元素的位置
    let (mut wi, mut wd) = (0usize, 0f32);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let d = (g - w).abs() / (1.0 + w.abs());
        if d > wd {
            wd = d;
            wi = i;
        }
    }
    let m_idx = wi / (h * w) as usize;
    let hw_idx = wi % (h * w) as usize;
    eprintln!(
        "[gpu]   最差 [{wi}] (m={m_idx} hw={hw_idx}): gpu={:.6} want={:.6} diff={:.6}",
        got[wi],
        want[wi],
        (got[wi] - want[wi]).abs()
    );
    // 打印该 M 行的首尾
    let row = m_idx * (h * w) as usize;
    eprintln!(
        "[gpu]   m={m_idx} 行首: got={:.4} want={:.4}",
        got[row], want[row]
    );
    eprintln!(
        "[gpu]   m={m_idx} 行尾: got={:.4} want={:.4}",
        got[row + (h * w) as usize - 1],
        want[row + (h * w) as usize - 1]
    );
    assert!(e < 1e-3, "conv_gemm 超容差: {e}");
}

/// 最小共享内存测试：16 线程各自写入 shared，barrier 后读邻居。
/// 如果这个都不过，说明 naga + patch 的共享内存基础就坏了。
#[test]
fn shared_memory_basic() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);
    // ★ 内核读写 SSBO 偏移 0（data[0..31]）——必须用第一个分配（offset=0）
    let probe = arena.alloc(256).unwrap();
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();

    // _shared_test 读 data[0..16] 写 data[16..32]（不走 PC）
    // SAFETY: probe 在偏移 0；前 32 float 在块内。
    unsafe {
        let base = &probe;
        let s = std::slice::from_raw_parts_mut(base.ptr as *mut f32, 16);
        for (i, v) in s.iter_mut().enumerate() {
            *v = i as f32 * 1.5;
        }
        let o = std::slice::from_raw_parts_mut(base.ptr.add(64) as *mut f32, 16);
        o.fill(-999.0);

        let cb = ctx.inner.device.alloc_reusable_cb().unwrap();
        let pc = super::pipeline::PcParams { p_off: 0 };
        record_dispatch(&dev, cb, &ks, "_shared_test", pc.bytes(), [1, 1, 1]);
        ctx.inner.device.end_reusable_cb(cb).unwrap();
        ctx.inner.device.submit_wait_cb(cb).unwrap();

        // 验证：data[i+16] = s[(i+1)%16] = input[(i+1)%16]
        let got = std::slice::from_raw_parts(base.ptr.add(64) as *const f32, 16);
        for i in 0..16 {
            let want = ((i + 1) % 16) as f32 * 1.5;
            assert_eq!(got[i], want, "shared memory: [{i}] got={}", got[i]);
        }
    }
    eprintln!("[gpu] 共享内存基础测试通过");
}

/// fp16 GEMM 对拍：f16 精度 ~3 位十进制（rel ~1e-3），断言 5e-3。
#[test]
fn conv_gemm_f16_vs_cpu() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);

    let (ci, m, h, w) = (32u32, 24u32, 16u32, 12u32);
    let hw = h * w;
    let x_n = (ci * hw) as usize;
    let w_n = (m * ci) as usize;
    let out_n = (m * hw) as usize;
    // f16 packed：每 2 个 f16 = 1 个 uint
    let x16_words = x_n / 2;
    let w16_words = w_n / 2;
    let out16_words = out_n / 2;
    arena
        .alloc(((x16_words + w16_words + m as usize / 2 + out16_words) * 4 + 128) as vk::DeviceSize)
        .unwrap();
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();

    let mut seed = 0xa5a5_5a5a_1234_5678_u64;
    // 生成 f32 测试数据
    let mut xv = vec![0f32; x_n];
    let mut wv = vec![0f32; w_n];
    let mut bv = vec![0f32; m as usize];
    for v in xv.iter_mut() {
        *v = lcg(&mut seed);
    }
    for v in wv.iter_mut() {
        *v = lcg(&mut seed) * 0.5;
    }
    for v in bv.iter_mut() {
        *v = lcg(&mut seed) * 0.1;
    }

    // 转换为 f16 packed
    let x16 = crate::vulkan::fp16::f32_to_f16_words(&xv);
    let w16 = crate::vulkan::fp16::f32_to_f16_words(&wv);
    let b16 = crate::vulkan::fp16::f32_to_f16_words(&bv);

    // 上传（偏移按 f16 元素索引）
    let x_reg = arena.alloc(x16_words as vk::DeviceSize * 4).unwrap();
    let w_reg = arena.alloc(w16_words as vk::DeviceSize * 4).unwrap();
    let b_reg = arena.alloc((m as usize / 2) as vk::DeviceSize * 4).unwrap();
    let out_reg = arena.alloc(out16_words as vk::DeviceSize * 4).unwrap();
    let pp = arena.alloc(128).unwrap();

    // SAFETY: 持久映射；长度与分配一致。
    unsafe {
        std::ptr::copy_nonoverlapping(x16.as_ptr(), x_reg.ptr as *mut u32, x16.len());
        std::ptr::copy_nonoverlapping(w16.as_ptr(), w_reg.ptr as *mut u32, w16.len());
        std::ptr::copy_nonoverlapping(b16.as_ptr(), b_reg.ptr as *mut u32, b16.len());
    }

    let cb = ctx.inner.device.alloc_reusable_cb().unwrap();
    // conv_gemm_f16 参数序与 conv.comp 相同（偏移按 f16 元素索引）
    let mut pb = ParamBlock::new();
    pb.u(x_reg.offset as u32 / 4 * 2) // f16 元素索引 = word 索引 * 2
        .u(w_reg.offset as u32 / 4 * 2)
        .u(b_reg.offset as u32 / 4 * 2)
        .u(OFF_NONE) // 无残差
        .u(out_reg.offset as u32 / 4 * 2)
        .u(1)
        .u(ci)
        .u(h)
        .u(w)
        .u(m)
        .u(1)
        .u(1)
        .u(1)
        .u(1)
        .u(0)
        .u(0)
        .u(1)
        .u(0) // group=1, act=0
        .f(0.0)
        .f(0.0)
        .f(0.0)
        .u(hw)
        .u(w);
    let pc = pb.finish(&pp);
    // SAFETY: cb 录制态；PC 与 conv_gemm_f16.comp 参数对应。
    unsafe {
        record_dispatch(
            &dev,
            cb,
            &ks,
            "conv_gemm_f16",
            pc.bytes(),
            [hw / 2 / 16 + 1, m / 16 + 1, 1],
        );
    }
    ctx.inner.device.end_reusable_cb(cb).unwrap();
    ctx.inner.device.submit_wait_cb(cb).unwrap();

    // 读回 f16 packed 并转 f32
    // SAFETY: 已等信号。
    let out16 =
        unsafe { std::slice::from_raw_parts(out_reg.ptr as *const u32, out16_words).to_vec() };
    let got = crate::vulkan::fp16::f16_words_to_f32(&out16);

    // CPU 参考（f32 精度）
    let mut want = vec![0f32; out_n];
    for mi in 0..m as usize {
        for j in 0..hw as usize {
            let mut acc = 0f32;
            for c in 0..ci as usize {
                acc += xv[c * hw as usize + j] * wv[mi * ci as usize + c];
            }
            want[mi * hw as usize + j] = acc + bv[mi];
        }
    }
    let e = rel_err(&got, &want);
    eprintln!("[gpu] conv_gemm_f16(ci{ci} m{m} {h}x{w}) rel|diff| = {e:.3e}");
    assert!(e < 5e-3, "fp16 GEMM 超容差: {e}");
}

/// fp16 vs fp32 GEMM 性能对比（单内核多次重放计时）。
#[test]
fn gemm_f16_vs_f32_perf() {
    let Some(ctx) = open_or_skip() else { return };
    let dev = ctx.inner.device.raw().clone();
    let mut arena = Arena::new(dev.clone(), ctx.inner.mem_types.staging, true);

    // 用 det 典型形状：ci=32, m=64, hw=207360（大）
    let (ci, m, hw) = (32u32, 64u32, 207_360u32);
    let x_n = (ci * hw) as usize;
    let w_n = (m * ci) as usize;
    let out_n = (m * hw) as usize;
    // f32 需要：x + w + bias + out = (32+2048+64+13271040)*4 ≈ 53 MB
    // f16 需要：一半 ≈ 27 MB
    // 总 arena ≈ 80 MB
    arena.alloc(90_000_000).unwrap();
    let (buf, buf_size) = arena.chunk_range().unwrap();
    let ks = KernelSet::new(&dev, buf, buf_size).unwrap();

    let mut seed = 0xfeed_beef_dead_1234_u64;
    // f32 路径
    let x32 = arena.alloc(x_n as vk::DeviceSize * 4).unwrap();
    let w32 = arena.alloc(w_n as vk::DeviceSize * 4).unwrap();
    let b32 = arena.alloc(m as vk::DeviceSize * 4).unwrap();
    let out32 = arena.alloc(out_n as vk::DeviceSize * 4).unwrap();
    fill(&x32, x_n.min(1_000_000), &mut seed); // 填一部分就够了（不用全填，只测性能）
    fill(&w32, w_n, &mut seed);

    // f16 路径
    let xv = read(&x32, x_n.min(1_000_000));
    let wv = read(&w32, w_n);
    let x16 = crate::vulkan::fp16::f32_to_f16_words(&xv);
    // 只上传前面的部分（完整太大，用部分填充测性能）
    let x16_partial = &x16[..x16.len().min(500_000)];
    let w16 = crate::vulkan::fp16::f32_to_f16_words(&wv);

    let x16_reg = arena.alloc(x16.len() as vk::DeviceSize * 2).unwrap();
    let w16_reg = arena.alloc(w16.len() as vk::DeviceSize * 4).unwrap();
    let b16_reg = arena.alloc(m as vk::DeviceSize * 2).unwrap();
    let out16_reg = arena.alloc(out_n as vk::DeviceSize * 2).unwrap();
    // SAFETY: 持久映射。
    unsafe {
        std::ptr::copy_nonoverlapping(x16.as_ptr(), x16_reg.ptr as *mut u32, x16.len());
        std::ptr::copy_nonoverlapping(w16.as_ptr(), w16_reg.ptr as *mut u32, w16.len());
    }

    let pp = arena.alloc(256).unwrap();

    // f32 GEMM 计时
    let cb32 = ctx.inner.device.alloc_reusable_cb().unwrap();
    let mut pb = ParamBlock::new();
    pb.u(el(&x32))
        .u(el(&w32))
        .u(el(&b32))
        .u(OFF_NONE)
        .u(el(&out32))
        .u(1)
        .u(ci)
        .u(1)
        .u(1)
        .u(m)
        .u(1)
        .u(1)
        .u(1)
        .u(1)
        .u(0)
        .u(0)
        .u(1)
        .u(0)
        .f(0.0)
        .f(0.0)
        .f(0.0)
        .u(hw)
        .u(1);
    let pc32 = pb.finish(&pp);
    // SAFETY: cb 录制态。
    unsafe {
        record_dispatch(
            &dev,
            cb32,
            &ks,
            "conv_gemm",
            pc32.bytes(),
            [hw / 16 + 1, m / 16 + 1, 1],
        );
    }
    ctx.inner.device.end_reusable_cb(cb32).unwrap();

    // f16 GEMM 计时
    let cb16 = ctx.inner.device.alloc_reusable_cb().unwrap();
    let mut pb = ParamBlock::new();
    pb.u(x16_reg.offset as u32 / 4 * 2)
        .u(w16_reg.offset as u32 / 4 * 2)
        .u(b16_reg.offset as u32 / 4 * 2)
        .u(OFF_NONE)
        .u(out16_reg.offset as u32 / 4 * 2)
        .u(1)
        .u(ci)
        .u(1)
        .u(1)
        .u(m)
        .u(1)
        .u(1)
        .u(1)
        .u(1)
        .u(0)
        .u(0)
        .u(1)
        .u(0)
        .f(0.0)
        .f(0.0)
        .f(0.0)
        .u(hw)
        .u(1);
    let pc16 = pb.finish(&pp);
    // SAFETY: 同上。
    unsafe {
        record_dispatch(
            &dev,
            cb16,
            &ks,
            "conv_gemm_f16",
            pc16.bytes(),
            [hw / 2 / 16 + 1, m / 16 + 1, 1],
        );
    }
    ctx.inner.device.end_reusable_cb(cb16).unwrap();

    // 各跑 10 次取中位
    for (name, cb) in [("f32", cb32), ("f16", cb16)] {
        let mut times = Vec::new();
        for _ in 0..10 {
            let t0 = std::time::Instant::now();
            ctx.inner.device.submit_wait_cb(cb).unwrap();
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "[gpu] GEMM {name} (ci{ci} m{m} hw{hw}): 中位 {:.2} ms",
            times[5]
        );
    }
}
