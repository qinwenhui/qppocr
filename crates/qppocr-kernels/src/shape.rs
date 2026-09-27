//! 形状类算子与 matmul/batchnorm：concat / slice / transpose / reshape /
//! squeeze / reduce_mean / matmul / batchnorm。
//!
//! slice 与 transpose 共用 [`copy_strided`]：目标元素
//! `dst[o*inner + k] = src[base + Σ_{i<r-1} idx_i·stride_i + k·stride_{r-1}]`。
//! 曾经每输出元素重算一遍偏移（每维一次 `%`/`/`）；odometer 每外层步一次
//! 加法一次比较。最内维 `stride == 1` 时内循环是 memcpy 而不是 inner 次
//! 标量 load——`Transpose [3,6,8,40,15]` 正是这种。

use crate::buf::F32Buf;
use crate::gemm::sgemm;
use crate::par;

/// 泛型 strided 拷贝（f32 版）。`shape/stride` 按输出维度（含最内维）给出。
///
/// SAFETY（并行形态）：`dst` 的 `[outer_begin*inner, outer_end*inner)` 与
/// 并发调用者不相交。
fn copy_strided_f32(
    src: &[f32],
    dst: &mut [f32],
    shape: &[i64],
    stride: &[i64],
    base: i64,
    outer_begin: i64,
    outer_end: i64,
) {
    let r = shape.len();
    if r == 0 || outer_end <= outer_begin {
        return;
    }
    let inner = shape[r - 1];
    let istr = stride[r - 1];
    if inner <= 0 {
        return;
    }
    // 在 outer_begin 处播种。这是唯一做除法的地方：每线程每节点一次。
    let mut idx = vec![0i64; r.saturating_sub(1).max(1)];
    let mut off = base;
    let mut rem = outer_begin;
    for i in (0..r.saturating_sub(1)).rev() {
        idx[i] = rem % shape[i];
        rem /= shape[i];
        off += idx[i] * stride[i];
    }
    for o in outer_begin..outer_end {
        // ★ dst 是调用方给的**相对切片**（从 outer_begin 起，长
        //   (outer_end-outer_begin)·inner）——必须用相对 o 索引。曾经用
        //   绝对 o：单块路径（outer_begin=0，tiny/small 的日常张量都走它）
        //   恰好掩住错位；大张量并行多块后 ob>0 即越界 panic（3918×2772
        //   照片 + medium 档稳定复现，2026-09-25）。
        let d = &mut dst
            [((o - outer_begin) * inner) as usize..((o - outer_begin + 1) * inner) as usize];
        if istr == 1 {
            d.copy_from_slice(&src[off as usize..(off + inner) as usize]);
        } else {
            for (k, dv) in d.iter_mut().enumerate() {
                *dv = src[(off + k as i64 * istr) as usize];
            }
        }
        // 进位（保 base 不动）
        for i in (0..r.saturating_sub(1)).rev() {
            off += stride[i];
            idx[i] += 1;
            if idx[i] < shape[i] {
                break;
            }
            off -= idx[i] * stride[i];
            idx[i] = 0;
        }
    }
}

/// i64 版（Shape/Gather 等整型张量）。
fn copy_strided_i64(
    src: &[i64],
    dst: &mut [i64],
    shape: &[i64],
    stride: &[i64],
    base: i64,
    outer_begin: i64,
    outer_end: i64,
) {
    let r = shape.len();
    if r == 0 || outer_end <= outer_begin {
        return;
    }
    let inner = shape[r - 1];
    let istr = stride[r - 1];
    if inner <= 0 {
        return;
    }
    let mut idx = vec![0i64; r.saturating_sub(1).max(1)];
    let mut off = base;
    let mut rem = outer_begin;
    for i in (0..r.saturating_sub(1)).rev() {
        idx[i] = rem % shape[i];
        rem /= shape[i];
        off += idx[i] * stride[i];
    }
    for o in outer_begin..outer_end {
        let d = &mut dst
            [((o - outer_begin) * inner) as usize..((o - outer_begin + 1) * inner) as usize];
        if istr == 1 {
            d.copy_from_slice(&src[off as usize..(off + inner) as usize]);
        } else {
            for (k, dv) in d.iter_mut().enumerate() {
                *dv = src[(off + k as i64 * istr) as usize];
            }
        }
        for i in (0..r.saturating_sub(1)).rev() {
            off += stride[i];
            idx[i] += 1;
            if idx[i] < shape[i] {
                break;
            }
            off -= idx[i] * stride[i];
            idx[i] = 0;
        }
    }
}

/// 张量的两种有效载荷（kernels 层的最小张量表示；core 的 Tensor 包装它）。
#[derive(Debug, Clone, PartialEq)]
pub enum Payload {
    /// f32（池化缓冲）。
    F32(F32Buf),
    /// i64（含 I32/BOOL，按 ONNX 惯例提升）。
    I64(Vec<i64>),
}

/// 载荷的借用视图：执行器传权重/中间量用，不克隆数据。
#[derive(Debug, Clone, Copy)]
pub enum PayloadRef<'a> {
    /// f32。
    F32(&'a [f32]),
    /// i64。
    I64(&'a [i64]),
}

impl Payload {
    /// 借用视图。
    pub fn as_ref(&self) -> PayloadRef<'_> {
        match self {
            Payload::F32(v) => PayloadRef::F32(v.as_slice()),
            Payload::I64(v) => PayloadRef::I64(v),
        }
    }
    /// f32 载荷的切片（Payload::F32 时）。
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Payload::F32(v) => Some(v.as_slice()),
            Payload::I64(_) => None,
        }
    }
}

/// slice 的区间解析（负索引、步长、钳位），镜像  `slice_range`。
fn slice_range(start: i64, end: i64, step: i64, n: i64) -> (i64, i64) {
    let mut start = start;
    let mut end = end;
    if start < 0 {
        start += n;
    }
    if end < 0 && !(step < 0 && end == i64::MIN) {
        end += n;
    }
    if step > 0 {
        start = start.max(0);
        end = end.min(n);
        (start.min(n), start.max(end.min(n)))
    } else {
        start = start.min(n - 1);
        end = end.max(-1);
        (start, start.min(end))
    }
}

/// Slice。`starts/ends/axes/steps` 与 ONNX 属性同名同义；`axes` 空表示
/// 沿前 `starts.len()` 维。
pub fn slice_tensor(
    x: PayloadRef<'_>,
    x_shape: &[i64],
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
) -> (Payload, Vec<i64>) {
    let r = x_shape.len();
    let ax: Vec<i64> = if axes.is_empty() {
        (0..starts.len() as i64).collect()
    } else {
        axes.to_vec()
    };
    let st: Vec<i64> = if steps.is_empty() {
        vec![1; starts.len()]
    } else {
        steps.to_vec()
    };
    let mut b = vec![0i64; r];
    let mut e: Vec<i64> = x_shape.to_vec();
    let mut sp = vec![1i64; r];
    for i in 0..ax.len() {
        let a = if ax[i] < 0 { ax[i] + r as i64 } else { ax[i] } as usize;
        let (bb, ee) = slice_range(starts[i], ends[i], st[i], x_shape[a]);
        b[a] = bb;
        e[a] = ee;
        sp[a] = st[i];
    }
    let mut shape = vec![0i64; r];
    let mut count = 1i64;
    for i in 0..r {
        let len = if sp[i] > 0 {
            (e[i] - b[i] + sp[i] - 1) / sp[i]
        } else {
            (b[i] - e[i] + (-sp[i]) - 1) / (-sp[i])
        };
        let len = len.max(0);
        shape[i] = len;
        count *= len;
    }
    let y = match x {
        PayloadRef::F32(d) => {
            let mut out = F32Buf::with_zeroed(count as usize);
            if count > 0 {
                slice_copy(x_shape, &shape, &b, &sp, CopyKind::F32(d, &mut out));
            }
            Payload::F32(out)
        }
        PayloadRef::I64(d) => {
            let mut out = vec![0i64; count as usize];
            if count > 0 {
                slice_copy(x_shape, &shape, &b, &sp, CopyKind::I64(d, &mut out));
            }
            Payload::I64(out)
        }
    };
    (y, shape)
}

enum CopyKind<'a> {
    F32(&'a [f32], &'a mut F32Buf),
    I64(&'a [i64], &'a mut Vec<i64>),
}

fn slice_copy(x_shape: &[i64], out_shape: &[i64], b: &[i64], sp: &[i64], mut kind: CopyKind<'_>) {
    let r = x_shape.len();
    // 输入步长（按输入维）
    let mut xstr = vec![1i64; r];
    for i in (0..r.saturating_sub(1)).rev() {
        xstr[i] = xstr[i + 1] * x_shape[i + 1];
    }
    // 输出沿 dim i 走一步 = 输入走 sp[i] 步：贡献 sp[i]·xstr[i]
    let mut stride = vec![0i64; r];
    let mut base = 0i64;
    for i in 0..r {
        stride[i] = sp[i] * xstr[i];
        base += b[i] * xstr[i];
    }
    let count: i64 = out_shape.iter().product();
    let outer = if *out_shape.last().unwrap_or(&1) > 0 {
        count / out_shape[r - 1]
    } else {
        0
    };
    match &mut kind {
        CopyKind::F32(src, out) => {
            let dstp = par::SyncPtr::new(out.as_mut_slice().as_mut_ptr());
            par::parallel_for_elems(outer as usize, count as usize, |ob, oe| {
                // SAFETY: [ob, oe) 的外层步与其他块不相交。
                // SAFETY: [ob, oe) 的外层步与其他并行块不相交。
                let dst = unsafe {
                    dstp.offset(ob * out_shape[r - 1] as usize)
                        .slice((oe - ob) * out_shape[r - 1] as usize)
                };
                copy_strided_f32(src, dst, out_shape, &stride, base, ob as i64, oe as i64);
            });
        }
        CopyKind::I64(src, out) => {
            let dstp = par::SyncPtr::new(out.as_mut_ptr());
            par::parallel_for_elems(outer as usize, count as usize, |ob, oe| {
                // SAFETY: [ob, oe) 的外层步与其他并行块不相交。
                let dst = unsafe {
                    dstp.offset(ob * out_shape[r - 1] as usize)
                        .slice((oe - ob) * out_shape[r - 1] as usize)
                };
                copy_strided_i64(src, dst, out_shape, &stride, base, ob as i64, oe as i64);
            });
        }
    }
}

/// Transpose。`perm` 空表示反转。
pub fn transpose_tensor(x: PayloadRef<'_>, x_shape: &[i64], perm: &[i64]) -> (Payload, Vec<i64>) {
    let r = x_shape.len();
    let p: Vec<usize> = if perm.is_empty() {
        (0..r).rev().collect()
    } else {
        perm.iter()
            .map(|&v| (if v < 0 { v + r as i64 } else { v }) as usize)
            .collect()
    };
    let shape: Vec<i64> = p.iter().map(|&i| x_shape[i]).collect();
    let mut xstr = vec![1i64; r];
    for i in (0..r.saturating_sub(1)).rev() {
        xstr[i] = xstr[i + 1] * x_shape[i + 1];
    }
    // 每个输出维对应一个输入步长
    let istr: Vec<i64> = p.iter().map(|&i| xstr[i]).collect();
    let total: i64 = shape.iter().product();
    let outer = if *shape.last().unwrap_or(&1) > 0 {
        total / shape[r - 1]
    } else {
        1
    };
    match x {
        PayloadRef::F32(d) => {
            let mut out = F32Buf::with_zeroed(total as usize);
            let dstp = par::SyncPtr::new(out.as_mut_slice().as_mut_ptr());
            par::parallel_for_elems(outer as usize, total as usize, |b, e| {
                // SAFETY: [b, e) 的外层步与其他并行块不相交。
                let dst = unsafe {
                    dstp.offset(b * shape[r - 1] as usize)
                        .slice((e - b) * shape[r - 1] as usize)
                };
                copy_strided_f32(d, dst, &shape, &istr, 0, b as i64, e as i64);
            });
            (Payload::F32(out), shape)
        }
        PayloadRef::I64(d) => {
            let mut out = vec![0i64; total as usize];
            let dstp = par::SyncPtr::new(out.as_mut_ptr());
            par::parallel_for_elems(outer as usize, total as usize, |b, e| {
                // SAFETY: [b, e) 的外层步与其他并行块不相交。
                let dst = unsafe {
                    dstp.offset(b * shape[r - 1] as usize)
                        .slice((e - b) * shape[r - 1] as usize)
                };
                copy_strided_i64(d, dst, &shape, &istr, 0, b as i64, e as i64);
            });
            (Payload::I64(out), shape)
        }
    }
}

/// Concat（F32 或 I64，按第一个输入的种类）。`axis` 支持负索引。
pub fn concat_any(
    xs: &[PayloadRef<'_>],
    xs_shape: &[&[i64]],
    axis: i64,
    out: &mut Payload,
) -> Vec<i64> {
    assert!(!xs.is_empty(), "concat empty");
    let r = xs_shape[0].len();
    let axis = if axis < 0 { axis + r as i64 } else { axis } as usize;
    assert!(axis < r, "concat axis");
    let mut shape = xs_shape[0].to_vec();
    let mut concat_dim = 0i64;
    for s in xs_shape {
        for (i, (&d, &d0)) in s.iter().zip(&shape).enumerate() {
            if i != axis {
                assert!(d == d0, "concat shape mismatch");
            }
        }
        concat_dim += s[axis];
    }
    shape[axis] = concat_dim;
    let outer: i64 = shape[..axis].iter().product();
    let inner: i64 = shape[axis + 1..].iter().product();

    // 纯内存搬运：`Concat [1,32,752,992]` 拷 95 MB 曾单线程跑 21 ms
    // （单核带宽的三分之一）。第 i 个输入对 outer o 的通道块在两侧都是
    // `shape[axis]·inner` 个连续元素——工作项是切成 ~128 KB 块的 (outer,
    // channel) 平面。只按通道切不够：两个输入的 concat 只有 2 个工作项。
    let elem = match xs[0] {
        PayloadRef::F32(_) => 4,
        PayloadRef::I64(_) => 8,
    };
    let mut offs = Vec::with_capacity(xs.len().min(16));
    {
        let mut acc = 0i64;
        for s in xs_shape.iter().take(16) {
            offs.push(acc);
            acc += s[axis];
        }
    }
    const BLOCK_ELEMS: i64 = 1 << 15; // 128 KB of f32
    let blocks = 1i64.max((inner + BLOCK_ELEMS - 1) / BLOCK_ELEMS);
    let units = (outer * concat_dim * blocks) as usize;

    let total = outer * concat_dim * inner;
    match out {
        Payload::F32(yd) => {
            // SAFETY: concat 的每个元素都由下方块拷贝写满。
            unsafe { yd.resize_uninit(total as usize) };
            let dstp = par::SyncPtr::new(yd.as_mut_slice().as_mut_ptr());
            let body = |ub: usize, ue: usize| {
                for u in ub..ue {
                    let (o, rem) = (
                        u as i64 / (concat_dim * blocks),
                        u as i64 % (concat_dim * blocks),
                    );
                    let (ch, blk) = (rem / blocks, rem % blocks);
                    let i0 = blk * BLOCK_ELEMS;
                    let len = BLOCK_ELEMS.min(inner - i0);
                    if len <= 0 {
                        continue;
                    }
                    // 哪个输入拥有输出通道 ch，以及它自己的第几通道
                    let mut si = 0usize;
                    while si + 1 < xs.len().min(16) && offs[si + 1] <= ch {
                        si += 1;
                    }
                    let PayloadRef::F32(src) = xs[si] else {
                        unreachable!()
                    };
                    let d = xs_shape[si][axis];
                    let src_off = ((o * d) + (ch - offs[si])) * inner + i0;
                    let dst_off = ((o * concat_dim) + ch) * inner + i0;
                    unsafe {
                        // SAFETY: 块 (o, ch, blk) 与其他块不相交。
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr().add(src_off as usize),
                            dstp.get().add(dst_off as usize),
                            len as usize,
                        );
                    }
                }
            };
            if par::worth_forking_bytes(total as usize * elem) {
                par::parallel_for(units.min(1 << 30), 1, body);
            } else {
                body(0, units);
            }
        }
        Payload::I64(yd) => {
            yd.clear();
            yd.resize(total as usize, 0);
            let dstp = par::SyncPtr::new(yd.as_mut_ptr());
            let body = |ub: usize, ue: usize| {
                for u in ub..ue {
                    let (o, rem) = (
                        u as i64 / (concat_dim * blocks),
                        u as i64 % (concat_dim * blocks),
                    );
                    let (ch, blk) = (rem / blocks, rem % blocks);
                    let i0 = blk * BLOCK_ELEMS;
                    let len = BLOCK_ELEMS.min(inner - i0);
                    if len <= 0 {
                        continue;
                    }
                    let mut si = 0usize;
                    while si + 1 < xs.len().min(16) && offs[si + 1] <= ch {
                        si += 1;
                    }
                    let PayloadRef::I64(src) = xs[si] else {
                        unreachable!()
                    };
                    let d = xs_shape[si][axis];
                    let src_off = ((o * d) + (ch - offs[si])) * inner + i0;
                    let dst_off = ((o * concat_dim) + ch) * inner + i0;
                    // SAFETY: 块 (o, ch, blk) 与其他并行块不相交。
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr().add(src_off as usize),
                            dstp.get().add(dst_off as usize),
                            len as usize,
                        );
                    }
                }
            };
            if par::worth_forking_bytes(total as usize * elem) {
                par::parallel_for(units.min(1 << 30), 1, body);
            } else {
                body(0, units);
            }
        }
    }
    shape
}

/// Reshape：`-1` 推断、`0` 表示保留输入该维。纯形状变换（数据不动，clone）。
pub fn reshape_shape(x_shape: &[i64], shape_in: &[i64]) -> Vec<i64> {
    let total: i64 = x_shape.iter().product();
    let mut known = 1i64;
    let mut infer_idx: isize = -1;
    let mut shape = shape_in.to_vec();
    for (i, &d) in shape_in.iter().enumerate() {
        if d == -1 {
            assert!(infer_idx < 0, "reshape: multiple -1");
            infer_idx = i as isize;
        } else if d == 0 {
            // 0 = 保留输入的该维——仍计入乘积，否则推断的 -1 维会错
            shape[i] = x_shape[i];
            known *= shape[i];
        } else {
            known *= d;
        }
    }
    if infer_idx >= 0 {
        shape[infer_idx as usize] = total / known;
    }
    assert!(
        shape.iter().product::<i64>() == total,
        "reshape: numel mismatch"
    );
    shape
}

/// ReduceMean。`axes` 空表示全部维度归约。
pub fn reduce_mean(
    x: &[f32],
    x_shape: &[i64],
    axes: &[i64],
    keepdims: bool,
    out: &mut F32Buf,
) -> Vec<i64> {
    let r = x_shape.len();
    let mut red = vec![false; r];
    if axes.is_empty() {
        red.iter_mut().for_each(|v| *v = true);
    }
    for &a in axes {
        let a = if a < 0 { a + r as i64 } else { a } as usize;
        red[a] = true;
    }
    // 连续归约段拆成 outer / mid / inner（f64 累加与基准一致）
    let (mut outer, mut mid, mut inner) = (1i64, 1i64, 1i64);
    let mut state = 0;
    for (i, &is_red) in red.iter().enumerate() {
        if is_red {
            mid *= x_shape[i];
            state = 1;
        } else if state == 0 {
            outer *= x_shape[i];
        } else {
            inner *= x_shape[i];
        }
    }
    let mut shape = Vec::new();
    for (i, &is_red) in red.iter().enumerate() {
        if !is_red {
            shape.push(x_shape[i]);
        } else if keepdims {
            shape.push(1);
        }
    }
    // SAFETY: 每元素写一次。
    unsafe { out.resize_uninit((outer * inner) as usize) };
    let op = par::SyncPtr::new(out.as_mut_slice().as_mut_ptr());
    let numel: i64 = x_shape.iter().product();
    par::parallel_for_elems((outer * inner) as usize, numel as usize, |b, e| {
        for lin in b as i64..e as i64 {
            let (o, i) = (lin / inner, lin % inner);
            let base = o * mid * inner + i;
            let mut s = 0f64;
            for m in 0..mid {
                s += x[(base + m * inner) as usize] as f64;
            }
            // SAFETY: 输出元素 lin 与其他块不相交。
            unsafe { *op.get().add(lin as usize) = (s / mid as f64) as f32 };
        }
    });
    shape
}

/// 批量 matmul（右对齐批次广播）。A: `[..., M, K]`，B: `[..., K, N]`；
/// B 可以是 rank-2 `[K,N]` 被所有批次共享。
pub fn matmul(
    a: &[f32],
    a_shape: &[i64],
    b: &[f32],
    b_shape: &[i64],
    out: &mut F32Buf,
) -> Vec<i64> {
    let ra = a_shape.len();
    let rb = b_shape.len();
    assert!(ra >= 2 && rb >= 2, "matmul rank");
    let (m, k) = (a_shape[ra - 2], a_shape[ra - 1]);
    let (k2, n) = (b_shape[rb - 2], b_shape[rb - 1]);
    assert!(k == k2, "matmul inner dim mismatch");
    let rr = ra.max(rb) - 2;
    let ba: Vec<i64> = a_shape[..ra - 2].to_vec();
    let bb: Vec<i64> = b_shape[..rb - 2].to_vec();
    let mut bshape = vec![1i64; rr];
    for i in 0..rr {
        let da = if ba.len() as i64 - rr as i64 + i as i64 >= 0 {
            ba[(ba.len() as i64 - rr as i64 + i as i64) as usize]
        } else {
            1
        };
        let db = if bb.len() as i64 - rr as i64 + i as i64 >= 0 {
            bb[(bb.len() as i64 - rr as i64 + i as i64) as usize]
        } else {
            1
        };
        assert!(da == db || da == 1 || db == 1, "matmul batch broadcast");
        bshape[i] = da.max(db);
    }
    let nbatch: i64 = bshape.iter().product();
    let abatch: i64 = ba.iter().product();
    let bbatch: i64 = bb.iter().product();
    let mut out_shape = bshape.clone();
    out_shape.push(m);
    out_shape.push(n);
    // SAFETY: sgemm 每元素写满。
    unsafe { out.resize_uninit((nbatch * m * n) as usize) };
    let act = crate::activation::Activation::default();

    let amat = (m * k) as usize;
    let bmat = (k * n) as usize;
    if nbatch == 1 {
        sgemm(
            a, b, out, m as usize, n as usize, k as usize, n as usize, None, &act,
        );
        return out_shape;
    }

    // 单个批次大到能喂饱整机的，交给 sgemm：rec 的分类头 [B,T,80]x[80,6906]，
    // 行并行版本每个输出行把 2.2 MB 权重整读一遍，sgemm 按 N 分块让每列面板
    // 只读一次——同样算术 ~10 倍少的流量。
    if (m * n * k) as f64 >= par::thresholds().gemm_par_min {
        // A 逐批次不同、B 相同且 A 是纯堆叠：整节点折成一个 M 乘出来的 GEMM。
        // 每个输出元素仍是同序的 k 和，sgemm 的结果不依赖 M 怎么分块——位级安全。
        let folded = nbatch * m;
        if nbatch > 1
            && abatch == nbatch
            && bbatch == 1
            && folded <= i32::MAX as i64
            && n <= i32::MAX as i64
            && k <= i32::MAX as i64
        {
            sgemm(
                a,
                b,
                out,
                folded as usize,
                n as usize,
                k as usize,
                n as usize,
                None,
                &act,
            );
            return out_shape;
        }
        for bi in 0..nbatch {
            let ap = &a[if abatch == 1 {
                0
            } else {
                ((bi % abatch) * amat as i64) as usize
            }..];
            let bp = &b[if bbatch == 1 {
                0
            } else {
                ((bi % bbatch) * bmat as i64) as usize
            }..];
            let cp = &mut out[(bi * m * n) as usize..];
            sgemm(
                ap, bp, cp, m as usize, n as usize, k as usize, n as usize, None, &act,
            );
        }
        return out_shape;
    }

    // 否则（attention）：小矩阵，按行外积，切在行上让负载均衡跨全部行
    let rows = (nbatch * m) as usize;
    let op = par::SyncPtr::new(out.as_mut_ptr());
    par::parallel_for(rows.min(1 << 29), 1, |tb, te| {
        for t in tb..te {
            let (bi, mm) = ((t as i64) / m, (t as i64) % m);
            let arow = &a[if abatch == 1 {
                0
            } else {
                ((bi % abatch) * amat as i64) as usize
            } + (mm * k) as usize..];
            let bp = &b[if bbatch == 1 {
                0
            } else {
                ((bi % bbatch) * bmat as i64) as usize
            }..];
            // SAFETY: 行 t 与其他块不相交。此路径累加进 C，先自己清零
            // （0 + x == x，和不变）。
            let crow = unsafe { op.offset(t * n as usize).slice(n as usize) };
            crow.fill(0.0);
            for kk in 0..k as usize {
                let av = arow[kk];
                let braw = &bp[kk * n as usize..];
                for (cj, cv) in crow.iter_mut().enumerate() {
                    // fma：基准的 `C[n] += av*b` 在 -ffp-contract 下即此形态
                    *cv = av.mul_add(braw[cj], *cv);
                }
            }
        }
    });
    out_shape
}

/// BatchNormalization（推断语义：scale/var 作用于通道维）。
/// 一般会在图优化里折进 Conv；这里处理未折的情形。
#[allow(clippy::too_many_arguments)] //  ops.hpp 签名镜像
pub fn batchnorm(
    x: &[f32],
    x_shape: &[i64],
    scale: &[f32],
    bias: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
    out: &mut F32Buf,
) {
    let n = x_shape[0] as usize;
    let c = x_shape[1] as usize;
    let plane = usize::checked_div(x.len(), n * c).unwrap_or(0);
    // SAFETY: 每元素写一次。
    unsafe { out.resize_uninit(x.len()) };
    let op = par::SyncPtr::new(out.as_mut_ptr());
    // 平面索引是 nc*plane、通道索引是 nc%C——纯 c*plane 只对 N==1 成立。
    // 单元是整通道平面，成本读 N*C*plane 元素。
    par::parallel_for_elems(n * c, n * c * plane, |nb, ne| {
        for nc in nb..ne {
            let ch = nc % c;
            let a = scale[ch] / (var[ch] + eps).sqrt();
            // bias − mean·a 的 fnmadd 形态：GCC 把 c − a·b 收缩成
            // vfnmadd（单次舍入）。(-m) 的符号翻转是精确的，等价。
            let b = (-mean[ch]).mul_add(a, bias[ch]);
            // SAFETY: 平面 nc 与其他块不相交。
            let seg = unsafe { op.offset(nc * plane).slice(plane) };
            for (i, v) in seg.iter_mut().enumerate() {
                // x·a + b 的 fma 形态（GCC -ffp-contract 收缩，Rust 不收缩）
                *v = x[nc * plane + i].mul_add(a, b);
            }
        }
    });
}
