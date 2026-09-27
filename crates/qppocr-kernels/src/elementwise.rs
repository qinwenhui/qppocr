//! 广播二元算子。
//!
//! numpy 风格广播，`y = a op b`。五条路径（`binary_op` 的）：
//!
//! 1. 一侧是单元素（PP-OCR 最常见：hardswish 的 ×3 与 ÷6、SE 块的逐通道
//!    缩放）——完全可向量化；
//! 2. 尾部若干维上一侧稠密、另一侧常量（SE 块 `[1,C,1,1]`×`[1,C,H,W]`、
//!    bias 加）——内层 run 向量化；
//! 3. 无广播（同形）——一趟平坦 pass；
//! 4. 通用：odomoter 走法（每元素 O(1) 摊还，代替每维一次取模+除法）。
//!
//! `binary_op_inplace` 是每条残差 Add 和 `[C,1,1]` 缩放走的路径，多一条
//! 「b 在最内维既非常量也不稠密」的纯标量走法（分类器的 `[N,T,C] += [C]`
//! 曾经掉进这里，1.8 ms 在任何线程数下都不动——后来 run 路径学会吃
//! 稠密 run 才救回来）。
//!
//! 位级说明：每条路径对每个元素恰好按同样顺序结合一次，路径间逐位一致
//! （原注释「bit-identical」的前提）。

use crate::buf::F32Buf;
use crate::par;

/// 二元算子种类。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// 加。
    Add,
    /// 减。
    Sub,
    /// 乘。
    Mul,
    /// 除。
    Div,
    /// 幂。⚠ `f32::powf` 走 libm，跨实现有 ulp 级差异；GELU 子图正常会被
    /// 融合掉，这个算子极少真的执行。
    Pow,
}

impl BinOp {
    /// op 码（0..4 = + - * / pow）；向量路径用。
    #[inline]
    pub fn code(self) -> u8 {
        match self {
            BinOp::Add => 0,
            BinOp::Sub => 1,
            BinOp::Mul => 2,
            BinOp::Div => 3,
            BinOp::Pow => 4,
        }
    }
    /// 施加二元运算（测试的朴素参考也用它）。
    #[inline]
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
            BinOp::Pow => a.powf(b),
        }
    }
    #[inline]
    fn apply_inplace_rev(self, dst: f32, b: f32) -> f32 {
        // 就地版语义：dst op= b（除 Pow 是 dst = pow(dst, b) 外都同 apply）
        self.apply(dst, b)
    }
}

/// 广播元数据：shape 与两侧步长，全部按**反转序**存（下标 0 = 最快变化维）。
struct BroadcastMeta {
    shape: Vec<i64>,
    /// 就地路径只读 `strb`；`binary_op` 自算两侧步长。
    #[allow(dead_code)]
    stra: Vec<i64>,
    strb: Vec<i64>,
}

/// 反转序取维：`i < rr ? s[rr-1-i] : 1`。
#[inline]
fn dim_at(s: &[i64], i: usize) -> i64 {
    let rr = s.len();
    if i < rr { s[rr - 1 - i] } else { 1 }
}

fn broadcast_meta(a_shape: &[i64], b_shape: &[i64]) -> Option<BroadcastMeta> {
    let ra = a_shape.len();
    let rb = b_shape.len();
    let r = ra.max(rb);
    let mut shape = vec![1i64; r];
    for i in 0..r {
        let (da, db) = (dim_at(a_shape, i), dim_at(b_shape, i));
        if !(da == db || da == 1 || db == 1) {
            return None;
        }
        shape[i] = da.max(db);
    }
    let mut stra = vec![0i64; r];
    let mut strb = vec![0i64; r];
    let mut acc = 1i64;
    for i in 0..ra {
        stra[i] = if dim_at(a_shape, i) == 1 { 0 } else { acc };
        acc *= dim_at(a_shape, i);
    }
    let mut acc = 1i64;
    for i in 0..rb {
        strb[i] = if dim_at(b_shape, i) == 1 { 0 } else { acc };
        acc *= dim_at(b_shape, i);
    }
    Some(BroadcastMeta { shape, stra, strb })
}

/// `a` 的形状是否已经等于 `a op b` 的广播结果（能就地做，省一次分配和
/// 一整趟额外的内存搬运）。
pub fn binary_can_inplace(a_shape: &[i64], b_shape: &[i64]) -> bool {
    let Some(m) = broadcast_meta(a_shape, b_shape) else {
        return false;
    };
    if m.shape.len() != a_shape.len() {
        return false;
    }
    (0..a_shape.len()).all(|i| a_shape[i] == m.shape[a_shape.len() - 1 - i])
}

/// `out[n,c,s] = feature[n,c,s] · gate[n,c] + residual[n,c,s]`。
///
/// 对应「门控块」`Add(Mul(特征, 门), 残差)`——PP-OCRv6 主干里每个大核块都有
/// 一个：门是 `[1,C,1,1]` 的逐通道标量（SE 路径算出来的），单独跑 Mul 再接
/// Add 要多写一趟、多读一趟中间的 12 MB 特征图。
///
/// ⚠ **不能用 fma**：独立的 Mul 先舍入一次、Add 再舍入一次，融合版必须复现
/// 这两次舍入（`(f·g) + r`），否则低位就分叉。Rust 默认不做浮点收缩，照字面
/// 写即可。
///
/// `hw` 是每个通道平面的元素数，`c` 是通道数（gate 的长度）。
#[allow(clippy::too_many_arguments)]
pub fn mul_add_scale(
    feature: &[f32],
    gate: &[f32],
    residual: &[f32],
    out: &mut [f32],
    c: usize,
    hw: usize,
) {
    if hw == 0 || c == 0 || feature.is_empty() {
        return;
    }
    assert_eq!(feature.len(), residual.len(), "mul_add_scale: 输入不等长");
    assert_eq!(feature.len(), out.len(), "mul_add_scale: 输出不等长");
    let planes = feature.len() / hw;
    let op = par::SyncPtr::new(out.as_mut_ptr());
    par::parallel_for_units(planes, |b, e| {
        for p in b..e {
            let gv = gate[p % c];
            let s = p * hw;
            // SAFETY: 平面 p 的区间与其他并行块不相交。
            let o = unsafe { op.offset(s).slice(hw) };
            let f = &feature[s..s + hw];
            let r = &residual[s..s + hw];
            for i in 0..hw {
                o[i] = f[i] * gv + r[i];
            }
        }
    });
}

/// [`mul_add_scale`] 的**分配输出**版本。core 是 `forbid(unsafe_code)`，
/// 拿不到未初始化缓冲；这里把「分配 + 写满」封在 kernels 内部，省掉
/// `with_zeroed` 那趟 12 MB 的清零。
pub fn mul_add_scale_alloc(
    feature: &[f32],
    gate: &[f32],
    residual: &[f32],
    c: usize,
    hw: usize,
) -> F32Buf {
    let mut out = F32Buf::new();
    // SAFETY: mul_add_scale 对每个输出元素恰好写一次（hw·c == len 时写满）。
    unsafe { out.resize_uninit(feature.len()) };
    mul_add_scale(feature, gate, residual, out.as_mut_slice(), c, hw);
    out
}

/// `a`（稠密、已持有广播输出形状）`op= b`（广播到 a 上）。
///
/// 最内若干维上 b 为常量时构成一个可向量化的 run；b 非常量但沿该维稠密
/// （步长 1）时，两个操作数在 run 上都连续，同一套机制把 broadcast 换成 load。
pub fn binary_op_inplace(a: &mut [f32], a_shape: &[i64], b: &[f32], b_shape: &[i64], op: BinOp) {
    let r = a_shape.len();
    let total: i64 = a_shape.iter().product();
    if total <= 0 {
        return;
    }
    let total = total as usize;
    debug_assert_eq!(a.len(), total);

    // 稠密同形：一趟平坦 pass。v6 backbone 的每条残差 Add 都走这里，
    // 曾经掉进下面的多维走法（每元素推进一次 r 维下标向量），
    // 6x160x3x80 上 ~0.5 ms vs 这里 ~0.03 ms。
    if a_shape == b_shape {
        let pa = par::SyncPtr::new(a.as_mut_ptr());
        let opc = op.code();
        par::parallel_for_elems(total, total, |b0, e0| {
            // SAFETY: 区间 [b0, e0) 与其他并行块不相交；a/b 等长。
            unsafe { crate::arch::binary_flat_inplace(pa.get(), b.as_ptr(), b0, e0, opc) };
        });
        return;
    }

    let Some(m) = broadcast_meta(a_shape, b_shape) else {
        panic!("inplace broadcast mismatch");
    };
    let BroadcastMeta {
        shape, strb: sb, ..
    } = m;
    let pa = par::SyncPtr::new(a.as_mut_ptr());

    // b 为常量的最内维构成一个 run；否则 b 在最内维稠密（步长 1）时同样适用。
    let mut k = 0;
    while k < r && sb[k] == 0 {
        k += 1;
    }
    let b_dense_run = k == 0 && !sb.is_empty() && sb[0] == 1;
    if b_dense_run {
        k = 1;
    }
    if k == 0 {
        // b 沿最内维变化且不稠密：没有可向量化的 run，纯逐元素走 b 的步长。
        // （ 这里也是串行。）
        let mut idx = vec![0i64; r];
        let mut ib: i64 = 0;
        for lin in 0..total {
            let vb = b[ib as usize];
            // SAFETY: 串行走法独占整个 a（此分支不 fork）。
            unsafe {
                let p = pa.get().add(lin);
                *p = op.apply_inplace_rev(*p, vb);
            }
            for i in 0..r {
                idx[i] += 1;
                ib += sb[i];
                if idx[i] < shape[i] {
                    break;
                }
                idx[i] = 0;
                ib -= sb[i] * shape[i];
            }
        }
        return;
    }

    let runlen: i64 = shape[..k].iter().product();
    let no = r - k;
    let nouter = (total as i64 / runlen) as usize;

    let kern = |o0: usize, o1: usize| {
        let mut idx = vec![0i64; no.max(1)];
        let mut ib: i64 = 0;
        // 把 odometer 推进到 o0（ 原样：每块从 0 起推）
        for _ in 0..o0 {
            for i in 0..no {
                let di = k + i;
                idx[i] += 1;
                ib += sb[di];
                if idx[i] < shape[di] {
                    break;
                }
                idx[i] = 0;
                ib -= sb[di] * shape[di];
            }
        }
        let opc = op.code(); // 向量/标量分支共用
        for o in o0..o1 {
            let cv = b[ib as usize];
            // SAFETY: 输出 run [o*runlen, (o+1)*runlen) 两两不相交；
            // b 稠密时 bsrc 从 ib 起有 runlen 个元素（步长 1 的保证）。
            unsafe {
                crate::arch::binary_run_inplace(
                    pa.get(),
                    b.as_ptr(),
                    o,
                    runlen as usize,
                    ib as usize,
                    b_dense_run,
                    cv,
                    opc,
                )
            };
            for i in 0..no {
                let di = k + i;
                idx[i] += 1;
                ib += sb[di];
                if idx[i] < shape[di] {
                    break;
                }
                idx[i] = 0;
                ib -= sb[di] * shape[di];
            }
        }
    };
    par::parallel_for_elems(nouter, total, kern);
}

/// `y = a op b`（分配输出），返回 `(数据, 正向形状)`。
pub fn binary_op(
    a: &[f32],
    a_shape: &[i64],
    b: &[f32],
    b_shape: &[i64],
    op: BinOp,
) -> (F32Buf, Vec<i64>) {
    let ra = a_shape.len();
    let rb = b_shape.len();
    let r = ra.max(rb);
    let mut shape = vec![1i64; r];
    for i in 0..r {
        let (da, db) = (dim_at(a_shape, i), dim_at(b_shape, i));
        assert!(
            da == db || da == 1 || db == 1,
            "broadcast shape mismatch: a={a_shape:?} b={b_shape:?}"
        );
        shape[i] = da.max(db);
    }
    let mut stra = vec![0i64; r];
    let mut strb = vec![0i64; r];
    {
        let mut acc = 1i64;
        for i in 0..ra {
            stra[i] = if dim_at(a_shape, i) == 1 { 0 } else { acc };
            acc *= dim_at(a_shape, i);
        }
        let mut acc = 1i64;
        for i in 0..rb {
            strb[i] = if dim_at(b_shape, i) == 1 { 0 } else { acc };
            acc *= dim_at(b_shape, i);
        }
    }
    let total: i64 = shape.iter().product();
    let out_shape: Vec<i64> = shape.iter().rev().copied().collect();
    if total <= 0 {
        return (F32Buf::new(), out_shape);
    }
    let total = total as usize;
    // SAFETY: 每条路径对每个输出元素恰好写一次（标量/向量/odometer 同）。
    let mut y = unsafe { F32Buf::with_uninit(total) };
    let yp = par::SyncPtr::new(y.as_mut_ptr());

    let nobc = (0..r).all(|i| stra[i] != 0 && strb[i] != 0);
    // b 沿最内维（反转序 dim0）稠密 = 步长 1——run 向量化可用 load
    #[allow(clippy::needless_range_loop)]
    let b_dense_inner = r > 0 && strb[0] == 1;

    // 快路径 1：一侧是单元素
    if a.len() == 1 || b.len() == 1 {
        let a_is_scalar = a.len() == 1;
        let sv = if a_is_scalar { a[0] } else { b[0] };
        let v = if a_is_scalar { b } else { a };
        let opc = op.code();
        par::parallel_for_elems(total, total, |b0, e0| {
            // SAFETY: 区间 [b0, e0) 与其他并行块不相交；v 同长。
            unsafe {
                crate::arch::binary_scalar_bcast(v.as_ptr(), sv, a_is_scalar, yp.get(), b0, e0, opc)
            };
        });
        return (y, out_shape);
    }

    // 快路径 2：尾部维上一侧稠密、另一侧常量 → 内层 run
    {
        let mut ka = 0;
        let mut exp = 1i64;
        for i in 0..r {
            if stra[i] == 0 && strb[i] == exp {
                ka += 1;
            } else {
                break;
            }
            exp *= shape[i];
        }
        let mut kb = 0;
        let mut exp = 1i64;
        for i in 0..r {
            if strb[i] == 0 && stra[i] == exp {
                kb += 1;
            } else {
                break;
            }
            exp *= shape[i];
        }
        if ka > 0 || kb > 0 {
            let a_const = ka >= kb;
            let k = if a_const { ka } else { kb };
            let runlen: i64 = shape[..k].iter().product();
            let no = r - k;
            let nouter = (total as i64 / runlen) as usize;
            let kern = |o0: usize, o1: usize| {
                let mut idx = vec![0i64; no.max(1)];
                let (mut ia, mut ib): (i64, i64) = (0, 0);
                for _ in 0..o0 {
                    for i in 0..no {
                        let di = k + i;
                        idx[i] += 1;
                        ia += stra[di];
                        ib += strb[di];
                        if idx[i] < shape[di] {
                            break;
                        }
                        idx[i] = 0;
                        ia -= stra[di] * shape[di];
                        ib -= strb[di] * shape[di];
                    }
                }
                let opc = op.code();
                let b_dense_run = b_dense_inner;
                for o in o0..o1 {
                    let cv = if a_const {
                        a[ia as usize]
                    } else {
                        b[ib as usize]
                    };
                    let dense = if a_const {
                        &b[ib as usize..]
                    } else {
                        &a[ia as usize..]
                    };
                    // SAFETY: 输出 run [o*runlen, (o+1)*runlen) 两两不相交；
                    // b 稠密时 dense 从 ib 起有 runlen 个元素（步长 1）。
                    if a_const {
                        // dense 切片头 = b.as_ptr()+ib，而内核内部还会 add(ib)——
                        // 传全量头 b.as_ptr()，偏移只在内核里做一次
                        let b_head = b.as_ptr();
                        let _ = dense;
                        // SAFETY: 输出 run [o*runlen,(o+1)*runlen) 不相交；
                        // b 稠密时 b_head+ib 起有 runlen 个元素（步长 1）。
                        unsafe {
                            crate::arch::binary_run_alloc(
                                yp.get(),
                                &cv,
                                b_head,
                                o,
                                runlen as usize,
                                ib as usize,
                                b_dense_run,
                                cv,
                                true,
                                opc,
                            )
                        };
                        // 尾部推进
                        for i in 0..no {
                            let di = k + i;
                            idx[i] += 1;
                            ia += stra[di];
                            ib += strb[di];
                            if idx[i] < shape[di] {
                                break;
                            }
                            idx[i] = 0;
                            ia -= stra[di] * shape[di];
                            ib -= strb[di] * shape[di];
                        }
                        continue;
                    }
                    // SAFETY: 输出 run [o*runlen,(o+1)*runlen) 不相交。
                    let dst = unsafe { yp.offset(o * runlen as usize).slice(runlen as usize) };
                    for (j, d) in dst.iter_mut().enumerate() {
                        let (va, vb) = if a_const {
                            (cv, dense[j])
                        } else {
                            (dense[j], cv)
                        };
                        *d = op.apply(va, vb);
                    }
                    for i in 0..no {
                        let di = k + i;
                        idx[i] += 1;
                        ia += stra[di];
                        ib += strb[di];
                        if idx[i] < shape[di] {
                            break;
                        }
                        idx[i] = 0;
                        ia -= stra[di] * shape[di];
                        ib -= strb[di] * shape[di];
                    }
                }
            };
            par::parallel_for_elems(nouter, total, kern);
            return (y, out_shape);
        }
    }

    // 通用路径：odometer 走法（每元素 O(1) 摊还，代替每维一次取模+除法）
    let scalar_kernel = |b0: usize, e0: usize| {
        let mut idx = vec![0i64; r];
        let (mut ia, mut ib): (i64, i64) = (0, 0);
        // 推进 odometer 到 b0（ 原样：从 0 起推，每块一次）
        for _ in 0..b0 {
            for i in 0..r {
                idx[i] += 1;
                ia += stra[i];
                ib += strb[i];
                if idx[i] < shape[i] {
                    break;
                }
                idx[i] = 0;
                ia -= stra[i] * shape[i];
                ib -= strb[i] * shape[i];
            }
        }
        for lin in b0 as i64..e0 as i64 {
            // SAFETY: 元素 lin 与其他并行块不相交。
            unsafe { *yp.get().add(lin as usize) = op.apply(a[ia as usize], b[ib as usize]) };
            for i in 0..r {
                idx[i] += 1;
                ia += stra[i];
                ib += strb[i];
                if idx[i] < shape[i] {
                    break;
                }
                idx[i] = 0;
                ia -= stra[i] * shape[i];
                ib -= strb[i] * shape[i];
            }
        }
    };
    if nobc {
        // SyncPtr 是 *mut 语义；这里指向只读数据，cast 后包装（不可变访问）。
        let (pa_src, pb_src) = (
            par::SyncPtr::new(a.as_ptr() as *mut f32),
            par::SyncPtr::new(b.as_ptr() as *mut f32),
        );
        let opc = op.code();
        par::parallel_for_elems(total, total, |b0, e0| {
            // SAFETY: 区间 [b0, e0) 与其他并行块不相交；a/b 等长。
            unsafe { crate::arch::binary_flat(pa_src.get(), pb_src.get(), yp.get(), b0, e0, opc) };
        });
    } else {
        par::parallel_for_elems(total, total, scalar_kernel);
    }
    (y, out_shape)
}

// ================================================================ 区间标量判据
// arch 分发层的标量分支。op 码语义与 `BinOp::apply` 一致（pow 走 libm），
// 与向量内核的标量尾巴同一套表达式——三侧（AVX2/NEON/标量）逐位同值。

/// 按 op 码施加二元运算（0..4 = + - * / pow）。
#[inline]
pub(crate) fn apply_code(a: f32, b: f32, op: u8) -> f32 {
    match op {
        0 => a + b,
        1 => a - b,
        2 => a * b,
        3 => a / b,
        _ => a.powf(b),
    }
}

/// 同形平坦就地：`dst[i] op= src[i]`，区间 `[b, e)`。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交；`dst`/`src` 等长。
#[allow(unsafe_op_in_unsafe_fn)] // 裸指针循环：前置条件见 # Safety 段
pub(crate) unsafe fn binary_flat_inplace_scalar(
    dst: *mut f32,
    src: *const f32,
    b: usize,
    e: usize,
    op: u8,
) {
    for i in b..e {
        let r = apply_code(*dst.add(i), *src.add(i), op);
        *dst.add(i) = r;
    }
}

/// run 路径就地的区间段形式（NEON 内核的尾巴从段起点续跑）。
///
/// # Safety
///
/// `dst` 起的 `[j0, j1)` 与并发调用者不相交。
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_op_in_unsafe_fn)] // 裸指针循环：前置条件见 # Safety 段
pub(crate) unsafe fn binary_run_inplace_seg(
    dst: *mut f32,
    bsrc: *const f32,
    j0: usize,
    j1: usize,
    ib: usize,
    b_dense_run: bool,
    cv: f32,
    op: u8,
) {
    for j in j0..j1 {
        let d = *dst.add(j);
        let bv = if b_dense_run { *bsrc.add(ib + j) } else { cv };
        *dst.add(j) = apply_code(d, bv, op);
    }
}

/// run 路径就地：`dst[o*runlen + j] op= bsrc[ib + j]`（b 稠密）或 `op= cv`。
///
/// # Safety
///
/// 输出 run `[o*runlen, (o+1)*runlen)` 必须与并发调用者不相交。
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_op_in_unsafe_fn)] // 裸指针循环：前置条件见 # Safety 段
pub(crate) unsafe fn binary_run_inplace_scalar(
    dst: *mut f32,
    bsrc: *const f32,
    o: usize,
    runlen: usize,
    ib: usize,
    b_dense_run: bool,
    cv: f32,
    op: u8,
) {
    binary_run_inplace_seg(
        dst.add(o * runlen),
        bsrc,
        0,
        runlen,
        ib,
        b_dense_run,
        cv,
        op,
    );
}

/// 单元素广播：`py[i] = op(sv, v[i])` 或 `op(v[i], sv)`（`a_scalar` 定方向）。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交；`v` 同长。
#[allow(unsafe_op_in_unsafe_fn)] // 裸指针循环：前置条件见 # Safety 段
pub(crate) unsafe fn binary_scalar_bcast_scalar(
    v: *const f32,
    sv: f32,
    a_scalar: bool,
    py: *mut f32,
    b: usize,
    e: usize,
    op: u8,
) {
    for i in b..e {
        let x = *v.add(i);
        let (va, vb) = if a_scalar { (sv, x) } else { (x, sv) };
        *py.add(i) = apply_code(va, vb, op);
    }
}

/// run 路径分配的区间段形式（NEON 内核的尾巴从段起点续跑）。
///
/// # Safety
///
/// `dst` 起的 `[j0, j1)` 与并发调用者不相交。
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_op_in_unsafe_fn)] // 裸指针循环：前置条件见 # Safety 段
pub(crate) unsafe fn binary_run_alloc_seg(
    dst: *mut f32,
    pa_const: *const f32,
    bsrc: *const f32,
    j0: usize,
    j1: usize,
    ib: usize,
    b_dense_run: bool,
    cv: f32,
    a_const: bool,
    op: u8,
) {
    for j in j0..j1 {
        let bv = if b_dense_run { *bsrc.add(ib + j) } else { cv };
        let av = if a_const { *pa_const } else { *dst.add(j) };
        *dst.add(j) = apply_code(av, bv, op);
    }
}

/// run 路径分配：`py[o*runlen + j] = op(a 值, b[j] 或 cv)`；a 恒常量。
///
/// # Safety
///
/// 输出 run `[o*runlen, (o+1)*runlen)` 必须与并发调用者不相交。
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_op_in_unsafe_fn)] // 裸指针循环：前置条件见 # Safety 段
pub(crate) unsafe fn binary_run_alloc_scalar(
    py: *mut f32,
    pa_const: *const f32,
    bsrc: *const f32,
    o: usize,
    runlen: usize,
    ib: usize,
    b_dense_run: bool,
    cv: f32,
    a_const: bool,
    op: u8,
) {
    binary_run_alloc_seg(
        py.add(o * runlen),
        pa_const,
        bsrc,
        0,
        runlen,
        ib,
        b_dense_run,
        cv,
        a_const,
        op,
    );
}

/// 同形平坦：`y[i] = a[i] op b[i]`（写新缓冲），区间 `[b, e)`。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交；`pa`/`pb`/`py` 等长。
#[allow(unsafe_op_in_unsafe_fn)] // 裸指针循环：前置条件见 # Safety 段
pub(crate) unsafe fn binary_flat_scalar(
    pa: *const f32,
    pb: *const f32,
    py: *mut f32,
    b: usize,
    e: usize,
    op: u8,
) {
    for i in b..e {
        let r = apply_code(*pa.add(i), *pb.add(i), op);
        *py.add(i) = r;
    }
}
