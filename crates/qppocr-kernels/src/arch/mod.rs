//! 内核分发层：每个 SIMD 内核调用点的唯一入口。
//!
//! 这里是 crate 里**唯一**出现 `target_arch` 接线的地方：每个入口先问
//! 当前后端（AVX2 / NEON / 标量，含 [`crate::force_backend`] 的强制覆盖），
//! 选中的后端不存在时落到标量参考实现。内核文件（`gemm`、`conv`、……）
//! 因此完全不含架构条件——新增一个架构后端 = 一个后端文件 + 本模块每个
//! 入口一行。
//!
//! 位级契约：三个后端对每个输出元素执行**同一串运算**——GEMM 的 k 升序
//! FMA 链、bias 的加法位置、softmax 水平归约的结合树，逐条在各后端的
//! 实现里镜像。`tests/bitexact.rs` 按架构自动选 SIMD 侧做逐位对拍，
//! 标量实现永远是判据。
//!
//! # Safety
//!
//! 各入口的 `# Safety` 前置条件与其微内核形态一致：输入区间长度由参数
//! 给出、输出区间与其他并行块不相交、形状已由上层（`sgemm`/`conv2d`/
//! ……）的断言校验。标量分支的前置条件相同。

// 本模块全部是 unsafe 分发体，前置条件由各入口的 `# Safety` 段声明、
// 由上层调用方保证；函数体内不再逐调用包 unsafe（与 `x86` 模块同款约定）。
#![allow(unsafe_op_in_unsafe_fn)]

// ================================================================ 选择器

/// 本次调用是否走 AVX2 内核（非 x86-64 编译期折叠为 false）。
#[allow(dead_code)] // 分发臂在非 x86_64 目标上不引用它（编译期即知无 AVX2）
#[inline]
fn avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        crate::current_backend() == crate::Backend::Avx2
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// 本次调用是否走 NEON 内核（非 aarch64 编译期折叠为 false）。
/// f32 NEON 是 aarch64 基线指令集，无需运行时探测。
#[allow(dead_code)] // 分发臂在非 aarch64 目标上不引用它（编译期即知无 NEON）
#[inline]
fn neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        crate::current_backend() == crate::Backend::Neon
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// 从裸指针还原 bias 切片（`m` 行）；空指针 = 无 bias。
#[inline]
fn bias_slice(bias: *const f32, m: usize) -> Option<&'static [f32]> {
    if bias.is_null() {
        None
    } else {
        // SAFETY: 分发层的调用方（sgemm/conv2d）已断言 bias 长度 ≥ m。
        Some(unsafe { std::slice::from_raw_parts(bias, m) })
    }
}

// ================================================================ sgemm

/// sgemm 面板体：`[pb, pe)` 号 N 面板（每面板 32 列），4 行块寄存器累加，
/// k 升序 FMA 链、bias 在 store 阶段后置（M%4 尾行的非整面板分支除外——
/// 那条以 bias 起种，见 `gemm` 模块头的位级说明）。
///
/// # Safety
///
/// `c` 的列区间 `[p*32, p*32+nn)`（所有行）必须与并发调用者不相交；
/// `a` ≥ m·k、`b` ≥ k·n、`bias` ≥ m（非空时）。
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_panel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    bias: *const f32,
    pb: usize,
    pe: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        crate::x86::sgemm_panel_avx2(a, b, c, m, n, k, ldc, bias, pb, pe);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() {
        crate::aarch64::sgemm_panel_neon(a, b, c, m, n, k, ldc, bias, pb, pe);
        return;
    }
    // SAFETY: 形状契约同上；切片由参数重建，长度即契约长度。
    unsafe {
        crate::gemm::panel_body(
            std::slice::from_raw_parts(a, m * k),
            std::slice::from_raw_parts(b, k * n),
            c,
            m,
            n,
            k,
            ldc,
            bias_slice(bias, m),
            pb,
            pe,
        )
    };
}

/// implicit-GEMM 面板体：`bptrs` 每个 k 一行一个指针（行内步长 `SW`），
/// 与 [`sgemm_panel`] 同一 k 序、同一 bias 位置。
///
/// # Safety
///
/// 同 [`sgemm_panel`]；`bptrs` ≥ k，各指针指向可读的 slab（含右侧 slack）。
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_panel_bptrs<const SW: usize>(
    a: *const f32,
    bptrs: *const *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    bias: *const f32,
    pb: usize,
    pe: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        crate::x86::sgemm_panel_bptrs_avx2::<SW>(a, bptrs, c, m, n, k, ldc, bias, pb, pe);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() {
        crate::aarch64::sgemm_panel_bptrs_neon::<SW>(a, bptrs, c, m, n, k, ldc, bias, pb, pe);
        return;
    }
    // SAFETY: 形状契约同上。
    unsafe { crate::gemm::sgemm_panel_bptrs_scalar::<SW>(a, bptrs, c, m, n, k, ldc, bias, pb, pe) };
}

/// sgemm 窄 N 路径：整行计算，K 上外积。32 列主体 bias 后置，
/// N%32 尾列以 bias 起种。
///
/// # Safety
///
/// 行区间 `[mb, me)` 必须与并发调用者不相交。
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_mrows(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    bias: *const f32,
    mb: usize,
    me: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        crate::x86::sgemm_mrows_avx2(a, b, c, n, k, ldc, bias, mb, me);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() {
        crate::aarch64::sgemm_mrows_neon(a, b, c, n, k, ldc, bias, mb, me);
        return;
    }
    // SAFETY: 行区间契约同上；切片长度由形状给出。
    unsafe {
        crate::gemm::m_rows_body(
            std::slice::from_raw_parts(a, m * k),
            std::slice::from_raw_parts(b, k * n),
            c,
            m,
            n,
            k,
            ldc,
            bias_slice(bias, m),
            mb,
            me,
        )
    };
}

// ================================================================ conv

/// 深度卷积一个 (n, channel) 输出平面，输入是已补零平面（无边界判断）。
/// 累加顺序 ky 外层、kx 内层、从 0 起，bias 最后加。
///
/// # Safety
///
/// `yc` 指向 `oh*ow` 个可写元素；`xp` 指向 `((oh-1)*sh+kh)*pwidth` 个可读
/// 元素；`wc` 指向 `kh*kw` 个可读元素。
#[allow(clippy::too_many_arguments)]
pub unsafe fn depthwise_plane_padded(
    yc: *mut f32,
    xp: *const f32,
    wc: *const f32,
    oh: usize,
    ow: usize,
    pwidth: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    bias: f32,
) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        crate::x86::depthwise_plane_padded_avx2(yc, xp, wc, oh, ow, pwidth, kh, kw, sh, sw, bias);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() {
        crate::aarch64::depthwise_plane_padded_neon(
            yc, xp, wc, oh, ow, pwidth, kh, kw, sh, sw, bias,
        );
        return;
    }
    // SAFETY: 平面长度契约同上（pheight 由输出形状与核高推出）。
    unsafe {
        crate::conv::depthwise_plane_scalar(
            std::slice::from_raw_parts_mut(yc, oh * ow),
            std::slice::from_raw_parts(xp, ((oh - 1) * sh + kh) * pwidth),
            std::slice::from_raw_parts(wc, kh * kw),
            oh,
            ow,
            pwidth,
            kh,
            kw,
            sh,
            sw,
            bias,
        )
    };
}

/// ConvTranspose 的行内积：输入列 `[j0, j1)` 的每个元素散射出相邻两个
/// 输出（kx 奇偶），c 通道升序 FMA 链。向量后端按本架构步长分批，
/// 不足一批的尾巴逐元素。
///
/// # Safety
///
/// `xr` 起有 `c*ch_plane` 个可读元素；`w0`/`w1` 各 `c` 个；`orow` 起有
/// `2*(j1-j0)` 个可写元素（位于输出行内）。
#[allow(clippy::too_many_arguments)]
pub unsafe fn convt_rows(
    xr: *const f32,
    ch_plane: usize,
    w0: *const f32,
    w1: *const f32,
    c: usize,
    j0: usize,
    j1: usize,
    orow: *mut f32,
) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        let mut j = j0;
        while j + 8 <= j1 {
            crate::x86::convt_row_vec(xr, ch_plane, w0, w1, c, j, orow);
            j += 8;
        }
        crate::conv::convt_row_scalar(xr, ch_plane, w0, w1, c, j, j1, orow);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() {
        let mut j = j0;
        while j + 4 <= j1 {
            crate::aarch64::convt_row_neon(xr, ch_plane, w0, w1, c, j, orow);
            j += 4;
        }
        crate::conv::convt_row_scalar(xr, ch_plane, w0, w1, c, j, j1, orow);
        return;
    }
    crate::conv::convt_row_scalar(xr, ch_plane, w0, w1, c, j0, j1, orow);
}

// ================================================================ 激活（区间）

/// `y = max(0, x)`，就地，区间 `[b, e)`。NaN→0。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交。
pub unsafe fn relu_seg(t: *mut f32, b: usize, e: usize) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        crate::x86::relu_vec(t, b, e);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() {
        crate::aarch64::relu_vec(t, b, e);
        return;
    }
    crate::activation::relu_seg_scalar(t, b, e);
}

/// `y = clip(0, 1, fma(x, alpha, beta))`。clamp 顺序 max(0, min(1, v))。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交；`x`/`y` 等长。
#[allow(clippy::too_many_arguments)]
pub unsafe fn hardsigmoid_seg(
    x: *const f32,
    y: *mut f32,
    b: usize,
    e: usize,
    alpha: f32,
    beta: f32,
) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        crate::x86::hardsigmoid_vec(x, y, b, e, alpha, beta);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() {
        crate::aarch64::hardsigmoid_vec(x, y, b, e, alpha, beta);
        return;
    }
    crate::activation::hardsigmoid_seg_scalar(x, y, b, e, alpha, beta);
}

/// `y = 1 / (1 + exp(-x))`，区间 `[b, e)`。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交；`x`/`y` 等长。
pub unsafe fn sigmoid_seg(x: *const f32, y: *mut f32, b: usize, e: usize) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        crate::x86::sigmoid_vec(x, y, b, e);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() {
        crate::aarch64::sigmoid_vec(x, y, b, e);
        return;
    }
    crate::activation::sigmoid_seg_scalar(x, y, b, e);
}

/// `t = c3·t·(erf(t/c1) + c2)`，就地（融合 GELU），区间 `[b, e)`。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交。
#[allow(clippy::too_many_arguments)]
pub unsafe fn gelu_seg(t: *mut f32, b: usize, e: usize, c1: f32, c2: f32, c3: f32) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        crate::x86::gelu_vec(t, b, e, c1, c2, c3);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() {
        crate::aarch64::gelu_vec(t, b, e, c1, c2, c3);
        return;
    }
    crate::activation::gelu_seg_scalar(t, b, e, c1, c2, c3);
}

/// `t = clip(t, lo, hi)`，就地，区间 `[b, e)`。NaN 语义与标量版对齐。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交。
pub unsafe fn clip_seg(t: *mut f32, b: usize, e: usize, lo: f32, hi: f32) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        crate::x86::clip_vec(t, b, e, lo, hi);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() {
        crate::aarch64::clip_vec(t, b, e, lo, hi);
        return;
    }
    crate::activation::clip_seg_scalar(t, b, e, lo, hi);
}

/// 最后一维 softmax 的一行：max / exp / 水平归约按固定的 lane 结构与
/// 结合树（`((v0+v4)+(v1+v5)) + ((v2+v6)+(v3+v7))`）。行宽 ≥ 8 才有
/// 向量路径，否则标量参考实现。
///
/// # Safety
///
/// `row` 指向 `inner` 个可读写元素，且该行与其他并行块不相交。
pub unsafe fn softmax_row(row: *mut f32, inner: usize) {
    #[cfg(target_arch = "x86_64")]
    if avx2() && inner >= 8 {
        crate::x86::softmax_row_vec(row, inner);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() && inner >= 8 {
        crate::aarch64::softmax_row_neon(row, inner);
        return;
    }
    crate::activation::softmax_row_scalar(row, inner);
}

// ================================================================ 二元

/// 同形平坦就地：`dst[i] op= src[i]`，区间 `[b, e)`。op 码：0..3 = + - * /
///（4 = pow 无向量路径，恒标量）。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交；`dst`/`src` 等长。
pub unsafe fn binary_flat_inplace(dst: *mut f32, src: *const f32, b: usize, e: usize, op: u8) {
    #[cfg(target_arch = "x86_64")]
    if avx2() && op <= 3 {
        crate::x86::binary_flat_inplace_vec(dst, src, b, e, op);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() && op <= 3 {
        crate::aarch64::binary_flat_inplace_vec(dst, src, b, e, op);
        return;
    }
    crate::elementwise::binary_flat_inplace_scalar(dst, src, b, e, op);
}

/// run 路径就地：`dst[o*runlen + j] op= bsrc[ib + j]`（b 稠密）或
/// `op= cv`（b 常量广播），`j ∈ [0, runlen)`。
///
/// # Safety
///
/// 输出 run `[o*runlen, (o+1)*runlen)` 必须与并发调用者不相交；b 稠密时
/// `bsrc + ib` 起有 `runlen` 个元素。
#[allow(clippy::too_many_arguments)]
pub unsafe fn binary_run_inplace(
    dst: *mut f32,
    bsrc: *const f32,
    o: usize,
    runlen: usize,
    ib: usize,
    b_dense_run: bool,
    cv: f32,
    op: u8,
) {
    #[cfg(target_arch = "x86_64")]
    if avx2() && op <= 3 {
        crate::x86::binary_run_inplace_vec(dst, bsrc, o, runlen, ib, b_dense_run, cv, op);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() && op <= 3 {
        crate::aarch64::binary_run_inplace_vec(dst, bsrc, o, runlen, ib, b_dense_run, cv, op);
        return;
    }
    crate::elementwise::binary_run_inplace_scalar(dst, bsrc, o, runlen, ib, b_dense_run, cv, op);
}

/// 单元素广播：`py[i] = op(sv, v[i])` 或 `op(v[i], sv)`（`a_scalar` 决定
/// 方向），区间 `[b, e)`。
///
/// # Safety
///
/// 区间 `[b, e)` 必须与并发调用者不相交；`v` 同长。
pub unsafe fn binary_scalar_bcast(
    v: *const f32,
    sv: f32,
    a_scalar: bool,
    py: *mut f32,
    b: usize,
    e: usize,
    op: u8,
) {
    #[cfg(target_arch = "x86_64")]
    if avx2() && op <= 3 {
        crate::x86::binary_scalar_vec(v, sv, a_scalar, py, b, e, op);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() && op <= 3 {
        crate::aarch64::binary_scalar_vec(v, sv, a_scalar, py, b, e, op);
        return;
    }
    crate::elementwise::binary_scalar_bcast_scalar(v, sv, a_scalar, py, b, e, op);
}

/// run 路径分配：`py[o*runlen + j] = op(a 值, b[j] 或 cv)`；a 恒常量。
///
/// # Safety
///
/// 输出 run `[o*runlen, (o+1)*runlen)` 必须与并发调用者不相交；b 稠密时
/// `bsrc + ib` 起有 `runlen` 个元素。
#[allow(clippy::too_many_arguments)]
pub unsafe fn binary_run_alloc(
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
    #[cfg(target_arch = "x86_64")]
    if avx2() && op <= 3 {
        crate::x86::binary_run_alloc_vec(
            py,
            pa_const,
            bsrc,
            o,
            runlen,
            ib,
            b_dense_run,
            cv,
            a_const,
            op,
        );
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() && op <= 3 {
        crate::aarch64::binary_run_alloc_vec(
            py,
            pa_const,
            bsrc,
            o,
            runlen,
            ib,
            b_dense_run,
            cv,
            a_const,
            op,
        );
        return;
    }
    crate::elementwise::binary_run_alloc_scalar(
        py,
        pa_const,
        bsrc,
        o,
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
pub unsafe fn binary_flat(
    pa: *const f32,
    pb: *const f32,
    py: *mut f32,
    b: usize,
    e: usize,
    op: u8,
) {
    #[cfg(target_arch = "x86_64")]
    if avx2() && op <= 3 {
        crate::x86::binary_flat_vec(pa, pb, py, b, e, op);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() && op <= 3 {
        crate::aarch64::binary_flat_vec(pa, pb, py, b, e, op);
        return;
    }
    crate::elementwise::binary_flat_scalar(pa, pb, py, b, e, op);
}

// ================================================================ 池化 / resize / 膨胀

/// 池化 2x2 s1 的行内相：`out[j] = max(max(r0[j], r1[j]), max(r0[j+1],
/// r1[j+1]))`；`r1` 为 null 表示最后一行（钳制）。`vm` 是标量路径的
/// scratch 行（长度 w+1；向量路径不使用）。
///
/// # Safety
///
/// `r0`/`r1`（非空时）至少 `w` 个可读元素；`out` 至少 `w` 个可写元素。
pub unsafe fn pool2x2_row(r0: *const f32, r1: *const f32, vm: *mut f32, out: *mut f32, w: usize) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        crate::x86::pool2x2_row_vec(r0, r1, vm, out, w);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() {
        crate::aarch64::pool2x2_row_neon(r0, r1, vm, out, w);
        return;
    }
    crate::pool2d::pool2x2_row_scalar(r0, r1, vm, out, w);
}

/// resize_bilinear 的行内积：8 个输出列一批（NEON 为 4）。
/// `ix0/ix1/fx` 是预计算的源列映射，权重结合顺序固定（首项两乘，其余
/// fma）。
///
/// # Safety
///
/// 输出行区间与其他并行块不相交；映射表长度覆盖 `[ox0, ox1)`，源列界内。
#[allow(clippy::too_many_arguments)]
pub unsafe fn bilinear_row(
    r0: *const f32,
    r1: *const f32,
    ix0: *const i32,
    ix1: *const i32,
    fx: *const f32,
    ly: f32,
    dst: *mut f32,
    ox0: usize,
    ox1: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        crate::x86::bilinear_row_vec(r0, r1, ix0, ix1, fx, ly, dst, ox0, ox1);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() {
        crate::aarch64::bilinear_row_neon(r0, r1, ix0, ix1, fx, ly, dst, ox0, ox1);
        return;
    }
    crate::resize::bilinear_row_scalar(r0, r1, ix0, ix1, fx, ly, dst, ox0, ox1);
}

/// 2×2 最大值膨胀的行内核（`x ∈ [1, w)`）：`out[x] = max(cur[x-1],
/// cur[x], up[x-1], up[x])`（`up` 为 None = 第一行）。
///
/// # Safety
///
/// `cur` 至少 `w` 字节；`up` 非空时至少 `w` 字节；`out` 至少 `w` 字节。
pub unsafe fn dilate2x2_row(cur: &[u8], up: Option<&[u8]>, out: *mut u8, w: usize) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        crate::x86::dilate2x2_row_avx2(cur, up, out, w);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon() {
        crate::aarch64::dilate2x2_row_neon(cur, up, out, w);
        return;
    }
    crate::resize::dilate2x2_row_scalar(cur, up, out, w);
}
