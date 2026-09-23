// GEMM A/B — Rust 侧。逐句对等 bench.cpp：同样的 4 行 x 4 个 AVX 向量、
// 同样的 k 内层 broadcast+FMA 顺序、同样的 N-panel 32 列、同样的 tail 处理。
// 用裸指针而不是 slice 下标，理由和 C++ 用 float* 一样：不要把 bounds check
// 混进这个对比里 —— 这一条本身就是评估结论的一部分。
#![allow(clippy::needless_range_loop)]

use std::arch::x86_64::*;
use std::time::Instant;

#[inline(always)]
unsafe fn store_row(c: *mut f32, nn: usize, full: bool, v0: __m256, v1: __m256, v2: __m256, v3: __m256) {
    if full {
        _mm256_storeu_ps(c, v0);
        _mm256_storeu_ps(c.add(8), v1);
        _mm256_storeu_ps(c.add(16), v2);
        _mm256_storeu_ps(c.add(24), v3);
    } else {
        let mut t = [0f32; 32];
        _mm256_storeu_ps(t.as_mut_ptr(), v0);
        _mm256_storeu_ps(t.as_mut_ptr().add(8), v1);
        _mm256_storeu_ps(t.as_mut_ptr().add(16), v2);
        _mm256_storeu_ps(t.as_mut_ptr().add(24), v3);
        for n in 0..nn {
            *c.add(n) = t[n];
        }
    }
}

unsafe fn sgemm_kernel(a_mat: *const f32, b_mat: *const f32, c_mat: *mut f32, m: usize, n: usize, k: usize, ldc: usize) {
    let mut n0 = 0usize;
    while n0 < n {
        let nn = std::cmp::min(32, n - n0);
        let full = nn == 32;
        let mut row = 0usize;
        while row + 4 <= m {
            let a0 = a_mat.add((row + 0) * k);
            let a1 = a_mat.add((row + 1) * k);
            let a2 = a_mat.add((row + 2) * k);
            let a3 = a_mat.add((row + 3) * k);
            let z = _mm256_setzero_ps();
            let (mut c00, mut c01, mut c02, mut c03) = (z, z, z, z);
            let (mut c10, mut c11, mut c12, mut c13) = (z, z, z, z);
            let (mut c20, mut c21, mut c22, mut c23) = (z, z, z, z);
            let (mut c30, mut c31, mut c32, mut c33) = (z, z, z, z);
            for kk in 0..k {
                let b = b_mat.add(kk * n + n0);
                let bv0 = _mm256_loadu_ps(b);
                let bv1 = if nn > 8 { _mm256_loadu_ps(b.add(8)) } else { z };
                let bv2 = if nn > 16 { _mm256_loadu_ps(b.add(16)) } else { z };
                let bv3 = if nn > 24 { _mm256_loadu_ps(b.add(24)) } else { z };
                let av = _mm256_broadcast_ss(&*a0.add(kk));
                c00 = _mm256_fmadd_ps(av, bv0, c00); c01 = _mm256_fmadd_ps(av, bv1, c01);
                c02 = _mm256_fmadd_ps(av, bv2, c02); c03 = _mm256_fmadd_ps(av, bv3, c03);
                let av = _mm256_broadcast_ss(&*a1.add(kk));
                c10 = _mm256_fmadd_ps(av, bv0, c10); c11 = _mm256_fmadd_ps(av, bv1, c11);
                c12 = _mm256_fmadd_ps(av, bv2, c12); c13 = _mm256_fmadd_ps(av, bv3, c13);
                let av = _mm256_broadcast_ss(&*a2.add(kk));
                c20 = _mm256_fmadd_ps(av, bv0, c20); c21 = _mm256_fmadd_ps(av, bv1, c21);
                c22 = _mm256_fmadd_ps(av, bv2, c22); c23 = _mm256_fmadd_ps(av, bv3, c23);
                let av = _mm256_broadcast_ss(&*a3.add(kk));
                c30 = _mm256_fmadd_ps(av, bv0, c30); c31 = _mm256_fmadd_ps(av, bv1, c31);
                c32 = _mm256_fmadd_ps(av, bv2, c32); c33 = _mm256_fmadd_ps(av, bv3, c33);
            }
            store_row(c_mat.add((row + 0) * ldc + n0), nn, full, c00, c01, c02, c03);
            store_row(c_mat.add((row + 1) * ldc + n0), nn, full, c10, c11, c12, c13);
            store_row(c_mat.add((row + 2) * ldc + n0), nn, full, c20, c21, c22, c23);
            store_row(c_mat.add((row + 3) * ldc + n0), nn, full, c30, c31, c32, c33);
            row += 4;
        }
        while row < m {
            let a = a_mat.add(row * k);
            let c = c_mat.add(row * ldc + n0);
            let z = _mm256_setzero_ps();
            let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
            for kk in 0..k {
                let b = b_mat.add(kk * n + n0);
                let av = _mm256_broadcast_ss(&*a.add(kk));
                c0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b), c0);
                if nn > 8 { c1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b.add(8)), c1); }
                if nn > 16 { c2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b.add(16)), c2); }
                if nn > 24 { c3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b.add(24)), c3); }
            }
            store_row(c, nn, full, c0, c1, c2, c3);
            row += 1;
        }
        n0 += 32;
    }
}

fn main() {
    let shapes: [(usize, usize, usize); 4] = [
        (3136, 64, 576),
        (1024, 256, 256),
        (512, 256, 576),
        (64, 64, 64),
    ];
    for (m, n, k) in shapes {
        let mut a = vec![0.001f32; m * k];
        let mut b = vec![0.002f32; k * n];
        let mut c = vec![0f32; m * n];
        let mut i = 0;
        while i < a.len() { a[i] = 0.5; i += 7; }
        let mut i = 0;
        while i < b.len() { b[i] = 0.25; i += 11; }

        let mut best = f64::MAX;
        for _ in 0..7 {
            let t0 = Instant::now();
            unsafe { sgemm_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, n) };
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if ms < best { best = ms; }
        }
        let gflops = 2.0 * (m * n * k) as f64 / (best * 1e6);
        println!("{:<18} M={:<5} N={:<4} K={:<4}  best {:8.3} ms  {:7.2} GFLOPS   chk={:.4}",
                 "rust", m, n, k, best, gflops, c[m / 2 * n + n / 2]);
    }
}
