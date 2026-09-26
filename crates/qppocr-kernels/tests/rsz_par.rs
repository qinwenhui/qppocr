//! 并行图像缩放与标量参考的逐位一致性。
#[test]
fn resize_parallel_bitexact() {
    // 标量参考（与 core 的旧实现同式）
    fn scalar(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
        let mut dst = vec![0u8; dw * dh * 3];
        let (sx, sy) = (sw as f32 / dw as f32, sh as f32 / dh as f32);
        let (mut x0, mut x1, mut fx) = (vec![0usize; dw], vec![0usize; dw], vec![0f32; dw]);
        for x in 0..dw {
            let v = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
            let i0 = (v as usize).min(sw - 1);
            x0[x] = i0;
            x1[x] = (i0 + 1).min(sw - 1);
            fx[x] = v - i0 as f32;
        }
        for y in 0..dh {
            let v = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
            let j0 = (v as usize).min(sh - 1);
            let j1 = (j0 + 1).min(sh - 1);
            let fy = v - j0 as f32;
            let (r0, r1) = (&src[j0 * sw * 3..], &src[j1 * sw * 3..]);
            for x in 0..dw {
                let (a, b) = (x0[x] * 3, x1[x] * 3);
                let lx = fx[x];
                for ch in 0..3 {
                    let (p00, p01) = (r0[a + ch] as f32, r0[b + ch] as f32);
                    let (p10, p11) = (r1[a + ch] as f32, r1[b + ch] as f32);
                    let t = p00 * (1.0 - lx) * (1.0 - fy);
                    let t = (p01 * lx).mul_add(1.0 - fy, t);
                    let t = (p10 * (1.0 - lx)).mul_add(fy, t);
                    let val = (p11 * lx).mul_add(fy, t);
                    dst[(y * dw + x) * 3 + ch] = (val + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        }
        dst
    }
    let (sw, sh) = (848usize, 816usize);
    let src: Vec<u8> = (0..sw * sh * 3).map(|i| (i * 7919 % 256) as u8).collect();
    for &(dw, dh) in &[(832usize, 832usize), (400, 48), (960, 1216), (17, 33)] {
        let want = scalar(&src, sw, sh, dw, dh);
        let mut got = vec![0u8; dw * dh * 3];
        qppocr_kernels::resize::resize_bilinear_rgb_u8(&src, sw, sh, &mut got, dw, dh);
        assert_eq!(got, want, "resize {dw}x{dh} 与标量不一致");
    }
}

/// 2×2 膨胀的 AVX2 版 vs 标量参考逐位一致。
#[test]
fn dilate2x2_bitexact() {
    fn scalar(mask: &[u8], h: usize, w: usize) -> Vec<u8> {
        let mut d = vec![0u8; h * w];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let mut v = 0u8;
                for dy in -1..=0i32 {
                    for dx in -1..=0i32 {
                        let (ny, nx) = (y + dy, x + dx);
                        if ny < 0 || ny >= h as i32 || nx < 0 || nx >= w as i32 {
                            continue;
                        }
                        v = v.max(mask[ny as usize * w + nx as usize]);
                    }
                }
                d[y as usize * w + x as usize] = v;
            }
        }
        d
    }
    for &(h, w) in &[(832usize, 832usize), (37, 129), (1, 1), (5, 300), (300, 5)] {
        let mask: Vec<u8> = (0..h * w)
            .map(|i| ((i * 2654435761usize) >> 24) as u8 % 3)
            .collect();
        let mut got = vec![0u8; h * w];
        qppocr_kernels::resize::dilate2x2_max(&mask, &mut got, h, w);
        assert_eq!(got, scalar(&mask, h, w), "dilate {h}x{w}");
    }
}
