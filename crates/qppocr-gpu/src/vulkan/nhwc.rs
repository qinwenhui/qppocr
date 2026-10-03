//! NHWC-f32 计划路径的纯数据变换（n_ 内核族的 Rust 侧配套）。
//!
//! - 激活布局：NHWC f32，C 补到 %4（vec4 粒度），pad 通道恒为 0
//!   （entry 零填充、conv 权重零填充 ⇒ pad 输出为 0）。
//! - 区域大小以 **u32 word = f32 元素**计——与 Layout/旧内核同一单位制；
//!   n_ 内核内部 `>>2` 折算 vec4。
//! - 权重装载期一次性重排（k-major / tap-major，见各 .comp 头注释）。

use qppocr_core::tensor::Tensor;

/// 通道数补到 %4。
pub(crate) fn cpad4(c: i64) -> u32 {
    ((c + 3) / 4) as u32 * 4
}

/// NHWC f32 张量 [N,C,H,W] 的 word 数（= 元素数，含 Cpad4）。
pub(crate) fn nhwc_words(shape: &[i64]) -> u32 {
    if shape.len() != 4 {
        panic!("n_ 路径只收 rank-4：{shape:?}");
    }
    (shape[0] as u64 * shape[2] as u64 * shape[3] as u64 * cpad4(shape[1]) as u64) as u32
}

/// 稠密 conv 权重 [Co,Ci,kh,kw] → k-major（n_conv.comp 的布局）：
/// f16vec4 下标 (t*cin4v + ci4)*4*nv + i*nv + n4，元素 = 4 个输出通道。
pub(crate) fn repack_conv_w(w: &[f32], co: usize, cin: usize, kh: usize, kw: usize) -> Vec<u32> {
    let (cop, cip) = (cpad4(co as i64) as usize, cpad4(cin as i64) as usize);
    let (cin4v, nv, taps) = (cip / 4, cop / 4, kh * kw);
    let mut v = vec![0f32; taps * cin4v * 4 * nv * 4];
    for t in 0..taps {
        let (ky, kx) = (t / kw, t % kw);
        for ci4 in 0..cin4v {
            for ci in ci4 * 4..ci4 * 4 + 4 {
                for n4 in 0..nv {
                    let base = (((t * cin4v + ci4) * 4 + (ci % 4)) * nv + n4) * 4;
                    for (l, co_i) in (n4 * 4..n4 * 4 + 4).enumerate() {
                        v[base + l] = if ci < cin && co_i < co {
                            w[co_i * cin * kh * kw + ci * kh * kw + ky * kw + kx]
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
    }
    v.iter().map(|f| f.to_bits()).collect()
}

/// depthwise 权重 [Co,1,kh,kw] → [t][c4]（n_conv_dw.comp 的布局）。
pub(crate) fn repack_dw_w(w: &[f32], c: usize, kh: usize, kw: usize) -> Vec<u32> {
    let cp = cpad4(c as i64) as usize;
    let cv4 = cp / 4;
    let mut v = vec![0f32; kh * kw * cv4 * 4];
    for t in 0..kh * kw {
        let (ky, kx) = (t / kw, t % kw);
        for c4 in 0..cv4 {
            for (l, ch) in (c4 * 4..c4 * 4 + 4).enumerate() {
                v[(t * cv4 + c4) * 4 + l] = if ch < c {
                    w[ch * kh * kw + ky * kw + kx]
                } else {
                    0.0
                };
            }
        }
    }
    v.iter().map(|f| f.to_bits()).collect()
}

/// ConvTranspose 权重 [Ci,Co,kh,kw] → 每 tap 一张 [Ci][nv]（n_convt.comp）。
pub(crate) fn repack_convt_w(w: &[f32], ci: usize, co: usize, kh: usize, kw: usize) -> Vec<u32> {
    let (cip, cop) = (cpad4(ci as i64) as usize, cpad4(co as i64) as usize);
    let nv = cop / 4;
    let mut v = vec![0f32; kh * kw * cip * nv * 4];
    for t in 0..kh * kw {
        let (ky, kx) = (t / kw, t % kw);
        for ci_i in 0..cip {
            for n4 in 0..nv {
                let base = ((t * cip + ci_i) * nv + n4) * 4;
                for (l, co_i) in (n4 * 4..n4 * 4 + 4).enumerate() {
                    v[base + l] = if ci_i < ci && co_i < co {
                        w[(ci_i * co + co_i) * kh * kw + ky * kw + kx]
                    } else {
                        0.0
                    };
                }
            }
        }
    }
    v.iter().map(|f| f.to_bits()).collect()
}

/// conv bias [Co] f32 → [CoPad4] words。
pub(crate) fn conv_bias(bias: &[f32], co: i64) -> Vec<u32> {
    let cop = cpad4(co) as usize;
    let mut v = vec![0f32; cop];
    let n = bias.len().min(cop);
    v[..n].copy_from_slice(&bias[..n]);
    v.iter().map(|f| f.to_bits()).collect()
}

/// 任意 f32 initializer → NHWC f32 words（Cpad4 补 0）。
/// gate 形 [N,C,1,1] 走同一路径（HW=1 退化成通道平铺）；rank<4
///（如 Resize 的 f32 scales [4]）按前置 1 补齐到 [1,C,1,1]。
pub(crate) fn init_to_nhwc(t: &Tensor) -> Vec<u32> {
    let mut s: Vec<i64> = t.shape.clone();
    if s.len() == 1 {
        // rank-1 = 通道向量：补成 [1, V, 1, 1]（NHWC 展开为平铺 V）。
        // front-pad 会得到 [1,1,1,V]——C=1/W=V，值散布到每 cpad 个位置，
        // 通道广播读到的 vec4 只有 lane0 有值（Add.104 实测踩过）。
        s.insert(0, 1);
        while s.len() < 4 {
            s.push(1);
        }
    } else {
        while s.len() < 4 {
            s.insert(0, 1);
        }
    }
    if s.len() > 4 {
        panic!("n_ 路径的 initializer 需 rank≤4：{s:?}");
    }
    let (nb, c, h, w) = (s[0] as usize, s[1] as usize, s[2] as usize, s[3] as usize);
    let cp = cpad4(s[1]) as usize;
    let hw = h * w;
    let mut v = vec![0f32; nb * hw * cp];
    for n in 0..nb {
        for pos in 0..hw {
            for ch in 0..c {
                v[n * hw * cp + pos * cp + ch] = t.f32[n * c * hw + ch * hw + pos];
            }
        }
    }
    v.iter().map(|f| f.to_bits()).collect()
}
