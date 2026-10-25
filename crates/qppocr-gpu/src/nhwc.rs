//! NHWC 计划路径的纯数据变换（n_ 内核族的 Rust 侧配套）。
//!
//! - 激活布局：NHWC f32，C 补到 %4（vec4 粒度），pad 通道恒为 0
//!   （entry 零填充、conv 权重零填充 ⇒ pad 输出为 0）。
//! - 区域大小以 **u32 word** 计——与 Layout/旧内核同一单位制；
//!   n_ 内核内部 `>>2` 折算 vec4。
//! - 权重装载期一次性重排（k-major / tap-major，见各 .comp 头注释）。
//!
//! # 权重的 fp16 存储（`*_f16` 那几个）
//!
//! 权重可以按 fp16 存，**激活与累加仍是 f32**——只换权重的字节宽度。
//! 这不是「加一个 f16 张量类型」：`DType`、解析器、模型文件、CPU 后端
//! 全都不动，转换只发生在这里（重排的那一刻），与竞品在装载期做
//! `(Half)f32[...]` 是同一件事。
//!
//! 实测动机见 `bench/gpuhealth/logs/` 的权重装载归因：`n_conv8` 的热点
//! 1×1 上权重装载占 58%，而 fp16 把这段足迹砍半，探针实测拿到 1.7~2.6×
//! （吃掉整个权重装载成本的 73~88%）。
//!
//! **索引语义必须与 f32 版逐位对应**：f32 下一个 `float4` 覆盖的 4 个
//! 通道，与 fp16 下一个 `half4` 覆盖的是同样的 4 个。于是内核侧的下标
//! 表达式一字不动，只有基址的折算从 `>>2`（float4）变 `>>1`（half4）——
//! 这正是探针 `p_whalf` 的单变量构造，也是这条路的正确性依据。

use qppocr_core::tensor::Tensor;

/// 通道数补到 %4。
pub fn cpad4(c: i64) -> u32 {
    ((c + 3) / 4) as u32 * 4
}

/// MMA 权重段的**尾垫**（word 数）：`n_conv8_mma*` 的 B 分片是 8 列宽，
/// 列块宽 32。当 `cpad4(Co)` 不是 32 的整数倍时，最后一块的分片会读到
/// 权重行尾之后——
///
/// - 行内越读落到同一张表的下一行（别的 k 的权重，有限值）；
/// - **表尾越读**落到权重段之后，那一段必须存在且是定值，否则可能读到
///   arena 里的任意位型（含 NaN）或越出缓冲。
///
/// 垫 `MMA_W_TAIL` 个 0 把第二种兜住。最大越读量：列块基
/// `(nblk-1)*32`、分片列偏移最大 24（`sgid&1` 的 16 + `j*8` 的 8），
/// 加 8 列宽 → 最多越过行尾 32 个 float。
///
/// **布局透明**：只在 blob 末尾追加，任何合法下标处的字节一字不变，
/// 所以同一份 `w_off` 仍可被 `n_conv8` 原样读（它的行 stride 是
/// `cpad4(Co)`，与垫之前完全相同）。
pub const MMA_W_TAIL: usize = 32;

/// NHWC f32 张量 [N,C,H,W] 的 word 数（= 元素数，含 Cpad4）。
pub fn nhwc_words(shape: &[i64]) -> u32 {
    if shape.len() != 4 {
        panic!("n_ 路径只收 rank-4：{shape:?}");
    }
    (shape[0] as u64 * shape[2] as u64 * shape[3] as u64 * cpad4(shape[1]) as u64) as u32
}

/// 稠密 conv 权重 [Co,Ci,kh,kw] → k-major（n_conv.comp 的布局）：
/// f32vec4 下标 (t*cin4v + ci4)*4*nv + i*nv + n4，元素 = 4 个输出通道。
pub fn repack_conv_w(w: &[f32], co: usize, cin: usize, kh: usize, kw: usize) -> Vec<u32> {
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

/// 同 [`repack_conv_w`]，但把权重压成 fp16。
///
/// 返回值是 f32 版的**一半长**（两个 half 挤进一个 word）：`n_conv8h`
/// 读权重时把它当 `device half4*`，一个 half4 = 8 B = 2 个 word，覆盖
/// 的 4 个通道与 f32 版的 `float4` 完全相同。
///
/// 静态段因此直接减半——arena 更小、计划重建时的主机侧上传也更少。
/// 段起始恒 16 B 对齐（见 `Layout::alloc_static`），所以内核侧
/// `w_off >> 1` 折 half4 下标是精确的，**不需要额外的对齐padding**。
pub fn repack_conv_w_f16(w: &[f32], co: usize, cin: usize, kh: usize, kw: usize) -> Vec<u32> {
    // 先把重排做出来（k-major 索引语义只有一份实现，不复制第二遍），
    // 再整体压成 half。两段分离是为了「换精度」这件事不可能悄悄改到
    // 布局——布局由 `repack_conv_w` 独家负责。
    pack_f16_words(&repack_conv_w(w, co, cin, kh, kw))
}

/// 4 B/元素 的 word 序列 → 2 B/元素：word `j` = (元素 `2j`, 元素 `2j+1`)。
///
/// 低位是前一个元素——`half4` 按地址递增读 `.x/.y/.z/.w`，于是 half4
/// 下标 X 恰好覆盖元素 `4X..4X+3`，与 f32 的 float4 下标 X 对齐。
fn pack_f16_words(words: &[u32]) -> Vec<u32> {
    debug_assert!(words.len() % 2 == 0, "fp16 打包要求元素数为偶");
    let mut out = Vec::with_capacity(words.len() / 2);
    for c in words.chunks_exact(2) {
        let lo = f32_to_f16_bits(f32::from_bits(c[0])) as u32;
        let hi = f32_to_f16_bits(f32::from_bits(c[1])) as u32;
        out.push(lo | (hi << 16));
    }
    out
}

/// f32 → IEEE binary16 位型（就近偶数）。
///
/// 自己写而不是用标准库：**Rust 1.97 的 `f16` 还是 unstable**
/// （`E0658`，issue #116909），且仓库不引第三方 half 依赖。正确性由
/// `f32_to_f16_matches_ieee_reference` 对 CPython 的 `struct.pack('<e', v)`
/// 期望值逐条核对——**不自己算一遍再自己对**。
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mant = bits & 0x007f_ffff;
    let exp = ((bits >> 23) & 0xff) as i32;
    if exp == 0xff {
        // inf / NaN：NaN 保留“非零尾数”这一位，免得把 NaN 洗成 inf。
        let m = if mant != 0 { 0x0200 | (mant >> 13) as u16 } else { 0 };
        return sign | 0x7c00 | m;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // 超出 f16 范围 → inf（与硬件一致）
    }
    if e <= 0 {
        if e < -10 {
            return sign; // 小到连次正规都不够 → 0
        }
        let m = mant | 0x0080_0000; // 补回隐含首位
        let shift = (14 - e) as u32; // 14..=24
        let mut q = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (q & 1) == 1) {
            q += 1; // 进位到 0x400 时自然滚成最小正规数，位型正确
        }
        return sign | q as u16;
    }
    let mut h = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1; // 进位溢出到指数域是正确行为
    }
    sign | h as u16
}

/// depthwise 权重 [Co,1,kh,kw] → [t][c4]（n_conv_dw.comp 的布局）。
pub fn repack_dw_w(w: &[f32], c: usize, kh: usize, kw: usize) -> Vec<u32> {
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
pub fn repack_convt_w(w: &[f32], ci: usize, co: usize, kh: usize, kw: usize) -> Vec<u32> {
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
pub fn conv_bias(bias: &[f32], co: i64) -> Vec<u32> {
    let cop = cpad4(co) as usize;
    let mut v = vec![0f32; cop];
    let n = bias.len().min(cop);
    v[..n].copy_from_slice(&bias[..n]);
    v.iter().map(|f| f.to_bits()).collect()
}

/// 任意 f32 initializer → NHWC f32 words（Cpad4 补 0）。
/// gate 形 [N,C,1,1] 走同一路径（HW=1 退化成通道平铺）；rank<4
///（如 Resize 的 f32 scales [4]）按前置 1 补齐到 [1,C,1,1]。
pub fn init_to_nhwc(t: &Tensor) -> Vec<u32> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 期望值由 CPython 的 `struct.pack('<e', v)` 生成（CPython 实现的就是
    /// IEEE binary16 就近偶数）——**不自己算一遍再自己对**，那样等于没测。
    ///
    /// 这条极其关键：`p_whalf` 探针里 fp16 的误差读数是 1.5e-5（≈ 纯 fp16
    /// 舍入的量级，而 `p_base` 是 3e-8）。**只有转换本身是对的**，那个
    /// 1.7~2.6× 才是「权重变窄」赚的，而不是「读到了垃圾」。
    #[test]
    fn f32_to_f16_matches_ieee_reference() {
        let cases: &[(f32, u16)] = &[
            (0.0, 0x0000),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-1.0, 0xbc00),
            (0.5, 0x3800),
            (1.0 / 3.0, 0x3555),
            (65504.0, 0x7bff),   // f16 最大正规数
            (65520.0, 0x7c00),   // 溢出：就近偶数恰好进位到 inf
            (65536.0, 0x7c00),
            (1e-5, 0x00a8),
            (6e-8, 0x0001),                  // 次正规
            (1e-9, 0x0000),                  // 太小 → 0
            (6.103_515_6e-5, 0x0400),       // 最小正规数
            (5.960_464_5e-8, 0x0001),       // 最小次正规数
            (0.1, 0x2e66),
            (-123.456, 0xd7b7),
            (f32::INFINITY, 0x7c00),
            (f32::NEG_INFINITY, 0xfc00),
        ];
        for &(v, want) in cases {
            let got = f32_to_f16_bits(v);
            assert_eq!(got, want, "f32 {v} → 期望 0x{want:04x}，实得 0x{got:04x}");
        }
        // NaN 的尾数必须非零，否则就退化成 inf 了。
        let nan = f32_to_f16_bits(f32::NAN);
        assert_eq!(nan & 0x7c00, 0x7c00, "NaN 的指数域应为全 1");
        assert_ne!(nan & 0x03ff, 0, "NaN 的尾数不该被洗成 0（那就是 inf）");
    }

    /// 打包的**下标语义**是整个 fp16 路径的正确性支点：half4 下标 X 必须
    /// 恰好覆盖元素 `4X..4X+3`（低位在前）。这里把两版并排比——f32 版
    /// float4 下标 X 覆盖的 4 个元素，与 fp16 版 half4 下标 X 覆盖的
    /// 必须逐个相等。
    #[test]
    fn f16_pack_preserves_half4_indexing() {
        // 3×5 conv，Co=8、Ci=4：nv=2、cin4v=1、taps=9，元素数 9*1*4*2*4=288。
        let (co, cin, kh, kw) = (8usize, 4usize, 3usize, 3usize);
        let w: Vec<f32> = (0..co * cin * kh * kw)
            .map(|i| (i as f32 * 0.37).sin() * 1.7)
            .collect();
        let w32 = repack_conv_w(&w, co, cin, kh, kw);
        let w16 = repack_conv_w_f16(&w, co, cin, kh, kw);
        assert_eq!(w16.len(), w32.len() / 2, "fp16 段长应是 f32 的一半");

        for x in 0..w32.len() / 4 {
            // f32 视图：float4 下标 X 的 4 个元素。
            let f32v: Vec<f32> = (0..4).map(|l| f32::from_bits(w32[x * 4 + l])).collect();
            // fp16 视图：half4 下标 X = word 2X/2X+1 里的 4 个 half。
            let mut f16v = [0f32; 4];
            for l in 0..4 {
                let half = ((w16[(x * 4 + l) / 2] >> (16 * ((x * 4 + l) % 2))) & 0xffff) as u16;
                f16v[l] = f16_bits_to_f32(half);
            }
            // 比的是**索引**，不是数值：同一个位置的元素应当等于它的 fp16
            // 舍入。数值不等（fp16 本来就窄）不算错，位置串了才算错。
            for l in 0..4 {
                let want = f16_bits_to_f32(f32_to_f16_bits(f32v[l]));
                assert_eq!(
                    f16v[l], want,
                    "half4 下标 {x} 的 lane {l} 与 f32 版 float4 对不上（位置串了）"
                );
            }
        }
    }

    /// 测试用的 half → f32（位级精确回读，仅用于上面的对拍）。
    fn f16_bits_to_f32(h: u16) -> f32 {
        let sign = ((h as u32) & 0x8000) << 16;
        let exp = ((h >> 10) & 0x1f) as i32;
        let mant = (h & 0x3ff) as u32;
        let bits = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                // 次正规：规格化后回填指数
                let mut e = -1i32;
                let mut m = mant;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (((e + 127 - 14) as u32) << 23) | ((m & 0x3ff) << 13)
            }
        } else if exp == 0x1f {
            sign | 0x7f80_0000 | (mant << 13)
        } else {
            sign | (((exp - 15 + 127) as u32) << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }
}
