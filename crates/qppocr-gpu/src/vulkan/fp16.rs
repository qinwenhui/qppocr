//! fp16 全管线：中间张量全部 f16 存储（带宽减半），计算保持 f32 精度。
//!
//! 存储：SSBO 里的 `uint`，每个 uint 打包 2 个 f16（`packHalf2x16`）。
//! 元素偏移仍按 float 索引（逻辑上 `data[i]` = 第 i 个 f16 值），
//! 物理访问 `data_words[i/2]` 的 (i%2) 半边。
//!
//! 管线形态：
//! ```
//! f32 输入 → [to_f16] → f16 中间张量 ×N 层 → [from_f16] → f32 输出
//! ```
//! 转换内核各一趟读写，成本 ~2×张量大小 / 带宽——相比 N 层 conv 的
//! 读写总量，开销可忽略。
//!
//! 计算精度：内核读 f16 → 解包为 f32 → FMA 累加（f32）→ 结果打包 f16。
//! 权重精度：装载期一次性转换为 f16——det 模型对权重量化到 f16 的
//! 精度损失在逐字符对拍里不构成差异（竞品全部用 fp16 推理）。

use ash::vk;
use qppocr_core::error::Result;

use super::device::VulkanDevice;
use super::memory::Arena;
use super::pipeline::{KernelSet, PcUnary, record_dispatch};

/// f32 → f16 转换内核（元素数需偶数；装载期 padding 保证）。
/// 输入：f32 data[in_off .. in_off+n]
/// 输出：f16 packed data[out_off/2 .. ]（n/2 个 uint）
pub(crate) fn record_to_f16(
    dev: &ash::Device,
    cb: vk::CommandBuffer,
    ks: &KernelSet,
    in_off: u32,
    out_off: u32,
    n: u32, // 元素数（f16 计）
) {
    let pc = PcUnary { in_off, out_off, n };
    // SAFETY: cb 录制态；PC 与 to_f16.comp 参数对应。
    unsafe {
        record_dispatch(dev, cb, ks, "to_f16", pc.bytes(), [n / 2 / 64 + 1, 1, 1]);
    }
}

/// f16 → f32 转换内核。
pub(crate) fn record_from_f16(
    dev: &ash::Device,
    cb: vk::CommandBuffer,
    ks: &KernelSet,
    in_off: u32,
    out_off: u32,
    n: u32,
) {
    let pc = PcUnary { in_off, out_off, n };
    // SAFETY: 同上。
    unsafe {
        record_dispatch(dev, cb, ks, "from_f16", pc.bytes(), [n / 2 / 64 + 1, 1, 1]);
    }
}

/// f32 切片 → f16 packed words（装载期权重转换用）。
pub fn f32_to_f16_words(src: &[f32]) -> Vec<u32> {
    let n = src.len().div_ceil(2) * 2;
    let mut out = Vec::with_capacity(n / 2);
    for i in (0..n).step_by(2) {
        let a = src.get(i).copied().unwrap_or(0.0) as f16_helper;
        let b = src.get(i + 1).copied().unwrap_or(0.0) as f16_helper;
        out.push(pack_half2x16(a, b));
    }
    out
}

/// f16 packed words → f32 切片（测试/诊断用）。
pub fn f16_words_to_f32(src: &[u32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(src.len() * 2);
    for &w in src {
        let (a, b) = unpack_half2x16(w);
        out.push(a);
        out.push(b);
    }
    out
}

// ---- f32 ↔ f16 位操作（不引 half crate，直接位运算） ----

/// f32 → f16 的位级转换（round-to-nearest-even，IEEE 754）。
/// 精度：尾数 23→10 位截断 + 舍入；值域 [±65504]；denormal → 0。
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 0xFF {
        // Inf / NaN
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    if exp == 0 && mant == 0 {
        return sign; // ±0
    }
    // 舍入到最近的 f16
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F {
        return sign | 0x7C00; // overflow → Inf
    }
    if new_exp <= 0 {
        // subnormal 或 0
        if new_exp < -10 {
            return sign; // too small → ±0
        }
        // subnormal：mant >> (14 - new_exp) + 舍入
        let shift = 14 - new_exp;
        let mant = mant | 0x80_0000; // implicit 1
        let rounded = mant >> shift;
        // round-to-nearest-even
        let rem = mant & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        let mut r = rounded;
        if rem > half || (rem == half && (rounded & 1) != 0) {
            r += 1;
        }
        return sign | r as u16;
    }
    // normal
    let mant16 = mant >> 13;
    let rem = mant & 0x1FFF;
    let mut m = mant16;
    if rem > 0x1000 || (rem == 0x1000 && (mant16 & 1) != 0) {
        m += 1;
        if m == 0x400 {
            m = 0;
            return sign | (((new_exp + 1) as u16) << 10);
        }
    }
    sign | ((new_exp as u16) << 10) | m as u16
}

/// f16 → f32 的位级转换。
fn f16_to_f32_bits(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    let bits = match (exp, mant) {
        (0, 0) => sign, // ±0
        (0, m) => {
            // subnormal
            // 逐步正规化
            let mut e = -1i32;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            sign | (((127 - 15 + e + 1) as u32) << 23) | (m << 13)
        }
        (0x1F, 0) => sign | 0x7F80_0000,               // ±Inf
        (0x1F, _) => sign | 0x7F80_0000 | 0x0040_0000, // NaN
        _ => sign | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

fn pack_half2x16(a: f32, b: f32) -> u32 {
    (f32_to_f16_bits(a) as u32) | ((f32_to_f16_bits(b) as u32) << 16)
}

fn unpack_half2x16(w: u32) -> (f32, f32) {
    (f16_to_f32_bits(w as u16), f16_to_f32_bits((w >> 16) as u16))
}

// 类型别名让 f32_to_f16_words 的 as 转换语义清晰
type f16_helper = f32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        let vals = [
            0.0f32, 1.0, -1.0, 0.5, -0.5, 3.14159, 1e-5, 65504.0, -65504.0,
        ];
        let words = f32_to_f16_words(&vals);
        let back = f16_words_to_f32(&words);
        for (i, (a, b)) in vals.iter().zip(back.iter()).enumerate() {
            let rel = (a - b).abs() / (1.0 + a.abs());
            assert!(rel < 0.001, "f16 roundtrip [{i}]: {a} → {b} (rel={rel})");
        }
    }

    #[test]
    fn f16_edge_cases() {
        assert_eq!(f32_to_f16_bits(0.0), 0);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
        assert_eq!(f32_to_f16_bits(-1.0), 0xBC00);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7C00);
        assert!(f32_to_f16_bits(f32::NAN) & 0x7C00 == 0x7C00);
    }
}
