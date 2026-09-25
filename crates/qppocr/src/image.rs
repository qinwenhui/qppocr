//! 图像输入：核心只吃已解码的 RGB 像素，
//! 解码与推理解耦——`image` crate 是 feature 门控的可选项。

pub use qppocr_core::pipeline::image::Image;

/// 从已解码的 RGB 字节构造（`w * h * 3`）。不涉及解码器。
pub fn rgb_from_bytes(w: u32, h: u32, data: Vec<u8>) -> Result<Image, crate::Error> {
    if data.len() != (w as usize) * (h as usize) * 3 {
        return Err(crate::Error::Image(format!(
            "RGB 数据长度 {} 与 {}x{} 不符（期望 {}）",
            data.len(),
            w,
            h,
            (w as usize) * (h as usize) * 3
        )));
    }
    Ok(Image {
        w: w as i32,
        h: h as i32,
        c: 3,
        orig_w: w as i32,
        orig_h: h as i32,
        data,
    })
}

/// 解码图像文件（PNG/JPEG；feature `image-decode`）。
#[cfg(feature = "image-decode")]
pub fn decode_file(path: impl AsRef<std::path::Path>) -> Result<Image, crate::Error> {
    decode_bytes(&std::fs::read(path)?)
}

/// 只读图片头部取尺寸（不解码像素；feature `image-decode`）。
///
/// 批量调度估算一张图要多「重」时用（对应 CLI 批量调度的
/// 头部探测）——解码本身保持全分辨率。
#[cfg(feature = "image-decode")]
pub fn probe_dimensions(path: impl AsRef<std::path::Path>) -> Result<(u32, u32), crate::Error> {
    image::ImageReader::open(path)
        .map_err(|e| crate::Error::Image(format!("打开失败: {e}")))?
        .into_dimensions()
        .map_err(|e| crate::Error::Image(format!("读头部失败: {e}")))
}

/// 解码图像字节（PNG/JPEG；feature `image-decode`）。
///
/// JPEG 的 **EXIF Orientation 自动应用**（对齐 PaddleOCR 官方流水线与
/// 浏览器显示）：竖拍照片的原始像素是横躺的，不转正的话检测框与显示
/// 坐标系是两套。PNG 无普遍的 EXIF 实践，不做。
#[cfg(feature = "image-decode")]
pub fn decode_bytes(bytes: &[u8]) -> Result<Image, crate::Error> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| crate::Error::Image(format!("解码失败: {e}")))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let out = Image {
        w: w as i32,
        h: h as i32,
        c: 3,
        orig_w: w as i32,
        orig_h: h as i32,
        data: rgb.into_raw(),
    };
    let o = if bytes.len() > 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
        exif_orientation(bytes)
    } else {
        1
    };
    Ok(apply_orientation(out, o))
}

/// 从 JPEG 字节流读 EXIF Orientation（无 EXIF/解析失败返回 1 = 不转）。
/// 只扫 APP1 段与 IFD0 的 0x0112 短整数——手写解析，零依赖。
#[cfg(feature = "image-decode")]
fn exif_orientation(jpeg: &[u8]) -> u8 {
    let rd16 = |b: &[u8], le: bool| -> u16 {
        if le {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        }
    };
    let rd32 = |b: &[u8], le: bool| -> u32 {
        if le {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        }
    };
    let mut i = 2usize; // 跳过 SOI
    while i + 4 <= jpeg.len() {
        if jpeg[i] != 0xFF {
            return 1; // 段流错位，放弃
        }
        let marker = jpeg[i + 1];
        if marker == 0xD8 || (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            i += 2;
            continue;
        }
        if marker == 0xDA || marker == 0xD9 {
            return 1; // 到扫描数据/结尾，没有 EXIF
        }
        let seg_len = rd16(&jpeg[i + 2..], false) as usize;
        if marker == 0xE1
            && seg_len >= 8
            && i + 10 + 8 <= jpeg.len()
            && &jpeg[i + 4..i + 10] == b"Exif\x00\x00"
        {
            let t = &jpeg[i + 10..];
            let le = t.len() >= 2 && t[0] == b'I' && t[1] == b'I';
            let be = t.len() >= 2 && t[0] == b'M' && t[1] == b'M';
            if !le && !be {
                return 1;
            }
            let ifd_off = rd32(&t[4..], le) as usize;
            if ifd_off + 2 > t.len() {
                return 1;
            }
            let n = rd16(&t[ifd_off..], le) as usize;
            for e in 0..n {
                let base = ifd_off + 2 + e * 12;
                if base + 12 > t.len() {
                    return 1;
                }
                if rd16(&t[base..], le) == 0x0112 {
                    let v = rd16(&t[base + 8..], le);
                    return v.clamp(1, 8) as u8;
                }
            }
            return 1;
        }
        i += 2 + seg_len;
    }
    1
}

/// 按方向值转正像素（1-8；1 原样返回不拷贝）。消费原 `Image`。
#[cfg(feature = "image-decode")]
fn apply_orientation(mut img: Image, o: u8) -> Image {
    if o <= 1 {
        return img;
    }
    let (w, h, c) = (img.w as usize, img.h as usize, 3usize);
    let src = std::mem::take(&mut img.data);
    let (nw, nh) = match o {
        5..=8 => (h, w),
        _ => (w, h),
    };
    let mut dst = vec![0u8; nw * nh * c];
    for y in 0..h {
        for x in 0..w {
            // 源像素 (x, y) 在显示图上的落点
            let (dx, dy) = match o {
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (y, w - 1 - x),
                7 => (h - 1 - y, w - 1 - x),
                8 => (h - 1 - y, x),
                _ => (x, y),
            };
            let s = (y * w + x) * c;
            let d = (dy * nw + dx) * c;
            dst[d..d + c].copy_from_slice(&src[s..s + c]);
        }
    }
    img.w = nw as i32;
    img.h = nh as i32;
    img.orig_w = nw as i32;
    img.orig_h = nh as i32;
    img.data = dst;
    img
}

#[cfg(all(test, feature = "image-decode"))]
mod exif_tests {
    use super::*;

    /// 构造只含 Orientation 的最小 JPEG 头（SOI + APP1/Exif + IFD0 一条）。
    fn synth(orientation: u16, little: bool) -> Vec<u8> {
        let mut tiff: Vec<u8> = Vec::new();
        let put16 = |v: u16, le: bool| -> Vec<u8> {
            if le {
                v.to_le_bytes().to_vec()
            } else {
                v.to_be_bytes().to_vec()
            }
        };
        let put32 = |v: u32, le: bool| -> Vec<u8> {
            if le {
                v.to_le_bytes().to_vec()
            } else {
                v.to_be_bytes().to_vec()
            }
        };
        tiff.extend_from_slice(if little { b"II" } else { b"MM" });
        tiff.extend_from_slice(&put16(0x2A, little));
        tiff.extend_from_slice(&put32(8, little)); // IFD0 在 TIFF 头后
        tiff.extend_from_slice(&put16(1, little)); // 一个条目
        tiff.extend_from_slice(&put16(0x0112, little)); // Orientation
        tiff.extend_from_slice(&put16(3, little)); // SHORT
        tiff.extend_from_slice(&put32(1, little)); // count
        tiff.extend_from_slice(&put16(orientation, little));
        tiff.extend_from_slice(&[0, 0]); // 值槽补齐
        tiff.extend_from_slice(&put32(0, little)); // next IFD

        let mut jpeg = vec![0xFF, 0xD8];
        let mut app1 = b"Exif\x00\x00".to_vec();
        app1.extend_from_slice(&tiff);
        jpeg.extend_from_slice(&[0xFF, 0xE1]);
        jpeg.extend_from_slice(&(app1.len() as u16 + 2).to_be_bytes());
        jpeg.extend_from_slice(&app1);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);
        jpeg
    }

    #[test]
    fn parses_both_endians() {
        for o in [1u16, 3, 6, 8] {
            assert_eq!(exif_orientation(&synth(o, true)) as u16, o, "LE {o}");
            assert_eq!(exif_orientation(&synth(o, false)) as u16, o, "BE {o}");
        }
        assert_eq!(exif_orientation(&[0xFF, 0xD8, 0xFF, 0xD9]), 1);
    }

    #[test]
    fn orientation_pixel_math() {
        let mk = |w: usize, h: usize| -> Image {
            let mut data = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    data.extend_from_slice(&[(x * 100 + y) as u8, 0, 0]);
                }
            }
            Image {
                w: w as i32,
                h: h as i32,
                c: 3,
                orig_w: w as i32,
                orig_h: h as i32,
                data,
            }
        };
        let clone = |im: &Image| -> Image {
            Image {
                w: im.w,
                h: im.h,
                c: im.c,
                orig_w: im.orig_w,
                orig_h: im.orig_h,
                data: im.data.clone(),
            }
        };
        let px = |im: &Image, x: usize, y: usize| -> u8 { im.data[(y * im.w as usize + x) * 3] };
        let img = mk(2, 3);
        // 6 = 90°CW：显示 3x2，源 (x,y) → 显示 (y, w-1-x)
        let r6 = apply_orientation(clone(&img), 6);
        assert_eq!((r6.w, r6.h), (3, 2));
        assert_eq!(px(&r6, 0, 0), px(&img, 1, 0));
        // 8 = 90°CCW：源 (x,y) → 显示 (h-1-y, x)
        let r8 = apply_orientation(clone(&img), 8);
        assert_eq!((r8.w, r8.h), (3, 2));
        assert_eq!(px(&r8, 2, 0), px(&img, 0, 0));
        // 3 = 180°
        let r3 = apply_orientation(clone(&img), 3);
        assert_eq!(px(&r3, 1, 2), px(&img, 0, 0));
        // 2 = 水平镜像
        let r2 = apply_orientation(clone(&img), 2);
        assert_eq!(px(&r2, 1, 0), px(&img, 0, 0));
    }
}
