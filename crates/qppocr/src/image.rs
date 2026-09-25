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
#[cfg(feature = "image-decode")]
pub fn decode_bytes(bytes: &[u8]) -> Result<Image, crate::Error> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| crate::Error::Image(format!("解码失败: {e}")))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok(Image {
        w: w as i32,
        h: h as i32,
        c: 3,
        orig_w: w as i32,
        orig_h: h as i32,
        data: rgb.into_raw(),
    })
}
