//! CUDA 后端（枚举级预留）：dlopen 驱动（Windows `nvcuda.dll` /
//! Linux `libcuda.so.1`），**零编译期 CUDA SDK 依赖**——无 N 卡、无驱动
//! 的机器照样编译与运行（枚举为空）。
//!
//! 范围（架构定案）：设备枚举（名称 / 驱动版本）+ `create_session`
//! 明确报「内核未实现」。计算内核是 Phase 3（等 N 卡验证环境），
//! 届时在此模块补驱动 API 子集（上下文 / 模块 / launch），planner
//! 与 Vulkan 共用。
//!
//! 签名手写：CUDA 驱动 C ABI 稳定（`cuInit` 等自 2007 年不变），
//! 只取枚举所需的最小子集。

use qppocr_core::device::{DeviceContext, DeviceInfo, DeviceKind, DeviceSession, SessionOptions};
use qppocr_core::error::{Error, Result};
use qppocr_core::onnx::model::Graph;
use qppocr_core::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

/// CUDA 驱动 API 的状态码（0 = CUDA_SUCCESS）。
type CuStatus = u32;

// ---- 驱动函数的最小签名子集（枚举够用） ----
type CuInit = unsafe extern "C" fn(flags: u32) -> CuStatus;
type CuDriverGetVersion = unsafe extern "C" fn(version: *mut i32) -> CuStatus;
type CuDeviceGetCount = unsafe extern "C" fn(count: *mut i32) -> CuStatus;
type CuDeviceGet = unsafe extern "C" fn(device: *mut i32, ordinal: i32) -> CuStatus;
type CuDeviceGetName = unsafe extern "C" fn(name: *mut i8, len: i32, device: i32) -> CuStatus;

#[cfg(windows)]
const DRIVER_NAMES: &[&str] = &["nvcuda.dll"];
#[cfg(not(windows))]
const DRIVER_NAMES: &[&str] = &["libcuda.so.1", "libcuda.so"];

/// 按平台惯例名装载驱动库。失败 = `None`（没有 N 卡驱动是常态，不当错误）。
fn load_driver() -> Option<libloading::Library> {
    for name in DRIVER_NAMES {
        // SAFETY: 只按惯例名装载 CUDA 驱动库；库由驱动安装方提供，
        // 加载后仅调用下方手写签名、与官方头文件一致的 C ABI 函数。
        if let Ok(lib) = unsafe { libloading::Library::new(*name) } {
            return Some(lib);
        }
    }
    None
}

/// `[i8; N]` 的 C 字符串 → String。
fn cstr_i8(buf: &[i8]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// 驱动版本（如 12040 → "cuda 12.4"）。
fn api_str(version: i32) -> String {
    format!("cuda {}.{}", version / 1000, version % 1000 / 10)
}

/// 枚举本机全部 CUDA 设备。无驱动 / `cuInit` 失败 = 空列表。
pub(crate) fn enumerate() -> Vec<DeviceInfo> {
    enumerate_inner().unwrap_or_default()
}

fn enumerate_inner() -> Option<Vec<DeviceInfo>> {
    let lib = load_driver()?;
    // SAFETY: 符号名与签名均来自稳定的 CUDA 驱动 C ABI；返回值逐个检查。
    unsafe {
        let init: libloading::Symbol<CuInit> = lib.get(b"cuInit\0").ok()?;
        if init(0) != 0 {
            return None;
        }
        let get_ver: libloading::Symbol<CuDriverGetVersion> =
            lib.get(b"cuDriverGetVersion\0").ok()?;
        let mut ver = 0i32;
        if get_ver(&mut ver) != 0 {
            return None;
        }
        let get_count: libloading::Symbol<CuDeviceGetCount> =
            lib.get(b"cuDeviceGetCount\0").ok()?;
        let mut count = 0i32;
        if get_count(&mut count) != 0 {
            return None;
        }
        let get: libloading::Symbol<CuDeviceGet> = lib.get(b"cuDeviceGet\0").ok()?;
        let get_name: libloading::Symbol<CuDeviceGetName> = lib.get(b"cuDeviceGetName\0").ok()?;
        let mut out = Vec::new();
        for ordinal in 0..count {
            let mut dev = 0i32;
            if get(&mut dev, ordinal) != 0 {
                continue;
            }
            let mut buf = [0i8; 256];
            if get_name(buf.as_mut_ptr(), buf.len() as i32, dev) != 0 {
                continue;
            }
            out.push(DeviceInfo {
                kind: DeviceKind::Cuda,
                name: cstr_i8(&buf),
                api: api_str(ver),
            });
        }
        Some(out)
    }
}

/// 一个 CUDA 设备的上下文：驱动库 + 选定设备的名称与驱动版本。
///
/// Phase 3 加 CUcontext / 模块装载；现在保证「驱动已装载、已 cuInit」。
pub struct CudaContext {
    /// 驱动库的存活期即符号的有效期，必须持有。
    _lib: libloading::Library,
    name: String,
    api: String,
}

impl CudaContext {
    /// 装载驱动并 `cuInit`，打开第 `index` 个设备（`None` = 第一个）。
    pub fn open(index: Option<u32>) -> Result<Self> {
        let lib = load_driver().ok_or_else(|| {
            Error::Device(
                "找不到 CUDA 驱动库（nvcuda.dll / libcuda.so.1）——未装 \
                 NVIDIA 驱动时属常态；用 --device vulkan 或 --device cpu"
                    .into(),
            )
        })?;
        // SAFETY: 同 [`enumerate_inner`]——符号与签名来自稳定的驱动 C ABI。
        unsafe {
            let init: libloading::Symbol<CuInit> = lib
                .get(b"cuInit\0")
                .map_err(|e| Error::Device(format!("解析 cuInit 失败: {e}")))?;
            let st = init(0);
            if st != 0 {
                return Err(Error::Device(format!("cuInit 失败: 状态码 {st}")));
            }
            let mut count = 0i32;
            {
                let get_count: libloading::Symbol<CuDeviceGetCount> = lib
                    .get(b"cuDeviceGetCount\0")
                    .map_err(|e| Error::Device(format!("解析 cuDeviceGetCount 失败: {e}")))?;
                let st = get_count(&mut count);
                if st != 0 {
                    return Err(Error::Device(format!("cuDeviceGetCount 失败: 状态码 {st}")));
                }
            }
            if count == 0 {
                return Err(Error::Device(
                    "CUDA 驱动在位但没有任何设备（仅装驱动未接 N 卡？）；\
                     用 --device vulkan 或 --device cpu"
                        .into(),
                ));
            }
            let ordinal = match index {
                Some(i) => {
                    let i = i as i32;
                    if i >= count {
                        return Err(Error::Device(format!(
                            "--device cuda:{i} 超出范围（共 {count} 个 CUDA 设备）"
                        )));
                    }
                    i
                }
                None => 0,
            };
            let get: libloading::Symbol<CuDeviceGet> = lib
                .get(b"cuDeviceGet\0")
                .map_err(|e| Error::Device(format!("解析 cuDeviceGet 失败: {e}")))?;
            let mut dev = 0i32;
            let st = get(&mut dev, ordinal);
            if st != 0 {
                return Err(Error::Device(format!("cuDeviceGet 失败: 状态码 {st}")));
            }
            let mut name_buf = [0i8; 256];
            let get_name: libloading::Symbol<CuDeviceGetName> = lib
                .get(b"cuDeviceGetName\0")
                .map_err(|e| Error::Device(format!("解析 cuDeviceGetName 失败: {e}")))?;
            let st = get_name(name_buf.as_mut_ptr(), name_buf.len() as i32, dev);
            if st != 0 {
                return Err(Error::Device(format!("cuDeviceGetName 失败: 状态码 {st}")));
            }
            let mut ver = 0i32;
            let get_ver: libloading::Symbol<CuDriverGetVersion> = lib
                .get(b"cuDriverGetVersion\0")
                .map_err(|e| Error::Device(format!("解析 cuDriverGetVersion 失败: {e}")))?;
            let st = get_ver(&mut ver);
            if st != 0 {
                return Err(Error::Device(format!(
                    "cuDriverGetVersion 失败: 状态码 {st}"
                )));
            }
            Ok(Self {
                _lib: lib,
                name: cstr_i8(&name_buf),
                api: api_str(ver),
            })
        }
    }
}

impl DeviceContext for CudaContext {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cuda
    }

    fn info(&self) -> DeviceInfo {
        DeviceInfo {
            kind: DeviceKind::Cuda,
            name: self.name.clone(),
            api: self.api.clone(),
        }
    }

    fn create_session(
        &self,
        graph: Graph,
        initializers: HashMap<String, Tensor>,
        _opts: &SessionOptions,
    ) -> Result<Arc<dyn DeviceSession>> {
        // 枚举级预留（架构定案）：内核是 Phase 3，等 N 卡验证环境。
        drop(graph);
        drop(initializers);
        Err(Error::Device(
            "CUDA 计算内核未实现（Phase 3 计划——枚举先行是为了定型接口）；\
             当前用 --device vulkan 或 --device cpu"
                .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 有 N 卡驱动的机器列出设备；无驱动（本机开发环境/CI）自动跳过。
    #[test]
    fn enumerate_or_open_reports() {
        match CudaContext::open(None) {
            Ok(ctx) => {
                let i = ctx.info();
                eprintln!("[cuda] opened: {} ({})", i.name, i.api);
            }
            Err(e) => eprintln!("[cuda] 跳过（无 N 卡驱动）: {e}"),
        }
        for d in enumerate() {
            eprintln!("[cuda] {} ({})", d.name, d.api);
        }
    }
}
