//! 模型来源与校验。

use crate::sha256::sha256;

/// 模型档位。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum Tier {
    /// 最快。det 1.8 MB + rec 4.5 MB。
    Tiny,
    /// 默认。准确率与速度的平衡点。
    #[default]
    Small,
    /// 最准（⚠ 未经基准验证，）。
    Medium,
}

impl Tier {
    /// 目录名（`ModelSource::Dir` 的子目录布局）。
    pub fn dir_name(self) -> &'static str {
        match self {
            Tier::Tiny => "tiny",
            Tier::Small => "small",
            Tier::Medium => "medium",
        }
    }
}

/// 上游官方原件的 SHA-256（PaddleOCR 发布，HuggingFace `inference.onnx`）。
/// 来源：附录 D，全部已下载核对。
const UPSTREAM_SHA: &[(Tier, Kind, &str)] = &[
    (
        Tier::Tiny,
        Kind::Det,
        "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8",
    ),
    (
        Tier::Tiny,
        Kind::Rec,
        "9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6",
    ),
    (
        Tier::Small,
        Kind::Det,
        "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e",
    ),
    (
        Tier::Small,
        Kind::Rec,
        "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
    ),
    (
        Tier::Medium,
        Kind::Det,
        "eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1",
    ),
    (
        Tier::Medium,
        Kind::Rec,
        "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba",
    ),
];

/// 方向分类模型（三档共用同一个）。
#[allow(dead_code)] // fetch feature 落地时用
pub const CLS_SHA: &str = "dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2";

/// 字典的 SHA-256（small/medium 共用一份 18708 行）。
#[allow(dead_code)] // fetch feature 落地时用
pub const DICT_SHA: &str = "118d0f0714ad2a37668c23d6541f2c3feb65b8214041265b567f7fd5b3365d8e";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Det,
    Rec,
}

/// 期望的 SHA-256（按档位与模型种类查上游官方值）。
fn expected_sha(tier: Tier, kind: Kind) -> Option<&'static str> {
    UPSTREAM_SHA
        .iter()
        .find(|(t, k, _)| *t == tier && *k == kind)
        .map(|(_, _, s)| *s)
}

/// 模型文件的来源。
#[non_exhaustive]
pub enum ModelSource {
    /// 从目录读。布局（`fetch` 也下载成这个布局）：
    ///
    /// ```text
    /// models/
    /// ├── tiny/  ├── det.onnx
    /// │          └── rec.onnx
    /// ├── small/ ├── det.onnx
    /// │          └── rec.onnx
    /// ├── medium/├── det.onnx
    /// │          └── rec.onnx
    /// ├── cls.onnx            （可选，三档共用）
    /// └── dict.txt            （可选；rec 模型不带内嵌字典时必需）
    /// ```
    ///
    /// SHA-256 默认校验（与官方原件不符时报错——防下载损坏与调包）；
    /// 自备重导出模型用 [`EngineBuilder::verify_sha256`](crate::EngineBuilder::verify_sha256)`(false)`。
    Dir(PathBuf),
    /// 由调用方提供字节（WASM / 移动端从 asset 取）。
    /// `cls` 与 `dict` 可选；`dict` 为 `None` 时用 rec 模型的内嵌字典
    ///（上游原件不带，见 [`Dictionary`](crate::Dictionary)）。
    Bytes {
        /// 检测模型。
        det: Vec<u8>,
        /// 识别模型。
        rec: Vec<u8>,
        /// 方向分类模型。
        cls: Option<Vec<u8>>,
        /// 字典文本（每行一个字符）。
        dict: Option<String>,
    },
}

use std::path::PathBuf;

/// 装载结果：三个模型的字节 + 字典来源。
pub(crate) struct Loaded {
    pub det: Vec<u8>,
    pub rec: Vec<u8>,
    pub cls: Option<Vec<u8>>,
    pub dict: crate::Dictionary,
}

impl ModelSource {
    /// 读取全部模型与字典。`tier` 决定子目录；SHA 校验按 `verify` 开关。
    pub(crate) fn load(&self, tier: Tier, verify: bool) -> Result<Loaded, crate::Error> {
        match self {
            ModelSource::Dir(root) => {
                let tier_dir = root.join(tier.dir_name());
                let det_path = tier_dir.join("det.onnx");
                let rec_path = tier_dir.join("rec.onnx");
                let det = std::fs::read(&det_path).map_err(|e| {
                    crate::Error::Model(format!(
                        "读不到检测模型 {}: {e}\
                         \n  期望布局: <dir>/{}/det.onnx（见 ModelSource::Dir 文档）",
                        det_path.display(),
                        tier.dir_name()
                    ))
                })?;
                let rec = std::fs::read(&rec_path).map_err(|e| {
                    crate::Error::Model(format!("读不到识别模型 {}: {e}", rec_path.display()))
                })?;
                if verify {
                    check_sha(&det, tier, Kind::Det, &det_path)?;
                    check_sha(&rec, tier, Kind::Rec, &rec_path)?;
                }
                let cls = std::fs::read(root.join("cls.onnx")).ok();
                // 字典：rec 内嵌（转换版）或目录里的 dict.txt / ppocr_keys.txt
                let dict = match load_dict_file(root, &tier_dir) {
                    Some(text) => crate::Dictionary::Text(text),
                    None => crate::Dictionary::Embedded,
                };
                Ok(Loaded {
                    det,
                    rec,
                    cls,
                    dict,
                })
            }
            ModelSource::Bytes {
                det,
                rec,
                cls,
                dict,
            } => Ok(Loaded {
                det: det.clone(),
                rec: rec.clone(),
                cls: cls.clone(),
                dict: match dict {
                    Some(s) => crate::Dictionary::Text(s.clone()),
                    None => crate::Dictionary::Embedded,
                },
            }),
        }
    }
}

/// 目录里的字典查找：`{tier}/dict.txt` → `dict.txt` → `ppocr_keys.txt`。
fn load_dict_file(root: &std::path::Path, tier_dir: &std::path::Path) -> Option<String> {
    for p in [
        tier_dir.join("dict.txt"),
        root.join("dict.txt"),
        root.join("ppocr_keys.txt"),
    ] {
        if let Ok(t) = std::fs::read_to_string(&p) {
            return Some(t);
        }
    }
    None
}

/// 校验失败时报错（带两个已知值与跳过方法——防呆不挡路）。
fn check_sha(
    data: &[u8],
    tier: Tier,
    kind: Kind,
    path: &std::path::Path,
) -> Result<(), crate::Error> {
    let want = match expected_sha(tier, kind) {
        Some(s) => s,
        None => return Ok(()),
    };
    let got = sha256(data).hex();
    if got != want {
        // 「不要静默降级」与「不挡自备模型」的平衡：报错，但把出路写清楚。
        // 是 Err 不是 panic——库不该替宿主应用决定「崩掉」（模型损坏/
        // 被调包是数据问题，应用层要能接住并引导用户）。
        return Err(crate::Error::Model(format!(
            "模型 SHA-256 不匹配（{}）:\n  期望（上游官方）: {want}\n  实际:            {got}\n\
             若这是你自己重导出的模型，用 EngineBuilder::verify_sha256(false) 跳过校验。",
            path.display()
        )));
    }
    Ok(())
}
