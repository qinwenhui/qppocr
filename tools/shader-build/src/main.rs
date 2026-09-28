//! GLSL → SPIR-V 编译工具（glslc 后端）。
//!
//! 仓库政策：SPIR-V **签入**（qppocr-gpu 构建零着色器依赖），本工具只在
//! 改着色器时手动运行（`cargo run --manifest-path tools/shader-build/Cargo.toml`）。
//!
//! 为什么是 glslc 而不是 naga：着色器要用 GL_KHR_cooperative_matrix、
//! subgroup 内建和 fp16 显式算术（Intel XMX 路径），naga 均不支持，且其
//! barrier/packHalf2x16 有实现 bug；glslc 生成的 SPIR-V 直接正确。
//!
//! 编译器解析顺序（找到第一个可用的）：
//! 1. 环境变量 `QPPOCR_GLSLC` 指定的可执行文件
//! 2. PATH 上的 `glslc`
//! 3. `%VULKAN_SDK%/Bin/glslc.exe`（Windows Vulkan SDK）
//! 4. WSL 内的 `glslc`（`apt install glslc`；Windows 路径自动转 /mnt/…）

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let glslc = resolve_glslc();
    println!("glslc: {}", glslc.display());

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/qppocr-gpu/shaders");
    let src_dir = root.join(".");
    let out_dir = root.join("spirv");
    std::fs::create_dir_all(&out_dir).unwrap();

    let mut sources: Vec<PathBuf> = std::fs::read_dir(&src_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "comp"))
        .collect();
    sources.sort();
    assert!(!sources.is_empty(), "shaders/ 下没有 .comp 源");

    for src in &sources {
        let name = src.file_stem().unwrap().to_str().unwrap().to_string();
        let out = out_dir.join(format!("{name}.spv"));
        compile(&glslc, src, &out);
        let bytes = std::fs::read(&out).unwrap();
        assert!(
            bytes.len() >= 20 && bytes[..4] == [0x03, 0x02, 0x23, 0x07],
            "{name}.spv 魔数/字节序不对"
        );
        println!("{name}.comp -> {} 字节 SPIR-V", bytes.len());
    }
}

/// 解析出 glslc 可执行文件路径。
///
/// WSL 兜底返回固定字符串 `wsl:`（不真实存在），由 [`compile`] 特判为
/// `wsl.exe -e glslc` 调用并把路径翻译成 /mnt 形式。
fn resolve_glslc() -> PathBuf {
    if let Ok(p) = std::env::var("QPPOCR_GLSLC") {
        let p = PathBuf::from(p);
        if probe(&p) {
            return p;
        }
        panic!("QPPOCR_GLSLC 指定的 {p:?} 不可执行（--version 失败）");
    }
    if let Some(p) = look_path("glslc") {
        return p;
    }
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let p = Path::new(&sdk).join("Bin").join("glslc.exe");
        if probe(&p) {
            return p;
        }
    }
    if probe(Path::new("wsl:")) {
        return PathBuf::from("wsl:");
    }
    panic!(
        "找不到 glslc。任选其一：\n  \
         1) 设置 QPPOCR_GLSLC 指向 glslc 可执行文件\n  \
         2) 把 glslc 加进 PATH（或安装 Vulkan SDK）\n  \
         3) WSL 里 `apt install glslc`"
    );
}

/// 在 PATH 上找可执行文件（Windows 上补 .exe/.bat 探测）。
fn look_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    let exts: &[&str] = if cfg!(windows) {
        &["", ".exe", ".bat", ".cmd"]
    } else {
        &[""]
    };
    for dir in std::env::split_paths(&path) {
        for ext in exts {
            let p = dir.join(format!("{name}{ext}"));
            if p.is_file() && probe(&p) {
                return Some(p);
            }
        }
    }
    None
}

/// 试运行 `--version`，能跑通才算可用。
fn probe(p: &Path) -> bool {
    if p == Path::new("wsl:") {
        // 依次试默认发行版和 Ubuntu-24.04
        return Command::new("wsl.exe")
            .args(["-e", "glslc", "--version"])
            .output()
            .is_ok_and(|o| o.status.success())
            || Command::new("wsl.exe")
                .args(["-d", "Ubuntu-24.04", "-e", "glslc", "--version"])
                .output()
                .is_ok_and(|o| o.status.success());
    }
    Command::new(p).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// 单个 .comp → .spv。目标环境 vulkan1.3（SPIR-V 1.6，coopmat KHR 需要的底座）。
fn compile(glslc: &Path, src: &Path, out: &Path) {
    let src_abs = dunce_canonical(src);
    let out_abs = dunce_canonical_parent(out);

    let status = if glslc == Path::new("wsl:") {
        let src_w = to_wsl_path(&src_abs);
        let out_w = to_wsl_path(&out_abs);
        run(Command::new("wsl.exe").args([
            "-e", "glslc",
            "--target-env=vulkan1.3", "-O",
            &src_w, "-o", &out_w,
        ]))
    } else {
        run(Command::new(glslc)
            .args(["--target-env=vulkan1.3", "-O"])
            .arg(src_abs)
            .arg("-o")
            .arg(out_abs))
    };
    if !status {
        panic!("glslc 编译失败: {}", src.display());
    }
}

fn run(cmd: &mut Command) -> bool {
    match cmd.output() {
        Ok(o) => {
            if !o.status.success() {
                eprintln!("{}", String::from_utf8_lossy(&o.stderr));
                eprintln!("{}", String::from_utf8_lossy(&o.stdout));
            }
            o.status.success()
        }
        Err(e) => {
            eprintln!("无法执行 {cmd:?}: {e}");
            false
        }
    }
}

/// canonicalize，剥掉 Windows 的 \\?\ 前缀（WSL 不认识）。
fn dunce_canonical(p: &Path) -> PathBuf {
    let c = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let s = c.to_string_lossy().replace(r"\\?\", "");
    PathBuf::from(s)
}

/// 输出文件可能还不存在：canonicalize 父目录再拼文件名。
fn dunce_canonical_parent(p: &Path) -> PathBuf {
    let name = p.file_name().unwrap().to_string_lossy().into_owned();
    dunce_canonical(&p.parent().unwrap()).join(name)
}

/// `D:\a\b.comp` → `/mnt/d/a/b.comp`（只在 wsl: 兜底路径下调用）。
fn to_wsl_path(p: &Path) -> String {
    let s = p.to_string_lossy().replace('\\', "/");
    let bytes = s.as_bytes();
    if bytes.len() >= 3 && bytes[1] == b':' && bytes[2] == b'/' {
        let drive = s[..1].to_ascii_lowercase();
        format!("/mnt/{drive}/{}", &s[3..])
    } else {
        s
    }
}
