//! GLSL → SPIR-V 编译工具。
//!
//! 仓库政策：SPIR-V **签入**（qppocr-gpu 构建零着色器依赖），本工具只在
//! 改着色器时手动运行（`cargo run --manifest-path tools/shader-build/Cargo.toml`），
//! CI 用它做新鲜度校验（重编译 + git diff --exit-code）。
//! 用 naga（纯 Rust）——不引 Vulkan SDK / glslang 的 C++ 依赖。

use std::path::PathBuf;

fn main() {
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
        let text = std::fs::read_to_string(src).unwrap();
        let spv = compile(&name, &text);
        let out = out_dir.join(format!("{name}.spv"));
        let bytes: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&out, bytes).unwrap();
        println!("{name}.comp -> {} 字节 SPIR-V", spv.len() * 4);
    }
}

/// naga 的 GLSL 前端 → 校验 → SPIR-V 后端。
fn compile(name: &str, text: &str) -> Vec<u32> {
    use naga::back::spv;
    use naga::front::glsl;
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    use naga::ShaderStage;

    let mut frontend = glsl::Frontend::default();
    let module = frontend
        .parse(&glsl::Options::from(ShaderStage::Compute), text)
        .map_err(|e| format!("{name}: GLSL 解析失败\n{e}"))
        .unwrap();
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| format!("{name}: 校验失败\n{e}"))
        .unwrap();
    spv::write_vec(
        &module,
        &info,
        &spv::Options::default(),
        Some(&spv::PipelineOptions {
            shader_stage: ShaderStage::Compute,
            entry_point: "main".into(),
        }),
    )
    .map_err(|e| format!("{name}: SPIR-V 生成失败\n{e}"))
    .unwrap()
}
