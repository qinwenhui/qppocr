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
        let mut spv = compile(&name, &text);
        patch_barriers(&mut spv);
        let out = out_dir.join(format!("{name}.spv"));
        let bytes: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&out, bytes).unwrap();
        println!("{name}.comp -> {} 字节 SPIR-V", spv.len() * 4);
    }
}

/// SPIR-V 后处理：修复 naga 的 OpControlBarrier 三重 bug。
///
/// naga 对 GLSL `barrier()` 生成：
/// 1. 执行域/内存域有时是 Subgroup(3)/Device(1) 而非 Workgroup(2)
/// 2. 语义缺 WorkgroupMemory（0x80）——共享内存跨线程不可见
/// 3. 语义含多余的 AtomicCounterMemory(0x100)/SubgroupMemory(0x40)
///
/// **不能 patch 原常量**——naga 把 scope 常量（值 1=Device）与内核
/// 逻辑常量（如 `i+1` 的 1）合并为同一 OpConstant，改值会破坏内核。
///
/// 正确方案：**追加两个新 OpConstant**（Workgroup scope 和正确语义），
/// 然后把 barrier 的操作数重定向到新常量。
fn patch_barriers(words: &mut Vec<u32>) {
    const OP_CONSTANT: u32 = 43;
    const OP_TYPE_INT: u32 = 21;
    const OP_CONTROL_BARRIER: u32 = 224;
    const OP_FUNCTION: u32 = 54;
    const WORKGROUP: u32 = 2;
    const SEMANTICS: u32 = 0x10 | 0x80; // AcquireRelease | WorkgroupMemory

    // 1) 找所有 OpControlBarrier，确认需要 patch
    let mut barrier_positions: Vec<usize> = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let opcode = words[i] & 0xFFFF;
        let wc = (words[i] >> 16) as usize;
        if wc == 0 { break; }
        if opcode == OP_CONTROL_BARRIER && wc == 4 {
            barrier_positions.push(i);
        }
        i += wc;
    }
    if barrier_positions.is_empty() { return; }

    // 2) 找 uint 类型 ID（OpTypeInt width=32 signed=0）
    let mut uint_type_id = 0;
    let mut i = 5;
    while i < words.len() {
        let opcode = words[i] & 0xFFFF;
        let wc = (words[i] >> 16) as usize;
        if wc == 0 { break; }
        if opcode == OP_TYPE_INT && wc == 4 {
            // OpTypeInt: [wc|op, result_id, width, signedness]
            if words[i + 2] == 32 && words[i + 3] == 0 {
                uint_type_id = words[i + 1];
                break;
            }
        }
        i += wc;
    }
    if uint_type_id == 0 { return; } // 找不到 uint 类型，放弃

    // 3) 找第一个 OpFunction 的位置（新常量插在它之前）
    let mut first_fn = words.len();
    let mut i = 5;
    while i < words.len() {
        let opcode = words[i] & 0xFFFF;
        let wc = (words[i] >> 16) as usize;
        if wc == 0 { break; }
        if opcode == OP_FUNCTION {
            first_fn = i;
            break;
        }
        i += wc;
    }

    // 4) 分配新常量 ID（当前 bound 起始）
    let old_bound = words[3];
    let scope_id = old_bound;
    let sem_id = old_bound + 1;
    let new_bound = old_bound + 2;

    // 5) 构造新 OpConstant 指令
    //    OpConstant: [4<<16|43, type_id, result_id, value]
    let new_scope = [4u32 << 16 | OP_CONSTANT, uint_type_id, scope_id, WORKGROUP];
    let new_sem = [4u32 << 16 | OP_CONSTANT, uint_type_id, sem_id, SEMANTICS];

    // 6) 在 first_fn 之前插入新常量
    let mut insert_pos = first_fn;
    for &w in new_sem.iter().rev() {
        words.insert(insert_pos, w);
    }
    for &w in new_scope.iter().rev() {
        words.insert(insert_pos, w);
    }

    // 7) 更新 bound
    words[3] = new_bound;

    // 8) 重定向所有 barrier 的操作数（插入后位置偏移了 8 个 word）
    let offset = 8; // 2 条 OpConstant × 4 words
    for &bp in &barrier_positions {
        let pos = bp + offset;
        // OpControlBarrier: [wc|op, exec, mem, sem]
        words[pos + 1] = scope_id;  // exec → Workgroup
        words[pos + 2] = scope_id;  // mem → Workgroup
        words[pos + 3] = sem_id;    // semantics → AcquireRelease|WorkgroupMemory
    }

    println!(
        "  [patch] {} 个 barrier 重定向到新常量（scope=Workgroup sem=AcquireRelease|WorkgroupMemory）",
        barrier_positions.len()
    );
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
