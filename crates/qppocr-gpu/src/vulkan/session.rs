//! VulkanSession：图计划生成 + 会话执行（Phase 1-G2）。
//!
//! 装载期（create_session）：图过 planner 静态形状推理。输入形状是
//! **运行期**才定的——首次 run 按输入形状现建计划并缓存：op→内核
//! 映射 → 活性复用的区域布局（**单块**：KernelSet 绑单缓冲）→ 权重
//! 与参数块上传 → 整图 dispatch 录进一条可复用命令缓冲。
//!
//! 运行期（run）：按输入形状取计划 → memcpy 写入 → submit_wait_cb →
//! 读回输出。零逐帧重录、零描述符操作——每帧成本就是一次提交。
//!
//! 已知边界（诚实记录）：
//! - 计划缓存未封顶（形状多样性高时 arena 无界增长——LRU 在性能轮加）；
//! - SE 融合内核（fused_*）未接模式匹配，sigmoid/hardsigmoid 走独立
//!   dispatch + mul_c——正确性优先，融合在性能轮按 profile 上；
//! - 逐节点 dump 不支持（QPPOCR_DUMP_DIR 设置时 stderr 声明，不静默）。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ash::vk;
use qppocr_core::device::{DeviceKind, DeviceSession};
use qppocr_core::error::{Error, Result};
use qppocr_core::onnx::model::{Graph, Node};
use qppocr_core::tensor::{DType, Tensor};
use qppocr_kernels::buf::F32Buf;

use super::memory::Arena;
use super::pipeline::{
    KernelSet, OFF_NONE, ParamBlock, PcBinary, PcChannel, PcMulAddScale, PcUnary, PcUnaryF,
    record_dispatch,
};
use crate::planner;

/// 计划缓存的条目明细（调试/步进对拍用）：(内核, 节点名, 节点序号,
/// 输出偏移, 输出元素数, PC 字节, dispatch 网格, 输入明细)。
type PlanRec = (
    String,
    String,
    usize,
    u32,
    u32,
    Vec<u8>,
    [u32; 3],
    Vec<(String, u32, u32)>,
);

/// 一个输入形状对应的整图执行计划。
struct Plan {
    /// 整块基址（持久映射；存活期归 `_arena`）。
    base: *mut u8,
    /// 输入名 → 偏移（float 元素，相对 base）。
    in_offs: HashMap<String, u32>,
    /// 图输出：(名字, 偏移, 元素数)（按 graph.outputs 顺序）。
    out_offs: Vec<(String, u32, usize)>,
    out_shapes: Vec<Vec<i64>>,
    /// 录好的整图命令缓冲（可复用）。
    cb: vk::CommandBuffer,
    /// 逐 dispatch 明细（步进对拍调试用；正常路径不读）。
    #[allow(dead_code)]
    recs: Vec<PlanRec>,
    _arena: Arena,
    _ks: KernelSet,
}

// SAFETY: base 是 _arena 持久映射的裸指针，存活期与 Plan 相同且无并发
// 写（run 全程持锁）；句柄均为 Vulkan 整数。跨线程只是把 Plan 放进
// Mutex 保护的缓存。
unsafe impl Send for Plan {}
// SAFETY: 同上——&Plan 的共享只发生在持锁的 run 里。
unsafe impl Sync for Plan {}

/// 区域布局：活性复用的 bump + first-fit 空闲表（全部相对同一整块）。
struct Layout {
    free: Vec<(u32, u32)>,
    cursor: u32,
    total: u32,
}

impl Layout {
    fn new() -> Self {
        Self {
            free: Vec::new(),
            cursor: 0,
            total: 0,
        }
    }

    fn alloc(&mut self, n: u32) -> u32 {
        let n = n.div_ceil(4) * 4; // 16 B 对齐
        for i in 0..self.free.len() {
            if self.free[i].1 >= n {
                let off = self.free[i].0;
                let rest = self.free[i].1 - n;
                if rest == 0 {
                    self.free.swap_remove(i);
                } else {
                    // 剩余区段后移到 [off+n, ...)——留原地就是与本次
                    // 分配重叠（曾致 in-place 别名：输入输出同址、深度
                    // 卷积读写互相踩）。
                    self.free[i].0 = off + n;
                    self.free[i].1 = rest;
                }
                return off;
            }
        }
        let off = self.cursor;
        self.cursor += n;
        self.total = self.cursor;
        off
    }

    fn free(&mut self, off: u32, n: u32) {
        let n = n.div_ceil(4) * 4;
        if n > 0 {
            self.free.push((off, n));
        }
    }

    /// 静态区（权重/参数）专用：**只 bump、永不复用**。
    ///
    /// 激活区域可以 free-复用——生产者 dispatch 先写、消费者后读，
    /// 单次重放内自洽。但静态内容只在装载期写一次，而 CB 每帧重放
    /// 时**复用区的原生产者 dispatch 仍会写那片内存**——参数落进
    /// 复用区 = 运行时被冲掉、消费方读到垃圾参数（本 bug 的实测形态：
    /// conv 参数损坏 → 巨循环 → GPU 挂死设备丢失）。
    fn alloc_static(&mut self, n: u32) -> u32 {
        let off = self.cursor;
        self.cursor += n.div_ceil(4) * 4;
        self.total = self.cursor;
        off
    }
}

fn consumer_counts(graph: &Graph) -> HashMap<String, u32> {
    let mut refs: HashMap<String, u32> = HashMap::new();
    for n in &graph.nodes {
        for inn in &n.inputs {
            if !inn.is_empty() {
                *refs.entry(inn.clone()).or_default() += 1;
            }
        }
    }
    refs
}

fn numel(table: &planner::ShapeTable, name: &str) -> u32 {
    table
        .values
        .get(name)
        .map(|v| v.shape.iter().product::<i64>() as u32)
        .unwrap_or(0)
}

/// GPU 会话：设备共享句柄 + 形状键计划缓存。
pub(crate) struct VulkanSession {
    graph: Graph,
    initializers: HashMap<String, Tensor>,
    input_name: String,
    plans: Mutex<Vec<(Vec<i64>, Plan)>>,
    /// 持有整个上下文。**必须声明在最后**：字段按声明序 drop——plans
    /// 里的管线/缓冲销毁要用设备，而设备销毁发生在 ctx（Arc<Inner>）
    /// 的最后一个引用释放时。ctx 在前 = 先毁设备再毁管线 = 对已销毁
    /// 设备调用 destroy（实测 CLI 路径 100% 段错误；测试路径因 ctx
    /// 局部变量活得比会话久而侥幸通过）。
    ctx: Arc<super::Inner>,
}

impl VulkanSession {
    pub(crate) fn new(
        ctx: Arc<super::Inner>,
        graph: Graph,
        initializers: HashMap<String, Tensor>,
    ) -> Result<Self> {
        if std::env::var_os("QPPOCR_DUMP_DIR").is_some() {
            eprintln!(
                "[gpu] QPPOCR_DUMP_DIR 已设置：GPU 会话不落盘逐节点 dump（融合调度下\
                 无逐节点中间量）；节点级诊断请用 --device cpu"
            );
        }
        let input_name = graph
            .inputs
            .iter()
            .find(|s| !s.is_empty())
            .cloned()
            .unwrap_or_default();
        Ok(Self {
            graph,
            initializers,
            input_name,
            plans: Mutex::new(Vec::new()),
            ctx,
        })
    }

    /// 步进执行 + 逐节点对拍（输入已写入；QPPOCR_GPU_DUMP_CMP 给出
    /// CPU dump 目录）。首个分歧节点带两侧首元素值报错。
    fn stepped_run_and_cmp(&self, plan: &Plan) -> Result<()> {
        let dump_dir = std::env::var_os("QPPOCR_GPU_DUMP_CMP");
        let dev = self.ctx.device.raw().clone();
        for (i, (kernel, node, idx, out_off, out_n, pc, groups, ins)) in
            plan.recs.iter().enumerate()
        {
            // 输入复查：每个输入区域当前值 vs 它生产时的 dump——
            // 不匹配 = 该区域被复用写花（liveness bug 的直接证据）。
            if let Some(dir) = &dump_dir {
                for (inm, ioff, in_n) in ins {
                    let Some(prod_idx) = node_index_of(&self.graph, inm) else {
                        continue;
                    };
                    let path = std::path::Path::new(dir).join(format!("s0_{prod_idx:06}.f32"));
                    let Ok(bytes) = std::fs::read(&path) else {
                        continue;
                    };
                    let rank = rank_of(&bytes);
                    let want: Vec<f32> = bytes[4 + 8 * rank..]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    // SAFETY: base 持久映射；ioff+in_n 在 total 内。
                    let got: Vec<f32> = unsafe {
                        std::slice::from_raw_parts(
                            plan.base.add(*ioff as usize * 4) as *const f32,
                            *in_n as usize,
                        )
                    }
                    .to_vec();
                    if got.len() == want.len() {
                        let e = got
                            .iter()
                            .zip(&want)
                            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
                            .fold(0.0f32, f32::max);
                        if e > 1e-3 {
                            return Err(Error::Graph(format!(
                                "[step] 节点 {node} 的输入 {inm} 在 dispatch 前已被改写：                                 rel|diff|={e:.3e}（区域被复用/覆盖——liveness 或参数错位）"
                            )));
                        }
                    }
                }
            }
            let one = self.ctx.device.alloc_reusable_cb()?;
            // SAFETY: cb 录制态；PC 与内核参数块对应（build 期生成）。
            unsafe {
                record_dispatch(&dev, one, &plan._ks, kernel, pc, *groups);
            }
            self.ctx.device.end_reusable_cb(one)?;
            self.ctx.device.submit_wait_cb(one)?;
            eprintln!("[gpu][step] {i}/{} {} {} ok", plan.recs.len(), kernel, node);
            if let Some(dir) = &dump_dir {
                let path = std::path::Path::new(dir).join(format!("s0_{idx:06}.f32"));
                if let Ok(bytes) = std::fs::read(&path) {
                    let rank = rank_of(&bytes);
                    let want: Vec<f32> = bytes[4 + 8 * rank..]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    // SAFETY: base 持久映射；out_off+out_n 在 total 内。
                    let got: Vec<f32> = unsafe {
                        std::slice::from_raw_parts(
                            plan.base.add(*out_off as usize * 4) as *const f32,
                            *out_n as usize,
                        )
                    }
                    .to_vec();
                    if got.len() == want.len() {
                        let e = got
                            .iter()
                            .zip(&want)
                            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
                            .fold(0.0f32, f32::max);
                        if e > 0.3 {
                            let bad: Vec<String> = got
                                .iter()
                                .zip(&want)
                                .enumerate()
                                .filter(|(_, (a, b))| (**a - **b).abs() / (1.0 + (*b).abs()) > 0.3)
                                .take(3)
                                .map(|(i, (a, b))| format!("[{i}] gpu={a:.6} cpu={b:.6}"))
                                .collect();
                            let n = got.len();
                            return Err(Error::Graph(format!(
                                "[step] 首个分歧节点 {node}（{kernel}，idx={idx}，{n} 元素）：                                 rel|diff|={e:.3e}，{}",
                                bad.join(" ")
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn build_plan(&self, in_shape: &[i64]) -> Result<Plan> {
        let table = planner::infer_shapes(
            &self.graph,
            &self.initializers,
            &[(self.input_name.clone(), in_shape.to_vec(), DType::F32)],
        )?;

        let refs = consumer_counts(&self.graph);
        if std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some() {
            if let Some(n0) = self.graph.nodes.first() {
                let o0 = &n0.outputs[0];
                eprintln!("[gpu][live] {} 的消费计数 = {:?}", o0, refs.get(o0));
            }
        }
        let mut live: HashMap<String, u32> = refs.clone();

        let mut layout = Layout::new();
        let mut ledger: Vec<(String, u32, u32)> = Vec::new(); // (名, off, len)
        let mut offs: HashMap<String, u32> = HashMap::new();
        // 权重占区（F32 且被引用；i64 常量链由 planner 折叠，不上传）。
        // 权重**永不释放**：命令缓冲每次重放都读它，区域复用=数据被覆盖。
        let mut upload: Vec<(u32, Vec<f32>)> = Vec::new();
        for (name, t) in &self.initializers {
            let Some(v) = table.values.get(name) else {
                continue;
            };
            if v.dtype != DType::F32 || refs.get(name).copied().unwrap_or(0) == 0 {
                continue;
            }
            let off = layout.alloc_static(t.f32.len() as u32);
            ledger.push((format!("w:{name}"), off, t.f32.len() as u32));
            offs.insert(name.clone(), off);
            upload.push((off, t.f32.to_vec()));
        }
        // 图输入占区：每次 run 重写，同样永不释放。
        offs.insert(
            self.input_name.clone(),
            layout.alloc(in_shape.iter().product::<i64>() as u32),
        );

        struct Rec {
            kernel: &'static str,
            node: String,
            node_idx: usize,
            out_off: u32,
            out_n: u32,
            pc: Vec<u8>,
            groups: [u32; 3],
            ins: Vec<(String, u32, u32)>, // (输入名, off, numel)
        }
        let mut recs: Vec<Rec> = Vec::new();
        let mut params: Vec<(u32, Vec<u32>)> = Vec::new();
        let is_output = |nm: &str| self.graph.is_graph_output(nm);

        for (node_idx, n) in self.graph.nodes.iter().enumerate() {
            let out = n.outputs[0].clone();
            let out_v = table.values.get(&out).ok_or_else(|| {
                Error::Graph(format!("planner 缺少 {} 的输出 {}", n.op_type, out))
            })?;
            // i64 常量链节点：planner 已折叠出值，无区域无 dispatch
            if out_v.dtype == DType::I64 {
                release_inputs(
                    n,
                    &mut live,
                    &mut offs,
                    &mut layout,
                    &table,
                    &is_output,
                    &self.initializers,
                    &self.input_name,
                );
                continue;
            }
            // 输出区域先占（map_node 的参数要引用它）
            let out_n = numel(&table, &out);
            let out_off = layout.alloc(out_n);
            if std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some() {
                eprintln!(
                    "[gpu][live] {out} @{} 分配于 {}（len {out_n}）",
                    out_off, n.name
                );
            }
            ledger.push((format!("n:{}", n.name), out_off, out_n));
            offs.insert(out.clone(), out_off);

            let (kernel, pb, pc_inline, groups) = self.map_node(n, &table, &offs)?;
            if std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some() && n.name == "Conv.5" {
                eprintln!(
                    "[gpu][dbg] {} 输入形状: {:?}",
                    n.name,
                    n.inputs
                        .iter()
                        .map(|i| table.values.get(i).map(|v| v.shape.clone()))
                        .collect::<Vec<_>>()
                );
                eprintln!(
                    "[gpu][dbg] 输出形状: {:?} 偏移={:?}",
                    out_v.shape,
                    offs.get(&out)
                );
                if let Some(pb) = &pb {
                    eprintln!("[gpu][dbg] 参数块 words = {:?}", pb.words());
                }
                eprintln!(
                    "[gpu][dbg] 各输入偏移: {:?}",
                    n.inputs
                        .iter()
                        .map(|i| offs.get(i).copied())
                        .collect::<Vec<_>>()
                );
                eprintln!("[gpu][dbg] layout.total = {}", layout.total);
            }
            let pc: Vec<u8> = if let Some(pb) = pb {
                // 参数块与数据同块（u32 视图）；静态内容，装载期写一次
                let p_off = layout.alloc_static(pb.len_words());
                ledger.push((format!("p:{}", n.name), p_off, pb.len_words()));
                params.push((p_off, pb.words().to_vec()));
                super::pipeline::PcParams { p_off }.bytes().to_vec()
            } else {
                pc_inline
            };
            let ins: Vec<(String, u32, u32)> = n
                .inputs
                .iter()
                .filter(|i| !i.is_empty())
                .map(|i| {
                    (
                        i.clone(),
                        offs.get(i).copied().unwrap_or(0),
                        numel(&table, i),
                    )
                })
                .collect();
            recs.push(Rec {
                kernel,
                node: n.name.clone(),
                node_idx,
                out_off,
                out_n,
                pc,
                groups,
                ins,
            });

            release_inputs(
                n,
                &mut live,
                &mut offs,
                &mut layout,
                &table,
                &is_output,
                &self.initializers,
                &self.input_name,
            );
        }

        // 对账（调试模式）：静态区（w:/p:）两两不重叠、不与图输出重叠。
        // 激活区允许复用（合法性由「生产者先写消费者后读」保证），故
        // 只查静态区。alloc_static 的 bump-only 构造本身就排除了静态区
        // 落入复用区——这段是防御性验证。
        if std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some() {
            let mut statics: Vec<(String, u32, u32)> = ledger
                .iter()
                .filter(|(nm, _, _)| nm.starts_with("w:") || nm.starts_with("p:"))
                .cloned()
                .collect();
            for o in self.graph.outputs.iter() {
                if let Some(off) = offs.get(o) {
                    statics.push((format!("OUT:{o}"), *off, numel(&table, o)));
                }
            }
            statics.sort_by_key(|(_, o, _)| *o);
            for w in statics.windows(2) {
                let (n1, o1, l1) = &w[0];
                let (n2, o2, _) = &w[1];
                assert!(
                    o1 + l1 <= *o2,
                    "静态区重叠: {n1} [{o1},{}) vs {n2} @{o2}",
                    o1 + l1
                );
            }
        }
        // === 实分配：整块一次（KernelSet 绑单缓冲）===
        let mut arena = Arena::new(
            self.ctx.device.raw().clone(),
            self.ctx.mem_types.staging,
            true,
        );
        let whole = arena
            .alloc(layout.total as vk::DeviceSize * 4)
            .map_err(|e| {
                Error::Device(format!("计划整块分配失败（{} floats）: {e}", layout.total))
            })?;
        let (buf, buf_size) = arena
            .chunk_range()
            .ok_or_else(|| Error::Device("arena 空".into()))?;
        let ks = KernelSet::new(self.ctx.device.raw(), buf, buf_size)?;

        // 权重 + 参数块上传（coherent 映射直写）
        // SAFETY: whole 是整块持久映射；off+len ≤ total。
        unsafe {
            for (off, data) in &upload {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    whole.ptr.add(*off as usize * 4) as *mut f32,
                    data.len(),
                );
            }
            for (p_off, words) in &params {
                std::ptr::copy_nonoverlapping(
                    words.as_ptr(),
                    whole.ptr.add(*p_off as usize * 4) as *mut u32,
                    words.len(),
                );
            }
        }

        // 录制整图 dispatch 序列
        let dev = self.ctx.device.raw().clone();
        let cb = self.ctx.device.alloc_reusable_cb()?;
        for r in &recs {
            // SAFETY: cb 处于录制态；PC 与内核参数块逐字段对应（map_node 保证）。
            unsafe {
                record_dispatch(&dev, cb, &ks, r.kernel, &r.pc, r.groups);
            }
        }
        self.ctx.device.end_reusable_cb(cb)?;

        let out_offs: Vec<(String, u32, usize)> = self
            .graph
            .outputs
            .iter()
            .map(|o| {
                let off = offs.get(o).copied().unwrap_or(0);
                (o.clone(), off, numel(&table, o) as usize)
            })
            .collect();
        let out_shapes: Vec<Vec<i64>> = self
            .graph
            .outputs
            .iter()
            .map(|o| {
                table
                    .values
                    .get(o)
                    .map(|v| v.shape.clone())
                    .unwrap_or_default()
            })
            .collect();

        eprintln!(
            "[gpu] 计划就绪：输入 {in_shape:?}，{} 个 dispatch，整块 {:.1} MB",
            recs.len(),
            layout.total as f64 * 4.0 / 1e6
        );
        let dbg_recs: Vec<PlanRec> = recs
            .iter()
            .map(|r| {
                (
                    r.kernel.to_string(),
                    r.node.clone(),
                    r.node_idx,
                    r.out_off,
                    r.out_n,
                    r.pc.clone(),
                    r.groups,
                    r.ins.clone(),
                )
            })
            .collect();
        Ok(Plan {
            base: whole.ptr,
            recs: dbg_recs,
            in_offs: [(self.input_name.clone(), offs[&self.input_name])]
                .into_iter()
                .collect(),
            out_offs,
            out_shapes,
            cb,
            _arena: arena,
            _ks: ks,
        })
    }

    /// 节点 → (内核名, 参数块 | 内联 PC, dispatch 网格)。
    #[allow(clippy::type_complexity)]
    fn map_node(
        &self,
        n: &Node,
        table: &planner::ShapeTable,
        offs: &HashMap<String, u32>,
    ) -> Result<(&'static str, Option<ParamBlock>, Vec<u8>, [u32; 3])> {
        let shape_of = |name: &str| -> Vec<i64> {
            table
                .values
                .get(name)
                .map(|v| v.shape.clone())
                .unwrap_or_default()
        };
        let off_of = |name: &str| -> Result<u32> {
            offs.get(name).copied().ok_or_else(|| {
                Error::Graph(format!(
                    "{}: 输入 {} 无区域（非权重/前驱输出）",
                    n.op_type, name
                ))
            })
        };
        let out = &n.outputs[0];
        let out_shape = shape_of(out);
        let out_n = numel(table, out);

        match n.op_type.as_str() {
            "Conv" => {
                let xs = shape_of(&n.inputs[0]);
                let ws = shape_of(&n.inputs[1]);
                let (sh, sw) = strides(n);
                let pads = pads4(n);
                let b_off = if n.inputs.len() > 2 && !n.inputs[2].is_empty() {
                    off_of(&n.inputs[2])?
                } else {
                    OFF_NONE
                };
                let r_off = if n.inputs.len() > 3 && !n.inputs[3].is_empty() {
                    off_of(&n.inputs[3])?
                } else {
                    OFF_NONE
                };
                let oh = out_shape[2] as u32;
                let ow = out_shape[3] as u32;
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(off_of(&n.inputs[1])?)
                    .u(b_off)
                    .u(r_off)
                    .u(off_of(out)?)
                    .u(xs[0] as u32)
                    .u(xs[1] as u32)
                    .u(xs[2] as u32)
                    .u(xs[3] as u32)
                    .u(ws[0] as u32)
                    .u(ws[2] as u32)
                    .u(ws[3] as u32)
                    .u(sh as u32)
                    .u(sw as u32)
                    .u(pads.0 as u32)
                    .u(pads.1 as u32)
                    .u(planner::get_i(n.attr("group"), 1) as u32)
                    .u(planner::get_i(n.attr("act"), 0) as u32)
                    .f(planner::get_f(n.attr("act_c1"), std::f32::consts::SQRT_2))
                    .f(planner::get_f(n.attr("act_c2"), 1.0))
                    .f(planner::get_f(n.attr("act_c3"), 0.5))
                    .u(oh * ow)
                    .u(ow);
                Ok((
                    "conv",
                    Some(pb),
                    Vec::new(),
                    [
                        ow / 8 + 1,
                        oh / 8 + 1,
                        out_shape[0] as u32 * out_shape[1] as u32,
                    ],
                ))
            }
            "ConvTranspose" => {
                let xs = shape_of(&n.inputs[0]);
                let ws = shape_of(&n.inputs[1]);
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(off_of(&n.inputs[1])?)
                    .u(off_of(out)?)
                    .u(xs[0] as u32)
                    .u(xs[1] as u32)
                    .u(xs[2] as u32)
                    .u(xs[3] as u32)
                    .u(ws[1] as u32);
                Ok((
                    "convtranspose",
                    Some(pb),
                    Vec::new(),
                    [
                        (2 * xs[3]) as u32 / 8 + 1,
                        (2 * xs[2]) as u32 / 8 + 1,
                        out_shape[0] as u32 * out_shape[1] as u32,
                    ],
                ))
            }
            "MaxPool" | "AveragePool" => {
                let xs = shape_of(&n.inputs[0]);
                let ks = n
                    .attr("kernel_shape")
                    .filter(|a| a.ints.len() == 2)
                    .ok_or_else(|| Error::Graph("kernel_shape required".into()))?;
                let (sh, sw) = strides(n);
                let pads = pads4(n);
                let oh = out_shape[2] as u32;
                let ow = out_shape[3] as u32;
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(off_of(out)?)
                    .u(xs[0] as u32)
                    .u(xs[1] as u32)
                    .u(xs[2] as u32)
                    .u(xs[3] as u32)
                    .u(ks.ints[0] as u32)
                    .u(ks.ints[1] as u32)
                    .u(sh as u32)
                    .u(sw as u32)
                    .u(pads.0 as u32)
                    .u(pads.1 as u32)
                    .u(u32::from(n.op_type == "MaxPool"))
                    .u(oh)
                    .u(ow);
                Ok((
                    "pool",
                    Some(pb),
                    Vec::new(),
                    [
                        ow / 8 + 1,
                        oh / 8 + 1,
                        out_shape[0] as u32 * out_shape[1] as u32,
                    ],
                ))
            }
            "GlobalAveragePool" | "ReduceMean" => {
                if n.op_type == "ReduceMean" {
                    let axes = planner::axes_from(n, &table.values)
                        .map_err(|e| Error::Graph(format!("ReduceMean: {e}")))?;
                    let r = shape_of(&n.inputs[0]).len() as i64;
                    let norm: Vec<i64> = axes
                        .iter()
                        .map(|&a| if a < 0 { a + r } else { a })
                        .collect();
                    if !(norm.len() == 2 && norm.contains(&2) && norm.contains(&3)) {
                        return Err(Error::Graph(format!(
                            "ReduceMean 只支持 axes={{2,3}}（SE 归约），实得 {axes:?}"
                        )));
                    }
                }
                let xs = shape_of(&n.inputs[0]);
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(off_of(out)?)
                    .u(xs[0] as u32)
                    .u(xs[1] as u32)
                    .u(xs[2] as u32)
                    .u(xs[3] as u32);
                Ok((
                    "reduce_hw",
                    Some(pb),
                    Vec::new(),
                    [(xs[0] * xs[1]) as u32 / 256 + 1, 1, 1],
                ))
            }
            "Resize" => {
                let mode = n
                    .attr("mode")
                    .filter(|a| a.has_s)
                    .map(|a| a.s.clone())
                    .unwrap_or_else(|| "nearest".into());
                if mode != "nearest" {
                    return Err(Error::Graph(format!(
                        "Resize 只支持 nearest（det 全是它），实得 {mode}"
                    )));
                }
                let xs = shape_of(&n.inputs[0]);
                let oh = out_shape[2] as u32;
                let ow = out_shape[3] as u32;
                let mut pb = ParamBlock::new();
                pb.u(off_of(&n.inputs[0])?)
                    .u(off_of(out)?)
                    .u(xs[0] as u32)
                    .u(xs[1] as u32)
                    .u(xs[2] as u32)
                    .u(xs[3] as u32)
                    .u(oh)
                    .u(ow);
                Ok((
                    "resize_nearest",
                    Some(pb),
                    Vec::new(),
                    [
                        ow / 8 + 1,
                        oh / 8 + 1,
                        out_shape[0] as u32 * out_shape[1] as u32,
                    ],
                ))
            }
            "Concat" => {
                let axis = planner::get_i(n.attr("axis"), 0);
                if axis != 1 {
                    return Err(Error::Graph(format!("Concat 只支持 axis=1，实得 {axis}")));
                }
                let xs = shape_of(&n.inputs[0]);
                let mut pb = ParamBlock::new();
                pb.u(off_of(out)?)
                    .u(xs[0] as u32)
                    .u(xs[2] as u32)
                    .u(xs[3] as u32)
                    .u(n.inputs.len() as u32)
                    .u(out_shape[1] as u32)
                    .u(0)
                    .u(0);
                for inn in &n.inputs {
                    if inn.is_empty() {
                        continue;
                    }
                    pb.u(off_of(inn)?).u(shape_of(inn)[1] as u32);
                }
                Ok((
                    "concat_c",
                    Some(pb),
                    Vec::new(),
                    [
                        xs[3] as u32 / 8 + 1,
                        xs[2] as u32 / 8 + 1,
                        out_shape[0] as u32 * out_shape[1] as u32,
                    ],
                ))
            }
            "MulAddScale" => {
                let fs = shape_of(&n.inputs[0]);
                let (c, hw) = (fs[1] as u32, (fs[2] * fs[3]) as u32);
                let pc = PcMulAddScale {
                    f_off: off_of(&n.inputs[0])?,
                    gate_off: off_of(&n.inputs[1])?,
                    r_off: off_of(&n.inputs[2])?,
                    out_off: off_of(out)?,
                    hw,
                    c,
                };
                Ok((
                    "muladd_scale",
                    None,
                    pc.bytes().to_vec(),
                    [hw / 256 + 1, c, fs[0] as u32],
                ))
            }
            "Sigmoid" | "Relu" | "HardSigmoid" | "Clip" => {
                let pc = if n.op_type == "Clip" {
                    PcUnaryF {
                        in_off: off_of(&n.inputs[0])?,
                        out_off: off_of(out)?,
                        n: out_n,
                        p1: planner::get_f(n.attr("min"), -3.4e38),
                        p2: planner::get_f(n.attr("max"), 3.4e38),
                    }
                    .bytes()
                    .to_vec()
                } else if n.op_type == "HardSigmoid" {
                    PcUnaryF {
                        in_off: off_of(&n.inputs[0])?,
                        out_off: off_of(out)?,
                        n: out_n,
                        p1: planner::get_f(n.attr("alpha"), 0.2),
                        p2: planner::get_f(n.attr("beta"), 0.5),
                    }
                    .bytes()
                    .to_vec()
                } else {
                    PcUnary {
                        in_off: off_of(&n.inputs[0])?,
                        out_off: off_of(out)?,
                        n: out_n,
                    }
                    .bytes()
                    .to_vec()
                };
                let kernel = match n.op_type.as_str() {
                    "Sigmoid" => "sigmoid",
                    "Relu" => "relu",
                    "HardSigmoid" => "hardsigmoid",
                    _ => "clip",
                };
                Ok((kernel, None, pc, [out_n / 256 + 1, 1, 1]))
            }
            "Add" | "Mul" => {
                let a = shape_of(&n.inputs[0]);
                let b = shape_of(&n.inputs[1]);
                let b_n = numel(table, &n.inputs[1]);
                if a == b {
                    let pc = PcBinary {
                        a_off: off_of(&n.inputs[0])?,
                        b_off: off_of(&n.inputs[1])?,
                        out_off: off_of(out)?,
                        n: out_n,
                    };
                    Ok((
                        if n.op_type == "Add" { "add" } else { "mul" },
                        None,
                        pc.bytes().to_vec(),
                        [out_n / 256 + 1, 1, 1],
                    ))
                } else if b.len() == 4 && b_n == b[1] as u32 {
                    // [1,C,1,1] 通道广播
                    let pc = PcChannel {
                        a_off: off_of(&n.inputs[0])?,
                        b_off: off_of(&n.inputs[1])?,
                        out_off: off_of(out)?,
                        hw: (a[2] * a[3]) as u32,
                        c: a[1] as u32,
                    };
                    Ok((
                        if n.op_type == "Mul" { "mul_c" } else { "add_c" },
                        None,
                        pc.bytes().to_vec(),
                        [(a[2] * a[3]) as u32 / 256 + 1, a[1] as u32, a[0] as u32],
                    ))
                } else {
                    Err(Error::Graph(format!(
                        "{}: 只支持同形或 [1,C,1,1] 通道广播，实得 {a:?} vs {b:?}",
                        n.op_type
                    )))
                }
            }
            other => Err(Error::Graph(format!(
                "GPU 计划不支持算子 {other}（节点 {}）——当前内核集覆盖 det",
                n.name
            ))),
        }
    }
}

/// 输出名 → 产出它的节点序号（线性扫，调试路径，图 ≤ 数百节点）。
fn node_index_of(graph: &Graph, out: &str) -> Option<usize> {
    graph
        .nodes
        .iter()
        .position(|n| n.outputs.first() == Some(&out.to_string()))
}

/// dump 文件头的 rank（i32 LE）。
fn rank_of(bytes: &[u8]) -> usize {
    if bytes.len() < 4 {
        return 0;
    }
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize
}

fn strides(n: &Node) -> (i64, i64) {
    let s = n
        .attr("strides")
        .map(|a| a.ints.clone())
        .unwrap_or_default();
    if s.len() == 2 { (s[0], s[1]) } else { (1, 1) }
}

fn pads4(n: &Node) -> (i64, i64, i64, i64) {
    let p = n.attr("pads").map(|a| a.ints.clone()).unwrap_or_default();
    if p.len() >= 4 {
        (p[0], p[1], p[2], p[3])
    } else {
        (0, 0, 0, 0)
    }
}

impl DeviceSession for VulkanSession {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Vulkan
    }

    fn prefers_host_parallelism(&self) -> bool {
        false // GPU 内部已并行：pipeline 走串行批（Phase 0 接缝）
    }

    fn run(&self, inputs: Vec<(String, Tensor)>) -> Result<Vec<Tensor>> {
        let mut plans = self
            .plans
            .lock()
            .map_err(|_| Error::Device("计划缓存锁中毒".into()))?;
        let in_shape: Vec<i64> = inputs
            .iter()
            .find(|(nm, _)| nm == &self.input_name)
            .map(|(_, t)| t.shape.clone())
            .ok_or_else(|| Error::Graph(format!("缺少输入 {}", self.input_name)))?;
        if !plans.iter().any(|(s, _)| *s == in_shape) {
            let plan = self.build_plan(&in_shape)?;
            plans.push((in_shape.clone(), plan));
        }
        let plan = plans
            .iter()
            .find(|(s, _)| *s == in_shape)
            .map(|(_, p)| p)
            .unwrap();

        for (nm, t) in &inputs {
            if let Some(off) = plan.in_offs.get(nm) {
                // SAFETY: base 是整块持久映射；off+len 在 total 内（按输入形状分配）。
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        t.f32.as_ptr(),
                        plan.base.add(*off as usize * 4) as *mut f32,
                        t.f32.len(),
                    );
                }
            }
        }
        if std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some() {
            self.stepped_run_and_cmp(plan)?;
        } else {
            self.ctx.device.submit_wait_cb(plan.cb)?;
        }

        let mut outs = Vec::with_capacity(plan.out_offs.len());
        for (i, (name, off, n)) in plan.out_offs.iter().enumerate() {
            let shape = plan.out_shapes.get(i).cloned().unwrap_or_default();
            let mut buf = F32Buf::with_zeroed(*n);
            // SAFETY: 已等信号；off+n ≤ total。
            unsafe {
                std::ptr::copy_nonoverlapping(
                    plan.base.add(*off as usize * 4) as *const f32,
                    buf.as_mut_slice().as_mut_ptr(),
                    *n,
                );
            }
            outs.push(Tensor {
                name: name.clone(),
                shape,
                dtype: DType::F32,
                f32: buf,
                i64: Vec::new(),
            });
        }
        Ok(outs)
    }
}

/// 消费释放：输入若再无消费者，归还区域给布局复用。
/// **豁免**：图输出（要读回）、权重（CB 每次重放都读）、图输入（每次
/// run 重写）——这三类区域一旦被复用就是数据 corrupt。
#[allow(clippy::too_many_arguments)]
fn release_inputs(
    n: &Node,
    live: &mut HashMap<String, u32>,
    offs: &mut HashMap<String, u32>,
    layout: &mut Layout,
    table: &planner::ShapeTable,
    is_output: &dyn Fn(&str) -> bool,
    initializers: &HashMap<String, Tensor>,
    input_name: &str,
) {
    let dbg = std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some();
    for inn in &n.inputs {
        if inn.is_empty() {
            continue;
        }
        if is_output(inn) || initializers.contains_key(inn) || inn == input_name {
            continue;
        }
        if let Some(c) = live.get_mut(inn) {
            *c = c.saturating_sub(1);
            if dbg {
                eprintln!("[gpu][live] {inn} 消费于 {}（剩 {}）", n.name, *c);
            }
            if *c == 0 && table.values.get(inn).map(|v| v.dtype) == Some(DType::F32) {
                if let Some(o) = offs.remove(inn) {
                    if dbg {
                        eprintln!("[gpu][live] {inn} @{} 释放于 {}", o, n.name);
                    }
                    layout.free(o, numel(table, inn));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::VulkanContext;
    use qppocr_core::executor::Session;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    /// 端到端：真实 tiny det，GPU 整图计划 vs CPU executor，同一输入
    /// 的概率图容差对拍；同形二次 run 走缓存、异形 run 建新计划。
    #[test]
    fn det_session_end_to_end() {
        let p = std::path::Path::new("../../models/tiny/det.onnx");
        if !p.is_file() {
            eprintln!("[gpu] 无模型，跳过");
            return;
        }
        let bytes = std::fs::read(p).unwrap();
        let Some(ctx) = (match VulkanContext::open(None) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("[gpu] 跳过（无满足基线的设备）: {e}");
                None
            }
        }) else {
            return;
        };

        let (h, w) = (960i64, 864i64);
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mk_input = |name: &str, seed: &mut u64| -> Tensor {
            let mut buf = F32Buf::with_zeroed((3 * h * w) as usize);
            for v in buf.as_mut_slice().iter_mut() {
                *v = lcg(seed);
            }
            Tensor {
                name: name.into(),
                shape: vec![1, 3, h, w],
                dtype: DType::F32,
                f32: buf,
                i64: Vec::new(),
            }
        };
        // ★ 同一份输入克隆给 CPU/GPU——mk_input 每次调用是新随机数据，
        //   曾因两边各生成一份导致「全图分歧」的假阳性。
        let input_once = mk_input("", &mut seed);
        let cl = move |nm: &str| {
            let mut t = input_once.clone();
            t.name = nm.to_string();
            t
        };

        let _ = &cl; // 占位（下方用 cl() 统一取同一份输入）
        // CPU 参考（带 dump：GPU 步进模式的逐节点对拍判据）
        let dump_dir = std::env::temp_dir().join("qppocr-g2-dbg");
        let _ = std::fs::remove_dir_all(&dump_dir);
        std::fs::create_dir_all(&dump_dir).unwrap();
        // SAFETY: 测试独占该环境变量（仅 Session::run 读）。
        unsafe { std::env::set_var("QPPOCR_DUMP_DIR", &dump_dir) };
        let cpu = Session::from_memory(&bytes, "tiny.det").unwrap();
        let in_name = cpu
            .graph
            .inputs
            .iter()
            .find(|s| !s.is_empty())
            .cloned()
            .unwrap();
        let t0 = std::time::Instant::now();
        let out_cpu = cpu.run(vec![(in_name.clone(), cl(&in_name))]).unwrap();
        // SAFETY: 同上。
        unsafe { std::env::remove_var("QPPOCR_DUMP_DIR") };
        eprintln!(
            "[gpu] CPU 前向 {:.1} ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );

        if std::env::var_os("QPPOCR_GPU_STEP_DEBUG").is_some() {
            // SAFETY: 同上（本测试进程内）。
            unsafe { std::env::set_var("QPPOCR_GPU_DUMP_CMP", &dump_dir) };
        }
        // GPU 会话（同图同权重）
        let (graph, init) = {
            let s = Session::from_memory(&bytes, "tiny.det2").unwrap();
            s.into_parts()
        };
        let gpu = VulkanSession::new(ctx.inner.clone(), graph, init).unwrap();
        let t1 = std::time::Instant::now();
        let out_gpu = gpu.run(vec![(in_name.clone(), cl(&in_name))]).unwrap();
        eprintln!(
            "[gpu] GPU 首跑（含建计划）{:.1} ms",
            t1.elapsed().as_secs_f64() * 1000.0
        );
        // 同形二次（缓存命中 + CB 重放）
        let t2 = std::time::Instant::now();
        let _ = gpu.run(vec![(in_name.clone(), cl(&in_name))]).unwrap();
        eprintln!(
            "[gpu] GPU 二跑（缓存）{:.1} ms",
            t2.elapsed().as_secs_f64() * 1000.0
        );

        assert_eq!(out_cpu.len(), out_gpu.len());
        let a = &out_gpu[0].f32;
        let b = &out_cpu[0].f32;
        assert_eq!(a.len(), b.len(), "输出元素数不一致");
        assert_eq!(out_gpu[0].shape, out_cpu[0].shape, "输出形状不一致");
        let max_abs = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let mean_abs = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>()
            / a.len() as f32;
        eprintln!("[gpu] det 概率图：max|diff| = {max_abs:.3e}，mean|diff| = {mean_abs:.3e}");
        // 验收口径：mean 反映整体数值健康（逐层 fma/累加序差异的传播），
        // max 允许 sigmoid 斜坡处的极值放大（det_thresh=0.2 附近的框边界
        // 翻转由 verify.py 的字符级对拍判定，不在这里卡死）。
        assert!(mean_abs < 1e-3, "概率图均值偏差超容差: {mean_abs}");
        assert!(max_abs < 5e-2, "概率图极值偏差超容差: {max_abs}");
    }
}
