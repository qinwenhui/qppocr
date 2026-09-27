//! 手写的 protobuf wire-format 读取器（`onnx_parser.cpp` ）。
//!
//! ONNX 就是普通 protobuf，只走我们需要的字段，不引 libprotobuf/prost
//! （ 会让构建依赖 protoc，与「纯 Rust、
//! 无外部依赖」的定位矛盾）。**支持 opset 14**——上游 det 就是 opset 14
//! 。
//!
//! 字段号（onnx.proto）：
//! - ModelProto：ir_version=1, producer_name=2, producer_version=3,
//!   domain=4, model_version=5, doc_string=6, graph=7, opset_import=8,
//!   metadata_props=14
//! - GraphProto：node=1, name=2, initializer=5, input=11, output=12
//! - NodeProto：input=1, output=2, name=3, op_type=4, attribute=5
//! - AttributeProto：name=1, f=2, i=3, s=4, t=5, floats=7, ints=8, type=20
//! - TensorProto：dims=1, data_type=2, float_data=4, int32_data=5,
//!   int64_data=7, name=8, raw_data=9, double_data=10
//! - ValueInfoProto：name=1
//! - OperatorSetIdProto：domain=1, version=2
//! - StringStringEntryProto：key=1, value=2

use crate::error::{Error, Result};
use crate::onnx::model::{Attribute, Graph, Node};
use crate::tensor::{DType, Tensor};

/// 光标读取器。所有方法在越界时报**字节偏移**——同款，错误信息能直接
/// 定位到文件的哪个位置。
struct Reader<'a> {
    p: usize,
    end: usize,
    buf: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader {
            p: 0,
            end: buf.len(),
            buf,
        }
    }

    fn sub(&self, start: usize, len: usize) -> Reader<'a> {
        Reader {
            p: 0,
            end: len,
            buf: &self.buf[start..start + len],
        }
    }

    fn off(&self) -> usize {
        self.p
    }

    fn eof(&self) -> bool {
        self.p >= self.end
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v: u64 = 0;
        let mut shift = 0u32;
        loop {
            if self.p >= self.end {
                return Err(Error::Parse(format!("truncated varint @{}", self.off())));
            }
            let b = self.buf[self.p];
            self.p += 1;
            v |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift > 63 {
                return Err(Error::Parse("varint too long".into()));
            }
        }
        Ok(v)
    }

    fn fixed32(&mut self) -> Result<u32> {
        if self.p + 4 > self.end {
            return Err(Error::Parse(format!("truncated fixed32 @{}", self.off())));
        }
        let v = u32::from_le_bytes(self.buf[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        Ok(v)
    }

    fn fixed64(&mut self) -> Result<u64> {
        if self.p + 8 > self.end {
            return Err(Error::Parse("truncated fixed64".into()));
        }
        let v = u64::from_le_bytes(self.buf[self.p..self.p + 8].try_into().unwrap());
        self.p += 8;
        Ok(v)
    }

    /// 长度前缀的字段：返回 (起始, 长度)。
    fn bytes(&mut self) -> Result<(usize, usize)> {
        let len = self.varint()? as usize;
        if self.p + len > self.end {
            return Err(Error::Parse(format!("truncated bytes @{}", self.off())));
        }
        let start = self.p;
        self.p += len;
        Ok((start, len))
    }

    fn str_at(&self, start: usize, len: usize) -> String {
        String::from_utf8_lossy(&self.buf[start..start + len]).into_owned()
    }

    /// 跳过未知字段。
    fn skip(&mut self, wire: u32) -> Result<()> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => {
                self.fixed64()?;
            }
            2 => {
                self.bytes()?;
            }
            5 => {
                self.fixed32()?;
            }
            _ => {
                return Err(Error::Parse(format!(
                    "bad wire type {wire} @{}",
                    self.off()
                )));
            }
        }
        Ok(())
    }
}

/// 原始 dtype 码（onnx.proto TensorProto.DataType）。
const DT_FLOAT: u64 = 1;
const DT_UINT8: u64 = 2;
const DT_INT8: u64 = 3;
const DT_UINT16: u64 = 4;
const DT_INT16: u64 = 5;
const DT_INT32: u64 = 6;
const DT_INT64: u64 = 7;
const DT_DOUBLE: u64 = 11;
const DT_BOOL: u64 = 9;

fn parse_tensor(r: &mut Reader, force_name: &str) -> Result<Tensor> {
    let mut t = Tensor::default();
    // 权重装载走临时 Vec（一次性，池锁不划算），收尾拷进 F32Buf
    let mut f32v: Vec<f32> = Vec::new();
    let mut raw_dtype: u64 = DT_FLOAT;
    while !r.eof() {
        let key = r.varint()?;
        let field = (key >> 3) as u32;
        let wire = (key & 7) as u32;
        match field {
            1 => {
                // dims（packed varint）
                if wire == 2 {
                    let (s, l) = r.bytes()?;
                    let mut rd = r.sub(s, l);
                    while !rd.eof() {
                        t.shape.push(rd.varint()? as i64);
                    }
                } else {
                    t.shape.push(r.varint()? as i64);
                }
            }
            2 => raw_dtype = r.varint()?,
            4 => {
                // float_data：packed 定长 little-endian f32 流
                if wire == 2 {
                    let (s, l) = r.bytes()?;
                    let n = l / 4;
                    f32v.reserve(n);
                    for i in 0..n {
                        let b = r.buf[s + i * 4..s + i * 4 + 4].try_into().unwrap();
                        f32v.push(f32::from_le_bytes(b));
                    }
                } else {
                    f32v.push(f32::from_bits(r.fixed32()?));
                }
            }
            5 => {
                // int32_data：protobuf 的 int32 是 VARINT 类型，packed 与
                // 非 packed 两种形态都是 varint，不是 fixed32——按 fixed32
                // 读会在短 payload 上越界并带偏整个流
                //（ 注释：症状是 EOF 处 "truncated fixed32"）。
                if wire == 2 {
                    let (s, l) = r.bytes()?;
                    let mut rd = r.sub(s, l);
                    while !rd.eof() {
                        t.i64.push(rd.varint()? as u32 as i32 as i64);
                    }
                } else if wire == 0 {
                    t.i64.push(r.varint()? as u32 as i32 as i64);
                } else if wire == 5 {
                    // 容忍不合规的 fixed32 写入方
                    t.i64.push(r.fixed32()? as i32 as i64);
                }
            }
            7 => {
                // int64_data
                if wire == 2 {
                    let (s, l) = r.bytes()?;
                    let mut rd = r.sub(s, l);
                    while !rd.eof() {
                        t.i64.push(rd.varint()? as i64);
                    }
                } else {
                    t.i64.push(r.varint()? as i64);
                }
            }
            8 => {
                let (s, l) = r.bytes()?;
                t.name = r.str_at(s, l);
            }
            9 => {
                // raw_data：小端，元素宽度取决于 data_type
                let (s, l) = r.bytes()?;
                let n = l;
                match raw_dtype {
                    DT_FLOAT => {
                        let cnt = n / 4;
                        f32v.reserve(cnt);
                        for i in 0..cnt {
                            let b = r.buf[s + i * 4..s + i * 4 + 4].try_into().unwrap();
                            f32v.push(f32::from_le_bytes(b));
                        }
                    }
                    DT_DOUBLE => {
                        let cnt = n / 8;
                        for i in 0..cnt {
                            let b = r.buf[s + i * 8..s + i * 8 + 8].try_into().unwrap();
                            f32v.push(f64::from_le_bytes(b) as f32);
                        }
                    }
                    DT_INT64 => {
                        let cnt = n / 8;
                        t.i64.reserve(cnt);
                        for i in 0..cnt {
                            let b = r.buf[s + i * 8..s + i * 8 + 8].try_into().unwrap();
                            t.i64.push(i64::from_le_bytes(b));
                        }
                    }
                    DT_INT32 => {
                        let cnt = n / 4;
                        for i in 0..cnt {
                            let b = r.buf[s + i * 4..s + i * 4 + 4].try_into().unwrap();
                            t.i64.push(i32::from_le_bytes(b) as i64);
                        }
                    }
                    DT_BOOL | DT_UINT8 | DT_INT8 => {
                        t.i64.reserve(n);
                        for i in 0..n {
                            let v = r.buf[s + i];
                            t.i64.push(if raw_dtype == DT_INT8 {
                                v as i8 as i64
                            } else {
                                v as i64
                            });
                        }
                    }
                    DT_UINT16 => {
                        let cnt = n / 2;
                        for i in 0..cnt {
                            let b = r.buf[s + i * 2..s + i * 2 + 2].try_into().unwrap();
                            t.i64.push(u16::from_le_bytes(b) as i64);
                        }
                    }
                    DT_INT16 => {
                        let cnt = n / 2;
                        for i in 0..cnt {
                            let b = r.buf[s + i * 2..s + i * 2 + 2].try_into().unwrap();
                            t.i64.push(i16::from_le_bytes(b) as i64);
                        }
                    }
                    other => {
                        return Err(Error::Parse(format!(
                            "raw_data: unsupported data_type {other}"
                        )));
                    }
                }
            }
            10 => {
                // double_data：转 f32
                if wire == 2 {
                    let (s, l) = r.bytes()?;
                    let mut rd = r.sub(s, l);
                    while !rd.eof() {
                        f32v.push(f64::from_bits(rd.fixed64()?) as f32);
                    }
                } else {
                    f32v.push(f64::from_bits(r.fixed64()?) as f32);
                }
                if raw_dtype == DT_DOUBLE {
                    raw_dtype = DT_FLOAT;
                }
            }
            _ => r.skip(wire)?,
        }
    }
    t.f32 = qppocr_kernels::buf::F32Buf::from_vec(&f32v);
    if !force_name.is_empty() {
        t.name = force_name.to_string();
    }
    // 归一化：I32/BOOL → I64，F64 → F32（内核只吃这两种）
    t.dtype = match raw_dtype {
        DT_FLOAT => DType::F32,
        DT_DOUBLE => DType::F32,
        _ => DType::I64,
    };
    Ok(t)
}

fn parse_attribute(r: &mut Reader) -> Result<Attribute> {
    let mut a = Attribute::default();
    while !r.eof() {
        let key = r.varint()?;
        let field = (key >> 3) as u32;
        let wire = (key & 7) as u32;
        match field {
            1 => {
                let (s, l) = r.bytes()?;
                a.name = r.str_at(s, l);
            }
            2 => {
                a.f = f32::from_bits(r.fixed32()?);
                a.has_f = true;
            }
            3 => {
                a.i = r.varint()? as i64;
                a.has_i = true;
            }
            4 => {
                let (s, l) = r.bytes()?;
                a.s = r.str_at(s, l);
                a.has_s = true;
            }
            5 => {
                let (s, l) = r.bytes()?;
                let mut rd = r.sub(s, l);
                a.tensor = Some(parse_tensor(&mut rd, "")?);
            }
            7 => {
                // floats（packed fixed32）
                if wire == 2 {
                    let (s, l) = r.bytes()?;
                    let mut rd = r.sub(s, l);
                    while !rd.eof() {
                        a.floats.push(f32::from_bits(rd.fixed32()?));
                    }
                } else {
                    a.floats.push(f32::from_bits(r.fixed32()?));
                }
            }
            8 => {
                // ints（packed varint）
                if wire == 2 {
                    let (s, l) = r.bytes()?;
                    let mut rd = r.sub(s, l);
                    while !rd.eof() {
                        a.ints.push(rd.varint()? as i64);
                    }
                } else {
                    a.ints.push(r.varint()? as i64);
                }
            }
            _ => r.skip(wire)?,
        }
    }
    Ok(a)
}

fn parse_node(r: &mut Reader) -> Result<Node> {
    let mut n = Node::default();
    while !r.eof() {
        let key = r.varint()?;
        let field = (key >> 3) as u32;
        let wire = (key & 7) as u32;
        match field {
            1 => {
                let (s, l) = r.bytes()?;
                n.inputs.push(r.str_at(s, l));
            }
            2 => {
                let (s, l) = r.bytes()?;
                n.outputs.push(r.str_at(s, l));
            }
            3 => {
                let (s, l) = r.bytes()?;
                n.name = r.str_at(s, l);
            }
            4 => {
                let (s, l) = r.bytes()?;
                n.op_type = r.str_at(s, l);
            }
            5 => {
                let (s, l) = r.bytes()?;
                let mut rd = r.sub(s, l);
                n.attrs.push(parse_attribute(&mut rd)?);
            }
            _ => r.skip(wire)?,
        }
    }
    Ok(n)
}

/// 从 ValueInfoProto 里抠 field 1（名字）。
fn value_info_name(r: &mut Reader) -> Option<String> {
    while !r.eof() {
        let k = r.varint().ok()?;
        let f2 = (k >> 3) as u32;
        let w2 = (k & 7) as u32;
        if f2 == 1 && w2 == 2 {
            let (s, l) = r.bytes().ok()?;
            return Some(r.str_at(s, l));
        }
        r.skip(w2).ok()?;
    }
    None
}

fn parse_graph(r: &mut Reader, g: &mut Graph) -> Result<()> {
    // 先攒子块再解析：同款（节点解析与图级字段交错无影响，但保持顺序）
    let mut node_chunks: Vec<(usize, usize)> = Vec::new();
    let mut init_chunks: Vec<(usize, usize)> = Vec::new();
    while !r.eof() {
        let key = r.varint()?;
        let field = (key >> 3) as u32;
        let wire = (key & 7) as u32;
        if wire != 2 {
            // 图级标量不存在，防御性跳过
            r.skip(wire)?;
            continue;
        }
        let (s, l) = r.bytes()?;
        match field {
            1 => node_chunks.push((s, l)),
            2 => g.graph_name = r.str_at(s, l),
            5 => init_chunks.push((s, l)),
            11 => {
                let mut rd = r.sub(s, l);
                if let Some(nm) = value_info_name(&mut rd) {
                    g.inputs.push(nm);
                }
            }
            12 => {
                let mut rd = r.sub(s, l);
                if let Some(nm) = value_info_name(&mut rd) {
                    g.outputs.push(nm);
                }
            }
            _ => {}
        }
    }
    for (s, l) in node_chunks {
        let mut rd = r.sub(s, l);
        g.nodes.push(parse_node(&mut rd)?);
    }
    for (s, l) in init_chunks {
        let mut rd = r.sub(s, l);
        let t = parse_tensor(&mut rd, "")?;
        g.initializers.insert(t.name.clone(), t);
    }
    Ok(())
}

/// 从内存解析 ONNX 模型。`display_name` 只用于报告。
pub fn load_onnx_memory(data: &[u8], display_name: &str) -> Result<Graph> {
    let mut r = Reader::new(data);
    let mut g = Graph::default();
    while !r.eof() {
        let key = r.varint()?;
        let field = (key >> 3) as u32;
        let wire = (key & 7) as u32;
        if wire != 2 {
            if wire == 0 {
                let v = r.varint()?;
                if field == 1 {
                    g.ir_version = v as i64;
                }
            } else {
                r.skip(wire)?;
            }
            continue;
        }
        let (s, l) = r.bytes()?;
        match field {
            2 => g.producer_name = r.str_at(s, l),
            3 => g.producer_version = r.str_at(s, l),
            6 => {
                // doc_string（首行；field 4 是 domain）
                let full = r.str_at(s, l);
                g.doc_string = full.split('\n').next().unwrap_or("").to_string();
            }
            8 => {
                // opset_import：{domain=1, version=2}
                let mut orr = r.sub(s, l);
                let mut ver: i64 = 0;
                let mut dom = String::from("ai.onnx");
                while !orr.eof() {
                    let k = orr.varint()?;
                    let f2 = (k >> 3) as u32;
                    let w2 = (k & 7) as u32;
                    if f2 == 1 && w2 == 2 {
                        let (s2, l2) = orr.bytes()?;
                        dom = orr.str_at(s2, l2);
                    } else if f2 == 2 && w2 == 0 {
                        ver = orr.varint()? as i64;
                    } else {
                        orr.skip(w2)?;
                    }
                }
                if dom == "ai.onnx" || dom.is_empty() || g.opset_version == 0 {
                    g.opset_version = ver;
                }
            }
            7 => {
                let mut rd = r.sub(s, l);
                parse_graph(&mut rd, &mut g)?;
            }
            14 => {
                // metadata_props：{key=1, value=2}
                let mut er = r.sub(s, l);
                let mut k = String::new();
                let mut v = String::new();
                while !er.eof() {
                    let k2 = er.varint()?;
                    let f2 = (k2 >> 3) as u32;
                    let w2 = (k2 & 7) as u32;
                    if f2 == 1 && w2 == 2 {
                        let (s2, l2) = er.bytes()?;
                        k = er.str_at(s2, l2);
                    } else if f2 == 2 && w2 == 2 {
                        let (s2, l2) = er.bytes()?;
                        v = er.str_at(s2, l2);
                    } else {
                        er.skip(w2)?;
                    }
                }
                if !k.is_empty() {
                    g.metadata.insert(k, v);
                }
            }
            _ => {}
        }
    }
    if g.nodes.is_empty() {
        return Err(Error::Parse("model has no nodes (parse failure?)".into()));
    }

    g.model_path = display_name.to_string();
    g.file_bytes = data.len() as u64;

    // Constant 折叠进 initializers，让执行器看到平坦的图。
    // Constant：outputs[0] = 名字，attr value/value_float/value_ints/value_floats。
    {
        let mut kept: Vec<Node> = Vec::with_capacity(g.nodes.len());
        for n in std::mem::take(&mut g.nodes) {
            if n.op_type == "Constant" {
                let out_name = n.outputs[0].clone();
                let mut t = Tensor::default();
                t.name = out_name.clone();
                let mut ok = false;
                for a in &n.attrs {
                    match a.name.as_str() {
                        "value" if a.tensor.is_some() => {
                            let mut tt = a.tensor.clone().unwrap();
                            tt.name = out_name.clone();
                            t = tt;
                            ok = true;
                            break;
                        }
                        "value_float" => {
                            t.dtype = DType::F32;
                            t.f32 = qppocr_kernels::buf::F32Buf::from_vec(&[a.f]);
                            ok = true;
                            break;
                        }
                        "value_floats" => {
                            t.dtype = DType::F32;
                            t.f32 = qppocr_kernels::buf::F32Buf::from_vec(&a.floats);
                            ok = true;
                            break;
                        }
                        "value_int" | "value_ints" => {
                            t.dtype = DType::I64;
                            t.i64 = a.ints.clone();
                            ok = true;
                            break;
                        }
                        _ => {}
                    }
                }
                // 0 维但 1 元素的张量是合法的
                if ok {
                    g.initializers.insert(out_name.clone(), t);
                    continue;
                }
                // 不支持的 Constant 形态——保留节点（执行器会大声报错）
                kept.push(n);
            } else {
                kept.push(n);
            }
        }
        g.nodes = kept;
    }

    // Identity 消除：Identity 是纯改名，导出器的清理趟会成批吐出它们
    //（PP-OCRv6 det 有 147 个）。执行它们等于整张特征图白 memcpy。
    {
        let mut alias: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        for n in &g.nodes {
            if n.op_type == "Identity" && !n.inputs.is_empty() && !n.outputs.is_empty() {
                alias.insert(n.outputs[0].clone(), n.inputs[0].clone());
            }
        }
        if !alias.is_empty() {
            // 解链：a -> b -> c 变成 a -> c
            let resolve = |nm: &str| -> String {
                let mut cur = nm.to_string();
                for _ in 0..64 {
                    match alias.get(&cur) {
                        Some(next) => cur = next.clone(),
                        None => break,
                    }
                }
                cur
            };
            let mut out: Vec<Node> = Vec::with_capacity(g.nodes.len());
            for mut n in std::mem::take(&mut g.nodes) {
                if n.op_type == "Identity" {
                    continue;
                }
                for inn in n.inputs.iter_mut() {
                    if !inn.is_empty() {
                        *inn = resolve(inn);
                    }
                }
                out.push(n);
            }
            for o in g.outputs.iter_mut() {
                *o = resolve(o);
            }
            g.nodes = out;
        }
    }

    // 模型特化：只做结构匹配，不匹配的原样保留。
    g.dropped_identity = crate::graph::optimize::drop_identity_pairs(&mut g);
    g.fused_gelu = crate::graph::optimize::fuse_gelu(&mut g);
    g.folded_bias = crate::graph::optimize::fold_conv_bias(&mut g);
    // 最后跑，让它看到最终的 Conv 节点（fuse_gelu 造出它要匹配的
    // FusedGelu，fold_conv_bias 可能刚给它们补了 bias）。
    g.fused_conv_act = crate::graph::optimize::fuse_conv_activation(&mut g);

    g.node_count = g.nodes.len() as i64;
    for t in g.initializers.values() {
        g.param_count += t.numel();
    }
    Ok(g)
}

/// 从文件解析。
pub fn load_onnx(path: &std::path::Path) -> Result<Graph> {
    let data = std::fs::read(path)
        .map_err(|e| Error::Io(format!("cannot read model file {}: {e}", path.display())))?;
    load_onnx_memory(&data, &path.display().to_string())
}
