//! 逐算子的静态形状推理。
//!
//! 每条规则都注明它镜像的内核/executor 位置——改内核的形状语义时
//! 这里必须同步（判据：tests/plan.rs 的真模型 oracle 对拍）。

use std::collections::HashMap;

use qppocr_core::error::{Error, Result};
use qppocr_core::onnx::model::Node;
use qppocr_core::tensor::DType;

use super::{Const, Value, axes_from, get_i};

fn miss(n: &Node, idx: usize) -> Error {
    Error::Graph(format!(
        "missing input {}",
        n.inputs.get(idx).unwrap_or(&String::new())
    ))
}

fn inp<'a>(n: &Node, idx: usize, values: &'a HashMap<String, Value>) -> Result<&'a Value> {
    values
        .get(n.inputs.get(idx).map(String::as_str).unwrap_or(""))
        .ok_or_else(|| miss(n, idx))
}

/// ONNX auto_pad 解析。镜像 `executor.rs::resolve_pads`（SAME_UPPER/
/// LOWER 的 end-can-overhang 语义原样）。
#[allow(clippy::too_many_arguments)]
fn resolve_pads(
    n: &Node,
    in_h: i64,
    in_w: i64,
    kh: i64,
    kw: i64,
    sh: i64,
    sw: i64,
    dh: i64,
    dw: i64,
) -> (i64, i64, i64, i64) {
    let (mut ph, mut pw, mut peh, mut pew) = (0i64, 0, 0, 0);
    let ap = n.attr("auto_pad").filter(|a| a.has_s).map(|a| a.s.clone());
    if let Some(a) = n.attr("pads") {
        if a.ints.len() >= 4 {
            ph = a.ints[0];
            pw = a.ints[1];
            peh = a.ints[2];
            pew = a.ints[3];
        }
    }
    if ap.as_deref() != Some("SAME_UPPER") && ap.as_deref() != Some("SAME_LOWER") {
        return (ph, pw, peh, pew);
    }
    let (eff_kh, eff_kw) = ((kh - 1) * dh + 1, (kw - 1) * dw + 1);
    let oh = in_h.div_euclid(sh) + i64::from(in_h % sh != 0);
    let ow = in_w.div_euclid(sw) + i64::from(in_w % sw != 0);
    let th = ((oh - 1) * sh + eff_kh - in_h).max(0);
    let tw = ((ow - 1) * sw + eff_kw - in_w).max(0);
    if ap.as_deref() == Some("SAME_UPPER") {
        ph = th / 2;
        peh = th - ph;
        pw = tw / 2;
        pew = tw - pw;
    } else {
        peh = th / 2;
        ph = th - peh;
        pew = tw / 2;
        pw = tw - pew;
    }
    (ph, pw, peh, pew)
}

/// 右对齐广播（镜像 `elementwise.rs::broadcast_meta` 的形状部分）。
fn broadcast(a: &[i64], b: &[i64]) -> Result<Vec<i64>> {
    let r = a.len().max(b.len());
    let dim_at = |s: &[i64], i: usize| -> i64 {
        let rr = s.len();
        if i < rr { s[rr - 1 - i] } else { 1 }
    };
    // 内核按「右起第 i 维」比较；结果表要反转回行主序。
    let mut out = vec![1i64; r];
    for (i, slot) in out.iter_mut().enumerate() {
        let (da, db) = (dim_at(a, i), dim_at(b, i));
        if !(da == db || da == 1 || db == 1) {
            return Err(Error::Graph(format!(
                "广播失败：a[..]={da} vs b[..]={db}（右对齐第 {i} 维）"
            )));
        }
        *slot = da.max(db);
    }
    out.reverse();
    Ok(out)
}

/// slice 区间解析。镜像 `kernels/shape.rs::slice_range`。
fn slice_range(start: i64, end: i64, step: i64, n: i64) -> (i64, i64) {
    let (mut start, mut end) = (start, end);
    if start < 0 {
        start += n;
    }
    if end < 0 && !(step < 0 && end == i64::MIN) {
        end += n;
    }
    if step > 0 {
        start = start.max(0);
        end = end.min(n);
        (start.min(n), start.max(end.min(n)))
    } else {
        start = start.min(n - 1);
        end = end.max(-1);
        (start.max(-1), start.min(end.max(-1)))
    }
}

/// slice 的目标形状（镜像 `slice_tensor` 的形状部分；只支持正步长——
/// 负步长在我们模型里不存在，出现了就点名）。
fn slice_shape(
    x_shape: &[i64],
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
) -> Result<Vec<i64>> {
    let r = x_shape.len();
    let ax: Vec<i64> = if axes.is_empty() {
        (0..starts.len() as i64).collect()
    } else {
        axes.to_vec()
    };
    let st: Vec<i64> = if steps.is_empty() {
        vec![1; starts.len()]
    } else {
        steps.to_vec()
    };
    let mut b = vec![0i64; r];
    let mut e: Vec<i64> = x_shape.to_vec();
    let mut sp = vec![1i64; r];
    for i in 0..ax.len() {
        let a = if ax[i] < 0 { ax[i] + r as i64 } else { ax[i] } as usize;
        let (bb, ee) = slice_range(starts[i], ends[i], st[i], x_shape[a]);
        b[a] = bb;
        e[a] = ee;
        sp[a] = st[i];
    }
    let mut shape = Vec::with_capacity(r);
    for i in 0..r {
        if sp[i] <= 0 {
            return Err(Error::Graph("负步长 slice 未支持".into()));
        }
        shape.push((e[i] - b[i] + sp[i] - 1) / sp[i]);
    }
    Ok(shape)
}

/// 对常量做 slice（数据搬运用；元素量 = 形状链，个位数）。
fn slice_const_i64(
    data: &[i64],
    x_shape: &[i64],
    shape: &[i64],
    b: &[i64],
    sp: &[i64],
) -> Vec<i64> {
    // 旧 strides（行主序）
    let r = x_shape.len();
    let mut strides = vec![1i64; r];
    for i in (0..r.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * x_shape[i + 1];
    }
    let mut out = Vec::with_capacity(shape.iter().product::<i64>() as usize);
    let mut idx = vec![0usize; r];
    'outer: loop {
        let mut flat = 0i64;
        for i in 0..r {
            flat += (b[i] + idx[i] as i64 * sp[i]) * strides[i];
        }
        out.push(data[flat as usize]);
        // 行主序 odometer 递增
        for i in (0..r).rev() {
            idx[i] += 1;
            if idx[i] as i64 >= shape[i] {
                idx[i] = 0;
            } else {
                continue 'outer;
            }
        }
        break;
    }
    out
}

/// Reshape 形状（`-1` 推断、`0` 保留）。镜像 `kernels/shape.rs::reshape_shape`
/// （assert 换成 Err）。
fn reshape_shape(x_shape: &[i64], shape_in: &[i64]) -> Result<Vec<i64>> {
    let total: i64 = x_shape.iter().product();
    let mut known = 1i64;
    let mut infer_idx: isize = -1;
    let mut shape = shape_in.to_vec();
    for (i, &d) in shape_in.iter().enumerate() {
        if d == -1 {
            if infer_idx >= 0 {
                return Err(Error::Graph("reshape: 多个 -1".into()));
            }
            infer_idx = i as isize;
        } else if d == 0 {
            shape[i] = *x_shape
                .get(i)
                .ok_or_else(|| Error::Graph("reshape: 0 越界".into()))?;
            known *= shape[i];
        } else {
            known *= d;
        }
    }
    if infer_idx >= 0 {
        if known == 0 || total % known != 0 {
            return Err(Error::Graph(format!(
                "reshape: -1 推不出（{total} / {known}）"
            )));
        }
        shape[infer_idx as usize] = total / known;
    }
    Ok(shape)
}

/// Squeeze 形状。镜像 `executor.rs::squeeze_shape`。
fn squeeze_shape(x_shape: &[i64], axes: &[i64]) -> Vec<i64> {
    let r = x_shape.len();
    if axes.is_empty() {
        return x_shape.iter().copied().filter(|&d| d != 1).collect();
    }
    let mut drop = vec![false; r];
    for &a in axes {
        let a = if a < 0 { a + r as i64 } else { a } as usize;
        drop[a] = true;
    }
    x_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !drop[*i])
        .map(|(_, &d)| d)
        .collect()
}

/// Transpose 形状（perm 空 = 全反转）。镜像 `transpose_tensor`。
fn transpose_shape(x_shape: &[i64], perm: &[i64]) -> Result<Vec<i64>> {
    let r = x_shape.len();
    let p: Vec<usize> = if perm.is_empty() {
        (0..r).rev().collect()
    } else {
        perm.iter()
            .map(|&a| (if a < 0 { a + r as i64 } else { a }) as usize)
            .collect()
    };
    if p.len() != r || p.iter().collect::<std::collections::HashSet<_>>().len() != r {
        return Err(Error::Graph("transpose: perm 非法".into()));
    }
    Ok(p.iter().map(|&i| x_shape[i]).collect())
}

/// 推理单个节点的输出。
pub(crate) fn infer_node(n: &Node, values: &HashMap<String, Value>) -> Result<Value> {
    let op = n.op_type.as_str();
    let passthrough = |dtype: DType| -> Result<Value> {
        let x = inp(n, 0, values)?;
        Ok(Value {
            shape: x.shape.clone(),
            dtype,
            konst: None,
        })
    };
    match op {
        // ---- 形状保持（值计算留在内核）----
        "BatchNormalization" | "Relu" | "Sigmoid" | "HardSigmoid" | "FusedGelu" | "Erf"
        | "Sqrt" | "Clip" | "Softmax" => passthrough(DType::F32),
        "Identity" => {
            let x = inp(n, 0, values)?;
            Ok(x.clone())
        }
        // ---- 四维卷积族 ----
        "Conv" => {
            let x = inp(n, 0, values)?;
            let w = inp(n, 1, values)?;
            if x.shape.len() != 4 || w.shape.len() != 4 {
                return Err(Error::Graph("Conv: 只支持 NCHW 4D".into()));
            }
            // strides/dilations 两元素取法镜像 executor 的 strides_of /
            // dilations_of；group 不影响输出形状。
            let strides = n
                .attr("strides")
                .map(|a| a.ints.clone())
                .unwrap_or_default();
            let (sh, sw) = if strides.len() == 2 {
                (strides[0], strides[1])
            } else {
                (1, 1)
            };
            let dils = n
                .attr("dilations")
                .map(|a| a.ints.clone())
                .unwrap_or_default();
            let (dh, dw) = if dils.len() == 2 {
                (dils[0], dils[1])
            } else {
                (1, 1)
            };
            // 内核不支持 dilation（conv.rs 的 assert），planner 同步这一事实。
            if dh != 1 || dw != 1 {
                return Err(Error::Graph("Conv: dilation 不支持（内核同款限制）".into()));
            }
            let (kh, kw) = (w.shape[2], w.shape[3]);
            let (ph, pw, peh, pew) =
                resolve_pads(n, x.shape[2], x.shape[3], kh, kw, sh, sw, dh, dw);
            // 镜像 conv.rs::conv2d_out_shape 的整除公式（原始 kh，无膨胀项）
            let oh = (x.shape[2] + ph + peh - kh) / sh + 1;
            let ow = (x.shape[3] + pw + pew - kw) / sw + 1;
            Ok(Value {
                shape: vec![x.shape[0], w.shape[0], oh, ow],
                dtype: DType::F32,
                konst: None,
            })
        }
        "ConvTranspose" => {
            let x = inp(n, 0, values)?;
            let w = inp(n, 1, values)?;
            // 内核只支持 2x2 s2 p0（conv.rs 的 assert）——planner 同步
            // 这个事实：输出 = [N, w.shape[1]（C_out）, H*2, W*2]。
            let strides = n
                .attr("strides")
                .map(|a| a.ints.clone())
                .unwrap_or_default();
            let (sh, sw) = if strides.len() == 2 {
                (strides[0], strides[1])
            } else {
                (1, 1)
            };
            let pads = n.attr("pads").map(|a| a.ints.clone()).unwrap_or_default();
            let (ph, pw) = if pads.len() >= 2 {
                (pads[0], pads[1])
            } else {
                (0, 0)
            };
            let (kh, kw) = (w.shape[2], w.shape[3]);
            let supported = kh == 2 && kw == 2 && sh == 2 && sw == 2 && ph == 0 && pw == 0;
            if !supported {
                return Err(Error::Graph(
                    "ConvTranspose: 内核只支持 2x2 s2 p0（conv.rs 同款限制）".into(),
                ));
            }
            Ok(Value {
                shape: vec![x.shape[0], w.shape[1], x.shape[2] * 2, x.shape[3] * 2],
                dtype: DType::F32,
                konst: None,
            })
        }
        // ---- 池化 ----
        "GlobalAveragePool" => {
            let x = inp(n, 0, values)?;
            Ok(Value {
                shape: vec![x.shape[0], x.shape[1], 1, 1],
                dtype: DType::F32,
                konst: None,
            })
        }
        "AveragePool" | "MaxPool" => {
            let x = inp(n, 0, values)?;
            let ks = n
                .attr("kernel_shape")
                .filter(|a| a.ints.len() == 2)
                .ok_or_else(|| Error::Graph("kernel_shape required".into()))?;
            let strides = n
                .attr("strides")
                .map(|a| a.ints.clone())
                .unwrap_or_default();
            let (sh, sw) = if strides.len() == 2 {
                (strides[0], strides[1])
            } else {
                (1, 1)
            };
            let (ph, pw, peh, pew) = resolve_pads(
                n, x.shape[2], x.shape[3], ks.ints[0], ks.ints[1], sh, sw, 1, 1,
            );
            // 镜像 pool2d.rs:48-49 的 saturating_sub 公式
            let oh = (x.shape[2] + ph + peh).saturating_sub(ks.ints[0]) / sh + 1;
            let ow = (x.shape[3] + pw + pew).saturating_sub(ks.ints[1]) / sw + 1;
            Ok(Value {
                shape: vec![x.shape[0], x.shape[1], oh, ow],
                dtype: DType::F32,
                konst: None,
            })
        }
        // ---- 逐元素（广播）----
        "Add" | "Sub" | "Mul" | "Div" | "Pow" => {
            let a = inp(n, 0, values)?;
            let b = inp(n, 1, values)?;
            let shape = broadcast(&a.shape, &b.shape).map_err(|e| {
                Error::Graph(format!(
                    "{e}；inputs: {} = {:?}, {} = {:?}",
                    n.inputs[0], a.shape, n.inputs[1], b.shape
                ))
            })?;
            Ok(Value {
                shape,
                dtype: DType::F32,
                konst: None,
            })
        }
        "MulAddScale" => {
            let f = inp(n, 0, values)?;
            Ok(Value {
                shape: f.shape.clone(),
                dtype: DType::F32,
                konst: None,
            })
        }
        // ---- Resize：目标尺寸来自常量（sizes/scales）----
        "Resize" => {
            let x = inp(n, 0, values)?;
            let (oh, ow): (i64, i64);
            if n.inputs.len() >= 4 && !n.inputs[3].is_empty() {
                let sizes = values
                    .get(&n.inputs[3])
                    .and_then(|v| v.konst.as_ref())
                    .ok_or_else(|| {
                        Error::Graph(format!("Resize 的 sizes {} 非装载期常量", n.inputs[3]))
                    })?
                    .as_i64();
                // 镜像 executor：扁平数组末两位（不是按 rank）
                oh = sizes[sizes.len() - 2];
                ow = sizes[sizes.len() - 1];
            } else if n.inputs.len() >= 3 && !n.inputs[2].is_empty() {
                let sc = values
                    .get(&n.inputs[2])
                    .and_then(|v| v.konst.as_ref())
                    .ok_or_else(|| {
                        Error::Graph(format!("Resize 的 scales {} 非装载期常量", n.inputs[2]))
                    })?
                    .as_f32();
                let (sh_, sw_) = (sc[sc.len() - 2], sc[sc.len() - 1]);
                // 镜像 executor 的 as usize 截断
                oh = (x.shape[2] as f32 * sh_) as i64;
                ow = (x.shape[3] as f32 * sw_) as i64;
            } else {
                return Err(Error::Graph("Resize: no scales/sizes".into()));
            }
            Ok(Value {
                shape: vec![x.shape[0], x.shape[1], oh, ow],
                dtype: DType::F32,
                konst: None,
            })
        }
        // ---- 形状/常量链 ----
        "Shape" => {
            let x = inp(n, 0, values)?;
            Ok(Value {
                shape: vec![x.shape.len() as i64],
                dtype: DType::I64,
                konst: Some(Const::I64(x.shape.clone())),
            })
        }
        "Cast" => {
            let x = inp(n, 0, values)?;
            let to = get_i(n.attr("to"), 1);
            let dtype = if to == 1 || to == 10 {
                DType::F32
            } else {
                DType::I64
            };
            let konst = x.konst.as_ref().map(|c| match dtype {
                DType::F32 => Const::F32(c.as_f32()),
                DType::I64 => Const::I64(c.as_i64()),
            });
            Ok(Value {
                shape: x.shape.clone(),
                dtype,
                konst,
            })
        }
        "Slice" => {
            let x = inp(n, 0, values)?;
            // starts/ends/axes/steps：输入（opset 13+）或属性（≤9）
            let (starts, ends, axes, steps) =
                if n.inputs.len() >= 3 && !n.inputs[1].is_empty() && !n.inputs[2].is_empty() {
                    let grab = |nm: &str| -> Result<Vec<i64>> {
                        values
                            .get(nm)
                            .and_then(|v| v.konst.as_ref())
                            .map(|c| c.as_i64())
                            .ok_or_else(|| Error::Graph(format!("Slice: {nm} 非装载期常量")))
                    };
                    let axes = if n.inputs.len() > 3 && !n.inputs[3].is_empty() {
                        grab(&n.inputs[3])?
                    } else {
                        Vec::new()
                    };
                    let steps = if n.inputs.len() > 4 && !n.inputs[4].is_empty() {
                        grab(&n.inputs[4])?
                    } else {
                        Vec::new()
                    };
                    (grab(&n.inputs[1])?, grab(&n.inputs[2])?, axes, steps)
                } else {
                    let attr_ints = |nm: &str| -> Vec<i64> {
                        n.attr(nm).map(|a| a.ints.clone()).unwrap_or_default()
                    };
                    (
                        attr_ints("starts"),
                        attr_ints("ends"),
                        attr_ints("axes"),
                        Vec::new(),
                    )
                };
            let shape = slice_shape(&x.shape, &starts, &ends, &axes, &steps)?;
            // i64 常量链折叠（F32 的 Slice 出现在热路径上——数据留给内核）
            let konst = match (&x.konst, x.dtype) {
                (Some(Const::I64(data)), DType::I64) => {
                    let r = x.shape.len();
                    let ax: Vec<i64> = if axes.is_empty() {
                        (0..starts.len() as i64).collect()
                    } else {
                        axes.clone()
                    };
                    let mut b = vec![0i64; r];
                    let mut sp = vec![1i64; r];
                    for i in 0..ax.len() {
                        let a = if ax[i] < 0 { ax[i] + r as i64 } else { ax[i] } as usize;
                        let (bb, ee) = slice_range(
                            starts[i],
                            ends[i],
                            if steps.is_empty() { 1 } else { steps[i] },
                            x.shape[a],
                        );
                        b[a] = bb;
                        sp[a] = if steps.is_empty() { 1 } else { steps[i] };
                        let _ = ee;
                    }
                    Some(Const::I64(slice_const_i64(data, &x.shape, &shape, &b, &sp)))
                }
                _ => None,
            };
            Ok(Value {
                shape,
                dtype: x.dtype,
                konst,
            })
        }
        "Concat" => {
            let axis = get_i(n.attr("axis"), 0);
            let mut xs: Vec<&Value> = Vec::new();
            for nm in &n.inputs {
                if nm.is_empty() {
                    continue;
                }
                xs.push(
                    values
                        .get(nm)
                        .ok_or_else(|| Error::Graph(format!("Concat: missing {nm}")))?,
                );
            }
            let r = xs[0].shape.len();
            let axis = if axis < 0 { axis + r as i64 } else { axis } as usize;
            let mut shape = xs[0].shape.clone();
            for x in &xs[1..] {
                if x.shape.len() != r {
                    return Err(Error::Graph("Concat: 秩不一致".into()));
                }
                for (i, (&d0, &d1)) in shape.iter().zip(x.shape.iter()).enumerate() {
                    if i != axis && d0 != d1 {
                        return Err(Error::Graph("Concat: 非轴维不一致".into()));
                    }
                }
                shape[axis] += x.shape[axis];
            }
            // i64 常量折叠（镜像 concat_any 的轴拼接）
            let all_const = xs.iter().all(|x| matches!(x.konst, Some(Const::I64(_))));
            let konst = if all_const {
                let mut data: Vec<i64> = Vec::new();
                for x in &xs {
                    data.extend_from_slice(x.konst.as_ref().unwrap().as_i64().as_slice());
                }
                Some(Const::I64(data))
            } else {
                None
            };
            Ok(Value {
                shape,
                dtype: xs[0].dtype,
                konst,
            })
        }
        "Transpose" => {
            let x = inp(n, 0, values)?;
            let perm = n.attr("perm").map(|a| a.ints.clone()).unwrap_or_default();
            let shape = transpose_shape(&x.shape, &perm)?;
            let konst = match (&x.konst, x.dtype) {
                (Some(Const::I64(data)), DType::I64) => {
                    // 通用置换：输出坐标 → 逆 perm 得源坐标 → 源扁平偏移。
                    let r = x.shape.len();
                    let p: Vec<usize> = if perm.is_empty() {
                        (0..r).rev().collect()
                    } else {
                        perm.iter()
                            .map(|&a| (if a < 0 { a + r as i64 } else { a }) as usize)
                            .collect()
                    };
                    let mut strides = vec![1i64; r];
                    for i in (0..r.saturating_sub(1)).rev() {
                        strides[i] = strides[i + 1] * x.shape[i + 1];
                    }
                    let mut out = Vec::with_capacity(data.len());
                    let mut idx = vec![0i64; r];
                    'outer: loop {
                        let mut src_flat = 0i64;
                        for (i, &pi) in p.iter().enumerate() {
                            src_flat += idx[i] * strides[pi];
                        }
                        out.push(data[src_flat as usize]);
                        for i in (0..r).rev() {
                            idx[i] += 1;
                            if idx[i] >= shape[i] {
                                idx[i] = 0;
                            } else {
                                continue 'outer;
                            }
                        }
                        break;
                    }
                    Some(Const::I64(out))
                }
                _ => None,
            };
            Ok(Value {
                shape,
                dtype: x.dtype,
                konst,
            })
        }
        "Reshape" => {
            let x = inp(n, 0, values)?;
            let shp = values
                .get(&n.inputs[1])
                .and_then(|v| v.konst.as_ref())
                .map(|c| c.as_i64())
                .ok_or_else(|| Error::Graph("Reshape: 目标形状非装载期常量".into()))?;
            let shape = reshape_shape(&x.shape, &shp)?;
            let konst = x.konst.as_ref().map(|c| match c {
                Const::I64(v) => Const::I64(v.clone()),
                Const::F32(v) => Const::F32(v.clone()),
            });
            Ok(Value {
                shape,
                dtype: x.dtype,
                konst,
            })
        }
        "Squeeze" => {
            let x = inp(n, 0, values)?;
            let axes = axes_from(n, values)?;
            let shape = squeeze_shape(&x.shape, &axes);
            let konst = x.konst.as_ref().map(|c| match c {
                Const::I64(v) => Const::I64(v.clone()),
                Const::F32(v) => Const::F32(v.clone()),
            });
            Ok(Value {
                shape,
                dtype: x.dtype,
                konst,
            })
        }
        "Unsqueeze" => {
            let x = inp(n, 0, values)?;
            let axes = axes_from(n, values)?;
            let rr = x.shape.len() + axes.len();
            let mut ins = vec![false; rr];
            for &a in &axes {
                let idx = if a < 0 { a + rr as i64 } else { a } as usize;
                if idx >= rr {
                    return Err(Error::Graph("Unsqueeze: axis 越界".into()));
                }
                ins[idx] = true;
            }
            let mut shape = Vec::with_capacity(rr);
            let mut k = 0usize;
            for &is_ins in ins.iter() {
                shape.push(if is_ins {
                    1
                } else {
                    let v = x.shape[k];
                    k += 1;
                    v
                });
            }
            let konst = x.konst.as_ref().map(|c| match c {
                Const::I64(v) => Const::I64(v.clone()),
                Const::F32(v) => Const::F32(v.clone()),
            });
            Ok(Value {
                shape,
                dtype: x.dtype,
                konst,
            })
        }
        "ReduceMean" => {
            let x = inp(n, 0, values)?;
            let axes = axes_from(n, values)?;
            let r = x.shape.len();
            let keep = get_i(n.attr("keepdims"), 1) != 0;
            let mut red = vec![false; r];
            if axes.is_empty() {
                red.iter_mut().for_each(|v| *v = true);
            }
            for &a in &axes {
                let a = if a < 0 { a + r as i64 } else { a } as usize;
                red[a] = true;
            }
            let shape: Vec<i64> = if keep {
                x.shape
                    .iter()
                    .zip(red.iter())
                    .map(|(&d, &rd)| if rd { 1 } else { d })
                    .collect()
            } else {
                x.shape
                    .iter()
                    .zip(red.iter())
                    .filter(|&(_, rd)| !*rd)
                    .map(|(&d, _)| d)
                    .collect()
            };
            Ok(Value {
                shape,
                dtype: DType::F32,
                konst: None,
            })
        }
        "MatMul" => {
            let a = inp(n, 0, values)?;
            let b = inp(n, 1, values)?;
            let (ra, rb) = (a.shape.len(), b.shape.len());
            if ra < 2 || rb < 2 {
                return Err(Error::Graph("matmul rank".into()));
            }
            let (m, k) = (a.shape[ra - 2], a.shape[ra - 1]);
            let (k2, n_dim) = (b.shape[rb - 2], b.shape[rb - 1]);
            if k != k2 {
                return Err(Error::Graph("matmul inner dim mismatch".into()));
            }
            // 右对齐批次广播（镜像 shape.rs::matmul）
            let batch = broadcast(&a.shape[..ra - 2], &b.shape[..rb - 2])?;
            let mut shape = batch;
            shape.push(m);
            shape.push(n_dim);
            Ok(Value {
                shape,
                dtype: DType::F32,
                konst: None,
            })
        }
        other => Err(Error::Graph(format!(
            "planner 不支持的算子: {other}（内核集合 = executor 的算子集）"
        ))),
    }
}
