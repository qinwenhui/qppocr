# qppocr-core

The engine inside qppocr: a hand-written ONNX reader, a graph optimizer, an
arena executor, and the detection / orientation / recognition pipeline built
on top of them. No external inference runtime — models are parsed, scheduled,
and run by this crate plus `qppocr-kernels`.

## Components

- **onnx** — protobuf parsing for ONNX files without a protobuf dependency.
  Reads ModelProto/GraphProto, initializers, attributes, and metadata
  (the recognizer ships its dictionary as a metadata property).
- **graph::optimize** — load-time rewrites, applied only when the pattern
  matches: constant folding, Identity elimination, GELU subgraph collapse
  (Div/Erf/Add/Mul/Mul to one node), Conv+activation fusion,
  Conv+BatchNorm folding (affine folded into weights and bias),
  Conv+ResidualAdd fusion into the conv store pass, and per-channel bias
  absorption. Every rewrite is value-preserving; changes to floating-point
  rounding are gated on corpus-level output checks.
- **executor** — a `Session` runs a graph into an arena of reference-counted
  tensors. Readiness scheduling with an indegree counter keeps memory
  residency low; buffers come from the kernels crate pool. Outputs are
  deterministic across thread counts.
- **pipeline** — det → cls → rec. DB postprocess (binarize, dilate, connected
  components, min-area rect, unclip), perspective crop with rotation-aware
  packing, 0/180 orientation classification, CTC decode with per-character
  coordinate spans mapped back to source-image coordinates.

## Running a model directly

`Session` is usable for any ONNX graph whose operators the kernels support —
not only the bundled PP-OCR models:

```rust
use qppocr_core::executor::Session;
use qppocr_core::tensor::{DType, Tensor};

let session = Session::open(std::path::Path::new("model.onnx"))?;
let input = Tensor {
    name: "x".into(),
    shape: vec![1, 3, 64, 64],
    dtype: DType::F32,
    f32: qppocr_kernels::buf::F32Buf::from_vec(&input_data),
    i64: Vec::new(),
};
let outputs = session.run(vec![("x".into(), input)])?;
```

Most applications should use the `qppocr` facade, which owns model loading,
image decoding, and the pipeline configuration.

License: MIT OR Apache-2.0.
