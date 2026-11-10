# qppocr-kernels

Numerical kernels for the qppocr inference engine: GEMM, 2-D convolution,
pooling, resize, and activation primitives, plus the thread pool and tensor
buffer machinery they share. This is the only crate in the workspace that
contains `unsafe`, and it is the only one that talks to SIMD intrinsics.

## What is in it

- **GEMM** — panel micro-kernels (AVX2 on x86-64, NEON on aarch64) with a
  scalar fallback. Supports bias
  and an activation epilogue applied in the store pass, so fused operators do
  not re-read the output. An implicit-GEMM entry point (`sgemm_bptrs_serial`)
  lets strided convolutions run without materializing the im2col patch matrix.
- **conv2d** — one entry point that dispatches by shape: dedicated 1x1 GEMM
  path, grouped im2col path with row tiling, depthwise path over a padded
  plane with register accumulation, and ConvTranspose (2x2, stride 2).
- **Fusion epilogues** — bias, activation (GELU/ReLU), BatchNorm affine
  (folded into weights at load time), and residual add can all be folded into
  the conv store pass through `conv2d_res` / `sgemm_res`.
- **Pooling / resize** — max/avg pooling, global average pool, bilinear and
  nearest resize, 2x2 dilate.
- **Activations** — vectorized erf/exp polynomials. The scalar versions are
  exact lane-for-lane mirrors of the vector versions, so scalar and SIMD runs
  produce bit-identical output — on both vector backends.
- **par** — a small fork-join thread pool with batch wakeup, main-thread
  participation, and a serial-execution mode that keeps split decisions
  identical to the parallel path.
- **buf::F32Buf** — pooled tensor buffers. Blocks are returned to size-class
  buckets instead of the system allocator, which avoids page faults on every
  node output.

CPU features are detected at runtime; binaries built without any special
`target-cpu` setting still take the AVX2 path when the host supports it. On
aarch64, NEON is part of the baseline ISA and is always used. Kernel dispatch
lives in one module (`arch`): adding a backend is one file plus one arm per
entry point, and kernel code itself contains no architecture conditionals.

## When to use it directly

Most users want the `qppocr` facade. Use this crate directly if you need the
kernels standalone — for example to run convolutions from your own graph
runtime, or to build image pipelines out of the resize/normalize primitives.

```rust
use qppocr_kernels::activation::Activation;
use qppocr_kernels::buf::F32Buf;
use qppocr_kernels::conv::{conv2d, ConvParams};

// x: NCHW input, w: [out_ch, in_ch, kh, kw]
let mut y = F32Buf::new();
let out_shape = conv2d(
    &x, &[1, 3, 64, 64],
    &w, &[16, 3, 3, 3],
    Some(&bias),
    &ConvParams { sh: 1, sw: 1, ph: 1, pw: 1, peh: 1, pew: 1, dh: 1, dw: 1, group: 1 },
    &Activation::relu(),
    &mut y,
);
```

Float kernels are deterministic by construction: split and accumulation order
is fixed and does not depend on thread count.

## Features

- `parallel` (default) — thread pool and forking. Turn it off for
  single-threaded or no-std-adjacent builds; every kernel then runs inline.

License: MIT OR Apache-2.0.
