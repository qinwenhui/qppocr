# qppocr

A pure-Rust OCR engine for the PP-OCRv6 detection / orientation / recognition
models. It parses and runs the official ONNX files directly — no ONNX Runtime,
no native inference dependency. Kernels are hand-written AVX2 with a scalar
fallback, so one binary runs on any x86-64 host and picks up SIMD at runtime.

## What you get

Given an image, one call returns text lines with reading order, quads in
source-image coordinates, confidences, per-character spans, and stage
timings:

```rust
use qppocr::{Engine, Tier};

let engine = Engine::builder()
    .tier(Tier::Small)          // tiny | small | medium
    .build(std::path::Path::new("models"))?;

let img = qppocr::decode_file("photo.jpg")?;
let result = engine.run(&img)?;

for line in &result.lines {
    println!("{:.3}  {}", line.confidence, line.text);
    // line.pts: quad corners; line.chars: per-character boxes
}
```

`models/` is expected to hold the upstream files laid out as
`{tiny,small,medium}/{det,rec}.onnx` and a shared `cls.onnx`. SHA-256 of the
model files is verified at load time unless verification is disabled.

## Model tiers and presets

- `Tier::Tiny` / `Small` / `Medium` trade accuracy for cost. The models are
  the official upstream ONNX exports; the engine ships no weights.
- Presets: `Speed` (recognizer input height 40, region retry off), `Balanced`
  (default, height 48), `Accuracy` (adds region retry and clutter rejection).
  Region retry re-runs detection on low-confidence regions; it is an
  application-level feature and off by default.
- Orientation classification (0/180) can be disabled per engine
  (`detect_orientation(false)`), saving one pass per line.

## Features

- `image-decode` (default) — JPEG/PNG decoding via pure-Rust codecs,
  including EXIF orientation. Turn it off when the host supplies pixels.
- `parallel` (default) — multithreaded kernels. Off = single-threaded engine.
- `serde` — serializable result types.
- `std` — off for no_std targets; models then load from bytes via
  `ModelSource`.

## CLI

The repository ships a reference CLI (`qppocr-cli`) for batch runs and
per-stage timing inspection:

```bash
qppocr img/*.jpg --tier small --json --workers 8
```

License: MIT OR Apache-2.0.
