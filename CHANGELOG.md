# Changelog

本项目的全部显著变化都记录在这个文件里。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。
每个版本写清**行为变化**，尤其是默认值变化（DESIGN.md §10）。

## [Unreleased]

### 新增

- workspace 骨架：`qppocr`（门面）/ `qppocr-core`（引擎主体）/
  `qppocr-kernels`（唯一允许 `unsafe` 的算子内核）/ `qppocr-cli`（参考 CLI）。
- CI：构建 + 测试（三平台）、MSRV（1.85）、clippy、rustfmt、cargo-deny。
- 内核标量版全量移植（阶段 1a）：sgemm/im2col/conv2d/convtranspose、
  广播二元、激活（erf/exp 多项式系数照抄）、池化、resize、形状类、
  matmul/batchnorm。并行走 rayon（feature 门控），fork 阈值照搬 tuning.hpp。
- 内核测试：test_ops.cpp 用例集移植 + 补充覆盖（softmax/pool/resize/
  transpose/slice/concat/matmul/reduce_mean/convtranspose），自包含、
  无需模型与语料；并行与单线程两种配置都过。
- sgemm 的 AVX2+FMA 微内核（4×4 分块、尾部面板按 nn 门控加载）与
  逐位对拍测试：9 组形状 × bias/serial 组合，AVX2 与标量输出逐位相同。
- GEMM 性能对拍（交错 5 轮取最好）：Rust/C++ = 0.80~0.87x，判据
  ≤1.05x 大幅超额（tools/migration-bench/README.md）。
- 其余算子的 AVX2 向量路径与逐位对拍：激活五件套、二元平坦/run、
  depthwise 内层、convT interleave、softmax 向量相、pool 2x2 可分
  快路径（含钳制行哨兵修复）。5 组 bitexact 测试全绿。
- ONNX 解析 + 图优化 + 执行器（阶段 2）：手写 protobuf 读取器
  （opset 7/11/14 全实测）、Constant 折叠、Identity 消除、GELU 子图
  坍缩、bias 折叠、conv+act 融合、带引用计数释放的 arena 执行器。
  上游 det(opset 14)/rec 前向通过。
- 逐位对拍（docs/CROSSCHECK.md）：cls 251/251 节点逐字节一致；rec
  76/77、det 107/177（其余为同一 libm 尾巴根因的 ≤1 ulp 传播）。
  抓出并修复七类真错误，其中 GCC「intrinsic 之间也做 FMA 收缩」
  （汇编实证）是最大的一类。
- det → cls → rec 完整流水线（阶段 3）：DB 后处理（二值化/膨胀/连通域/
  旋转卡壳/box_score/unclip）、透视裁剪（8×8 单应求解）、凸包框合并、
  0/180 方向分类、按宽度分批、CTC 解码、像素空格（rec_space_gap）、
  区域重试。36 参数照搬 tuning.hpp。
- 文本级对拍：上游原件全套模型（C++ 跑不了的配置）在 testdata 15 张图
  114 行上与 C++（转换版）**113 行逐字符相同**；同时完成 §4.4 的
  上游参数复验。修复区域重试偏移、上游 cls 的 attribute 式 Slice
  （C++ 在此段错误）、cls_view 开窗对上游模型有害三个问题。
- 性能：池化缓冲（pool.hpp/buf.hpp 移植，阶段 3 遗留的短板）。
  `F32Buf` 按 2^k 分桶回收块（保守默认 64 MB 上限，§6.3 不照搬 C++
  的 256 MB）；GEMM/binary/conv 等写满型内核走 `resize_uninit`，
  权重不再每次 run 克隆（arena 两级查找）。mixed.png 端到端
  143→107 ms（C++ 80 ms，1.34x；此前 545 ms）。binary_op 曾三趟
  内存流量（清零分配 + 计算 + from_vec 拷贝）→ 单趟。附带
  QPPOCR_PROF=1 的 per-op/per-shape profile（对标 C++ LEAN_PROF2）。
  对拍无回归：逐节点 76/77、文本 113/114 维持。
- perf(conv): im2col 串行化修复嵌套并行——micro-bench 中 stride-2 的
  im2col 路径从 1.93x 慢反超到 0.74~0.83x 快。根因：C++ 的 im2col 带
  serial 参数（conv 的 tile 调用全部传 true），我的移植丢了这层语义，
  每个 2 行 tile 又嵌套进 rayon（160 个微任务吃掉 1.2ms）。附带
  conv_bench / im2col_split 两个分解计时 example。
- feat(api): 外部接入反馈的系统性修复（Tauri 评估 + GUI 勘误清单）：
  - **serde 接线做实**（原为死开关）：`OcrResult`/`TextLine`/`Timings`/
    `PipelineConfig`（core）与 `Tier`/`Preset`/`Config`/`Advanced`（facade）
    挂派生；`Tier`/`Preset` 序列化为 `"tiny"`/`"balanced"` 小写字符串，
    `Config`/`Advanced` 缺省字段取默认——**部分 JSON 即预设覆盖语义**，
    下游不再需要手写镜像 DTO
  - **`detect_orientation` 默认 false→true**（倒置翻正是 OCR 引擎的预期
    默认，对齐 PaddleOCR 与本 crate CLI；无 cls.onnx 自动跳过）；
    且 **false 现在真跳过 cls 阶段**（原来只把阈值设无穷大——cls 照
    加载照推理，白付时间与内存）
  - `PipelineConfig` 从 facade 导出（`engine.config()` 的返回类型可命名）
  - `Engine` 手写 `Debug`（摘要式，支持 `expect_err` 等错误处理路径）
  - **字典/模型错配硬校验**：rec 输出类别数 ≠ 字符表长度时首次 run 报
    `Err`（原注释承诺过未实现；错字典曾静默整表错位——`ctc_decode`
    越界静默跳过）。删行字典实测拦截
  - 验证 example：`api_verify`（`--features serde`，七项断言全过）
- feat(api): `TextLine::retried` 逐行区域重试标记——下游需要知道**哪几行**
  被重读（此前只有整图 `num_det_retried`）。true = 来自重试遍（首遍弱行
  被整体替换）；注意语义是「来自第二遍」而非「必然更好」（img-030 实测
  替换行 conf 0.28 仍弱）。`retry_flags` example 八图验证不变量。
- fix(api): 门面解码 re-export 挂 `image-decode` 门控——`default-features =
  false` 时 `decode_file/decode_bytes/probe_dimensions` 不存在，无条件
  `pub use` 让整个 crate 编不过（外部用户实测）。七组 feature 矩阵全过。
- feat(api): **逐字坐标**（对标 PaddleOCR `return_word_box`）：
  `TextLine.chars: Vec<CharSpan>`——每字符四角点四边形（原图坐标，
  与 `pts` 同约定），空格框 = 裁剪图真实空白 run。CTC 时间步往回映射，
  性能损耗 ≈ 0（rec_post 不变），穿过全部坐标变换链（翻转/竖排转正/
  重试偏移等）；`char_boxes` example 三图全过，文本逐字符不变。
- fix(api): SHA 校验失败从 `panic!` 改为 `Err(Error::Model)`——库不该替
  宿主应用决定崩掉（外部评估者实测报告；模型损坏/调包是数据问题，
  应用层要能接住并引导用户）。提示文案不变（期望/实际值 + 跳过校验
  的出路）。损坏模型复测：返回 Err 不再 panic。
- feat(cli): 批量 `--workers`（进程扇出，C++ `auto_workers`/`proc_pool`
  的移植）——100 图 14.0→5.9 s（C++ 5.7 s，平手）。worker 数自动
  （头部探测平均 MP：大图 cap 2 / 小图 cap 8 / 核数减半）或显式；
  连续切块按序拼接保持输入序与 JSON 合并零解析；扇出 vs 顺序同线程数
  0/100 差异。途中实测并否决了进程内「fork 宽度收窄」方案：fork_mu
  串行化下收窄反而降并行度，且 gemm 分轴随线程数变会破坏位级一致。
  同步新增 `qppocr::thread_count()`（池大小查询）与
  `qppocr::probe_dimensions()`（头部探测，不解码像素）。
- perf(mem)+bench(small): 权重单份化 + small 档全量对比。
  `Session::open`/`from_memory` 的 `initializers.clone()` 改 `mem::take`
  ——权重（small ≈33 MB、tiny ≈7 MB）原本双份常驻。实测 PeakWS：
  small receipt 213→181 MB（反超 C++ 194 约 7%）、small big 225→191
  （反超 15%）、tiny receipt/big 各降 8 MB（反超 10~12%）。
  bench_pipeline.py 参数化 `--tier`；small 档成绩（BENCH.md §4b）：
  单图中位 0.796x、K=2 **0.858x 反超**（tiny 的 K=2 差距随计算密度
  增大消失，印证串行段争用定位）、K=4 0.649x、准确率 48.3%/CER
  39.70% 两边一致。
- feat(api): `Advanced` 从 10 项补齐到全部 29 项行为参数（tuning.hpp
  对账：C++ `OcrConfig` 30 个行为参数——除 cls_model 走构造参数外
  全量同名同默认值；此前 19 项含 `enhance_contrast`/`upscale`/
  `det_max_side` 在公开 API 不可达）。闭包初始值仍取预设生效后的值，
  覆盖顺序 预设 → Config → Advanced 不变。功能验证：enhance+upscale=2
  +det_max_side=1280 组合实跑通过。
- perf(mem): det 前处理消除两处全图拷贝——`crop_src` 改借用（对齐
  C++ 的指针语义，仅 enhance_contrast 开启才物化）、超帽路径免
  「先克隆再被 resize 替换」的瞬时整份峰值、`work` 于 db_postprocess
  后显式释放。big.png（3000×2000）PeakWS 152.5→135.6 MB（反超
  C++ 141.8 约 4.3%），receipt 122.3 MB 维持领先（C++ 129.8）；
  文本逐字符一致、速度无回归（顺手省掉每 run 一次全图 memcpy）。
- feat(par): 线程数管道接通——`par::set_threads`（池首用时定容，显式
  请求不设 16 上限，对齐 C++ `resolve_threads`）+ `QPPOCR_THREADS`
  环境变量（对标 `LEAN_THREADS`）+ `EngineBuilder::threads`/CLI
  `--threads` 落地（此前为装饰性配置）。join 改自适应（自旋 2048 轮
  后让出，无争用零成本、核满时不烧核挡 straggler）。新增
  `tools/sweep_parallel.py` 并行度扫描（BENCH.md §5）：单进程延迟
  最优 t=14~16（默认不动）；吞吐峰在总线程 ≈1.2~1.5× 逻辑核
  （rust K=6×t=3 = 25.0 img/s 追平 cpp 峰值）；K=2 差距定位为
  串行段内存争用底差 8%（t=1 无池仍在）× 池放大，非唤醒机制
  （信号量版实测更慢，已回退）。
- perf(pool): 移植 C++ fork-join 线程池（util.hpp `ThreadPool` Windows
  路径）替代 rayon——批量唤醒、主线程参与、自旋 join、参与者计数屏障
  （epoch 推进被 join 门控 ⇒ 栈上 Job 生存期安全）。跨线程并发
  `fork_join` 用 `fork_mu` 串行化而非 C++ 的挂死语义；drop rayon
  依赖（kernels 零第三方依赖）。TP_CHUNKS 回归 tuning.hpp 默认 8。
  端到端交错基准（bench v3）：Σ中位 rust 563.5 vs C++ 572.2 ms
  （**0.985x，整体反超**；v2 为 1.379x），det 单模型 33.5→26.9 ms
  （C++ 30.5），K=4 并发 0.959x，内存峰值 123 vs 130 MB。含
  resize 双线性 AVX2（bilinear_row_vec）与全局 FMA 构建旗标
  （.cargo/config.toml，对齐 C++ `-mavx2 -mfma`；workspace 级配置
  不传播给下游依赖者）。
- 公开 API（阶段 4）：`qppocr` 门面 crate——`Engine::new(tier, dir)`
  三行上手；`EngineBuilder`（tier/preset/config/advanced/threads/
  verify_sha256）；`Config`（Option 字段 = 预设覆盖语义）+ `Preset`
  （Speed/Balanced/Accuracy）+ `Advanced`（不承诺稳定的基准常数，
  只经 builder.advanced 进入）；`ModelSource::Dir/Bytes` + 上游官方
  SHA-256 校验（手写 FIPS 180-4，零依赖，NIST 向量测试）；字典三路
  查找（{tier}/dict.txt → dict.txt → ppocr_keys.txt → 内嵌）；
  feature 门控（parallel/image-decode/serde）。`#![deny(missing_docs)]`
  + `cargo doc` 零警告 + 不依赖模型的 doctest。
