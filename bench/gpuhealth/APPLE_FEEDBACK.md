# 给 Apple 的反馈稿（Feedback Assistant 用）

> 两条独立的问题，建议**分开提交**。第 1 条是主要问题（GPU 访存通路退降，必须重启），
> 第 2 条是可稳定复现的驱动健壮性问题（未绑定 buffer → BIF0 页故障）。
>
> 文中每个数字都是实测的。**推断部分一律标注为「推断」**，不要当成结论提交。
> 提交时建议附上 `logs/` 与 `evidence/` 里的原始文件（见文末清单）。

---

## 第 1 条（主问题）：GPU 访存带宽掉到约 1/70，只能重启恢复

### English

> **Title:** GPU memory bandwidth collapses to ~1/70 of normal (95 GB/s → 0.4–1.5 GB/s) while the GPU keeps running at full clock; only a reboot restores it
>
> **Configuration**
> - MacBook Air (Mac16,12), Apple M4, 10-core GPU (IORegistry class `AGXAcceleratorG16G`), 24 GB unified memory
> - macOS 26.3.1 (build 25D2128)
> - No third-party kernel extensions, no system extensions, no virtual displays
> - AC power connected; no memory pressure
>
> **Symptom**
> After running a heavy Metal compute workload, GPU memory bandwidth collapses and *never*
> recovers until reboot:
> - `MTLBlitCommandEncoder` copy between two 256 MB `StorageModePrivate` buffers (i.e. pure
>   GPU DMA, not involving WindowServer, the display, or any capture stream): **0.42–1.52 GB/s**
>   vs **82.6–95.6 GB/s** on a freshly booted machine (measured best-of-5, ramp 7 ms → 7.3 ms).
> - A CPU `memcpy` of the same 256 MB, at the same moment: **136–147 GB/s**. So DRAM and the
>   CPU path are fine; only the GPU's path is degraded.
> - All four access paths degrade **uniformly** — DMA read+write / DMA write-only (`fillBuffer`) /
>   shader read+write / shader write-only (64 MB each): **1.38–1.93 GB/s**.
> - An empty command buffer still submits in **0.03 ms**, so this is not submission/driver latency.
> - A 4 MB transfer still reaches ~7 GB/s (cache path usable); 16 MB and above fall to ~1.3 GB/s.
> - `MTLCreateSystemDefaultDevice` reports maxed-out utilization with **no client process
>   running**: `"Device Utilization %"=100`, `"Tiler Utilization %"=100` (a freshly booted machine
>   reads ~20%).
>
> **The GPU is NOT throttled.** This is the key observation:
> `sudo powermetrics -n 1 -i 1000 --samplers gpu_power` while the GPU is under load and the
> bandwidth is collapsed reads:
> ```
> GPU HW active frequency: 1470 MHz
> GPU HW active residency: 100.00% (1470 MHz: 100%)
> GPU SW requested state: (P9 : 100%)
> GPU SW state: (SW_P9 : 100%)
> GPU idle residency: 0.00%
> GPU Power: 5699 mW
> ```
> i.e. **maximum core clock, 100% residency, the driver is requesting the highest p-state, and
> the GPU is drawing ~5.7 W** — yet it moves data 70× slower. There is no frequency throttling,
> no thermal throttling, no power-management anomaly.
>
> **No error is reported.** `recoveryCount=0`, `lastRecoveryTime=0`, **no `gpuEvent` report is
> generated**, and the kernel log is clean. The only observable correlate in IORegistry is
> `Device/Tiler Utilization %` stuck at 100%.
>
> **Impact / blast radius** — this is not merely a benchmark problem:
> WindowServer must composite the desktop through the same GPU memory path, so the whole UI
> becomes unusable. Worse, WindowServer was terminated by the watchdog **three times** in one
> session (`/Library/Logs/DiagnosticReports/WindowServer-*.ips`, `bug_type 409`,
> `namespace: WATCHDOG`, `"(1 monitored services unresponsive) … 40 seconds since last successful
> checkin"`). The third one is decisive: `is_alive_func returned unhealthy : WindowServer
> initialization not complete (post IOKitWaitQuiet)` — i.e. **WindowServer cannot even finish
> initializing after the reboot, stuck waiting on the IOKit graphics driver**. Physically the
> machine showed a black screen and could only be recovered by holding the power button.
>
> **Killing the application does not help.** A control experiment: SIGTERM'd the stuck client,
> confirmed it exited, waited 8 s, re-measured — still 1.40 GB/s. There is no API to cancel a
> submitted `MTLCommandBuffer`, so the degraded state outlives every client process.
>
> **What triggers it / what does not** (each tested for 30 s on a freshly booted, verified-healthy
> machine, re-measuring bandwidth afterwards):
> - **Does NOT trigger it:** 30 s idle; **257,792** compute dispatches (in-bounds, randomized
>   addresses and sizes, with a `memoryBarrierWithScope` after every dispatch); **20,064** large
>   (1–128 MB) `StorageModePrivate` buffer allocations + `fillBuffer` + free; 24 live
>   `StorageModeShared` buffers of 12 different sizes (634 MB total) with 30,048 in-bounds
>   accesses. All of these re-measured at 71–94 GB/s afterwards.
> - **Triggers it — a STEP, not cumulative decay.** Measured 2026-10-08 on a freshly booted machine:
>   a batch of **10 images** failed to complete within 180 s, and the machine re-measured at
>   **0.77 GB/s** — having read **95.24 GB/s** immediately before the batch. One batch, no ramp-up.
>   (An earlier uncontrolled observation of "monotonic decay" over a session is superseded by this.)
> - **The trigger is the amount of GPU work in a batch, NOT concurrency.** Three controlled arms;
>   between them only the image count and worker count differ:
>
>   | images | worker processes | result |
>   |---|---|---|
>   | 2 | 2 | completed in 215 ms — **81.6 GB/s** (healthy) |
>   | 10 | 2 | failed (180 s timeout) — **0.92 GB/s** |
>   | 10 | 5 | failed (180 s timeout) — **0.77 GB/s** |
>
>   The same image count wedges with 2 processes *or* 5; only the smaller batch is fine.
> - **Retraction of an earlier claim in this report.** A previous revision of this document stated
>   that the trigger was the **number of concurrent Metal client processes**. That was wrong: it came
>   from a comparison in which process count and per-process work changed *together*. The
>   experiment above separates them, and the separation went against our hypothesis. We are leaving
>   this note in rather than silently editing it.
> - **Driver-side GPU memory is not the cause either.** The 2-process wedge left
>   `Alloc system memory` at **0.98 GB**; the 5-process wedge at **2.06 GB**. It merely scales with
>   process count — resource footprint does not explain the collapse.
> - **Sign of the degradation starting early:** within the 10-image batch the *first* images to
>   finish were already slow (142–178 ms/image) versus **93 ms/image** on the healthy 2-image batch.
> - **Correction to our own earlier conclusion.** We previously reported "contention cannot explain
>   the collapse", based on a control in which *one* competing process saturated the GPU — the blit
>   fell only 22% (94.9 → 74.3 GB/s). That control varied GPU *occupancy* while keeping
>   address-space diversity trivial (2 buffers, 1 extra process). Pure GPU-time contention is ruled
>   out by it; multi-client address-space pressure is not.
>
> **Corroborating symptom (visible in Activity Monitor, no root needed)**
> While degraded, WindowServer's GPU usage sits at **80–100%** (mostly 100%); on a freshly booted
> machine the same process reads **~8%**. This is a *consequence*, not a cause — see the
> contention control below — and it is a cheap way for anyone to see the state without root.
>
> **GPU-time contention ruled out; multi-client address-space pressure is NOT ruled out.** Control
> experiment: with a second process saturating the GPU (257k in-bounds compute dispatches, the same
> load that does not wedge the machine), the 256 MB blit measures **74.3 / 78.1 GB/s** — a drop of
> only ~22% from the uncontended 94.9 GB/s. So *sharing GPU time* cannot account for a 70× collapse.
> Caveat we want to state plainly: that control had 1 extra process and 2 buffers, i.e. it varied
> occupancy without varying address-space diversity. It therefore does not speak to the
> multi-process case, which our 2026-10-08 worker-count experiment now implicates.
>
> **Ruled out by measurement**
> - Memory pressure: 88–92% free, `Pages stored in compressor` 0, `Compressions/Decompressions`
>   0/0, `Swapins/Swapouts/Pageouts` 0/0/0, largest resident process 379 MB.
> - Battery / low-power throttling: AC connected (`ioreg ExternalConnected=Yes, IsCharging=Yes`).
> - Thermal throttling: `pmset -g therm` reports no warnings.
> - GPU frequency/PM throttling: see powermetrics above.
> - Remote-desktop/capture software holding the GPU: all its threads idle in `mach_msg`/`kevent`,
>   zero Metal frames; and the 95 GB/s healthy readings were taken **while** it was capturing.
> - Third-party kexts / system extensions: none installed.
>
> **What we are asking for**
> - A root cause for why the GPU's memory path can degrade ~70× and persist for the remaining
>   uptime, with full clock, full power, no throttle, and no error report.
> - A **user-space recovery path** for compute queues. On Windows, TDR (Timeout Detection and
>   Recovery) force-resets a hung GPU and kills the offending application's device context after
>   ~2 s, so an application cannot leave the system unusable. On macOS, the watchdog only kills
>   WindowServer; the bad GPU/driver state survives until reboot. Right now a misbehaving (or
>   merely heavy) user-space application can drive the whole machine into "physical black screen,
>   power-button required".
>
> **Attachments**: `logs/baseline_reboot1.txt` (healthy baseline on a freshly booted machine),
> `logs/pm_healthy.txt`, `logs/pm_wedged.txt` (the two powermetrics samples),
> `evidence/WindowServer-2026-10-08-162218.ips`, `…-162638.ips`, `…-163058.ips`,
> `evidence/forceReset-full-2026-10-08-165259.0002.diag`.

### 中文（便于你核对）

**标题**：GPU 访存带宽掉到正常值的约 1/70（95 GB/s → 0.4–1.5 GB/s），而 GPU 仍以满频运行；只有重启能恢复

**配置**
- MacBook Air（Mac16,12），Apple M4，10 核 GPU（IORegistry 类名 `AGXAcceleratorG16G`），24 GB 统一内存
- macOS 26.3.1（build 25D2128）
- 未安装任何第三方内核扩展 / 系统扩展 / 虚拟显示器
- AC 供电；无内存压力

**症状**
跑一段重型 Metal compute 负载之后，GPU 的访存带宽崩塌，且**直到重启都不会恢复**：
- 两块 256 MB `StorageModePrivate` buffer 之间的 `MTLBlitCommandEncoder` 拷贝（纯 GPU DMA，不经过 WindowServer、不涉及显示器、不涉及任何采集流）：**0.42–1.52 GB/s**，而刚重启的机器上是 **82.6–95.6 GB/s**（取 5 轮最好成绩，耗时从约 5.6 ms 涨到约 1030 ms）
- 同一时刻对同一块 256 MB 做 CPU `memcpy`：**136–147 GB/s**。所以 DRAM 和 CPU 通路正常，**只有 GPU 的路径坏了**
- 四条访存路径**均匀**退化 —— DMA 读+写 / DMA 纯写（`fillBuffer`）/ shader 读+写 / shader 纯写（各 64 MB）：**1.38–1.93 GB/s**
- 空 command buffer 提交仍只要 **0.03 ms**，所以不是提交/驱动延迟
- 4 MB 的小传输还能到约 7 GB/s（缓存路径可用）；16 MB 以上掉到约 1.3 GB/s
- **没有任何客户端进程在跑**时，设备仍报 `"Device Utilization %"=100`、`"Tiler Utilization %"=100`（刚重启的机器约 20%）

**GPU 没有被降频 —— 这是最关键的一点。** 在带宽已经崩塌、GPU 处于负载下的时候运行
`sudo powermetrics -n 1 -i 1000 --samplers gpu_power`，读数是：
```
GPU HW active frequency: 1470 MHz
GPU HW active residency: 100.00% (1470 MHz: 100%)
GPU SW requested state: (P9 : 100%)
GPU SW state: (SW_P9 : 100%)
GPU idle residency: 0.00%
GPU Power: 5699 mW
```
即：**核心频率满档、驻留 100%、驱动请求最高 p-state、功耗约 5.7 W** —— 却搬不动数据（慢 70 倍）。
没有频率降档、没有温控降频、没有电源管理异常。

**没有任何错误被报告。** `recoveryCount=0`、`lastRecoveryTime=0`、**没有产生任何 `gpuEvent` 报告**、内核日志干净。IORegistry 里唯一的相关观察是 `Device/Tiler Utilization %` 卡在 100%。

**影响范围（远不止"基准测不准"）**：WindowServer 合成桌面必须走同一条 GPU 访存通路，所以整个 UI 卡到不可用。更严重的是，WindowServer 在一次会话中被看门狗**判死三次**（`/Library/Logs/DiagnosticReports/WindowServer-*.ips`，`bug_type 409`，`namespace: WATCHDOG`，`"(1 monitored services unresponsive) … 40 seconds since last successful checkin"`）。第三次的 `details` 是决定性的：`is_alive_func returned unhealthy : WindowServer initialization not complete (post IOKitWaitQuiet)` —— **重启之后 WindowServer 连初始化都完不成，卡在等 IOKit 图形驱动**。物理在现场看到的是**黑屏**，只能长按电源强制重启。

**杀掉应用没用。** 对照实验：给卡住的客户端发 SIGTERM、确认它已退出、等 8 秒、复测 —— 仍是 **1.40 GB/s**。`MTLCommandBuffer` 没有取消 API，所以退化状态在所有客户端进程消失后依然存在。

**能触发 / 不能触发**（每一项都是在刚重启且已验证健康的机器上压 30 秒，之后再复测带宽）：
- **不能触发**：空闲 30 秒；**257,792 条** compute dispatch（全部界内、地址与尺寸随机、每条 dispatch 后加 `memoryBarrierWithScope`）；**20,064 次**大块（1–128 MB）`StorageModePrivate` buffer 分配 + `fillBuffer` + 释放；24 块不同尺寸的 `StorageModeShared` buffer 同时存活（共 634 MB）+ 30,048 次界内访存。以上之后复测都是 71–94 GB/s。
- **能触发——而且是台阶，不是累积衰减。** 2026-10-08 在刚重启的机器上实测：**一批 10 张图**在 180 秒内没能完成，随后复测 **0.77 GB/s**，而**这批之前**实测是 **95.24 GB/s**。就一批，没有爬坡过程。（早先那次"整个会话内单调衰减"的观察未做负载控制，已被这次测量取代。）
- **变量是「一批里的 GPU 工作量」，不是并发度。** 三个受控臂，臂与臂之间只差图数和 worker 数：

  | 图数 | worker 进程数 | 结果 |
  |---|---|---|
  | 2 | 2 | 215 ms 跑完 —— **81.6 GB/s**（健康） |
  | 10 | 2 | 未完成（180 s 超时）—— **0.92 GB/s** |
  | 10 | 5 | 未完成（180 s 超时）—— **0.77 GB/s** |

  同样的图数，用 2 个进程**或** 5 个进程都打瘸；只有小批是好的。
- **撤回本报告早先版本里的一个说法。** 本报告上一版写过「触发点是**并发 Metal 客户端进程数**」。那是错的：它出自一个「进程数与每进程工作量同时变化」的对照。上面的实验把两者拆开了，而拆开的结果**与我们的假设相反**。我们把这条撤回记录留在这里，而不是悄悄删掉。
- **驱动侧显存也不是原因。** 2 进程瘸掉时 `Alloc system memory` 是 **0.98 GB**，5 进程瘸掉时是 **2.06 GB** —— 它只是随进程数增长，资源占用解释不了崩塌。
- **退化在批内很早就开始了**：10 图那批里**最先跑完的几张就已经慢了**（142~178 ms/张），而健康的 2 图批是 **93 ms/张**。
- **对我们自己此前结论的一处更正。** 我们先前报告过"竞争无法解释崩塌"，依据是一个对照实验：**单个**竞争进程把 GPU 打满，blit 只掉 22%（94.9 → 74.3 GB/s）。那个对照只改变了 GPU 的**占空**，而地址空间多样性极低（2 块 buffer、多 1 个进程）。它排除了纯粹的"GPU 时间竞争"，但**没有**排除多客户端地址空间压力。

**已实测排除**
- 内存压力：88–92% 空闲、`Pages stored in compressor` 0、`Compressions/Decompressions` 0/0、`Swapins/Swapouts/Pageouts` 0/0/0、最大常驻进程 379 MB
- 电池/低电量降频：AC 供电（`ioreg ExternalConnected=Yes, IsCharging=Yes`）
- 温控降频：`pmset -g therm` 无任何告警
- GPU 频率/电源管理降频：见上面的 powermetrics
- 远控/采集软件占着 GPU：它的所有线程都空闲在 `mach_msg`/`kevent`，零个 Metal 帧；而且那些 95 GB/s 的健康读数**正是在它采集时**测到的
- 第三方内核扩展 / 系统扩展：未安装

**我们的诉求**
- 说明 GPU 访存路径为什么能退化约 70 倍，并在剩余运行时间里一直保持 —— 满频、满功耗、无节流、无任何错误报告
- **给 compute 队列提供一个用户态可用的恢复路径**。Windows 有 TDR（超时检测与恢复）：GPU 上的命令超过约 2 秒没完成，系统会强制重置 GPU 并杀掉肇事应用的设备上下文，所以应用**无法**把系统搞成不可用。macOS 上看门狗只是杀掉 WindowServer，坏的 GPU/驱动状态一直保留到重启。现在的结果是：一个（哪怕只是重度使用 GPU 的）用户态应用，可以把整台机器推到**黑屏、必须长按电源**的地步。

**附件**：`logs/baseline_reboot1.txt`（刚重启机器上的健康基线）、`logs/pm_healthy.txt`、
`logs/pm_wedged.txt`（两份 powermetrics 样本）、`evidence/WindowServer-2026-10-08-162218.ips`、
`…-162638.ips`、`…-163058.ips`、`evidence/forceReset-full-2026-10-08-165259.0002.diag`

---

## 第 2 条（可稳定复现，建议单独提）：未绑定的 buffer → GPU 从地址 0 取数 → BIF0 页故障

### English

> **Title:** A compute kernel reading an unbound buffer argument reads from GPU address 0 and raises a BIF0 page fault, degrading the whole GPU until reboot
>
> If a `MTLComputeCommandEncoder` is dispatched with a buffer argument **declared by the kernel
> but never bound** via `setBuffer:offset:atIndex:`, the GPU reads from **virtual address 0**
> instead of faulting cleanly at the argument level, which raises a **BIF0 page fault** and pushes
> the device into a degraded state (the same state described in the report above).
>
> Evidence, `/Library/Logs/DiagnosticReports/gpuEvent-qppocr_metal-*.ips` (`bug_type 284`):
> ```
> "restart_reason_desc": "BIF0 page fault"
> "bif0_fault": { "is_read": true, "address": 0 }
> "guilty_dm": 3
> "signature": 563
> ```
> Two such reports were generated ~2 s apart and drove `recoveryCount` to 2.
>
> We would expect an unbound buffer argument to trap deterministically (and, ideally, to be
> diagnosed as an API misuse) rather than to produce a GPU-side read from address 0 with
> system-wide consequences.
>
> We have fixed this on our side. A minimal reproduction is a kernel whose signature declares
> `constant uint* p [[buffer(1)]]`, dispatched after binding only buffer 0.

### 中文

**标题**：compute 内核读取一个未绑定的 buffer 参数时会从 GPU 地址 0 取数，触发 BIF0 页故障，并使整块 GPU 退化到重启为止

如果 `MTLComputeCommandEncoder` 被 dispatch 时，某个内核**声明**了但从未通过 `setBuffer:offset:atIndex:` **绑定**的 buffer 参数，GPU 会从**虚拟地址 0** 取数（而不是在参数层面干净地报错），触发 **BIF0 页故障**，并把设备推进退化态（即上一条报告描述的那个状态）。

证据，`/Library/Logs/DiagnosticReports/gpuEvent-qppocr_metal-*.ips`（`bug_type 284`）：
```
"restart_reason_desc": "BIF0 page fault"
"bif0_fault": { "is_read": true, "address": 0 }
"guilty_dm": 3
"signature": 563
```
两份这样的报告相隔约 2 秒产生，把 `recoveryCount` 推到了 2。

我们的预期是：未绑定的 buffer 参数应当被确定性地捕获（最好能直接诊断为 API 误用），而不是产生一次"GPU 侧从地址 0 读"并带来系统级后果的访问。

我们这边已经修了。最小复现是一个签名里声明了 `constant uint* p [[buffer(1)]]` 的内核，dispatch 时只绑定了 buffer 0。

---

## 提交时的建议

1. **两条分开提**。第 1 条的标题直接点出"满频但带宽掉 70 倍、只能重启"，工程师一眼就能抓到重点。
2. **一定要附 `powermetrics` 的两份样本**（健康态 + 退化态）。"满频 1470 MHz 却搬不动数据"是全篇最有说服力的一句。
3. **附上 WindowServer 的看门狗报告**，它把问题从"GPU 慢"升级成"整机不可用"，优先级会不一样。
4. 如果 Feedback Assistant 允许录屏，退化态下录一段**桌面拖动窗口**的屏 —— 这是"影响整机可用性"最直观的证据。
5. 第 1 条里的"随累积工作量单调衰减"**明确标注为假设**，不要写成结论；否则被证伪会连带削弱整份报告的可信度。
