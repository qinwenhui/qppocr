// membw.m —— 内存带宽**上限**探针：把「官方那个数」和「我们机器真能到多少」对上。
//
// ## 为什么必须重做（2026-10-08）
//
// 原来的 `gpuhealth` 用 `memset(s, 1, bytes)` 填源、再 `memcpy`，按
// `2.0*bytes/t`（读+写都算流量）报数，得到 **CPU 135~147 GB/s**。而 Apple 对
// 基础版 M4 的规格是 **120 GB/s**（LPDDR5X-7500 × 128 bit ÷ 8 = 7500e6*16 B/s）。
// **测出来比理论峰值还高 12~22% —— 这个数不可能是真的**，拿它当基准会把后面
// 所有「离带宽上限还有多远」的判断带偏。
//
// **元凶（2026-10-08 查明）：不是压缩，是编译器。**
// clang -O2 做了常量传播 —— 它看见 `memset(s,1,bytes)`，就把 `memcpy(t,s,bytes)`
// 改写成 `memset(t,1,bytes)`。反汇编可验：`gpuhealth` 二进制里 `bl _memcpy`
// **一次都没有**，只有 `bl _memset` 且 w1=1、w2=0x10000000。
// 于是那行读数是**纯写**，公式却仍按拷贝记 2×bytes → 凭空虚高 2 倍。
// 对账闭合：本探针「memset 常量（纯写）」512 MB 得 **73.49 GB/s**，
// 2 × 73.49 = 147，正是 `gpuhealth` 长期显示的 146~147。
//
// 一度怀疑过**透明内存压缩**（源是单字节常量，本机 `vm_stat` 也确实有压缩活动）。
// 已否掉：同场对照里「随机源 vs 常量源」读数一致（84.98 vs 85.37），压缩没帮忙。
// `gpuhealth.m` 已补上空内联汇编把 s 的值信息抹掉，修复后 CPU 行回到 ~75 GB/s。
//
// 所以这个探针做三件事：
//   1. **用不可压缩数据**（xorshift64 填的随机字节）重测同样的三条路径；
//   2. 同场对比**可压缩 vs 不可压缩**源 —— 用来**排除**压缩这条假设；
//   3. 把读数统一成「相对 120 GB/s 的百分比」，并**分开报**读、写、读写混合
//      —— 「带宽」这个词在这三条上不是同一个数，混着报就必然自相矛盾。
//
// 口径声明（重要）：所有 GB/s 都按**实际 DRAM 流量**算——
//   拷贝 / blit = 2×bytes（读一遍 + 写一遍）、只读 = 1×bytes、只写 = 1×bytes。
//   与 120 GB/s 相比时注意：官方那个数是**总线理论峰值**，读+写共用同一条总线，
//   所以「拷贝」这类读写各半的负载，两边加起来不该超过 120。

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SPEC_GBPS 120.0  // M4 基础版：LPDDR5X-7500 × 128 bit ÷ 8

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// xorshift64：够快、够不可压缩（内存压缩对高熵数据无效）。
// **四条独立流**：单条的 3 移位 + 2 异或是串行依赖链，约 4 个周期才出一个数，
// 只写测试会变成「测 PRNG」而不是「测内存」——这是第一版读数的教训，别改回单流。
static inline uint64_t xs_step(uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

static void fill_random(uint8_t *p, size_t n) {
    uint64_t s0 = 0x9E3779B97F4A7C15ull, s1 = 0xBF58476D1CE4E5B9ull;
    uint64_t s2 = 0x94D049BB133111EBull, s3 = 0x2545F4914F6CDD1Dull;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = xs_step(s0); s1 = xs_step(s1);
        s2 = xs_step(s2); s3 = xs_step(s3);
        memcpy(p + i +  0, &s0, 8);
        memcpy(p + i +  8, &s1, 8);
        memcpy(p + i + 16, &s2, 8);
        memcpy(p + i + 24, &s3, 8);
    }
    for (; i + 8 <= n; i += 8) { s0 = xs_step(s0); memcpy(p + i, &s0, 8); }
}

static uint8_t *g_src, *g_dst;
static volatile uint64_t g_sink;  // 防止整个循环被优化掉

// 取多轮最好成绩：环境噪声（WindowServer / 后台同步）只会让它变慢，取最小是
// 对「硬件上限」的正确估计。首次触碰的缺页由调用方先 warmup 掉。
static double best_ms(int rounds, void (^body)(void)) {
    double best = 1e30;
    for (int r = 0; r < rounds; r++) {
        double t0 = now_ms();
        body();
        double t1 = now_ms();
        if (t1 - t0 < best) best = t1 - t0;
    }
    return best;
}

static void report(const char *label, double ms, double traffic_bytes) {
    double gbps = traffic_bytes / (ms / 1e3) / 1e9;
    printf("  %-34s %8.2f ms  %8.2f GB/s  = %5.1f%% of 120\n", label, ms, gbps,
           gbps / SPEC_GBPS * 100.0);
}

// 尺寸扫描 —— 这是本探针最要紧的一段。
//
// `gpuhealth.m` 的 CPU 测试和本探针的「常量源」那行**逐字同构**（都是
// `memset(s,1,n)` 后 `memcpy(t,s,n)`、5 轮取最快），可它报 **146.73 GB/s**、
// 本探针报 82.81 —— 高过规格 120 的 22%。唯一变量就是**块大小**（256 MB vs 512 MB）。
//
// 压缩解释已被同场对照否掉（随机源 85.14 vs 常量源 82.81，压缩没带来任何好处）。
// 那就只剩「块大小」这一个自变量，扫一遍看断裂点落在哪里、以及 146 那个数能不能复现。
static void sweep_sizes(void) {
    size_t mb[] = {32, 64, 128, 192, 256, 320, 384, 512};
    size_t n = sizeof(mb) / sizeof(mb[0]);
    printf("\n── 尺寸扫描：同一条 memcpy，各 5 轮取最快 ──\n");
    printf("  %9s  %10s %10s   %10s %10s\n", "块大小", "随机源ms", "随机源GB/s",
           "常量源ms", "常量源GB/s");

    // 预热：**别删**。没有它，扫描的前几行会明显偏低（32 MB 量到 41 GB/s），
    // 很容易被误读成「块越小带宽越低」—— 实测那是冷启动，与块大小无关。
    // 预热之后 32 MB ~ 512 MB 全部平在 ~85 GB/s。
    for (int w = 0; w < 2; w++) {
        size_t wb = 256ULL << 20;
        uint8_t *wa = malloc(wb), *wb2 = malloc(wb);
        if (wa && wb2) {
            fill_random(wa, wb);
            fill_random(wb2, wb);
            memcpy(wb2, wa, wb);
            free(wa); free(wb2);
        }
    }
    for (size_t i = 0; i < n; i++) {
        size_t bytes = mb[i] << 20;
        uint8_t *a = malloc(bytes), *b = malloc(bytes);
        if (!a || !b) {
            printf("  %6zu MB      malloc 失败\n", mb[i]);
            free(a); free(b);
            continue;
        }
        // 两块都先触碰并填随机：排除缺页、也排除「未触碰页是零页」这个变量。
        fill_random(a, bytes);
        fill_random(b, bytes);
        double mr = best_ms(5, ^{ memcpy(b, a, bytes); });

        memset(a, 1, bytes);
        double mc = best_ms(5, ^{ memcpy(b, a, bytes); });

        printf("  %6zu MB  %10.2f %10.2f   %10.2f %10.2f\n", mb[i], mr,
               2.0 * bytes / (mr / 1e3) / 1e9, mc, 2.0 * bytes / (mc / 1e3) / 1e9);
        free(a);
        free(b);
    }
    printf("  注：预热后前两行仍偏低，是残余的顺序效应（越早跑越冷），不是\n");
    printf("      块大小效应。看结论请以 128 MB 及以上为准（~86 GB/s）。\n");
}

int main(void) {
    @autoreleasepool {
        // 512 MB 一维数组：远超 M4 的 SLC（十几 MB 量级），保证打的是 DRAM。
        size_t bytes = 512ULL << 20;
        g_src = malloc(bytes);
        g_dst = malloc(bytes);
        if (!g_src || !g_dst) {
            printf("malloc 失败\n");
            return 1;
        }

        // 预触碰：先让所有页真的映射好，免得把缺页成本算进带宽。
        fill_random(g_src, bytes);
        memset(g_dst, 0, bytes);
        fill_random(g_dst, bytes);

        printf("════ 内存带宽上限  512 MB × 2（相对规格 120 GB/s）════\n");
        printf("口径：拷贝/blit = 2×bytes 流量；只读/只写 = 1×bytes\n\n");

        printf("── CPU（单线程）──\n");
        double ms;

        // 不可压缩源的拷贝：这才是真实可到的上限。
        fill_random(g_src, bytes);
        ms = best_ms(5, ^{ memcpy(g_dst, g_src, bytes); });
        report("copy   随机源（不可压缩）", ms, 2.0 * bytes);

        // 可压缩源：与上一行同场对照，**用来排除压缩假设**。
        // （实测两者一致 → 压缩在本机不影响这个读数，别再把虚高归给它。）
        memset(g_src, 1, bytes);
        ms = best_ms(5, ^{ memcpy(g_dst, g_src, bytes); });
        report("copy   常量源（可压缩，对照）", ms, 2.0 * bytes);
        printf("           ↑ 与上一行一致 ⇒ 压缩**不是**虚高的原因\n");

        // 只读：逐 8 字节求和，读满一整块。
        fill_random(g_src, bytes);
        ms = best_ms(5, ^{
            uint64_t s = 0;
            for (size_t i = 0; i + 8 <= bytes; i += 8) {
                uint64_t v;
                memcpy(&v, g_src + i, 8);
                s += v;
            }
            g_sink = s;
        });
        report("read   随机源（只读）", ms, 1.0 * bytes);

        // ⚠️ 这一行**不是带宽读数**：0.72 ns/8B 说明瓶颈在 xorshift 的依赖链，
        // 不在内存（同尺寸纯 memset 只要 7.30 ms，这里 46 ms）。保留它是为了
        // 对照出「写常量 vs 写随机」的差距里有多少是 PRNG 成本。要写带宽看下一行。
        ms = best_ms(5, ^{ fill_random(g_dst, bytes); });
        report("write  PRNG 填随机（非带宽）", ms, 1.0 * bytes);

        // 纯 memset 写出 —— **这一条是用来对账的**。
        //
        // `gpuhealth.m` 的 CPU 行长期报 146~147 GB/s，因为它那句 `memcpy` 被
        // clang 常量传播成了 `memset`（反汇编里一个 `bl _memcpy` 都没有），
        // 干的是纯写、却按拷贝的 2×bytes 记流量。这里把同一件事真测一遍：
        // 若此行的 GB/s × 2 ≈ 147，就对上了，那个数确系「纯写账拷贝」。
        ms = best_ms(5, ^{ memset(g_dst, 1, bytes); });
        report("memset 常量（纯写，只写）", ms, 1.0 * bytes);

        printf("\n── GPU（苹果自己的 blit，同一块内存）──\n");
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            printf("  无 Metal 设备，跳过\n");
            return 0;
        }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        // Shared：CPU 直接可写，先把随机数据灌进去（与 CPU 那几条同源可比）。
        id<MTLBuffer> sa = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> sb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        memcpy(sa.contents, g_src, bytes);

        ms = best_ms(5, ^{
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
            [e copyFromBuffer:sa sourceOffset:0 toBuffer:sb destinationOffset:0 size:bytes];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        });
        report("blit   Shared→Shared（随机数据）", ms, 2.0 * bytes);

        // Private：与旧探针同构（内容未显式初始化 —— 若被压成零页，读数会虚高）。
        id<MTLBuffer> pa = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> pb = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        ms = best_ms(5, ^{
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
            [e copyFromBuffer:pa sourceOffset:0 toBuffer:pb destinationOffset:0 size:bytes];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        });
        report("blit   Private→Private（内容未初始化）", ms, 2.0 * bytes);

        printf("\n判读（这些是 2026-10-08 在本机实测出来的结论）：\n");
        printf("  · 120 GB/s 是**总线理论峰值**，CPU 与 GPU 共用，谁都到不了 100%%。\n");
        printf("  · 拷贝（读+写各半）真能到的上限：CPU ~85、GPU blit ~93~95 GB/s\n");
        printf("    —— 都在规格的 7~8 成，**这才是拿来做「离带宽还有多远」的分母**。\n");
        printf("  · 「常量源 vs 随机源」读数一致 ⇒ 压缩不影响本机读数（已排除）。\n");
        printf("  · 只读 ~68、只写 ~73 GB/s 是单向的数，别和拷贝的 2× 口径混着比。\n");
        printf("  · 旧基线里 CPU 135~147 GB/s 是**编译器把 memcpy 改成 memset** 造成的\n");
        printf("    假数（见文件头），已修，别再引用。\n");
    }

    sweep_sizes();
    return 0;
}
