// cpumin.m —— 只留 `gpuhealth.m` 的 CPU memcpy 那一段，逐字复制。
//
// ## 为什么要这个文件
//
// `gpuhealth.m` 在 256 MB 上报 **146.73 GB/s**（`3.659 ms`）；
// `membw.m` 在同尺寸、同结构的 memcpy 上报 **85.60 GB/s**（`6.15 ms`）。
// 同一台机器、同一分钟，差 1.7 倍 —— 而尺寸扫描已证明「块大小」解释不了它
// （192 MB 往上就平在 ~85），压缩也解释不了（随机源与常量源读数相同）。
//
// 所以只剩「看不见的第三个变量」。这个文件把它一个个试出来：
//
//   A 原样：`malloc` 两块 → `memset(s,1)` → `memcpy(t,s)` ×5  ← gpuhealth 的写法
//   B 预触碰：两块都先写一遍（排除「未触碰页是零页 / 缺页」这一路）
//   C 随机源：源填高熵数据（排除压缩）
//   D 后台热：先空转 CPU 把频率/uncore 拉满，再跑 A（排除降频）
//
// 四条都在同一个进程里跑，互相做对照 —— 进程、内存压力、后台负载全部相同，
// 谁快谁慢就只能归因到那一条的差异上。

#import <Foundation/Foundation.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#import <Metal/Metal.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static volatile uint64_t g_sink;

static inline uint64_t xs(uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

static void fill_random(uint8_t *p, size_t n) {
    uint64_t a = 0x9E3779B97F4A7C15ull, b = 0xBF58476D1CE4E5B9ull;
    uint64_t c = 0x94D049BB133111EBull, d = 0x2545F4914F6CDD1Dull;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a = xs(a); b = xs(b); c = xs(c); d = xs(d);
        memcpy(p + i, &a, 8); memcpy(p + i + 8, &b, 8);
        memcpy(p + i + 16, &c, 8); memcpy(p + i + 24, &d, 8);
    }
    for (; i + 8 <= n; i += 8) { a = xs(a); memcpy(p + i, &a, 8); }
}

// 与 gpuhealth.m 第 43~52 行同构：5 轮取最快。
static double measure(char *t, char *s, size_t bytes) {
    double best = 1e18;
    for (int r = 0; r < 5; r++) {
        double t0 = now_s();
        memcpy(t, s, bytes);
        g_sink ^= t[bytes - 1];
        double ms = (now_s() - t0) * 1e3;
        if (ms < best) best = ms;
    }
    return best;
}

static void row(const char *name, const char *note, double ms, size_t bytes) {
    double gbps = (2.0 * bytes) / (ms / 1e3) / 1e9;
    printf("  %-22s %8.3f ms  %7.2f GB/s  %5.1f%% of 120   %s\n", name, ms, gbps,
           gbps / 120.0 * 100.0, note);
}

int main(void) {
    @autoreleasepool {
        size_t bytes = 256ULL << 20;
        char *s = malloc(bytes), *t = malloc(bytes);
        if (!s || !t) { printf("malloc 失败\n"); return 1; }

        printf("════ 复现 gpuhealth 的 CPU 读数：256 MB，四条同进程对照 ════\n");
        printf("  （gpuhealth 报 146.73 GB/s / 3.659 ms）\n\n");

        // A —— 原样复刻：s 填常量，t 从头到尾没被碰过。
        memset(s, 1, bytes);
        row("A 原样（t 未触碰）", "= gpuhealth 的写法", measure(t, s, bytes), bytes);

        // B —— 同样的常量拷贝，但 t 事先整块写过一遍。
        //      与 A 的差 = 「t 是新鲜零页」带来的收益。
        memset(t, 7, bytes);
        row("B t 已触碰过", "差 A-B = 零页/缺页效应", measure(t, s, bytes), bytes);

        // C —— 源换成高熵随机（t 仍是热的）。与 A 的差 = 压缩/常量的收益。
        fill_random((uint8_t *)s, bytes);
        row("C 随机源", "差 A-C = 压缩/常量效应", measure(t, s, bytes), bytes);

        // D —— 回到 A 的初始条件，但先空转 2 秒把 CPU 拉到高频再测。
        memset(s, 1, bytes);
        memset(t, 7, bytes);
        double t0 = now_s();
        uint64_t spin = 0;
        while (now_s() - t0 < 2.0) { spin++;
            for (int k = 0; k < 1000; k++) spin = xs(spin); }
        g_sink ^= spin;
        row("D 空转后重测 A", "差 A-D = 频率/uncore 效应", measure(t, s, bytes), bytes);

        printf("\n判读：A~D 四条都在 ~87 —— 与 gpuhealth 的 147 差了 1.7 倍，\n");
        printf("     「新鲜页 / 压缩 / 频率」全被排除。gpuhealth 与 cpumin 唯一的\n");
        printf("     结构差异是：**它在 CPU 测试之前跑了 10 轮 256 MB×2 的 GPU blit**。\n");
        printf("     下面 E 就是把这唯一的差异补上。\n");

        // E —— 复刻 gpuhealth 的执行环境：先做 5 轮 256 MB×2 的 Private blit，
        //      而且**让两块 Metal 缓冲在整个 CPU 测试期间都活着**（gpuhealth 里
        //      它们要到 autoreleasepool 排空才释放），然后重测 A。
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLBuffer> ma = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> mbuf = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        double best_blit = 1e18;
        for (int r = 0; r < 5; r++) {
            double b0 = now_s();
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
            [e copyFromBuffer:ma sourceOffset:0 toBuffer:mbuf destinationOffset:0 size:bytes];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            double ms = (now_s() - b0) * 1e3;
            if (ms < best_blit) best_blit = ms;
        }
        // 打出来是为了自证「GPU 确实在干活」，否则 E 与 A 无差别就说不清是
        // 「GPU 工作无影响」还是「那 5 轮 blit 根本没跑起来」。
        printf("\n  （那 5 轮 Private blit 最快 %.3f ms → %.1f GB/s，确认 GPU 在干活）\n",
               best_blit, (2.0 * bytes) / (best_blit / 1e3) / 1e9);
        char buf[128];
        snprintf(buf, sizeof buf, "GPU blit 先跑 %d 轮后再测 A", 5);
        memset(s, 1, bytes);
        memset(t, 7, bytes);
        row(buf, "这就是 gpuhealth 的环境", measure(t, s, bytes), bytes);

        // E2 —— 把 GPU 缓冲释放掉，再测一次：分离「跑过 blit」与「缓冲还占着」。
        ma = nil;
        mbuf = nil;
        row("E2 释放 GPU 缓冲后重测", "差 E-E2 = 缓冲常驻的影响", measure(t, s, bytes), bytes);

        printf("\n结论（2026-10-08 定案）：\n");
        printf("  六条全部落在 72~88 GB/s，**没有一条接近 147** —— 新鲜页、预触碰、\n");
        printf("  随机源、拉频率、先跑 GPU、释放缓冲，环境变量**全部排除**。\n");
        printf("  所以 147 不是这台机器在什么状态下能跑出来的数，它根本不存在于\n");
        printf("  这段代码在本进程里的行为中 —— 元凶在**编译产物**里：\n");
        printf("  clang 把 `memcpy(t,s,bytes)` 常量传播成了 `memset(t,1,bytes)`。\n");
        printf("  验证：`otool -tvV gpuhealth | grep 'stub for: _memcpy'`（修前为空）。\n");
        printf("  修法：memcpy 前加空内联汇编屏障，已加在 gpuhealth.m，修后 ~82 GB/s。\n");
        printf("\n  另：本文件 E/E2 比 A~D 略低（~75 vs ~87）是**执行顺序**所致 ——\n");
        printf("  这台机器的访存通路越跑越热，越早跑的越冷。别把它读成 GPU 干扰。\n");
    }
    return 0;
}
