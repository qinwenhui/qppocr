// 判别实验：到底哪种 GPU 负载会把「访存通路」打瘸？
//
//   ./stress inbounds [秒]   严格界内的重型访存 —— 大 pool + 大量小 dispatch，全部界内
//   ./stress churn    [秒]   大块 Private buffer 反复分配/释放（同时活着的始终很少）
//   ./stress manybuf  [秒]   **大量不同尺寸的 Shared buffer 同时活着** —— 引擎计划集的真实形态
//   ./stress mixed    [秒]   三者各 1/3
//   ./stress idle     [秒]   对照：什么都不做，只等同样长的时间
//
// 每种模式都是三段式：测健康 → 跑 stress N 秒 → 复测健康 → 打印前后对比。
// 健康测法与 gpuhealth 完全一致（256 MB Apple blit，取最好一轮），所以数字可直接比。
//
// 判别逻辑（跑在刚重启的健康机器上）：
//   inbounds 后掉到 1.x GB/s  → 我们的 shader 只要「界内大量访存」就能打瘸，与越界无关
//   churn    后掉到 1.x GB/s  → 是「大块显存分配/释放的搅动」打瘸的
//   两个都健康                  → 需要引擎里更具体的行为（越界 / 特定地址模式），下一步拿引擎二分
//   idle     也掉                → 与我们的负载无关，是系统里别的东西在周期性打瘸（远控/驱动）
//
// 编译: clang -fobjc-arc -framework Foundation -framework Metal -O2 -o stress stress.m
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// ── 确定性伪随机（不用 rand()，便于复现）──
static uint32_t g_seed = 20261008u;
static uint32_t xorshift(void) {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 17; g_seed ^= g_seed << 5;
    return g_seed;
}

static id<MTLDevice>      g_dev;
static id<MTLCommandQueue> g_q;
static id<MTLComputePipelineState> g_spill;

// 唯一的内核：对 float4 数组做一次界内读-改-写。kernel 里再挡一次界。
static const char *SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void kspill(device float4* buf [[buffer(0)]],\n"
"                   constant uint& n  [[buffer(1)]],\n"
"                   uint g [[thread_position_in_grid]]) {\n"
"  if (g >= n) return;\n"              // ← 双保险：界内
"  float4 v = buf[g];\n"
"  v = fma(v, 1.0000001f, 0.5f);\n"
"  buf[g] = v;\n"
"}\n";

// ── 健康测量：256 MB 苹果自己的 blit，取最好一轮，返回 GB/s ──
static double measure_health(const char *label) {
    size_t bytes = 256ULL << 20;
    id<MTLBuffer> a = [g_dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> b = [g_dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    double best = 1e18;
    for (int r = 0; r < 3; r++) {
        double t0 = now_s();
        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
        [e copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:bytes];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        double ms = (now_s() - t0) * 1e3;
        if (ms < best) best = ms;
    }
    double gbps = (2.0 * bytes) / (best / 1e3) / 1e9;
    printf("    %-14s blit %9.3f ms → %7.2f GB/s\n", label, best, gbps);
    return gbps;
}

// ── 模式 A：严格界内的重型访存（模仿引擎的形状）──
// 一个 256 MB 的 private pool，反复用不同大小的 dispatch 去踩它，
// 每个 dispatch 的访问范围都被 clamp 在 pool 内；每 8 条 dispatch 一个 command buffer。
static void stress_inbounds(double secs) {
    size_t pool_bytes = 256ULL << 20;
    id<MTLBuffer> pool = [g_dev newBufferWithLength:pool_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> side = [g_dev newBufferWithLength:64ULL << 20 options:MTLResourceStorageModePrivate];

    // 模仿引擎的形状多样性：不同大小的 dispatch
    static const size_t shapes[] = {16u<<10, 64u<<10, 256u<<10, 1u<<20, 4u<<20, 16u<<20};
    const int nshapes = (int)(sizeof(shapes)/sizeof(shapes[0]));

    double t0 = now_s();
    long long dispatches = 0, blits = 0;
    while (now_s() - t0 < secs) {
        double cb0 = now_s();
        while (now_s() - cb0 < 0.050 && now_s() - t0 < secs) {   // 每个 CB 约 50 ms 的活儿
            id<MTLCommandBuffer> cb = [g_q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:g_spill];
            for (int k = 0; k < 8; k++) {
                size_t bytes = shapes[xorshift() % nshapes];
                if (bytes > pool_bytes) bytes = pool_bytes;
                uint32_t n = (uint32_t)(bytes / 16);
                // Metal 对带 offset 的 buffer 绑定要求 256 字节对齐；off+bytes 恒在 pool 内
                uint64_t off = (uint64_t)(xorshift() % (uint32_t)((pool_bytes - bytes) / 256 + 1)) * 256;
                [en setBuffer:pool offset:off atIndex:0];
                [en setBytes:&n length:4 atIndex:1];
                [en dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [en memoryBarrierWithScope:MTLBarrierScopeBuffers];
                dispatches++;
            }
            [en endEncoding];
            [cb commit]; [cb waitUntilCompleted];
        }
        // 偶尔来一条界内 blit（引擎也会在共享 arena 里搬数据）
        {
            size_t bytes = 8ULL << 20;
            id<MTLCommandBuffer> cb = [g_q commandBuffer];
            id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
            [e copyFromBuffer:pool sourceOffset:0 toBuffer:side destinationOffset:0 size:bytes];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            blits++;
        }
    }
    printf("    inbounds 结束：%lld 条 dispatch，%lld 条 blit，全界内\n", dispatches, blits);
}

// ── 模式 B：大块 Private buffer 的分配/释放搅动（模仿 arena 翻倍增长）──
static void stress_churn(double secs) {
    static const size_t sizes[] = {1u<<20, 4u<<20, 16u<<20, 64u<<20, 128u<<20};
    const int nsizes = (int)(sizeof(sizes)/sizeof(sizes[0]));
    double t0 = now_s();
    long long allocs = 0;
    while (now_s() - t0 < secs) {
        @autoreleasepool {
            for (int k = 0; k < 16; k++) {
                size_t bytes = sizes[xorshift() % nsizes];
                id<MTLBuffer> buf = [g_dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
                // 让驱动真的给它分配物理页：一次界内 blit 填充
                id<MTLCommandBuffer> cb = [g_q commandBuffer];
                id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
                [e fillBuffer:buf range:NSMakeRange(0, bytes) value:0x3F];
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                allocs++;
                buf = nil;   // 立刻释放
            }
        }
    }
    printf("    churn 结束：%lld 次「大块分配 + 释放」\n", allocs);
}

// ── 模式 C：模仿引擎的**计划集**——大量不同尺寸的 buffer 同时活着 ──
//
// 这是前两个模式都没碰过的轴，而它恰好是引擎的真实形态：
//   - 引擎为**每个输入形状**建一份计划，每份计划一整块 arena；
//   - 尺寸从 1.4 MB 到 137.8 MB 不等（下面这张表就是实测的 12 份计划）；
//   - 旧 arena 被旧计划持有而不释放；计划缓存预算 384 MB → 同时活着很多块。
// 而 inbounds / churn 都只用 1~2 块 buffer（churn 虽分配了 2 万次，但都是
// 立刻释放，**同时活着的始终很少**）。所以"多块不同尺寸同时活着"这个形态，
// 前两个模式**根本没覆盖**。
//
// 关键差异还有一点：下表用的是 `StorageModeShared`（引擎 arena 就是 Shared，
// 还要 CPU 直接写 contents()），而 inbounds/churn 用的是 Private。
static void stress_manybuf(double secs) {
    // 实测的 12 份计划（一图 tiny 的 det/cls/rec 全部形状），单位 MB。
    static const size_t plan_mb[] = {2, 14, 1, 138, 10, 4, 26, 18, 19, 32, 26, 27};
    const int nplan = (int)(sizeof(plan_mb) / sizeof(plan_mb[0]));
    const int rounds = 2;                    // 铺两轮 → 同时活着约 620 MB
    const int nbuf = nplan * rounds;

    NSMutableArray<id<MTLBuffer>> *bufs = [NSMutableArray arrayWithCapacity:nbuf];
    size_t total = 0;
    for (int i = 0; i < nbuf; i++) {
        size_t bytes = plan_mb[i % nplan] << 20;
        // 引擎 arena 的真实存储模式：Shared（CPU/GPU 同址）。
        [bufs addObject:[g_dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]];
        total += bytes;
    }
    printf("    铺开 %d 块 Shared buffer（%d 种尺寸），共 %.0f MB，全部同时活着\n",
           nbuf, nplan, total / 1048576.0);

    double t0 = now_s();
    long long passes = 0;
    while (now_s() - t0 < secs) {
        // 照引擎的节奏走一遍：每块都推一次界内访存
        for (int i = 0; i < nbuf; i++) {
            size_t bytes = plan_mb[i % nplan] << 20;
            uint32_t n = (uint32_t)(bytes / 16);
            id<MTLCommandBuffer> cb = [g_q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:g_spill];
            [en setBuffer:bufs[i] offset:0 atIndex:0];
            [en setBytes:&n length:4 atIndex:1];
            [en dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [en endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            passes++;
        }
    }
    printf("    manybuf 结束：%lld 次「对同时活着的计划集做界内访存」\n", passes);
    [bufs removeAllObjects];
}

static void stress_mixed(double secs) {
    stress_inbounds(secs / 3);
    stress_churn(secs / 3);
    stress_manybuf(secs / 3);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *mode = argc > 1 ? argv[1] : "idle";
        double secs = argc > 2 ? atof(argv[2]) : 30.0;

        g_dev = MTLCreateSystemDefaultDevice();
        g_q = [g_dev newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:SRC]
                                                  options:nil error:&err];
        if (!lib) { printf("shader 编译失败: %s\n", [[err description] UTF8String]); return 1; }
        g_spill = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kspill"]
                                                       error:&err];
        if (!g_spill) { printf("pipeline 建失败: %s\n", [[err description] UTF8String]); return 1; }

        printf("芯片 %s ｜ 模式 %s ｜ 时长 %.0f 秒\n", [[g_dev name] UTF8String], mode, secs);
        printf("  ── 前测 ──\n");
        double before = measure_health("前测");

        printf("  ── 施压中 ──\n");
        double t0 = now_s();
        if      (!strcmp(mode, "inbounds")) stress_inbounds(secs);
        else if (!strcmp(mode, "churn"))    stress_churn(secs);
        else if (!strcmp(mode, "manybuf"))  stress_manybuf(secs);
        else if (!strcmp(mode, "mixed"))    stress_mixed(secs);
        else if (!strcmp(mode, "idle"))     { while (now_s() - t0 < secs) usleep(200000); printf("    idle 结束：什么都没做\n"); }
        else { printf("未知模式 %s（用 inbounds / churn / manybuf / mixed / idle）\n", mode); return 2; }

        printf("  ── 后测 ──\n");
        double after = measure_health("后测");

        printf("\n  ═══ 结论 ═══\n");
        printf("  前 %.2f GB/s → 后 %.2f GB/s（%.2f×）\n", before, after, after / before);
        if (before < 10.0) {
            // 前测就已经是瘸的：整台机器需要重启，这次施压的对比没有意义
            printf("  ⛔ 前测就只有 %.2f GB/s —— 这台机器本来就是瘸的，本次对比无意义。\n", before);
            printf("     先重启，再跑本探针。\n");
            return 3;
        }
        if (after >= 60.0)      printf("  ✅ 访存通路健康（≥60 GB/s）—— 这个负载不是凶手\n");
        else if (after >= 10.0) printf("  ⚠️  明显变慢但没到瘸（10~60 GB/s）—— 可疑，需要复跑确认\n");
        else                    printf("  ❌ 被打瘸了（<10 GB/s）—— 找到凶手了！重启前别再跑 GPU\n");
    }
    return 0;
}
