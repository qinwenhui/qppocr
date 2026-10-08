// 分清两件完全不同的事：
//   (a) 带宽被吃光  → 耗时随拷贝尺寸线性增长
//   (b) 每次命令缓冲区有个固定的大延迟 → 空命令缓冲区也要等很久
// 空 command buffer 那次是关键：它不搬任何字节，耗时就是纯粹的提交/启动开销。
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <time.h>
#include <stdio.h>

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        printf("芯片 %s\n\n", [[dev name] UTF8String]);

        // ── 关键测试 1：空命令缓冲区（不搬一个字节）──
        for (int r = 0; r < 3; r++) {
            double t0 = now_ms();
            id<MTLCommandBuffer> cb = [q commandBuffer];
            [cb commit]; [cb waitUntilCompleted];
            printf("  空命令缓冲区 #%d：%8.3f ms\n", r + 1, now_ms() - t0);
        }

        // ── 关键测试 2：尺寸扫描（分带宽 vs 固定开销）──
        size_t sizes[] = {4u<<20, 16u<<20, 64u<<20, 256u<<20};
        printf("\n  尺寸扫描（看耗时是否随尺寸线性长）\n");
        for (int s = 0; s < 4; s++) {
            size_t bytes = sizes[s];
            id<MTLBuffer> a = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
            double best = 1e18;
            for (int r = 0; r < 4; r++) {
                double t0 = now_ms();
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
                [e copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:bytes];
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                double ms = now_ms() - t0;
                if (ms < best) best = ms;
            }
            printf("    %4zu MB：%9.3f ms  →  %7.2f GB/s\n", bytes >> 20, best, (2.0*bytes)/(best/1e3)/1e9);
        }

        // ── 关键测试 3：20 次 4 MB 拷贝，放 1 个 vs 20 个命令缓冲区 ──
        // 如果「20 个」远慢于「1 个」，说明每次提交有巨额固定成本，
        // 与"有人在用 GPU"无关 —— 那是驱动/提交路径的问题。
        size_t mb = 4u<<20;
        id<MTLBuffer> sa = [dev newBufferWithLength:mb options:MTLResourceStorageModePrivate];
        id<MTLBuffer> sb = [dev newBufferWithLength:mb options:MTLResourceStorageModePrivate];
        double t0 = now_ms();
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
            for (int i = 0; i < 20; i++)
                [e copyFromBuffer:sa sourceOffset:0 toBuffer:sb destinationOffset:0 size:mb];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
        double one = now_ms() - t0;
        t0 = now_ms();
        for (int i = 0; i < 20; i++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
            [e copyFromBuffer:sa sourceOffset:0 toBuffer:sb destinationOffset:0 size:mb];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
        double many = now_ms() - t0;
        printf("\n  20×4MB 放 1 个命令缓冲区：%9.3f ms\n", one);
        printf("  20×4MB 放 20 个命令缓冲区：%9.3f ms\n", many);
        printf("  每次提交的固定成本 ≈ %.3f ms\n", (many - one) / 20.0);
    }
    return 0;
}
