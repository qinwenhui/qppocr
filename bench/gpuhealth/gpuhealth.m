// GPU 体检：一条苹果自己的 blit + 一条 CPU memcpy，同一块 256 MB 内存。
// 不含任何我写的内核，输出一眼可读。5 秒跑完。
//   编译: clang -fobjc-arc -framework Foundation -framework Metal -O2 -o gpuhealth gpuhealth.m
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <time.h>
#include <string.h>
#include <stdio.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(void) {
    @autoreleasepool {
        size_t bytes = 256ULL << 20;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        printf("芯片: %s | 缓冲: 256 MB\n\n", [[dev name] UTF8String]);

        id<MTLBuffer> a = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];

        double best = 1e18;
        for (int r = 0; r < 5; r++) {
            double t0 = now_s();
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
            [e copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:bytes];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            double ms = (now_s() - t0) * 1e3;
            printf("  第%d轮 blit %9.3f ms → %7.2f GB/s\n", r + 1, ms, (2.0 * bytes) / (ms / 1e3) / 1e9);
            if (ms < best) best = ms;
        }
        double gpu = (2.0 * bytes) / (best / 1e3) / 1e9;
        printf("GPU（苹果自己的 blit 拷贝）：%9.3f ms  →  %7.2f GB/s\n", best, gpu);

        char *s = malloc(bytes), *t = malloc(bytes);
        volatile char sink = 0;
        memset(s, 1, bytes);
        // ⚠️ 这一句是 2026-10-08 补的，别删。
        //
        // 没有它，clang -O2 会做**常量传播**：它看见上面刚 `memset(s,1,bytes)`，
        // 就把下面的 `memcpy(t,s,bytes)` 直接改写成 `memset(t,1,bytes)`。
        // 反汇编可验：原二进制里 `bl _memcpy` **一次都没有**，只有 `bl _memset`
        // 且 w1=1、w2=0x10000000。
        //
        // 后果是这一行读数是**纯写**，而下面的公式仍按拷贝记 `2.0*bytes`（读+写），
        // 于是 CPU 那行凭空虚高整整 2 倍 —— 长期显示 146~147 GB/s，
        // **比 Apple 规格的 120 GB/s 总线峰值还高 22%**，本身就是自证的假数。
        // （一度被误当成「内存压缩虚高」，同场对照早就否掉了：随机源 vs 常量源
        //   读数一致。真正的元凶一直是编译器。）
        //
        // 空内联汇编 + memory clobber：抹掉 s 的值信息，编译器再也无法这么改写。
        __asm__ __volatile__("" : "+r"(s), "+r"(t) : : "memory");
        // 预热一轮（不计时）：这台机器的访存通路冷态首跑会慢 15% 以上。
        // 「5 轮取最快」救不了它 —— 五轮全在冷态里。membw 的尺寸扫描早先
        // 把它误读成「块越小越慢」，其实与块大小无关，是冷启动。
        memcpy(t, s, bytes);
        sink ^= t[bytes - 1];
        best = 1e18;
        for (int r = 0; r < 5; r++) {
            double t0 = now_s();
            memcpy(t, s, bytes);
            sink ^= t[bytes - 1];
            double ms = (now_s() - t0) * 1e3;
            if (ms < best) best = ms;
        }
        double cpu = (2.0 * bytes) / (best / 1e3) / 1e9;
        printf("CPU（memcpy 同一块内存）    ：%9.3f ms  →  %7.2f GB/s\n", best, cpu);

        printf("\n参考值（本机 2026-10-08 实测，见 bench/gpuhealth/membw）：\n");
        printf("  Apple 对 M4 的规格 = 120 GB/s 总线**理论**峰值，CPU 与 GPU 共用。\n");
        printf("  实测可到：GPU blit 拷贝 ~93 GB/s（512 MB）、CPU memcpy ~86 GB/s\n");
        printf("  —— 都是理论值的 7~8 成，两侧同量级才对。\n");
        printf("实测 CPU/GPU = %.1f 倍。若在 10 倍以上，GPU 通路不正常。\n", cpu / gpu);
        (void)sink; free(s); free(t);
    }
    return 0;
}
