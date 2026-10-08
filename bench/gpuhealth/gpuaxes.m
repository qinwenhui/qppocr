// 把 GPU 的四条访存路径分开测：DMA 读+写 / DMA 纯写 / shader 读+写 / shader 纯写
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <time.h>
#include <stdio.h>
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static const char* SRC =
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void kcopy(device const float4* s [[buffer(0)]], device float4* t [[buffer(1)]], constant uint& n [[buffer(2)]], uint g [[thread_position_in_grid]]){ if(g<n) t[g]=s[g]; }\n"
"kernel void kfill(device float4* t [[buffer(1)]], constant uint& n [[buffer(2)]], uint g [[thread_position_in_grid]]){ if(g<n) t[g]=float4(1.0); }\n";
int main(void){@autoreleasepool{
  id<MTLDevice> d=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[d newCommandQueue];
  NSError* e=nil; id<MTLLibrary> lib=[d newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&e];
  if(!lib){printf("编译失败: %s\n",[[e description]UTF8String]);return 1;}
  size_t B=64u<<20; uint32_t n4=(uint32_t)(B/16);
  id<MTLBuffer> a=[d newBufferWithLength:B options:MTLResourceStorageModePrivate];
  id<MTLBuffer> b=[d newBufferWithLength:B options:MTLResourceStorageModePrivate];
  id<MTLComputePipelineState> pc=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kcopy"] error:nil];
  id<MTLComputePipelineState> pf=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kfill"] error:nil];
  struct { const char* name; int kind; } tests[] = {
    {"DMA  拷读+写",0},{"DMA  纯写(fill)",1},{"SM   拷读+写",2},{"SM   纯写",3}};
  for(int t=0;t<4;t++){
    double best=1e18;
    for(int r=0;r<4;r++){
      double t0=now_ms(); id<MTLCommandBuffer> cb=[q commandBuffer];
      if(tests[t].kind<2){ id<MTLBlitCommandEncoder> en=[cb blitCommandEncoder];
        if(tests[t].kind==0)[en copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:B];
        else [en fillBuffer:b range:NSMakeRange(0,B) value:0x3f];
        [en endEncoding];
      } else { id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
        [en setComputePipelineState: tests[t].kind==2?pc:pf];
        [en setBuffer:a offset:0 atIndex:0]; [en setBuffer:b offset:0 atIndex:1];
        [en setBytes:&n4 length:4 atIndex:2];
        [en dispatchThreads:MTLSizeMake(n4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [en endEncoding]; }
      [cb commit]; [cb waitUntilCompleted];
      double ms=now_ms()-t0; if(ms<best)best=ms;
    }
    double moved = tests[t].kind==0||tests[t].kind==2 ? 2.0*B : 1.0*B;
    printf("  %-16s %9.3f ms → %7.2f GB/s\n", tests[t].name, best, moved/(best/1e3)/1e9);
  }
  return 0;}}
