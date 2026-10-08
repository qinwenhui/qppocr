// 纯 ALU 吞吐测试（吞吐受限：6 条互相独立的 FMA 链，不访存）。
// 目的：区分「整块 GPU 被降频」与「只有访存通路坏」。
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <time.h>
#include <stdio.h>
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static const char* SRC =
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void kalu(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint g [[thread_position_in_grid]]){\n"
"  float a = float(g & 255u) * 1.0001f, b = 1.0000001f, c = 0.9999999f;\n"
"  float d1 = 1.1f, d2 = 1.2f, d3 = 1.3f, d4 = 1.4f, d5 = 1.5f;\n"
"  for (uint i = 0; i < iters; ++i) {\n"
"    a = fma(a, b, c); d1 = fma(d1, b, c); d2 = fma(d2, b, c);\n"
"    d3 = fma(d3, b, c); d4 = fma(d4, b, c); d5 = fma(d5, b, c);\n"
"  }\n"
"  if (a == 12345.678f) out[g] = a + d1 + d2 + d3 + d4 + d5;\n"
"}\n";
int main(void){@autoreleasepool{
  id<MTLDevice> d=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[d newCommandQueue];
  NSError* e=nil; id<MTLLibrary> lib=[d newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&e];
  if(!lib){printf("编译失败: %s\n",[[e description]UTF8String]);return 1;}
  id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kalu"] error:nil];
  uint32_t THREADS = 4u<<20;
  id<MTLBuffer> out=[d newBufferWithLength:THREADS*4 options:MTLResourceStorageModePrivate];
  printf("  纯 ALU 吞吐受限（6 条独立 FMA 链，零访存）\n");
  uint32_t iters_arr[3]={256,1024,4096};
  for(int k=0;k<3;k++){
    uint32_t iters=iters_arr[k]; double best=1e18;
    for(int r=0;r<3;r++){
      double t0=now_ms(); id<MTLCommandBuffer> cb=[q commandBuffer];
      id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
      [en setComputePipelineState:p]; [en setBuffer:out offset:0 atIndex:0];
      [en setBytes:&iters length:4 atIndex:1];
      [en dispatchThreads:MTLSizeMake(THREADS,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
      double ms=now_ms()-t0; if(ms<best)best=ms;
    }
    double flop=(double)THREADS*iters*6.0*2.0;
    printf("    iters=%5u：%9.3f ms → %8.2f GFLOP/s\n",iters,best,flop/(best/1e3)/1e9);
  }
  printf("\n  M4 十核 GPU fp32 峰值量级：数 TFLOP/s。若这里也掉到 1/68，说明整块 GPU 被降频；\n");
  printf("  若这里正常，则病只在访存通路。\n");
  return 0;}}
