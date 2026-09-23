// GEMM A/B — C++ 侧。结构逐句照抄 src/ops.cpp 的 sgemm（4 行 x 4 个 AVX 向量、
// k 内层 broadcast+FMA、N-panel 32 列），去掉线程池、激活和 bias 以外的分支，
// 只留内核，这样比的是「同一套算法在两个语言/编译器下的代码质量」。
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, int ldc) {
    for (int n0 = 0; n0 < N; n0 += 32) {
        const int nn = std::min(32, N - n0);
        const bool full = (nn == 32);
        int m = 0;
        for (; m + 4 <= M; m += 4) {
            const float* a0 = A + (size_t)(m + 0) * K;
            const float* a1 = A + (size_t)(m + 1) * K;
            const float* a2 = A + (size_t)(m + 2) * K;
            const float* a3 = A + (size_t)(m + 3) * K;
            __m256 c00 = _mm256_setzero_ps(), c01 = c00, c02 = c00, c03 = c00;
            __m256 c10 = c00, c11 = c00, c12 = c00, c13 = c00;
            __m256 c20 = c00, c21 = c00, c22 = c00, c23 = c00;
            __m256 c30 = c00, c31 = c00, c32 = c00, c33 = c00;
            for (int k = 0; k < K; ++k) {
                const float* b = B + (size_t)k * N + n0;
                __m256 bv0 = _mm256_loadu_ps(b);
                __m256 bv1 = (nn > 8) ? _mm256_loadu_ps(b + 8) : _mm256_setzero_ps();
                __m256 bv2 = (nn > 16) ? _mm256_loadu_ps(b + 16) : _mm256_setzero_ps();
                __m256 bv3 = (nn > 24) ? _mm256_loadu_ps(b + 24) : _mm256_setzero_ps();
                __m256 av;
                av = _mm256_broadcast_ss(a0 + k);
                c00 = _mm256_fmadd_ps(av, bv0, c00); c01 = _mm256_fmadd_ps(av, bv1, c01);
                c02 = _mm256_fmadd_ps(av, bv2, c02); c03 = _mm256_fmadd_ps(av, bv3, c03);
                av = _mm256_broadcast_ss(a1 + k);
                c10 = _mm256_fmadd_ps(av, bv0, c10); c11 = _mm256_fmadd_ps(av, bv1, c11);
                c12 = _mm256_fmadd_ps(av, bv2, c12); c13 = _mm256_fmadd_ps(av, bv3, c13);
                av = _mm256_broadcast_ss(a2 + k);
                c20 = _mm256_fmadd_ps(av, bv0, c20); c21 = _mm256_fmadd_ps(av, bv1, c21);
                c22 = _mm256_fmadd_ps(av, bv2, c22); c23 = _mm256_fmadd_ps(av, bv3, c23);
                av = _mm256_broadcast_ss(a3 + k);
                c30 = _mm256_fmadd_ps(av, bv0, c30); c31 = _mm256_fmadd_ps(av, bv1, c31);
                c32 = _mm256_fmadd_ps(av, bv2, c32); c33 = _mm256_fmadd_ps(av, bv3, c33);
            }
            float* c;
            c = C + (size_t)(m + 0) * ldc + n0;
            if (full) { _mm256_storeu_ps(c, c00); _mm256_storeu_ps(c + 8, c01); _mm256_storeu_ps(c + 16, c02); _mm256_storeu_ps(c + 24, c03); }
            else { alignas(16) float t[32]; _mm256_storeu_ps(t, c00); _mm256_storeu_ps(t + 8, c01); _mm256_storeu_ps(t + 16, c02); _mm256_storeu_ps(t + 24, c03); for (int n = 0; n < nn; ++n) c[n] = t[n]; }
            c = C + (size_t)(m + 1) * ldc + n0;
            if (full) { _mm256_storeu_ps(c, c10); _mm256_storeu_ps(c + 8, c11); _mm256_storeu_ps(c + 16, c12); _mm256_storeu_ps(c + 24, c13); }
            else { alignas(16) float t[32]; _mm256_storeu_ps(t, c10); _mm256_storeu_ps(t + 8, c11); _mm256_storeu_ps(t + 16, c12); _mm256_storeu_ps(t + 24, c13); for (int n = 0; n < nn; ++n) c[n] = t[n]; }
            c = C + (size_t)(m + 2) * ldc + n0;
            if (full) { _mm256_storeu_ps(c, c20); _mm256_storeu_ps(c + 8, c21); _mm256_storeu_ps(c + 16, c22); _mm256_storeu_ps(c + 24, c23); }
            else { alignas(16) float t[32]; _mm256_storeu_ps(t, c20); _mm256_storeu_ps(t + 8, c21); _mm256_storeu_ps(t + 16, c22); _mm256_storeu_ps(t + 24, c23); for (int n = 0; n < nn; ++n) c[n] = t[n]; }
            c = C + (size_t)(m + 3) * ldc + n0;
            if (full) { _mm256_storeu_ps(c, c30); _mm256_storeu_ps(c + 8, c31); _mm256_storeu_ps(c + 16, c32); _mm256_storeu_ps(c + 24, c33); }
            else { alignas(16) float t[32]; _mm256_storeu_ps(t, c30); _mm256_storeu_ps(t + 8, c31); _mm256_storeu_ps(t + 16, c32); _mm256_storeu_ps(t + 24, c33); for (int n = 0; n < nn; ++n) c[n] = t[n]; }
        }
        for (; m < M; ++m) {
            const float* a = A + (size_t)m * K;
            float* c = C + (size_t)m * ldc + n0;
            __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
            for (int k = 0; k < K; ++k) {
                const float* b = B + (size_t)k * N + n0;
                __m256 av = _mm256_broadcast_ss(a + k);
                c0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b), c0);
                if (nn > 8) c1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b + 8), c1);
                if (nn > 16) c2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b + 16), c2);
                if (nn > 24) c3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b + 24), c3);
            }
            if (full) { _mm256_storeu_ps(c, c0); _mm256_storeu_ps(c + 8, c1); _mm256_storeu_ps(c + 16, c2); _mm256_storeu_ps(c + 24, c3); }
            else { alignas(16) float t[32]; _mm256_storeu_ps(t, c0); _mm256_storeu_ps(t + 8, c1); _mm256_storeu_ps(t + 16, c2); _mm256_storeu_ps(t + 24, c3); for (int n = 0; n < nn; ++n) c[n] = t[n]; }
        }
    }
}

int main() {
    struct Shape { int M, N, K; };
    const Shape shapes[] = {
        {3136, 64, 576},    // det 的 3x3 conv（im2col 后）
        {1024, 256, 256},   // 中等 conv
        {512, 256, 576},
        {64, 64, 64},       // 小特征图，fork 阈值以下
    };
    for (const Shape& s : shapes) {
        std::vector<float> A((size_t)s.M * s.K, 0.001f), B((size_t)s.K * s.N, 0.002f), C((size_t)s.M * s.N, 0.f);
        for (size_t i = 0; i < A.size(); i += 7) A[i] = 0.5f;
        for (size_t i = 0; i < B.size(); i += 11) B[i] = 0.25f;
        double best = 1e18;
        for (int rep = 0; rep < 7; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            sgemm_kernel(A.data(), B.data(), C.data(), s.M, s.N, s.K, s.N);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        double gflops = 2.0 * s.M * s.N * s.K / (best * 1e6);
        printf("%-18s M=%-5d N=%-4d K=%-4d  best %8.3f ms  %7.2f GFLOPS   chk=%.4f\n",
               "cpp", s.M, s.N, s.K, best, gflops, C[s.M / 2 * (size_t)s.N + s.N / 2]);
    }
    return 0;
}
