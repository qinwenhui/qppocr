// 模拟引擎的分配模式：常驻权重 + 反复申请的中间张量。
#include <cstdio>
#include <vector>
#include <chrono>
#include <cstring>
#include <windows.h>
#include <psapi.h>
int main() {
    std::vector<std::vector<float>> weights;
    for (int i = 0; i < 8; ++i) weights.emplace_back(2'500'000, 0.01f);   // ~80 MB 常驻
    double touched = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < 400; ++iter) {
        std::vector<std::vector<float>> tmp;
        for (int j = 0; j < 24; ++j) tmp.emplace_back((size_t)(20000 + j * 7000), 0.5f);
        for (auto& t : tmp) touched += t[t.size() / 2];
        for (auto& t : tmp) touched += t[0];
    }
    auto t1 = std::chrono::steady_clock::now();
    PROCESS_MEMORY_COUNTERS pmc{};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    printf("cpp   wall %7.1f ms  peakWS %6.1f MB  touched=%.1f\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count(),
           pmc.PeakWorkingSetSize / 1048576.0, touched);
}
