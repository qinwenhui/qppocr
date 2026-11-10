//! CPU 特性/后端探测探针。x86_64 上打印 ISA 位，其余架构只报后端。
fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        println!("avx2={}", std::arch::is_x86_feature_detected!("avx2"));
        println!("fma={}", std::arch::is_x86_feature_detected!("fma"));
        println!("avx512f={}", std::arch::is_x86_feature_detected!("avx512f"));
    }
    let b = qppocr_kernels::detect_backend();
    println!("backend={b:?}");
}
