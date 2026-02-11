#pragma once


#define FORCE_INLINE __attribute__((always_inline))

namespace perf_gemm {

constexpr int NUM_XCDS = 8;
constexpr int AMDGCN_WAVEFRONT_SIZE = 64;

using bfloat16_t = __bf16;

template<typename dtype, int N>
struct PackN_t {
    using t = __attribute__((vector_size(N * sizeof(dtype)))) dtype;
    static constexpr auto n = N;
    static constexpr auto H = N / 2; 
    union {
        dtype x[N];
        t pack;
        struct { dtype low[H], high[H]; };
    };
};

using bf16x4_t = PackN_t<bfloat16_t, 4>;
using fp32x4_t = PackN_t<float, 4>;


__device__ FORCE_INLINE constexpr bfloat16_t fast_f32tob16(float f) {
#ifdef FAST_UNSAFE_CAST
    union {
        float fp32;
        unsigned int u32;
    } u = {f};
    u.u32 += 0x7FFF + ((u.u32 >> 16) & 1);
    auto ret = u.u32 >> 16;
    return reinterpret_cast<bfloat16_t &>(ret);
#else
    return static_cast<bfloat16_t>(f);
#endif
}


__device__ __host__ FORCE_INLINE constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

template<int a, int b>
__device__ __host__ FORCE_INLINE constexpr int exact_div() {
    static_assert(a % b == 0);
    return a / b;
}


} // namespace gemm_rs