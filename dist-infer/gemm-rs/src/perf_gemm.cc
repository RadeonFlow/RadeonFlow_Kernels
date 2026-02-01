#include <c10/hip/HIPException.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <ck/utility/data_type.hpp>
#include <ck/utility/amd_buffer_addressing.hpp>
#include <ck/utility/ignore.hpp>
#define FAST_UNSAFE_CAST
// #define DEBUG_SIGNAL_CYCLE
// #define SWIZZLE_XCD_PID
// #define SWIZZLE_L2_TILE

#define FORCE_INLINE __attribute__((always_inline))

namespace roc_isa {
    constexpr int AMDGCN_WAVEFRONT_SIZE = 64;
namespace issue_latency {
    constexpr int v_mfma_f32_16x16x16_bf16 = 4;
    constexpr int ds_read_b128 = 2 * 4;
    constexpr int ds_write_b128 = 5 * 4;
    constexpr int buffer_load_dwordx2 = 1 * 2;
} // namespace issue_latency

// Trait for different GEMM size categories
enum class GemmSizeCategory {
    LARGE,   // 256x224, 224x256, 256x256
    MIDDLE,  // 256x128, 128x256, 128x128
    SMALL    // 128x64, 64x128, 128x32, 32x128, 64x32, 32x32
};

template<int BM, int BN>
struct GemmSizeTrait {
    // Large GEMM
    // (256x224) MFMA 224, DS_READ 30, DS_WRITE 15, BUFFER_LOAD 30
    // Middle GEMM
    // (256X128) MFMA 128, DS_READ_24, DS_WRITE 12, BUFFER_LOAD 24
    // (128X128) MFMA 64,  DS_READ 16, DS_WRITE 8,  BUFFER_LOAD 16
    // Small GEMM
    // (128x64) MFMA 32, DS_WRITE 6, DS_READ 12, BUFFER_LOAD 12
    // (128x32) MFMA 16, DS_WRITE 5, DS_READ 8 , BUFFER_LOAD 10
    // (64x32)  MFMA 8,  DS_WRITE 3, DS_READ 6 , BUFFER_LOAD 6
    static constexpr GemmSizeCategory category = 
        ((BM == 256 && BN == 224) || (BM == 224 && BN == 256) || (BM == 256 && BN == 256)) ? GemmSizeCategory::LARGE :
        ((BM == 256 && BN == 128) || (BM == 128 && BN == 256) || (BM == 128 && BN == 128)) ? GemmSizeCategory::MIDDLE :
        GemmSizeCategory::SMALL;
};

// Schedule configuration trait based on category
template<GemmSizeCategory category>
struct ScheduleConfig;

template<>
struct ScheduleConfig<GemmSizeCategory::LARGE> {
    // Stage 1: DS_WRITE(1) -> MFMA(2) -> VMEM(1) -> MFMA(3)
    static constexpr int stage1_ds_write = 1;
    static constexpr int stage1_mfma_before_vmem = 2;
    static constexpr int stage1_vmem = 1;
    static constexpr int stage1_mfma_after_vmem = 3;
    
    // Stage 2: MFMA(2) -> DS_READ(1)
    static constexpr int stage2_mfma = 2;
    static constexpr int stage2_ds_read = 1;
};

template<>
struct ScheduleConfig<GemmSizeCategory::MIDDLE> {
    // Stage 1: DS_WRITE(1) -> MFMA(2) -> VMEM(1) -> MFMA(1)
    static constexpr int stage1_ds_write = 1;
    static constexpr int stage1_mfma_before_vmem = 2;
    static constexpr int stage1_vmem = 1;
    static constexpr int stage1_mfma_after_vmem = 1;
    
    // Stage 2: MFMA(1) -> DS_READ(2)
    static constexpr int stage2_mfma = 1;
    static constexpr int stage2_ds_read = 2;
};

template<>
struct ScheduleConfig<GemmSizeCategory::SMALL> {
    // Stage 1: DS_WRITE(1) -> MFMA(1) -> VMEM(1) -> MFMA(1)
    static constexpr int stage1_ds_write = 1;
    static constexpr int stage1_mfma_before_vmem = 1;
    static constexpr int stage1_vmem = 1;
    static constexpr int stage1_mfma_after_vmem = 1;
    
    // Stage 2: MFMA(1) -> DS_READ(1)
    static constexpr int stage2_mfma = 1;
    static constexpr int stage2_ds_read = 1;
};

template<int BM, int BN, int BK, int NUM_THREADS, int WARP_M, int WARP_N>
struct InstCalculator {
    static constexpr int v_mfma_f32_16x16x16_bf16 = (BM * BN * BK) / (WARP_M * WARP_N) / (16*16*16);
    // Compiler will merge two ds_{read,write}_b64 to ds_{read,write`}2st64_b64
    static constexpr int ds_read_b128_a = (BM * BK / WARP_M) / 64 / 8;
    static constexpr int ds_read_b128_b = (BN * BK / WARP_N) / 64 / 8;
    static constexpr int ds_read_b128 = ds_read_b128_a + ds_read_b128_b;
    static constexpr int ds_write_b128_a = (BM * BK) / NUM_THREADS / 8;
    static constexpr int ds_write_b128_b = (BN * BK) / NUM_THREADS / 8;
    static constexpr int ds_write_b128 = ds_write_b128_a + ds_write_b128_b;
    static constexpr int buffer_load_dwordx2_a = (BM * BK) / NUM_THREADS / 4;
    static constexpr int buffer_load_dwordx2_b = (BN * BK) / NUM_THREADS / 4;
    static constexpr int buffer_load_dwordx2 = buffer_load_dwordx2_a + buffer_load_dwordx2_b;

    // Get schedule configuration based on BM and BN
    using size_trait = GemmSizeTrait<BM, BN>;
    using schedule_config = ScheduleConfig<size_trait::category>;
};

} // namespace roc_isa

namespace test {
    constexpr int BM = 128, BN = 128;
    constexpr int WARP_M = 2, WARP_N = 2, NUM_THREADS = 256, BK = 64;
    constexpr int MFMA_NUM = roc_isa::InstCalculator<BM, BN, BK, NUM_THREADS, WARP_M, WARP_N>::v_mfma_f32_16x16x16_bf16;
    constexpr int DS_READ_NUM = roc_isa::InstCalculator<BM, BN, BK, NUM_THREADS, WARP_M, WARP_N>::ds_read_b128;
    constexpr int DS_WRITE_NUM = roc_isa::InstCalculator<BM, BN, BK, NUM_THREADS, WARP_M, WARP_N>::ds_write_b128;
    constexpr int BUFFER_LOAD_NUM = roc_isa::InstCalculator<BM, BN, BK, NUM_THREADS, WARP_M, WARP_N>::buffer_load_dwordx2;
}


using bfloat16_t = __bf16;

__device__ __host__ FORCE_INLINE constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

template<int a, int b>
__device__ __host__ FORCE_INLINE constexpr int exact_div() {
    static_assert(a % b == 0);
    return a / b;
}

__device__ __host__ FORCE_INLINE constexpr int i_min(int a, int b) {
    return a < b ? a : b;
}

__device__ __host__ FORCE_INLINE constexpr int i_max(int a, int b) {
    return a > b ? a : b;
}


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
using bf16x8_t = PackN_t<bfloat16_t, 8>;
using fp32x8_t = PackN_t<float, 8>;
using i32x4_t = PackN_t<int32_t, 4>;

#define FORCE_INLINE __attribute__((always_inline))


__device__ ck::int32x4_t inline make_wave_buffer_resource(const void* ptr, uint32_t size = 0xffffffff) {
    ck::int32x4_t res;
    
    // Pack the 64-bit pointer into two 32-bit integers
    uint64_t ptr_val = reinterpret_cast<uint64_t>(ptr);
    res.x = static_cast<uint32_t>(ptr_val);
    res.y = static_cast<uint32_t>(ptr_val >> 32);
    
    // Set buffer size and format
    res.z = size;  // Buffer size in bytes
    res.w = 0x00020000;  // hardcoded for gfx942
    
    res.x = __builtin_amdgcn_readfirstlane(res.x);
    res.y = __builtin_amdgcn_readfirstlane(res.y);
    res.z = __builtin_amdgcn_readfirstlane(res.z);
    res.w = __builtin_amdgcn_readfirstlane(res.w);
    return res;
}

__device__ FORCE_INLINE bfloat16_t fast_f32tob16(float f) {
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


__device__ FORCE_INLINE float fast_b16tof32(bfloat16_t bf) {
#ifdef FAST_UNSAFE_CAST
    union {
        float fp32;
        unsigned int u32;
    } u;
    u.u32 = (reinterpret_cast<unsigned short&>(bf)) << 16;
    return u.fp32;
#else
    return static_cast<float>(bf);
#endif
}

__device__ void block_sync_lds() {
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_s_barrier();
}

__device__ void block_sync_gds() {
    __builtin_amdgcn_s_waitcnt(0xf70);
    __builtin_amdgcn_s_barrier();
}

template<int num_tile_m, int num_tile_n, int GROUP_SIZE_N = 8>
__device__ FORCE_INLINE void compute_tile_indices(
    int tile_id, 
    int &tile_m_id, 
    int &tile_n_id
) {
    if constexpr (GROUP_SIZE_N == 0) {
        // No swizzle
        tile_m_id = tile_id / num_tile_n;
        tile_n_id = tile_id % num_tile_n;
    } else {
        // Swizzle pattern for better L2 cache locality
        // Groups tiles in blocks of num_tile_m x GROUP_SIZE_N
        constexpr int num_pid_in_group = num_tile_m * GROUP_SIZE_N;
        
        // Which group does this tile belong to?
        const int group_id = tile_id / num_pid_in_group;
        
        // First N-dimension tile in this group
        const int first_pid_n = group_id * GROUP_SIZE_N;
        
        // Actual group size (handling boundary case)
        const int group_size_n = min(GROUP_SIZE_N, num_tile_n - first_pid_n);
        
        // Position within the group
        const int idx_in_group = tile_id % num_pid_in_group;
        
        // Swizzled tile indices: alternate N then M within group
        tile_n_id = first_pid_n + (idx_in_group % group_size_n);
        tile_m_id = idx_in_group / group_size_n;
    }
}

// M-dimension grouped version for better L2 cache locality when M > N
template<int num_tile_m, int num_tile_n, int GROUP_SIZE_M>
__device__ FORCE_INLINE void compute_tile_indices_m_grouped(
    int tile_id, 
    int &tile_m_id, 
    int &tile_n_id
) {
    if constexpr (GROUP_SIZE_M == 0) {
        // No swizzle
        tile_m_id = tile_id % num_tile_m;
        tile_n_id = tile_id / num_tile_m;
    } else {
        // Swizzle pattern for better L2 cache locality
        // Groups tiles in blocks of GROUP_SIZE_M x num_tile_n
        constexpr int num_pid_in_group = GROUP_SIZE_M * num_tile_n;
        
        // Which group does this tile belong to?
        const int group_id = tile_id / num_pid_in_group;
        
        // First M-dimension tile in this group
        const int first_pid_m = group_id * GROUP_SIZE_M;
        
        // Actual group size (handling boundary case)
        const int group_size_m = min(GROUP_SIZE_M, num_tile_m - first_pid_m);
        
        // Position within the group
        const int idx_in_group = tile_id % num_pid_in_group;
        
        // Swizzled tile indices: alternate M then N within group
        tile_m_id = first_pid_m + (idx_in_group % group_size_m);
        tile_n_id = idx_in_group / group_size_m;
    }
}

const int WORLD_SIZE = 8;
constexpr size_t MAX_IPC_MEM_SIZE = 256 * (1UL << 20); // 256MB
constexpr size_t SIGNAL_BUF_SIZE = 1 * (1UL << 20); // 1MB
constexpr int NUM_THREADS = 256;
constexpr int NUM_SMS = 304;
// Fused kernel: pid < NUM_GEMM_SMS runs GEMM, next NUM_RS_SMS pids run ReduceScatter
template<
    int M, int N, int K, bool LOAD_BIAS,
    int BM, int BN, int BK,
    int NUM_SMS, int NUM_GEMM_SMS, int NUM_RS_SMS,
    int NUM_THREADS, int WARP_M, int WARP_N, int GROUP_SIZE_N,
    int SPLIT_K = 1
>
__launch_bounds__(NUM_THREADS)
__global__ void fused_gemm_rs_kernel(
    const bfloat16_t *x,               // GEMM: M x K
    const bfloat16_t *w,               // GEMM: N x K
    const bfloat16_t *b,               // GEMM: N
    const std::array<bfloat16_t*, WORLD_SIZE> c_all, // unified C buffers for all ranks (GEMM writes c_all[rank], RS reads all)
    const std::array<int*, WORLD_SIZE> signal_all,   // unified signal buffers for all ranks (GEMM writes signal_all[rank], RS reads all)
    int signal_val,
    bfloat16_t *rs_out,                // RS out: [M / WORLD_SIZE, N]
    float *workspace,                  // temporary FP32 workspace: [SPLIT_K, M, N]
    int rank
) {
    static_assert(SPLIT_K >= 1, "SPLIT_K must be >= 1");
    static_assert(K % SPLIT_K == 0, "K must be divisible by SPLIT_K");

    const int pid = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const int tid = threadIdx.x;
    const int lane_id = __lane_id();
    const int warp_id = __builtin_amdgcn_readfirstlane(tid / roc_isa::AMDGCN_WAVEFRONT_SIZE);
    __builtin_assume(pid >= 0 && pid < NUM_SMS);
    __builtin_assume(tid >= 0 && tid < NUM_THREADS);
    __builtin_assume(lane_id >= 0 && lane_id < 64);

    if (pid < NUM_GEMM_SMS) {
        // GEMM Kernel
        constexpr int num_tile_m = ceil_div(M, BM);
        constexpr int num_tile_n = ceil_div(N, BN);    
        constexpr int num_tiles = num_tile_m * num_tile_n * SPLIT_K;
        // each split handles K_per_split
        constexpr int K_per_split = exact_div<K, SPLIT_K>();
        constexpr int num_tile_k = ceil_div(K_per_split, BK);

        using inst_nums = roc_isa::InstCalculator<BM, BN, BK, NUM_THREADS, WARP_M, WARP_N>;
        static_assert(BK % 4 == 0 && NUM_THREADS * 4 % BK == 0);
        constexpr int WM = 16, WN = 16, WK = 16;

        constexpr int Frag_M = exact_div<BM, WM * WARP_M>();
        constexpr int Frag_N = exact_div<BN, WN * WARP_N>();
        constexpr int Frag_K = exact_div<BK, WK>();
        const int warp_m = warp_id / WARP_N;
        const int warp_n = warp_id % WARP_N;
        using FragX = bf16x4_t;
        using FragW = bf16x4_t;
        using FragC = fp32x4_t;
        __shared__ bfloat16_t s_x[BM][BK];
        __shared__ bfloat16_t s_w[BN][BK];
        bf16x4_t vgpr_x[ceil_div(BM * BK, NUM_THREADS * 4)];
        bf16x4_t vgpr_w[ceil_div(BN * BK, NUM_THREADS * 4)];

        FragC frag_c[Frag_M][Frag_N];
        FragX frag_x[Frag_M][Frag_K];
        FragW frag_w[Frag_N][Frag_K];
        fp32x4_t out_fp32[Frag_M][Frag_N]; // AccVGPR -> VGPR Buffer
        auto b_arr = ck::make_wave_buffer_resource<bfloat16_t>(const_cast<bfloat16_t*>(b), N);
        auto c_arr = ck::make_wave_buffer_resource<bfloat16_t>(c_all[rank], M * N);
        auto x_arr = ck::make_wave_buffer_resource<bfloat16_t>(const_cast<bfloat16_t*>(x), M * K);
        auto w_arr = ck::make_wave_buffer_resource<bfloat16_t>(const_cast<bfloat16_t*>(w), N * K);
        auto *signal_arr = reinterpret_cast<int (*)[num_tile_n]>(signal_all[rank]);
            

        // TODO: optimized use ck::amd_buffer_store 
        int last_tile_m_id = num_tile_m, last_tile_n_id = num_tile_n;
        auto release_signal = [&]() FORCE_INLINE {
            if (tid == 0) {
                __hip_atomic_store(&signal_arr[last_tile_m_id][last_tile_n_id], signal_val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            }
            
        };

        constexpr int NUM_XCDS = 8;
        for (int tile_id=pid; tile_id<num_tiles; tile_id+=NUM_GEMM_SMS) {
            int tile_id1 = (tile_id % NUM_XCDS) * (NUM_GEMM_SMS / NUM_XCDS) + (tile_id / NUM_XCDS);
            int split_k_id = tile_id1 % SPLIT_K;
            int tile_n_id = (tile_id1 / SPLIT_K) % num_tile_n;
            int tile_m_id = (tile_id1 / SPLIT_K) / num_tile_n;
            last_tile_m_id = tile_m_id;
            last_tile_n_id = tile_n_id;
            compute_tile_indices<num_tile_m, num_tile_n, GROUP_SIZE_N>(tile_id / SPLIT_K, tile_m_id, tile_n_id);
            int m = tile_m_id * BM;
            int n = tile_n_id * BN;

            int k_offset = split_k_id * K_per_split * sizeof(bfloat16_t);
            int v_offset = ((tid * 4 / BK) * K + (tid * 4 % BK)) * sizeof(bfloat16_t);
            auto load_vgpr = [&](int k) FORCE_INLINE {
                uint32_t src_addr_shift = ((K_per_split % BK == 0) || (k + tid * 4 % BK < K_per_split)) ? 0 : 0x80000000;
                ck::static_for<0, sizeof(vgpr_x) / sizeof(vgpr_x[0]), 1>{}([&](auto t) {
                    int s_offset = ((m * K + k) + t * NUM_THREADS * 4 / BK * K) * sizeof(bfloat16_t);
                    vgpr_x[t] = __builtin_bit_cast(bf16x4_t, ck::amd_buffer_load_impl_raw<sizeof(bf16x4_t)>(
                        x_arr, v_offset + src_addr_shift, s_offset + k_offset));
                });
                ck::static_for<0, sizeof(vgpr_w) / sizeof(vgpr_w[0]), 1>{}([&](auto t) {
                    int s_offset = ((n * K + k) + t * NUM_THREADS * 4 / BK * K) * sizeof(bfloat16_t);
                    vgpr_w[t] = __builtin_bit_cast(bf16x4_t, ck::amd_buffer_load_impl_raw<sizeof(bf16x4_t)>(
                        w_arr, v_offset + src_addr_shift, s_offset + k_offset));
                });
                
            };


            auto load_lds = [&]() FORCE_INLINE {
                // diagonal swizzle, shape=[16, 64] dtype=bfloat16
                #pragma unroll
                for (int t=0;t<sizeof(vgpr_x)/sizeof(vgpr_x[0]);++t) {
                    int row0 = t * NUM_THREADS * 4 / BK;
                    int row1 = tid * 4 / BK;
                    int col0 = tid * 4 % BK;
                    int col1 = BK <= 64 ? (row1 * 4 + col0) % BK : ((row0 + row1) * 4 + col0) % BK;
                    *reinterpret_cast<bf16x4_t*>(&s_x[row0 + row1][col1]) = vgpr_x[t];
                }
                #pragma unroll
                for (int t=0;t<sizeof(vgpr_w)/sizeof(vgpr_w[0]);++t) {
                    int row0 = t * NUM_THREADS * 4 / BK;
                    int row1 = tid * 4 / BK;
                    int col0 = tid * 4 % BK;
                    int col1 = BK <= 64 ? (row1 * 4 + col0) % BK : ((row0 + row1) * 4 + col0) % BK;
                    *reinterpret_cast<bf16x4_t*>(&s_w[row0 + row1][col1]) = vgpr_w[t];
                }
            };

            auto zero_all_frags = [&]() FORCE_INLINE {
                ck::static_for<0, Frag_M, 1>{}([&](auto i) {
                    ck::static_for<0, Frag_N, 1>{}([&](auto j) {
                        ck::static_for<0, 4, 1>{}([&](auto t) { frag_c[i][j].x[t] = 0; });
                    });
                });
            };



            auto frags_load = [&]() {
                #pragma unroll
                for (int k=0;k<Frag_K;++k) {
                    #pragma unroll
                    for (int i=0;i<Frag_M;++i) {
                        const int row1 = (warp_m * Frag_M + i) * WM;
                        const int row0 = lane_id % 16;
                        const int col0 = k * 16 + lane_id / 16 * 4;
                        const int col1 = BK <= 64 ? (row0 * 4 + col0) % BK : ((row0 + row1) * 4 + col0) % BK;
                        frag_x[i][k] = *reinterpret_cast<const bf16x4_t*>(&s_x[row0 + row1][col1]);
                    }
                }
                #pragma unroll
                for (int k=0;k<Frag_K;++k) {
                    #pragma unroll
                    for (int j=0;j<Frag_N;++j) {
                        const int row1 = (warp_n * Frag_N + j) * WN;
                        const int row0 = lane_id % 16;
                        const int col0 = k * 16 + lane_id / 16 * 4;
                        const int col1 = BK <= 64 ? (row0 * 4 + col0) % BK : ((row0 + row1) * 4 + col0) % BK;
                        frag_w[j][k] = *reinterpret_cast<const bf16x4_t*>(&s_w[row0 + row1][col1]);
                    }
                }
            };

            auto frags_mfma = [&] { 
                #pragma unroll
                for (int i=0;i<Frag_M;++i) {
                    #pragma unroll
                    for (int j=0;j<Frag_N;++j) {
                        #pragma unroll
                        for (int k=0;k<Frag_K;++k) {   
                            // a: [16][16], b: [16][16], c: [16][16]
                            // mfma requires a: row-major, b: col-major, out: col-major
                            // so we compute w^T * x^T = c^T so we can treat out as col-major
                            frag_c[i][j].pack = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(frag_w[j][k].pack, frag_x[i][k].pack, frag_c[i][j].pack, 0, 0, 0);
                        }
                    }
                }
            };
            

            auto store_frags = [&]() FORCE_INLINE {
                
                // AccVGPR -> VGPR
                #pragma unroll
                for (int i=0; i<Frag_M; ++i) {
                    #pragma unroll
                    for (int j=0; j<Frag_N; ++j) {
                        #pragma unroll
                        for (int t=0; t<4; ++t) {
                            out_fp32[i][j].x[t] = frag_c[i][j].x[t];
                        }
                    }
                }


                if constexpr (SPLIT_K == 1) {
                    #pragma unroll
                    for (int i=0; i<Frag_M; ++i) {
                        #pragma unroll
                        for (int j=0; j<Frag_N; ++j) {
                            int col = lane_id / 16 * 4;
                            uint32_t src_addr_shift = (N % BN == 0) || (n + (j + warp_n * Frag_N) * WN + col < N) ? 0 : 0x80000000;
                            int b_s_offset = (n + (j + warp_n * Frag_N) * WN) * sizeof(bfloat16_t);
                            int b_v_offset = col * sizeof(bfloat16_t) + src_addr_shift;
                            auto b_vec = __builtin_bit_cast(bf16x4_t, ck::amd_buffer_load_impl_raw<sizeof(bf16x4_t)>(
                                b_arr, b_v_offset, b_s_offset));
                            #pragma unroll
                            for (int t = 0; t < 4; ++t) {
                                out_fp32[i][j].x[t] += static_cast<float>(LOAD_BIAS ? b_vec.x[t] : 0);
                            }
                        }
                    }

                    ck::static_for<0, Frag_M, 1>{}([&](auto i) {
                        ck::static_for<0, Frag_N, 1>{}([&](auto j) {
                            bf16x4_t c_out_bf16;
                            #pragma unroll
                            for (int t = 0; t < 4; ++t) c_out_bf16.x[t] = fast_f32tob16(out_fp32[i][j].x[t]);
                            int row = lane_id % 16;
                            int col = lane_id / 16 * 4;
                            uint32_t src_addr_shift = (N % BN == 0) || (n + (j + warp_n * Frag_N) * WN + col < N) ? 0 : 0x80000000;
                            int b_s_offset = (n + (j + warp_n * Frag_N) * WN) * sizeof(bfloat16_t);
                            int c_s_offset = b_s_offset + (m + (i + warp_m * Frag_M) * WM) * N * sizeof(bfloat16_t);
                            int c_v_offset = col * sizeof(bfloat16_t) + src_addr_shift + (row * N) * sizeof(bfloat16_t);
                            ck::amd_buffer_store_impl_raw<sizeof(bf16x4_t), ck::AmdBufferCoherenceEnum::WAVE_NT1>(c_out_bf16.pack, c_arr, c_v_offset, c_s_offset);
                        });
                    });
                } else {
                    // SPLIT_K > 1: store FP32 partials into workspace [split_id, M, N] (row-major floats)
                    auto ws_arr = ck::make_wave_buffer_resource<float>(workspace + split_k_id * M * N, M * N);
                    ck::static_for<0, Frag_M, 1>{}([&](auto i) {
                        ck::static_for<0, Frag_N, 1>{}([&](auto j) {
                            int row = lane_id % 16;
                            int col = lane_id / 16 * 4;
                            uint32_t src_addr_shift = (N % BN == 0) || (n + (j + warp_n * Frag_N) * WN + col < N) ? 0 : 0x80000000;
                            int b_s_offset = (n + (j + warp_n * Frag_N) * WN) * sizeof(float);
                            int c_s_offset = b_s_offset + (m + (i + warp_m * Frag_M) * WM) * N * sizeof(float);
                            int c_v_offset = col * sizeof(float) + src_addr_shift + (row * N) * sizeof(float);
                            ck::amd_buffer_store_impl_raw<sizeof(fp32x4_t), ck::AmdBufferCoherenceEnum::WAVE_NT1>(out_fp32[i][j].pack, ws_arr, c_v_offset, c_s_offset);
                        });
                    });
                }
            };




            load_vgpr(0);                    // GDS -> VGPR #0
            load_lds();                      // VGPR -> LDS #0
            load_vgpr(1 * BK);               // GDS -> VGPR #1
            zero_all_frags();
            // __builtin_amdgcn_s_waitcnt(0x70);
            // __builtin_amdgcn_s_barrier();
            block_sync_lds();
            // release_signal();
            frags_load();                    // LDS -> FRAG #0
            __builtin_amdgcn_sched_barrier(0);
            for (int tile_k_id = 1; tile_k_id < (num_tile_k - 1); ++tile_k_id) {
                block_sync_lds();
                // Stage 1
                load_lds();                             // VGPR -> LDS #1
                load_vgpr((tile_k_id + 1) * BK);        // GDS -> VGPR #2(k+1)
                frags_mfma();                           // MFMA #0(k-1)
                #pragma unroll
                for (int k = 0; k < inst_nums::buffer_load_dwordx2; ++k) {
                    __builtin_amdgcn_sched_group_barrier(0x200, inst_nums::schedule_config::stage1_ds_write, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, inst_nums::schedule_config::stage1_mfma_before_vmem, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x020, inst_nums::schedule_config::stage1_vmem, 0); // VMEM read
                    __builtin_amdgcn_sched_group_barrier(0x008, inst_nums::schedule_config::stage1_mfma_after_vmem, 0); // MFMA
                }
                block_sync_lds();
                // Stage 2                       
                frags_load();                           // LDS -> FRAG #1(k)
                #pragma unroll
                for (int k = 0; k < inst_nums::ds_read_b128; ++k) {
                    __builtin_amdgcn_sched_group_barrier(0x008, inst_nums::schedule_config::stage2_mfma, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x100, inst_nums::schedule_config::stage2_ds_read, 0); // DS read
                }
                __builtin_amdgcn_sched_barrier(0);
    
            }
            frags_mfma();                               // MFMA #1(n-2)
            block_sync_lds();
            load_lds();                                 // VGPR -> LDS #2(n-1)
            block_sync_lds();
            frags_load();                               // LDS -> FRAG #2(n-1)
            frags_mfma();                               // MFMA #2(n-1)
            
            // __builtin_amdgcn_sched_barrier(0);
            store_frags();
            __syncthreads();
            if (tid == 0) {
                __hip_atomic_store(&signal_arr[tile_m_id][tile_n_id], signal_val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
        // block_sync_gds();
        // release_signal();
    } else if (pid >= NUM_SMS - NUM_RS_SMS) {
        const int pid_rs = pid - (NUM_SMS - NUM_RS_SMS);
        const int initial_tile_id = pid_rs * WARP_M * WARP_N + warp_id;
        const int batch_tile = WARP_M * WARP_N * NUM_RS_SMS;
        constexpr int M_per_rank = M / WORLD_SIZE;
        constexpr int num_tile_m = ceil_div(M_per_rank, BM);
        constexpr int num_tile_n = ceil_div(N, BN);
        i32x4_t swizzle_c[WORLD_SIZE]; // SGPR
        int* swizzle_signal = signal_all[(lane_id / 2) % WORLD_SIZE]; // VGPR
        // TODO: benchmark PIPELINE_STAGES
        constexpr int TOTAL_STAGES = BM * BN / 64 / 8;
        constexpr int PIPELINE_STAGES = i_min(4, i_max(TOTAL_STAGES / 2, 1));
        bf16x8_t reg_c_buf[PIPELINE_STAGES][WORLD_SIZE];
        fp32x8_t reg_c_sum[PIPELINE_STAGES];
        ck::static_for<0, WORLD_SIZE, 1>{}([&](auto k) {
            int r = __builtin_amdgcn_readfirstlane((pid_rs + k) % WORLD_SIZE);
            swizzle_c[k].pack = ck::make_wave_buffer_resource<bfloat16_t>(c_all[r] + rank * M_per_rank * N, M_per_rank * N);
        });

        for (int tile_id=initial_tile_id; tile_id<num_tile_m*num_tile_n; tile_id+=batch_tile) {
            int tile_m_id, tile_n_id;
            compute_tile_indices<num_tile_m, num_tile_n, 0>(tile_id, tile_m_id, tile_n_id);

            int signal_id_m_begin = (rank * M_per_rank + tile_m_id * BM) / BM;
            int signal_id_m_close = std::min(rank * M_per_rank + M_per_rank - 1, rank * M_per_rank + tile_m_id * BM + BM - 1) / BM;
            int signal_id_n = tile_n_id;

#ifdef DEBUG_SIGNAL_CYCLE
            long long begin;
            if (M == 8192 && N == 8192 && rank == 0 && tid == 0) {
                begin = clock64();
            }
#endif

            // __builtin_amdgcn_sched_barrier(0);
            // __builtin_amdgcn_s_setprio(0);
            // __builtin_amdgcn_sched_barrier(0);

            if (lane_id < WORLD_SIZE * 2) {
                const int signal_id = lane_id % 2;
                const int target_rank = lane_id / 2;
                if (signal_id_m_begin + signal_id <= signal_id_m_close) {
                    auto *signal_arr = reinterpret_cast<const int (*)[num_tile_n]>(swizzle_signal);
                    while (__hip_atomic_load(&signal_arr[signal_id_m_begin + signal_id][signal_id_n], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != signal_val) {
                        // __builtin_amdgcn_s_sleep(4);
                    }
                }
            }

            // __builtin_amdgcn_sched_barrier(0);
            // __builtin_amdgcn_s_setprio(1);
            // __builtin_amdgcn_sched_barrier(0);
            // __threadfence_system();

            // #pragma unroll
            // for (int r = 0; r < WORLD_SIZE; ++r) {
            //     // int v_offset = signal_id_m_begin + lane_id <= signal_id_m_close ? lane_id * N * sizeof(int) : 0x80000000;
            //     // int s_offset = (signal_id_m_begin * N + signal_id_n) * sizeof(int);
            //     // while (!(v_offset == 0x80000000 || ck::amd_buffer_load_impl_raw<sizeof(int), ck::AmdBufferCoherenceEnum::SYSTEM_NT1>(swizzle_signal[r].pack, v_offset, s_offset) == signal_val)) {
                    
            //     // }
            //     if (signal_id_m_begin + lane_id <= signal_id_m_close) {
            //         auto *signal_arr = reinterpret_cast<const int (*)[num_tile_n]>(signal_all[r]);
            //         while (__hip_atomic_load(&signal_arr[signal_id_m_begin + lane_id][signal_id_n], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != signal_val) {}
            //     }
            // }
#ifdef DEBUG_SIGNAL_CYCLE
            if (tid == 0 && M == 8192 && N == 8192 && rank == 0) {
                auto duration = clock64() - begin;
                printf("rank0 wait signal: %lld cycles\n", duration);
            }
#endif
            // __threadfence_system();

            auto buf_load = [&](int t) FORCE_INLINE {
                int v_offset = ((lane_id * 8 / BN) * N + (lane_id * 8 % BN)) * sizeof(bfloat16_t);
                int s_offset = ((t * 8 * 64 / BN + tile_m_id * BM) * N + (tile_n_id * BN)) * sizeof(bfloat16_t);
                uint32_t src_addr_shift = ((N % BN == 0) || ((tile_n_id * BN) + (lane_id * 8 % BN) < N)) ? 0 : 0x80000000;
                int stage = t % PIPELINE_STAGES;
                ck::static_for<0, WORLD_SIZE, 1>{}([&](auto r) {
                    reg_c_buf[stage][r] = __builtin_bit_cast(bf16x8_t, ck::amd_buffer_load_impl_raw<sizeof(bf16x8_t), ck::AmdBufferCoherenceEnum::SYSTEM_NT0>(
                        swizzle_c[r].pack, v_offset + src_addr_shift, s_offset));
                });
            };

            auto compute_store = [&](int t) FORCE_INLINE {
                int stage = t % PIPELINE_STAGES;
                int v_offset = ((lane_id * 8 / BN) * N + (lane_id * 8 % BN)) * sizeof(bfloat16_t);
                int s_offset = ((t * 8 * 64 / BN + tile_m_id * BM) * N + (tile_n_id * BN)) * sizeof(bfloat16_t);
                uint32_t src_addr_shift = ((N % BN == 0) || ((tile_n_id * BN) + (lane_id * 8 % BN) < N)) ? 0 : 0x80000000;
                ck::static_for<0, WORLD_SIZE, 1>{}([&](auto r) {
                    ck::static_for<0, 8, 1>{}([&](auto k) { 
                        if constexpr(r == 0) reg_c_sum[stage].x[k] = reg_c_buf[stage][r].x[k];
                        else reg_c_sum[stage].x[k] += reg_c_buf[stage][r].x[k];
                    });
                });
                auto out_arr = ck::make_wave_buffer_resource<bfloat16_t>(rs_out, M_per_rank * N);
                bf16x8_t out_val;
                ck::static_for<0, 8, 1>{}([&](auto k) { out_val.x[k] = static_cast<bfloat16_t>(reg_c_sum[stage].x[k]); });
                ck::amd_buffer_store_impl_raw<sizeof(bf16x8_t), ck::AmdBufferCoherenceEnum::WAVE_NT1>(out_val.pack, out_arr, v_offset + src_addr_shift, s_offset);
            };

    
            
            #pragma unroll
            for (int t = 0; t < PIPELINE_STAGES; ++t) {
                buf_load(t);
            }
            if constexpr (PIPELINE_STAGES < TOTAL_STAGES) {
                #pragma unroll PIPELINE_STAGES
                for (int t = 0; t < TOTAL_STAGES - PIPELINE_STAGES; t++) {
                    compute_store(t);
                    buf_load(t + PIPELINE_STAGES);
                }
            }
            #pragma unroll
            for (int t = TOTAL_STAGES - PIPELINE_STAGES; t < TOTAL_STAGES; t++) {
                compute_store(t);
            }
        }
        
    }
}

template<int M, int N, bool LOAD_BIAS, int SPLITK_K, int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE)
__global__ void reduce_kernel(const float *workspace, const bfloat16_t *b, bfloat16_t *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;
    if (idx >= M * N) return;
    static_assert(N % 4 == 0, "N must be multiple of 4");
    auto *ws_arr = reinterpret_cast<const float (*)[M][N]>(workspace); // [SPLITK, M, N]
    auto *b_arr = reinterpret_cast<const bfloat16_t (*)>(b); // [N]
    auto *c_arr = reinterpret_cast<bfloat16_t (*)[N]>(c); // [M, N]
    fp32x4_t sum = {};
    int row = idx / N, col = idx % N;
    #pragma unroll
    for (auto k = 0; k < SPLITK_K; ++k) {
        fp32x4_t data = *reinterpret_cast<const fp32x4_t*>(&ws_arr[k][row][col]);
        ck::static_for<0, 4, 1>{}([&](auto i) { sum.x[i] += data.x[i]; });
    }
    if constexpr (LOAD_BIAS) {
        bf16x4_t bias = *reinterpret_cast<const bf16x4_t*>(&b_arr[col]);
        ck::static_for<0, 4, 1>{}([&](auto i) { sum.x[i] += static_cast<float>(bias.x[i]); });
    }
    bf16x4_t out;
    ck::static_for<0, 4, 1>{}([&](auto i) { out.x[i] = fast_f32tob16(sum.x[i]); });
    *reinterpret_cast<bf16x4_t*>(&c_arr[row][col]) = *reinterpret_cast<bf16x4_t*>(&out);
}

template<
    int M, int N, int K, int LOAD_BIAS,
    int BM, int BN, int BK,
    int NUM_SMS, int NUM_GEMM_SMS, int NUM_RS_SMS,
    int NUM_THREADS, int WARP_M, int WARP_N, int GROUP_SIZE_N,
    int SPLIT_K = 1
>
void kernel_launcher(
    const bfloat16_t* x, const bfloat16_t* w, const bfloat16_t* b,
    const std::array<bfloat16_t*, WORLD_SIZE>& c_all,
    const std::array<int*, WORLD_SIZE>& signal_all,
    int signal_val, bfloat16_t* rs_out, float* workspace, int rank)
{
    dim3 grid(NUM_SMS);
    dim3 block(NUM_THREADS);
    auto stream = at::cuda::getCurrentHIPStream().stream();
    if constexpr (SPLIT_K == 1) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fused_gemm_rs_kernel<M,N,K,LOAD_BIAS,BM,BN,BK,NUM_SMS,NUM_GEMM_SMS,NUM_RS_SMS,NUM_THREADS,WARP_M,WARP_N,GROUP_SIZE_N,SPLIT_K>),
            grid, block, 0, stream,
            x, w, b, c_all, signal_all, signal_val, rs_out,
            /*workspace*/ static_cast<float*>(nullptr),
            rank);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fused_gemm_rs_kernel<M,N,K,LOAD_BIAS,BM,BN,BK,NUM_SMS,NUM_GEMM_SMS,NUM_RS_SMS,NUM_THREADS,WARP_M,WARP_N,GROUP_SIZE_N,SPLIT_K>),
            grid, block, 0, stream,
            x, w, b, c_all, signal_all, signal_val, rs_out,
            workspace,
            rank);
        constexpr int BLOCK_SIZE = 512;
        dim3 block_reduce(BLOCK_SIZE);
        dim3 grid_reduce(exact_div<M * N, BLOCK_SIZE * 4>());
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(reduce_kernel<M,N,LOAD_BIAS,SPLIT_K,BLOCK_SIZE>),
            grid, block, 0, stream,
            workspace, b, c_all[rank]);
    }
}

#ifndef __PERF_GEMM_HEADER__

using KernelFn = void (*)(const bfloat16_t* x, const bfloat16_t* w, const bfloat16_t* b,
    const std::array<bfloat16_t*, WORLD_SIZE>& c_all,
    const std::array<int*, WORLD_SIZE>& signal_all,
    int signal_val, bfloat16_t* rs_out, float* workspace, int rank);

struct KernelRegistery {
    std::unordered_map<int64_t, KernelFn> gemm_map;
    std::unordered_map<int64_t, KernelFn> rs_map;
    std::unordered_map<int64_t, KernelFn> fused_full_map;

    void* workspace_ptr = nullptr;
    size_t workspace_size = 0;
};

union ShapeKey {
    struct { uint16_t M, N, K, PAD = 0; } data;
    int64_t key;
    static_assert(sizeof(data) == sizeof(key));
};


    
KernelRegistery& get_kernel_registry() {
    static KernelRegistery registry;
    constexpr bool WB = true; // w/ bias
    constexpr bool WO = false; // w/o bias
#ifndef __GPUMODE_BENCHMARK__
#define REGISTER_KERNELS(M,N,K,LOAD_BIAS,BM,BN,BK,WARP_M,WARP_N,GROUP_SIZE_N,NUM_RS_SMS,SPLIT_K) \
    registry.rs_map[ShapeKey{M,N,0}.key] = kernel_launcher<M,N,K,LOAD_BIAS,BM,BN,BK,NUM_SMS,0,NUM_RS_SMS,NUM_THREADS,WARP_M,WARP_N,GROUP_SIZE_N,SPLIT_K>; \
    registry.gemm_map[ShapeKey{M,N,K}.key] = kernel_launcher<M,N,K,LOAD_BIAS,BM,BN,BK,NUM_SMS,NUM_SMS - NUM_RS_SMS,0,NUM_THREADS,WARP_M,WARP_N,GROUP_SIZE_N,SPLIT_K>; \
    registry.fused_full_map[ShapeKey{M,N,K}.key] = kernel_launcher<M,N,K,LOAD_BIAS,BM,BN,BK,NUM_SMS,NUM_SMS - NUM_RS_SMS,NUM_RS_SMS,NUM_THREADS,WARP_M,WARP_N,GROUP_SIZE_N,SPLIT_K>;
// #define REGISTER_KERNELS(M,N,K,LOAD_BIAS,BM,BN,BK,WARP_M,WARP_N,GROUP_SIZE_N,NUM_RS_SMS,SPLIT_K) \
//     registry.gemm_map[ShapeKey{M,N,K}.key] = kernel_launcher<M,N,K,LOAD_BIAS,BM,BN,BK,NUM_SMS,NUM_SMS - NUM_RS_SMS,0,NUM_THREADS,WARP_M,WARP_N,GROUP_SIZE_N,SPLIT_K>; 
    

    if (__builtin_expect(registry.workspace_ptr == nullptr, false)) {
        // minimal registration updated with SPLIT_K (use 1 for now)
        // REGISTER_KERNELS(64,   7168, 2304, WB, 32,  64,  64, 2, 2, 16, 8, 1);
        // REGISTER_KERNELS(512,  4096, 1536, WB, 64,  128, 64, 2, 2, 16, 8, 1);
        // REGISTER_KERNELS(2048, 2880, 360,  WB, 128, 128, 64, 2, 2, 16, 8, 1);
        // REGISTER_KERNELS(4096, 4096, 512,  WB, 224, 256, 64, 2, 2, 16, 8, 1);
        // REGISTER_KERNELS(8192, 4096, 1792, WB, 224, 256, 64, 2, 2, 8,  8, 1);
        // REGISTER_KERNELS(8192, 8192, 3696, WB, 224, 256, 64, 2, 2, 8,  8, 1);

        // AG
        // REGISTER_KERNELS(64,   2304, 7168, WB, 64,  64,  64, 2, 2, 0, 4, 16);
        // REGISTER_KERNELS(512,  1536, 4096, WB, 64,  128, 64, 2, 2, 16, 4, 1);
        // REGISTER_KERNELS(2048, 360,  2880, WB, 128, 64,  64, 2, 2, 16, 4, 1);
        // REGISTER_KERNELS(4096, 512,  4096, WB, 256, 128, 64, 2, 2, 16, 4, 1);
        // REGISTER_KERNELS(8192, 1792, 4096, WB, 256, 224, 64, 2, 2, 32, 4, 1);
        // REGISTER_KERNELS(8192, 3696, 8192, WB, 256, 224, 64, 2, 2, 32, 0, 1);


        // REGISTER_KERNELS(8192, 8192, 3696, WB, 256, 224, 64, 2, 2, 16, 8, 1);  // 619.00 TFLOPS

        REGISTER_KERNELS(64  , 7168, 2304, WB, 32 , 64 , 128, 2, 2, 8 , 32, 1 );  // 90.92 TFLOPS
        REGISTER_KERNELS(512 , 4096, 1536, WB, 128, 64 , 128, 2, 2, 8 , 48, 1 );  // 194.13 TFLOPS
        REGISTER_KERNELS(2048, 2880, 360 , WB, 128, 256, 64, 2, 2, 8 , 48, 1 );  // 150.06 TFLOPS
        REGISTER_KERNELS(4096, 4096, 512 , WB, 256, 128, 64, 2, 2, 16, 48, 1 );  // 240.03 TFLOPS
        REGISTER_KERNELS(8192, 4096, 1792, WB, 224, 256, 64, 2, 2, 16, 32, 1 );  // 491.16 TFLOPS
        REGISTER_KERNELS(8192, 8192, 3696, WB, 224, 256, 64, 2, 2, 0 , 8, 1 );  // 620.40 TFLOPS

        // REGISTER_KERNELS(64  , 2304, 7168, WB, 32 , 64 , 128, 2, 2, 0 , 8, 4);  // 51.87 TFLOPS
        // REGISTER_KERNELS(512 , 1536, 4096, WB, 64 , 64 , 128, 2, 2, 32, 8, 1);  // 166.24 TFLOPS
        // REGISTER_KERNELS(2048, 360 , 2880, WB, 64 , 64 , 128, 2, 2, 8 , 8, 1);  // 161.53 TFLOPS
        // REGISTER_KERNELS(4096, 512 , 4096, WB, 128, 64 , 128, 2, 2, 40, 8, 1);  // 253.96 TFLOPS
        // REGISTER_KERNELS(8192, 1792, 4096, WB, 256, 224, 64,  2, 2, 8 , 8, 1);  // 494.63 TFLOPS
        // REGISTER_KERNELS(8192, 3696, 8192, WB, 256, 224, 64,  2, 2, 32, 8, 1);  // 577.45 TFLOPS

        constexpr size_t PREALLOC_WORKSPACE = 2 * 1024UL * 1024UL * 1024UL; // 2GB
        registry.workspace_size = PREALLOC_WORKSPACE;
        C10_HIP_CHECK(hipMalloc(&registry.workspace_ptr, registry.workspace_size));
        C10_HIP_CHECK(hipMemset(registry.workspace_ptr, 0, registry.workspace_size));
    }

#undef REGISTER_KERNELS

#else
// #define REGISTER_KERNELS(M,N,K,LOAD_BIAS,BM,BN,BK,WARP_M,WARP_N,GROUP_SIZE_N,NUM_RS_SMS,SPLIT_K) \
//     registry.fused_full_map[ShapeKey{M,N,K}.key] = kernel_launcher<M,N,K,LOAD_BIAS,BM,BN,BK,NUM_SMS,NUM_SMS - NUM_RS_SMS,NUM_RS_SMS,NUM_THREADS,WARP_M,WARP_N,GROUP_SIZE_N,SPLIT_K>;

#define REGISTER_KERNELS(M,N,K,LOAD_BIAS,BM,BN,BK,WARP_M,WARP_N,GROUP_SIZE_N,NUM_RS_SMS,SPLIT_K) \
    registry.rs_map[ShapeKey{M,N,0}.key] = kernel_launcher<M,N,K,LOAD_BIAS,BM,BN,BK,NUM_SMS,0,NUM_RS_SMS,NUM_THREADS,WARP_M,WARP_N,GROUP_SIZE_N,SPLIT_K>; \
    registry.gemm_map[ShapeKey{M,N,K}.key] = kernel_launcher<M,N,K,LOAD_BIAS,BM,BN,BK,NUM_SMS,NUM_SMS - NUM_RS_SMS,0,NUM_THREADS,WARP_M,WARP_N,GROUP_SIZE_N,SPLIT_K>; \
    registry.fused_full_map[ShapeKey{M,N,K}.key] = kernel_launcher<M,N,K,LOAD_BIAS,BM,BN,BK,NUM_SMS,NUM_SMS - NUM_RS_SMS,NUM_RS_SMS,NUM_THREADS,WARP_M,WARP_N,GROUP_SIZE_N,SPLIT_K>;

    if (__builtin_expect(registry.gemm_map.empty(), false)) {
        // // Online Benchmark parameters here
        REGISTER_KERNELS(64  , 7168, 2304, WB, 32 , 64, 128, 2, 2, 8 , 48, 1 );  // 90.92 TFLOPS
        REGISTER_KERNELS(512 , 4096, 1536, WB, 128, 64, 128, 2, 2, 8 , 48, 1 );  // 194.13 TFLOPS
        REGISTER_KERNELS(2048, 2880, 360 , WB, 128, 256, 64, 2, 2, 8 , 48, 1 );  // 150.06 TFLOPS
        REGISTER_KERNELS(4096, 4096, 512 , WB, 256, 128, 64, 2, 2, 16, 48, 1 );  // 240.03 TFLOPS
        REGISTER_KERNELS(8192, 4096, 1792, WB, 224, 256, 64, 2, 2, 16, 32, 1 );  // 491.16 TFLOPS
        REGISTER_KERNELS(8192, 8192, 3696, WB, 224, 256, 64, 2, 2, 0 , 8, 1 );  // 620.40 TFLOPS
        // // New shapes
        REGISTER_KERNELS(64,   2880, 360,  WB, 32,  64,  64, 2, 2, 8, 16, 1);
        REGISTER_KERNELS(64,   3584, 1792, WB, 32,  64,  64, 2, 2, 8, 16, 1);
        REGISTER_KERNELS(512,  3584, 1792, WB, 64,  128, 64, 2, 2, 8, 16, 1);
        REGISTER_KERNELS(512,  4608, 4608, WB, 64,  128, 64, 2, 2, 8, 16, 1);
        REGISTER_KERNELS(2048, 4096, 896,  WB, 128, 128, 64, 2, 2, 8, 16, 1);
        REGISTER_KERNELS(2048, 8192, 3840, WB, 128, 256, 64, 2, 2, 8, 8,  1);
        REGISTER_KERNELS(4096, 2880, 360,  WB, 256, 128, 64, 2, 2, 8, 16, 1);
        REGISTER_KERNELS(4096, 8192, 256,  WB, 224, 256, 64, 2, 2, 8, 8,  1);
        REGISTER_KERNELS(8192, 3584, 1792, WB, 224, 256, 64, 2, 2, 8, 8,  1);
        REGISTER_KERNELS(8192, 4608, 4608, WB, 224, 256, 64, 2, 2, 8, 8,  1);
        REGISTER_KERNELS(8192, 8192, 3584, WB, 224, 256, 64, 2, 2, 8, 8,  1);


    }
#endif



    return registry;
}



torch::Tensor launch_gemm(torch::Tensor &x, torch::Tensor &w, torch::Tensor &b, torch::Tensor &signal, int signal_val, std::optional<torch::Tensor> out_opt) {
    auto M = x.size(0);
    auto N = w.size(0);
    auto K = x.size(1);
    auto out = out_opt ? *out_opt : torch::empty({M, N}, x.options());
    auto &registry = get_kernel_registry();
    auto kernel_it = registry.gemm_map.find(ShapeKey{static_cast<uint16_t>(M),static_cast<uint16_t>(N),static_cast<uint16_t>(K)}.key);
    TORCH_CHECK(kernel_it != registry.gemm_map.end(), "Unsupported GEMM size: ", M, "x", N, "x", K);
    int device = 0;
    C10_HIP_CHECK(hipGetDevice(&device));
    int rank = device;

    std::array<bfloat16_t*, WORLD_SIZE> c_ptrs{};
    c_ptrs[rank] = reinterpret_cast<bfloat16_t *>(out.data_ptr());
    std::array<int*, WORLD_SIZE> signal_ptrs{};
    signal_ptrs[rank] = reinterpret_cast<int *>(signal.data_ptr());

    kernel_it->second(
        reinterpret_cast<const bfloat16_t *>(x.const_data_ptr()),
        reinterpret_cast<const bfloat16_t *>(w.const_data_ptr()),
        reinterpret_cast<const bfloat16_t *>(b.const_data_ptr()),
        c_ptrs, signal_ptrs, signal_val,
        reinterpret_cast<bfloat16_t *>(out.data_ptr()),
        static_cast<float*>(registry.workspace_ptr),
        rank);
    return out;
}

torch::Tensor launch_reduce_scatter(std::array<torch::Tensor, WORLD_SIZE> &c, std::array<torch::Tensor, WORLD_SIZE> &signal, int signal_val, int rank) {
    auto M = c[0].size(0);
    auto N = c[0].size(1);
    auto out = torch::empty({M / WORLD_SIZE, N}, c[0].options());
    auto &registry = get_kernel_registry();
    auto key = ShapeKey{static_cast<uint16_t>(M),static_cast<uint16_t>(N),0}.key;
    auto it = registry.rs_map.find(key);
    TORCH_CHECK(it != registry.rs_map.end(), "Unsupported ReduceScatter size: ", M, "x", N);

    std::array<bfloat16_t*, WORLD_SIZE> c_ptrs{};
    std::array<int*, WORLD_SIZE> signal_ptrs{};
    for (int i = 0; i < WORLD_SIZE; i++) {
        c_ptrs[i] = reinterpret_cast<bfloat16_t *>(c[i].data_ptr());
        signal_ptrs[i] = reinterpret_cast<int *>(signal[i].data_ptr());
    }

    it->second(/*x*/static_cast<const bfloat16_t*>(nullptr),
               /*w*/static_cast<const bfloat16_t*>(nullptr),
               /*b*/static_cast<const bfloat16_t*>(nullptr),
               c_ptrs, signal_ptrs, signal_val,
               reinterpret_cast<bfloat16_t *>(out.data_ptr()),
               static_cast<float*>(registry.workspace_ptr),
               rank);
    return out;
}

torch::Tensor launch_fused(
    torch::Tensor &x,
    torch::Tensor &w,
    torch::Tensor &b,
    std::array<torch::Tensor, WORLD_SIZE> &c,       // unified C buffers for all ranks
    std::array<torch::Tensor, WORLD_SIZE> &signal,  // unified signal buffers for all ranks
    int signal_val,
    int rank
) {
    dim3 grid(NUM_SMS);
    dim3 block(NUM_THREADS);

    auto M = x.size(0);
    auto N = w.size(0);
    auto K = x.size(1);
    auto rs_out = torch::empty({M / WORLD_SIZE, N}, c.front().options());
    auto &registry = get_kernel_registry();
    auto key = ShapeKey{static_cast<uint16_t>(M),static_cast<uint16_t>(N),static_cast<uint16_t>(K)}.key;
    auto it = registry.fused_full_map.find(key);
    std::array<bfloat16_t*, WORLD_SIZE> c_ptrs{};
    std::array<int*, WORLD_SIZE> signal_ptrs{};
    for (int i = 0; i < WORLD_SIZE; i++) {
        c_ptrs[i] = reinterpret_cast<bfloat16_t *>(c[i].data_ptr());
        signal_ptrs[i] = reinterpret_cast<int *>(signal[i].data_ptr());
    }
    it->second(
        reinterpret_cast<const bfloat16_t *>(x.const_data_ptr()),
        reinterpret_cast<const bfloat16_t *>(w.const_data_ptr()),
        reinterpret_cast<const bfloat16_t *>(b.const_data_ptr()),
        c_ptrs, signal_ptrs, signal_val,
        reinterpret_cast<bfloat16_t *>(rs_out.data_ptr()),
        static_cast<float*>(registry.workspace_ptr),
        rank);
    return rs_out;
}


class GemmRS {
private:
    int rank_;
    int world_size_;
    void *ipc_mems_[WORLD_SIZE];
    void *sig_buf_[WORLD_SIZE];
    
    void check_device() {
        int device;
        C10_HIP_CHECK(hipGetDevice(&device));
        TORCH_CHECK(device == rank_);
    }

public:
    GemmRS(int rank, int world_size): rank_(rank), world_size_(world_size) {
        C10_HIP_CHECK(hipExtMallocWithFlags(&ipc_mems_[rank_], MAX_IPC_MEM_SIZE, hipDeviceMallocUncached));
        // C10_HIP_CHECK(hipMalloc(&ipc_mems_[rank_], MAX_IPC_MEM_SIZE));
        C10_HIP_CHECK(hipMemset(ipc_mems_[rank_], 0, MAX_IPC_MEM_SIZE));
        sig_buf_[rank_] = reinterpret_cast<std::byte*>(ipc_mems_[rank_]) + MAX_IPC_MEM_SIZE - SIGNAL_BUF_SIZE;
        TORCH_CHECK(world_size_ == WORLD_SIZE, "Only support world_size = ", WORLD_SIZE);
    }
    ~GemmRS() {}

    pybind11::bytearray get_ipc_handle() {
        check_device();
        hipIpcMemHandle_t ipc_handle;
        C10_HIP_CHECK(hipIpcGetMemHandle(&ipc_handle, ipc_mems_[rank_]));
        return {ipc_handle.reserved, HIP_IPC_HANDLE_SIZE};
    }

    auto init_dist(const std::vector<pybind11::bytearray> &ipc_handles) {
        check_device();
        int world_size = ipc_handles.size();
        TORCH_CHECK(world_size == WORLD_SIZE, "Mismatched world size");
        for (int i = 0; i < world_size; i++) {
            if (i == rank_) continue;
            hipIpcMemHandle_t handle;
            auto handle_buf = std::string(ipc_handles[i]);
            TORCH_CHECK(handle_buf.size() == HIP_IPC_HANDLE_SIZE);
            std::memcpy(handle.reserved, handle_buf.data(), HIP_IPC_HANDLE_SIZE);
            C10_HIP_CHECK(hipIpcOpenMemHandle(&ipc_mems_[i], handle, hipIpcMemLazyEnablePeerAccess));
            sig_buf_[i] = reinterpret_cast<std::byte*>(ipc_mems_[i]) + MAX_IPC_MEM_SIZE - SIGNAL_BUF_SIZE; // last for signal
        }
    }

    std::array<torch::Tensor, WORLD_SIZE> get_c_tensors(int M, int N) {
        check_device();
        TORCH_CHECK(M % world_size_ == 0, "M must be divisible by world_size");

        std::array<torch::Tensor, WORLD_SIZE> c;
        for (int i = 0; i < world_size_; i++) {
            TORCH_CHECK(ipc_mems_[i] != nullptr, "IPC memory not initialized");
            c[i] = torch::from_blob(ipc_mems_[i], {M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, rank_));
        }
        return c;
    }

    std::array<torch::Tensor, WORLD_SIZE> get_signal_tensors() {
        check_device();
        std::array<torch::Tensor, WORLD_SIZE> signal;
        for (int i = 0; i < world_size_; i++) {
            TORCH_CHECK(sig_buf_[i] != nullptr, "Signal buffer not initialized");
            signal[i] = torch::from_blob(sig_buf_[i], {1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, rank_));
        }
        return signal;
    }

};


PYBIND11_MODULE(perf_gemm, m) {
    m.def("launch_gemm", &launch_gemm, "Launch GEMM kernel", 
        pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("b"), pybind11::arg("signal"), pybind11::arg("signal_val"), pybind11::arg("out") = std::nullopt);
    m.def("launch_reduce_scatter", &launch_reduce_scatter, "Launch ReduceScatter kernel",
        pybind11::arg("c"), pybind11::arg("signal"), pybind11::arg("signal_val"), pybind11::arg("rank"));
    m.def("launch_fused", &launch_fused, "Launch fused GEMM+RS kernel",
        pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("b"),
        pybind11::arg("c"), pybind11::arg("signal"),
        pybind11::arg("signal_val"), pybind11::arg("rank"));
    m.def("__debug_get_workspace_tensor", [](int M, int N, int split_k){
        auto &registry = get_kernel_registry();
        TORCH_CHECK(registry.workspace_ptr != nullptr, "Workspace not initialized");
        size_t required_size = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float) * split_k;
        TORCH_CHECK(required_size <= registry.workspace_size, "Requested workspace size ", required_size, " exceeds preallocated size ", registry.workspace_size);
        return torch::from_blob(registry.workspace_ptr, {static_cast<long>(split_k), M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    });
    pybind11::class_<GemmRS>(m, "GemmRS")
        .def(pybind11::init<int, int>(), pybind11::arg("rank"), pybind11::arg("world_size"))
        .def("get_ipc_handle", &GemmRS::get_ipc_handle, "Get IPC handle for the current rank")
        .def("init_dist", &GemmRS::init_dist, "Initialize distributed GemmRS with IPC handles")
        .def("get_c_tensors", &GemmRS::get_c_tensors, "Get tensors for C matrices")
        .def("get_signal_tensors", &GemmRS::get_signal_tensors, "Get tensors for signal buffers");
}

#endif // __PERF_GEMM_HEADER__
