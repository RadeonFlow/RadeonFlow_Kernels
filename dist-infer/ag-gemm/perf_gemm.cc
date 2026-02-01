#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <ck/utility/data_type.hpp>
#include <ck/utility/amd_buffer_addressing.hpp>
#include <ck/utility/ignore.hpp>
#define FAST_UNSAFE_CAST

#define FORCE_INLINE __attribute__((always_inline))

#define LOCAL_TEST


// Perf GEMM Common Begin


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

// Perf GEMM Common End

#define DO_PRAGMA_(x) _Pragma(#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)
#define UNROLL DO_PRAGMA(unroll)
#define UNROLL_N(n) DO_PRAGMA(unroll n)

using b16 = bfloat16_t;

constexpr int WARP_SIZE = 64;

template <typename T> constexpr T const_min(T a, T b) { return a > b ? b : a; }

template <int TILE_M, int TILE_K, int K, int NUM_DST>
__device__ inline void warp_copy_tile(b16 *(&dst)[NUM_DST], const b16 *src) {
    static_assert(TILE_K % WARP_SIZE == 0);
    constexpr int VEC_SIZE = TILE_K / WARP_SIZE * sizeof(b16);
    static_assert(VEC_SIZE == 16 || VEC_SIZE == 8 || VEC_SIZE == 4 || VEC_SIZE == 2);
    using cp_t = typename ck::vector_type<int8_t, VEC_SIZE>::type;
    constexpr int NUM_STAGES = const_min(8, exact_div<TILE_M, 2>());
    cp_t regs[NUM_STAGES];

    const auto src_rsrc = ck::make_wave_buffer_resource_with_default_range(src);
    ck::int32x4_t dst_rsrc[NUM_DST];
    UNROLL
    for (int i = 0; i < NUM_DST; i++) {
        dst_rsrc[i] = ck::make_wave_buffer_resource_with_default_range(dst[i]);
    }
    const auto lane_id = threadIdx.x % WARP_SIZE;

    auto load_row = [&](int reg_idx, int row_idx) {
        const int soffset = row_idx * K * sizeof(b16);
        const int voffset = lane_id * VEC_SIZE;
        regs[reg_idx] = ck::amd_buffer_load_impl_raw<
            VEC_SIZE, ck::AmdBufferCoherenceEnum::SYSTEM_NT0>(
            src_rsrc, voffset, soffset
        );
    };
    auto store_row = [&](int reg_idx, int row_idx) {
        UNROLL
        for (int i = 0; i < NUM_DST; i++) {
            const int soffset = row_idx * K * sizeof(b16);
            const int voffset = lane_id * VEC_SIZE;
            ck::amd_buffer_store_impl_raw<
                VEC_SIZE, ck::AmdBufferCoherenceEnum::SYSTEM_NT0>(
                regs[reg_idx], dst_rsrc[i], voffset, soffset
            );
        }
    };

    /*
    ld 0
    ld 1
    ld 2
    ld 3
    st 0
    ld 4
    st 1
    ld 5
    st 2

    ld 6
    st 3
    ld 7
    st 4
    ld 8
    st 5
    ld 9
    st 6

    ld 10
    st 7
    ld 11
    st 8
    st 9
    st 10
    st 11
    */
    static_assert(TILE_M % (NUM_STAGES * 2) == 0);
    auto copy_prologue = [&](int row_idx=0) {
        UNROLL
        for (int i = 0; i < NUM_STAGES - 1; i++) {
            load_row(i % NUM_STAGES, row_idx + i);
        }
        UNROLL
        for (int i = 0; i < NUM_STAGES - 1; i++) {
            load_row((i + NUM_STAGES - 1) % NUM_STAGES, row_idx + i + NUM_STAGES - 1);
            store_row(i % NUM_STAGES, row_idx + i);
        }
    };
    // row_idx: 0, TILE_M - (NUM_STAGES * 2), NUM_STAGES
    auto copy_loop_body = [&](int row_idx) {
        UNROLL
        for (int i = 0; i < NUM_STAGES; i++) {
            auto load_off = (NUM_STAGES - 1) * 2;
            auto store_off = (NUM_STAGES - 1);
            load_row((i + load_off) % NUM_STAGES, row_idx + i + load_off);
            store_row((i + store_off) % NUM_STAGES, row_idx + i + store_off);
        }
    };
    // row_idx: TILE_M - 2
    auto copy_epilogue = [&](int row_idx=TILE_M-2) {
        UNROLL
        for (int i = 0; i < 2; i++) {
            auto load_off = row_idx;
            auto store_off = row_idx - (NUM_STAGES - 1);
            load_row((i + load_off) % NUM_STAGES, i + load_off);
            store_row((i + store_off) % NUM_STAGES, i + store_off);
        }
        UNROLL
        for (int i = 0; i < NUM_STAGES - 1; i++) {
            auto store_off = row_idx - (NUM_STAGES - 1) + 2;
            store_row((i + store_off) % NUM_STAGES, i + store_off);
        }
    };

    copy_prologue();
    for(int i = 0; i < TILE_M - (NUM_STAGES * 2); i += NUM_STAGES) {
        copy_loop_body(i);
    }
    copy_epilogue();

    // auto copy_loop_body = [&](int row_idx) {
    //     load_row(0, row_idx + 2);
    //     // sync, 1 ld, 8 st in flight
    //     store_row(1, row_idx + 1);
    //     load_row(1, row_idx + 3);
    //     // sync, 1 ld, 8 st in flight
    //     store_row(0, row_idx + 2);
    // };

    // constexpr int UNROLL_FACTOR = const_min(TILE_M / NUM_STAGES, 8);
    // constexpr int INNER_M = UNROLL_FACTOR * NUM_STAGES;
    // static_assert(TILE_M % NUM_STAGES == 0);
    // static_assert(TILE_M >= INNER_M);

    // load_row(0, 0);
    // load_row(1, 1);
    // store_row(0, 0);
    // for (int i = 0; i < TILE_M - INNER_M; i += INNER_M) {
    //     asm(";main loop begin");
    //     UNROLL
    //     for (int j = 0; j < INNER_M; j += NUM_STAGES) {
    //         copy_loop_body(i + j);
    //     }
    //     asm(";main loop end");
    // }
    // UNROLL
    // for (int i = TILE_M - INNER_M; i < TILE_M - NUM_STAGES; i += NUM_STAGES) {
    //     copy_loop_body(i);
    // }
    // store_row(1, TILE_M - 1);
}

using signal_t = int[128][128];
#ifdef LOCAL_TEST
constexpr int WORLD_SIZE = 2;
#else
constexpr int WORLD_SIZE = 8;
#endif
constexpr int NUM_SMS = 304;
constexpr int AG_SMS = 16;
constexpr int GEMM_SMS = NUM_SMS - AG_SMS;

constexpr int CHUNK_K = 128;
constexpr int CHUNK_M = 64;

struct workspace_t {
    int grid_barrier;
};

constexpr int MAX_M = 8192;
constexpr int MAX_N = 29568;
constexpr int MAX_K = 8192;
constexpr int MAX_M_LOCAL = MAX_M / WORLD_SIZE;

// global variables
struct ipc_mem_t {
    // FIXME: reset signals
    signal_t nvl_recv_signals;
    signal_t nvl_consume_signals;
    int nvl_barrier[WORLD_SIZE];
};

struct ipc_cache_t {
    b16 nvl_recv_x[MAX_M * MAX_K];
};

struct global_t {
    // config
    int rank;
    int m, n, k;
    int next_signal;

    // buffers
    ipc_mem_t *ipc_mems[WORLD_SIZE] = {};
    ipc_cache_t *ipc_caches[WORLD_SIZE] = {};
    workspace_t *workspace;
};

template <typename T> __device__ inline void st_relaxed_sys(T *ptr, T val) {
    __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T> __device__ inline T ld_relaxed_sys(T *ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T> __device__ inline void st_release_global(T *ptr, T val) {
    __hip_atomic_store(ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T> __device__ inline T ld_acquire_global(T *ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

template <int M, int K, int NUM_AG_SMS, int NUM_AG_WARPS, int CHUNK_M, int CHUNK_K>
__device__ inline void send_kernel(const int sm_id, const b16 *x, global_t global) {
    const auto num_sms = NUM_AG_SMS;
    const auto num_warps = blockDim.x / WARP_SIZE;
    const int num_global_warps = num_sms * num_warps;

    // put soffset to sgpr
    const auto warp_id =
        __builtin_amdgcn_readfirstlane(threadIdx.x / WARP_SIZE);
    const auto global_warp_id = sm_id * num_warps + warp_id;
    const auto lane_id = threadIdx.x % WARP_SIZE;

    if (global_warp_id >= NUM_AG_WARPS) {
        return;
    }

    const auto rank = global.rank;

    static_assert(M % WORLD_SIZE == 0);
    constexpr auto M_LOCAL = M / WORLD_SIZE;

    constexpr auto NUM_CHUNKS_M = ceil_div(M_LOCAL, CHUNK_M);
    constexpr auto NUM_CHUNKS_K = ceil_div(K, CHUNK_K);
    constexpr auto TAIL_CHUNK_M = M_LOCAL - (NUM_CHUNKS_M - 1) * CHUNK_M;
    constexpr auto TAIL_CHUNK_K = K - (NUM_CHUNKS_K - 1) * CHUNK_K;

    for (int i = global_warp_id; i < NUM_CHUNKS_M * NUM_CHUNKS_K * WORLD_SIZE;
         i += NUM_AG_WARPS) {
        const auto dst_rank = i % WORLD_SIZE;
        const auto chunk_id = i / WORLD_SIZE;
        // TODO: maybe k first
        const auto chunk_k = chunk_id / NUM_CHUNKS_M;
        const auto chunk_m = chunk_id % NUM_CHUNKS_M;

        const auto m_begin = chunk_m * CHUNK_M;
        const auto k_begin = chunk_k * CHUNK_K;
        const auto offset = m_begin * K + k_begin;

        const auto chunk_src = x + offset;
        b16 *chunk_dst[1] = {
            global.ipc_caches[dst_rank]->nvl_recv_x + rank * M_LOCAL * K +
            offset
        };

        if (TAIL_CHUNK_M != CHUNK_M && chunk_m == NUM_CHUNKS_M - 1) {
            if (TAIL_CHUNK_K != CHUNK_K && chunk_k == NUM_CHUNKS_K - 1) {
                warp_copy_tile<TAIL_CHUNK_M, TAIL_CHUNK_K, K>(
                    chunk_dst, chunk_src
                );
            } else {
                warp_copy_tile<TAIL_CHUNK_M, CHUNK_K, K>(chunk_dst, chunk_src);
            }
        } else {
            if (TAIL_CHUNK_K != CHUNK_K && chunk_k == NUM_CHUNKS_K - 1) {
                warp_copy_tile<CHUNK_M, TAIL_CHUNK_K, K>(chunk_dst, chunk_src);
            } else {
                warp_copy_tile<CHUNK_M, CHUNK_K, K>(chunk_dst, chunk_src);
            }
        }

        if (lane_id == 0) {
            st_relaxed_sys(
                &global.ipc_mems[dst_rank]
                        ->nvl_recv_signals[chunk_k][rank * NUM_CHUNKS_M + chunk_m],
                global.next_signal
            );

            // if (lane_id == 0) {
            //     printf("m %d, k %d\n", chunk_m, chunk_k);
            // }

            constexpr int NUM_PREFETCH = 1;
            if (chunk_k >= NUM_PREFETCH) {
                auto &consume = global.ipc_mems[dst_rank]
                        ->nvl_recv_signals[chunk_k - NUM_PREFETCH][rank * NUM_CHUNKS_M + chunk_m];
                while (global.next_signal != ld_relaxed_sys(&consume))
                    __builtin_amdgcn_s_sleep(5);
            }
        }
    }
}

template<int M, int N, int K, int BM, int BN, int BK, int NUM_SMS, int NUM_THREADS, int WARP_M, int WARP_N, int SPLIT_K, int GROUP_SIZE_M, int CHUNK_M, int CHUNK_K>
__launch_bounds__(NUM_THREADS)
__global__ void gemm_kernel(
    const bfloat16_t *x, // M x K
    const bfloat16_t *w, // N x K
    const bfloat16_t *b, // N
    bfloat16_t *c,       // M x N
    global_t global
) {
    constexpr int NUM_CONCURRENT_CHUNK_K = 1;
    constexpr int NUM_SEND_WARPS = ceil_div(M / WORLD_SIZE, CHUNK_M) * WORLD_SIZE * NUM_CONCURRENT_CHUNK_K;
    constexpr int NUM_AG_SMS =  ceil_div(NUM_SEND_WARPS, NUM_THREADS / WARP_SIZE);
    constexpr int NUM_GEMM_SMS = NUM_SMS - NUM_AG_SMS;
    if (blockIdx.x >= NUM_GEMM_SMS) {
        // allgather
        const int sm_id = blockIdx.x - NUM_GEMM_SMS;
        send_kernel<M, K, NUM_AG_SMS, NUM_SEND_WARPS, CHUNK_M, CHUNK_K>(sm_id, x, global);
        return;
    }
    x = global.ipc_caches[global.rank]->nvl_recv_x;
    auto signal = &global.ipc_mems[global.rank]->nvl_recv_signals;

    const int pid = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const int tid = threadIdx.x;
    const int lane_id = __lane_id();
    const int warp_id = __builtin_amdgcn_readfirstlane(tid / roc_isa::AMDGCN_WAVEFRONT_SIZE);
    __builtin_assume(pid >= 0 && pid < NUM_SMS);
    __builtin_assume(tid >= 0 && tid < NUM_THREADS);
    __builtin_assume(lane_id >= 0 && lane_id < 64);

    // gemm

    constexpr int COMM_K = CHUNK_K;
    auto wait_signal = [&](int tile_m_id, int tile_k_id) {
        constexpr int M_LOCAL = exact_div<M, WORLD_SIZE>();
        constexpr int COMM_M = const_min(M_LOCAL, CHUNK_M);

        auto signal_k_id = tile_k_id / exact_div<COMM_K, BK>();

        static_assert(M >= BM && M % BM == 0);
        int signal_m_id_begin, signal_m_id_end;
        if constexpr (COMM_M < BM) {
            signal_m_id_begin = tile_m_id * exact_div<BM, COMM_M>();
            signal_m_id_end = signal_m_id_begin + exact_div<BM, COMM_M>();
        } else {
            signal_m_id_begin = tile_m_id / exact_div<COMM_M, BM>();
            signal_m_id_end = signal_m_id_begin + 1;
        }

        auto &sig = *signal;
        
        if(warp_id == 0 && lane_id < (signal_m_id_end - signal_m_id_begin)) {
            // printf("wait m %d k %d\n", signal_m_id, signal_k_id);
            while(global.next_signal != ld_relaxed_sys(&sig[signal_k_id][signal_m_id_begin + lane_id]))
                __builtin_amdgcn_s_sleep(5);
            if (tile_k_id % exact_div<COMM_K, BK>() == 0) {
                auto &consume = global.ipc_mems[global.rank]->nvl_consume_signals[signal_k_id][signal_m_id_begin + lane_id];
                st_relaxed_sys(&consume, global.next_signal);
            }
            // __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
        }
        __syncthreads();
    };

    // Perf GEMM
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
    auto c_arr = ck::make_wave_buffer_resource<bfloat16_t>(c, M * N);
    auto x_arr = ck::make_wave_buffer_resource<bfloat16_t>(const_cast<bfloat16_t*>(x), M * K);
    auto w_arr = ck::make_wave_buffer_resource<bfloat16_t>(const_cast<bfloat16_t*>(w), N * K);

    constexpr bool LOAD_BIAS = true;
    constexpr int NUM_XCDS = 8;
    for (int tile_id=pid; tile_id<num_tiles; tile_id+=NUM_GEMM_SMS) {
        int tile_id1 = (tile_id % NUM_XCDS) * (NUM_GEMM_SMS / NUM_XCDS) + (tile_id / NUM_XCDS);
        // int tile_id1 = tile_id;
        int split_k_id = tile_id1 % SPLIT_K;
        int tile_n_id = (tile_id1 / SPLIT_K) % num_tile_n;
        int tile_m_id = (tile_id1 / SPLIT_K) / num_tile_n;
        compute_tile_indices_m_grouped<num_tile_m, num_tile_n, GROUP_SIZE_M>(tile_id / SPLIT_K, tile_m_id, tile_n_id);
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



        auto load_frag = [&](int tile_kk, const bfloat16_t *ptr, bf16x4_t &frag, bool permute) {
            // ptr: [16][16]
            const int row0 = lane_id % 16;
            const int col0 = tile_kk * 16 + lane_id / 16 * 4;
            const int col1 = (row0 * 4 + col0) % BK;
            frag = *reinterpret_cast<const bf16x4_t*>(&ptr[row0 * BK + col1]);
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
                // TODO: workspace + split_k_id * M * N
                auto ws_arr = ck::make_wave_buffer_resource<float>(nullptr, M * N);
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



        wait_signal(tile_m_id, 0);
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
            if ((tile_k_id + 1) % exact_div<COMM_K, BK>() == 0) {
                wait_signal(tile_m_id, tile_k_id + 1);
            }
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
        // __syncthreads();
        // if (tid == 0) {
        //     __hip_atomic_store(&signal_arr[tile_m_id][tile_n_id], signal_val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        // }
    }
    // block_sync_gds();
    // release_signal();
}

constexpr int64_t pack_mnk(int m, int n, int k) {
    return (int64_t(m) << 32) | (int64_t(n) << 16) | int64_t(k);
}

// TODO: fix 2880 ck=512
// clang-format off
#ifdef LOCAL_TEST
#define SWITCH_GEMM_MNK(m, n, k, MACRO, ...) \
    switch (pack_mnk(m, n, k)) { \
        case pack_mnk(64  , 2304, 7168): MACRO(64  , 2304, 7168, 32,  32,  128, 2, 2, 1, 8, 64, 512, ##__VA_ARGS__); break; \
        case pack_mnk(128 , 1536, 4096): MACRO(128 , 1536, 4096, 64,  64,  128, 2, 2, 1, 32, 64, 512, ##__VA_ARGS__); break; \
        case pack_mnk(512 , 360 , 2880): MACRO(512 , 360 , 2880, 64,  64,  128, 2, 2, 1, 8,  64, 128, ##__VA_ARGS__); break; \
        case pack_mnk(1024, 512 , 4096): MACRO(1024, 512 , 4096, 128, 64,  128, 2, 2, 1, 40, 64, 512, ##__VA_ARGS__); break; \
        case pack_mnk(2048, 1792, 4096): MACRO(2048, 1792, 4096, 256, 224, 64,  2, 2, 1, 32, 64, 128, ##__VA_ARGS__); break; \
        case pack_mnk(2048, 3696, 8192): MACRO(2048, 3696, 8192, 256, 224, 64,  2, 2, 1, 32, 64, 128, ##__VA_ARGS__); break; \
        default: fprintf(stderr, "invalid mnk: %d %d %d", m, n, k); \
    }
#else
// bm, bn, bk, wm, wn
#define SWITCH_GEMM_MNK(m, n, k, MACRO, ...) \
    switch (pack_mnk(m, n, k)) { \
        case pack_mnk(64, 2304, 7168):   MACRO(64, 2304,  7168,  32,  32,  128, 2, 2, 1, 8,  64, 512, ##__VA_ARGS__); break; \
        case pack_mnk(512, 1536, 4096):  MACRO(512, 1536,  4096, 64,  64,  128, 2, 2, 1, 32, 64, 512, ##__VA_ARGS__); break; \
        case pack_mnk(2048, 360, 2880):  MACRO(2048, 360,  2880, 64,  64,  128, 2, 2, 1, 8,  64, 128, ##__VA_ARGS__); break; \
        case pack_mnk(4096, 512, 4096):  MACRO(4096, 512,  4096, 128, 64,  128, 2, 2, 1, 40, 64, 512, ##__VA_ARGS__); break; \
        case pack_mnk(8192, 1792, 4096): MACRO(8192, 1792, 4096, 256, 224, 64,  2, 2, 1, 32, 64, 128, ##__VA_ARGS__); break; \
        case pack_mnk(8192, 3696, 8192): MACRO(8192, 3696, 8192, 256, 224, 64,  2, 2, 1, 32, 64, 128, ##__VA_ARGS__); break; \
        default: fprintf(stderr, "invalid mnk"); \
    }
#endif
// clang-format on

class AgGemm {
  private:
    global_t global{};

  public:
    AgGemm(int rank, int m, int n, int k) {
        global.rank = rank;
        global.m = m;
        global.n = n;
        global.k = k;
        global.next_signal = 0;
    }

    ~AgGemm() {
        for (auto i = 0; i < WORLD_SIZE; i++) {
            auto ipc_mem = global.ipc_mems[i];
            if (ipc_mem && i != global.rank) {
                C10_HIP_CHECK(hipIpcCloseMemHandle(ipc_mem));
            }
            auto ipc_cache = global.ipc_caches[i];
            if (ipc_cache && i != global.rank) {
                C10_HIP_CHECK(hipIpcCloseMemHandle(ipc_cache));
            }
        }
        auto local_mem = global.ipc_mems[global.rank];
        if (local_mem) {
            C10_HIP_CHECK(hipFree(local_mem));
        }
        auto local_cache = global.ipc_caches[global.rank];
        if (local_cache) {
            C10_HIP_CHECK(hipFree(local_cache));
        }
        if (global.workspace) {
            C10_HIP_CHECK(hipFree(global.workspace));
        }
    }

    auto get_ipc_handle() -> pybind11::bytearray {
        void *ws;
        C10_HIP_CHECK(hipMalloc(&ws, sizeof(workspace_t)));
        C10_HIP_CHECK(hipMemset(ws, 0, sizeof(workspace_t)));
        global.workspace = reinterpret_cast<workspace_t *>(ws);

        void *ptr;
        C10_HIP_CHECK(hipExtMallocWithFlags(
            &ptr, sizeof(ipc_mem_t), hipDeviceMallocUncached
        ));
        C10_HIP_CHECK(hipMemset(ptr, 0, sizeof(ipc_mem_t)));
        global.ipc_mems[global.rank] = reinterpret_cast<ipc_mem_t *>(ptr);

        // C10_HIP_CHECK(hipMalloc(&ptr, sizeof(ipc_cache_t)));
        C10_HIP_CHECK(hipExtMallocWithFlags(
            &ptr, sizeof(ipc_cache_t), hipDeviceMallocUncached
        ));
        C10_HIP_CHECK(hipMemset(ptr, 0, sizeof(ipc_cache_t)));
        global.ipc_caches[global.rank] = reinterpret_cast<ipc_cache_t *>(ptr);

        std::vector<hipIpcMemHandle_t> handles(2);
        C10_HIP_CHECK(hipIpcGetMemHandle(&handles[0], global.ipc_mems[global.rank])
        );
        C10_HIP_CHECK(
            hipIpcGetMemHandle(&handles[1], global.ipc_caches[global.rank])
        );
        return {
            reinterpret_cast<char *>(handles.data()), HIP_IPC_HANDLE_SIZE * 2
        };
    }

    auto init(const std::vector<pybind11::bytearray> &ipc_handles) {
        for (int i = 0; i < WORLD_SIZE; i++) {
            if (i == global.rank) {
                continue;
            }
            hipIpcMemHandle_t handle;
            auto handle_buf = std::string(ipc_handles[i]);
            auto handles =
                reinterpret_cast<hipIpcMemHandle_t *>(handle_buf.data());
            void *ptr;
            C10_HIP_CHECK(hipIpcOpenMemHandle(
                &ptr, handles[0], hipIpcMemLazyEnablePeerAccess
            ));
            global.ipc_mems[i] = reinterpret_cast<ipc_mem_t *>(ptr);
            C10_HIP_CHECK(hipIpcOpenMemHandle(
                &ptr, handles[1], hipIpcMemLazyEnablePeerAccess
            ));
            global.ipc_caches[i] = reinterpret_cast<ipc_cache_t *>(ptr);
        }
    }

    auto get_x_full() {
        auto x_full = torch::from_blob(
            global.ipc_caches[global.rank]->nvl_recv_x, {global.m, global.k},
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA)
        );
        return x_full;
    }

    auto get_signal() {
        auto signal = torch::from_blob(
            global.ipc_mems[global.rank]->nvl_recv_signals, {128, 128},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)
        );
        return signal;
    }

    void reset() {
        auto stream = at::cuda::getCurrentHIPStream().stream();
        auto &signal = global.ipc_mems[global.rank]->nvl_recv_signals;
        C10_HIP_CHECK(hipMemsetAsync(&signal, 0, sizeof(signal), stream));
    }

    auto
    perf_gemm(torch::Tensor &x, torch::Tensor &w, torch::Tensor &b) {
        auto m_local = x.size(0);
        const auto m = m_local * WORLD_SIZE;
        auto n = w.size(0);
        auto k = w.size(1);
        auto out = torch::empty({m, n}, x.options());

        auto x_ptr = reinterpret_cast<const bfloat16_t *>(x.const_data_ptr());
        auto w_ptr = reinterpret_cast<const bfloat16_t *>(w.const_data_ptr());
        auto b_ptr = reinterpret_cast<const bfloat16_t *>(b.const_data_ptr());
        auto o_ptr = reinterpret_cast<bfloat16_t *>(out.data_ptr());

        // udpate signal
        global.next_signal++;

        constexpr int NUM_THREADS = 256;

        dim3 grid(NUM_SMS);
        dim3 block(NUM_THREADS);

        auto stream = at::cuda::getCurrentHIPStream().stream();

        // clang-format off
        #define LAUNCH_PERF(m, n, k, bm, bn, bk, wm, wn, splitk, group_m, cm, ck) gemm_kernel<m, n, k, bm, bn, bk, NUM_SMS, NUM_THREADS, wm, wn, splitk, group_m, cm, ck><<<grid, block, 0, stream>>>(x_ptr, w_ptr, b_ptr, o_ptr, global)
        SWITCH_GEMM_MNK(m, n, k, LAUNCH_PERF)
        // clang-format on
        return out;
    }
};

PYBIND11_MODULE(ag_gemm, m) {
    py::class_<AgGemm>(m, "AgGemm")
        .def(py::init<int, int, int, int>())
        .def("get_ipc_handle", &AgGemm::get_ipc_handle)
        .def("init", &AgGemm::init)
        .def("get_x_full", &AgGemm::get_x_full)
        .def("reset", &AgGemm::reset)
        .def(
            "perf_gemm", &AgGemm::perf_gemm, py::arg("x"), py::arg("w"),
            py::arg("b")
        )
        .def("get_signal", &AgGemm::get_signal);
}