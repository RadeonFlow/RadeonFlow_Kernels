#include "gemm_rs_kernel.h"
#include <unordered_map>
#include <functional>
#include <hip/hip_runtime.h>
#include <ck/utility/type.hpp>
#include <ck/utility/data_type.hpp>
#include <ck/utility/amd_buffer_addressing.hpp>
#include <c10/hip/HIPException.h>
#define FAST_UNSAFE_CAST
// #define SWIZZLE_XCD_PID
// #define SWIZZLE_L2_TILE
// #define FORCE_LOAD_BIAS
#define FORCE_INLINE __attribute__((always_inline))


namespace gemm_rs {


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
    u.u32 = reinterpret_cast<unsigned short&>(bf) << 16;
    return u.fp32;
#else
    return static_cast<float>(bf);
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


__device__ FORCE_INLINE inline void block_sync_lds() {
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_s_barrier();
}


template<int num_tile_m, int num_tile_n, int GROUP_SIZE_M = 16>
__device__ FORCE_INLINE inline void compute_tile_indices(
    int tile_id, 
    int &tile_m_id, 
    int &tile_n_id
) {
    static_assert(GROUP_SIZE_M % 8 == 0);
    if constexpr (GROUP_SIZE_M > 0) {
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
    } else {
        tile_m_id = tile_id / num_tile_n;
        tile_n_id = tile_id % num_tile_n;
    }
}

template<int NUM_SMS, int NUM_GEMM_SMS, bool REMAP_XCD>
__device__ int FORCE_INLINE remap_xcd_pid(int pid) {
    if constexpr(REMAP_XCD) {
        return (pid % NUM_XCDS) * (NUM_GEMM_SMS / NUM_XCDS) + (pid / NUM_XCDS);
    } else {
        return pid;
    }
}


template<int num_tile_n>
struct EpilogueSignal {
    FORCE_INLINE __device__ void operator()(int tid, int tile_m_id, int tile_n_id, int *signal_ptr, int round_trip) const {
        if (tid == 0) {
            auto signal_arr = reinterpret_cast<int (*)[num_tile_n]>(signal_ptr);
            __hip_atomic_store(&signal_arr[tile_m_id][tile_n_id], round_trip, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
};


template <int M, int N, int K,
    int BM, int BN, int BK, 
    int NUM_THREADS, int WARP_M, int WARP_N, 
    int WORLD_SIZE, int NUM_GEMM_SMS, int NUM_RS_SMS,
    int GROUP_SIZE_M, bool REMAP_XCD
>
__launch_bounds__(NUM_THREADS)
__global__ void gemm_kernel(
    const bfloat16_t *x, // [M, K]
    const bfloat16_t *w, // [N, K]
    const bfloat16_t *b, // [N] or nullptr
    std::array<bfloat16_t *, WORLD_SIZE> c, // WORLD_SIZE * [M, N]
    std::array<int*, WORLD_SIZE> signal, // WORLD_SIZE * [M / BM][N / BN]
    bfloat16_t *out, // [M / WORLD_SIZE, N]
    int rank,
    int round_trip
) {
    const int tid = threadIdx.x;
    const int pid = blockIdx.x;
    __builtin_assume(tid >= 0 && tid < NUM_THREADS);
    __builtin_assume(pid >= 0 && pid < (NUM_GEMM_SMS + NUM_RS_SMS));
    const int lane_id = __lane_id();
    __builtin_assume(lane_id >= 0 && lane_id < 64);


    if (pid < NUM_GEMM_SMS) { /* GEMM */

        auto *c_ptr = c[rank];
        auto *signal_ptr = signal[rank];
        constexpr int v_mfma_f32_16x16x16_bf16 = (BM * BN * BK) / (WARP_M * WARP_N) / (16*16*16);
        // Compiler will merge two ds_{read,write}_b64 to ds_{read,write`}2st64_b64
        constexpr int ds_read_b128_a = (BM * BK / WARP_M) / 64 / 8;
        constexpr int ds_read_b128_b = (BN * BK / WARP_N) / 64 / 8;
        constexpr int ds_read_b128 = ds_read_b128_a + ds_read_b128_b;
        constexpr int ds_write_b128_a = (BM * BK) / NUM_THREADS / 8;
        constexpr int ds_write_b128_b = (BN * BK) / NUM_THREADS / 8;
        constexpr int ds_write_b128 = ds_write_b128_a + ds_write_b128_b;
        constexpr int buffer_load_dwordx2_a = (BM * BK) / NUM_THREADS / 4;
        constexpr int buffer_load_dwordx2_b = (BN * BK) / NUM_THREADS / 4;
        constexpr int buffer_load_dwordx2 = buffer_load_dwordx2_a + buffer_load_dwordx2_b;
        const int pid0 = blockIdx.x;
        constexpr int NUM_XCDS = 8;
        constexpr int num_tile_m = ceil_div(M, BM);
        constexpr int num_tile_n = ceil_div(N, BN);
        constexpr int num_tile_k = ceil_div(K, BK);
        constexpr int num_tiles = num_tile_m * num_tile_n;
    #ifdef SWIZZLE_XCD_PID
        const int pid = (pid0 % NUM_XCDS) * (NUM_GEMM_SMS / NUM_XCDS) + (pid0 / NUM_XCDS);
    #else
        const int pid = pid0;
    #endif
        
        const int tid = threadIdx.x;
        const int lane_id = __lane_id();
        __builtin_assume(pid >= 0 && pid < NUM_GEMM_SMS + NUM_RS_SMS);
        __builtin_assume(tid >= 0 && tid < NUM_THREADS);
        __builtin_assume(lane_id >= 0 && lane_id < 64);
        // each thread load 4 elements
        static_assert(BK % 4 == 0 && NUM_THREADS * 4 % BK == 0);
        constexpr int WM = 16, WN = 16, WK = 16;

        constexpr int Frag_M = exact_div<BM, WM * WARP_M>();
        constexpr int Frag_N = exact_div<BN, WN * WARP_N>();
        constexpr int Frag_K = exact_div<BK, WK>();
        const int warp_id = __builtin_amdgcn_readfirstlane(tid / 64);
        // const int warp_id = 0;
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

        auto load_vgpr = [&](int m, int n, int k) FORCE_INLINE {
            auto x_arr = ck::make_wave_buffer_resource<bfloat16_t>(const_cast<bfloat16_t*>(x), M * K);
            auto w_arr = ck::make_wave_buffer_resource<bfloat16_t>(const_cast<bfloat16_t*>(w), N * K);
            int v_offset = ((tid * 4 / BK) * K + (tid * 4 % BK)) * sizeof(bfloat16_t);
            uint32_t src_addr_shift = (K % BK == 0) || (k + tid * 4 % BK < K) ? 0 : 0x80000000;
            ck::static_for<0, sizeof(vgpr_x) / sizeof(vgpr_x[0]), 1>{}([&](auto t) {
                int s_offset = ((m * K + k) + t * NUM_THREADS * 4 / BK * K) * sizeof(bfloat16_t);
                vgpr_x[t] = __builtin_bit_cast(bf16x4_t, ck::amd_buffer_load_impl_raw<sizeof(bf16x4_t)>(
                    x_arr, v_offset + src_addr_shift, s_offset));
            });
            ck::static_for<0, sizeof(vgpr_w) / sizeof(vgpr_w[0]), 1>{}([&](auto t) {
                int s_offset = ((n * K + k) + t * NUM_THREADS * 4 / BK * K) * sizeof(bfloat16_t);
                vgpr_w[t] = __builtin_bit_cast(bf16x4_t, ck::amd_buffer_load_impl_raw<sizeof(bf16x4_t)>(
                    w_arr, v_offset + src_addr_shift, s_offset));
            });

        };

        auto load_lds = [&]() FORCE_INLINE {
            // diagonal swizzle, shape=[16, 64] dtype=bfloat16
            #pragma unroll
            for (int t=0;t<sizeof(vgpr_x)/sizeof(vgpr_x[0]);++t) {
                int row0 = t * NUM_THREADS * 4 / BK;
                int row1 = tid * 4 / BK;
                int col0 = tid * 4 % BK;
                int col1 = (row1 * 4 + col0) % BK;
                *reinterpret_cast<bf16x4_t*>(&s_x[row0 + row1][col1]) = vgpr_x[t];
            }
            #pragma unroll
            for (int t=0;t<sizeof(vgpr_w)/sizeof(vgpr_w[0]);++t) {
                int row0 = t * NUM_THREADS * 4 / BK;
                int row1 = tid * 4 / BK;
                int col0 = tid * 4 % BK;
                int col1 = (row1 * 4 + col0) % BK;
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


        auto frags_load = [&]() FORCE_INLINE {
            ck::static_for<0, Frag_K, 1>{}([&](auto k) {
                ck::static_for<0, Frag_M, 1>{}([&](auto i) {
                        const int row1 = (warp_m * Frag_M + i) * WM;
                        const int row0 = lane_id % 16;
                        const int col0 = k * 16 + lane_id / 16 * 4;
                        const int col1 = (row0 * 4 + col0) % BK;
                        frag_x[i][k] = *reinterpret_cast<const bf16x4_t*>(&s_x[row0 + row1][col1]);
                });
                ck::static_for<0, Frag_N, 1>{}([&](auto j) {
                        const int row1 = (warp_n * Frag_N + j) * WN;
                        const int row0 = lane_id % 16;
                        const int col0 = k * 16 + lane_id / 16 * 4;
                        const int col1 = (row0 * 4 + col0) % BK;
                        frag_w[j][k] = *reinterpret_cast<const bf16x4_t*>(&s_w[row0 + row1][col1]);
                });
            });
        };

        auto frags_mfma = [&]() FORCE_INLINE {
            ck::static_for<0, Frag_M, 1>{}([&](auto i) {
                ck::static_for<0, Frag_N, 1>{}([&](auto j) {
                    ck::static_for<0, Frag_K, 1>{}([&](auto k) {
                        // a: [16][16], b: [16][16], c: [16][16]
                        // mfma requires a: row-major, b: col-major, out: col-major
                        // so we compute w^T * x^T = c^T so we can treat out as col-major
                        frag_c[i][j].pack = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(frag_w[j][k].pack, frag_x[i][k].pack, frag_c[i][j].pack, 0, 0, 0);
                    });
                });
            });
        };
        

        auto store_frags = [&](int m, int n) FORCE_INLINE {
            auto b_arr = ck::make_wave_buffer_resource<bfloat16_t>(const_cast<bfloat16_t*>(b), N);
            auto c_arr = ck::make_wave_buffer_resource<bfloat16_t>(c_ptr, M * N);
            fp32x4_t c_out[Frag_M][Frag_N];
            ck::static_for<0, Frag_M, 1>{}([&](auto i) {
                ck::static_for<0, Frag_N, 1>{}([&](auto j) {
                    ck::static_for<0, 4, 1>{}([&](auto t) {
                        // v_accvgpr_read_b32
                        c_out[i][j].x[t] = frag_c[i][j].x[t];
                    });
                    // c_out: [16][16]
                    int row = lane_id % 16;
                    int col = lane_id / 16 * 4;
                    uint32_t src_addr_shift = (N % BN == 0) || (n + (j + warp_n * Frag_N) * WN + col < N) ? 0 : 0x80000000;
                    // load b
                    int b_s_offset = (n + (j + warp_n * Frag_N) * WN) * sizeof(bfloat16_t);
                    int b_v_offset = col * sizeof(bfloat16_t) + src_addr_shift;
                    auto b_vec = __builtin_bit_cast(bf16x4_t, ck::amd_buffer_load_impl_raw<sizeof(bf16x4_t)>(
                        b_arr, b_v_offset, b_s_offset));
                    // compute c
                    bf16x4_t c_out_bf16;
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        c_out_bf16.x[t] = fast_f32tob16(c_out[i][j].x[t] + b_vec.x[t]);
                    }
                    // write c
                    int c_s_offset = b_s_offset + (m + (i + warp_m * Frag_M) * WM) * N * sizeof(bfloat16_t);
                    int c_v_offset = b_v_offset + (row * N) * sizeof(bfloat16_t);
                    ck::amd_buffer_store_impl_raw<sizeof(bf16x4_t), ck::AmdBufferCoherenceEnum::WAVE_NT0>(c_out_bf16.pack, c_arr, c_v_offset, c_s_offset);
                });
            });
        };

        

        for (int tile_id=pid; tile_id<num_tiles; tile_id+=NUM_GEMM_SMS) {
    #ifdef SWIZZLE_L2_TILE
            int tile_m_id, tile_n_id;
            compute_tile_indices<num_tile_m, num_tile_n>(tile_id, tile_m_id, tile_n_id);
    #else
            // int tile_m_id = tile_id / num_tile_n;
            // int tile_n_id = tile_id % num_tile_n;
            int tile_m_id = tile_id % num_tile_m;
            int tile_n_id = tile_id / num_tile_m;
    #endif
            int m = tile_m_id * BM;
            int n = tile_n_id * BN;
            load_vgpr(m, n, 0);       // GDS -> VGPR #0
            load_lds();               // VGPR -> LDS #0
            load_vgpr(m, n, 1 * BK);  // GDS -> VGPR #1
            zero_all_frags();
            block_sync_lds();
            frags_load();             // LDS -> FRAG #0
            __builtin_amdgcn_sched_barrier(0);
            // #pragma clang loop unroll_count(2)
            // #pragma unroll 2
            // #pragma unroll
            for (int tile_k_id = 1; tile_k_id < (num_tile_k - 1); ++tile_k_id) {
                asm volatile(R"(
                    ; Main Loop Begin
                )" ::: "memory");
                block_sync_lds();
                // Stage 1
                load_lds();                             // VGPR -> LDS #1
                load_vgpr(m, n, (tile_k_id + 1) * BK);  // GDS -> VGPR #2(k+1)
                frags_mfma();                           // MFMA #0(k-1)
                // 120
                #pragma unroll
                for (int k = 0; k < buffer_load_dwordx2 / 2; ++k) {
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, 3, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    __builtin_amdgcn_sched_group_barrier(0x008, 3, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    __builtin_amdgcn_sched_group_barrier(0x008, 3, 0); // MFMA
                }

                block_sync_lds();
                // Stage 2                       
                frags_load();                           // LDS -> FRAG #1(k)
                // 60
                #pragma unroll
                for (int k = 0; k < ds_read_b128; ++k) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                }
                __builtin_amdgcn_sched_barrier(0);
                asm volatile(R"(
                    ; Main Loop End
                )" ::: "memory");
            }
            frags_mfma();                               // MFMA #1(n-2)
            block_sync_lds();
            load_lds();                                 // VGPR -> LDS #2(n-1)
            block_sync_lds();
            frags_load();                               // LDS -> FRAG #2(n-1)
            frags_mfma();                               // MFMA #2(n-1)
            store_frags(m, n);
            
            __builtin_amdgcn_s_barrier();
            if (tid == 0) {
                __hip_atomic_store(&signal_ptr[tile_m_id * num_tile_n + tile_n_id], round_trip, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
    } else { /* Reduce Scatter */
        constexpr int M_per_rank = exact_div<M, WORLD_SIZE>();
        static_assert(BN % 8 == 0);
        constexpr int NUM_TILES_M = ceil_div(M_per_rank, BM);
        constexpr int NUM_TILES_N = ceil_div(N, BN);
        constexpr int TOTAL_TILES = NUM_TILES_M * NUM_TILES_N;
        constexpr int ELEMENTS_PER_THREAD = 8;
        struct rs_vec_t { bfloat16_t data[ELEMENTS_PER_THREAD]; };
        static_assert(N % ELEMENTS_PER_THREAD == 0);
        const int rs_pid = pid - NUM_GEMM_SMS;
        for (int tile_id = rs_pid; tile_id < TOTAL_TILES; tile_id += NUM_RS_SMS) {
            int tile_m = tile_id % NUM_TILES_M;
            int tile_n = tile_id / NUM_TILES_M;
            float accum[BM * BN / NUM_THREADS] = {};
            static_assert((sizeof(accum) / sizeof(float)) % ELEMENTS_PER_THREAD == 0);
            #pragma clang loop unroll_count(4)
            for (int r = 0; r < WORLD_SIZE; r++) {
                int swizzle_rank = (rs_pid + r) % WORLD_SIZE;
                const int M_begin = rank * M_per_rank;
                const int M_end = (rank + 1) * M_per_rank;
                // since tile may be not well aligned, we have to wait at most two signal
                // we should wait signal[signal_m0 .. signal_m1][signal_n]
                const int tile_row_begin = M_begin + tile_m * BM;
                const int tile_row_end = min(tile_row_begin + BM, M_end);
                int signal_m0 = tile_row_begin / BM;
                int signal_m1 = (tile_row_end - 1) / BM;
                int signal_n = tile_n;
                int signal_m = signal_m0 + tid;
                if (signal_m <= signal_m1) {
                    auto *signal_arr = reinterpret_cast<int (*)[NUM_TILES_N]>(signal[swizzle_rank]);
                    while (__hip_atomic_load(&signal_arr[signal_m][signal_n], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != round_trip) {} // spin wait
                }
                __syncthreads();
                auto input_arr = ck::make_wave_buffer_resource<bfloat16_t>(c[swizzle_rank] + M_begin * N, M_per_rank * N);
                // auto input_arr = reinterpret_cast<bfloat16_t (*)[N]>(c[swizzle_rank] + M_begin * N);
                #pragma unroll
                for (int t = 0; t < sizeof(accum) / sizeof(float); t+=ELEMENTS_PER_THREAD) {
                    int i = (t * NUM_THREADS + tid * ELEMENTS_PER_THREAD) / BN;
                    int j = (t * NUM_THREADS + tid * ELEMENTS_PER_THREAD) % BN;
                    int global_i = tile_m * BM + i;
                    int global_j = tile_n * BN + j;
                    auto vec = ck::bit_cast<rs_vec_t>(ck::amd_buffer_load_impl_raw<sizeof(rs_vec_t)>(
                        input_arr, (global_i * N + global_j) * sizeof(bfloat16_t), 0));
                    #pragma unroll
                    for (int k = 0; k < ELEMENTS_PER_THREAD; ++k) accum[t + k] += fast_b16tof32(vec.data[k]);
                }
            }
            // auto out_arr = reinterpret_cast<bfloat16_t (*)[N]>(out);
            auto out_arr = ck::make_wave_buffer_resource<bfloat16_t>(out, M_per_rank * N);
            #pragma unroll
            for (int t = 0; t < sizeof(accum) / sizeof(float); t+=ELEMENTS_PER_THREAD) {
                int i = (t * NUM_THREADS + tid * ELEMENTS_PER_THREAD) / BN;
                int j = (t * NUM_THREADS + tid * ELEMENTS_PER_THREAD) % BN;
                int global_i = tile_m * BM + i;
                int global_j = tile_n * BN + j;
                rs_vec_t vec;
                #pragma unroll
                for (int k = 0; k < ELEMENTS_PER_THREAD; ++k) vec.data[k] = fast_f32tob16(accum[t + k]);
                using r_t = typename ck::vector_type<int8_t, sizeof(rs_vec_t)>::type;
                ck::amd_buffer_store_impl_raw<sizeof(rs_vec_t)>(ck::bit_cast<r_t>(vec),
                    out_arr, (global_i * N + global_j) * sizeof(bfloat16_t), 0
                );
            }
        }
    }
}

constexpr long IntKey(int M, int N, int K, int WORLD_SIZE, bool bias) {
    union {
        struct {uint16_t M, N, K, WORLD_SIZE;} data;
        long key;
    } u { .data={
        static_cast<uint16_t>(M) ,
        static_cast<uint16_t>(N),
        static_cast<uint16_t>(K),
        static_cast<uint16_t>(WORLD_SIZE)}};
    return u.key;
}


constexpr int WORLD_SIZE = 8;
constexpr int MI300X_NUM_SMS = 304;
constexpr int RS_SMS = 16;
constexpr int GEMM_SMS = MI300X_NUM_SMS - RS_SMS;

// Simuate IntraNode environment for test
class GEMMFactory {
private:
    std::unordered_map<long, std::function<void(const bfloat16_t* x, const bfloat16_t* w, const bfloat16_t *b, bfloat16_t* c, hipStream_t stream)>> gemm_map_;
    std::unordered_map<long, std::function<void(const bfloat16_t *const *inputs, bfloat16_t *out, int *sig[], int rank, hipStream_t stream)>> rs_map_;
    std::unordered_map<long, std::function<void(const bfloat16_t *x, const bfloat16_t *w, const bfloat16_t *b, bfloat16_t *c[], int *sig[], bfloat16_t *out, int rank, hipStream_t stream)>> gemm_rs_map_;
    int *dummy_signal_buf_;
    bfloat16_t *zero_bias_buf_;
    int round_trip_ = 0;
    
    template <int M, int N, int K, int BM, int BN, int BK, int WARP_M, int WARP_N, int GROUP_SIZE_M>
    inline void RegisterGEMM() {
        static_assert(M * N * sizeof(bfloat16_t) <= MAX_IPC_MEM_SIZE, "C size exceeds MAX_IPC_MEM_SIZE");
        constexpr int NUM_THREADS = WARP_M * WARP_N * AMDGCN_WAVEFRONT_SIZE;
        TORCH_CHECK(gemm_map_.count(IntKey(M, N, K, 8, 0)) == 0);
        gemm_map_[IntKey(M, N, K, 8, 0)] = [&](const bfloat16_t *x, const bfloat16_t *w, const bfloat16_t *b, bfloat16_t *c, hipStream_t stream) {
            std::array<int*, WORLD_SIZE> signal = {dummy_signal_buf_, nullptr};
            std::array<bfloat16_t*, WORLD_SIZE> c_ptrs = {c, nullptr};
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_kernel<M, N, K, BM, BN, BK, NUM_THREADS, WARP_M, WARP_N, WORLD_SIZE, 304, 0, GROUP_SIZE_M, false>),
                dim3(304), dim3(NUM_THREADS), 0, stream,
                x, w, b, c_ptrs, signal, nullptr, 0, 0
            );
        };
#ifdef __ENABLE_LOCAL_DEBUG__
        TORCH_CHECK(rs_map_.count(IntKey(M, N, 0, WORLD_SIZE, false)) == 0);
        rs_map_[IntKey(M, N, 0, WORLD_SIZE, false)] = [&](const bfloat16_t *const *inputs, bfloat16_t *out, int *sig[], int rank, hipStream_t stream) mutable {
            std::array<int*, WORLD_SIZE> signal;
            std::array<bfloat16_t*, WORLD_SIZE> c_ptrs;
            for (int i = 0; i < WORLD_SIZE; i++) {
                signal[i] = sig[i];
                c_ptrs[i] = const_cast<bfloat16_t*>(inputs[i]);
            }
            int round_trip = ++round_trip_;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_kernel<M, N, K, BM, BN, BK, NUM_THREADS, WARP_M, WARP_N, WORLD_SIZE, 0, RS_SMS, GROUP_SIZE_M, false>),
                dim3(MI300X_NUM_SMS), dim3(NUM_THREADS), 0, stream,
                nullptr, nullptr, nullptr, c_ptrs, signal, out, rank, round_trip
            );
        };
#endif
        TORCH_CHECK(gemm_rs_map_.count(IntKey(M, N, K, WORLD_SIZE, false)) == 0);
        gemm_rs_map_[IntKey(M, N, K, WORLD_SIZE, false)] = [&](const bfloat16_t *x, const bfloat16_t *w, const bfloat16_t *b, bfloat16_t *c[], int *sig[], bfloat16_t *out, int rank, hipStream_t stream) mutable {
            std::array<int*, WORLD_SIZE> signal;
            std::array<bfloat16_t*, WORLD_SIZE> c_ptrs;
            for (int i = 0; i < WORLD_SIZE; i++) {
                signal[i] = sig[i];
                c_ptrs[i] = c[i];
            }
            // TODO: sync all rank
            // C10_HIP_CHECK(hipDeviceSynchronize()); // make sure memset is done before kernel launch

            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_kernel<M, N, K, BM, BN, BK, NUM_THREADS, WARP_M, WARP_N, WORLD_SIZE, GEMM_SMS, RS_SMS, GROUP_SIZE_M, false>),
                dim3(MI300X_NUM_SMS), dim3(NUM_THREADS), 0, stream,
                x, w, b, c_ptrs, signal, out, rank, ++round_trip_
            );
        };
    }

    GEMMFactory() {
        C10_HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&dummy_signal_buf_), SIGNAL_BUF_SIZE, hipDeviceMallocUncached)); // dummy buffer for signal
        C10_HIP_CHECK(hipMemset(dummy_signal_buf_, 0, SIGNAL_BUF_SIZE));
#ifdef FORCE_LOAD_BIAS
        C10_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&zero_bias_buf_), 8192 * sizeof(bfloat16_t)));
        C10_HIP_CHECK(hipMemset(zero_bias_buf_, 0, 8192 * sizeof(bfloat16_t)));
#endif
        // RegisterGEMM<8192, 8192, 2048>();
        // RegisterGEMM<4096, 4096, 4096>();
        // RegisterGEMM<4096, 4096, 2048>();
        
        // int M, int N, int K, int BM, int BN, int BK, int WARP_M, int WARP_N, int GROUP_SIZE_M
        RegisterGEMM<64,   7168, 2304, 32,  128, 64, 2, 2, 0 >();      // 64 x 7168 x (18432/8)
        RegisterGEMM<512,  4096, 1536, 64,  128, 64, 2, 2, 8 >();      // 512 x 4096 x (12288/8)
        RegisterGEMM<2048, 2880, 360,  128, 128, 64, 2, 2, 8 >();      // 2048 x 2880 x (2880/8)
        RegisterGEMM<4096, 4096, 512,  128, 256, 64, 2, 2, 8 >();      // 4096 x 4096 x (4096/8)
        RegisterGEMM<8192, 4096, 1792, 256, 224, 64, 2, 2, 16>();      // 8192 x 4096 x (14336/8)
        RegisterGEMM<8192, 8192, 3696, 224, 256, 64, 2, 2, 16>();      // 8192 x 8192 x (29568/8)
    }

public:
    static GEMMFactory* get() {
        static std::unique_ptr<GEMMFactory> instance;
        if (!instance) {
            instance = std::unique_ptr<GEMMFactory>(new GEMMFactory());
        }
        return instance.get();
    }

    void *get_completed_signal_for_next(hipStream_t stream) {
        int round_trip = round_trip_ + 1;
        C10_HIP_CHECK(hipMemsetD32Async(dummy_signal_buf_, round_trip, SIGNAL_BUF_SIZE / sizeof(int), stream));
        return dummy_signal_buf_;
    }

    void launch_gemm(const bfloat16_t *x_ptr, const bfloat16_t *w_ptr, const bfloat16_t *b_ptr, bfloat16_t *c_ptr, int M, int N, int K, hipStream_t stream) {
        auto key = IntKey(M, N, K, 8, 0);
        auto it = gemm_map_.find(key);
        TORCH_CHECK(it != gemm_map_.end(), "Unsupported GEMM size: ", M, "x", N, "x", K);
        auto func = it->second;
        func(x_ptr, w_ptr, b_ptr, c_ptr, stream);
    }

    void launch_rs(const bfloat16_t *inputs[], bfloat16_t *out, int *signal_buf[], int rank, int world_size, int M, int N, hipStream_t stream) {
        auto key = IntKey(M, N, 0, world_size, false); // K is not used in RS, set to a dummy value
        auto it = rs_map_.find(key);
        TORCH_CHECK(it != rs_map_.end(), "Unsupported RS size: ", M, "x", N);
        auto func = it->second;
        func(inputs, out, signal_buf, rank, stream);
    }

    void launch_gemm_rs(const bfloat16_t *x_ptr, const bfloat16_t *w_ptr, const bfloat16_t *b_ptr, bfloat16_t *c_ptr[], int *signal_buf[], bfloat16_t *out, int rank, int M, int N, int K, int world_size, hipStream_t stream) {
        bool bias = b_ptr != nullptr;
        auto key = IntKey(M, N, K, world_size, bias);
        auto it = gemm_rs_map_.find(key);
        TORCH_CHECK(it != gemm_rs_map_.end(), "Unsupported GEMM+RS size: ", M, "x", N, "x", K, "-", bias);
        auto func = it->second;
        func(x_ptr, w_ptr, b_ptr, c_ptr, signal_buf, out, rank, stream);
    }
    
};



void launch_gemm(const void *x, const void *w, const void *b, void *out, int M, int N, int K, hipStream_t stream) {
    auto *x_ptr = reinterpret_cast<const bfloat16_t *>(x);
    auto *w_ptr = reinterpret_cast<const bfloat16_t *>(w);
    auto *b_ptr = reinterpret_cast<const bfloat16_t *>(b);
    auto *c_ptr = reinterpret_cast<bfloat16_t *>(out);
    GEMMFactory::get()->launch_gemm(x_ptr, w_ptr, b_ptr, c_ptr, M, N, K, stream);
}

void launch_rs(const void *inputs[], void *output, int rank, int world_size, int M, int N, hipStream_t stream) {
    auto *inputs_ptr = reinterpret_cast<const bfloat16_t **>(inputs);
    auto *output_ptr = reinterpret_cast<bfloat16_t *>(output);
    std::vector<int*> signal_buf(world_size, reinterpret_cast<int*>(GEMMFactory::get()->get_completed_signal_for_next(stream)));
    GEMMFactory::get()->launch_rs(inputs_ptr, output_ptr, signal_buf.data(), rank, world_size, M, N, stream);
}

void launch_gemm_rs_dist(const void *x, const void *w,const void *b, void *rs_buf[], void *sig_buf[], void *output, int rank, int world_size, int M, int N, int K, hipStream_t stream) {
    auto * x_ptr = reinterpret_cast<const bfloat16_t *>(x);
    auto * w_ptr = reinterpret_cast<const bfloat16_t *>(w);
    auto * b_ptr = reinterpret_cast<const bfloat16_t *>(b);
    auto * out_ptr = reinterpret_cast<bfloat16_t *>(output);
    auto *rs_buf_ptr = reinterpret_cast<bfloat16_t **>(const_cast<void **>(rs_buf));
    auto *sig_buf_ptr = reinterpret_cast<int **>(const_cast<void **>(sig_buf));
    GEMMFactory::get()->launch_gemm_rs(x_ptr, w_ptr, b_ptr, rs_buf_ptr, sig_buf_ptr, out_ptr, rank, M, N, K, world_size, stream);
}

void launch_gemm_rs(const void *x, const void *w,const void *b, void *rs_buf[], void *sig_buf, void *output, int rank, int world_size, int M, int N, int K, hipStream_t stream) {
    std::vector<void*> signal_buf(world_size);
    for (int i = 0; i < world_size; i++) {
        if (i == rank) {
            signal_buf[i] = sig_buf;
        } else {
            signal_buf[i] = GEMMFactory::get()->get_completed_signal_for_next(stream);
        }

    }
    launch_gemm_rs_dist(x, w, b, rs_buf, signal_buf.data(), output, rank, world_size, M, N, K, stream);
}

} // namespace gemm_rs