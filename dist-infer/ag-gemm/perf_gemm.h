#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <ck/utility/data_type.hpp>
#include <ck/utility/amd_buffer_addressing.hpp>
#include <ck/utility/ignore.hpp>
#define FAST_UNSAFE_CAST
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

    __device__ static FORCE_INLINE void schedule_loop() {
        if constexpr ((BM == 256 && BN == 224) || (BM == 224 && BN == 256)) {
            // Large GEMM
            // MFMA 224, DS_READ 30, DS_WRITE 15, BUFFER_LOAD 30
            #pragma unroll
            for (int k = 0; k < buffer_load_dwordx2; ++k) { // 150
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 3, 0); // MFMA
            }
            #pragma unroll
            for (int k = 0; k < ds_read_b128; ++k) { // 60
                __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            }
        } else if constexpr ((BM == 256 && BN == 128) || (BM == 128 && BN == 256) || (BM == 128 && BN == 128)) {
            // Middle GEMM
            // (256X128) MFMA 128, DS_READ_24, DS_WRITE 12, BUFFER_LOAD 24
            // (128X128) MFMA 64,  DS_READ 16, DS_WRITE 8,  BUFFER_LOAD 16
            #pragma unroll
            for (int k = 0; k < buffer_load_dwordx2; ++k) { // 96, 48
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            }
            #pragma unroll
            for (int k = 0; k < ds_read_b128; ++k) { // 24, 32
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 2, 0); // DS read
            }
        } else if constexpr ((BM == 128 && BN == 32) || (BM == 32 && BN == 128) 
            || (BM == 128 && BN == 64) || (BM == 64 && BN == 128) 
            || (BM == 64 && BN == 32) || (BM == 32 && BN == 32)
            || (BM == 32 && BN == 32)
        ) {
            // Small GEMM
            // (128x64) MFMA 32, DS_WRITE 6, DS_READ 12, BUFFER_LOAD 12
            // (128x32) MFMA 16, DS_WRITE 5, DS_READ 8 , BUFFER_LOAD 10
            // (64x32)  MFMA 8,  DS_WRITE 3, DS_READ 6 , BUFFER_LOAD 6
            #pragma unroll
            for (int k = 0; k < buffer_load_dwordx2; ++k) { // 20
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            }
            #pragma unroll
            for (int k = 0; k < ds_read_b128; ++k) { // 8
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            }
        } else {
            static_assert((BM == 256 && BN == 224) || (BM == 224 && BN == 256) ||
                          (BM == 256 && BN == 128) || (BM == 128 && BN == 256) ||
                          (BM == 128 && BN == 128) ||
                          (BM == 128 && BN == 64) || (BM == 64 && BN == 128) ||
                          (BM == 128 && BN == 32) || (BM == 32 && BN == 128) ||
                          (BM == 64 && BN == 32) || (BM == 32 && BN == 64) || (BM == 32 && BN == 32),
                          "Unsupported BM, BN");
        }

    }
};

} // namespace roc_isa



using bfloat16_t = __bf16;

__device__ __host__ FORCE_INLINE inline constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

template<int a, int b>
__device__ __host__ FORCE_INLINE constexpr int exact_div() {
    static_assert(a % b == 0);
    return a / b;
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


__device__ FORCE_INLINE inline bfloat16_t fast_f32tob16(float f) {
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


__device__ FORCE_INLINE inline float fast_b16tof32(bfloat16_t bf) {
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



__device__ inline void block_sync_lds() {
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_s_barrier();
}

template<int num_tile_m, int num_tile_n, int GROUP_SIZE_M = 16>
__device__ __forceinline__ void compute_tile_indices(
    int tile_id, 
    int &tile_m_id, 
    int &tile_n_id
) {
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

using signal_t = int[128][128];

template<int M, int N, int K, int BM, int BN, int BK, int NUM_SMS, int NUM_GEMM_SMS, int NUM_THREADS, int WARP_M, int WARP_N>
__launch_bounds__(NUM_THREADS)
__global__ void gemm_kernel(
    const bfloat16_t *x, // M x K
    const bfloat16_t *w, // N x K
    const bfloat16_t *b, // N
    bfloat16_t *c,       // M x N
    signal_t *signal
) {
    const int pid0 = blockIdx.x;
    constexpr int GEMM_SMS = NUM_SMS;
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
    using inst_nums = roc_isa::InstCalculator<BM, BN, BK, NUM_THREADS, WARP_M, WARP_N>;
    const int tid = threadIdx.x;
    const int lane_id = __lane_id();
    __builtin_assume(pid >= 0 && pid < NUM_SMS);
    __builtin_assume(tid >= 0 && tid < NUM_THREADS);
    __builtin_assume(lane_id >= 0 && lane_id < 64);
    // each thread load 4 elements
    static_assert(BK % 4 == 0 && NUM_THREADS * 4 % BK == 0);
    constexpr int WM = 16, WN = 16, WK = 16;

    constexpr int Frag_M = exact_div<BM, WM * WARP_M>();
    constexpr int Frag_N = exact_div<BN, WN * WARP_N>();
    constexpr int Frag_K = exact_div<BK, WK>();
    const int warp_id = __builtin_amdgcn_readfirstlane(tid / roc_isa::AMDGCN_WAVEFRONT_SIZE);
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
        });
        ck::static_for<0, Frag_K, 1>{}([&](auto k){
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
        auto c_arr = ck::make_wave_buffer_resource<bfloat16_t>(c, M * N);
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
                ck::amd_buffer_store_impl_raw<sizeof(bf16x4_t), ck::AmdBufferCoherenceEnum::WAVE_NT1>(c_out_bf16.pack, c_arr, c_v_offset, c_s_offset);
            });
        });
    };

    auto wait_signal = [&](int tile_m_id, int tile_k_id) {
        if (!signal) {
            return;
        }

        constexpr int M_LOCAL = M / 8;
        constexpr int COMM_M = M_LOCAL < 256 ? M_LOCAL : 256;
        constexpr int COMM_K = 512;

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
        
        if(warp_id == 0 && lane_id == 0) {
            for (int signal_m_id = signal_m_id_begin; signal_m_id < signal_m_id_end; ++signal_m_id) {
                while(!__hip_atomic_load(&sig[signal_m_id][signal_k_id], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM))
                    ;
            }
            // __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
        }
        // __builtin_amdgcn_s_barrier();
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    };


    for (int tile_id=pid; tile_id<num_tiles; tile_id+=NUM_GEMM_SMS) {
#ifdef SWIZZLE_L2_TILE
        int tile_m_id, tile_n_id;
        compute_tile_indices<num_tile_m, num_tile_n>(tile_id, tile_m_id, tile_n_id);
#else
        int tile_m_id = tile_id / num_tile_n;
        int tile_n_id = tile_id % num_tile_n;
#endif
        int m = tile_m_id * BM;
        int n = tile_n_id * BN;
        wait_signal(tile_m_id, 0);
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
            // asm volatile(R"(
            //     ; Main Loop Begin
            // )" ::: "memory");
            block_sync_lds();
            // Stage 1
            load_lds();                             // VGPR -> LDS #1
            if ((tile_k_id + 1) % 8 == 0) {
                wait_signal(tile_m_id, tile_k_id + 1);
            }
            load_vgpr(m, n, (tile_k_id + 1) * BK);  // GDS -> VGPR #2(k+1)
            frags_mfma();                           // MFMA #0(k-1)
            block_sync_lds();
            // Stage 2                       
            frags_load();                           // LDS -> FRAG #1(k)
            inst_nums::schedule_loop();
            __builtin_amdgcn_sched_barrier(0);
            // asm volatile(R"(
            //     ; Main Loop End
            // )" ::: "memory");
        }
        frags_mfma();                               // MFMA #1(n-2)
        block_sync_lds();
        load_lds();                                 // VGPR -> LDS #2(n-1)
        block_sync_lds();
        frags_load();                               // LDS -> FRAG #2(n-1)
        frags_mfma();                               // MFMA #2(n-1)
        store_frags(m, n);
    }
}
