#pragma once
#include "common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <ck/utility/data_type.hpp>
#include <ck/utility/amd_buffer_addressing.hpp>


namespace perf_gemm {


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
int __device__ remap_xcd_pid(int pid) {
    if constexpr(REMAP_XCD) {
        return (pid % NUM_XCDS) * (NUM_GEMM_SMS / NUM_XCDS) + (pid / NUM_XCDS);
    } else {
        return pid;
    }
}


struct EpilogueNOP {
    void operator()(int tid, int tile_m, int tile_n, void *signal_ptr) const {}
};

template<
    int M, int N, int K, 
    int BM, int BN, int BK, 
    int NUM_SMS, int NUM_GEMM_SMS, int NUM_THREADS, 
    int WARP_M, int WARP_N,
    int GROUP_SIZE_M, bool REMAP_XCD,
    typename Epilogue = EpilogueNOP
>
__device__ void FORCE_INLINE gemm_kernel(
    const bfloat16_t *x, // M x K
    const bfloat16_t *w, // N x K
    const bfloat16_t *b, // N
    bfloat16_t *c,       // M x N
    int *signal_ptr,
    int signal_val
) {
    constexpr int num_tile_m = ceil_div(M, BM);
    constexpr int num_tile_n = ceil_div(N, BN);
    constexpr int num_tile_k = ceil_div(K, BK);
    constexpr int num_tiles = num_tile_m * num_tile_n;
    const int pid = remap_xcd_pid<NUM_SMS, NUM_GEMM_SMS, REMAP_XCD>(threadIdx.x);    
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
    const int warp_id = __builtin_amdgcn_readfirstlane(tid / AMDGCN_WAVEFRONT_SIZE);
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
                ck::amd_buffer_store_impl_raw<sizeof(bf16x4_t)>(c_out_bf16.pack, c_arr, c_v_offset, c_s_offset);
            });
        });
    };

    

    for (int tile_id=pid; tile_id<num_tiles; tile_id+=NUM_GEMM_SMS) {
        int tile_m_id, tile_n_id;
        compute_tile_indices<num_tile_m, num_tile_n, GROUP_SIZE_M>(tile_id, tile_m_id, tile_n_id);
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
        // Epilogue{}(tid, tile_m_id, tile_n_id, signal_ptr, signal_val);
    }
}

} // namespace perf_gemm