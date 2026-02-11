// remove pytorch restriction
#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__
#include <hip/hip_fp16.h>

#include <ATen/ATen.h>
#include <ATen/ops/from_blob.h>
#include <c10/hip/HIPStream.h>
#include <torch/extension.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include "perf_gemm.h"
#include <ck/utility/amd_buffer_addressing.hpp>
#include <ck/version.h>

#include "rocwmma/rocwmma.hpp"
#include "rocwmma/rocwmma_coop.hpp"

namespace mma = rocwmma;
using f16 = mma::float16_t;
using b16 = mma::bfloat16_t;
using f32 = mma::float32_t;
using i32 = mma::int32_t;
using i64 = mma::int64_t;

#define USE_DBG 0
#define USE_ASSERT 0

#define DO_PRAGMA_(x) _Pragma(#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)
#define UNROLL DO_PRAGMA(unroll)
#define UNROLL_N(n) DO_PRAGMA(unroll n)
#define STR(x) #x
#define TO_STR(x) STR(x)

#define ASSERT(cond)                                                           \
    do {                                                                       \
        if (USE_ASSERT && !(cond)) {                                           \
            __assert_fail(#cond, __FILE__, __LINE__, __PRETTY_FUNCTION__);     \
        }                                                                      \
    } while (0)

#define DBG(fmt, ...)                                                          \
    do {                                                                       \
        if (USE_DBG && threadIdx.x % 64 == 0) {                                \
            printf(fmt "\n", ##__VA_ARGS__);                                   \
        }                                                                      \
    } while (0)

#define HOST_DBG(fmt, ...)                                                     \
    do {                                                                       \
        fprintf(stderr, fmt "\n", ##__VA_ARGS__);                              \
    } while (0)

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            fprintf(                                                           \
                stderr, "HIP error: %s (%d)\n  at %s:%d\n",                    \
                hipGetErrorString(err), err, __FILE__, __LINE__                \
            );                                                                 \
        }                                                                      \
    } while (0)

constexpr i32 WORLD_SIZE = 2;
constexpr i32 WARP_SIZE = 64;
constexpr i32 BLOCK_SIZE = 512;
constexpr i32 NUM_SMS = 304;

template <typename T> constexpr T ceil_div(T a, T b) { return (a + b - 1) / b; }

template <int N, typename T> struct vec_t {
    using type = __attribute__((__vector_size__(N))) T;
    static_assert(N % sizeof(T) == 0);
    constexpr static i32 nelem = N / sizeof(T);
    constexpr static i32 nelem_per_warp = WARP_SIZE * nelem;

    static __device__ void copy(T *dst, const T *src) {
        auto val =
            __builtin_nontemporal_load(reinterpret_cast<const type *>(src));
        __builtin_nontemporal_store(val, reinterpret_cast<type *>(dst));
    }

    template <int N_ELEM>
    __device__ static inline void warp_copy(T *dst, const T *src) {
        static_assert(N_ELEM % nelem_per_warp == 0);
        const auto lane_id = threadIdx.x % WARP_SIZE;
        UNROLL
        for (int i = 0; i < N_ELEM / nelem_per_warp; i++) {
            auto src_ptr = reinterpret_cast<const type *>(
                src + i * nelem_per_warp + lane_id * nelem
            );
            auto val = __builtin_nontemporal_load(src_ptr);
            auto dst_ptr = reinterpret_cast<type *>(
                dst + i * nelem_per_warp + lane_id * nelem
            );
            __builtin_nontemporal_store(val, dst_ptr);
        }
    }

    template <int N_ELEM> using accum_type = type[N_ELEM / nelem_per_warp];

    template <int N_ELEM>
    __device__ static inline void
    warp_accum(accum_type<N_ELEM> &acc, const T *src, f32 weight) {
        static_assert(N_ELEM % nelem_per_warp == 0);
        const auto lane_id = threadIdx.x % WARP_SIZE;
        UNROLL
        for (int i = 0; i < N_ELEM / nelem_per_warp; i++) {
            auto ptr = reinterpret_cast<const type *>(
                src + i * nelem_per_warp + lane_id * nelem
            );
            auto val = __builtin_nontemporal_load(ptr);
            UNROLL
            for (int j = 0; j < nelem; j++) {
                acc[i][j] += val[j] * weight;
            }
        }
    }

    template <int N_ELEM>
    __device__ static inline void
    warp_accum_store(T *dst, accum_type<N_ELEM> &acc) {
        static_assert(N_ELEM % nelem_per_warp == 0);
        const auto lane_id = threadIdx.x % WARP_SIZE;
        UNROLL
        for (int i = 0; i < N_ELEM / nelem_per_warp; i++) {
            auto ptr = reinterpret_cast<const type *>(
                dst + i * nelem_per_warp + lane_id * nelem
            );
            __builtin_nontemporal_store(acc[i], ptr);
        }
    }

    template <int N_ELEM, bool NT = true>
    __device__ static inline void
    warp_load(accum_type<N_ELEM> &acc, const T *src) {
        static_assert(N_ELEM % nelem_per_warp == 0);
        const auto lane_id = threadIdx.x % WARP_SIZE;
        UNROLL
        for (int i = 0; i < N_ELEM / nelem_per_warp; i++) {
            auto src_ptr = reinterpret_cast<const type *>(
                src + i * nelem_per_warp + lane_id * nelem
            );
            if constexpr (NT) {
                acc[i] = __builtin_nontemporal_load(src_ptr);
            } else {
                acc[i] = *src_ptr;
            }
        }
    }

    template <int N_ELEM, bool NT = true>
    __device__ static inline void
    warp_store(T *dst, const accum_type<N_ELEM> &acc) {
        static_assert(N_ELEM % nelem_per_warp == 0);
        const auto lane_id = threadIdx.x % WARP_SIZE;
        UNROLL
        for (int i = 0; i < N_ELEM / nelem_per_warp; i++) {
            auto dst_ptr = reinterpret_cast<type *>(
                dst + i * nelem_per_warp + lane_id * nelem
            );
            if constexpr (NT) {
                __builtin_nontemporal_store(acc[i], dst_ptr);
            } else {
                *dst_ptr = acc;
            }
        }
    }
};

constexpr i32 MAX_M = 8192;
constexpr i32 MAX_N = 29568;
constexpr i32 MAX_K = 8192;
constexpr i32 MAX_M_LOCAL = MAX_M / WORLD_SIZE;

constexpr i32 CHUNK_K = 512;
constexpr i32 CHUNK_M = 256;
constexpr i32 MAX_NUM_CHUNKS_K = ceil_div(MAX_K, CHUNK_K);
constexpr i32 MAX_NUM_CHUNKS_M = ceil_div(MAX_M, CHUNK_M);

struct workspace_t {
    i32 grid_barrier;
};

// global variables
struct ipc_mem_t {
    // FIXME: reset signals
    signal_t nvl_recv_signals;
    i32 nvl_barrier[WORLD_SIZE];
};

struct ipc_cache_t {
    b16 nvl_recv_x[MAX_M * MAX_K];
};

struct global_t {
    // config
    i32 rank;
    i32 m, n, k;

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

__device__ __forceinline__ void syncwarp() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

template <typename T> constexpr T const_min(T a, T b) { return a > b ? b : a; }

template <i32 TILE_M, i32 TILE_K, i32 K, i32 NUM_DST>
__device__ inline void warp_copy_tile(b16 *(&dst)[NUM_DST], const b16 *src) {
    static_assert(TILE_K % WARP_SIZE == 0);
    constexpr i32 VEC_SIZE = TILE_K % 512 == 0 ? 16 : 2;
    using cp_t = typename ck::vector_type<int8_t, VEC_SIZE>::type;
    cp_t regs[2];

    const auto src_rsrc = ck::make_wave_buffer_resource_with_default_range(src);
    ck::int32x4_t dst_rsrc[NUM_DST];
    UNROLL
    for (int i = 0; i < NUM_DST; i++) {
        dst_rsrc[i] = ck::make_wave_buffer_resource_with_default_range(dst[i]);
    }
    const auto lane_id = threadIdx.x % WARP_SIZE;

    auto load_row = [&](int reg_idx, int row_idx) {
        const i32 soffset = row_idx * K * sizeof(b16);
        const i32 voffset = lane_id * VEC_SIZE;
        regs[reg_idx] = ck::amd_buffer_load_impl_raw<
            VEC_SIZE, ck::AmdBufferCoherenceEnum::SYSTEM_NT0>(
            src_rsrc, voffset, soffset
        );
    };
    auto store_row = [&](int reg_idx, int row_idx) {
        UNROLL
        for (int i = 0; i < NUM_DST; i++) {
            const i32 soffset = row_idx * K * sizeof(b16);
            const i32 voffset = lane_id * VEC_SIZE;
            ck::amd_buffer_store_impl_raw<
                VEC_SIZE, ck::AmdBufferCoherenceEnum::SYSTEM_NT0>(
                regs[reg_idx], dst_rsrc[i], voffset, soffset
            );
        }
    };

    auto copy_loop_body = [&](int row_idx) {
        load_row(0, row_idx + 2);
        // sync, 1 ld, 8 st in flight
        store_row(1, row_idx + 1);
        load_row(1, row_idx + 3);
        // sync, 1 ld, 8 st in flight
        store_row(0, row_idx + 2);
    };

    constexpr i32 NUM_STAGES = 2;
    constexpr i32 UNROLL_FACTOR = const_min(TILE_M / NUM_STAGES, 8);
    constexpr i32 INNER_M = UNROLL_FACTOR * NUM_STAGES;
    static_assert(TILE_M % NUM_STAGES == 0);
    static_assert(TILE_M >= INNER_M);

    load_row(0, 0);
    load_row(1, 1);
    store_row(0, 0);
    for (i32 i = 0; i < TILE_M - INNER_M; i += INNER_M) {
        asm(";main loop begin");
        UNROLL
        for (i32 j = 0; j < INNER_M; j += NUM_STAGES) {
            copy_loop_body(i + j);
        }
        asm(";main loop end");
    }
    UNROLL
    for (i32 i = TILE_M - INNER_M; i < TILE_M - NUM_STAGES; i += NUM_STAGES) {
        copy_loop_body(i);
    }
    store_row(1, TILE_M - 1);
}
struct send_args_t {
    b16 *x;
};

// push save an extra copy compared with pull
template <i32 M, i32 K, bool SYNC = true>
__global__ void send_kernel(send_args_t args, global_t global) {
    const auto num_sms = gridDim.x;
    const auto num_warps = blockDim.x / WARP_SIZE;
    const i32 num_global_warps = num_sms * num_warps;

    const auto sm_id = blockIdx.x;
    // put soffset to sgpr
    const auto warp_id =
        __builtin_amdgcn_readfirstlane(threadIdx.x / WARP_SIZE);
    const auto global_warp_id = sm_id * num_warps + warp_id;
    const auto lane_id = threadIdx.x % WARP_SIZE;

    const auto rank = global.rank;

    static_assert(M % WORLD_SIZE == 0);
    constexpr auto M_LOCAL = M / WORLD_SIZE;

    constexpr auto NUM_CHUNKS_M = ceil_div(M_LOCAL, CHUNK_M);
    constexpr auto NUM_CHUNKS_K = ceil_div(K, CHUNK_K);
    constexpr auto TAIL_CHUNK_M = M_LOCAL - (NUM_CHUNKS_M - 1) * CHUNK_M;
    constexpr auto TAIL_CHUNK_K = K - (NUM_CHUNKS_K - 1) * CHUNK_K;

    for (int i = global_warp_id; i < NUM_CHUNKS_M * NUM_CHUNKS_K * WORLD_SIZE;
         i += num_global_warps) {
        const auto dst_rank = i % WORLD_SIZE;
        const auto chunk_id = i / WORLD_SIZE;
        // TODO: maybe k first
        const auto chunk_k = chunk_id / NUM_CHUNKS_M;
        const auto chunk_m = chunk_id % NUM_CHUNKS_M;

        const auto m_begin = chunk_m * CHUNK_M;
        const auto k_begin = chunk_k * CHUNK_K;
        const auto offset = m_begin * K + k_begin;

        const auto chunk_src = args.x + offset;
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

        if constexpr (!SYNC) {
            // __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
            st_relaxed_sys(
                &global.ipc_mems[dst_rank]
                     ->nvl_recv_signals[rank * NUM_CHUNKS_M + chunk_m][chunk_k],
                1
            );
        }
    }

    if constexpr (!SYNC) {
        return;
    }

    __syncthreads();
    auto &grid_barrier = global.workspace->grid_barrier;
    if (warp_id == 0 && lane_id == 0) {
        __atomic_fetch_add(&grid_barrier, 1, __ATOMIC_RELAXED);
    }
    static_assert(WORLD_SIZE <= WARP_SIZE);
    if (global_warp_id == 0 && lane_id < WORLD_SIZE) {
        while (ld_acquire_global(&grid_barrier) != num_sms)
            ;
        st_relaxed_sys(&global.ipc_mems[lane_id]->nvl_barrier[rank], 1);
        st_release_global(&grid_barrier, 0);
        while (!ld_relaxed_sys(&global.ipc_mems[rank]->nvl_barrier[lane_id]))
            ;
        __builtin_amdgcn_wave_barrier();
        // safe to reset here because there would be a barrier after
        // custom kernel
        st_relaxed_sys(&global.ipc_mems[rank]->nvl_barrier[lane_id], 0);
    }
}

constexpr i64 pack_mnk(i32 m, i32 n, i32 k) {
    return (i64(m) << 32) | (i64(n) << 16) | i64(k);
}

// clang-format off
#define SWITCH_MNK(m, n, k, MACRO, ...) \
    switch (pack_mnk(m, n, k)) { \
        /*case pack_mnk(64, 2304, 7168): MACRO(64, 2304, 7168, ##__VA_ARGS__); break; \
        case pack_mnk(512, 1536, 4096): MACRO(512, 1536, 4096, ##__VA_ARGS__); break; \
        case pack_mnk(2048, 360, 2880): MACRO(2048, 360, 2880, ##__VA_ARGS__); break; \
        case pack_mnk(4096, 512, 4096): MACRO(4096, 512, 4096, ##__VA_ARGS__); break; \
        case pack_mnk(8192, 1792, 4096): MACRO(8192, 1792, 4096, ##__VA_ARGS__); break; \
        case pack_mnk(8192, 3696, 8192): MACRO(8192, 3696, 8192, ##__VA_ARGS__); break; \
        */case pack_mnk(2048, 3696, 8192): MACRO(2048, 3696, 8192, ##__VA_ARGS__); break; \
        default: ASSERT(false); \
    }

// bm, bn, bk, wm, wn
#define SWITCH_GEMM_MNK(m, n, k, MACRO, ...) \
    switch (pack_mnk(m, n, k)) { \
        /*case pack_mnk(64, 2304, 7168): MACRO(64, 2304, 7168, 32, 64, 64, 2, 2, ##__VA_ARGS__); break; \
        case pack_mnk(512, 1536, 4096): MACRO(512, 1536, 4096, 32, 64, 64, 2, 2, ##__VA_ARGS__); break; \
        case pack_mnk(2048, 360, 2880): MACRO(2048, 360, 2880, 128, 128, 64, 2, 2, ##__VA_ARGS__); break; \
        case pack_mnk(4096, 512, 4096): MACRO(4096, 512, 4096, 256, 128, 64, 2, 2, ##__VA_ARGS__); break; \
        case pack_mnk(8192, 1792, 4096): MACRO(8192, 1792, 4096, 256, 224, 64, 2, 2, ##__VA_ARGS__); break; \
        case pack_mnk(8192, 3696, 8192): MACRO(8192, 3696, 8192, 256, 224, 64, 2, 2, ##__VA_ARGS__); break; \
        */case pack_mnk(2048, 3696, 8192): MACRO(2048, 3696, 8192, 256, 224, 64, 2, 2, ##__VA_ARGS__); break; \
        default: ASSERT(false); \
    }
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
    }

    ~AgGemm() {
        for (auto i = 0; i < WORLD_SIZE; i++) {
            auto ipc_mem = global.ipc_mems[i];
            if (ipc_mem && i != global.rank) {
                HIP_CHECK(hipIpcCloseMemHandle(ipc_mem));
            }
            auto ipc_cache = global.ipc_caches[i];
            if (ipc_cache && i != global.rank) {
                HIP_CHECK(hipIpcCloseMemHandle(ipc_cache));
            }
        }
        auto local_mem = global.ipc_mems[global.rank];
        if (local_mem) {
            HIP_CHECK(hipFree(local_mem));
        }
        auto local_cache = global.ipc_caches[global.rank];
        if (local_cache) {
            HIP_CHECK(hipFree(local_cache));
        }
        if (global.workspace) {
            HIP_CHECK(hipFree(global.workspace));
        }
    }

    auto get_ipc_handle() -> pybind11::bytearray {
        void *ws;
        HIP_CHECK(hipMalloc(&ws, sizeof(workspace_t)));
        HIP_CHECK(hipMemset(ws, 0, sizeof(workspace_t)));
        global.workspace = reinterpret_cast<workspace_t *>(ws);

        void *ptr;
        HIP_CHECK(hipExtMallocWithFlags(
            &ptr, sizeof(ipc_mem_t), hipDeviceMallocUncached
        ));
        HIP_CHECK(hipMemset(ptr, 0, sizeof(ipc_mem_t)));
        global.ipc_mems[global.rank] = reinterpret_cast<ipc_mem_t *>(ptr);

        HIP_CHECK(hipMalloc(&ptr, sizeof(ipc_cache_t)));
        // HIP_CHECK(hipExtMallocWithFlags(
        //     &ptr, sizeof(ipc_cache_t), hipDeviceMallocUncached
        // ));
        HIP_CHECK(hipMemset(ptr, 0, sizeof(ipc_cache_t)));
        global.ipc_caches[global.rank] = reinterpret_cast<ipc_cache_t *>(ptr);

        std::vector<hipIpcMemHandle_t> handles(2);
        HIP_CHECK(hipIpcGetMemHandle(&handles[0], global.ipc_mems[global.rank])
        );
        HIP_CHECK(
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
            ASSERT(handle_buf.size() == HIP_IPC_HANDLE_SIZE * 2);
            auto handles =
                reinterpret_cast<hipIpcMemHandle_t *>(handle_buf.data());
            void *ptr;
            HIP_CHECK(hipIpcOpenMemHandle(
                &ptr, handles[0], hipIpcMemLazyEnablePeerAccess
            ));
            global.ipc_mems[i] = reinterpret_cast<ipc_mem_t *>(ptr);
            HIP_CHECK(hipIpcOpenMemHandle(
                &ptr, handles[1], hipIpcMemLazyEnablePeerAccess
            ));
            global.ipc_caches[i] = reinterpret_cast<ipc_cache_t *>(ptr);
        }
    }

    void send(torch::Tensor &x, bool sync) {
        send_args_t args{
            .x = reinterpret_cast<b16 *>(x.contiguous().data_ptr())
        };
        auto stream = at::cuda::getCurrentHIPStream().stream();
        dim3 block(256, 1, 1);
        // 1 SM ~= 8 GB/s, 16 SM ~= 40 GB/s
        dim3 grid(16, 1, 1);
        // clang-format off
        #define LAUNCH_SEND(m, n, k, sync) send_kernel<m, k, sync><<<grid, block, 0, stream>>>(args, global)
        if (sync) {
            SWITCH_MNK(global.m, global.n, global.k, LAUNCH_SEND, true)
        } else {
            SWITCH_MNK(global.m, global.n, global.k, LAUNCH_SEND, false)
        }
        // clang-format on
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
        HIP_CHECK(hipMemsetAsync(&signal, 0, sizeof(signal), stream));
    }

    auto
    perf_gemm(torch::Tensor &x, torch::Tensor &w, torch::Tensor &b, bool sync) {
        auto m = x.size(0);
        auto n = w.size(0);
        auto k = w.size(1);
        auto out = torch::empty({m, n}, x.options());

        auto x_ptr = reinterpret_cast<const bfloat16_t *>(x.const_data_ptr());
        auto w_ptr = reinterpret_cast<const bfloat16_t *>(w.const_data_ptr());
        auto b_ptr = reinterpret_cast<const bfloat16_t *>(b.const_data_ptr());
        auto o_ptr = reinterpret_cast<bfloat16_t *>(out.data_ptr());

        constexpr i32 GEMM_THREADS = 256;
        constexpr i32 GEMM_SMS = NUM_SMS;

        dim3 grid(GEMM_SMS);
        dim3 block(GEMM_THREADS);

        auto stream = at::cuda::getCurrentHIPStream().stream();
        auto signal = &global.ipc_mems[global.rank]->nvl_recv_signals;
        if (sync) {
            signal = nullptr;
        }

// TODO: tune num_gemm_sms
// clang-format off
        #define LAUNCH_PERF(m, n, k, bm, bn, bk, wm, wn) gemm_kernel<m, n, k, bm, bn, bk, GEMM_SMS, GEMM_SMS, GEMM_THREADS, wm, wn><<<grid, block, 0, stream>>>(x_ptr, w_ptr, b_ptr, o_ptr, signal)
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
        .def("send", &AgGemm::send, py::arg("x"), py::arg("sync") = true)
        .def("get_x_full", &AgGemm::get_x_full)
        .def("reset", &AgGemm::reset)
        .def(
            "perf_gemm", &AgGemm::perf_gemm, py::arg("x"), py::arg("w"),
            py::arg("b"), py::arg("sync") = true
        )
        .def("get_signal", &AgGemm::get_signal);
    m.def("ck_version", []() { return TO_STR(CK_COMMIT_ID); });
}
