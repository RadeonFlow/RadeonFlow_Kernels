#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__
#include <hip/hip_fp16.h>

#include <ATen/ATen.h>
#include <ATen/ops/from_blob.h>
#include <c10/hip/HIPStream.h>
#include <torch/extension.h>

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <ck/utility/amd_buffer_addressing.hpp>
#include <ck/utility/amd_wave_read_first_lane.hpp>

#include "rocwmma/rocwmma.hpp"
#include "rocwmma/rocwmma_coop.hpp"

#define LOCAL_TEST

namespace mma = rocwmma;
using f16 = mma::float16_t;
using f32 = mma::float32_t;
using i32 = mma::int32_t;
using i64 = mma::int64_t;
using f8 = mma::float8_fnuz_t;

#define USE_DBG 0
#define USE_ASSERT 0

#define PRAGMA_UNROLL _Pragma("unroll")

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


#ifdef LOCAL_TEST
constexpr i32 WORLD_SIZE = 2;
constexpr i32 MAX_NUM_EXPERTS = 64;
#else
constexpr i32 WORLD_SIZE = 8;
constexpr i32 MAX_NUM_EXPERTS = 256;
#endif

constexpr i32 MAX_TOPK = 8;
constexpr i32 MAX_HIDDEN_DIM = 7168;
constexpr i32 MAX_MAX_NUM_TOKENS = 256;

constexpr i32 MAX_NUM_LOCAL_EXPERTS = MAX_NUM_EXPERTS / WORLD_SIZE;

constexpr i32 WARP_SIZE = 64;

// NOTE: 64, 1024 both result in poor performance
constexpr i32 BLOCK_SIZE = 256;
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
    __device__ static inline void warp_mul(T *dst, const T *src, f16 weight) {
        static_assert(N_ELEM % nelem_per_warp == 0);
        const auto lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int i = 0; i < N_ELEM / nelem_per_warp; i++) {
            auto src_ptr = reinterpret_cast<const type *>(
                src + i * nelem_per_warp + lane_id * nelem
            );
            auto val = __builtin_nontemporal_load(src_ptr);
            PRAGMA_UNROLL
            for (int j = 0; j < nelem; j++) {
                val[j] *= weight;
            }
            auto dst_ptr = reinterpret_cast<const type *>(
                dst + i * nelem_per_warp + lane_id * nelem
            );
            __builtin_nontemporal_store(val, dst_ptr);
        }
    }

    template <int N_ELEM> using accum_type = type[N_ELEM / (WARP_SIZE * nelem)];

    template <int N_ELEM>
    __device__ static inline void
    warp_accum(accum_type<N_ELEM> &acc, const T *src, f32 weight) {
        static_assert(N_ELEM % (WARP_SIZE * nelem) == 0);
        const auto lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int i = 0; i < N_ELEM / (WARP_SIZE * nelem); i++) {
            auto ptr = reinterpret_cast<const type *>(
                src + i * (WARP_SIZE * nelem) + lane_id * nelem
            );
            auto val = __builtin_nontemporal_load(ptr);
            PRAGMA_UNROLL
            for (int j = 0; j < nelem; j++) {
                // order required to maintain precision
                acc[i][j] += val[j] * weight;
            }
        }
    }

    template <int N_ELEM>
    __device__ static inline void
    warp_accum_store(T *dst, accum_type<N_ELEM> &acc) {
        static_assert(N_ELEM % (WARP_SIZE * nelem) == 0);
        const auto lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int i = 0; i < N_ELEM / (WARP_SIZE * nelem); i++) {
            auto ptr = reinterpret_cast<const type *>(
                dst + i * (WARP_SIZE * nelem) + lane_id * nelem
            );
            __builtin_nontemporal_store(acc[i], ptr);
        }
    }
};

struct dispatch_args_t {
    i32 num_tokens;
    // [num_tokens, num_experts]
    i32 *topk_idx;
    // [num_tokens, hidden_dim]
    f16 *x;
};

struct combine_args_t {
    i32 num_tokens;
    // [num_tokens, num_topk]
    i32 *topk_idx;
    // [num_tokens, num_topk]
    f32 *topk_weight;
};

struct workspace_t {
    i32 grid_barrier;
    i32 barrier_flag;

    // [max_num_tokens, topk]
    i32 nvl_dst_idxs[MAX_MAX_NUM_TOKENS][MAX_TOPK];

    // [max_num_tokens, hidden_dim]
    f16 nvl_y[MAX_MAX_NUM_TOKENS][MAX_HIDDEN_DIM];
};

// global variables
struct ipc_mem_t {
    // [num_local_experts, max_num_tokens, hidden_dim]
    f16 nvl_recv_x[MAX_NUM_LOCAL_EXPERTS][WORLD_SIZE * MAX_MAX_NUM_TOKENS]
                  [MAX_HIDDEN_DIM];
    // [num_local_experts]
    i32 nvl_recv_count[MAX_NUM_LOCAL_EXPERTS];

    // we need 2 barriers to do ping-pong
    // so that next barrier would not overwrite previous one before all ranks
    // pass the previous barrier
    i32 nvl_barrier[2][WORLD_SIZE];

    i32 nvl_signal[WORLD_SIZE];
    i32 barrier;
};
struct global_t {
    // config
    i32 rank;
    i32 num_experts;
    i32 topk;
    i32 hidden_dim;
    i32 max_num_tokens;

    // buffers
    ipc_mem_t *ipc_mems[WORLD_SIZE] = {};
    workspace_t *workspace;

    i32 next_signal;
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

// timeout 500 ms
__device__ inline void barrier(
    const global_t &global, const i32 lane_id, const i32 warp_id,
    const i32 global_warp_id, const i32 num_sms, const i32 barrier_idx,
    const i64 timeout = 1 << 30
) {
    const auto rank = global.rank;

    // 0 is not a valid flag
    constexpr i32 barrier_flag = 1;

    __syncthreads();

    if (warp_id == 0 && lane_id == 0) {
        // FIXME: look like relaxed is fine
        __hip_atomic_fetch_add(
            &global.workspace->grid_barrier, 1, __ATOMIC_RELAXED,
            __HIP_MEMORY_SCOPE_AGENT
        );
    }
    // TODO: use the last atomic add warp instead of first global warp
    static_assert(WORLD_SIZE < WARP_SIZE);
    if (global_warp_id == 0) {
        if (lane_id < WORLD_SIZE) {
            auto start = clock64();
            // TODO: maybe relaxed
            while (ld_acquire_global(&global.workspace->grid_barrier) != num_sms
            ) {
                if (clock64() - start > timeout) {
                    DBG("grid barrier timeout, expected %d, actual %d", num_sms,
                        global.workspace->grid_barrier);
                }
            }
            // here we must use acquire/release to sync multiple l2s
            st_release_global(&global.workspace->grid_barrier, 0);
            // reset ping-pong nvl barrier flag
            st_relaxed_sys(
                &global.ipc_mems[rank]->nvl_barrier[!barrier_idx][lane_id], 0
            );
            // notify other ranks
            st_relaxed_sys(
                &global.ipc_mems[lane_id]->nvl_barrier[barrier_idx][rank],
                barrier_flag
            );
        }
    }
    // first warp of each SM waits
    if (warp_id == 0 && lane_id < WORLD_SIZE) {
        auto start = clock64();
        while (ld_relaxed_sys(
                   &global.ipc_mems[rank]->nvl_barrier[barrier_idx][lane_id]
               ) != barrier_flag) {
            if (clock64() - start > timeout) {
                DBG("nvl barrier timeout, expected %d, actual %d", barrier_flag,
                    global.ipc_mems[rank]->nvl_barrier[barrier_idx][lane_id]);
            }
        }
    }
    __syncthreads();
}

__global__ __launch_bounds__(BLOCK_SIZE
) void dispatch_kernel(const dispatch_args_t args, const global_t global) {
    const auto num_sms = NUM_SMS;
    const auto num_warps = BLOCK_SIZE / WARP_SIZE;
    const auto num_global_warps = num_sms * num_warps;

    const auto sm_id = blockIdx.x;
    const auto warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP_SIZE);
    const auto global_warp_id = sm_id * num_warps + warp_id;
    const auto lane_id = threadIdx.x % WARP_SIZE;

    const auto rank = global.rank;
    const auto hidden_dim = global.hidden_dim;
    const auto num_topk = global.topk;

    const auto num_local_experts = global.num_experts / WORLD_SIZE;
    const auto num_experts = global.num_experts;

    const auto num_tokens = args.num_tokens;
    const auto x = args.x;
    const auto topk_idxs = args.topk_idx;

    // step 1: send data
    // TODO: split token to chunks
    for (int i = global_warp_id; i < num_tokens * num_topk;
         i += num_global_warps) {
        const auto token_idx = i / num_topk;
        const auto topk_idx = i % num_topk;
        const auto expert_idx = topk_idxs[token_idx * num_topk + topk_idx];
        const auto dst_rank = expert_idx / num_local_experts;
        const auto dst_local_expert_idx = expert_idx % num_local_experts;

        // atomic count
        // 20 us
        auto dst_token_idx = 0;
        if (lane_id == 0) {
            dst_token_idx = atomicAdd(
                &global.ipc_mems[dst_rank]
                     ->nvl_recv_count[dst_local_expert_idx],
                1
            );
            global.workspace->nvl_dst_idxs[token_idx][topk_idx] = dst_token_idx;
        }
        dst_token_idx = __shfl(dst_token_idx, 0);

        // copy x to dst_rank's nvl_recv_x
        f16 *src_token = x + token_idx * hidden_dim;
        f16 *dst_token = global.ipc_mems[dst_rank]
                             ->nvl_recv_x[dst_local_expert_idx][dst_token_idx];
        // NOTE: ~30 us
        using cp_t = vec_t<16, f16>;
        ASSERT(hidden_dim % cp_t::nelem == 0);
        for (int j = lane_id * cp_t::nelem; j < hidden_dim;
             j += WARP_SIZE * cp_t::nelem) {
            cp_t::copy(dst_token + j, src_token + j);
        }
    }

    // TODO: maybe use per-dst-expert fine-grained sync

    // step 2: wait warps to finish
    // NOTE: 40-90 us
    // barrier(global, lane_id, warp_id, global_warp_id, num_sms, 0);

    // 1us
    __syncthreads();
    if (warp_id == 0 && lane_id == 0) {
        // FIXME: look like relaxed is fine
        __hip_atomic_fetch_add(
            &global.ipc_mems[rank]->barrier, 1, __ATOMIC_RELAXED,
            __HIP_MEMORY_SCOPE_AGENT
        );
    }
    
    if (global_warp_id == 0 && lane_id < WORLD_SIZE) {
        while (ld_relaxed_sys(&global.ipc_mems[rank]->barrier) != num_sms)
            ;
        st_relaxed_sys(&global.ipc_mems[lane_id]->nvl_signal[rank], global.next_signal);
        st_relaxed_sys(&global.ipc_mems[rank]->barrier, 0);
        __builtin_amdgcn_wave_barrier();
        while (ld_relaxed_sys(&global.ipc_mems[rank]->nvl_signal[lane_id]) != global.next_signal)
            ;
    }
    // no further code beyond barrier
}

template <i32 HIDDEN_DIM>
__global__ __launch_bounds__(BLOCK_SIZE
) void combine_kernel(combine_args_t args, global_t global) {
    const auto num_sms = NUM_SMS;
    const auto num_warps = BLOCK_SIZE / WARP_SIZE;
    const auto num_global_warps = num_sms * num_warps;

    const auto sm_id = blockIdx.x;
    const auto warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP_SIZE);
    const auto global_warp_id = sm_id * num_warps + warp_id;
    const auto lane_id = threadIdx.x % WARP_SIZE;

    const auto rank = global.rank;
    const auto num_topk = global.topk;
    const auto num_local_experts = global.num_experts / WORLD_SIZE;
    const auto num_experts = global.num_experts;

    const auto num_tokens = args.num_tokens;
    const auto topk_idxs = args.topk_idx;
    const auto topk_weight = args.topk_weight;

    constexpr i32 WARP_HIDDEN_DIM = 1024;
    constexpr auto NUM_TOKEN_CHUNKS = ceil_div(HIDDEN_DIM, WARP_HIDDEN_DIM);
    constexpr auto LAST_CHUNK_LEN =
        HIDDEN_DIM - (NUM_TOKEN_CHUNKS - 1) * WARP_HIDDEN_DIM;

    // step 0: reset before barrier to avoid got overwritten by other ranks
    // this is unsafe, need a barrier after combine
    if (global_warp_id == 0) {
        static_assert(WARP_SIZE >= MAX_NUM_LOCAL_EXPERTS);
        if (lane_id < num_local_experts) {
            st_relaxed_sys(&global.ipc_mems[rank]->nvl_recv_count[lane_id], 0);
        }
    }

    // barrier(global, lane_id, warp_id, global_warp_id, num_sms, 1);

    // step 1: send data
    for (int i = global_warp_id; i < num_tokens * NUM_TOKEN_CHUNKS;
         i += num_global_warps) {
        const auto token_idx = i / NUM_TOKEN_CHUNKS;
        const auto chunk_idx = i % NUM_TOKEN_CHUNKS;

        f16 *dst_chunk =
            global.workspace->nvl_y[token_idx] + chunk_idx * WARP_HIDDEN_DIM;

        if (LAST_CHUNK_LEN < WARP_HIDDEN_DIM &&
            chunk_idx == NUM_TOKEN_CHUNKS - 1) {
            // last unfull chunk
            using cp_t = vec_t<2, f16>;
            cp_t::accum_type<LAST_CHUNK_LEN> acc = {};

            for (int j = 0; j < num_topk; j++) {
                const auto topk_idx = j;
                const auto expert_idx =
                    topk_idxs[token_idx * num_topk + topk_idx];
                const auto dst_rank = expert_idx / num_local_experts;
                const auto dst_local_expert_idx =
                    expert_idx % num_local_experts;

                const auto dst_token_idx =
                    global.workspace->nvl_dst_idxs[token_idx][topk_idx];

                f32 weight = topk_weight[token_idx * num_topk + topk_idx];
                f16 *src_chunk =
                    global.ipc_mems[dst_rank]
                        ->nvl_recv_x[dst_local_expert_idx][dst_token_idx] +
                    chunk_idx * WARP_HIDDEN_DIM;

                cp_t::warp_accum<LAST_CHUNK_LEN>(acc, src_chunk, weight * (dst_rank + 1));
            }
            cp_t::warp_accum_store<LAST_CHUNK_LEN>(dst_chunk, acc);
        } else {
            // full chunk
            using cp_t = vec_t<16, f16>;
            cp_t::accum_type<WARP_HIDDEN_DIM> acc = {};

            for (int j = 0; j < num_topk; j++) {
                const auto topk_idx = j;
                const auto expert_idx =
                    topk_idxs[token_idx * num_topk + topk_idx];
                const auto dst_rank = expert_idx / num_local_experts;
                const auto dst_local_expert_idx =
                    expert_idx % num_local_experts;

                const auto dst_token_idx =
                    global.workspace->nvl_dst_idxs[token_idx][topk_idx];

                f32 weight = topk_weight[token_idx * num_topk + topk_idx];
                f16 *src_chunk =
                    global.ipc_mems[dst_rank]
                        ->nvl_recv_x[dst_local_expert_idx][dst_token_idx] +
                    chunk_idx * WARP_HIDDEN_DIM;

                cp_t::warp_accum<WARP_HIDDEN_DIM>(acc, src_chunk, weight * (dst_rank + 1));
            }
            cp_t::warp_accum_store<WARP_HIDDEN_DIM>(dst_chunk, acc);
        }
    }
}

template <i32 HIDDEN_DIM>
__global__ __launch_bounds__(BLOCK_SIZE) void ffn_kernel(global_t global) {
    const auto num_sms = gridDim.x;
    const auto num_warps = blockDim.x / WARP_SIZE;
    const i32 num_global_warps = num_sms * num_warps;

    const auto sm_id = blockIdx.x;
    const auto warp_id = threadIdx.x / WARP_SIZE;
    const auto global_warp_id = sm_id * num_warps + warp_id;
    const auto lane_id = threadIdx.x % WARP_SIZE;

    const auto rank = global.rank;
    const auto num_topk = global.topk;
    const auto num_local_experts = global.num_experts / WORLD_SIZE;
    const auto num_experts = global.num_experts;

    auto recv_x = global.ipc_mems[rank]->nvl_recv_x;
    auto recv_count = global.ipc_mems[rank]->nvl_recv_count;

    // divide warps into num_local_experts groups, each group handles one
    // local_expert_idx
    const auto num_groups = num_local_experts;
    const auto num_warps_per_group = ceil_div(num_global_warps, num_groups);
    const i32 group_id = global_warp_id / num_warps_per_group;
    const auto warp_id_per_group = global_warp_id % num_warps_per_group;

    const auto local_expert_idx = group_id;
    const auto num_tokens = recv_count[local_expert_idx];

    // consider the last group
    const auto num_warps_in_group = std::min(
        num_warps_per_group, num_global_warps - group_id * num_warps_per_group
    );
    for (int i = warp_id_per_group; i < num_tokens; i += num_warps_in_group) {
        // handle one token
        using cp_t = vec_t<16, f16>;
        constexpr auto LAST_HIDDEN_DIM = HIDDEN_DIM % cp_t::nelem_per_warp;

        auto token = recv_x[local_expert_idx][i];
        cp_t::warp_mul<HIDDEN_DIM - LAST_HIDDEN_DIM>(token, token, rank + 1);

        // handle remaining part
        if (LAST_HIDDEN_DIM > 0) {
            auto token =
                recv_x[local_expert_idx][i] + HIDDEN_DIM - LAST_HIDDEN_DIM;
            vec_t<2, f16>::warp_mul<LAST_HIDDEN_DIM>(token, token, rank + 1);
        }
    }
}

// clang-format off
#define SWITCH_HIDDEN(hidden, MACRO) \
    switch (hidden) { \
        case 2048: MACRO(2048); break; \
        case 2880: MACRO(2880); break; \
        case 4096: MACRO(4096); break; \
        case 6144: MACRO(6144); break; \
        case 7168: MACRO(7168); break; \
    }
// clang-format on

class All2all {
  private:
    global_t global{};

  public:
    All2all(
        int rank, int topk, int hidden_dim, int max_num_tokens, int num_experts
    ) {
        global.rank = rank;
        global.topk = topk;
        global.hidden_dim = hidden_dim;
        global.max_num_tokens = max_num_tokens;
        global.num_experts = num_experts;

        global.next_signal = 0;
    }

    ~All2all() {
        for (auto i = 0; i < WORLD_SIZE; i++) {
            auto ipc_mem = global.ipc_mems[i];
            if (ipc_mem && i != global.rank) {
                HIP_CHECK(hipIpcCloseMemHandle(ipc_mem));
            }
        }
        auto local_mem = global.ipc_mems[global.rank];
        if (local_mem) {
            HIP_CHECK(hipFree(local_mem));
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

        hipIpcMemHandle_t ipc_handle;
        HIP_CHECK(hipIpcGetMemHandle(&ipc_handle, ptr));
        return {ipc_handle.reserved, HIP_IPC_HANDLE_SIZE};
    }

    auto init(const std::vector<pybind11::bytearray> &ipc_handles) {
        for (int i = 0; i < WORLD_SIZE; i++) {
            if (i == global.rank) {
                continue;
            }
            hipIpcMemHandle_t handle;
            auto handle_buf = std::string(ipc_handles[i]);
            ASSERT(handle_buf.size() == HIP_IPC_HANDLE_SIZE);
            std::memcpy(
                handle.reserved, handle_buf.data(), HIP_IPC_HANDLE_SIZE
            );
            void *ptr;
            HIP_CHECK(
                hipIpcOpenMemHandle(&ptr, handle, hipIpcMemLazyEnablePeerAccess)
            );
            global.ipc_mems[i] = reinterpret_cast<ipc_mem_t *>(ptr);
        }
    }

    auto dispatch(torch::Tensor &topk_idx, torch::Tensor &x) {
        dispatch_args_t args{
            .num_tokens = static_cast<i32>(topk_idx.size(0)),
            .topk_idx = topk_idx.contiguous().data_ptr<i32>(),
            // torch doesn't support data_ptr<f16>()
            .x = reinterpret_cast<f16 *>(x.contiguous().data_ptr()),
        };

        global.next_signal++;

        auto stream = at::cuda::getCurrentHIPStream().stream();
        dim3 block(BLOCK_SIZE, 1, 1);
        dim3 grid(NUM_SMS, 1, 1);
        dispatch_kernel<<<grid, block, 0, stream>>>(args, global);
        // HIP_CHECK(hipStreamSynchronize(global.stream));

        auto recv_x = torch::from_blob(
            global.ipc_mems[global.rank]->nvl_recv_x,
            {global.num_experts / WORLD_SIZE,
             global.max_num_tokens * WORLD_SIZE, global.hidden_dim},
            {WORLD_SIZE * MAX_MAX_NUM_TOKENS * MAX_HIDDEN_DIM, MAX_HIDDEN_DIM, 1
            },
            torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)
        );
        auto recv_count = torch::from_blob(
            global.ipc_mems[global.rank]->nvl_recv_count,
            {global.num_experts / WORLD_SIZE},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)
        );
        return std::pair{recv_x, recv_count};
    }

    auto combine(torch::Tensor &topk_idx, torch::Tensor &topk_weight) {
        combine_args_t args{
            .num_tokens = static_cast<i32>(topk_idx.size(0)),
            .topk_idx = topk_idx.contiguous().data_ptr<i32>(),
            .topk_weight = topk_weight.contiguous().data_ptr<f32>(),
        };

        auto stream = at::cuda::getCurrentHIPStream().stream();
        dim3 block(BLOCK_SIZE, 1, 1);
        dim3 grid(NUM_SMS, 1, 1);
        // clang-format off
        #define LAUNCH_COMBINE(hidden) combine_kernel<(hidden)><<<grid, block, 0, stream>>>(args, global)
        SWITCH_HIDDEN(global.hidden_dim, LAUNCH_COMBINE)
        // clang-format on
        // HIP_CHECK(hipStreamSynchronize(global.stream));

        auto y = torch::from_blob(
            global.workspace->nvl_y, {global.max_num_tokens, global.hidden_dim},
            {MAX_HIDDEN_DIM, 1},
            torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)
        );
        return y;
    }

    // NOTE: 6-30 us
    void ffn() {
        auto stream = at::cuda::getCurrentHIPStream().stream();
        dim3 block(BLOCK_SIZE, 1, 1);
        dim3 grid(NUM_SMS, 1, 1);
        // clang-format off
        #define LAUNCH_FFN(hidden) ffn_kernel<hidden><<<grid, block, 0, stream>>>(global)
        SWITCH_HIDDEN(global.hidden_dim, LAUNCH_FFN)
        // clang-format on
    }
};

PYBIND11_MODULE(all2all, m) {
    py::class_<All2all>(m, "All2all")
        .def(py::init<int, int, int, int, int>())
        .def("get_ipc_handle", &All2all::get_ipc_handle)
        .def("init", &All2all::init)
        .def("dispatch", &All2all::dispatch)
        .def("combine", &All2all::combine)
        .def("ffn", &All2all::ffn);
}
