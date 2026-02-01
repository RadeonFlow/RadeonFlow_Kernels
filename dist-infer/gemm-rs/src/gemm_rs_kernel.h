#pragma once
#include <hip/hip_runtime.h>


namespace gemm_rs {

constexpr int MAX_WORLD_SIZE = 8;
constexpr size_t MAX_IPC_MEM_SIZE = 256 * (1UL << 20); // 256MB
constexpr size_t SIGNAL_BUF_SIZE = 1 * (1UL << 20); // 1MB

void launch_gemm(const void *x, const void *w, const void *b, void *out, int M, int N, int K, hipStream_t stream);

void launch_rs(const void *inputs[], void *output, int rank, int world_size, int M, int N, hipStream_t stream);

void launch_gemm_rs(const void *x, const void *w, const void *b, void *rs_buf[], void *sig_buf, void *output, int rank, int world_size, int M, int N, int K, hipStream_t stream);

void launch_gemm_rs_dist(const void *x, const void *w, const void *b, void *rs_buf[], void *sig_buf[], void *output, int rank, int world_size, int M, int N, int K, hipStream_t stream);


} // namespace gemm_rs