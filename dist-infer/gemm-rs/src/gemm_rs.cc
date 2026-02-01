#include "gemm_rs_kernel.h"
#include <hip/amd_detail/amd_hip_runtime.h>
#include <pybind11/cast.h>
#include <torch/extension.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <hip/hip_runtime.h>


namespace gemm_rs {


class GemmRS {
private:
    int rank_;
    int world_size_;
    void *ipc_mems_[MAX_WORLD_SIZE];
    void *sig_buf_[MAX_WORLD_SIZE];
    
    void check_device() {
        int device;
        C10_HIP_CHECK(hipGetDevice(&device));
        TORCH_CHECK(device == rank_);
    }

public:
    GemmRS(int rank, int world_size): rank_(rank), world_size_(world_size) {
        // C10_HIP_CHECK(hipExtMallocWithFlags(&ipc_mems_[rank_], MAX_IPC_MEM_SIZE, hipDeviceMallocUncached));
        C10_HIP_CHECK(hipMalloc(&ipc_mems_[rank_], MAX_IPC_MEM_SIZE));
        C10_HIP_CHECK(hipMemset(ipc_mems_[rank_], 0, MAX_IPC_MEM_SIZE));
        sig_buf_[rank_] = reinterpret_cast<std::byte*>(ipc_mems_[rank_]) + MAX_IPC_MEM_SIZE - SIGNAL_BUF_SIZE;
    }
    ~GemmRS() {}

    pybind11::bytearray get_ipc_handle() {
        check_device();
        hipIpcMemHandle_t ipc_handle;
        C10_HIP_CHECK(hipIpcGetMemHandle(&ipc_handle, ipc_mems_[rank_]));
        return {ipc_handle.reserved, HIP_IPC_HANDLE_SIZE};
    }

    auto init_dist(const std::vector<pybind11::bytearray> &ipc_handles) {
        int world_size = ipc_handles.size();
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


    torch::Tensor gemm_rs(const torch::Tensor& input, const torch::Tensor& weight, const c10::optional<torch::Tensor>& bias) {
        int M = input.size(0);
        int K = input.size(1);
        int N = weight.size(0);
        TORCH_CHECK(K == weight.size(1), "Incompatible GEMM size");
        auto stream = at::cuda::getCurrentHIPStream();
        auto out = torch::empty({M / world_size_, N}, input.options());
        TORCH_CHECK(M * N * sizeof(at::BFloat16) + SIGNAL_BUF_SIZE <= MAX_IPC_MEM_SIZE, "Input size exceeds MAX_IPC_MEM_SIZE");
        launch_gemm_rs_dist(input.const_data_ptr(), weight.const_data_ptr(), bias ? bias->const_data_ptr() : nullptr, ipc_mems_, sig_buf_, out.mutable_data_ptr(), rank_, world_size_, M, N, K, stream);
        return out;
    }
    
    torch::Tensor test_gemm(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor &bias) {
        int M = input.size(0);
        int K = input.size(1);
        int N = weight.size(0);
        TORCH_CHECK(K == weight.size(1), "Incompatible GEMM size");
        TORCH_CHECK(M * N * sizeof(at::BFloat16) + SIGNAL_BUF_SIZE <= MAX_IPC_MEM_SIZE, "Input size exceeds MAX_IPC_MEM_SIZE");
        auto stream = at::cuda::getCurrentHIPStream();
        // C10_HIP_CHECK(hipMemsetAsync(ipc_mems_[rank_], 0, M * N * sizeof(at::BFloat16), stream));
        launch_gemm(input.const_data_ptr(), weight.const_data_ptr(), bias.const_data_ptr(), ipc_mems_[rank_], M, N, K, stream);
        auto t = torch::from_blob(ipc_mems_[rank_], {M, N}, input.options().dtype(at::kBFloat16));
        // C10_HIP_CHECK(hipStreamSynchronize(stream));
        return t;
    }

    torch::Tensor test_rs(const std::vector<torch::Tensor>& inputs, int fake_rank) {
        int M = inputs[0].size(0);
        int N = inputs[0].size(1);
        auto stream = at::cuda::getCurrentHIPStream();
        TORCH_CHECK(inputs.size() <= MAX_WORLD_SIZE, "inputs size exceeds MAX_WORLD_SIZE");
        std::vector<const void *> input_ptrs;
        for (const auto& t : inputs) {
            TORCH_CHECK(t.size(0) == M && t.size(1) == N, "All inputs must have the same shape");
            input_ptrs.push_back(t.const_data_ptr());
        }
        int world_size = inputs.size();
        auto out = torch::empty({M / world_size, N}, inputs[0].options());
        launch_rs(input_ptrs.data(), out.mutable_data_ptr(), fake_rank, world_size, M, N, stream);
        return out;
    }

    torch::Tensor test_gemm_rs(const torch::Tensor& input, const torch::Tensor& weight, std::optional<torch::Tensor> bias, const std::vector<torch::Tensor>& all_inputs, int world_size, int fake_rank) {
        int M = input.size(0);
        int K = input.size(1);
        int N = weight.size(0);
        TORCH_CHECK(K == weight.size(1), "Incompatible GEMM size");
        auto stream = at::cuda::getCurrentHIPStream();
        auto out = torch::empty({M / world_size, N}, input.options());
        std::vector<void *> rs_bufs(world_size);
        for (int i = 0; i < world_size; i++) {
            rs_bufs[i] = all_inputs[i].mutable_data_ptr();
        }
        TORCH_CHECK(M * N * sizeof(at::BFloat16) + SIGNAL_BUF_SIZE <= MAX_IPC_MEM_SIZE, "Input size exceeds MAX_IPC_MEM_SIZE");
        launch_gemm_rs(input.const_data_ptr(), weight.const_data_ptr(), bias ? bias->const_data_ptr() : nullptr, rs_bufs.data(), sig_buf_[rank_], out.mutable_data_ptr(), fake_rank, world_size, M, N, K, stream);
        return out;
    }
};

} // namespace gemm_rs

PYBIND11_MODULE(gemm_rs, m) {
    pybind11::class_<gemm_rs::GemmRS>(m, "GemmRS")
        .def(pybind11::init<int, int>(), py::arg("rank"), py::arg("world_size"))
        .def("get_ipc_handle", &gemm_rs::GemmRS::get_ipc_handle)
        .def("init_dist", &gemm_rs::GemmRS::init_dist)
        .def("gemm_rs", &gemm_rs::GemmRS::gemm_rs)
        .def("test_gemm", &gemm_rs::GemmRS::test_gemm)
        .def("test_rs", &gemm_rs::GemmRS::test_rs)
        .def("test_gemm_rs", &gemm_rs::GemmRS::test_gemm_rs);
}