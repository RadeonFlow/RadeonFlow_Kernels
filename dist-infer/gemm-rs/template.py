#!POPCORN leaderboard  amd-gemm-rs
import sys
import time
from task import input_t, output_t
from typing import Optional
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline
import zlib
import base64
import os
os.environ.update(
    {
        "HSA_XNACK": "0",
        "CXX": "clang++",
        "PYTORCH_ROCM_ARCH": "gfx942",
    }
)


CPP_WRAPPER = ""
CUDA_SRC = ""

module = load_inline(
    name='perf_gemm',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20", "-U__HIP_NO_HALF_OPERATORS__", "-U__HIP_NO_HALF_CONVERSIONS__", "-D__GPUMODE_BENCHMARK__"],
    extra_cflags=["-Ofast", "-ffast-math", "-march=native", "-funroll-loops", "-fomit-frame-pointer"],
)

comm = None
should_udpate_comm = True
round_trip = 0

orignal_init_pg = dist.init_process_group

def hooked_init_pg(*args, **kwargs):
    global should_update_comm
    should_update_comm = True
    ret = orignal_init_pg(*args, **kwargs)
    # print0(f"init pg: {args}, {kwargs}", True)
    return ret

def dist_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)

def dist_print_err(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs, flush=True, file=sys.stderr)
        


def all_get_comm():
    global comm, should_update_comm, round_trip
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    if should_update_comm:
        should_update_comm = False
        round_trip = 0
        # create a new comm
        world_size = dist.get_world_size()
        dist_print(f"create new comm: {rank} {world_size}")
        del comm
        comm = module.GemmRS(rank, world_size)
        ipc_handle = comm.get_ipc_handle()
        ipc_handles = [None] * world_size
        dist.all_gather_object(ipc_handles, ipc_handle)
        comm.init_dist(ipc_handles)
        dist.barrier()
        torch.cuda.synchronize()
    round_trip += 1
    return comm, round_trip

dist.init_process_group = hooked_init_pg


def ref_kernel(data: input_t) -> output_t:
    """
    Reference kernel for Gemm-ReduceScatter operation.

    Args:
        data: Tuple of (input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor])
            - input: Local input tensor of shape [M, local_K].
            - weight: Weight tensor of shape [N, local_K].
            - bias: Optional bias tensor of shape [N] or None.
    Returns:
        Tuple containing:
            - output: Resulting tensor of shape [M // world_size, N].
    """
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()
    # matmul
    output = torch.matmul(input, weight.T)
    if bias is not None:
        output = output + bias
    # reduce scatter
    rs_output = torch.empty((M // world_size, N), dtype=output.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output)
    return rs_output

def ref_kernel_debug(data: input_t) -> output_t:
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    output = torch.matmul(input, weight.T)
    if bias is not None:
        output = output + bias
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    torch.distributed.barrier()
    rs_output = torch.empty((M // world_size, N), dtype=output.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    comm, round_trip = all_get_comm()
    if round_trip < 8:
        dist_print_err(f"rank {rank} gemm time: {((t1 - t0)*1e6):.2f}μs, perf: {((M*N*local_K*2)/(t1 - t0) * 1e-12):.2f}TFlops/s")
        dist_print_err(f"shape: {M}x{N}x{local_K}, time: {((t2 - t1)*1e6):.2f}μs, perf: {((M*N*2)/(t2 - t1) * 1e-9):.2f}GB/s")
    return rs_output

def gemm_then_rs_kernel_debug(data: input_t) -> output_t:
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    comm, round_trip = all_get_comm()
    ipc_tensor = comm.get_c_tensors(M, N)
    signal_tensor = comm.get_signal_tensors()
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    output = module.launch_gemm(input, weight, bias, signal_tensor[rank], round_trip, ipc_tensor[rank])
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    if round_trip < 8:
        dist_print_err(f"shape: {M}x{N}x{local_K}, gemm time: {((t1 - t0)*1e6):.2f}μs, perf: {((M*N*local_K*2)/(t1 - t0) * 1e-12):.2f}TFlops/s")
    torch.distributed.barrier()

    output = module.launch_reduce_scatter(ipc_tensor, signal_tensor, round_trip, rank)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    if round_trip < 8:
        dist_print_err(f"shape: {M}x{N}x{local_K}, rs time: {((t2 - t1)*1e6):.2f}μs, perf: {((M*N*2)/(t2 - t1) * 1e-9):.2f}GB/s")
    return output
    rs_output = ref_kernel(data)
    if not torch.allclose(output, rs_output, atol=1e-2, rtol=1e-2):
        dist_print_err("mismatch in gemm_then_rs_kernel")
        dist_print_err("output:", output)
        dist_print_err("rs_output:", rs_output)
        diff = torch.abs(output - rs_output)
        dist_print_err("mismatch count:", torch.sum(diff > 1e-2).item())
        dist_print_err("diff:", diff)
        dist_print_err("max diff:", torch.max(diff).item())
    return rs_output

def gemm_then_rs_kernel(data: input_t) -> output_t:
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    comm, round_trip = all_get_comm()
    ipc_tensor = comm.get_c_tensors(M, N)
    signal_tensor = comm.get_signal_tensors()
    output = module.launch_gemm(input, weight, bias, signal_tensor[rank], round_trip, ipc_tensor[rank])
    output = module.launch_reduce_scatter(ipc_tensor, signal_tensor, round_trip, rank)
    return output


def gemm_rs_kernel(data: input_t) -> output_t:
    input, weight, bias = data
    M, _ = input.shape
    N = weight.shape[0]
    rank = torch.distributed.get_rank()
    comm, round_trip = all_get_comm()
    ipc_tensor = comm.get_c_tensors(M, N)
    signal_tensor = comm.get_signal_tensors()
    rs_output = module.launch_fused(input, weight, bias, ipc_tensor, signal_tensor, round_trip, rank)
    # dist_print_err(f"gemm_rs_kernel {input.shape}x{weight.shape} rank {rank} round_trip {round_trip}")
    return rs_output

def gemm_rs_kernel_debug(data: input_t) -> output_t:
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    comm, round_trip = all_get_comm()
    ipc_tensor = comm.get_c_tensors(M, N)
    signal_tensor = comm.get_signal_tensors()
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    rs_output = module.launch_fused(input, weight, bias, ipc_tensor, signal_tensor, round_trip, rank)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    torch.distributed.barrier()
    if round_trip < 8:
        dist_print_err(f"shape {M}x{local_K}x{N} ? {torch.all(bias == 0).item()} fused time: {((t1 - t0)*1e6):.2f}μs, perf: {((M*N*local_K*2)/(t1 - t0) * 1e-12):.2f}TFlops/s")
    rs_output_ref = ref_kernel(data)
    if not torch.allclose(rs_output, rs_output_ref, atol=1e-2, rtol=1e-2):
        dist_print_err("mismatch in gemm_rs_kernel")
        dist_print_err("rs_output:", rs_output)
        dist_print_err("rs_output_ref:", rs_output_ref)
        diff = torch.abs(rs_output - rs_output_ref)
        dist_print_err("mismatch count:", torch.sum(diff > 1e-2).item())
        dist_print_err("diff:", diff)
        dist_print_err("max diff:", torch.max(diff).item())
    return rs_output_ref
    

def check_implementation(data: input_t) -> output_t:
    rf_res = gemm_rs_kernel(data)
    ref_res = ref_kernel(data)
    x_shape = data[0].shape
    w_shape = data[1].shape
    
    if not torch.allclose(rf_res, ref_res, atol=1e-2, rtol=1e-2):
        dist_print_err(f"{x_shape[0]}x{x_shape[1]}x{w_shape[0]} mismatch")
        dist_print_err("rf_res:", rf_res)
        dist_print_err("ref_res:", ref_res)
        diff = torch.abs(rf_res - ref_res)
        dist_print_err("mismatch count:", torch.sum(diff > 1e-2).item())
        dist_print_err("diff:", diff)
        dist_print_err("max diff:", torch.max(diff).item())
    return ref_res 

# custom_kernel = check_implementation
# custom_kernel = gemm_rs_kernel
custom_kernel = gemm_then_rs_kernel_debug
# custom_kernel = ref_kernel
# custom_kernel = ref_kernel_debug
# custom_kernel = gemm_rs_kernel_debug