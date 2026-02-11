import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

from task import input_t, output_t

CUDA_SRC = r"""
{{}}
"""

import sys
import os
import time
from filelock import FileLock
from contextlib import contextmanager
import functools

os.environ.update(
    {
        "CXX": "clang++",
        "PYTORCH_ROCM_ARCH": "gfx942",
        # "NCCL_P2P_DISABLE": "1",
        # "NCCL_IB_DISABLE": "1",
        # "NCCL_SHM_DISABLE": "1",
        # "NCCL_DEBUG": "WARNING",
    }
)

# don't overwrite existing source file to avoid recompile among multiple ranks
lock_path = "all2all-compile.lock"
with FileLock(lock_path):
    with open("all2all.cu", "w") as f:
        f.write(CUDA_SRC.replace("@", chr(92)))
    os.makedirs("torch-build", exist_ok=True)
    module = load(
        name="all2all",
        sources=["all2all.cu"],
        build_directory="torch-build",
        verbose=False,
        extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20", "-O2"],
        extra_cflags=["-O2"],
    )


def print0(out: str, all=False):
    rank = dist.get_rank()
    if rank == 0 or all:
        print(f"[rank {rank}] {out}", file=sys.stderr)


def barrier():
    dist.barrier()
    torch.cuda.current_stream().synchronize()


comm = None
should_udpate_comm = True

orignal_init_pg = dist.init_process_group


def hooked_init_pg(*args, **kwargs):
    global should_update_comm
    should_update_comm = True
    ret = orignal_init_pg(*args, **kwargs)
    # print0(f"init pg: {args}, {kwargs}", True)
    return ret


dist.init_process_group = hooked_init_pg


def all_get_comm(rank, world_size, cfg):
    global comm, should_update_comm
    config = (
        rank,
        cfg.experts_per_token,
        cfg.hidden_dim,
        cfg.max_num_tokens,
        cfg.num_experts,
    )
    if should_update_comm:
        should_update_comm = False
        # clean up old comm
        del comm
        # always set device first to avoid using wrong gpu
        torch.cuda.set_device(dist.get_rank())
        # create a new comm
        print0(f"create new comm: {config}", True)
        comm = module.All2all(*config)
        ipc_handle = comm.get_ipc_handle()
        ipc_handles = [None] * world_size
        dist.all_gather_object(ipc_handles, ipc_handle)
        comm.init(ipc_handles)
        barrier()
    return comm


from reference import PyTorchAllToAll, ref_kernel


def diff_allclose(ref, other, rtol, atol, max_print=10):
    diff = torch.abs(ref - other)
    mask = diff > (atol + rtol * torch.abs(ref))
    if mask.any():
        idx = mask.nonzero(as_tuple=False)
        print0(f"{idx.shape[0]} elements mismatch", True)
        for i in range(min(max_print, idx.shape[0])):
            coord = tuple(idx[i].tolist())
            print0(
                f"  coord={coord}: ref={ref[coord].item()}, other={other[coord].item()}",
                True,
            )


def all_assert(exp: bool, err: str):
    gathered = [False] * dist.get_world_size()
    dist.all_gather_object(gathered, exp)
    assert all(gathered), err


def sort_recv_x(recv_x: torch.Tensor, n_elem: int):
    out = recv_x
    # alphabetical sort
    for d in reversed(range(recv_x.shape[2])):
        indices = out[:, :, d].argsort(dim=-1, stable=True)
        indices_exp = indices.unsqueeze(-1).expand(*out.shape)
        out = out.gather(dim=1, index=indices_exp)
    return out[:, :, :n_elem]


def check_dispatch(recv_x: torch.Tensor, ref: torch.Tensor, ref_recv_cnt: torch.Tensor):
    recv_x = recv_x.clone()
    ref = ref.clone()

    all_assert(ref.shape == recv_x.shape, "dispatch shape mismatch")
    num_local_experts, max_num_tokens, hidden_dim = recv_x.shape
    max_num_tokens /= 8

    max_token = 0
    # mask out invalid token
    for i in range(num_local_experts):
        num_tokens = ref_recv_cnt[i].item()
        max_token = max(max_token, num_tokens)
        ref[i, num_tokens:] = 0
        recv_x[i, num_tokens:] = 0
    ref = ref[:, :max_token]
    recv_x = recv_x[:, :max_token]
    # only compare the first n_elem in each token
    n_elem = hidden_dim
    ref = sort_recv_x(ref, n_elem)
    recv_x = sort_recv_x(recv_x, n_elem)
    check = torch.allclose(ref, recv_x, rtol=1e-2, atol=5e-3)
    if not check:
        diff_allclose(ref, recv_x, 1e-2, 5e-3)
        print0(f"{ref=}", True)
        print0(f"{recv_x=}", True)
    all_assert(check, "dispatch result mismatch")


def check_combine(y: torch.Tensor, ref_y: torch.Tensor, num_tokens: int):
    y = y.clone()
    ref_y = ref_y.clone()

    all_assert(y.shape == ref_y.shape, "combine shape mismatch")
    _, hidden_dim = ref_y.shape
    # only compare the first n_elem in each token
    n_elem = hidden_dim
    y = y[:num_tokens, :n_elem]
    ref_y = ref_y[:num_tokens, :n_elem]
    check = torch.allclose(y, ref_y, rtol=1e-2, atol=5e-3)
    if not check:
        diff_allclose(ref_y, y, 1e-2, 5e-3)
        print0(f"{y=}", True)
        print0(f"{ref_y=}", True)
    all_assert(check, "combine result mismatch")


@contextmanager
def host_timer():
    end = None

    def wait_for_time():
        if end is None:
            return 0.0
        return (end - start) * 1000.0

    try:
        start = time.time()
        yield wait_for_time
    finally:
        end = time.time()


def report_host_time(name=""):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with host_timer() as t:
                result = func(*args, **kwargs)
            print0(f"{name}: {t():.3f}ms", True)
            return result

        return wrapper

    return decorator


@contextmanager
def timer():
    end = None

    def wait_for_time():
        if end is None:
            return 0.0
        end.synchronize()
        return start.elapsed_time(end)

    try:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        yield wait_for_time
    finally:
        end = torch.cuda.Event(enable_timing=True)
        end.record()


def custom_kernel_test(data: input_t, check=False) -> output_t:
    cfg, rank_data, rank, world_size = data
    # step 0: always set device first to avoid using wrong gpu
    torch.cuda.set_device(rank)
    # step 1: setup communicator
    # rank could change in different testcase, so make sure to update rank in comm
    if check:
        ata = PyTorchAllToAll(cfg, rank, world_size)
    with host_timer() as t_comm:
        comm = all_get_comm(rank, world_size, cfg)
    # step 2: dispatch
    with timer() as t_dispatch:
        recv_x, recv_count = comm.dispatch(rank_data.indices, rank_data.x)
    if check:
        expert_num, ref_recv_x, expert_meta = ata.dispatch(
            rank_data.x, rank_data.indices
        )
        torch.cuda.synchronize()
        check_dispatch(recv_x, ref_recv_x, expert_num)
    # step 3: ffn
    with timer() as t_ffn:
        comm.ffn()
    if check:
        ref_recv_x = ref_recv_x.to(cfg.out_dtype) * (1 + rank)
    # step 4: combine
    with timer() as t_combine:
        y = comm.combine(rank_data.indices, rank_data.weights)
    if check:
        ref_y = torch.zeros(
            (cfg.max_num_tokens, cfg.hidden_dim),
            dtype=cfg.out_dtype,
            device=rank_data.x.device,
        )
        ata.combine(ref_y, rank_data.weights, expert_meta, ref_recv_x, expert_num)
        torch.cuda.synchronize()
        check_combine(y, ref_y, rank_data.num_tokens)
    print0(
        f"comm: {t_comm():.3f}ms, dispatch: {t_dispatch():.3f}ms, ffn: {t_ffn():.3f}ms, combine: {t_combine():.3f}ms",
        True,
    )
    return y[: rank_data.num_tokens]


# @report_host_time("custom_kernel")
def custom_kernel_bench(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    # step 1: lazy set device and setup communicator
    # rank could change in different testcase, so make sure to update rank in comm
    comm = all_get_comm(rank, world_size, cfg)
    # step 2: dispatch
    recv_x, recv_count = comm.dispatch(rank_data.indices, rank_data.x)
    # step 3: ffn & combine
    y = comm.combine(rank_data.indices, rank_data.weights)
    return y[: rank_data.num_tokens]


def custom_kernel_repeat(data: input_t, fun=custom_kernel_test) -> output_t:
    for _ in range(10):
        fun(data)
    return fun(data)


def empty_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)

    gathered = [None] * world_size
    dist.all_gather_object(gathered, should_udpate_comm)
    all_assert(all(gathered) or not any(gathered), "should_update_comm mismatch")

    comm = all_get_comm(rank, world_size, cfg)
    return ref_kernel(data)


def custom_kernel_graph(data: input_t) -> output_t:
    # warmup: set device, init comm
    # avoid stream synchronizing in capture
    torch.cuda.set_device(dist.get_rank())
    custom_kernel_bench(data)
    torch.cuda.synchronize()
    # capture in correct device
    s = torch.cuda.Stream(device=dist.get_rank())
    # this is required, don't know why though
    with torch.cuda.stream(s):
        custom_kernel_bench(data)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        custom_kernel_bench(data)
    torch.cuda.set_device(dist.get_rank())
    # warmup graph
    g.replay()
    torch.cuda.synchronize()
    with host_timer() as t_graph:
        for _ in range(1):
            g.replay()
        torch.cuda.synchronize()
    with host_timer() as t_normal:
        for _ in range(1):
            ret = custom_kernel_bench(data)
        torch.cuda.synchronize()
    # cuda graph is slightly slower
    # normal: 0.241, graph: 0.277
    print0(f"normal: {t_normal():.3f}, graph: {t_graph():.3f}")
    return ret


custom_kernel = custom_kernel_bench
