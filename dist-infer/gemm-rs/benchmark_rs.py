import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'build'))

import perf_gemm

import torch
import time


WORLD_SIZE = 8

test_configs = [
    # {"M": 64, "N": 7168, "name": "64x7168"},
    # {"M": 512, "N": 4096, "name": "512x4096"},
    # {"M": 2048, "N": 2880, "name": "2048x2880"},
    # {"M": 4096, "N": 4096, "name": "4096x4096"},
    {"M": 8192, "N": 4096, "name": "8192x4096"},
    # {"M": 8192, "N": 8192, "name": "8192x8192"},
]


for config in test_configs:
    M, N = config["M"], config["N"]
    assert M % WORLD_SIZE == 0, f"M={M} must be divisible by WORLD_SIZE={WORLD_SIZE}"

    mem_used = []
    
    def generate_data():
        c_list = []
        for i in range(WORLD_SIZE):
            c = torch.randn((M, N), dtype=torch.bfloat16, device='cuda') * 2 - 1
            c_list.append(c)
            mem_used.append(c)
        return c_list

    def ref_fn(c_list, rank):
        # Reference implementation: sum all tensors and extract the rank's slice
        stacked = torch.stack(c_list, dim=0)  # [WORLD_SIZE, M, N]
        summed = torch.sum(stacked, dim=0)    # [M, N]
        M_per_rank = M // WORLD_SIZE
        out = summed[rank * M_per_rank : (rank + 1) * M_per_rank, :]
        mem_used.append(out)
        return out

    def our_fn(c_list, rank):
        dummy_signal = [torch.zeros(c_list[0].shape[0], c_list[0].shape[1], dtype=torch.int32, device='cuda')] * WORLD_SIZE
        out = perf_gemm.launch_reduce_scatter(c_list, dummy_signal, 0, rank)
        mem_used.append(out)
        return out

    def clear_all_cache():
        z = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device='cuda')
        z.fill_(42)
        mem_used.append(z)

    def benchmark(name: str, fn: callable, rank: int, repeat=5):
        records = []
        for _ in range(repeat):
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            c_list = generate_data()
            clear_all_cache()
            torch.cuda.synchronize()
            event_start.record()
            fn(c_list, rank)
            event_end.record()
            torch.cuda.synchronize()
            records.append(event_start.elapsed_time(event_end) * 1e3)
        
        tot = sum(records[2:]) / len(records[2:])  # skip first two warmup
        
        # Calculate bandwidth
        # Read: WORLD_SIZE tensors of size M * N * 2 bytes (bfloat16)
        # Write: 1 tensor of size (M / WORLD_SIZE) * N * 2 bytes
        bytes_read = WORLD_SIZE * M * N * 2
        bytes_write = (M // WORLD_SIZE) * N * 2
        total_bytes = bytes_read + bytes_write
        bandwidth_gb_s = total_bytes / tot / 1e3  # Convert to GB/s (μs -> s, bytes -> GB)
        
        print(f"{name}: {tot:.2f} μs {bandwidth_gb_s:.2f} GB/s, [{', '.join('%.2f' % t for t in records)}]")

    # Test for rank 0
    rank = 0
    c_list = generate_data()
    
    ref_t = ref_fn(c_list, rank)
    our_t = our_fn(c_list, rank)

    print(f"==== {config['name']} (rank={rank}) ====")
    
    # Correctness check
    try:
        torch.testing.assert_close(ref_t, our_t, atol=1e-2, rtol=1e-2)
        print("✓ Correctness check passed")
    except AssertionError as e:
        print(ref_t)
        print(our_t)
        print("✗ Correctness check failed:", e)

    # Performance benchmark
    benchmark('ref_fn', ref_fn, rank)
    benchmark('our_fn', our_fn, rank)
    benchmark('ref_fn', ref_fn, rank)
    benchmark('our_fn', our_fn, rank)
    print("=" * 20, end='\n\n')
