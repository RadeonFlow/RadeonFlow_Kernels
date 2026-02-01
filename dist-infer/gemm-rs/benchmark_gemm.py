import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'build'))

import perf_gemm

import torch
import time


test_configs = [
    {"M": 8192, "N": 8192, "K": 3696, "name": "8192x8192x3696"}, #warmup GPU
    {"M": 8192, "N": 3696, "K": 8192, "name": "8192x3696x8192"},
    
    {"M": 64, "N": 7168, "K": 2304, "name": "64x7168x2304"},
    {"M": 512, "N": 4096, "K": 1536, "name": "512x4096x1536"},
    {"M": 2048, "N": 2880, "K": 360, "name": "2048x2880x360"},
    {"M": 8192, "N": 4096, "K": 1792, "name": "8192x4096x1792"},
    {"M": 8192, "N": 8192, "K": 3696, "name": "8192x8192x3696"},
    
    {"M": 64, "N": 2304, "K": 7168, "name": "64x2304x7168"},
    {"M": 512, "N": 1536, "K": 4096, "name": "512x1536x4096"},
    {"M": 2048, "N": 360, "K": 2880, "name": "2048x360x2880"},
    {"M": 8192, "N": 1792, "K": 4096, "name": "8192x1792x4096"},
    {"M": 8192, "N": 3696, "K": 8192, "name": "8192x3696x8192"},



]


for config in test_configs:
    M, N, K = config["M"], config["N"], config["K"]


    mem_used = []
    def generate_data():
        torch.manual_seed(42)
        x = torch.rand((M, K), dtype=torch.bfloat16, device='cuda') * 2 - 1
        x *= 0.01
        w = torch.rand((N, K), dtype=torch.bfloat16, device='cuda') * 2 - 1
        w *= 0.01
        b = torch.randn((N,  ), dtype=torch.bfloat16, device='cuda') * 2 - 1
        b *= 0.01
        # b = torch.zeros((N,  ), dtype=torch.bfloat16, device='cuda')
        mem_used.append([x, w, b])
        return x, w, b

    x, w, b = generate_data()

    # torch.set_printoptions(threshold=torch.inf, linewidth=1000000000)
    def ref_fn(x, w, b):
        out =  torch.matmul(x, w.T)
        # mem_used.append(out)
        return out

    def our_fn(x, w, b):
        dummy_signal = torch.empty(x.shape[0], w.shape[0], dtype=torch.int32, device='cuda')
        out = perf_gemm.launch_gemm(x, w, b, dummy_signal, 0)
        # t = perf_gemm.__debug_get_workspace_tensor(M, N, 2)
        # torch.cuda.synchronize()
        # mem_used.append(out)
        # print(out.data_ptr())
        # torch.cuda.synchronize()
        # print(out.float())
        # print(t.sum(dim=0))
        # torch.set_printoptions(threshold=1000000000, linewidth=1000000000)
        # print(torch.stack([t.sum(dim=0)[:100, 1], out.float()[:100, 1]]))
        # print()
        # torch.testing.assert_close(t.sum(dim=0), out.float(), atol=1e-1, rtol=1e-1)
        # print("reduce correctness check")
        # # return out
        return out

    def clear_all_cache():
        z = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device='cuda')
        z.fill_(42)
        mem_used.append(z)
        

    def benchmark(name: str, fn: callable, repeat=100):
        records = []
        for _ in range(repeat):
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            x, w, b = generate_data()
            clear_all_cache()
            torch.cuda.synchronize()
            event_start.record()
            fn(x, w, b)
            event_end.record()
            torch.cuda.synchronize()
            records.append(event_start.elapsed_time(event_end) * 1e3)
        tot = sum(records[2:]) / len(records[2:]) # skip first two warmup
        
        tflops = 2 * M * N * K / tot / 1e6  # Corrected TFLOPS calculation
        print(f"{name}: {tot:.2f} μs {tflops:.2f} TFlops, [{', '.join('%.2f' % t for t in records[2:5])}]")



    print(f"==== {config['name']} ====")
    
    ref_t = ref_fn(x, w, b)
    our_t = our_fn(x, w, b)



    # correctness
    try:
        torch.testing.assert_close(ref_t + b, our_t, atol=1e-1, rtol=1e-2)
    except AssertionError as e:
        print(our_t)
        print(ref_t)
        print("Error:", e)

    # performace
    benchmark('ref_fn', ref_fn)
    benchmark('our_fn', our_fn)
    benchmark('ref_fn', ref_fn)
    benchmark('our_fn', our_fn)
    print("=" * 20, end='\n\n')
