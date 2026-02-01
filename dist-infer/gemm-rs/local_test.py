import torch
import os
import sys
import numpy as np
import argparse
import statistics

# Ensure the build directory is in the path
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'build'))

import gemm_rs

# Test configurations based on the provided shapes from test_perf.py
test_configs = [
    {"M": 64, "N": 7168, "K": 2304, "bias": True, "name": "64x7168x2304"},
    {"M": 512, "N": 4096, "K": 1536, "bias": True, "name": "512x4096x1536"},
    {"M": 2048, "N": 2880, "K": 360, "bias": True, "name": "2048x2880x360"},
    {"M": 4096, "N": 4096, "K": 512, "bias": True, "name": "4096x4096x512"},
    {"M": 8192, "N": 4096, "K": 1792, "bias": True, "name": "8192x4096x1792"},
    {"M": 8192, "N": 8192, "K": 3696, "bias": True, "name": "8192x8192x3696"},
]

WORLD_SIZE = 8

# Initialize GemmRS object
rs = gemm_rs.GemmRS(0, WORLD_SIZE)

def print_first_20_errors(expected, actual, test_name, rtol=1e-2, atol=1e-2):
    """Print the first 20 error values when tensors don't match"""
    expected_flat = expected.flatten()
    actual_flat = actual.flatten()
    
    # Calculate tolerances
    tolerance = atol + rtol * torch.abs(expected_flat)
    diff = torch.abs(expected_flat - actual_flat)
    error_mask = diff > tolerance
    
    error_indices = torch.nonzero(error_mask, as_tuple=False).flatten()
    
    if len(error_indices) > 0:
        print(f"    📝 {test_name} Error Details:")
        print(f"       Total errors: {len(error_indices)} out of {len(expected_flat)} elements")
        print(f"       Error rate: {len(error_indices) / len(expected_flat) * 100:.2f}%")
        print(f"    📋 First 20 errors:")
        print(f"       {'Index':<8} {'Expected':<12} {'Actual':<12} {'Diff':<12} {'RelErr%':<10}")
        print(f"       {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        
        for i, idx in enumerate(error_indices[:20]):
            idx = idx.item()
            exp_val = expected_flat[idx].item()
            act_val = actual_flat[idx].item()
            diff_val = abs(exp_val - act_val)
            rel_err = (diff_val / abs(exp_val)) * 100 if abs(exp_val) > 1e-10 else float('inf')
            
            print(f"       {idx:<8} {exp_val:<12.6f} {act_val:<12.6f} {diff_val:<12.6f} {rel_err:<10.2f}")

def elapsed_time(func: callable, repeat=100):
    """Measures the elapsed time of a function."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        func()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat

def test_gemm(x, w, b):
    """Tests the GEMM operation."""
    return rs.test_gemm(x, w, b)

def test_rs(all_ranks, target_rank):
    """Tests the RS (Reduce-Scatter) operation."""
    return rs.test_rs(all_ranks, target_rank)

def test_gemm_rs(x, w, b, all_inputs, world_size, fake_rank):
    """Tests the fused GEMM_RS operation."""
    return rs.test_gemm_rs(x, w, b, all_inputs, world_size, fake_rank)

def main():
    parser = argparse.ArgumentParser(description="Run performance tests for GEMM, RS, and GEMM_RS.")
    parser.add_argument('--gemm', action='store_true', help='Run only the GEMM test')
    parser.add_argument('--rs', action='store_true', help='Run only the RS test')
    parser.add_argument('--gemm-rs', action='store_true', help='Run only the GEMM_RS test')
    args = parser.parse_args()

    # If no specific test is selected, run all of them
    run_all = not (args.gemm or args.rs or args.gemm_rs)

    # Store execution times for geometric mean calculation
    times_rocblas_gemm = []
    times_our_gemm = []
    times_manual_rs = []
    times_our_rs = []
    times_baseline_gemm_rs = []
    times_our_gemm_rs = []
    
    # Track correctness statistics
    total_tests = 0
    gemm_matches = 0
    rs_matches = 0
    gemm_rs_matches = 0
    gemm_total = 0
    rs_total = 0
    gemm_rs_total = 0

    # Main testing loop
    for config in test_configs:
        M, N, K = config["M"], config["N"], config["K"]
        M_per_rank = M // WORLD_SIZE
        print(f"\n🧪 Testing {config['name']} (M={M}, N={N}, K={K})")

        # Create generator for reproducible results
        gen = torch.Generator(device="cuda")
        gen.manual_seed(42)

        # Generate input data
        x = (torch.rand((M, K), dtype=torch.bfloat16, device="cuda", generator=gen) * 2 - 1) * 0.01
        w = (torch.rand((N, K), dtype=torch.bfloat16, device="cuda", generator=gen) * 2 - 1) * 0.01
        b = (torch.rand((N,), dtype=torch.bfloat16, device="cuda", generator=gen) * 2 - 1) * 0.01 if config["bias"] else None
        
        if run_all or args.gemm:
            # --- rocBLAS Baseline (GEMM) ---
            rocblas_result = torch.matmul(x, w.T) 
            if b is not None:
                rocblas_result += b
            
            # --- Our GEMM Kernel ---
            try:
                # First check correctness
                our_gemm_result = test_gemm(x, w, b)
                
                # Verify results before timing
                try:
                    torch.testing.assert_close(our_gemm_result, rocblas_result, rtol=1e-2, atol=1e-2)
                    print(f"    ✅ GEMM Results match (within tolerance)")
                    gemm_matches += 1
                    
                    # Only run performance tests if correctness check passes
                    func_rocblas_gemm = lambda: torch.matmul(x, w.T)
                    elapsed_time(func_rocblas_gemm)  # Warmup
                    t_rocblas_gemm = elapsed_time(func_rocblas_gemm)
                    times_rocblas_gemm.append(t_rocblas_gemm)
                    rocblas_gemm_tflops = 2 * M * N * K / t_rocblas_gemm / 1e9
                    print(f"  - rocBLAS GEMM: {rocblas_gemm_tflops:.2f} TFLOPS ({t_rocblas_gemm * 1000:.1f} μs)")
                    
                    func_gemm = lambda: test_gemm(x, w, b)
                    elapsed_time(func_gemm)  # Warmup
                    t_gemm = elapsed_time(func_gemm)
                    times_our_gemm.append(t_gemm)
                    gemm_tflops = 2 * M * N * K / t_gemm / 1e9
                    gemm_speedup = t_rocblas_gemm / t_gemm
                    print(f"  - Our GEMM:     {gemm_tflops:.2f} TFLOPS ({t_gemm * 1000:.1f} μs) (Speedup: {gemm_speedup:.2f}x)")
                    
                except AssertionError as e:
                    print(f"    ❌ GEMM Results do not match: {str(e)}")
                    print_first_20_errors(rocblas_result, our_gemm_result, "GEMM")
                gemm_total += 1
            except Exception as e:
                print(f"  - Our GEMM:     FAILED - {str(e)}")

        if run_all or args.rs:
            torch.manual_seed(42)
            # --- RS (Reduce-Scatter) Test ---
            # Generate input for RS: simulate WORLD_SIZE ranks each with M_per_rank rows
            M_per_rank = M // WORLD_SIZE
            all_ranks = [
                torch.rand(M, N, dtype=torch.bfloat16, device="cuda", generator=gen)
                for _ in range(WORLD_SIZE)
            ]
            
            # Baseline for RS: manual reduce-scatter for rank 0
            def manual_rs():
                expected = torch.zeros((M_per_rank, N), dtype=torch.float32, device='cuda')
                for rank in range(WORLD_SIZE):
                    expected += all_ranks[rank][M_per_rank * 0: M_per_rank * (0 + 1), :].float()
                return expected.to(torch.bfloat16)

            manual_rs_result = manual_rs()
            
            # Our RS kernel (test for rank 0)
            try:
                # First check correctness
                our_rs_result = test_rs(all_ranks, 0)
                
                # Verify results before timing
                try:
                    torch.testing.assert_close(our_rs_result, manual_rs_result, rtol=1e-2, atol=1e-2)
                    print(f"    ✅ RS Results match (within tolerance)")
                    rs_matches += 1
                    
                    # Only run performance tests if correctness check passes
                    func_manual_rs = manual_rs
                    elapsed_time(func_manual_rs) # Warmup
                    t_manual_rs = elapsed_time(func_manual_rs)
                    times_manual_rs.append(t_manual_rs)
                    # TFLOPS for RS is based on (WORLD_SIZE - 1) additions for each element in the slice
                    rs_tflops_manual = (WORLD_SIZE - 1) * M_per_rank * N / t_manual_rs / 1e9
                    print(f"  - Manual RS:    {rs_tflops_manual:.2f} TFLOPS ({t_manual_rs * 1000:.1f} μs)")

                    func_rs = lambda: test_rs(all_ranks, 0)
                    elapsed_time(func_rs)  # Warmup
                    t_rs = elapsed_time(func_rs)
                    times_our_rs.append(t_rs)
                    rs_tflops = (WORLD_SIZE - 1) * M_per_rank * N / t_rs / 1e9
                    rs_speedup = t_manual_rs / t_rs
                    print(f"  - Our RS:       {rs_tflops:.2f} TFLOPS ({t_rs * 1000:.1f} μs) (Speedup: {rs_speedup:.2f}x)")
                    
                except AssertionError as e:
                    print(f"    ❌ RS Results do not match: {str(e)}")
                    print_first_20_errors(manual_rs_result, our_rs_result, "RS")
                rs_total += 1
            except Exception as e:
                print(f"  - Our RS:       FAILED - {str(e)}")

        if run_all or args.gemm_rs:
            # --- Fused GEMM_RS Test ---
            # GEMM_RS does GEMM first, then RS on the result
            # Generate input for GEMM_RS: simulate WORLD_SIZE ranks each contributing to GEMM
            
            # Create input matrices for each rank (simulating distributed GEMM inputs)
            torch.manual_seed(42)
            gen_gemm_rs = torch.Generator(device="cuda")
            gen_gemm_rs.manual_seed(42)
            
            # Generate GEMM inputs for all ranks with same range as main x, w, b
            all_inputs = []
            for rank in range(WORLD_SIZE):
                x_rank = (torch.rand((M, K), dtype=torch.bfloat16, device="cuda", generator=gen_gemm_rs) * 2 - 1) * 0.01
                w_rank = (torch.rand((N, K), dtype=torch.bfloat16, device="cuda", generator=gen_gemm_rs) * 2 - 1) * 0.01
                b_rank = (torch.rand((N,), dtype=torch.bfloat16, device="cuda", generator=gen_gemm_rs) * 2 - 1) * 0.01 if config["bias"] else None
                
                # Compute GEMM + BIAS for this rank
                gemm_result = torch.matmul(x_rank, w_rank.T)
                if b_rank is not None:
                    gemm_result += b_rank
                all_inputs.append(gemm_result)
            
            x = (torch.rand((M,K), dtype=torch.bfloat16, device="cuda", generator=gen_gemm_rs) * 2 - 1) * 0.01
            w = (torch.rand((N,K), dtype=torch.bfloat16, device="cuda", generator=gen_gemm_rs) * 2 - 1) * 0.01
            b = (torch.rand((N,), dtype=torch.bfloat16, device="cuda", generator=gen_gemm_rs) * 2 - 1) * 0.01 if config["bias"] else None
            # Create GEMM results from all ranks for the RS operation
            
            # Baseline: Manual GEMM + RS for rank 0
            def manual_gemm_rs():
                # The baseline is just doing RS on the pre-computed GEMM results for rank 0's slice
                M_per_rank = M // WORLD_SIZE
                start_idx = M_per_rank * 0  # rank 0
                end_idx = M_per_rank * (0 + 1)
                
                rs_result = torch.zeros((M_per_rank, N), dtype=torch.float32, device='cuda')
                for rank in range(WORLD_SIZE):
                    if rank == 0: # rank 0
                        res = torch.matmul(x, w.T)[start_idx:end_idx]
                        if b is not None:
                            res += b
                        rs_result += res.float()
                    else:
                        rs_result += all_inputs[rank][start_idx:end_idx, :].float()
                return rs_result.to(torch.bfloat16)
            baseline_gemm_rs_result = manual_gemm_rs()
            
            torch.cuda.synchronize()
            try:
                # First check correctness
                our_gemm_rs_result = test_gemm_rs(x, w, b, all_inputs, WORLD_SIZE, 0)
                # Verify results before timing
                try:
                    torch.testing.assert_close(our_gemm_rs_result, baseline_gemm_rs_result, rtol=1e-2, atol=1e-2)
                    print(f"    ✅ GEMM_RS Results match (within tolerance)")
                    gemm_rs_matches += 1
                    
                    # Only run performance tests if correctness check passes
                    func_baseline_gemm_rs = manual_gemm_rs
                    elapsed_time(func_baseline_gemm_rs)  # Warmup
                    t_baseline_gemm_rs = elapsed_time(func_baseline_gemm_rs)
                    times_baseline_gemm_rs.append(t_baseline_gemm_rs)
                    
                    # TFLOPS calculation: GEMM operations + RS operations
                    gemm_ops = WORLD_SIZE * 2 * M * N * K  # GEMM for all ranks
                    rs_ops = (WORLD_SIZE - 1) * (M // WORLD_SIZE) * N  # RS operations
                    total_ops = gemm_ops + rs_ops
                    gemm_rs_tflops_baseline = total_ops / t_baseline_gemm_rs / 1e9
                    print(f"  - Baseline GEMM+RS: {gemm_rs_tflops_baseline:.2f} TFLOPS ({t_baseline_gemm_rs * 1000:.1f} μs)")

                    # Reset for our kernel test
                    torch.cuda.synchronize()

                    func_gemm_rs = lambda: test_gemm_rs(x, w, b, all_inputs, WORLD_SIZE, 0)
                    elapsed_time(func_gemm_rs)  # Warmup
                    t_gemm_rs = elapsed_time(func_gemm_rs)
                    times_our_gemm_rs.append(t_gemm_rs)
                    gemm_rs_tflops = total_ops / t_gemm_rs / 1e9
                    gemm_rs_speedup = t_baseline_gemm_rs / t_gemm_rs
                    print(f"  - Our GEMM_RS:      {gemm_rs_tflops:.2f} TFLOPS ({t_gemm_rs * 1000:.1f} μs) (Speedup: {gemm_rs_speedup:.2f}x)")
                    
                except AssertionError as e:
                    print(f"    ❌ GEMM_RS Results do not match: {str(e)}")
                    print_first_20_errors(baseline_gemm_rs_result, our_gemm_rs_result, "GEMM_RS")
                gemm_rs_total += 1
            except Exception as e:
                print(f"  - Our GEMM_RS:      FAILED - {str(e)}")

    print(f"\n{'='*60}")
    print("📊 All tests completed.")
    
    # Display correctness statistics
    print(f"\n🎯 Correctness Statistics:")
    if gemm_total > 0:
        print(f"  - GEMM matches:    {gemm_matches}/{gemm_total}")
    if rs_total > 0:
        print(f"  - RS matches:      {rs_matches}/{rs_total}")
    if gemm_rs_total > 0:
        print(f"  - GEMM_RS matches: {gemm_rs_matches}/{gemm_rs_total}")
    
    # Calculate and display geometric means
    print("\n📈 Geometric Mean of Execution Times:")
    if times_rocblas_gemm:
        geom_mean_rocblas = statistics.geometric_mean(times_rocblas_gemm)
        print(f"  - rocBLAS GEMM:      {geom_mean_rocblas * 1000:.1f} μs")
    if times_our_gemm:
        geom_mean_our_gemm = statistics.geometric_mean(times_our_gemm)
        print(f"  - Our GEMM:          {geom_mean_our_gemm * 1000:.1f} μs")
    if times_manual_rs:
        geom_mean_manual_rs = statistics.geometric_mean(times_manual_rs)
        print(f"  - Manual RS:         {geom_mean_manual_rs * 1000:.1f} μs")
    if times_our_rs:
        geom_mean_our_rs = statistics.geometric_mean(times_our_rs)
        print(f"  - Our RS:            {geom_mean_our_rs * 1000:.1f} μs")
    if times_baseline_gemm_rs:
        geom_mean_baseline = statistics.geometric_mean(times_baseline_gemm_rs)
        print(f"  - Baseline GEMM+RS:  {geom_mean_baseline * 1000:.1f} μs")
    if times_our_gemm_rs:
        geom_mean_our_gemm_rs = statistics.geometric_mean(times_our_gemm_rs)
        print(f"  - Our GEMM_RS:       {geom_mean_our_gemm_rs * 1000:.1f} μs")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
    main()