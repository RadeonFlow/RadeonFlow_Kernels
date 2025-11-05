import torch
from kernels import get_kernel

# Load the kernel from the Hub
gemm = get_kernel("kernels-community/gemm")

# Matrix dimensions (must be supported - see gemm_launcher.cpp)
M, N, K = 1024, 1536, 7168
QUANT_SIZE = 128

# Setup device
device = torch.device("cuda")

# Create inputs - kernel expects A:(K,M), B:(K,N)
A_fp32 = torch.randn(M, K, device=device)
B_fp32 = torch.randn(K, N, device=device)

# Convert to FP8
A_fp8 = A_fp32.to(torch.float8_e4m3fnuz)
B_fp8 = B_fp32.to(torch.float8_e4m3fnuz)

# Create scale factors (uniform scaling)
A_scale = torch.ones(K // QUANT_SIZE, M, device=device, dtype=torch.float32)
B_scale = torch.ones(K // QUANT_SIZE, N // QUANT_SIZE, device=device, dtype=torch.float32)

C = torch.zeros(M, N, device=device, dtype=torch.bfloat16)

# Use the kernel
result = gemm.gemm(A_fp8, B_fp8, A_scale, B_scale, C)

# Print top 5 rows and columns of the result
print("Shape of result:", result.shape)
print("Top 5 rows and columns of result:", result[:5, :5])