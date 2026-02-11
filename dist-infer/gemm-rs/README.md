# AMD GEMM-Rs
## Build
```bash
cd /workspace/gemm-rs
source /opt/conda/bin/activate py_3.10
export PATH="/opt/ompi/bin:/opt/ucx/bin:/opt/cache/bin:/opt/rocm/llvm/bin:/opt/rocm/opencl/bin:/opt/rocm/hip/bin:/opt/rocm/hcc/bin:/opt/rocm/bin:/opt/conda/envs/py_3.10/bin:/opt/conda/bin:$PATH"
export CMAKE_HIP_ARCHITECTURES=gfx942
export PYTORCH_ROCM_ARCH=gfx942

# cmake -B build -S . -G Ninja -DTorch_DIR=/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/share/cmake/Torch/ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DAMDGPU_TARGETS=gfx942 -DCMAKE_BUILD_TYPE=Release
cmake -B build -S . -G Ninja -DTorch_DIR=/usr/local/lib/python3.12/dist-packages/torch/share/cmake/Torch/ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DAMDGPU_TARGETS=gfx942 -DCMAKE_BUILD_TYPE=Release
cmake --build build
# Local Test (single node)
python benchmark_gemm.py
python benchmark_rs.py
# Local Test (multi nodes)
python local_test.py
```
