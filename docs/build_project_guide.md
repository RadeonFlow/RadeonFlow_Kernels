# RadeonFlow Kernels Build Guide for ROCm 6.4.2

This guide documents the complete step-by-step process to build the RadeonFlow Kernels project on a system with ROCm 6.4.2 and AMD Instinct MI325X (gfx942 architecture).

## Prerequisites

- ROCm 6.4.2 installed on your system
- AMD Instinct MI300X or MI325X GPU (gfx942 architecture)
- Python 3.12 or later
- CMake 3.21 or later
- Standard build tools (make, gcc/clang)

## Build Steps

### Step 1: Create a Python Virtual Environment

**Why:** This project requires PyTorch with ROCm support. Using a virtual environment avoids conflicts with system packages and provides a clean, isolated environment.

```bash
cd /home/abdennacerhuggingface/RadeonFlow_Kernels
python3 -m venv rocm_env
```

### Step 2: Activate the Virtual Environment and Upgrade pip

```bash
source rocm_env/bin/activate
pip install --upgrade pip
```

### Step 3: Install PyTorch with ROCm 6.4 Support

**Why:** PyTorch provides the necessary ROCm libraries and is required for the project's correctness checking. The ROCm 6.4 version ensures compatibility with your system's ROCm installation.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
```

This installs:
- PyTorch 2.8.0+rocm6.4
- All necessary ROCm libraries (hipBLAS, hipBLASlt, MIOpen, etc.)

**Verify the installation:**

```bash
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('ROCm available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch version: 2.8.0+rocm6.4
ROCm available: True
```

### Step 4: Download LibTorch for ROCm 6.4

**Why:** The project uses LibTorch (C++ distribution of PyTorch) for building native code. We need the shared libraries version with dependencies.

```bash
cd /home/abdennacerhuggingface/RadeonFlow_Kernels
wget -c https://download.pytorch.org/libtorch/rocm6.4/libtorch-shared-with-deps-2.8.0%2Brocm6.4.zip -O libtorch-rocm6.4.zip
```

**Note:** Use `libtorch-shared-with-deps` (NOT `libtorch-cxx11-abi-shared-with-deps`) for compatibility with PyTorch wheel packages.

### Step 5: Extract LibTorch

**Why:** Extracts the LibTorch archive to the project directory.

```bash
unzip -q libtorch-rocm6.4.zip
```

This creates a `libtorch` directory containing:
- `include/` - C++ headers
- `lib/` - Shared libraries
- `share/` - CMake configuration files

### Step 6: Replace LibTorch Libraries with PyTorch Libraries [CRITICAL]

**Why:** This is the most critical step! LibTorch's bundled ROCm libraries may have slight version mismatches with your system's ROCm 6.4.2. Using PyTorch's libraries (which are already compatible with your system) prevents:
- Segmentation faults
- Incorrect computation results
- Algorithm index mismatches in hipBLASlt

```bash
# Get PyTorch library directory
PYTORCH_LIB=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
echo "PyTorch lib directory: $PYTORCH_LIB"

# Replace all .so files
cd /home/abdennacerhuggingface/RadeonFlow_Kernels/libtorch/lib
cp -v $PYTORCH_LIB/*.so* .
```

This replaces critical libraries including:
- `libhipblaslt.so` - Used by MoE kernel
- `libtorch_hip.so` - HIP runtime for PyTorch
- `libMIOpen.so` - Deep learning primitives
- All other ROCm libraries

### Step 7: Verify Configuration File

**Why:** The `config.cmake` file tells CMake where to find LibTorch and what GPU architecture to target.

Check the content of `config.cmake`:

```bash
cat config.cmake
```

It should contain:

```cmake
set(TARGET_VENDOR "AMD")
set(LIBTORCH_DIR /home/abdennacerhuggingface/RadeonFlow_Kernels/libtorch)
## AMD Specific Settings
set(TARGET_GPU_ARCH "gfx942")
set(CMAKE_HIP_ARCHITECTURES "gfx942")
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
```

**Note:** `gfx942` is for MI300X and MI325X. Adjust if you have a different GPU.

### Step 8: Fix CMakeLists.txt Duplicate Target Issue

**Why:** The original CMakeLists.txt has a duplicate definition of the `mla` library which causes CMake configuration to fail.

The issue is that `mla` library is defined twice:
1. Line 75: Basic definition for testing
2. Line 189: Full definition with Python bindings

**Fix:** Comment out the first definition since the second one is more complete.

Edit `CMakeLists.txt` and change lines 74-86 from:
```cmake
if (TARGET_VENDOR STREQUAL "AMD")
add_library(mla SHARED src/mla/mla.cpp)
target_link_libraries(mla hip::device timer)
set_target_properties(mla PROPERTIES COMPILE_FLAGS "--save-temps")

target_include_directories(mla PRIVATE include)

add_library(mla_test tests/mla/mla_test.cpp)
target_include_directories(mla_test PRIVATE tests/checker)
target_link_libraries(mla_test "${TORCH_LIBRARIES}")

endif()
```

To:
```cmake
if (TARGET_VENDOR STREQUAL "AMD")
# Commenting out this duplicate definition - the mla library is properly defined below at line 189
# add_library(mla SHARED src/mla/mla.cpp)
# target_link_libraries(mla hip::device timer)
# set_target_properties(mla PROPERTIES COMPILE_FLAGS "--save-temps")

# target_include_directories(mla PRIVATE include)

add_library(mla_test tests/mla/mla_test.cpp)
target_include_directories(mla_test PRIVATE tests/checker)
target_link_libraries(mla_test "${TORCH_LIBRARIES}")

endif()
```

Also update the second `mla` definition (around line 189) to include all necessary dependencies:

```cmake
add_library(mla SHARED src/mla/mla_pybind.cpp src/mla/mla.cpp)
set_source_files_properties(mla PROPERTIES LANGUAGE HIP)
target_include_directories(mla PRIVATE ${Python3_INCLUDE_DIRS} include)
target_link_libraries(mla PRIVATE ${TORCH_PYTHON_LIBRARY} ${TORCH_LIBRARIES} hip::device timer)
set_target_properties(mla PROPERTIES PREFIX "" COMPILE_FLAGS "--save-temps")
```

### Step 9: Create Build Directory

**Why:** CMake best practice is to use out-of-source builds to keep generated files separate from source code.

```bash
cd /home/abdennacerhuggingface/RadeonFlow_Kernels
rm -rf build  # Clean any previous build
mkdir build
```

### Step 10: Configure with CMake

**Why:** CMake reads the project configuration, finds dependencies, and generates build files (Makefiles).

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
```

**What this does:**
- Detects HIP compiler (ROCm's LLVM/Clang)
- Finds ROCm installation in `/opt/rocm`
- Locates LibTorch in the path specified in `config.cmake`
- Configures for gfx942 architecture
- Sets up build targets for:
  - `gemm` library (FP8 GEMM kernel)
  - `moe` library (MoE kernel)
  - `mla` library (MLA kernel)
  - Test executables: `gemm_checker`, `moe_topk_checker`, `mla_checker`
  - Various playground/benchmark executables

**Expected output includes:**
```
-- The HIP compiler identification is Clang 19.0.0
-- Found HIP: /opt/rocm-6.4.2
Building PyTorch for GPU arch: gfx942
ROCM_VERSION_DEV: 6.4.2
hipblaslt VERSION: 0.12.1
-- Configuring done
-- Generating done
```

### Step 11: Build the Project

**Why:** Compiles all source files into executables and libraries. Using `-j$(nproc)` parallelizes the build across all CPU cores for faster compilation.

```bash
make -j$(nproc)
```

**What gets built:**
- `libgemm.so` - FP8 GEMM kernel shared library
- `libmoe.so` - MoE kernel shared library  
- `mla.so` - MLA kernel Python module
- `gemm_checker` - GEMM test executable
- `moe_topk_checker` - MoE test executable
- `mla_checker` - MLA test executable
- Various benchmark and playground executables

**Expected output:**
```
[100%] Building HIP object ...
[100%] Linking CXX shared library libgemm.so
[100%] Linking CXX shared library libmoe.so
[100%] Linking CXX executable gemm_checker
[100%] Built target mla
```

### Step 12: Test the GEMM Kernel

**Why:** Verifies that the FP8 GEMM kernel works correctly and achieves expected performance.

```bash
cd /home/abdennacerhuggingface/RadeonFlow_Kernels
./build/gemm_checker
```

**Expected output:**
```
Found 18 test cases for GEMM
Benchmark mode enabled
=======================
✅ All 18 test cases passed!
-----------------------
✅ Test case 0: Best: [72.72 us, 310.08 TFLOPS], Slowest: [1835.84 us, 12.28 TFLOPS]
...
GeoMean - Best Time: 82.88 us, Best TFLOPS: 400.42
=======================
```

**Performance note:** The kernel achieves 400+ TFLOPS geometric mean, which is excellent for FP8 operations on MI325X.

### Step 13: Test MoE Kernel (Currently Requires Calibration)

**Why:** Tests the Mixture-of-Experts kernel. However, this will currently fail because hipBLASlt algorithm indices need recalibration for ROCm 6.4.2.

```bash
./build/moe_topk_checker
```

**Current status:** This will crash with a memory access fault because the hardcoded hipBLASlt algorithm indices are for ROCm 6.3.1.

**To fix:** Follow the recalibration process in `ROCM_6.4.2_UPGRADE_GUIDE.md` to benchmark and update algorithm indices in `src/moe/gemm_thirdparty.cpp`.

## Verification Checklist

After completing all steps, verify:

- ✅ ROCm 6.4.2 is installed and detected
- ✅ PyTorch 2.8.0+rocm6.4 is installed in virtual environment
- ✅ LibTorch libraries are replaced with PyTorch libraries
- ✅ CMake configuration succeeds without errors
- ✅ Project builds successfully (100% completion)
- ✅ `gemm_checker` runs and all tests pass
- ⚠️  `moe_topk_checker` requires algorithm calibration (see note below)
- ✅ Build artifacts exist:
  - `build/libgemm.so`
  - `build/libmoe.so`
  - `build/mla.so`
  - `build/gemm_checker`
  - `build/moe_topk_checker`
  - `build/mla_checker`

## Quick Test Commands

After building, run these commands to test kernels:

```bash
# Test GEMM kernel (should pass)
./build/gemm_checker

# Single run benchmark (no repeated runs)
./build/gemm_checker -b

# Profiling mode (disables correctness checks)
./build/gemm_checker -p
```

## Important Notes

### About MoE Kernel and ROCm 6.4.2

The MoE kernel uses hipBLASlt with hardcoded algorithm indices that were calibrated for ROCm 6.3.1. These indices are specific to the ROCm version and must be recalibrated for ROCm 6.4.2 to work correctly.

**Symptoms of uncalibrated algorithms:**
- Memory access faults
- Incorrect results
- Poor performance

**Solution:** See `ROCM_6.4.2_UPGRADE_GUIDE.md` for detailed calibration instructions.

### Why Replace LibTorch Libraries?

This step is **critical** and often overlooked. The issue is:

1. LibTorch bundles its own ROCm libraries (compiled with ROCm X.Y.Z)
2. Your system has ROCm 6.4.2 installed
3. PyTorch wheel was built against your exact ROCm version
4. Even minor version mismatches cause runtime errors

By replacing LibTorch's libraries with PyTorch's, you ensure all libraries are from the same ROCm build.

### GPU Architecture Notes

This project **only supports gfx942** architecture:
- AMD Instinct MI300X
- AMD Instinct MI325X

Check your GPU architecture:
```bash
rocminfo | grep gfx942
```

## Troubleshooting

### Issue: "Cannot find Torch" during CMake

**Solution:** Ensure `LIBTORCH_DIR` in `config.cmake` points to the correct path:
```bash
ls /home/abdennacerhuggingface/RadeonFlow_Kernels/libtorch/lib/libtorch.so
```

### Issue: HIP compiler not found

**Solution:** Ensure ROCm is in your PATH:
```bash
export PATH=/opt/rocm/bin:$PATH
```

### Issue: Build fails with "duplicate target mla"

**Solution:** Follow Step 8 to fix the CMakeLists.txt duplicate definition.

### Issue: GEMM test passes but MoE test crashes

**Solution:** This is expected - MoE kernel needs algorithm recalibration for ROCm 6.4.2. See `ROCM_6.4.2_UPGRADE_GUIDE.md`.

## Python Submissions (Optional)

For testing with Python (as required by GPUMode):

```bash
# Activate virtual environment
source rocm_env/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Generate GEMM submission
python ./scripts/gen_submission.py gemm

# Test GEMM
python ./scripts/eval.py test --prob=gemm
python ./scripts/eval.py performance --prob=gemm
```

## Summary

This build process:

1. ✅ Creates isolated Python environment
2. ✅ Installs PyTorch with ROCm 6.4 support
3. ✅ Downloads matching LibTorch
4. ✅ Replaces libraries to prevent version mismatches
5. ✅ Fixes CMake configuration issues
6. ✅ Builds all kernels and test executables
7. ✅ Successfully runs GEMM kernel tests
8. ⚠️ MoE kernel requires additional calibration

## Next Steps

After successful build:

1. ✅ GEMM kernel is ready to use
2. ⚠️ Calibrate MoE kernel algorithms (see `ROCM_6.4.2_UPGRADE_GUIDE.md`)
3. Run performance benchmarks with `-b` flag
4. Profile kernels using `rocprof` or `rocprof-compute`
5. Generate Python submissions for GPUMode

## References

- [Official README](README.md)
- [ROCm 6.4.2 Upgrade Guide](ROCM_6.4.2_UPGRADE_GUIDE.md)
- [Technical Report](TechnicalReport.md)
- [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)






