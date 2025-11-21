# RadeonFlow Kernels

🏆 Grand Prize Winner Project for AMD Developer Challenge 2025

### News 🔥
- [2025/11] [Easily Build and Share ROCm Kernels with Hugging Face](https://huggingface.co/blog/build-rocm-kernels) – Discover how to easily build and share RadeonFlow_Kernels using [Hugging Face's kernel-builder](https://github.com/huggingface/kernel-builder).

- [2025/06] [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html) now provides free AMD Instinct Accelerators, you can try out our project using their MI300X.

- [2025/06] FP8-GEMM, MoE and MLA kernels are open-sourced now!

## Introduction
This project contains implementations for three GPU kernels specifically tuned for AMD Instinct MI300X Accelerator, namely FP8 Blockwise-scaled GEMM, MoE and MLA. These GPU kernels achieve high performance (at least 8x speedup for FP8 GEMM and the other kernels, compared to the reference implementaion written in PyTorch from AMD) and won the grand prize for AMD Developer Challenge 2025: Inference Sprint.

See our [Techical Report](./TechnicalReport.md) for implementation details.

## Project Structure

We implemented the FP8-GEMM and MoE kernels in HIP, and MLA kernel in PyTorch. You can find the first two kernel sources in `src` folder, and MLA kernel source in `tests` folder separately.

1. HIP Kernel Source (**for FP8-GEMM and MoE**): You can find two directories (gemm and moe) inside the `src` folder, Each for a specific kernel. The entry function is:
    * `run` from gemm.cpp for FP8-GEMM kernel.
    * `run_topk` from moe.cpp for MoE kernel.

2. PyTorch Kernel Source (**for MLA**): You can find PyTorch implementation for MLA inside the `tests` folder.

3. **Checker**: This project includes a correctness checker and performance benchmarking tool written by C++, its source can be found in `tests/checker`. For problems implemented in HIP, there is a "client" to tell the checker how to test it, whose sources can be found in `tests/gemm` and `tests/moe`.

4. **Submission and Testing Scripts**
    * Submission Generator (gen_submission.py): GPUMode requires a single .py file with PyTorch's `load_inline` feature for HIP kernel submission. This generator automatically generates the python code for submission.
    * Evaluator (eval.py): This file is taken from GPUMode's repository. the code is adjusted a little bit so that we can test all the problems locally.

    You can find the usage of these scripts in later part of this document.

5. **Playground**: We have tried a number of different implementations, techniques and workarounds while developing this project, and we have some codes for mini-tests or benchmarks. It is not part of this project, and some may use codes from the Internet without proper credit. But we have decided to leave these codes here in case you need.

## Get Started

This project relies on LibTorch for correctness checking. You can download prebuilt binaries for LibTorch on pytorch.org.

The kernels no longer supports GPUs other than MI300X (though you can see some macros and codes trying to support RDNA4), and the result can only be reproduced on ROCm 6.3.1 (find explanation and guides for upgrading ROCm version located in later parts). So make sure you are building this project on the correct hardware and software version, or the compilation would fail or the results won't be correct.

* [IMPORTANT] After extracting LibTorch, replace all the .so files in LibTorch with the libraries bundled in your PyTorch's implementation! If the version of ROCm libraries are (even slightly) different, you will very likely to run into mystrious errors (like segmentation fault and result incorrect) on MoE kernel. In addition, you may get bad performance or fail to run the kernel, since we fixed the algorithm index for hipBLASlt.

To get started, create `config.cmake` in project root (the same directory as `README.md`)

```cmake
set(TARGET_VENDOR "AMD")
set(LIBTORCH_DIR PATH/TO/YOUR/LIBTORCH)
## AMD Specific Settings
set(TARGET_GPU_ARCH "gfx942")
# The following settings are deprecated.
# list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
# add_definitions(-DTEST_ON_RDNA4)
```

After that, run the following commands to build the project:
```
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=RELEASE
$ make -j
```

## Quick Test Using C++ with Checker (Recommended)
* Configure and build using CMakeLists.txt.
* Run `./build/gemm_checker` or `./build/moe_checker`. It is expected to see output like this:

```
root@ENC1-CLS01-SVR06:~/radeon-flow$ ./build/gemm_checker
Found 18 test cases for GEMM
Benchmark mode enabled
================================
✓ All 18 test cases passed!
--------------------------------
✓ Test case 0: Best: [76.33 us, 295.40 TFLOPS], Slowest: [1775.43 us, 12.70 TFLOPS]
✓ Test case 1: Best: [42.78 us, 225.91 TFLOPS], Slowest: [43.38 us, 222.78 TFLOPS]
✓ Test case 2: Best: [45.14 us, 187.31 TFLOPS], Slowest: [46.39 us, 182.29 TFLOPS]
✓ Test case 3: Best: [20.37 us, 184.52 TFLOPS], Slowest: [22.13 us, 169.81 TFLOPS]
✓ Test case 4: Best: [75.29 us, 399.31 TFLOPS], Slowest: [89.44 us, 336.13 TFLOPS]
✓ Test case 5: Best: [136.35 us, 496.12 TFLOPS], Slowest: [140.64 us, 480.99 TFLOPS]
... (More results omitted)
--------------------------------
GeoMean - Best Time: 92.03 us, Best TFLOPS: 360.59
```

If you only want to run the kernel ONCE, add '-b' argument for checker. If you want to profile the kernel using rocprof or rocprof-compute, add '-p' argument, which would disable correctness check pass, so that kernels from PyTorch won't show up in the profiling result.

## Test Using Official Python Scripts
For GEMM and MoE
* Run `python ./scripts/gen_submission.py [gemm/moe]` to generate a `submission.py` in current directory (You can submit this in GPUMode)
* Run `python ./scripts/eval.py test --prob=[gemm/moe]` to launch functional test
* Run `python ./scripts/eval.py performance --prob=[gemm/moe]` to launch performance test
  
For MLA
* Run `make benchmark`

## Guide for Upgrading ROCm Version for the Project

The implementation of MoE relies on hipBLASlt to compute FP16 GEMM. We manually test all the algorithms provided in hipBLASlt bundled in ROCm 6.3.1, and fixed the index for them. Since the indexes of algorithms are unstable, the MoE kernel may not work with newer versions for ROCm. But the upgrade process is relatively simple:
1. Find `LaunchGroupGEMM` and `LaunchGEMM` function calls in `moe.cpp`
2. Change one of these functions to `LaunchGroupedBench` or `LaunchGEMMBench`
3. Run the checker with "-b" argument, and the best algorithm index will be shown in stdout.
4. Replace the algorithm index in `initialize_gemm_thirdparty` function of gemm_thirdparty.cpp.
5. Change the function calls back, and they are expected to be working again with newer ROCm.

## Special Thanks to 

❤️ We would like to express our heartfelt gratitude to **Prithvi Mattur** and **Bingqing Guo** from AMD for their incredible support in organizing the competition and helping us arrange our travel schedule for the awards ceremony in the US. We also extend our sincere thanks to **GPUMode** and all the organizers who made this fascinating competition possible.

We would also like to extend our heartfelt gratitude to all the mentors, organizers, and community members who provided invaluable support and guidance throughout the competition. Your encouragement and expertise were instrumental in helping us achieve our goals. Thank you!

Feel free to reach out to the project authors by opening an issue if you need help or have suggestions!
