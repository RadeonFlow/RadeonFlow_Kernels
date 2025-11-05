cmake_minimum_required(VERSION 3.21) # HIP language support requires 3.21
include(config.cmake)
if(TARGET_VENDOR STREQUAL "AMD")
set(GPU_TARGETS ${TARGET_GPU_ARCH} CACHE STRING "GPU architectures to build for HIP")
set(ENV{PYTORCH_ROCM_ARCH} ${TARGET_GPU_ARCH})
endif()
set(CMAKE_CXX_STANDARD 20)
cmake_policy(VERSION 3.21.3...3.27)

if(TARGET_VENDOR STREQUAL "AMD")
project(RadeonFlow LANGUAGES HIP CXX)
else()
project(RadeonFlow LANGUAGES CXX CUDA)
endif()

find_package(Torch REQUIRED PATHS ${LIBTORCH_DIR})

include_directories(include)
include_directories(${LIBTORCH_DIR}/include)

if(TARGET_VENDOR STREQUAL "AMD")
    find_package(hip)
    find_package(composable_kernel COMPONENTS device_other_operations device_gemm_operations)
    add_compile_definitions(TEST_ON_HIP)
    add_compile_definitions(HIP_ENABLE_WARP_SYNC_BUILTINS)
else()
    add_compile_definitions(TEST_ON_CUDA)
endif()

add_library(timer SHARED src/utils/timer.cpp)
if (TARGET_VENDOR STREQUAL "AMD")
target_link_libraries(timer hip::host hip::device)
else()
set_source_files_properties(src/utils/timer.cpp PROPERTIES LANGUAGE CUDA)
endif()

# Problem 1: GEMM
add_library(gemm SHARED src/gemm/gemm_launcher.cpp src/gemm/gemm_kernel.cpp)
# set_target_properties(gemm PROPERTIES COMPILE_FLAGS "--maxrregcount=256")
if (TARGET_VENDOR STREQUAL "AMD")
target_link_libraries(gemm hip::device)
else()
set_source_files_properties(src/gemm/gemm.cpp PROPERTIES LANGUAGE CUDA)
endif()

target_link_libraries(gemm timer)

target_include_directories(gemm PRIVATE include)

add_library(gemm_test tests/gemm/gemm_test.cpp)
target_include_directories(gemm_test PRIVATE tests/checker)
target_link_libraries(gemm_test PRIVATE "${TORCH_LIBRARIES}")

# Problem 2: MoE


if (TARGET_VENDOR STREQUAL "AMD")
add_library(moe SHARED src/moe/moe.cpp src/moe/moe_topk_kernel.cpp src/moe/moe_gemm_pipeline_kernel.cpp src/moe/transpose.cpp src/moe/gemm_thirdparty.cpp)
target_link_libraries(moe hip::device hip::hipcub)
set_target_properties(moe PROPERTIES COMPILE_FLAGS "--save-temps")


target_include_directories(moe PRIVATE include)
target_link_libraries(moe timer)

add_library(moe_topk_test tests/moe/moe_topk_test.cpp)
target_include_directories(moe_topk_test PRIVATE tests/checker)
target_link_libraries(moe_topk_test "${TORCH_LIBRARIES}")

endif()

# Problem 3: MLA

if (TARGET_VENDOR STREQUAL "AMD")



add_library(mla_test tests/mla/mla_test.cpp)
target_include_directories(mla_test PRIVATE tests/checker)
target_link_libraries(mla_test "${TORCH_LIBRARIES}")

endif()

# Checker

add_executable(gemm_checker tests/checker/checker.cpp)
target_link_libraries(gemm_checker PRIVATE gemm_test "${TORCH_LIBRARIES}")
set_target_properties(gemm_checker PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}"
    SKIP_BUILD_RPATH FALSE)

if ((TARGET_VENDOR STREQUAL "AMD"))
add_executable(moe_topk_checker tests/checker/checker.cpp)
target_link_libraries(moe_topk_checker PRIVATE moe_topk_test "${TORCH_LIBRARIES}")
set_target_properties(moe_topk_checker PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}"
    SKIP_BUILD_RPATH FALSE)

add_executable(mla_checker tests/checker/checker.cpp)
target_link_libraries(mla_checker PRIVATE mla_test "${TORCH_LIBRARIES}")
set_target_properties(mla_checker PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}"
    SKIP_BUILD_RPATH FALSE)

endif()



# add_executable(checker checker.cpp shapes.cpp)
# add_library(reference reference.cpp)

# target_link_libraries(checker PRIVATE "${TORCH_LIBRARIES}" reference)
# target_link_libraries(reference PRIVATE "${TORCH_LIBRARIES}")

#add_executable(exp_runner experiment/exp_runner.cpp)
# target_link_libraries(exp_runner PRIVATE "${TORCH_LIBRARIES}")


# Playground

function(add_micro_test target_name source_file)
    add_executable(${target_name} ${source_file})
    if (TARGET_VENDOR STREQUAL "AMD")
        set_source_files_properties(${source_file} PROPERTIES LANGUAGE HIP)
        target_link_libraries(${target_name} hip::device)
    else()
        set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
        set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES "89")
    endif()
endfunction()

if (TARGET_VENDOR STREQUAL "AMD")
add_micro_test(wmma playground/wmma.cpp)
add_micro_test(gemm2 playground/gemm2.cpp)
add_micro_test(micro_io playground/micro_io.cpp)
add_micro_test(micro_wmma playground/micro_wmma.cpp)
add_micro_test(wmma_layout playground/wmma_layout.cpp)
add_micro_test(wmma_layout2 playground/wmma_layout2.cpp)
add_micro_test(gemm3 src/gemm/gemm_kernel.cpp)
add_micro_test(transpose src/gemm/transpose_kernel.cpp)
# set_target_properties(gemm3 PROPERTIES COMPILE_FLAGS "--save-temps")
add_micro_test(device_info playground/device_info.cpp)
add_micro_test(sgemm2 playground/sgemm2.cpp)
add_micro_test(sgemm3 playground/sgemm3.cpp)
add_micro_test(hipcub playground/hipcub.cpp)
if (${CMAKE_HIP_ARCHITECTURES} STREQUAL "gfx1201")
# those test is not supported on gfx942
add_micro_test(expert_group_gemm playground/expert_group_gemm.cpp)
endif()
if (${CMAKE_HIP_ARCHITECTURES} STREQUAL "gfx942")
# MI300 only micro test
add_micro_test(async_copy playground/async_copy.cpp)
add_micro_test(pipeline_gemm playground/pipeline_gemm.cpp)
target_link_libraries(pipeline_gemm rocblas)
set_target_properties(pipeline_gemm PROPERTIES COMPILE_FLAGS "--save-temps")
add_micro_test(async_copy_transpose playground/async_copy_transpose.cpp)
target_link_libraries(async_copy_transpose hipblas)
set_target_properties(async_copy_transpose PROPERTIES COMPILE_FLAGS "--save-temps")
add_micro_test(async_gemm playground/async_gemm.cpp)
set_target_properties(async_gemm PROPERTIES COMPILE_FLAGS "--save-temps")
# add_micro_test(mask_io playground/mask_io.cpp)
# set_target_properties(mask_io PROPERTIES COMPILE_FLAGS "--save-temps")
add_micro_test(benchmark_blaslt playground/benchmark_blaslt.cpp)
target_link_libraries(benchmark_blaslt roc::hipblaslt hip::device)
add_micro_test(benchmark_grouped_gemm playground/benchmark_grouped_gemm.cpp)
target_link_libraries(benchmark_grouped_gemm roc::hipblaslt hip::device)

add_micro_test(hipblaslt_minitest playground/hipblaslt_minitest.cpp)
target_link_libraries(hipblaslt_minitest roc::hipblaslt hip::device)
add_micro_test(transpose_perf playground/transpose_perf.cpp)
set_target_properties(transpose_perf PROPERTIES COMPILE_FLAGS "--save-temps")
endif()
else()

# set_target_properties(wmma PROPERTIES CUDA_ARCHITECTURES "89")
endif()

# add_executable(hipcub playground/hipcub.cpp)
# set_source_files_properties(playground/hipcub.cpp PROPERTIES LANGUAGE HIP)
# target_link_libraries(hipcub hip::host hip::hipcub ${TORCH_LIBRARIES})

find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
find_package(Torch CONFIG REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH ${TORCH_INSTALL_PREFIX}/lib)
add_library(mla SHARED src/mla/mla_pybind.cpp src/mla/mla.cpp)
set_source_files_properties(mla PROPERTIES LANGUAGE HIP)
target_include_directories(mla PRIVATE ${Python3_INCLUDE_DIRS} include)
target_link_libraries(mla PRIVATE ${TORCH_PYTHON_LIBRARY} ${TORCH_LIBRARIES} hip::device timer)
set_target_properties(mla PROPERTIES PREFIX "" COMPILE_FLAGS "--save-temps")
