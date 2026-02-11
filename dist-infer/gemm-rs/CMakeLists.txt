cmake_minimum_required(VERSION 3.21)
cmake_policy(VERSION 3.21.3...3.27)
set(PROJECT_NAME "gemm_rs")
project(${PROJECT_NAME} LANGUAGES HIP CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
find_package(Torch CONFIG REQUIRED)
find_package(HIP CONFIG REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH ${TORCH_INSTALL_PREFIX}/lib)
add_library(${PROJECT_NAME} SHARED src/gemm_rs.cc src/gemm_rs_kernel.cc)
set_source_files_properties(${PROJECT_NAME} PROPERTIES LANGUAGE HIP)
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_PYTHON_LIBRARY} ${TORCH_LIBRARIES} hip::device Python3::Python)
target_compile_definitions(${PROJECT_NAME} PRIVATE -D__ENABLE_LOCAL_DEBUG__)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

# perf gemm
add_library(perf_gemm SHARED src/perf_gemm.cc)
set_source_files_properties(perf_gemm PROPERTIES LANGUAGE HIP)
target_link_libraries(perf_gemm PRIVATE ${TORCH_PYTHON_LIBRARY} ${TORCH_LIBRARIES} hip::device Python3::Python)
target_compile_definitions(perf_gemm PRIVATE -D__ENABLE_LOCAL_DEBUG__)
set_target_properties(perf_gemm PROPERTIES PREFIX "")
target_compile_features(perf_gemm PRIVATE cxx_std_20)
target_compile_options(perf_gemm PRIVATE "--save-temps")

# for host compile
target_compile_definitions(${PROJECT_NAME} PRIVATE -D__${CMAKE_HIP_ARCHITECTURES}__)
# for debug
target_compile_options(${PROJECT_NAME} PRIVATE "--save-temps")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gline-tables-only")
# add_executable(test_basic_gemm src/test_basic_gemm.cc)
# target_link_libraries(test_basic_gemm PRIVATE ${TORCH_LIBRARIES} hip::device composable_kernel::device_other_operations composable_kernel::device_gemm_operations)
# add_executable(test_optimized_gemm src/test_optimized_gemm.cc)
# target_link_libraries(test_optimized_gemm PRIVATE ${TORCH_LIBRARIES} hip::device composable_kernel::device_other_operations composable_kernel::device_gemm_operations)
