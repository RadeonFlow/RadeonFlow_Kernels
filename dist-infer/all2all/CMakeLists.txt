cmake_minimum_required(VERSION 3.21)
cmake_policy(VERSION 3.21.3...3.27)
set(PROJECT_NAME "all2all")
project(${PROJECT_NAME} LANGUAGES HIP CXX)

find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
find_package(Torch CONFIG REQUIRED)
find_package(HIP CONFIG REQUIRED)

# required for python binding
find_library(TORCH_PYTHON_LIBRARY torch_python PATH ${TORCH_INSTALL_PREFIX}/lib)

add_library(${PROJECT_NAME} SHARED all2all.cpp)
set_source_files_properties(all2all.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_PYTHON_LIBRARY} ${TORCH_LIBRARIES} hip::device Python3::Python)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

# for host compile
target_compile_definitions(${PROJECT_NAME} PRIVATE -D__${CMAKE_HIP_ARCHITECTURES}__)
