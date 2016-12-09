cmake_minimum_required(VERSION 2.8.9)
project(gpiu)

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src /usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
find_package(CUDA REQUIRED)

cuda_add_executable(gpiu ${CMAKE_SOURCE_DIR}/src/gpiu.cu)
# The CUDA_NVCC_FLAGS don't seem to work!
set(CUDA_NVCC_FLAGS "-Werror;-Wall;-arch=compute_52,sm_52;-code=compute_52,sm_52")

target_link_libraries(gpiu curand)
