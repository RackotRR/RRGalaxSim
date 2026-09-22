cmake_minimum_required(VERSION 3.29)
project(RRGalaxSim CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_subdirectory(extern)
add_subdirectory(lib/rrgsim_common)
add_subdirectory(lib/rrgsim_io)
add_subdirectory(lib/rrgsim_nbody)
add_subdirectory(app)