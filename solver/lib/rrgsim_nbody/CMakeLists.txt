cmake_minimum_required(VERSION 3.29)
project(rrgsim_nbody LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(CCCL REQUIRED COMPONENTS libcudacxx)

# библиотека с общим кодом
add_library(${PROJECT_NAME} STATIC
    src/nbody.cu
    src/p2mesh.cu
    src/wave.cu
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        rrgsim_common
)