cmake_minimum_required(VERSION 3.29)
project(rrgsim CXX CUDA)

add_executable(${PROJECT_NAME}
    src/rrgsim.cu
    src/rrgsim_log.cpp
    src/rrgsim_conservation.cu
    src/rrgsim_nbody.cu
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)


target_link_libraries(${PROJECT_NAME}
    PUBLIC
        rrgsim_common
        rrgsim_io
        rrgsim_nbody
)
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)