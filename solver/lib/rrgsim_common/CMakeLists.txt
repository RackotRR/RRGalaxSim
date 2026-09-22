cmake_minimum_required(VERSION 3.29)
project(rrgsim_common LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(CCCL REQUIRED COMPONENTS libcudacxx)

# цель для настройки общих параметров
add_library(rrgsim_common_properties INTERFACE)
target_compile_definitions(rrgsim_common_properties
    INTERFACE
        FMT_UNICODE=0
)
target_link_libraries(rrgsim_common_properties
    INTERFACE
        CUDA::cudart
        CCCL::libcudacxx
        spdlog
        RR
        rrgsim_tl
)
target_link_options(rrgsim_common_properties
    INTERFACE
        /NODEFAULTLIB:LIBCMT
)
set_target_properties(rrgsim_common_properties
    PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# библиотека с общим кодом
add_library(rrgsim_common STATIC
    src/co_device_structs.cu
    src/co_particles_context.cu
    src/co_grid_context.cu
)
target_include_directories(rrgsim_common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(rrgsim_common
    PUBLIC
        rrgsim_common_properties
)