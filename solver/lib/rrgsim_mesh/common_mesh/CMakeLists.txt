cmake_minimum_required(VERSION 3.29)
project(rrgsim_common_mesh LANGUAGES CXX CUDA)

# библиотека с общим кодом
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(${PROJECT_NAME}
    INTERFACE
        rrgsim_common
)