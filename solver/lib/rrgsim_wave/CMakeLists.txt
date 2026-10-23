cmake_minimum_required(VERSION 3.29)
project(rrgsim_wave LANGUAGES CXX CUDA)

# библиотека с общим кодом
add_library(${PROJECT_NAME} STATIC
    src/wave.cu
    src/wave_steady_checker.cu
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