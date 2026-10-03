cmake_minimum_required(VERSION 3.29)
project(rrgsim_io LANGUAGES CXX CUDA)

# библиотека с общим кодом
add_library(${PROJECT_NAME} STATIC
    src/io_particles.cpp
    src/io_json.cpp
    src/io_handler.cpp
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
        rrgsim_tl
        nlohmann_json
    PRIVATE
        csv_wrapped
        fmt::fmt
        range-v3
)

# Увеличиваем глубину рекурсии для работы csv-parser
target_compile_options(${PROJECT_NAME}
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ftemplate-depth=256>
)