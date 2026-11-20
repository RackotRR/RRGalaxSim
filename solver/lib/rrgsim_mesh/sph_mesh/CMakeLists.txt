cmake_minimum_required(VERSION 3.29)
project(rrgsim_sph_mesh LANGUAGES CXX CUDA)

# библиотека с общим кодом
add_library(${PROJECT_NAME} STATIC
    src/sm_p2mesh.cu
    src/sm_mesh2p.cu
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        rrgsim_common_mesh
)