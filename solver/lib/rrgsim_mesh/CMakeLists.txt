cmake_minimum_required(VERSION 3.29)
project(rrgsim_mesh LANGUAGES CXX CUDA)

add_library(${PROJECT_NAME} STATIC
    src/m_common_mesh_solver.cu
    src/m_mesh_solver_rect.cu
    src/m_mesh_solver_sph.cu
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        rrgsim_common
)