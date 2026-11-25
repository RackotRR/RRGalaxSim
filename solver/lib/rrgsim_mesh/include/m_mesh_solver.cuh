#pragma once
#include <memory>
#include <stdexcept>
#include <m_common_mesh_solver.cuh>
#include <m_mesh_solver_rect.cuh>
#include <m_mesh_solver_sph.cuh>

namespace rrgsim::mesh {

using sMeshSolver = std::shared_ptr<MeshSolver>;
inline sMeshSolver make_mesh_solver(
    const rrgsim::common::GridInfoVariant& grid_info
)
{
    if (auto rect_grid_info = std::get_if<rrgsim::common::RectGridInfo>(&grid_info); rect_grid_info) {
        return std::make_shared<rrgsim::mesh::rect::RectMeshSolver>();
    }
    else if (auto sph_grid_info = std::get_if<rrgsim::common::SphGridInfo>(&grid_info); sph_grid_info) {
        return std::make_shared<rrgsim::mesh::sph::SphMeshSolver>();
    }
    else {
        throw std::runtime_error{ "make_mesh_solver error: invalid grid info variant" };
    }
}

} // namespace rrgsim::mesh