#pragma once
#include <memory>
#include <stdexcept>
#include <cm_p2mesh.cuh>
#include <rm_p2mesh.cuh>
#include <sm_p2mesh.cuh>

namespace rrgsim::mesh {

using sP2MeshSolver = std::shared_ptr<P2MeshSolver>;
inline sP2MeshSolver make_p2mesh_solver(
    const rrgsim::common::GridInfoVariant& grid_info
)
{
    if (auto rect_grid_info = std::get_if<rrgsim::common::RectGridInfo>(&grid_info); rect_grid_info) {
        return std::make_shared<rrgsim::mesh::rect::RectP2MeshSolver>();
    }
    else if (auto sph_grid_info = std::get_if<rrgsim::common::SphGridInfo>(&grid_info); sph_grid_info) {
        return std::make_shared<rrgsim::mesh::sph::SphP2MeshSolver>();
    }
    else {
        throw std::runtime_error{ "make_p2mesh_solver error: invalid grid info variant" };
    }
}

} // namespace rrgsim::mesh