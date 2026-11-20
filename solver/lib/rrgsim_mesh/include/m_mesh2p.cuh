#pragma once
#include <memory>
#include <stdexcept>
#include <cm_mesh2p.cuh>
#include <rm_mesh2p.cuh>
#include <sm_mesh2p.cuh>

namespace rrgsim::mesh {

using sMesh2PSolver = std::shared_ptr<Mesh2PSolver>;
inline sMesh2PSolver make_mesh2p_solver(
    const rrgsim::common::GridInfoVariant& grid_info
)
{
    if (auto rect_grid_info = std::get_if<rrgsim::common::RectGridInfo>(&grid_info); rect_grid_info) {
        return std::make_shared<rrgsim::mesh::rect::RectMesh2PSolver>();
    }
    else if (auto sph_grid_info = std::get_if<rrgsim::common::SphGridInfo>(&grid_info); sph_grid_info) {
        return std::make_shared<rrgsim::mesh::sph::SphMesh2PSolver>();
    }
    else {
        throw std::runtime_error{ "make_mesh2p_solver error: invalid grid info variant" };
    }
}

} // namespace rrgsim::mesh