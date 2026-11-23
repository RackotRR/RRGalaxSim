#include "sm_p2mesh.cuh"

namespace rrgsim::mesh::sph {

void SphP2MeshSolver::convert_particles_to_grid(
    const sParticlesContext_ particles_,
    sGridContext_ grid_
)
{
    throw std::runtime_error{ "Not implemented" };
}

void SphP2MeshSolver::project_particles_acceleration_on_grid(
    const sParticlesContext_ particles_context_,
    sGridContext_ grid_context_
)
{
    throw std::runtime_error{ "Not implemented" };
}

void SphP2MeshSolver::project_particles_grav_on_grid(
    const sParticlesContext_ particles_context_,
    sGridContext_ grid_context_
)
{
    throw std::runtime_error{ "Not implemented" };
}

void SphP2MeshSolver::calc_acceleration_field(
    const sGridContext_ grid_
)
{
    throw std::runtime_error{ "Not implemented" };
}

void SphP2MeshSolver::calc_grav_by_center_of_mass(
    rrgsim::common::sGridContext_ grid_,
    real3 center_of_mass,
    real total_mass
)
{
    throw std::runtime_error{ "Not implemented" };
}

const rrgsim::common::SphGridInfo&
SphP2MeshSolver::grid_info(
    const rrgsim::common::sGridContext_& grid_
)
{
    if (auto sph_grid_info = std::get_if<rrgsim::common::SphGridInfo>(&grid_->info); sph_grid_info) {
        return *sph_grid_info;
    }
    else {
        throw std::runtime_error{ "SphGridInfo::grid_info met unexpected grid info variant" };
    }
}

} // namespace rrgsim::mesh::sph