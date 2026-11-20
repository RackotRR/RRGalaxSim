#include "sm_mesh2p.cuh"

namespace rrgsim::mesh::sph {

void SphMesh2PSolver::project_grid_acceleration_on_particles(
    sGridContext_ grid_context_,
    sParticlesContext_ particles_context_
)
{
    throw std::runtime_error{ "Not implemented" };
}

void SphMesh2PSolver::project_grid_grav_on_particles(
    const sGridContext_ grid_context_,
    sParticlesContext_ particles_context_
)
{
    throw std::runtime_error{ "Not implemented" };
}

GridProjectionOnParticles
SphMesh2PSolver::project_grid_onto_particles(
    sGridContext_ grid_context_,
    sParticlesContext_ particles_context_
)
{
    throw std::runtime_error{ "Not implemented" };
}

} // namespace rrgsim::mesh::sph