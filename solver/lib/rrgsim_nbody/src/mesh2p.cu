#include <cuda_runtime.h>
#include <co_device_utils.cuh>
#include <co_device_structs.cuh>

#include "mesh2p.cuh"

#include <spdlog/spdlog.h>

namespace rrgsim::mesh2p {

using rrgsim::common::detail::ParticleCellInfo;
using rrgsim::common::detail::CellInfo;

/// @note Запуск по частицам
__global__ void projectCellMassOnParticles(
    const real* cell_mass,
    const ParticleCellInfo* particles_cell_info,
    const CellInfo* cell_info,
    real* particle_phi
)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    int i_cell = particles_cell_info[i_particle].cell_id;
    particle_phi[i_particle] = cell_mass[i_cell] / cell_info[i_cell].count;
}

/// @note Запуск по частицам
__global__ void projectCellPhiOnParticles(
    const real* cell_phi,
    const ParticleCellInfo* particles_cell_info,
    real* particle_phi
)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    int i_cell = particles_cell_info[i_particle].cell_id;
    particle_phi[i_particle] = cell_phi[i_cell];
}


GridProjectionOnParticles
project_grid_onto_particles(
    sGridContext_ grid_context_,
    sParticlesContext_ particles_context_
)
{
    spdlog::info("project_grid_onto_particles");
    RR::CUDA::CuDeviceSync();

    auto over = rrgsim::common::OverInfo::calc(0, particles_context_->info.ntotal);

    RR::CUDA::CuCall(projectCellPhiOnParticles, over.particles, over.blocks) (
        grid_context_->grav_curr_,
        grid_context_->particles_cell_info_,
        particles_context_->grav_
    );
    RR::CUDA::CuCall(projectCellMassOnParticles, over.particles, over.blocks) (
        grid_context_->mass_,
        grid_context_->particles_cell_info_,
        grid_context_->cell_info_,
        particles_context_->mass_
    );

    GridProjectionOnParticles projection;
    projection.grav = particles_context_->grav_.to_vector();
    projection.mass = particles_context_->mass_.to_vector();
    return projection;
}


} // namespace rrgsim::mesh2p