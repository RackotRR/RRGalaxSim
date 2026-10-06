#pragma once
#include <cuda_runtime.h>

#include <co_device_utils.cuh>
#include <co_device_structs.cuh>

#include <co_grid_context.h>

namespace rrgsim::nbody {
    using rrgsim::common::particles_info_;
    using rrgsim::common::grid_info_;
	using rrgsim::common::BLOCK_SIZE;
    using rrgsim::common::detail::ParticleCellInfo;
    using rrgsim::common::detail::CellInfo;


/// @note Запуск по частицам
inline __global__ void projectCellMassOnParticles(
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
inline __global__ void projectCellPhiOnParticles(
    const real* cell_phi,
    const ParticleCellInfo* particles_cell_info,
    real* particle_phi
)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    int i_cell = particles_cell_info[i_particle].cell_id;
    particle_phi[i_particle] = cell_phi[i_cell];
}

} // namespace rrgsim::nbody

