#pragma once
#include <co_sim_params.h>
#include <co_particles.h>

namespace rrgsim {

using rrgsim::common::ParticlesData;
using rrgsim::common::SimParams;
using rrgsim::common::WaveParams;
using rrgsim::common::GridInfo;

/// @brief Particle in cell
void integrate_PIC(
    ParticlesData particles_data,
    SimParams sim_params,
    GridInfo grid_info,
    WaveParams wave_params
);

} // namespace rrgsim