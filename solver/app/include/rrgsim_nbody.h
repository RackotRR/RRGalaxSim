#pragma once
#include <co_sim_params.h>
#include <co_particles.h>

namespace rrgsim {

using rrgsim::common::ParticlesData;
using rrgsim::common::SimParams;

void integrate_nbody(
    ParticlesData particles_data,
    SimParams sim_params
);

} // namespace rrgsim