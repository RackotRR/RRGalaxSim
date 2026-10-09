#pragma once
#include <co_particles_context.cuh>
#include <co_grid_context.cuh>

namespace rrgsim::mesh2p {

using rrgsim::common::GridProjectionOnParticles;
using rrgsim::common::sGridContext_;
using rrgsim::common::sParticlesContext_;

void project_grid_acceleration_on_particles(
    sGridContext_ grid_context_,
    sParticlesContext_ particles_context_
);

void project_grid_grav_on_particles(
    const sGridContext_ grid_context_,
    sParticlesContext_ particles_context_
);

GridProjectionOnParticles
project_grid_onto_particles(
    sGridContext_ grid_context_,
    sParticlesContext_ particles_context_
);

} // namespace rrgsim::mesh2p