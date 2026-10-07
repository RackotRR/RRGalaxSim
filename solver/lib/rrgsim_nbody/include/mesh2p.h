#pragma once
#include <co_particles_context.cuh>
#include <co_grid_context.h>

namespace rrgsim::mesh2p {

using rrgsim::common::GridProjectionOnParticles;
using rrgsim::common::sGridContext_;
using rrgsim::common::sParticlesContext_;

GridProjectionOnParticles
project_grid_onto_particles(
    sGridContext_ grid_context_,
    sParticlesContext_ particles_context_
);

} // namespace rrgsim::mesh2p