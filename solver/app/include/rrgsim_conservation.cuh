#pragma once
#include <nbody.cuh>
#include <filesystem>

#include <co_particles_context.cuh>
#include <co_grid_context.cuh>

namespace rrgsim::conservation {

void print_conservation(
    const rrgsim::nbody::ConservationInfo& info
);

void print_conservation(
    const rrgsim::nbody::ConservationInfo& info0,
    const rrgsim::nbody::ConservationInfo& info
);

void check_bounds(
    const std::vector<real3>& particles_pos,
    const rrgsim::common::GridInfo& grid_info
);

void print_sample_particle(
    rrgsim::common::sParticlesContext particles_context
);
void print_acc_projection(
    rrgsim::common::sGridContext grid_context
);
void print_grav_projection(
    rrgsim::common::sGridContext grid_context
);

} // namespace rrgsim::conservation