#pragma once
#include <nbody.h>
#include <filesystem>

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

} // namespace rrgsim::conservation