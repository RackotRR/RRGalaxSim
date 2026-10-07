#pragma once
#include <co_sim_params.h>
#include <co_device_structs.cuh>
#include <co_particles_context.cuh>
#include <co_grid_context.h>

namespace rrgsim::p2mesh {
    using common::sGridContext_;
    using common::sParticlesContext_;

    void convert_particles_to_grid(
        const sParticlesContext_ particles_,
        sGridContext_ grid_
    );

    void calc_acceleration_field(
        const sGridContext_ grid_
    );
} // namespace rrgsim::p2mesh