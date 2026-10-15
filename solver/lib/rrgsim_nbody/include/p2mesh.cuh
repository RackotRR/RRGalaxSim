#pragma once
#include <co_sim_params.h>
#include <co_device_structs.cuh>
#include <co_particles_context.cuh>
#include <co_grid_context.cuh>

namespace rrgsim::p2mesh {
    using common::sGridContext_;
    using common::sParticlesContext_;

    void convert_particles_to_grid(
        const sParticlesContext_ particles_,
        sGridContext_ grid_
    );

    void project_particles_acceleration_on_grid(
        const sParticlesContext_ particles_context_,
        sGridContext_ grid_context_
    );

    void project_particles_grav_on_grid(
        const sParticlesContext_ particles_context_,
        sGridContext_ grid_context_
    );

    void calc_acceleration_field(
        const sGridContext_ grid_
    );

    void calc_grav_by_center_of_mass(
        rrgsim::common::sGridContext_ grid_,
        real3 center_of_mass,
        real total_mass
    );
} // namespace rrgsim::p2mesh