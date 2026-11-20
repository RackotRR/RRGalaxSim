#pragma once
#include <co_particles_context.cuh>
#include <co_grid_context.cuh>

namespace rrgsim::mesh {
    using common::sGridContext_;
    using common::sParticlesContext_;

    struct P2MeshSolver {
        virtual void convert_particles_to_grid(
            const sParticlesContext_ particles_,
            sGridContext_ grid_
        ) = 0;

        virtual void project_particles_acceleration_on_grid(
            const sParticlesContext_ particles_context_,
            sGridContext_ grid_context_
        ) = 0;

        virtual void project_particles_grav_on_grid(
            const sParticlesContext_ particles_context_,
            sGridContext_ grid_context_
        ) = 0;

        virtual void calc_acceleration_field(
            const sGridContext_ grid_
        ) = 0;

        virtual void calc_grav_by_center_of_mass(
            rrgsim::common::sGridContext_ grid_,
            real3 center_of_mass,
            real total_mass
        ) = 0;
    };

} // namespace rrgsim::mesh