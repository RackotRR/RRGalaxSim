#pragma once
#include <cm_p2mesh.cuh>

namespace rrgsim::mesh::sph {

    class SphP2MeshSolver : public P2MeshSolver {
    public:
        void convert_particles_to_grid(
            const sParticlesContext_ particles_,
            sGridContext_ grid_
        ) override;

        void project_particles_acceleration_on_grid(
            const sParticlesContext_ particles_context_,
            sGridContext_ grid_context_
        ) override;

        void project_particles_grav_on_grid(
            const sParticlesContext_ particles_context_,
            sGridContext_ grid_context_
        ) override;

        void calc_acceleration_field(
            const sGridContext_ grid_
        ) override;

        void calc_grav_by_center_of_mass(
            rrgsim::common::sGridContext_ grid_,
            real3 center_of_mass,
            real total_mass
        ) override;

    public:
        const rrgsim::common::SphGridInfo& grid_info(
            const rrgsim::common::sGridContext_& grid_
        );
    };

} // namespace rrgsim::mesh::sph