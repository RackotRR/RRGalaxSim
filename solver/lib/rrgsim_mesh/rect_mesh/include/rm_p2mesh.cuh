#pragma once
#include <cm_p2mesh.cuh>

namespace rrgsim::mesh::rect {

    class RectP2MeshSolver : public P2MeshSolver {
    public:
        void project_particles_acceleration_on_grid(
            const sParticlesContext_ particles_,
            sGridContext_ grid_
        ) override;

        void project_particles_grav_on_grid(
            const sParticlesContext_ particles_,
            sGridContext_ grid_
        ) override;

        void calc_acceleration_field(
            const sGridContext_ grid_
        ) override;

        void calc_grav_by_center_of_mass(
            sGridContext_ grid_,
            real3 center_of_mass,
            real total_mass
        ) override;

    protected:

        void assign_particles_to_cells(
            const rrgsim::common::OverInfo& over,
            const sParticlesContext_ particles_context_,
            sGridContext_ grid_context_
        ) override;

        void deposit_mass(
            const rrgsim::common::OverInfo& over,
            const sParticlesContext_ particles_,
            sGridContext_ grid_
        ) override;

    private:

        const rrgsim::common::RectGridInfo& grid_info(
            const rrgsim::common::sGridContext_& grid_
        );

    };

} // namespace rrgsim::mesh::rect