#pragma once
#include <m_common_mesh_solver.cuh>

namespace rrgsim::mesh::sph {
    using rrgsim::common::sph_grid_info_;
    using rrgsim::common::SphGridInfo;

    class SphMeshSolver : public MeshSolver {
    public:

        const SphGridInfo& grid_info(
            const sGridContext_& grid_
        );

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
            sGridContext_ grid_,
            real3 center_of_mass,
            real total_mass
        ) override;

        void project_grid_acceleration_on_particles(
            sGridContext_ grid_context_,
            sParticlesContext_ particles_context_
        ) override;

        void project_grid_grav_on_particles(
            const sGridContext_ grid_context_,
            sParticlesContext_ particles_context_
        ) override;

    protected:
        void init(
            const sGridContext_ grid_
        ) override;

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
    };

    inline __device__
    size_t cell_index(int i, int j, int k) {
        return (size_t)i
            +  (size_t)sph_grid_info_.nr * (size_t)j
            +  (size_t)sph_grid_info_.nr * sph_grid_info_.nt * (size_t)k;
    }

} // namespace rrgsim::mesh::sph