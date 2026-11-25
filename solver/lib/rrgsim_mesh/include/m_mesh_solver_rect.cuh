#pragma once
#include <m_common_mesh_solver.cuh>

namespace rrgsim::mesh::rect {
    using rrgsim::common::rect_grid_info_;
    using rrgsim::common::RectGridInfo;

    class RectMeshSolver : public MeshSolver {
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

        void project_grid_acceleration_on_particles(
            sGridContext_ grid_,
            sParticlesContext_ particles_
        ) override;

        void project_grid_grav_on_particles(
            const sGridContext_ grid_,
            sParticlesContext_ particles_
        ) override;

    protected:
        void init(
            const sGridContext_ grid_
        ) override;

        void assign_particles_to_cells(
            const rrgsim::common::OverInfo& over,
            const sParticlesContext_ particles_,
            sGridContext_ grid_
        ) override;

        void deposit_mass(
            const rrgsim::common::OverInfo& over,
            const sParticlesContext_ particles_,
            sGridContext_ grid_
        ) override;

    private:

        const RectGridInfo& grid_info(
            const sGridContext_& grid_
        );

    };

    inline __device__ __forceinline__
    size_t cell_index(int i, int j, int k) {
        return (size_t)i
            +  (size_t)rect_grid_info_.nx * (size_t)j
            +  (size_t)rect_grid_info_.nx * rect_grid_info_.nx * (size_t)k;
    }

} // namespace rrgsim::mesh::rect