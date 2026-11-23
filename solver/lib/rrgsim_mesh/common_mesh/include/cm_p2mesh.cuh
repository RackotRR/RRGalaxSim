#pragma once
#include <co_particles_context.cuh>
#include <co_grid_context.cuh>

namespace rrgsim::mesh {
    using common::sGridContext_;
    using common::sParticlesContext_;

    class P2MeshSolver {
    public:
        void init_p2mesh_context(
            sGridContext_ grid_,
            const sParticlesContext_ particles_
        );

        void convert_particles_to_grid(
            const sParticlesContext_ particles_,
            sGridContext_ grid_
        );

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
            sGridContext_ grid_,
            real3 center_of_mass,
            real total_mass
        ) = 0;

    protected:
        virtual void assign_particles_to_cells(
            const rrgsim::common::OverInfo& over,
            const sParticlesContext_ particles_context_,
            sGridContext_ grid_context
        ) = 0;
    };

    namespace detail {

        __global__ void compute_prefix_sums_(
            rrgsim::common::detail::CellInfo* __restrict__ cellInfo,
            int* __restrict__ maxPBC,
            const int numCells
        );


        __global__ void adjust_global_prefix_sums_(
            rrgsim::common::detail::CellInfo* __restrict__ cellInfo,
            const int* __restrict__ maxPBC,
            const int numCells
        );

    } // namespace detail

} // namespace rrgsim::mesh