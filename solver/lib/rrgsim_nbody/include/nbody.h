#pragma once
#include <co_sim_params.h>
#include <co_device_structs.cuh>
#include <co_particles_context.h>

namespace rrgsim::nbody {
    using RR::CUDA::CuDarray;
    using common::sParticlesContext;
    using common::sParticlesContext_;
    using common::BLOCK_SIZE;
    using common::SimParams;

    struct ConservationInfo {
        real time = 0;
        real Ek = 0;
        real Ep = 0;
        real3 momentum = make_real3(0., 0., 0.);
        real3 angular = make_real3(0., 0., 0.);

        real E() const {
            return Ek + Ep;
        }
    };

    ConservationInfo calc_conservation(
        sParticlesContext host_context
    );

    void predict_step(
        sParticlesContext_ context,
        const SimParams& sim_params
    );

    void nbody_acceleration(
        sParticlesContext_ context,
        const SimParams& sim_params
    );

    void nbody_grav(
        sParticlesContext_ context,
        const SimParams& sim_params
    );

    void correct_step(
        sParticlesContext_ context,
        const SimParams& sim_params
    );

    inline int calc_blocks_count(int ntotal) {
        return (ntotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

} // namespace rrgsim::nbody