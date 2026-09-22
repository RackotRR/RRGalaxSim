#pragma once
#include <co_sim_params.h>
#include <co_device_structs.cuh>
#include <co_particles_context.h>
#include <co_grid_context.h>

namespace rrgsim::nbody {
    using common::sParticlesContext;
    using common::sParticlesContext_;
    using common::sGridContext;
    using common::sGridContext_;
    using common::SimParams;
    using common::BLOCK_SIZE;
    using RR::CUDA::CuDarray;

    void convert_particles_to_grid(
        const sParticlesContext_ particles_,
        sGridContext_ grid_
    );


} // namespace rrgsim::nbody