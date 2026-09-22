#pragma once
#include <RR/CUDA/CuCommon.cuh>

#include <memory>
#include <vector>

#include <rrgsim_tl.h>

#include <co_device_structs.cuh>
#include <co_particles.h>

namespace rrgsim::common {
    using RR::CUDA::CuDarray;

    // контекст расчёта на GPU
    struct ParticlesContext_ {
        double time = 0.; // current time

        CuDarray<double3> pos_; // particles position (GPU)
        CuDarray<double3> vel_; // particles velocity (GPU)
        CuDarray<double3> vel_predicted_; // particles velocity predicted (GPU)
        CuDarray<double3> acc_; // particles acceleration (GPU)
        CuDarray<double> mass_; // particles masses (GPU)
        CuDarray<double> soft2_; // particles softening squared (GPU)
        CuDarray<double> grav_; // particles gravitational potential (GPU)

        ParticlesInfo info;
    };
    using sParticlesContext_ = std::shared_ptr<ParticlesContext_>;

    // контекст для пост-процессинга на CPU (данные, которые копируются с шагом dt_save)
    struct ParticlesContext {
        double time = 0.;
        std::vector<double3> pos;
        std::vector<double3> vel;
        std::vector<double> mass;
        std::vector<double> grav;

        void fill_device_data(
            const sParticlesContext_& context_
        );
    };
    using sParticlesContext = std::shared_ptr<ParticlesContext>;


    sParticlesContext_
    initialize_particles_context_(
        const common::ParticlesData& particles_data
    );

    sParticlesContext
    initialize_particles_context(
        common::ParticlesData particles_data
    );

} // namespace rrgsim::common