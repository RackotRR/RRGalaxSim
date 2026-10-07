#include "rrgsim_nbody.h"
#include "rrgsim_conservation.cuh"

#include <co_particles_context.cuh>

#include <nbody.cuh>

#include <spdlog/spdlog.h>

namespace rrgsim {

void integrate_nbody(
    ParticlesData particles_data,
    SimParams sim_params
)
{
    auto particles_context_ = rrgsim::common::initialize_particles_context_(particles_data);
    auto particles_context = initialize_particles_context(std::move(particles_data));
    rrgsim::nbody::nbody_grav(
        particles_context_
    );
    rrgsim::nbody::nbody_acceleration(
        particles_context_
    );
    particles_context->grav = particles_context_->grav_.to_vector();
    const auto base_conservation_info = rrgsim::nbody::calc_conservation(particles_context);
    rrgsim::conservation::print_conservation(base_conservation_info);

    real time = 0.;
    real next_save = time + sim_params.dt_save;
    while (time < sim_params.time_max) {
        rrgsim::nbody::predict_step(
            particles_context_,
            sim_params
        );

        rrgsim::nbody::nbody_acceleration(
            particles_context_
        );

        rrgsim::nbody::correct_step(
            particles_context_,
            sim_params
        );

        time += sim_params.dt_dynamics;
        particles_context_->time = time;

        if (time >= next_save) {
            spdlog::info("Time to save: {}", time);

            rrgsim::nbody::nbody_grav(
                particles_context_
            );

            particles_context->fill_device_data(
                particles_context_
            );

            rrgsim::conservation::print_conservation(
                base_conservation_info,
                rrgsim::nbody::calc_conservation(particles_context)
            );

            next_save += sim_params.dt_save;
        }
    }
}

} // namespace rrgsim