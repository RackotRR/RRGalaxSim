#include "rrgsim_pic.h"

#include <spdlog/spdlog.h>

#include <co_grid_context.cuh>
#include <co_particles_context.cuh>

#include <nbody.cuh>
#include <p2mesh.cuh>
#include <mesh2p.cuh>
#include <wave.cuh>

#include "rrgsim_conservation.cuh"

namespace rrgsim {

void integrate_PIC(
    ParticlesData particles_data,
    SimParams sim_params,
    GridInfo grid_info,
    WaveParams wave_params
)
{
    auto particles_context_ = rrgsim::common::initialize_particles_context_(particles_data);
    auto particles_context = rrgsim::common::initialize_particles_context(std::move(particles_data));
    auto grid_context_ = rrgsim::common::initialize_grid_context_(grid_info);
    auto grid_context = rrgsim::common::initialize_grid_context(grid_info);
    rrgsim::conservation::check_bounds(particles_data.pos, grid_info);

    RR::CUDA::CuCopyToSymbol(
        grid_info,
        rrgsim::common::grid_info_,
        RR::CUDA::ToDevice
    );
    RR::CUDA::CuCopyToSymbol(
        wave_params,
        rrgsim::common::wave_params_,
        RR::CUDA::ToDevice
    );

    rrgsim::p2mesh::convert_particles_to_grid(particles_context_, grid_context_);
    rrgsim::wave::solve_wave_equation(grid_context_, wave_params);
    rrgsim::p2mesh::calc_acceleration_field(grid_context_);
    rrgsim::mesh2p::project_grid_acceleration_on_particles(grid_context_, particles_context_);

    // далее не требуется много итераций, используем начальное приближение
    wave_params.setup_iterations = 5000;

    const auto base_conservation_info = rrgsim::nbody::calc_conservation(particles_context);
    rrgsim::conservation::print_conservation(base_conservation_info);


    real time = 0.;
    real next_save = time + sim_params.dt_save;

    while (time < sim_params.time_max) {
        rrgsim::nbody::predict_step(particles_context_, sim_params);

        rrgsim::p2mesh::convert_particles_to_grid(particles_context_, grid_context_);
        rrgsim::wave::solve_wave_equation(grid_context_, wave_params);
        rrgsim::p2mesh::calc_acceleration_field(grid_context_);
        rrgsim::mesh2p::project_grid_acceleration_on_particles(grid_context_, particles_context_);

        rrgsim::nbody::correct_step(particles_context_, sim_params);

        time += sim_params.dt_dynamics;
        particles_context_->time = time;

        if (time >= next_save) {
            spdlog::info("Time to save: {}", time);

            rrgsim::mesh2p::project_grid_grav_on_particles(grid_context_, particles_context_);
            particles_context->fill_device_data(particles_context_);

            rrgsim::conservation::print_conservation(
                base_conservation_info,
                rrgsim::nbody::calc_conservation(particles_context)
            );

            next_save += sim_params.dt_save;
        }
    }
}

} // namespace rrgsim