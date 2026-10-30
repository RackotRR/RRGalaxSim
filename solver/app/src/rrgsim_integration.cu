#include "rrgsim_integration.cuh"
#include "rrgsim_integration_nbody.cuh"
#include "rrgsim_integration_rect.cuh"

#include <spdlog/spdlog.h>

#include <co_grid_context.cuh>
#include <co_particles_context.cuh>

#include <RR/Time/Timer.h>

#include "rrgsim_conservation.cuh"

#include <io_particles_bin.cuh>
#include <io_handler.h>

namespace rrgsim::integration {

void integrate(
    IntegrationParams params
)
{
    auto particles_context_ = rrgsim::common::initialize_particles_context_(params.particles_data);
    auto particles_context = rrgsim::common::initialize_particles_context(std::move(params.particles_data));

    rrgsim::common::sGridContext grid_context = nullptr;
    rrgsim::common::sGridContext_ grid_context_ = nullptr;

    params.mb_grid_info
        .transform(
            [&](const GridInfo& grid_info) {
                grid_context = rrgsim::common::initialize_grid_context(grid_info);
                grid_context_ = rrgsim::common::initialize_grid_context_(grid_info);
                rrgsim::conservation::check_bounds(particles_context->pos, grid_info);
            }
        );

    RR::Timer timer_init;
    timer_init.start();

    std::shared_ptr<IntegrationSolver> solver;
    if (params.sim_params.use_wave_model) {
        solver = std::make_shared<RectWaveIntegrationSolver>();
    }
    else {
        solver = std::make_shared<NBodyIntegrationSolver>();
    }

    solver->init(params, particles_context_, grid_context_);

    // Расчёт потенциала и ускорения в начальный момент времени
    solver->precalc();

    // Синхронизация частиц и сетки для сохранения данных
    solver->sync_particles_and_grid();

    particles_context->fill_device_data(particles_context_);
    const auto base_conservation_info = rrgsim::nbody::calc_conservation(particles_context);
    rrgsim::conservation::print_conservation(base_conservation_info);
    rrgsim::conservation::print_sample_particle(particles_context);

    if (grid_context_) {
        grid_context->fill_device_data(grid_context_);
        rrgsim::conservation::print_acc_projection(grid_context);
        rrgsim::conservation::print_grav_projection(grid_context);
        rrgsim::conservation::print_mass_projection(grid_context);
    }

    timer_init.finish();
    spdlog::info("Init time: {} seconds", timer_init.total<std::chrono::seconds>());

    size_t iteration = 0;
    real time = 0.;
    real next_save = time + params.sim_params.dt_save;

    size_t saves_without_dump = 0;

    RR::Timer timer_iterations;
    while (time < params.sim_params.time_max) {
        timer_iterations.start();

        solver->calc_step();

        ++iteration;
        time += params.sim_params.dt_dynamics;
        particles_context_->time = time;
        particles_context_->iter = iteration;

        if (grid_context_) {
            grid_context_->time = time;
            grid_context_->iter = iteration;
        }

        if (time >= next_save) {
            spdlog::info("Simulation time: {:.5f} / {:.5f} (iter: {})", time, params.sim_params.time_max, iteration);

            // подготавливаем ускорение и потенциал для вывода
            solver->sync_particles_and_grid();

            particles_context->fill_device_data(particles_context_);
            rrgsim::conservation::print_sample_particle(particles_context);
            rrgsim::conservation::print_conservation(base_conservation_info,
                rrgsim::nbody::calc_conservation(particles_context)
            );

            if (grid_context_) {
                grid_context->fill_device_data(grid_context_);
                rrgsim::conservation::check_bounds(particles_context->pos, grid_context->info);
                rrgsim::conservation::print_acc_projection(grid_context);
                rrgsim::conservation::print_grav_projection(grid_context);
                rrgsim::conservation::print_mass_projection(grid_context);
            }

            next_save += params.sim_params.dt_save;
            timer_iterations.finish();
            spdlog::info("Time between saves: {} seconds", timer_iterations.average<std::chrono::seconds>());

            long long saves_left = static_cast<long long>((params.sim_params.time_max - time) / params.sim_params.dt_save);
            spdlog::info("Time total: {} min (est. ~{} min)",
                timer_iterations.total<std::chrono::minutes>() +
                timer_init.total<std::chrono::minutes>(),
                // estimate in min.
                timer_iterations.average<std::chrono::milliseconds>() * saves_left / 60'000
            );

            ++saves_without_dump;
            if (saves_without_dump < params.sim_params.saves_without_dump) {
                std::ignore = io::write_particles_dump(
                        particles_context,
                        io::IOHandler::get_dump_dir()
                    )
                    .transform_error(
                        [](auto&& error) {
                            spdlog::error(error);
                        }
                    );
            }
        }
    }

    spdlog::info("integration completed ({} min)",
        timer_iterations.total<std::chrono::minutes>() +
        timer_init.total<std::chrono::minutes>()
    );
}

} // namespace rrgsim