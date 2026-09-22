#include <co_sim_params.h>
#include <co_particles.h>
#include <co_particles_context.h>
#include <co_grid_context.h>
#include <co_device_utils.cuh>

#include "io_particles.h"
#include "io_json.h"
#include "io_handler.h"

#include <fstream>
#include <numeric>

#include "rrgsim_log.h"
#include "rrgsim_conservation.h"
#include "nbody.h"
#include "p2mesh.h"

void integrate_nbody(
    rrgsim::common::ParticlesData particles_data,
    rrgsim::common::SimParams sim_params
) {
    auto particles_context_ = rrgsim::common::initialize_particles_context_(particles_data);
    auto particles_context = initialize_particles_context(std::move(particles_data));
    rrgsim::nbody::nbody_grav(
        particles_context_,
        sim_params
    );
    rrgsim::nbody::nbody_acceleration(
        particles_context_,
        sim_params
    );
    particles_context->grav = particles_context_->grav_.to_vector();
    const auto base_conservation_info = rrgsim::nbody::calc_conservation(particles_context);
    rrgsim::conservation::print_conservation(base_conservation_info);

    double time = 0.;
    double next_save = time + sim_params.dt_save;
    while (time < sim_params.time_max) {
        rrgsim::nbody::predict_step(
            particles_context_,
            sim_params
        );

        rrgsim::nbody::nbody_acceleration(
            particles_context_,
            sim_params
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
                particles_context_,
                sim_params
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

void check_grid(
    rrgsim::common::ParticlesData particles_data,
    rrgsim::common::GridInfo grid_info
)
{
    auto particles_context_ = rrgsim::common::initialize_particles_context_(particles_data);
    auto grid_context_ = rrgsim::common::initialize_grid_context_(grid_info);
    auto grid_context = rrgsim::common::initialize_grid_context(grid_info);

    rrgsim::nbody::convert_particles_to_grid(
        particles_context_,
        grid_context_
    );

    grid_context->fill_device_data(grid_context_);


    double mass_grid = std::accumulate(
        grid_context->mass.begin(),
        grid_context->mass.end(),
        0.
    );
    spdlog::info("Mass in grid: {}", mass_grid);
}

int main(int argc, const char** argv) {
    if (argc < 2) {
        return 1;
    }
    else if (argc < 3) {
        const char* work_dir = argv[1];
        std::filesystem::path work_dir_path = work_dir;

        rrgsim::log::setup_logging(work_dir_path);
        rrgsim::io::IOHandler::setup(work_dir_path);

        auto expected_parsed_params = rrgsim::io::parse_params_json(work_dir_path / "params.json");
        if (false == expected_parsed_params.has_value()) {
            spdlog::error("Error: {}", expected_parsed_params.error());
            return 0;
        }
        rrgsim::io::ParsedParams parsed = std::move(expected_parsed_params).value();

        auto expected_particles_data = rrgsim::io::read_simple_particles_data(parsed.galaxy_data);
        if (false == expected_particles_data.has_value()) {
            spdlog::error("Error: {}", expected_particles_data.error());
            return 0;
        }

        try {
            check_grid(
                std::move(expected_particles_data).value(),
                std::move(parsed.mb_grid_info).value()
            );
            // ::integrate_nbody(
            //     std::move(expected_particles_data).value(),
            //     std::move(parsed.sim_params)
            // );
        }
        catch (const std::exception& ex) {
            spdlog::error("Exception: {}", ex.what());
        }
    }

    return 0;
}