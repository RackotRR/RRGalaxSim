#include <co_sim_params.h>
#include <co_particles.h>
#include <co_particles_context.cuh>
#include <co_grid_context.cuh>
#include <co_device_utils.cuh>

#include "io_particles.h"
#include "io_json.h"
#include "io_handler.h"

#include <fstream>
#include <numeric>

#include "rrgsim_log.h"
#include "rrgsim_conservation.cuh"
#include "nbody.cuh"
#include "p2mesh.cuh"
#include "mesh2p.cuh"
#include "wave.cuh"

#include "rrgsim_nbody.h"

void print_grav_projection(
    rrgsim::common::sGridContext grid_context
)
{
    auto& handler = rrgsim::io::IOHandler::instance();
    size_t NX = grid_context->info.nx;

    for (size_t xi = 0; xi < NX; ++xi) {
        rrgsim::io::TagValue x{ "x", grid_context->info.domain_min + grid_context->info.dx * xi };
        rrgsim::io::TagValue g{ "grav", grid_context->grav[AT(xi, NX / 2, NX / 2)] };
        handler.append_table("grav_x_center", { x, g });
    }

    for (size_t yi = 0; yi < NX; ++yi) {
        rrgsim::io::TagValue y{ "y", grid_context->info.domain_min + grid_context->info.dx * yi };
        rrgsim::io::TagValue g{ "grav", grid_context->grav[AT(NX / 2, yi, NX / 2)] };
        handler.append_table("grav_y_center", { y, g });
    }

    for (size_t zi = 0; zi < NX; ++zi) {
        rrgsim::io::TagValue z{ "z", grid_context->info.domain_min + grid_context->info.dx * zi };
        rrgsim::io::TagValue g{ "grav", grid_context->grav[AT(NX / 2, NX / 2, zi)] };
        handler.append_table("grav_z_center", { z, g });
    }
}

void print_grid_nbody_projection(
    std::string filename,
    const std::vector<real>& nbody,
    const std::vector<real>& grid,
    const std::vector<real3>& pos
)
{
    auto& handler = rrgsim::io::IOHandler::instance();

    for (size_t i = 0; i < nbody.size(); ++i) {
        rrgsim::io::TagValue ii{ "i", (real)i };
        rrgsim::io::TagValue g_nbody{ "nbody", nbody[i] };
        rrgsim::io::TagValue g_grid{ "grid", grid[i] };
        rrgsim::io::TagValue x{ "x", pos[i].x };
        rrgsim::io::TagValue y{ "y", pos[i].y };
        rrgsim::io::TagValue z{ "z", pos[i].z };

        handler.append_table(filename, { ii, g_nbody, g_grid, x, y, z });
    }
};

void check_grid(
    rrgsim::common::ParticlesData particles_data,
    rrgsim::common::GridInfo grid_info,
    rrgsim::common::WaveParams wave_params
)
{
    RR::CUDA::CuCopyToSymbol(
        wave_params,
        rrgsim::common::wave_params_,
        RR::CUDA::ToDevice
    );

    auto particles_context_ = rrgsim::common::initialize_particles_context_(particles_data);
    auto grid_context_ = rrgsim::common::initialize_grid_context_(grid_info);
    auto grid_context = rrgsim::common::initialize_grid_context(grid_info);
    rrgsim::conservation::check_bounds(particles_data.pos, grid_info);

    rrgsim::p2mesh::convert_particles_to_grid(
        particles_context_,
        grid_context_
    );

    rrgsim::wave::solve_wave_equation(
        grid_context_,
        wave_params
    );

    grid_context->fill_device_data(grid_context_);
    print_grav_projection(grid_context);

    rrgsim::nbody::nbody_grav(particles_context_);
    auto grav_nbody = particles_context_->grav_.to_vector();
    auto mass_nbody = particles_data.mass;

    // conservation
    auto particles_context = rrgsim::common::initialize_particles_context(particles_data);
    particles_context->fill_device_data(particles_context_);
    auto conservation_nbody = rrgsim::nbody::calc_conservation(particles_context);

    auto grid_projected = rrgsim::mesh2p::project_grid_onto_particles(
        grid_context_,
        particles_context_
    );
    particles_context->grav = grid_projected.grav;
    auto conservation_grid_projected = rrgsim::nbody::calc_conservation(particles_context);
    spdlog::info("Conservation Grid Projected VS NBody");
    rrgsim::conservation::print_conservation(conservation_nbody, conservation_grid_projected);

    print_grid_nbody_projection("mass_nbody_grid", mass_nbody, grid_projected.mass, particles_data.pos);
    print_grid_nbody_projection("grav_nbody_grid", grav_nbody, grid_projected.grav, particles_data.pos);

    real mass_grid = std::accumulate(
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
                std::move(parsed.mb_grid_info).value(),
                parsed.mb_wave_params.value()
            );
            // ::integrate_nbody(
            //     std::move(expected_particles_data).value(),
            //     std::move(parsed.sim_params),
            // );
        }
        catch (const std::exception& ex) {
            spdlog::error("Exception: {}", ex.what());
        }
    }

    spdlog::default_logger()->flush();
    return 0;
}