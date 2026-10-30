#include <co_sim_params.h>

#include "io_particles_csv.h"
#include "io_json.h"
#include "io_handler.h"

#include "rrgsim_log.h"

#include "rrgsim_integration.cuh"


int main(int argc, const char** argv) {
    if (argc < 2) {
        return 1;
    }
    else if (argc < 3) {
        const char* work_dir = argv[1];
        std::filesystem::path work_dir_path = work_dir;

        rrgsim::log::setup_logging(work_dir_path);
        rrgsim::io::IOHandler::setup(work_dir_path);

        try {
            auto devices = RR::CUDA::CuGetDevices();
            for (const auto& device : devices) {
                spdlog::info("Device: {}", device.name);
            }
        }
        catch (const std::exception& ex) {
            spdlog::error("CUDA Error: {}", ex.what());
        }

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
            rrgsim::integration::IntegrationParams params;
            params.particles_data = std::move(expected_particles_data).value();
            params.sim_params = std::move(parsed.sim_params);
            params.mb_grid_info = std::move(parsed.mb_grid_info).value();
            params.mb_wave_params = std::move(parsed.mb_wave_params).value();

            rrgsim::integration::integrate(params);
        }
        catch (const std::exception& ex) {
            spdlog::error("Exception: {}", ex.what());
        }
    }

    spdlog::default_logger()->flush();
    return 0;
}