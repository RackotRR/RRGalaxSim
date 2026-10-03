#include <fstream>
#include <csv.hpp>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "io_particles.h"
#include "io_json.h"

namespace rrgsim::io {

    tl::expected<size_t, std::string>
    read_particles_data_component(
        const IniGalaxyComponentData& component_data,
        ParticlesData& particles_data
    )
    {
        spdlog::info("Read particles data with ini path={}", component_data.ini_file.string());

        csv::CSVFormat format;
        format.delimiter(' ');
        format.no_header();
        format.variable_columns(csv::VariableColumnPolicy::KEEP);

        csv::CSVReader reader(component_data.ini_file.string(), format);
        csv::CSVRow row;

        bool succeed = reader.read_row(row);
        if (false == succeed) {
            return tl::make_unexpected("Failed to read ini header");
        }

        size_t ini_N = row[0].get<size_t>();
        double ini_t = row[1].get<double>();
        spdlog::info("Header says: {} particles, time = {:.10f}", ini_N, ini_t);

        size_t prev_N = particles_data.pos.size();
        size_t N = prev_N + ini_N;
        particles_data.pos.reserve(N);
        particles_data.vel.reserve(N);
        particles_data.mass.reserve(N);
        particles_data.soft2.reserve(N);
        spdlog::debug("Current array sizes: {}", prev_N);
        spdlog::debug("New data size: {}", ini_N);
        spdlog::debug("New array sizes: {}", N);

        const double particle_mass = component_data.mass / N;
        const double particle_soft2 = component_data.soft * component_data.soft;

        auto& pos = particles_data.pos;
        auto& vel = particles_data.vel;
        auto& mass = particles_data.mass;
        auto& soft2 = particles_data.soft2;

        while (reader.read_row(row)) {
            pos.push_back(
                make_double3(
                    row[0].get<double>(),
                    row[1].get<double>(),
                    row[2].get<double>()
                )
            );

            vel.push_back(
                make_double3(
                    row[3].get<double>(),
                    row[4].get<double>(),
                    row[5].get<double>()
                )
            );

            mass.push_back(particle_mass);
            soft2.push_back(particle_soft2);

            if (pos.size() % 100'000 == 0) {
                spdlog::info("Read {}/{} row", pos.size(), N);
            }
        }

        if (pos.size() != N) {
            return tl::make_unexpected(
                fmt::format("Rows header/file mismatch ({}/{})", N, pos.size())
            );
        }

        return ini_N;
    }


    tl::expected<ParticlesData, std::string>
    read_simple_particles_data(
        const IniGalaxyData& ini_galaxy_data
    )
    {
        spdlog::info("Read particles data");

        try {
            ParticlesData particles_data;

            if (ini_galaxy_data.mb_ini_file_star) {
                IniGalaxyComponentData component_data;
                component_data.ini_file = ini_galaxy_data.mb_ini_file_star.value();
                component_data.mass = ini_galaxy_data.mass_star;
                component_data.soft = ini_galaxy_data.soft_star;

                auto expected_result = read_particles_data_component(component_data, particles_data);
                EXPECTED_CHECK(expected_result);
            }

            if (ini_galaxy_data.mb_ini_file_dark) {
                IniGalaxyComponentData component_data;
                component_data.ini_file = ini_galaxy_data.mb_ini_file_dark.value();
                component_data.mass = ini_galaxy_data.mass_dark;
                component_data.soft = ini_galaxy_data.soft_dark;

                auto expected_result = read_particles_data_component(component_data, particles_data);
                EXPECTED_CHECK(expected_result);
            }

            particles_data.info.ntotal = particles_data.pos.size();

            spdlog::info("{} particles loaded", particles_data.pos.size());
            if (particles_data.pos.size() % rrgsim::common::BLOCK_SIZE != 0) {
                spdlog::error("Particles count must be a multiple of {}", rrgsim::common::BLOCK_SIZE);
                return tl::make_unexpected("Unexpected particles data format");
            }

            return particles_data;
        }
        catch(const std::exception& ex) {
            return tl::make_unexpected(
                fmt::format("Can't read simple particles data: {}", ex.what())
            );
        }
    }

    tl::expected<ParticlesData, std::string>
    read_simple_particles_data(
        const fs::path& ini_json_path
    )
    {
        spdlog::info("Read particles data. Ini json path: {}", ini_json_path.string());

        try {
            std::ifstream stream{ ini_json_path };
            nlohmann::json ini_json; stream >> ini_json;

            return read_simple_particles_data(
                ini_json.get<IniGalaxyData>()
            );
        }
        catch(const std::exception& ex) {
            return tl::make_unexpected(
                fmt::format("Can't read simple particles data: {}", ex.what())
            );
        }
    }

} // namespace rrgsim::io