#include <fstream>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "io_particles_bin.cuh"

namespace rrgsim::io {

    tl::expected<void, std::string>
    write_particles_dump(
        const rrgsim::common::sParticlesContext context,
        const fs::path& dump_dir
    )
    {
        spdlog::info("Write particles dump. Time: {:.5f}", context->time);

        try {
            std::string dump_name = fmt::format("S_{:>5d}.bin", context->iter);
            std::ofstream stream{ dump_dir / dump_name, std::ios::binary };

            const size_t N = context->pos.size();
            const double time = context->time;

            stream.write(reinterpret_cast<const char*>(&N), sizeof(N));
            stream.write(reinterpret_cast<const char*>(&time), sizeof(time));

            stream.write(reinterpret_cast<const char*>(context->pos.data()), sizeof(real3) * N);
            stream.write(reinterpret_cast<const char*>(context->vel.data()), sizeof(real3) * N);

            return tl::expected<void, std::string>{};
        }
        catch(const std::exception& ex) {
            return tl::make_unexpected(
                fmt::format("Can't write particles dump: {}", ex.what())
            );
        }
    }

    tl::expected<ParticlesData, std::string>
    read_particles_dump(
        const IniGalaxyData& ini_galaxy_data,
        size_t iteration,
        const fs::path& dump_dir
    )
    {
        spdlog::info("Read particles dump");

        try {
            std::string dump_name = fmt::format("S_{:>5d}.bin", iteration);
            std::ifstream stream{ dump_dir / dump_name, std::ios::binary };

            size_t N = 0;
            double time = 0;

            stream.read(reinterpret_cast<char*>(&N), sizeof(N));
            stream.read(reinterpret_cast<char*>(&time), sizeof(time));

            const real particle_mass = ini_galaxy_data.mass_star / N;
            const real particle_soft2 = ini_galaxy_data.soft_star * ini_galaxy_data.soft_star;

            ParticlesData particles_data;
            particles_data.info.ntotal = N;
            particles_data.mass.resize(N, particle_mass);
            particles_data.soft2.resize(N, particle_soft2);
            particles_data.pos.resize(N);
            particles_data.vel.resize(N);

            stream.read(reinterpret_cast<char*>(particles_data.pos.data()), sizeof(real3) * N);
            stream.read(reinterpret_cast<char*>(particles_data.vel.data()), sizeof(real3) * N);

            return particles_data;
        }
        catch(const std::exception& ex) {
            return tl::make_unexpected(
                fmt::format("Can't read particles dump: {}", ex.what())
            );
        }
    }
} // namespace rrgsim::io