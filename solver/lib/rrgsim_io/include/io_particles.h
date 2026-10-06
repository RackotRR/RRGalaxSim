#pragma once
#include <vector>
#include <limits>
#include <filesystem>
#include <rrgsim_tl.h>
#include <cuda_runtime.h>
#include <co_particles.h>
#include <co_types.h>

namespace rrgsim::io {

    namespace fs = std::filesystem;
    using rrgsim::common::ParticlesData;

    constexpr real NAN_VALUE = std::numeric_limits<real>::quiet_NaN();

    struct IniGalaxyComponentData {
        real mass = NAN_VALUE;
        real soft = NAN_VALUE;
        fs::path ini_file;
    };

    struct IniGalaxyData {
        real mass_star = NAN_VALUE;
        real mass_dark = NAN_VALUE;
        real soft_star = NAN_VALUE;
        real soft_dark = NAN_VALUE;
        tl::optional<fs::path> mb_ini_file_star;
        tl::optional<fs::path> mb_ini_file_dark;
    };

    /// @brief
    /// @param component_data
    /// @param particles_data
    /// @return Количество прочитанных частиц, либо ошибка
    tl::expected<size_t, std::string>
    read_particles_data_component(
        const IniGalaxyComponentData& component_data,
        ParticlesData& particles_data
    );

    tl::expected<ParticlesData, std::string>
    read_simple_particles_data(
        const IniGalaxyData& galaxy_data
    );

    tl::expected<ParticlesData, std::string>
    read_simple_particles_data(
        const fs::path& ini_json_path
    );

} // namespace rrgsim_io