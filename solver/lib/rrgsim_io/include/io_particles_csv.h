#pragma once
#include <vector>

#include <co_particles.h>

#include "io_common.h"
#include "io_ini_galaxy_data.h"
#include <rrgsim_tl.h>

namespace rrgsim::io {

    using rrgsim::common::ParticlesData;

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