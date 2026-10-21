#pragma once
#include <vector>
#include <cuda_runtime.h>

#include <co_particles.h>
#include <co_particles_context.cuh>

#include "io_common.h"
#include "io_ini_galaxy_data.h"

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

    tl::expected<void, std::string>
    write_particles_dump(
        const rrgsim::common::sParticlesContext context
    );

} // namespace rrgsim_io