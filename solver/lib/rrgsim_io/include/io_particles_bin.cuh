#pragma once
#include <vector>
#include <cuda_runtime.h>

#include <co_particles.h>
#include <co_particles_context.cuh>

#include "io_common.h"
#include "io_ini_galaxy_data.h"

namespace rrgsim::io {

    using rrgsim::common::ParticlesData;

    tl::expected<void, std::string>
    write_particles_dump(
        const rrgsim::common::sParticlesContext context,
        const fs::path& dump_dir
    );

    tl::expected<ParticlesData, std::string>
    read_particles_dump(
        const IniGalaxyData& ini_galaxy_data,
        size_t iteration,
        const fs::path& dump_dir
    );

} // namespace rrgsim_io