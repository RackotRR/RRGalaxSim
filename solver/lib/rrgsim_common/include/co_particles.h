#pragma once
#include "cuda_runtime.h"
#include <vector>
#include "co_device_structs.cuh"
#include "co_types.h"

namespace rrgsim::common {

    struct ParticlesData {
        std::vector<real3> pos;
        std::vector<real3> vel;
        std::vector<real> mass;
        std::vector<real> soft2;

        ParticlesInfo info;
    };

} // rrgsim::common