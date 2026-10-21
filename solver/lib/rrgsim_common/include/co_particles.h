#pragma once
#include <vector>
#include "co_types.h"
#include "co_info_structs.h"

namespace rrgsim::common {

    struct ParticlesData {
        std::vector<real3> pos;
        std::vector<real3> vel;
        std::vector<real> mass;
        std::vector<real> soft2;

        ParticlesInfo info;
    };

} // rrgsim::common