#pragma once
#include <co_types.h>

namespace rrgsim::common {

    struct SimParams {
        real time_max = NAN_VALUE;
        real dt_save = NAN_VALUE;
        real dt_dynamics = NAN_VALUE;

        size_t saves_without_dump = std::numeric_limits<size_t>::max();

        bool use_wave_model = false;
    };


} // namespace rrgsim::common