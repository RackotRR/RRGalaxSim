#pragma once
#include <co_device_structs.cuh>

namespace rrgsim::common {

    struct SimParams {
        double time_max = NAN_VALUE;
        double dt_save = NAN_VALUE;
        double dt_dynamics = NAN_VALUE;

        bool use_wave_model = false;
    };


} // namespace rrgsim::common