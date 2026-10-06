#pragma once
#include <co_device_structs.cuh>
#include <co_types.h>

namespace rrgsim::common {

    struct SimParams {
        real time_max = NAN_VALUE;
        real dt_save = NAN_VALUE;
        real dt_dynamics = NAN_VALUE;

        bool use_wave_model = false;
    };


} // namespace rrgsim::common