#pragma once
#include <co_device_structs.cuh>
#include <co_grid_context.h>

namespace rrgsim::wave {

    void solve_wave_equation(
        rrgsim::common::sGridContext_ grid_,
        rrgsim::common::WaveParams wave_params
    );

} // namespace rrgsim::wave