#pragma once
#include <co_device_structs.cuh>
#include <co_grid_context.cuh>

namespace rrgsim::wave {

    void solve_wave_equation(
        rrgsim::common::sGridContext_ grid_,
        rrgsim::common::WaveParams wave_params
    );

    void fill_in_initial_grav(
        rrgsim::common::sGridContext_ grid_,
        real3 mass_center,
        real mass_value
    );

} // namespace rrgsim::wave