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

    /// Асимптотические ГУ: масса и центр масс подаются на вход
    void set_boundary_monopole(
        real total_mass,
        const real3& center_of_mass,
        bool radiative,
        real dx
    );

    /// Периодические ГУ: средняя плотность считается по сетке
    void set_boundary_periodic(
        real total_mass,
        const rrgsim::common::GridInfo& g,
        bool keep_layer = false
    );
} // namespace rrgsim::wave