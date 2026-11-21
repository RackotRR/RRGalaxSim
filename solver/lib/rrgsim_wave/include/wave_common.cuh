#pragma once
#include <co_grid_context.cuh>

namespace rrgsim::wave {

struct MonopoleBoundaryParams {
    real total_mass = 0;
    real3 center_of_mass;
    bool is_radiative;
};
struct PeriodicBoundaryParams {
    real total_mass = 0;
    bool keep_extra_dissipative_layer = false;
};
using BoundaryParamsVariant = std::variant<MonopoleBoundaryParams, PeriodicBoundaryParams>;

struct NewtonianGravParams {
    real mass_value;
    real3 mass_center;
};

struct Params {
    rrgsim::common::WaveParams wave_params;

    rrgsim::common::GridInfoVariant grid_info;

    BoundaryParamsVariant boundary_params;

    tl::optional<NewtonianGravParams> mb_initial_grav_params;
};

class WaveSolver {
public:
    WaveSolver(Params params)
        : params{ params }
    {
        setup_boundary_params();
    }

    virtual void setup(
        rrgsim::common::sGridContext_ grid_
    ) = 0;
    virtual void step(
        rrgsim::common::sGridContext_ grid_
    ) = 0;

    void solve(
        rrgsim::common::sGridContext_ grid_,
        int max_iterations
    );

private:
    void setup_boundary_params() const;

protected:
    Params params;

};

} // namespace rrgsim::wave