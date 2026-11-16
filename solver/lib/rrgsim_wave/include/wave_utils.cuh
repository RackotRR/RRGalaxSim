#pragma once
#include <co_info_structs.h>
#include <co_device_utils.cuh>

namespace rrgsim::wave {

inline double
calc_mean_density(
    double total_mass,
    const rrgsim::common::GridInfoVariant& grid_info
)
{
    using rrgsim::common::cube;

    struct GridVolumeCalculator {
        real operator()(const rrgsim::common::RectGridInfo& rect_grid) const {
            return cube(rect_grid.domain_l);
        }
        real operator()(const rrgsim::common::SphGridInfo& sph_grid) const {
            return 4. / 3. * PI * cube(sph_grid.radius_max);
        }
    } volume_calculator;

    real V = std::visit(
        volume_calculator,
        grid_info
    );

    return total_mass / V;
}

} // namespace rrgsim::wave