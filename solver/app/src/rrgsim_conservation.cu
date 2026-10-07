#include "rrgsim_conservation.cuh"
#include <spdlog/spdlog.h>
#include <co_device_utils.cuh>
#include <io_handler.h>

namespace rrgsim::conservation {

void save_angular_data_conservation(
    const rrgsim::nbody::ConservationInfo& info0,
    const rrgsim::nbody::ConservationInfo& info
)
{
    using rrgsim::io::TagValue;
    auto& io_handler = rrgsim::io::IOHandler::instance();

    {
        double ang_z_rel = info.angular.z / info0.angular.z - 1.;

        io_handler.append_table(
            "delta_ang_z",
            {
                { "t", info.time },
                { "ang_z", info.angular.z },
                { "ang_z_rel_signed", ang_z_rel },
                { "ang_z_rel", std::abs(ang_z_rel) }
            }
        );
    }

    {
        using rrgsim::common::norm;
        double ang_norm = norm(info.angular);
        double ang_norm0 = norm(info0.angular);
        double norm_rel = ang_norm / ang_norm0 - 1.;

        io_handler.append_table(
            "delta_ang_norm",
            {
                { "t", info.time },
                { "ang_norm", ang_norm },
                { "ang_norm_rel_signed", norm_rel },
                { "ang_norm_rel", std::abs(norm_rel) }
            }
        );
    }
}

void save_momentum_data_conservation(
    const rrgsim::nbody::ConservationInfo& info0,
    const rrgsim::nbody::ConservationInfo& info
)
{
    using rrgsim::io::TagValue;
    using rrgsim::common::norm;
    auto& io_handler = rrgsim::io::IOHandler::instance();

    double momentum_norm = norm(info.momentum);
    double momentum_norm0 = norm(info0.momentum);
    double momentum_norm_rel = momentum_norm / momentum_norm0 - 1.;

    io_handler.append_table(
        "delta_momentum",
        {
            { "t", info.time },
            { "momentum_norm", momentum_norm },
            { "momentum_diff", momentum_norm - momentum_norm0 },
            { "momentum_rel_signed", momentum_norm_rel },
            { "momentum_rel", std::abs(momentum_norm_rel) }
        }
    );
}

void save_energy_data_conservation(
    const rrgsim::nbody::ConservationInfo& info0,
    const rrgsim::nbody::ConservationInfo& info
)
{
    using rrgsim::io::TagValue;
    auto& io_handler = rrgsim::io::IOHandler::instance();

    double E_rel = info.E() / info0.E() - 1.;

    io_handler.append_table(
        "delta_energy",
        {
            { "t", info.time },
            { "E", info.E() },
            { "E_rel_signed", E_rel },
            { "E_rel", std::abs(E_rel) },
            { "Ek", info.Ek },
            { "Ep", info.Ep }
        }
    );
}

void print_conservation(
    const rrgsim::nbody::ConservationInfo& info
)
{
    spdlog::info("-- momentum:");
    spdlog::info("\t x {:.15f}", info.momentum.x);
    spdlog::info("\t y {:.15f}", info.momentum.y);
    spdlog::info("\t z {:.15f}", info.momentum.z);

    spdlog::info("-- angular momentum:");
    spdlog::info("\t x {:.15f}", info.angular.x);
    spdlog::info("\t y {:.15f}", info.angular.y);
    spdlog::info("\t z {:.15f}", info.angular.z);

    spdlog::info("-- energy: {:.15f}", info.E());
    spdlog::info("\t keenetic {:.15f}", info.Ek);
    spdlog::info("\t potential {:.15f}", info.Ep);
}
void print_conservation(
    const rrgsim::nbody::ConservationInfo& info0,
    const rrgsim::nbody::ConservationInfo& info
)
{
    using rrgsim::common::distance;
    using rrgsim::common::norm;

    double momentum_diff = distance(
        info.momentum,
        info0.momentum
    );
    double angular_momentum_diff = distance(
        info.angular,
        info0.angular
    );

    spdlog::info(
        "-- momentum: diff {} ; rel {}",
        momentum_diff,
        norm(info.momentum) / norm(info.momentum) - 1.
    );
    spdlog::trace(
        "\t {} -> {}",
        info0.momentum.x,
        info.momentum.x
    );
    spdlog::trace(
        "\t {} -> {}",
        info0.momentum.y,
        info.momentum.y
    );
    spdlog::trace(
        "\t {} -> {}",
        info0.momentum.z,
        info.momentum.z
    );

    spdlog::info(
        "-- angular momentum: diff {} ; rel {}",
        angular_momentum_diff,
        norm(info.angular) / norm(info0.angular) - 1.
    );
    spdlog::trace(
        "\t {} -> {}",
        info0.angular.x,
        info.angular.x
    );
    spdlog::trace(
        "\t {} -> {}",
        info0.angular.y,
        info.angular.y
    );
    spdlog::trace(
        "\t {} -> {}",
        info0.angular.z,
        info.angular.z
    );

    spdlog::info(
        "-- energy keenetic: diff {}", info.Ek - info0.Ek
    );
    spdlog::trace(
        "\t {} -> {}",
        info0.Ek,
        info.Ek
    );
    spdlog::info(
        "-- energy potential: diff {}", info.Ep - info0.Ep
    );
    spdlog::trace(
        "\t {} -> {}",
        info0.Ep,
        info.Ep
    );

    double energy_total = info.E();
    double energy_total0 = info0.E();
    spdlog::info(
        "-- energy total: diff {} ; rel {}",
        energy_total - energy_total0,
        energy_total / energy_total0 - 1.
    );
    spdlog::trace(
        "\t {} -> {}",
        energy_total0,
        energy_total
    );

    save_angular_data_conservation(info0, info);
    save_momentum_data_conservation(info0, info);
    save_energy_data_conservation(info0, info);
}

void check_bounds(
    const std::vector<real3>& particles_pos,
    const rrgsim::common::GridInfo& grid_info
)
{
    spdlog::info("Check bounds");
    if (particles_pos.empty()) {
        spdlog::debug("\t No particles_pos passed");
        return;
    }

    real max_x = particles_pos.front().x;
    real max_y = particles_pos.front().y;
    real max_z = particles_pos.front().z;
    real min_x = max_x;
    real min_y = max_y;
    real min_z = max_z;
    for (size_t i = 0; i < particles_pos.size(); ++i) {
        const real3& p = particles_pos[i];
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
        max_z = std::max(max_z, p.z);
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        min_z = std::min(min_z, p.z);
    }

    spdlog::info("\t X: [{} .. {}]", min_x, max_x);
    spdlog::info("\t Y: [{} .. {}]", min_y, max_y);
    spdlog::info("\t Z: [{} .. {}]", min_z, max_z);

#define MIN_BOUNDS_VIOLATION_MESSAGE "{0} min bounds violation! Particle {0}: {1}; domain_min: {2}"
#define MAX_BOUNDS_VIOLATION_MESSAGE "{0} max bounds violation! Particle {0}: {1}; domain_max: {2}"

    if (min_x < grid_info.domain_min) {
        spdlog::error(MIN_BOUNDS_VIOLATION_MESSAGE, "X", min_x, grid_info.domain_min);
    }
    if (min_y < grid_info.domain_min) {
        spdlog::error(MIN_BOUNDS_VIOLATION_MESSAGE, "Y", min_y, grid_info.domain_min);
    }
    if (min_z < grid_info.domain_min) {
        spdlog::error(MIN_BOUNDS_VIOLATION_MESSAGE, "Z", min_z, grid_info.domain_min);
    }
    if (max_x > grid_info.domain_max) {
        spdlog::error(MAX_BOUNDS_VIOLATION_MESSAGE, "X", max_x, grid_info.domain_max);
    }
    if (max_y > grid_info.domain_max) {
        spdlog::error(MAX_BOUNDS_VIOLATION_MESSAGE, "Y", max_y, grid_info.domain_max);
    }
    if (max_z > grid_info.domain_max) {
        spdlog::error(MAX_BOUNDS_VIOLATION_MESSAGE, "Z", max_z, grid_info.domain_max);
    }
}

} // namespace rrgsim::conservation