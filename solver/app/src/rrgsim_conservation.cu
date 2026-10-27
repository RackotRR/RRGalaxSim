#include "rrgsim_conservation.cuh"
#include <spdlog/spdlog.h>
#include <fmt/format.h>
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
    int precision = 20;

    {
        double ang_z_rel = info.angular.z / info0.angular.z - 1.;

        io_handler.append_table(
            "delta_ang_z",
            {
                { "t", info.time },
                { "ang_z", info.angular.z, precision },
                { "ang_z_rel_signed", ang_z_rel, precision },
                { "ang_z_rel", std::abs(ang_z_rel), precision }
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
                { "ang_norm", ang_norm, precision },
                { "ang_norm_rel_signed", norm_rel, precision },
                { "ang_norm_rel", std::abs(norm_rel), precision }
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
    int precision = 20;

    double momentum_norm = norm(info.momentum);
    double momentum_norm0 = norm(info0.momentum);
    double momentum_norm_rel = momentum_norm / momentum_norm0 - 1.;

    io_handler.append_table(
        "delta_momentum",
        {
            { "t", info.time, precision },
            { "momentum_norm", momentum_norm, precision },
            { "momentum_diff", momentum_norm - momentum_norm0, precision },
            { "momentum_rel_signed", momentum_norm_rel, precision },
            { "momentum_rel", std::abs(momentum_norm_rel), precision }
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
    int precision = 15;

    double E_rel = info.E() / info0.E() - 1.;

    io_handler.append_table(
        "delta_energy",
        {
            { "t", info.time },
            { "E", info.E(), precision },
            { "E_rel_signed", E_rel, precision },
            { "E_rel", std::abs(E_rel), precision },
            { "Ek", info.Ek, precision },
            { "Ep", info.Ep, precision }
        }
    );
}

void print_conservation(
    const rrgsim::nbody::ConservationInfo& info
)
{
    using rrgsim::common::norm;
    spdlog::info("------------------------------------------");
    spdlog::info("Conservation time: {}", info.time);
    spdlog::info("-- momentum: {:.15f}", norm(info.momentum));
    spdlog::info("\t x {:.15f}", info.momentum.x);
    spdlog::info("\t y {:.15f}", info.momentum.y);
    spdlog::info("\t z {:.15f}", info.momentum.z);

    spdlog::info("-- angular momentum: {:.15f}", norm(info.angular));
    spdlog::info("\t x {:.15f}", info.angular.x);
    spdlog::info("\t y {:.15f}", info.angular.y);
    spdlog::info("\t z {:.15f}", info.angular.z);

    spdlog::info("-- energy: {:.15f}", info.E());
    spdlog::info("\t keenetic {:.15f}", info.Ek);
    spdlog::info("\t potential {:.15f}", info.Ep);
}

static void print_conservation_momentum(
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
    double momentum_rel = norm(info.momentum) / norm(info0.momentum) - 1.;
    double momentum_rel_max = 0.15;

    if (std::abs(momentum_rel) > momentum_rel_max) {
        spdlog::info(
            "-- momentum: diff {} ; rel {}",
            momentum_diff,
            momentum_rel
        );
    }
    else {
        spdlog::info(
            "-- momentum: diff {} ; rel {}",
            momentum_diff,
            momentum_rel
        );
    }
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
}
static void print_conservation_angular(
    const rrgsim::nbody::ConservationInfo& info0,
    const rrgsim::nbody::ConservationInfo& info
)
{
    using rrgsim::common::distance;
    using rrgsim::common::norm;

    double angular_momentum_diff = distance(
        info.angular,
        info0.angular
    );
    double angular_momentum_rel = norm(info.angular) / norm(info0.angular) - 1.;
    double angular_momentum_rel_max = 0.15;

    if (std::abs(angular_momentum_rel) > angular_momentum_rel_max) {
        spdlog::warn(
            "-- angular momentum: diff {} ; rel {}",
            angular_momentum_diff,
            angular_momentum_rel
        );
    }
    else {
        spdlog::info(
            "-- angular momentum: diff {} ; rel {}",
            angular_momentum_diff,
            angular_momentum_rel
        );
    }
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
}

static void print_conservation_energy(
    const rrgsim::nbody::ConservationInfo& info0,
    const rrgsim::nbody::ConservationInfo& info
)
{
    double energy_total = info.E();
    double energy_total0 = info0.E();

    double energy_total_diff = energy_total - energy_total0;
    double energy_total_rel = energy_total / energy_total0 - 1.;
    double energy_total_rel_max = 0.15;

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

    if (std::abs(energy_total_rel) > energy_total_rel_max) {
        spdlog::warn(
            "-- energy total: diff {} ; rel {}",
            energy_total_diff,
            energy_total_rel
        );
    }
    else {
        spdlog::info(
            "-- energy total: diff {} ; rel {}",
            energy_total_diff,
            energy_total_rel
        );
    }
    spdlog::trace(
        "\t {} -> {}",
        energy_total0,
        energy_total
    );
}

void print_conservation(
    const rrgsim::nbody::ConservationInfo& info0,
    const rrgsim::nbody::ConservationInfo& info
)
{
    spdlog::info("------------------------------------------");
    spdlog::info("Conservation time: {}", info.time);

    print_conservation_momentum(info0, info);
    print_conservation_angular(info0, info);
    print_conservation_energy(info0, info);

    save_angular_data_conservation(info0, info);
    save_momentum_data_conservation(info0, info);
    save_energy_data_conservation(info0, info);
}

void check_bounds(
    const std::vector<real3>& particles_pos,
    const rrgsim::common::GridInfo& grid_info
)
{
    spdlog::debug("Check bounds");
    if (particles_pos.empty()) {
        spdlog::trace("\t No particles_pos passed");
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

    spdlog::debug("\t X: [{} .. {}]", min_x, max_x);
    spdlog::debug("\t Y: [{} .. {}]", min_y, max_y);
    spdlog::debug("\t Z: [{} .. {}]", min_z, max_z);

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

void print_sample_particle(
    rrgsim::common::sParticlesContext particles_context
)
{
    if (particles_context->pos.size() > 0) {
        auto& io_handler = rrgsim::io::IOHandler::instance();
        auto& p = particles_context->pos[0];
        auto& v = particles_context->vel[0];
        auto& g = particles_context->grav[0];
        auto& a = particles_context->acc[0];

        rrgsim::io::TagValue i{ "i", 0 };
        rrgsim::io::TagValue x{ "x", p.x };
        rrgsim::io::TagValue y{ "y", p.y };
        rrgsim::io::TagValue z{ "z", p.z };
        rrgsim::io::TagValue vx{ "vx", v.x };
        rrgsim::io::TagValue vy{ "vy", v.y };
        rrgsim::io::TagValue vz{ "vz", v.z };
        rrgsim::io::TagValue grav{ "grav", g };
        rrgsim::io::TagValue ax{ "ax", a.x };
        rrgsim::io::TagValue ay{ "ay", a.y };
        rrgsim::io::TagValue az{ "az", a.z };

        io_handler.append_table("sample_particle", { i, x, y, z, vx, vy, vz, grav, ax, ay, az } );
    }
}
void print_acc_projection(
    rrgsim::common::sGridContext grid_context
)
{
    auto& handler = rrgsim::io::IOHandler::instance();
    size_t NX = grid_context->info.nx;

    tl::optional<std::string> subdir = "a_grid";

    for (size_t xi = 0; xi < NX; ++xi) {
        rrgsim::io::TagValue x{ "x", grid_context->info.domain_min + grid_context->info.dx * xi };
        rrgsim::io::TagValue ax{ "ax", grid_context->acc[AT(xi, NX / 2, NX / 2)].x };
        rrgsim::io::TagValue ay{ "ay", grid_context->acc[AT(xi, NX / 2, NX / 2)].y };
        rrgsim::io::TagValue az{ "az", grid_context->acc[AT(xi, NX / 2, NX / 2)].z };
        handler.append_table(
            fmt::format("a_grid_center_{:.08f}", grid_context->time),
            { x, ax, ay, az },
            subdir
        );
    }
}

void print_grav_projection(
    rrgsim::common::sGridContext grid_context
)
{
    auto& handler = rrgsim::io::IOHandler::instance();
    size_t NX = grid_context->info.nx;

    tl::optional<std::string> subdir = "grav_grid";

    for (size_t i = 0; i < NX; ++i) {
        rrgsim::io::TagValue r{ "r", grid_context->info.domain_min + grid_context->info.dx * i };
        rrgsim::io::TagValue gx{ "gx", grid_context->grav[AT(i, NX / 2, NX / 2)] };
        rrgsim::io::TagValue gy{ "gy", grid_context->grav[AT(NX / 2, i, NX / 2)] };
        rrgsim::io::TagValue gz{ "gz", grid_context->grav[AT(NX / 2, NX / 2, i)] };
        handler.append_table(
            fmt::format("grav_axis_center_{:.10f}", grid_context->time),
            { r, gx, gy, gz },
            subdir
        );
    }
}

void print_mass_projection(
    rrgsim::common::sGridContext grid_context
)
{
    auto& handler = rrgsim::io::IOHandler::instance();
    size_t NX = grid_context->info.nx;

    tl::optional<std::string> subdir = "mass_grid";

    for (size_t i = 0; i < NX; ++i) {
        rrgsim::io::TagValue r{ "r", grid_context->info.domain_min + grid_context->info.dx * i };
        rrgsim::io::TagValue mx{ "mx", grid_context->mass[AT(i, NX / 2, NX / 2)] };
        rrgsim::io::TagValue my{ "my", grid_context->mass[AT(NX / 2, i, NX / 2)] };
        rrgsim::io::TagValue mz{ "mz", grid_context->mass[AT(NX / 2, NX / 2, i)] };
        handler.append_table(
            fmt::format("mass_axis_center_{:.10f}", grid_context->time),
            { r, mx, my, mz },
            subdir
        );
    }
}

} // namespace rrgsim::conservation