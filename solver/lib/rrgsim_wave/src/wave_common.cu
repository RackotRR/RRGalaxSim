#include "wave_common.cuh"
#include "wave_steady_checker.cuh"
#include <spdlog/spdlog.h>
#include <m_mesh_helper.cuh>

namespace rrgsim::wave {

void WaveSolver::solve(
    rrgsim::common::sGridContext_ grid_,
    int max_iterations
)
{
    spdlog::debug("solve_wave_equation");
    RR::CUDA::CuDeviceSync();

    SteadyStateChecker steady_state_checker{
        params.wave_params.tolerance,
        params.wave_params.steady_hold_steps
    };

    auto print_iteration = [&](int iter) {
        double rel_diff = steady_state_checker.calc_max_diff(
            grid_->grav_curr_,
            grid_->grav_prev_
        );
        spdlog::debug("wave_diss_iteration step: {} / {} (rel diff: {})", iter, max_iterations, rel_diff);
    };

    bool converged = false;
	for (int iter = 0; iter < max_iterations; ++iter) {
		if (iter && iter % params.wave_params.print_step == 0) {
            print_iteration(iter);
		}

        this->step(grid_);

		swap(grid_->grav_prev_, grid_->grav_curr_);
		swap(grid_->grav_curr_, grid_->grav_next_);

        // остановка по критерию установления
        if (iter && iter % params.wave_params.steady_check_step == 0) {
            double rel_diff = steady_state_checker.calc_max_diff(
                grid_->grav_curr_,
                grid_->grav_prev_
            );
            spdlog::trace("wave_diss_iteration check convergence: {} (rel diff: {})", iter, rel_diff);

            converged = steady_state_checker.check(rel_diff);
            if (converged) {
                spdlog::debug("solve_wave_equation: steady_check stopped at {} step", iter);
                break;
            }
        }

	}

    // если не сошлось, мы здесь по превышению количества итераций
    if (false == converged) {
        print_iteration(max_iterations);
    }

    spdlog::debug("solve_wave_equation finished");
}

void WaveSolver::setup_boundary_params() const {
    using rrgsim::common::bc_params_;
    using rrgsim::common::BoundaryParams;
    using rrgsim::common::BoundaryType;

    auto mesh_helper = rrgsim::mesh::make_mesh_helper(params.grid_info);

    if (auto monopole_boundary = std::get_if<MonopoleBoundaryParams>(&params.boundary_params); monopole_boundary) {
        BoundaryParams p;
        p.type = monopole_boundary->is_radiative
            ? BoundaryType::MonopoleRadiative
            : BoundaryType::MonopoleDirichlet;
        p.mass = monopole_boundary->total_mass;
        p.center_of_mass = monopole_boundary->center_of_mass;
        p.r_min = real(0.5) * mesh_helper->calc_min_dr();
        RR::CUDA::CuCopyToSymbol(p, bc_params_, RR::CUDA::ToDevice);
    }
    else if (auto periodic_boundary = std::get_if<PeriodicBoundaryParams>(&params.boundary_params); periodic_boundary) {
        BoundaryParams p;
        p.type = BoundaryType::Periodic;
        p.mass = periodic_boundary->total_mass;
        p.rho_mean = periodic_boundary->total_mass / mesh_helper->calc_volume();
        p.periodic_keep_layer = static_cast<int>(periodic_boundary->keep_extra_dissipative_layer);
        RR::CUDA::CuCopyToSymbol(p, bc_params_, RR::CUDA::ToDevice);
    }
    else {
        throw std::runtime_error{ "WaveEquationSolver::setup_boundary_params: invalid boundary params variant" };
    }
}

} // namespace rrgsim::wave