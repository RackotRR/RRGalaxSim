#include "wave.cuh"
#include "wave_utils.cuh"
#include "wave_steady_checker.cuh"

#include <spdlog/spdlog.h>
#include <co_device_utils.cuh>
#include <co_device_structs.cuh>

namespace rrgsim::wave {
	using rrgsim::common::BLOCK_SIZE;
	using rrgsim::common::particles_info_;
	using rrgsim::common::wave_params_;
	using rrgsim::common::bc_params_;
	using rrgsim::common::distance;
	using rrgsim::common::cube;
	using rrgsim::common::dot;



/// Асимптотические ГУ: масса и центр масс подаются на вход
void set_boundary_monopole(
    real total_mass,
    const real3& center_of_mass,
    bool radiative,
    real dx
)
{
    BoundaryParams p{};
    p.type  = radiative
        ? BoundaryType::MonopoleRadiative
        : BoundaryType::MonopoleDirichlet;
    p.mass  = total_mass;
    p.center_of_mass = center_of_mass;
    p.r_min = real(0.5) * dx;
    RR::CUDA::CuCopyToSymbol(p, bc_params_, RR::CUDA::ToDevice);
}

/// Периодические ГУ: средняя плотность считается по сетке
void set_boundary_periodic(
    real total_mass,
    const rrgsim::common::GridInfoVariant& g,
    bool keep_layer
)
{
    BoundaryParams p{};
    p.type                = BoundaryType::Periodic;
    p.mass                = total_mass;
    p.rho_mean            = calc_mean_density(total_mass, g);
    p.periodic_keep_layer = keep_layer ? 1 : 0;
    RR::CUDA::CuCopyToSymbol(p, bc_params_, RR::CUDA::ToDevice);
}

void solve_wave_equation(
    rrgsim::common::sGridContext_ grid_,
    rrgsim::common::WaveParams wave_params,
    int max_iterations
)
{
    spdlog::debug("solve_wave_equation");
    RR::CUDA::CuDeviceSync();

    SteadyStateChecker steady_state_checker{
        wave_params.tolerance,
        wave_params.steady_hold_steps
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
		if (iter && iter % wave_params.print_step == 0) {
            print_iteration(iter);
		}

        using rrgsim::common::BLOCK_SIZE_IN_3D;
        using rrgsim::common::OverInfo;

        unsigned threads_count = BLOCK_SIZE_IN_3D;
        unsigned blocks_count = static_cast<unsigned>(OverInfo::split_into_blocks(grid_->info.nx, BLOCK_SIZE_IN_3D));
        dim3 threads_per_block{ threads_count, threads_count, threads_count };
        dim3 blocks_per_grid{ blocks_count, blocks_count, blocks_count };
		RR::CUDA::CuCall(wave_diss_iteration, blocks_per_grid, threads_per_block) (
            grid_->grav_next_,
            grid_->grav_curr_,
            grid_->grav_prev_,
            grid_->mass_
		);
		swap(grid_->grav_prev_, grid_->grav_curr_);
		swap(grid_->grav_curr_, grid_->grav_next_);

        // остановка по критерию установления
        if (iter && iter % wave_params.steady_check_step == 0) {
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



__global__ void fill_in_initial_grav_(
    real* grav,
    real3 mass_center,
    real mass_value
)
{
	const int NX = grid_info_.nx;
	const real DX = grid_info_.dx;

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;
    if (ix >= NX || iy >= NX || iz >= NX) return;

    real3 pos = make_real3(
        grid_info_.domain_min + ix * DX,
        grid_info_.domain_min + iy * DX,
        grid_info_.domain_min + iz * DX
    );

    const real G = 1.;
    const real softening = 0.01;

    grav[AT(ix, iy, iz)] = -G * mass_value / (rrgsim::common::distance(pos, mass_center) + softening);
}

void fill_in_initial_grav(
    rrgsim::common::sGridContext_ grid_,
    real3 mass_center,
    real mass_value
)
{
    spdlog::debug("fill_in_initial_grav");

    using rrgsim::common::BLOCK_SIZE_IN_3D;
    using rrgsim::common::OverInfo;

    unsigned threads_count = BLOCK_SIZE_IN_3D;
    unsigned blocks_count = static_cast<unsigned>(OverInfo::split_into_blocks(grid_->info.nx, BLOCK_SIZE_IN_3D));
    dim3 threads_per_block{ threads_count, threads_count, threads_count };
    dim3 blocks_per_grid{ blocks_count, blocks_count, blocks_count };

    spdlog::trace("fill_in_initial_grav_([{}], [{}])", blocks_per_grid.x, threads_per_block.x);
    RR::CUDA::CuCall(fill_in_initial_grav_, blocks_per_grid, threads_per_block) (
        grid_->grav_curr_,
        mass_center,
        mass_value
    );

    RR::CUDA::CuCopyFromTo(
        grid_->grav_curr_,
        grid_->grav_prev_,
        RR::CUDA::DeviceToDevice
    );
}

} // namespace rrgsim::wave