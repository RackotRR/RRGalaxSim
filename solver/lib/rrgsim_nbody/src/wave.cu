#include "wave.h"
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <co_device_utils.cuh>

namespace rrgsim::wave {
	using rrgsim::common::BLOCK_SIZE;
	using rrgsim::common::particles_info_;
	using rrgsim::common::grid_info_;
	using rrgsim::common::wave_params_;
	using rrgsim::common::cube;
	using rrgsim::common::dot;


__global__ void wave_diss_iteration(
    real* phi_new,
    const real* phi,
    const real* phi_old,
    const real* mass
)
{
	const int NX =              grid_info_.nx;
	const real DX =           grid_info_.dx;
	const real SIM_L =        grid_info_.sim_l;
	const real BC_L =         grid_info_.bc_l;
	const real C =            wave_params_.wave_speed;
	const real DT =           wave_params_.dt;
	const real DISS_BASE =    wave_params_.diss_base;
	const real DISS_EXTRA =   wave_params_.diss_extra;

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

    int is_valid_idx =
           ix < NX
        && iy < NX
        && iz < NX;
    if (!is_valid_idx) return;


    real x = ix * DX;
    real y = iy * DX;
    real z = iz * DX;

	real rho = mass[AT(ix, iy, iz)] / (DX * DX * DX);
	const real G = 1.;
	real f = 4 * PI * G * rho;

    real diss = DISS_BASE;
#define _DISS_FUNC(x) (DISS_EXTRA * (x) * (x))

    if (x > -(SIM_L - BC_L)) {
        const real x_right = SIM_L - BC_L;
        diss += _DISS_FUNC(fabs(x - x_right));
    }
    else if (x < -(SIM_L - BC_L)) {
        const real x_left = -(SIM_L - BC_L);
        diss += _DISS_FUNC(fabs(x - x_left));
    }

    if (y > (SIM_L - BC_L)) {
        const real y_top = SIM_L - BC_L;
        diss += _DISS_FUNC(fabs(y - y_top));
    }
    else if (y < -(SIM_L - BC_L)) {
        const real y_bottom = -(SIM_L - BC_L);
        diss += _DISS_FUNC(fabs(y - y_bottom));
    }

    if (z > (SIM_L - BC_L)) {
        const real z_far = SIM_L - BC_L;
        diss += _DISS_FUNC(fabs(z - z_far));
    }
    else if (z < -(SIM_L - BC_L)) {
        const real z_near = -(SIM_L - BC_L);
        diss += _DISS_FUNC(fabs(z - z_near));
    }


#define _DIM 3
#define _CSQR (C * C)
#define _DT2 (DT * DT)
#define _DX2 (DX * DX)
#define _Q (diss * C * DT * 0.5)
#define _W (1 + _Q)
#define _K (_CSQR * _DT2 / _DX2)
#define _K_MAIN (_K / _W)
#define _K_F (-_CSQR * _DT2 / _W)
#define _K_ACTUAL (2 * (1. - _DIM * _K) / _W)
#define _K_OLD ((_Q - 1.) / _W)

#define _IS_I_EDGE(i) (i == 0 || i == NX - 1)
#define _IS_X_EDGE _IS_I_EDGE(ix)
#define _IS_Y_EDGE _IS_I_EDGE(iy)
#define _IS_Z_EDGE _IS_I_EDGE(iz)

    if (_IS_X_EDGE || _IS_Y_EDGE || _IS_Z_EDGE) {
        phi_new[AT(ix, iy, iz)] = 0.;
    }
    else {
        phi_new[AT(ix, iy, iz)] =
            phi[AT(ix - 1, iy, iz)] * _K_MAIN +
            phi[AT(ix + 1, iy, iz)] * _K_MAIN +
            phi[AT(ix, iy - 1, iz)] * _K_MAIN +
            phi[AT(ix, iy + 1, iz)] * _K_MAIN +
            phi[AT(ix, iy, iz - 1)] * _K_MAIN +
            phi[AT(ix, iy, iz + 1)] * _K_MAIN +
            phi[AT(ix, iy, iz)] * _K_ACTUAL +
            phi_old[AT(ix, iy, iz)] * _K_OLD +
            f * _K_F;
    }
}


void solve_wave_equation(
    rrgsim::common::sGridContext_ grid_,
    rrgsim::common::WaveParams wave_params
)
{
    spdlog::info("solve_wave_equation");

	for (int iter = 0; iter < wave_params.setup_iterations; ++iter) {
		if (iter && iter % 500 == 0) {
            spdlog::info("wave_diss_iteration step: {}", iter);
		}

        const int num_cells = rrgsim::common::cube(grid_->info.nx);
        const int num_cell_blocks = (num_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int over_cells = num_cell_blocks;
        const int over_blocks = BLOCK_SIZE;

        int nx = grid_->info.nx;

        unsigned threads_count = 8;
        unsigned blocks_count = static_cast<unsigned>(nx + threads_count - 1) / threads_count;
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
	}

    spdlog::info("solve_wave_equation succeed");
}

} // namespace rrgsim::wave