#pragma once
#include <cuda_runtime.h>

#include <co_device_utils.cuh>
#include <co_device_structs.cuh>

namespace rrgsim::nbody {
	using rrgsim::common::BLOCK_SIZE;
	using rrgsim::common::particles_info_;
	using rrgsim::common::cube;
	using rrgsim::common::dot;

/// @brief Расчёт ускорения с учётом того, что количество частиц кратно размеру блока
__global__ void acceleration_kernel_blocked_(
	real3* acc,
	const real3* pos,
	const real* mass,
	const real* soft2
)
{
	__shared__ real3 pos_other[BLOCK_SIZE];
	__shared__ real mass_other[BLOCK_SIZE];
	__shared__ real soft2_other[BLOCK_SIZE];

	real3 f_sum = make_real3(0.0, 0.0, 0.0);

	const int i_curr_global = threadIdx.x + blockIdx.x * blockDim.x;
	const real3 p_curr = pos[i_curr_global];
	const real soft2_curr = soft2[i_curr_global];

	for (int block = 0; block < gridDim.x; block++) {
		const int i_other_global = threadIdx.x + block * blockDim.x;
		pos_other[threadIdx.x] = pos[i_other_global];
		mass_other[threadIdx.x] = mass[i_other_global];
		soft2_other[threadIdx.x] = soft2[i_other_global];

		__syncthreads();

		for (int i_other_local = 0; i_other_local < blockDim.x; ++i_other_local) {
			const real3 dp = make_real3(
				pos_other[i_other_local].x - p_curr.x,
				pos_other[i_other_local].y - p_curr.y,
				pos_other[i_other_local].z - p_curr.z
			);
			const real denominator = 1. / sqrt(
				dot(dp, dp) +
				0.5 * (soft2_curr + soft2_other[i_other_local])
			);
			const real k = mass_other[i_other_local] * cube(denominator);
			f_sum.x += dp.x * k;
			f_sum.y += dp.y * k;
			f_sum.z += dp.z * k;
		}

		__syncthreads();

	}

	acc[i_curr_global] = f_sum;
}

__global__ void grav_kernel_blocked_(
	real* grav,
	const real3* pos,
	const real* mass,
	const real* soft2
)
{
	__shared__ real3 pos_other[BLOCK_SIZE];
	__shared__ real mass_other[BLOCK_SIZE];
	__shared__ real soft2_other[BLOCK_SIZE];

	real grav_sum = 0.;

	const int i_curr_global = threadIdx.x + blockIdx.x * blockDim.x;
	const real3 p_curr = pos[i_curr_global];
	const real soft2_curr = soft2[i_curr_global];

	for (int block = 0; block < gridDim.x; block++) {
		const int i_other_global = threadIdx.x + block * blockDim.x;
		pos_other[threadIdx.x] = pos[i_other_global];
		mass_other[threadIdx.x] = mass[i_other_global];
		soft2_other[threadIdx.x] = soft2[i_other_global];

		__syncthreads();

		for (int i_other_local = 0; i_other_local < blockDim.x; ++i_other_local) {
			const real3 dp = make_real3(
				pos_other[i_other_local].x - p_curr.x,
				pos_other[i_other_local].y - p_curr.y,
				pos_other[i_other_local].z - p_curr.z
			);
			grav_sum += mass_other[i_other_local] / sqrt(
				dot(dp, dp) +
				0.5 * (soft2_curr + soft2_other[i_other_local])
			);
		}

		__syncthreads();

	}

	// некоторые источники предлагают исключать самогравитацию:
	// real grav_self = mass[i_curr_global] / sqrt(soft2_curr);
	// grav[i_curr_global] = grav_self - grav_sum;

	// но в исходном коде этого нет, и для совпадения не использую:
	grav[i_curr_global] = -grav_sum;
}

/// @brief Шаг по координате
/// @param acc Ускорение (a_{i})
/// @param vel Скорость (v_{i})
/// @param [in, out] pos Координата (p_{i} -> p_{i+1})
/// @param vel_predict Предположение по скорости (v_{i+1}^{*})
/// @param dt Шаг по времени
__global__ void predict_step_(
	const real3* acc,
	const real3* vel,
	real3* pos,
	real3* vel_predict,
	real dt
)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= particles_info_.ntotal) {
		return;
	}

	const real3 v = vel[i];
	const real3 r = pos[i];
	const real3 a = acc[i];

	// v_{i+1}^{*} = v_{i} + dt * a_{i}
	vel_predict[i] = make_real3(
		v.x + dt * a.x,
		v.y + dt * a.y,
		v.z + dt * a.z
	);

	// p_{i+1} = p_{i} + dt * 0.5(v_{i} + v_{i+1}^{*})
	pos[i] = make_real3(
		r.x + dt * 0.5 * (v.x + vel_predict[i].x),
		r.y + dt * 0.5 * (v.y + vel_predict[i].y),
		r.z + dt * 0.5 * (v.z + vel_predict[i].z)
	);
}

/// @brief Уточнение скорости
/// @param acc_new Ускорение в новой точке (a_{i+1})
/// @param vel Скорость (v_{i} -> v_{i+1})
/// @param vel_predict Предположение по скорости (v_{i+1}^{*})
/// @param dt Шаг по времени
__global__ void correct_step_(
	const real3* acc_new,
	real3* vel,
	const real3* vel_predict,
	real dt
)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= particles_info_.ntotal) {
		return;
	}

	const real3 v = vel[i];
	const real3 vv = vel_predict[i];
	const real3 aa = acc_new[i];

	// v_{i+1} = 0.5 * (v_i + v_{i+1}^{*}) + 0.5 * dt * a_{i+1}
	vel[i] = make_real3(
		0.5 * (v.x + vv.x + dt * aa.x),
		0.5 * (v.y + vv.y + dt * aa.y),
		0.5 * (v.z + vv.z + dt * aa.z)
	);
}


} // namespace rrgsim::nbody