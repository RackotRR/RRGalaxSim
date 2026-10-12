#include <cuda_runtime.h>
#include <co_device_utils.cuh>
#include <co_device_structs.cuh>
#include <spdlog/spdlog.h>

#include "nbody.cuh"

namespace rrgsim::nbody {

using rrgsim::common::OverInfo;
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

void predict_step(
    sParticlesContext_ context_,
    const SimParams& sim_params
)
{
    spdlog::debug("NBody::predict_step (t={})", context_->time);

    auto over = OverInfo::calc(0, context_->info.ntotal);
    RR::CUDA::CuCall(predict_step_, over.particles, over.blocks) (
        context_->acc_,
        context_->vel_,
        context_->pos_,
        context_->vel_predicted_,
        sim_params.dt_dynamics
    );
}

void nbody_acceleration(
    sParticlesContext_ context_
)
{
    spdlog::debug("NBody::acceleration (t={})", context_->time);

    auto over = OverInfo::calc(0, context_->info.ntotal);
    RR::CUDA::CuCall(acceleration_kernel_blocked_, over.particles, over.blocks) (
        context_->acc_,
        context_->pos_,
        context_->mass_,
        context_->soft2_
    );
}

void nbody_grav(
    sParticlesContext_ context_
)
{
    spdlog::debug("NBody::grav (t={})", context_->time);

    auto over = OverInfo::calc(0, context_->info.ntotal);
    RR::CUDA::CuCall(grav_kernel_blocked_, over.particles, over.blocks) (
        context_->grav_,
        context_->pos_,
        context_->mass_,
        context_->soft2_
    );
}

void correct_step(
    sParticlesContext_ context_,
    const SimParams& sim_params
)
{
    spdlog::debug("NBody::correct_step (t={})", context_->time);

    auto over = OverInfo::calc(0, context_->info.ntotal);
    RR::CUDA::CuCall(correct_step_, over.particles, over.blocks) (
        context_->acc_,
        context_->vel_,
        context_->vel_predicted_,
        sim_params.dt_dynamics
    );
}

ConservationInfo calc_conservation(
    sParticlesContext context
)
{
    spdlog::debug("NBody::conservation");
    const auto& pos = context->pos;
    const auto& vel = context->vel;
    const auto& mass = context->mass;
    const auto& grav = context->grav;
    if (vel.size() != mass.size()) {
        throw std::runtime_error("Conservation calculation error: velocity and mass arrays size mismatch");
    }
    const size_t N = vel.size();

    ConservationInfo info;
    info.momentum = real3(0., 0., 0.);
    info.angular = real3(0., 0., 0.);
    info.Ek = 0;
    info.Ep = 0;

    for (size_t i = 0; i < N; ++i) {
        const real x = pos[i].x;
        const real y = pos[i].y;
        const real z = pos[i].z;
        const real m = mass[i];
        const real vx = vel[i].x;
        const real vy = vel[i].y;
        const real vz = vel[i].z;

        info.momentum.x += m * vx;
        info.momentum.y += m * vy;
        info.momentum.z += m * vz;

        info.angular.x += m * (vz * y - vy * z);
        info.angular.y += m * (vx * z - vz * x);
        info.angular.z += m * (vy * x - vx * y);

        info.Ek += m * (vx * vx + vy * vy + vz * vz);
        info.Ep += m * grav[i];
    }

    info.Ek *= 0.5;
    info.Ep *= 0.5;
    info.time = context->time;
    return info;
}

/// @brief Ядро для расчета центра масс (первый проход - суммирование)
__global__ void compute_center_of_mass_kernel(
    const real3* __restrict__ positions,
    const real* __restrict__ masses,
    real3* __restrict__ partial_sums,  // Для блочного суммирования
    real* __restrict__ partial_mass_sums,
    int num_particles)
{
    // Динамическая разделяемая память для блочной редукции
    extern __shared__ char shared_mem[];
    real3* shared_pos = reinterpret_cast<real3*>(shared_mem);
    real* shared_mass = reinterpret_cast<real*>(shared_mem + blockDim.x * sizeof(real3));

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Инициализация
    real3 local_sum = make_real3(0.0f, 0.0f, 0.0f);
    real local_mass = 0.0f;

    // Загрузка и умножение массы на позицию
    if (idx < num_particles) {
        real m = masses[idx];
        real3 pos = positions[idx];
        local_sum.x = pos.x * m;
        local_sum.y = pos.y * m;
        local_sum.z = pos.z * m;
        local_mass = m;
    }

    shared_pos[tid] = local_sum;
    shared_mass[tid] = local_mass;
    __syncthreads();

    // Блочная редукция
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_pos[tid].x += shared_pos[tid + s].x;
            shared_pos[tid].y += shared_pos[tid + s].y;
            shared_pos[tid].z += shared_pos[tid + s].z;
            shared_mass[tid] += shared_mass[tid + s];
        }
        __syncthreads();
    }

    // Запись результата блока
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_pos[0];
        partial_mass_sums[blockIdx.x] = shared_mass[0];
    }
}

/// @brief Финальное ядро для суммирования частичных результатов
__global__ void finalize_center_of_mass_kernel(
    const real3* __restrict__ partial_sums,
    const real* __restrict__ partial_mass_sums,
    int num_blocks,
    real3* __restrict__ center_of_mass,
    real* __restrict__ total_mass)
{
    extern __shared__ char shared_mem[];
    real3* shared_pos = reinterpret_cast<real3*>(shared_mem);
    real* shared_mass = reinterpret_cast<real*>(shared_mem + blockDim.x * sizeof(real3));

    int tid = threadIdx.x;

    real3 local_sum = make_real3(0.0f, 0.0f, 0.0f);
    real local_mass = 0.0f;

    // Загрузка с шагом
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        local_sum.x += partial_sums[i].x;
        local_sum.y += partial_sums[i].y;
        local_sum.z += partial_sums[i].z;
        local_mass += partial_mass_sums[i];
    }

    shared_pos[tid] = local_sum;
    shared_mass[tid] = local_mass;
    __syncthreads();

    // Редукция
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_pos[tid].x += shared_pos[tid + s].x;
            shared_pos[tid].y += shared_pos[tid + s].y;
            shared_pos[tid].z += shared_pos[tid + s].z;
            shared_mass[tid] += shared_mass[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        real M = shared_mass[0];
        if (M > 0.0f) {
            center_of_mass->x = shared_pos[0].x / M;
            center_of_mass->y = shared_pos[0].y / M;
            center_of_mass->z = shared_pos[0].z / M;
        } else {
            *center_of_mass = make_real3(0.0f, 0.0f, 0.0f);
        }
        *total_mass = M;
    }
}

real3 calc_mass_center(
	const sParticlesContext_ context_
)
{
	spdlog::debug("calc_mass_center");

	auto over = OverInfo::calc(0, context_->info.ntotal);

	RR::CUDA::CuDarray<real3> d_partial_sums(over.num_particle_blocks);
	RR::CUDA::CuDarray<real> d_partial_mass_sums(over.num_particle_blocks);
	RR::CUDA::CuDarray<real3> d_center_of_mass_sums(1);
	RR::CUDA::CuDarray<real> d_total_mass(1);

	// Размер разделяемой памяти
	size_t shared_mem_size = BLOCK_SIZE * (sizeof(real) + sizeof(real3));

	// Первый проход
	RR::CUDA::CuCall(compute_center_of_mass_kernel, over.num_particle_blocks, BLOCK_SIZE, shared_mem_size) (
		context_->pos_,
		context_->mass_,
		d_partial_sums,
		d_partial_mass_sums,
		over.num_particles
	);

	// Финальный проход
	RR::CUDA::CuCall(finalize_center_of_mass_kernel, 1, BLOCK_SIZE, shared_mem_size) (
		d_partial_sums,
		d_partial_mass_sums,
		over.num_particle_blocks,
		d_center_of_mass_sums,
		d_total_mass
	);

	RR::CUDA::CuDeviceSync();

	auto center_of_mass = d_center_of_mass_sums.to_vector();
	auto total_mass = d_total_mass.to_vector();
	spdlog::trace("\t total_mass: {}", total_mass.front());
	spdlog::trace("\t center_of_mass: ({}, {}, {})", center_of_mass.front().x, center_of_mass.front().y, center_of_mass.front().z);
	return center_of_mass.front();
}

} // namespace rrgsim::nbody