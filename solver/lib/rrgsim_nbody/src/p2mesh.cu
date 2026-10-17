#include <cuda_runtime.h>

#include <co_device_utils.cuh>
#include <co_device_structs.cuh>
#include <co_grid_context.cuh>

#include <spdlog/spdlog.h>
#include <rrgsim_tl.h>

#include "p2mesh.cuh"

#include <numeric>

namespace rrgsim::p2mesh {

using rrgsim::common::grid_info_;
using rrgsim::common::BLOCK_SIZE;
using rrgsim::common::detail::ParticleCellInfo;
using rrgsim::common::detail::CellInfo;
using rrgsim::common::OverInfo;
using rrgsim::common::SimParams;
using RR::CUDA::CuDarray;

namespace {
    __global__ void assign_particles_to_cells_(
        const real3* __restrict__ particles_pos,
        ParticleCellInfo* __restrict__ particles_cell_info,
        CellInfo* __restrict__ cellInfo,
        int* __restrict__ cellParticleCount
    )
    {
        int i_part = threadIdx.x + blockIdx.x * blockDim.x;

        real3 p = particles_pos[i_part];

        // Вычисление индексов ячейки
        int nx = grid_info_.nx;
        int ix = rrgsim::common::clamp(
            (int)((p.x - grid_info_.domain_min) / grid_info_.dx),
            (int)0,
            (int)(nx - 1));
        int iy = rrgsim::common::clamp(
            (int)((p.y - grid_info_.domain_min) / grid_info_.dx),
            (int)0,
            (int)(nx - 1));
        int iz = rrgsim::common::clamp(
            (int)((p.z - grid_info_.domain_min) / grid_info_.dx),
            (int)0,
            (int)(nx - 1));

        // Линейный индекс ячейки (Morton-подобный порядок для пространственной локальности)
        // Используем чередование битов для лучшей когерентности
        int i_cell = iz * nx * nx + iy * nx + ix;

        // Сохраняем номер ячейки для частицы
        particles_cell_info[i_part].cell_id = i_cell;

        // Атомарно увеличиваем счётчик частиц в ячейке
        int pos = atomicAdd(&cellParticleCount[i_cell], 1);

        // Сохраняем позицию внутри ячейки
        particles_cell_info[i_part].id_in_cell = pos;

        // Атомарно обновляем количество частиц в ячейке
        atomicAdd(&cellInfo[i_cell].count, 1);
    }

    __global__ void compute_prefix_sums_(
        CellInfo* __restrict__ cellInfo,
        int* __restrict__ maxPBC,
        const int numCells
    )
    {
        __shared__ int shared[BLOCK_SIZE];
        __shared__ int sharedPrev[BLOCK_SIZE];

        int i_cell = threadIdx.x + blockIdx.x * blockDim.x;
        int i_local = threadIdx.x;
        int i_block = blockIdx.x;

        // Загружаем количество частиц в ячейке
        if (i_cell >= numCells) {
            shared[i_local] = 0;
            sharedPrev[i_local] = 0;
        }
        else {
            int count = cellInfo[i_cell].count;
            shared[i_local]     = count;
            sharedPrev[i_local] = count;
        }

        __syncthreads();
        // Каскадное суммирование (параллельное сканирование)
        for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
            if (i_local + stride < BLOCK_SIZE) {
                shared[i_local + stride] += sharedPrev[i_local];
            }
            __syncthreads();

            // Обновляем предыдущие значения
            sharedPrev[i_local] = shared[i_local];
            __syncthreads();
        }

        // Первый поток в блоке сохраняет общую сумму блока
        if (i_local == 0) {
            int particles_in_block = shared[BLOCK_SIZE - 1];
            for (int i = i_block; i < gridDim.x; ++i) {
                atomicAdd(&(maxPBC[i]), particles_in_block);
            }
        }
    }

    __global__ void adjust_global_prefix_sums_(
        CellInfo* __restrict__ cellInfo,
        const int* __restrict__ maxPBC,
        const int numCells
    )
    {
        int i_cell = threadIdx.x + blockIdx.x * blockDim.x;
        int i_block = blockIdx.x;
        if (i_cell >= numCells) return;

        // Добавляем сумму всех предыдущих блоков
        if (i_block > 0) {
            cellInfo[i_cell].start_id += maxPBC[i_block - 1];
        }
    }

    __global__ void deposit_cell_mass_NGP_(
        const real* __restrict__ particle_mass,
        ParticleCellInfo* __restrict__ particles_cell_info,
        real* __restrict__ cellMasses
    )
    {
        int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
        int i_cell = particles_cell_info[i_particle].cell_id;
        atomicAdd(&cellMasses[i_cell], particle_mass[i_particle]);
    }

    /// @note Исходит из того, что масса всех частиц равна
    __global__ void deposit_cell_grav_NGP_(
        const real* __restrict__ particle_phi,
        const ParticleCellInfo* __restrict__ particles_cell_info,
        const CellInfo* __restrict__ cellInfo,
        real* __restrict__ cell_phi
    )
    {
        int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
        int i_cell = particles_cell_info[i_particle].cell_id;
        atomicAdd(&cell_phi[i_cell], particle_phi[i_particle] / cellInfo[i_cell].count);
    }

    /// @note Исходит из того, что масса всех частиц равна
    __global__ void deposit_cell_acc_NGP_(
        const real3* __restrict__ particle_acc,
        const ParticleCellInfo* __restrict__ particles_cell_info,
        const CellInfo* __restrict__ cellInfo,
        real3* __restrict__ cell_acc
    )
    {
        int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
        int i_cell = particles_cell_info[i_particle].cell_id;

        real3 a = particle_acc[i_particle];
        real inv_count = 1. / cellInfo[i_cell].count;

        atomicAdd(&cell_acc[i_cell].x, a.x * inv_count);
        atomicAdd(&cell_acc[i_cell].y, a.y * inv_count);
        atomicAdd(&cell_acc[i_cell].z, a.z * inv_count);
    }

    /// @brief CIC-депозиция массы частиц на сетку.
    /// Для каждой частицы раскладывает её массу по 8 ближайшим центрам ячеек
    /// с билинейными весами
    __global__ void deposit_cell_mass_CIC_(
        const real3* __restrict__ particle_pos,
        const real* __restrict__ particle_mass,
        real* __restrict__ cell_mass
    )
    {
        int p = blockIdx.x * blockDim.x + threadIdx.x;

        const int NX = grid_info_.nx;
        const real inv_dx = 1.f / grid_info_.dx;

        real3 pos = particle_pos[p];
        real  m   = particle_mass[p];

        // Физические координаты -> индексы центров ячеек (cell-centered, сдвиг -0.5)
        real u = (pos.x - grid_info_.domain_min) * inv_dx - 0.5f;
        real v = (pos.y - grid_info_.domain_min) * inv_dx - 0.5f;
        real w = (pos.z - grid_info_.domain_min) * inv_dx - 0.5f;

        // Клэмпим к допустимому диапазону центров ячеек [0, nx-1]
        const real eps = 1E-6;
        u = rrgsim::common::clamp<real>(u, 0, (NX - 1) - eps);
        v = rrgsim::common::clamp<real>(v, 0, (NX - 1) - eps);
        w = rrgsim::common::clamp<real>(w, 0, (NX - 1) - eps);

        int i0 = (int)floor(u);
        int j0 = (int)floor(v);
        int k0 = (int)floor(w);

        int i1 = min(i0 + 1, NX - 1);
        int j1 = min(j0 + 1, NX - 1);
        int k1 = min(k0 + 1, NX - 1);

        real fx = u - i0;
        real fy = v - j0;
        real fz = w - k0;

        // Веса CIC (билинейные)
        real wx0 = 1.f - fx, wx1 = fx;
        real wy0 = 1.f - fy, wy1 = fy;
        real wz0 = 1.f - fz, wz1 = fz;

        // 8 узлов, веса = произведения
        real w000 = wx0 * wy0 * wz0;
        real w100 = wx1 * wy0 * wz0;
        real w010 = wx0 * wy1 * wz0;
        real w110 = wx1 * wy1 * wz0;
        real w001 = wx0 * wy0 * wz1;
        real w101 = wx1 * wy0 * wz1;
        real w011 = wx0 * wy1 * wz1;
        real w111 = wx1 * wy1 * wz1;

        atomicAdd(&cell_mass[AT(i0, j0, k0)], m * w000);
        atomicAdd(&cell_mass[AT(i1, j0, k0)], m * w100);
        atomicAdd(&cell_mass[AT(i0, j1, k0)], m * w010);
        atomicAdd(&cell_mass[AT(i1, j1, k0)], m * w110);
        atomicAdd(&cell_mass[AT(i0, j0, k1)], m * w001);
        atomicAdd(&cell_mass[AT(i1, j0, k1)], m * w101);
        atomicAdd(&cell_mass[AT(i0, j1, k1)], m * w011);
        atomicAdd(&cell_mass[AT(i1, j1, k1)], m * w111);
    }

    __global__ void calc_acceleration_field_(
        real3* __restrict__ cell_acc, // сила на единицу массы
        const real* __restrict__ cell_phi
    )
    {
        int ix = threadIdx.x + blockIdx.x * blockDim.x;
        int iy = threadIdx.y + blockIdx.y * blockDim.y;
        int iz = threadIdx.z + blockIdx.z * blockDim.z;
        const int NX = grid_info_.nx;
        const real DX = grid_info_.dx;

        int xyz = AT(ix, iy, iz);

        real3 dphi;

        if (ix == 0) {
            dphi.x = cell_phi[AT(ix + 1, iy, iz)] - cell_phi[xyz];
        }
        else if (ix == NX - 1) {
            dphi.x = cell_phi[xyz] - cell_phi[AT(ix - 1, iy, iz)];
        }
        else {
            dphi.x = 0.5 * (cell_phi[AT(ix + 1, iy, iz)] - cell_phi[AT(ix - 1, iy, iz)]);
        }

        if (iy == 0) {
            dphi.y = cell_phi[AT(ix, iy + 1, iz)] - cell_phi[xyz];
        }
        else if (iy == NX - 1) {
            dphi.y = cell_phi[xyz] - cell_phi[AT(ix, iy - 1, iz)];
        }
        else {
            dphi.y = 0.5 * (cell_phi[AT(ix, iy + 1, iz)] - cell_phi[AT(ix, iy - 1, iz)]);
        }

        if (iz == 0) {
            dphi.z = cell_phi[AT(ix, iy, iz + 1)] - cell_phi[xyz];
        }
        else if (iz == NX - 1) {
            dphi.z = cell_phi[xyz] - cell_phi[AT(ix, iy, iz - 1)];
        }
        else {
            dphi.z = 0.5 * (cell_phi[AT(ix, iy, iz + 1)] - cell_phi[AT(ix, iy, iz - 1)]);
        }

        real coef = -1. / DX;
        dphi.x *= coef;
        dphi.y *= coef;
        dphi.z *= coef;

        cell_acc[xyz] = dphi;
    }

    /// @brief Ядро для расчета гравитационного потенциала
    /// Использует 3D сетку потоков
    __global__ void calc_grav_by_center_of_mass_(
        real* __restrict__ potential,
        real3 center_of_mass,
        real total_mass
    )
    {
        // 3D индексы
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;

        int nx = grid_info_.nx;

        // Проверка границ
        if (i >= nx || j >= nx || k >= nx) return;

        // Линейный индекс
        int idx = i + j * nx + k * nx * nx;

        // Координаты ячейки
        real x = grid_info_.domain_min + (i + 0.5) * grid_info_.dx;
        real y = grid_info_.domain_min + (j + 0.5) * grid_info_.dx;
        real z = grid_info_.domain_min + (k + 0.5) * grid_info_.dx;

        // Вектор от центра масс до точки
        real dx_pos = x - center_of_mass.x;
        real dy_pos = y - center_of_mass.y;
        real dz_pos = z - center_of_mass.z;

        // Квадрат расстояния
        real r2 = dx_pos * dx_pos + dy_pos * dy_pos + dz_pos * dz_pos;

        // Расстояние (с защитой от деления на ноль)
        real r = sqrtf(r2);

        // Минимальное расстояние для избежания сингулярности
        const real softening = 0.01;
        r = fmaxf(r, softening);

        // Гравитационный потенциал: phi = -G * M / r
        const real G = 1.;
        potential[idx] = -G * total_mass / r;
    }
} // namespace


void calc_acceleration_field(
    const sGridContext_ grid_
)
{
    spdlog::info("calc_acceleration_field");
    RR::CUDA::CuDeviceSync();

    using rrgsim::common::BLOCK_SIZE_IN_3D;

    unsigned blocks_count = static_cast<unsigned>(OverInfo::split_into_blocks(grid_->info.nx, BLOCK_SIZE_IN_3D));
    dim3 threads_per_block{ BLOCK_SIZE_IN_3D, BLOCK_SIZE_IN_3D, BLOCK_SIZE_IN_3D };
    dim3 blocks_per_grid{ blocks_count, blocks_count, blocks_count };

    RR::CUDA::CuCall(calc_acceleration_field_, blocks_per_grid, threads_per_block) (
        grid_->acc_,
        grid_->grav_curr_
    );
}

void init_p2mesh_context(
    sGridContext_ grid_,
    const sParticlesContext_ particles_
)
{
    tl::optional<OverInfo> mb_over;
    auto over = [&mb_over, &grid_, &particles_](void) -> const OverInfo& {
        if (false == mb_over.has_value()) {
            mb_over = OverInfo::calc(grid_->info.nx, particles_->info.ntotal);
        }
        return mb_over.value();
    };

    if (nullptr == grid_->particles_cell_info_) {
        grid_->particles_cell_info_ = CuDarray<ParticleCellInfo>(over().num_particles);
    }
    if (nullptr == grid_->cell_info_) {
        grid_->cell_info_ = CuDarray<CellInfo>(over().num_cells);
    }
    if (nullptr == grid_->cell_particles_count_) {
        grid_->cell_particles_count_ = CuDarray<int>(over().num_cells);
    }
    if (nullptr == grid_->particles_in_block_) {
        grid_->particles_in_block_ = CuDarray<int>(over().num_cell_blocks);
    }

    if (mb_over) {
        spdlog::info(
            "init_p2mesh_context: GPU memory occupied now - {} MB",
            CuDarray<real>::get_total_allocated_mb()
        );
    }
}

void convert_particles_to_grid(
    const sParticlesContext_ particles_,
    sGridContext_ grid_
)
{
    spdlog::debug("convert_particles_to_grid");
	RR::CUDA::CuDeviceSync();

    auto over = OverInfo::calc(grid_->info.nx, particles_->info.ntotal);

    init_p2mesh_context(grid_, particles_);

	grid_->particles_cell_info_.set_zero();
	grid_->cell_info_.set_zero();
	grid_->cell_particles_count_.set_zero();
	grid_->particles_in_block_.set_zero();
	grid_->mass_.set_zero();

    RR::CUDA::CuCall(assign_particles_to_cells_, over.particles, over.blocks) (
        particles_->pos_,
        grid_->particles_cell_info_,
        grid_->cell_info_,
        grid_->cell_particles_count_
    );
    RR::CUDA::CuCall(compute_prefix_sums_, over.cells, over.blocks) (
        grid_->cell_info_,
        grid_->particles_in_block_,
        over.num_cells
    );
    RR::CUDA::CuCall(adjust_global_prefix_sums_, over.cells, over.blocks) (
        grid_->cell_info_,
        grid_->particles_in_block_,
        over.num_cells
    );

	// RR::CUDA::CuCall(deposit_cell_mass_NGP_, over.particles, over.blocks) (
	// 	particles_->mass_,
	// 	grid_->particles_cell_info_,
	// 	grid_->mass_
	// );
	RR::CUDA::CuCall(deposit_cell_mass_CIC_, over.particles, over.blocks) (
		particles_->pos_,
		particles_->mass_,
		grid_->mass_
	);

}

void project_particles_acceleration_on_grid(
    const sParticlesContext_ particles_,
    sGridContext_ grid_
)
{
    auto over = OverInfo::calc(grid_->info.nx, particles_->info.ntotal);

	grid_->acc_.set_zero();

	RR::CUDA::CuCall(deposit_cell_acc_NGP_, over.particles, over.blocks) (
		particles_->acc_,
		grid_->particles_cell_info_,
		grid_->cell_info_,
		grid_->acc_
	);
}

void project_particles_grav_on_grid(
    const sParticlesContext_ particles_,
    sGridContext_ grid_
)
{
    auto over = OverInfo::calc(grid_->info.nx, particles_->info.ntotal);

    grid_->grav_curr_.set_zero();

	RR::CUDA::CuCall(deposit_cell_grav_NGP_, over.particles, over.blocks) (
		particles_->grav_,
		grid_->particles_cell_info_,
		grid_->cell_info_,
		grid_->grav_curr_
	);
}

void calc_grav_by_center_of_mass(
    rrgsim::common::sGridContext_ grid_,
    real3 center_of_mass,
    real total_mass
)
{
    using rrgsim::common::BLOCK_SIZE_IN_3D;
    using rrgsim::common::OverInfo;

    unsigned blocks_count = static_cast<unsigned>(OverInfo::split_into_blocks(grid_->info.nx, BLOCK_SIZE_IN_3D));
    dim3 threads_per_block{ BLOCK_SIZE_IN_3D, BLOCK_SIZE_IN_3D, BLOCK_SIZE_IN_3D };
    dim3 blocks_per_grid{ blocks_count, blocks_count, blocks_count };

    grid_->grav_curr_.set_zero();
    RR::CUDA::CuCall(calc_grav_by_center_of_mass_, blocks_per_grid, threads_per_block) (
        grid_->grav_curr_,
        center_of_mass,
        total_mass
    );
}

} // namespace rrgsim::p2mesh