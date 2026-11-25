#include <cuda_runtime.h>

#include <co_device_utils.cuh>
#include <co_device_structs.cuh>
#include <co_grid_context.cuh>

#include <spdlog/spdlog.h>

#include "m_mesh_solver_rect.cuh"

namespace rrgsim::mesh::rect {

using rrgsim::common::particles_info_;
using rrgsim::common::BLOCK_SIZE;
using rrgsim::common::detail::ParticleCellInfo;
using rrgsim::common::detail::CellInfo;
using rrgsim::common::OverInfo;
using RR::CUDA::CuDarray;

namespace {
    __global__ void assign_particles_to_cells_rect_(
        const real3* __restrict__ particles_pos,
        ParticleCellInfo* __restrict__ particles_cell_info,
        CellInfo* __restrict__ cellInfo,
        int* __restrict__ cellParticleCount
    )
    {
        int i_part = threadIdx.x + blockIdx.x * blockDim.x;

        real3 p = particles_pos[i_part];

        // Вычисление индексов ячейки
        int NX = rect_grid_info_.nx;
        int ix = rrgsim::common::clamp(
            static_cast<int>((p.x - rect_grid_info_.domain_min) / rect_grid_info_.dx),
            static_cast<int>(0),
            static_cast<int>(NX - 1));
        int iy = rrgsim::common::clamp(
            static_cast<int>((p.y - rect_grid_info_.domain_min) / rect_grid_info_.dx),
            static_cast<int>(0),
            static_cast<int>(NX - 1));
        int iz = rrgsim::common::clamp(
            static_cast<int>((p.z - rect_grid_info_.domain_min) / rect_grid_info_.dx),
            static_cast<int>(0),
            static_cast<int>(NX - 1));

        // Линейный индекс ячейки (Morton-подобный порядок для пространственной локальности)
        // Используем чередование битов для лучшей когерентности
        size_t i_cell = cell_index(ix, iy, iz);

        // Сохраняем номер ячейки для частицы
        particles_cell_info[i_part].cell_id = i_cell;

        // Атомарно увеличиваем счётчик частиц в ячейке
        int pos = atomicAdd(&cellParticleCount[i_cell], 1);

        // Сохраняем позицию внутри ячейки
        particles_cell_info[i_part].id_in_cell = pos;

        // Атомарно обновляем количество частиц в ячейке
        atomicAdd(&cellInfo[i_cell].count, 1);
    }

    /// @brief Вычисляет веса и индексы 8 узлов сетки для CIC-интерполяции в точке pos
    __device__ __forceinline__ DataCIC
    deposit_CIC_rect_(const real3& pos) {
        const int  nx         = rect_grid_info_.nx;
        const real dx         = rect_grid_info_.dx;
        const real inv_dx     = 1. / dx;
        const real domain_min = rect_grid_info_.domain_min;

        // Индексы ячейки и веса для трилинейной интерполяции
        using rrgsim::common::clamp;
        real u = clamp<real>((pos.x - domain_min) * inv_dx - 0.5, 0, nx - 1);
        real v = clamp<real>((pos.y - domain_min) * inv_dx - 0.5, 0, nx - 1);
        real w = clamp<real>((pos.z - domain_min) * inv_dx - 0.5, 0, nx - 1);

        int i0 = (int)u;
        int j0 = (int)v;
        int k0 = (int)w;

        int i1 = min(i0 + 1, nx - 1);
        int j1 = min(j0 + 1, nx - 1);
        int k1 = min(k0 + 1, nx - 1);

        real fx = u - i0;
        real fy = v - j0;
        real fz = w - k0;

        real wx0 = 1 - fx, wx1 = fx;
        real wy0 = 1 - fy, wy1 = fy;
        real wz0 = 1 - fz, wz1 = fz;

        DataCIC data;

        data.weights[0b000] = wx0*wy0*wz0;
        data.weights[0b100] = wx1*wy0*wz0;
        data.weights[0b010] = wx0*wy1*wz0;
        data.weights[0b110] = wx1*wy1*wz0;
        data.weights[0b001] = wx0*wy0*wz1;
        data.weights[0b101] = wx1*wy0*wz1;
        data.weights[0b011] = wx0*wy1*wz1;
        data.weights[0b111] = wx1*wy1*wz1;

        data.indices[0b000] = cell_index(i0, j0, k0);
        data.indices[0b100] = cell_index(i1, j0, k0);
        data.indices[0b010] = cell_index(i0, j1, k0);
        data.indices[0b110] = cell_index(i1, j1, k0);
        data.indices[0b001] = cell_index(i0, j0, k1);
        data.indices[0b101] = cell_index(i1, j0, k1);
        data.indices[0b011] = cell_index(i0, j1, k1);
        data.indices[0b111] = cell_index(i1, j1, k1);

        return data;
    }

    __global__ void deposit_real_CIC_rect_(
        const real3* __restrict__ particle_pos,
        const real*  __restrict__ particle_real,
        real*        __restrict__ cell_real
    )
    {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= particles_info_.ntotal) return;

        const real3 pos = particle_pos[p];
        const real  v   = particle_real[p];

        auto data_CIC = deposit_CIC_rect_(pos);

        #pragma unroll
        for (int i = 0; i < N_CIC; ++i) {
            size_t index = data_CIC.indices[i];
            real weight = data_CIC.weights[i];
            atomicAdd(&cell_real[index], v * weight);
        }
    }

    __global__ void deposit_real3_CIC_rect_(
        const real3* __restrict__ particle_pos,
        const real3* __restrict__ particle_real3,
        real3*       __restrict__ cell_real3
    )
    {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= particles_info_.ntotal) return;

        const real3 pos = particle_pos[p];
        const real3 v = particle_real3[p];

        auto data_CIC = deposit_CIC_rect_(pos);

        #pragma unroll
        for (int i = 0; i < N_CIC; ++i) {
            size_t index = data_CIC.indices[i];
            real weight = data_CIC.weights[i];
            atomicAdd(&cell_real3[index].x, v.x * weight);
            atomicAdd(&cell_real3[index].y, v.y * weight);
            atomicAdd(&cell_real3[index].z, v.z * weight);
        }
    }

    __global__ void project_real_CIC_rect_(
        const real3* __restrict__ particle_pos,
        const real*  __restrict__ cell_real,
        real*        __restrict__ particle_real
    )
    {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= particles_info_.ntotal) return;

        auto data_CIC = deposit_CIC_rect_(particle_pos[p]);

        real value = 0;

        #pragma unroll
        for (int i = 0; i < N_CIC; ++i) {
            size_t index = data_CIC.indices[i];
            real weight = data_CIC.weights[i];
            value += weight * cell_real[index];
        }

        particle_real[p] = value;
    }
    __global__ void project_real3_CIC_rect_(
        const real3* __restrict__ particle_pos,
        const real3* __restrict__ cell_real3,
        real3*       __restrict__ particle_real3
    )
    {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= particles_info_.ntotal) return;

        auto data_CIC = deposit_CIC_rect_(particle_pos[p]);

        real3 value = make_real3(0, 0, 0);

        #pragma unroll
        for (int i = 0; i < N_CIC; ++i) {
            size_t index = data_CIC.indices[i];
            real weight = data_CIC.weights[i];
            value.x += weight * cell_real3[index].x;
            value.y += weight * cell_real3[index].y;
            value.z += weight * cell_real3[index].z;
        }

        particle_real3[p] = value;
    }

    __global__ void mass_to_density_rect_(real* cell_rho) {
        const size_t n = rrgsim::common::cube((size_t)rect_grid_info_.nx);
        for (size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
            c < n;
            c += (size_t)gridDim.x * blockDim.x
        )
        {
            cell_rho[c] /= rrgsim::common::cube(rect_grid_info_.dx);
        }
    }

    __global__ void calc_acceleration_field_(
        real3* __restrict__ cell_acc, // сила на единицу массы
        const real* __restrict__ cell_phi
    )
    {
        int ix = threadIdx.x + blockIdx.x * blockDim.x;
        int iy = threadIdx.y + blockIdx.y * blockDim.y;
        int iz = threadIdx.z + blockIdx.z * blockDim.z;
        const int NX = rect_grid_info_.nx;
        const real DX = rect_grid_info_.dx;
        if (ix >= NX || iy >= NX || iz >= NX) return;

        size_t xyz = cell_index(ix, iy, iz);

        real3 dphi;

        if (ix == 0) {
            dphi.x = cell_phi[cell_index(ix + 1, iy, iz)] - cell_phi[xyz];
        }
        else if (ix == NX - 1) {
            dphi.x = cell_phi[xyz] - cell_phi[cell_index(ix - 1, iy, iz)];
        }
        else {
            dphi.x = 0.5 * (cell_phi[cell_index(ix + 1, iy, iz)] - cell_phi[cell_index(ix - 1, iy, iz)]);
        }

        if (iy == 0) {
            dphi.y = cell_phi[cell_index(ix, iy + 1, iz)] - cell_phi[xyz];
        }
        else if (iy == NX - 1) {
            dphi.y = cell_phi[xyz] - cell_phi[cell_index(ix, iy - 1, iz)];
        }
        else {
            dphi.y = 0.5 * (cell_phi[cell_index(ix, iy + 1, iz)] - cell_phi[cell_index(ix, iy - 1, iz)]);
        }

        if (iz == 0) {
            dphi.z = cell_phi[cell_index(ix, iy, iz + 1)] - cell_phi[xyz];
        }
        else if (iz == NX - 1) {
            dphi.z = cell_phi[xyz] - cell_phi[cell_index(ix, iy, iz - 1)];
        }
        else {
            dphi.z = 0.5 * (cell_phi[cell_index(ix, iy, iz + 1)] - cell_phi[cell_index(ix, iy, iz - 1)]);
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

        int NX = rect_grid_info_.nx;

        // Проверка границ
        if (i >= NX || j >= NX || k >= NX) return;

        // Координаты ячейки
        real x = rect_grid_info_.domain_min + (i + 0.5) * rect_grid_info_.dx;
        real y = rect_grid_info_.domain_min + (j + 0.5) * rect_grid_info_.dx;
        real z = rect_grid_info_.domain_min + (k + 0.5) * rect_grid_info_.dx;

        // Вектор от центра масс до точки
        real dx_pos = x - center_of_mass.x;
        real dy_pos = y - center_of_mass.y;
        real dz_pos = z - center_of_mass.z;

        // Квадрат расстояния
        real r2 = dx_pos * dx_pos + dy_pos * dy_pos + dz_pos * dz_pos;

        // Расстояние (с защитой от деления на ноль)
        real r = sqrt(r2);

        // Минимальное расстояние для избежания сингулярности
        const real softening = 0.01;
        r = max(r, softening);

        // Гравитационный потенциал: phi = -G * M / r
        const real G = 1.;
        potential[cell_index(i, j, k)] = -G * total_mass / r;
    }
} // namespace

void RectMeshSolver::init(
    const sGridContext_ grid_
)
{
    RR::CUDA::CuCopyToSymbol(
        grid_info(grid_),
        rrgsim::common::rect_grid_info_,
        RR::CUDA::ToDevice
    );
}

void RectMeshSolver::assign_particles_to_cells(
    const rrgsim::common::OverInfo& over,
    const sParticlesContext_ particles_,
    sGridContext_ grid_
)
{
    spdlog::trace("assign_particles_to_cells_rect_({}, {})", over.particles, over.blocks);
    RR::CUDA::CuCall(assign_particles_to_cells_rect_, over.particles, over.blocks) (
        particles_->pos_,
        grid_->particles_cell_info_,
        grid_->cell_info_,
        grid_->cell_particles_count_
    );
}

void RectMeshSolver::deposit_mass(
    const rrgsim::common::OverInfo& over,
    const sParticlesContext_ particles_,
    sGridContext_ grid_
)
{
    spdlog::trace("deposit_mass({}, {})", over.particles, over.blocks);
	RR::CUDA::CuCall(deposit_real_CIC_rect_, over.particles, over.blocks) (
		particles_->pos_,
		particles_->mass_,
		grid_->rho_
	);

    spdlog::trace("mass_to_density({}, {})", BLOCK_SIZE, BLOCK_SIZE);
    RR::CUDA::CuCall(mass_to_density_rect_, BLOCK_SIZE, BLOCK_SIZE) (
        grid_->rho_
    );
}

void RectMeshSolver::calc_acceleration_field(
    const sGridContext_ grid_
)
{
    spdlog::info("calc_acceleration_field");
    RR::CUDA::CuDeviceSync();

    using rrgsim::common::BLOCK_SIZE_IN_3D;

    unsigned blocks_count = static_cast<unsigned>(OverInfo::split_into_blocks(grid_info(grid_).nx, BLOCK_SIZE_IN_3D));
    dim3 threads_per_block{ BLOCK_SIZE_IN_3D, BLOCK_SIZE_IN_3D, BLOCK_SIZE_IN_3D };
    dim3 blocks_per_grid{ blocks_count, blocks_count, blocks_count };

    RR::CUDA::CuCall(calc_acceleration_field_, blocks_per_grid, threads_per_block) (
        grid_->acc_,
        grid_->grav_curr_
    );
}

void RectMeshSolver::project_particles_acceleration_on_grid(
    const sParticlesContext_ particles_,
    sGridContext_ grid_
)
{
    auto over = OverInfo::calc(
        rrgsim::common::get_grid_size(grid_->info),
        particles_->info.ntotal
    );

	grid_->acc_.set_zero();

	RR::CUDA::CuCall(deposit_real3_CIC_rect_, over.particles, over.blocks) (
		particles_->pos_,
		particles_->acc_,
		grid_->acc_
	);
}

void RectMeshSolver::project_particles_grav_on_grid(
    const sParticlesContext_ particles_,
    sGridContext_ grid_
)
{
    auto over = OverInfo::calc(
        rrgsim::common::get_grid_size(grid_->info),
        particles_->info.ntotal
    );

    grid_->grav_curr_.set_zero();

	RR::CUDA::CuCall(deposit_real_CIC_rect_, over.particles, over.blocks) (
		particles_->pos_,
		particles_->grav_,
		grid_->grav_curr_
	);
}

void RectMeshSolver::calc_grav_by_center_of_mass(
    rrgsim::common::sGridContext_ grid_,
    real3 center_of_mass,
    real total_mass
)
{
    using rrgsim::common::BLOCK_SIZE_IN_3D;
    using rrgsim::common::OverInfo;

    unsigned blocks_count = static_cast<unsigned>(OverInfo::split_into_blocks(grid_info(grid_).nx, BLOCK_SIZE_IN_3D));
    dim3 threads_per_block{ BLOCK_SIZE_IN_3D, BLOCK_SIZE_IN_3D, BLOCK_SIZE_IN_3D };
    dim3 blocks_per_grid{ blocks_count, blocks_count, blocks_count };

    grid_->grav_curr_.set_zero();
    RR::CUDA::CuCall(calc_grav_by_center_of_mass_, blocks_per_grid, threads_per_block) (
        grid_->grav_curr_,
        center_of_mass,
        total_mass
    );
}

const rrgsim::common::RectGridInfo&
RectMeshSolver::grid_info(
    const rrgsim::common::sGridContext_& grid_
)
{
    if (auto rect_grid_info = std::get_if<rrgsim::common::RectGridInfo>(&grid_->info); rect_grid_info) {
        return *rect_grid_info;
    }
    else {
        throw std::runtime_error{ "RectMeshSolver::grid_info met unexpected grid info variant" };
    }
}

void RectMeshSolver::project_grid_acceleration_on_particles(
    sGridContext_ grid_,
    sParticlesContext_ particles_
)
{
    auto over = OverInfo::calc(
        0,
        particles_->info.ntotal
    );

	RR::CUDA::CuCall(project_real3_CIC_rect_, over.particles, over.blocks) (
		particles_->pos_,
		grid_->acc_,
		particles_->acc_
	);
}

void RectMeshSolver::project_grid_grav_on_particles(
    const sGridContext_ grid_,
    sParticlesContext_ particles_
)
{
    auto over = OverInfo::calc(
        0,
        particles_->info.ntotal
    );

	RR::CUDA::CuCall(project_real_CIC_rect_, over.particles, over.blocks) (
		particles_->pos_,
		grid_->grav_curr_,
		particles_->grav_
	);
}

} // namespace rrgsim::rect_mesh