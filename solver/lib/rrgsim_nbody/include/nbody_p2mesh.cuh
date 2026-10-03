#pragma once
#include <cuda_runtime.h>

#include <co_device_utils.cuh>
#include <co_device_structs.cuh>

#include <co_grid_context.h>

namespace rrgsim::nbody {
    using rrgsim::common::particles_info_;
    using rrgsim::common::grid_info_;
	using rrgsim::common::BLOCK_SIZE;
    using rrgsim::common::detail::ParticleCellInfo;
    using rrgsim::common::detail::CellInfo;

// ====================================================
// ЯДРО 2: ИНИЦИАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ МАССИВОВ
// ====================================================
__global__ void assignParticlesToCells(
	const double3* particles_pos, // [N] исходные частицы
    ParticleCellInfo* particles_cell_info, // [N] инфо: x=ячейка, y=индекс в ячейке
    CellInfo* cellInfo,     // [TOTAL_CELLS] для подсчёта частиц
    int* cellParticleCount, // [TOTAL_CELLS] счётчик для atomicAdd
    const int numParticles
)
{
    int i_part = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_part >= numParticles) return;

    double3 p = particles_pos[i_part];

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

/// @brief ЯДРО 3: ВЫЧИСЛЕНИЕ ЧАСТИЧНЫХ СУММ (КАСКАДНЫЙ АЛГОРИТМ)
__global__ void computePrefixSums(
    CellInfo* cellInfo,    // [TOTAL_CELLS]
    int* maxPBC,           // [TOTAL_CELLS/BLOCK_SIZE]
    const int numCells)
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

// ====================================================
// ЯДРО 4: КОРРЕКТИРОВКА ГЛОБАЛЬНЫХ ПРЕФИКСНЫХ СУММ
// ====================================================
__global__ void adjustGlobalPrefixSums(
    CellInfo* cellInfo,     // [TOTAL_CELLS]
    const int* maxPBC,      // [TOTAL_CELLS/BLOCK_SIZE]
    const int numCells)
{
    int i_cell = threadIdx.x + blockIdx.x * blockDim.x;
    int i_block = blockIdx.x;
    if (i_cell >= numCells) return;

    // Добавляем сумму всех предыдущих блоков
    if (i_block > 0) {
        cellInfo[i_cell].start_id += maxPBC[i_block - 1];
    }
}

// ====================================================
__global__ void computeCellMassesUnsorted(
    const double* particle_mass,   	   // [N] упорядоченные частицы
    ParticleCellInfo* particles_cell_info,// [N] инфо: x=ячейка, y=индекс в ячейке
    double* cellMasses,                // [TOTAL_CELLS] результат
    const int numParticles)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_particle >= numParticles) return;

    int i_cell = particles_cell_info[i_particle].cell_id;
	atomicAdd(&cellMasses[i_cell], particle_mass[i_particle]);
}

// ====================================================
__global__ void computeCellPhiUnsorted(
    const double* particle_phi,   	   // [N] упорядоченные частицы
    const ParticleCellInfo* particles_cell_info,// [N] инфо: x=ячейка, y=индекс в ячейке
    const CellInfo* cellInfo,     		 // [TOTAL_CELLS]
    double* cell_phi,                // [TOTAL_CELLS] результат
    const int numParticles)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_particle >= numParticles) return;

    int i_cell = particles_cell_info[i_particle].cell_id;
	atomicAdd(&cell_phi[i_cell], particle_phi[i_particle] / cellInfo[i_cell].count);
}

} // namespace rrgsim::nbody

