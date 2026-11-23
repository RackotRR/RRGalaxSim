#include "cm_p2mesh.cuh"
#include <co_info_structs.h>
#include <spdlog/spdlog.h>

namespace rrgsim::mesh {

    using rrgsim::common::BLOCK_SIZE;
    using rrgsim::common::OverInfo;
    using RR::CUDA::CuDarray;
    using rrgsim::common::detail::CellInfo;
    using rrgsim::common::detail::ParticleCellInfo;

    namespace detail {

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

    } // detail

void P2MeshSolver::init_p2mesh_context(
    sGridContext_ grid_,
    const sParticlesContext_ particles_
)
{
    tl::optional<OverInfo> mb_over;
    auto over = [&mb_over, &grid_, &particles_, this](void) -> const OverInfo& {
        if (false == mb_over.has_value()) {
            mb_over = OverInfo::calc(
                rrgsim::common::get_grid_size(grid_->info),
                particles_->info.ntotal
            );
        }
        return mb_over.value();
    };

    if (nullptr == grid_->particles_cell_info_) {
        spdlog::trace("init particles_cell_info_[{}]", over().num_particles);
        grid_->particles_cell_info_ = CuDarray<ParticleCellInfo>(over().num_particles);
        spdlog::trace("allocated: {}", CuDarray<real>::get_total_allocated_mb());
    }
    if (nullptr == grid_->cell_info_) {
        spdlog::trace("init cell_info_[{}]", over().num_cells);
        grid_->cell_info_ = CuDarray<CellInfo>(over().num_cells);
        spdlog::trace("allocated: {}", CuDarray<real>::get_total_allocated_mb());
    }
    if (nullptr == grid_->cell_particles_count_) {
        spdlog::trace("init cell_particles_count_[{}]", over().num_cells);
        grid_->cell_particles_count_ = CuDarray<int>(over().num_cells);
        spdlog::trace("allocated: {}", CuDarray<real>::get_total_allocated_mb());
    }
    if (nullptr == grid_->particles_in_block_) {
        spdlog::trace("init particles_in_block_[{}]", over().num_cell_blocks);
        grid_->particles_in_block_ = CuDarray<int>(over().num_cell_blocks);
        spdlog::trace("allocated: {}", CuDarray<real>::get_total_allocated_mb());
    }

    if (mb_over) {
        spdlog::info(
            "init_p2mesh_context: GPU memory occupied now - {} MB",
            CuDarray<real>::get_total_allocated_mb()
        );
    }
}

} // namespace rrgsim::mesh