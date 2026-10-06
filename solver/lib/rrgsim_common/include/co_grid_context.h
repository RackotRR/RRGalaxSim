#pragma once
#include <RR/CUDA/CuCommon.cuh>

#include <memory>
#include <vector>

#include <rrgsim_tl.h>

#include <co_device_structs.cuh>
#include <co_particles.h>
#include <co_types.h>

namespace rrgsim::common {
    using RR::CUDA::CuDarray;

    namespace detail {
        struct CellInfo{
            int count; // число частиц в ячейке
            int start_id; // накопленная сумма частиц (префиксная сумма) - идекс в массиве частиц
        };

        struct ParticleCellInfo {
            int cell_id; // индекс ячейки
            int id_in_cell; // индекс частицы в ячейке
        };
    } // namespace detail


    // контекст расчёта на GPU
    struct GridContext_ {
        real time = 0.; // current time

        CuDarray<real> mass_; // cell masses (GPU)
        CuDarray<real> grav_prev_; // cell gravitational potential (GPU)
        CuDarray<real> grav_curr_; // cell gravitational potential (GPU)
        CuDarray<real> grav_next_; // cell gravitational potential (GPU)
        CuDarray<real3> acc_; // acceleration (GPU)

        CuDarray<detail::ParticleCellInfo> particles_cell_info_;
        CuDarray<detail::CellInfo> cell_info_;
        CuDarray<int> cell_particles_count_;
        CuDarray<int> particles_in_block_;

        GridInfo info;
    };
    using sGridContext_ = std::shared_ptr<GridContext_>;

    // контекст для пост-процессинга на CPU (данные, которые копируются с шагом dt_save)
    struct GridContext {
        real time = 0.;
        std::vector<real> mass;
        std::vector<real> grav;

        void fill_device_data(
            const sGridContext_ context_
        );
    };
    using sGridContext = std::shared_ptr<GridContext>;


    sGridContext_
    initialize_grid_context_(
        GridInfo grid_info
    );

    sGridContext
    initialize_grid_context(
        GridInfo grid_info
    );

} // namespace rrgsim::common