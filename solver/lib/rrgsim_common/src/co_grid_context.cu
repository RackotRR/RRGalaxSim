#include "co_grid_context.h"
#include "co_device_utils.cuh"
#include "co_device_structs.cuh"
#include <spdlog/spdlog.h>

namespace rrgsim::common {

void GridContext::fill_device_data(
    const sGridContext_ context_
)
{
    context_->grav_curr_.to_vector(this->grav);
    context_->mass_.to_vector(this->mass);
}

sGridContext_
initialize_grid_context_(
    GridInfo grid_info
)
{
    using RR::CUDA::CuDarray;
    sGridContext_ context_ = std::make_shared<GridContext_>();

    size_t nx3 = rrgsim::common::cube(grid_info.nx);

    context_->mass_ = CuDarray<real>(nx3);
    context_->grav_prev_ = CuDarray<real>(nx3);
    context_->grav_curr_ = CuDarray<real>(nx3);
    context_->grav_next_ = CuDarray<real>(nx3);
    context_->acc_ = CuDarray<real3>(nx3);

    context_->info = grid_info;
    RR::CUDA::CuCopyToSymbol(context_->info, rrgsim::common::grid_info_, RR::CUDA::ToDevice);

    spdlog::info("initialize_grid_context_: GPU memory occupied now - {} MB", CuDarray<real>::get_total_allocated_mb());

    return context_;
}

sGridContext
initialize_grid_context(
    GridInfo grid_info
)
{
    sGridContext context = std::make_shared<GridContext>();

    size_t nx3 = rrgsim::common::cube(grid_info.nx);
    context->grav = std::vector<real>(nx3);
    context->mass = std::vector<real>(nx3);
    return context;
}

} // namespace rrgsim::common