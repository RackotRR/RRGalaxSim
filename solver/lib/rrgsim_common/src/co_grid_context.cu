#include "co_grid_context.cuh"
#include "co_device_utils.cuh"
#include "co_device_structs.cuh"
#include <spdlog/spdlog.h>

namespace rrgsim::common {

void GridContext::fill_device_data(
    const sGridContext_ context_
)
{
    context_->grav_curr_.to_vector(this->grav);
    context_->rho_.to_vector(this->rho);
    context_->acc_.to_vector(this->acc);

    this->time = context_->time;
    this->iter = context_->iter;
}

size_t get_grid_size(const GridInfoVariant& grid_info) {
    if (auto rect_grid_info = std::get_if<RectGridInfo>(&grid_info); rect_grid_info)  {
        return rrgsim::common::cube(rect_grid_info->nx);
    }
    else if (auto sph_grid_info = std::get_if<SphGridInfo>(&grid_info); sph_grid_info) {
        return sph_grid_info->nr *
                sph_grid_info->nt *
                sph_grid_info->np;
    }
    else {
        throw std::runtime_error{ "get_grid_size with invalid grid_info variant" };
    }
}

void fill_in_grid_info_(const GridInfoVariant& grid_info) {
    if (auto rect_grid_info = std::get_if<RectGridInfo>(&grid_info); rect_grid_info)  {
        RR::CUDA::CuCopyToSymbol(*rect_grid_info, rrgsim::common::rect_grid_info_, RR::CUDA::ToDevice);
    }
    else if (auto sph_grid_info = std::get_if<SphGridInfo>(&grid_info); sph_grid_info) {
        RR::CUDA::CuCopyToSymbol(*sph_grid_info, rrgsim::common::sph_grid_info_, RR::CUDA::ToDevice);
    }
    else {
        throw std::runtime_error{ "fill_in_grid_info_ with invalid grid_info variant" };
    }
}

sGridContext_
initialize_grid_context_(
    GridInfoVariant grid_info
)
{
    using RR::CUDA::CuDarray;
    sGridContext_ context_ = std::make_shared<GridContext_>();

    size_t N = get_grid_size(grid_info);

    context_->rho_ = CuDarray<real>(N);
    context_->grav_prev_ = CuDarray<real>(N);
    context_->grav_curr_ = CuDarray<real>(N);
    context_->grav_next_ = CuDarray<real>(N);
    context_->acc_ = CuDarray<real3>(N);

    context_->info = grid_info;
    fill_in_grid_info_(grid_info);

    spdlog::info("initialize_grid_context_: GPU memory occupied now - {} MB", CuDarray<real>::get_total_allocated_mb());

    return context_;
}

sGridContext
initialize_grid_context(
    GridInfoVariant grid_info
)
{
    spdlog::debug("initialize_grid_context");
    sGridContext context = std::make_shared<GridContext>();

    size_t N = get_grid_size(grid_info);
    context->grav = std::vector<real>(N);
    context->rho = std::vector<real>(N);
    context->info = grid_info;
    return context;
}

} // namespace rrgsim::common