#pragma once
#include <memory>
#include <stdexcept>
#include "wave_common.cuh"
#include "wave_rect.cuh"
#include "wave_sph.cuh"

namespace rrgsim::wave {

    using sWaveSolver = std::shared_ptr<WaveSolver>;
    inline sWaveSolver make_wave_solver(
        const Params& params
    )
    {
        if (auto rect_grid_info = std::get_if<rrgsim::common::RectGridInfo>(&params.grid_info); rect_grid_info) {
            return std::make_shared<rrgsim::wave::rect::WaveRectSolver>(params);
        }
        else if (auto sph_grid_info = std::get_if<rrgsim::common::SphGridInfo>(&params.grid_info); sph_grid_info) {
            return std::make_shared<rrgsim::wave::sph::WaveSphSolver>(params);
        }
        else {
            throw std::runtime_error{ "make_wave_common_solver error: invalid grid info variant" };
        }
    }

} // namespace rrgsim::wave