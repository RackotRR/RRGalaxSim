#pragma once
#include <memory>
#include <stdexcept>
#include "wave_common.cuh"
#include "wave_rect.cuh"
#include "wave_sph.cuh"

namespace rrgsim::wave {

    using sWaveCommonSolver = std::shared_ptr<WaveSolver>;
    inline sWaveCommonSolver make_wave_common_solver(
        const rrgsim::common::GridInfoVariant& grid_info
    )
    {
        if (auto rect_grid_info = std::get_if<rrgsim::common::RectGridInfo>(&grid_info); rect_grid_info) {
            return std::make_shared<rrgsim::wave::rect::WaveRectSolver>(*rect_grid_info);
        }
        else if (auto sph_grid_info = std::get_if<rrgsim::common::SphGridInfo>(&grid_info); sph_grid_info) {
            return std::make_shared<rrgsim::wave::sph::WaveSphSolver>(*sph_grid_info);
        }
        else {
            throw std::runtime_error{ "make_wave_common_solver error: invalid grid info variant" };
        }
    }

} // namespace rrgsim::wave