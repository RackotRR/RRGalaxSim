#pragma once
#include "co_info_structs.h"
#include "co_boundary.cuh"

namespace rrgsim::common {

/// @brief Заполняется при инициализации контекста частиц
extern __constant__ ParticlesInfo particles_info_;

/// @brief Заполняется при инициализации контекста сферической сетки
extern __constant__ SphGridInfo sph_grid_info_;

/// @brief Заполняется при инициализации контекста прямоугольной сетки
extern __constant__ RectGridInfo rect_grid_info_;

/// @brief Заполняется при чтении параметров
extern __constant__ WaveParams wave_params_;

/// @brief Заполняется при чтении параметров
extern __constant__ BoundaryParams bc_params_;

} // namespace rrgsim::common