#pragma once
#include "co_info_structs.h"

namespace rrgsim::common {

/// @brief Заполняется при инициализации контекста частиц
extern __constant__ ParticlesInfo particles_info_;

/// @brief Заполняется при инициализации контекста сетки
extern __constant__ GridInfo grid_info_;

/// @brief Заполняется при чтении параметров
extern __constant__ WaveParams wave_params_;


} // namespace rrgsim::common