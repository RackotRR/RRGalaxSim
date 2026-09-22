#pragma once
#include <limits>

namespace rrgsim::common {

constexpr double NAN_VALUE = std::numeric_limits<double>::quiet_NaN();

#define PI 3.14159265358979
constexpr int BLOCK_SIZE = 256;

struct ParticlesInfo {
    int ntotal;
};

/// @brief Равномерная сетка
struct GridInfo {
    /// @brief Количество ячеек в одном измерении
    int nx;

    /// @brief Шаг по координате
    double dx;

    /// @brief Область моделирования в одном измерении
    double domain_l;
    double domain_min; // -domain_l * 0.5 <-- заполняется солвером при чтении
    double domain_max; // +domain_l * 0.5 <-- заполняется солвером при чтении

    /// @brief Область граничных условий: domain_l = bc_l + sim_l + bc_l
    double bc_l;

    /// @brief Основная область моделирования
    double sim_l;
};

/// @brief Параметры волновой модели
struct WaveParams {
    double diss_base;

    double diss_extra;

    double wave_speed;

    int setup_iterations;

    /// @brief Шаг по времени для интегрирования
    /// @note Выводится солвером из условия CFL
    double dt;
};

/// @brief Заполняется при инициализации контекста частиц
extern __constant__ ParticlesInfo particles_info_;

/// @brief Заполняется при инициализации контекста сетки
extern __constant__ GridInfo grid_info_;

/// @brief Заполняется при чтении параметров
extern __constant__ WaveParams wave_params_;

} // namespace rrgsim::common