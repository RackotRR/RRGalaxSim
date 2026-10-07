#pragma once
#include <limits>
#include <co_types.h>

namespace rrgsim::common {

constexpr real NAN_VALUE = std::numeric_limits<real>::quiet_NaN();

#define PI 3.14159265358979
constexpr int BLOCK_SIZE = 256;
constexpr int BLOCK_SIZE_IN_3D = 8;

struct ParticlesInfo {
    int ntotal;
};

/// @brief Равномерная сетка
struct GridInfo {
    /// @brief Количество ячеек в одном измерении
    int nx;

    /// @brief Шаг по координате
    real dx;

    /// @brief Область моделирования в одном измерении
    real domain_l;
    real domain_min; // -domain_l * 0.5 <-- заполняется солвером при чтении
    real domain_max; // +domain_l * 0.5 <-- заполняется солвером при чтении

    /// @brief Область граничных условий: domain_l = bc_l + sim_l + bc_l
    real bc_l;

    /// @brief Основная область моделирования
    real sim_l;
};

/// @brief Параметры волновой модели
struct WaveParams {
    real diss_base;

    real diss_extra;

    real wave_speed;

    int setup_iterations;

    /// @brief Шаг по времени для интегрирования
    /// @note Выводится солвером из условия CFL
    real dt;
};

/// @brief Заполняется при инициализации контекста частиц
extern __constant__ ParticlesInfo particles_info_;

/// @brief Заполняется при инициализации контекста сетки
extern __constant__ GridInfo grid_info_;

/// @brief Заполняется при чтении параметров
extern __constant__ WaveParams wave_params_;

class OverInfo {
public:
    int num_cells = 0;
    int num_particles = 0;
    int num_cell_blocks = 0;
    int num_particle_blocks = 0;
    int /*over*/ particles = 0;
    int /*over*/ cells = 0;
    int /*over*/ blocks = 0;

    static int split_into_blocks(int ntotal, int block_size) {
        return (ntotal + block_size - 1) / block_size;
    }

    static OverInfo calc(int grid_nx, int particles_count) {
        OverInfo over;
        over.num_cells = grid_nx * grid_nx * grid_nx;
        over.num_particles = particles_count;
        over.num_cell_blocks = split_into_blocks(over.num_cells, BLOCK_SIZE);
        over.num_particle_blocks = split_into_blocks(over.num_particles, BLOCK_SIZE);
        over.particles = over.num_particle_blocks;
        over.cells = over.num_cell_blocks;
        over.blocks = BLOCK_SIZE;
        return over;
    }
private:
    OverInfo() = default;
};

} // namespace rrgsim::common