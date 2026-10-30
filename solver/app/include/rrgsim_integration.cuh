#pragma once
#include <co_sim_params.h>
#include <co_particles.h>
#include <co_particles_context.cuh>
#include <co_grid_context.cuh>
#include <rrgsim_tl.h>

namespace rrgsim::integration {

using rrgsim::common::ParticlesData;
using rrgsim::common::SimParams;
using rrgsim::common::WaveParams;
using rrgsim::common::GridInfo;

struct IntegrationParams {
    ParticlesData particles_data;
    SimParams sim_params;
    tl::optional<GridInfo> mb_grid_info;
    tl::optional<WaveParams> mb_wave_params;
};

class IntegrationSolver {
public:

    virtual ~IntegrationSolver() = default;

    // инициализация контекстов
    virtual void init(
        const IntegrationParams& params,
        rrgsim::common::sParticlesContext_ particles_context_,
        rrgsim::common::sGridContext_ grid_context_
    ) = 0;

    // расчёт потенциала и ускорения для первого шага
    virtual void precalc() = 0;

    // расчёт шага интегрирования по времени
    virtual void calc_step() = 0;

    // синхронизация частиц и сетки
    virtual void sync_particles_and_grid() = 0;
};

void integrate(
    IntegrationParams params
);

} // namespace rrgsim::integration