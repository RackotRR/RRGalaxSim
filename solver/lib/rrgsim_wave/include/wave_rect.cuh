#pragma once
#include "wave_common.cuh"

namespace rrgsim::wave::rect {

class WaveRectSolver : public WaveSolver {
public:
    WaveRectSolver(Params params);

    void setup(
        rrgsim::common::sGridContext_ grid_
    ) override;

    void step(
        rrgsim::common::sGridContext_ grid_
    ) override;

private:
    void fill_in_initial_grav(
        rrgsim::common::sGridContext_ grid_,
        const NewtonianGravParams& initial_grav_params
    ) const;

private:
    dim3 threads_per_block;
    dim3 blocks_per_grid;
};

} // namespace rrgsim::wave::rect