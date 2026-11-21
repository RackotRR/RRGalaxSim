#pragma once
#include "wave_common.cuh"

namespace rrgsim::wave::sph {

class WaveSphSolver : public WaveSolver {
public:
    WaveSphSolver(Params params)
        : WaveSolver{ std::move(params) }
    {
    }

    void setup(
        rrgsim::common::sGridContext_ grid_
    ) override {}

    void step(
        rrgsim::common::sGridContext_ grid_
    ) override {}

private:
    dim3 threads_per_block;
    dim3 blocks_per_grid;
};

} // namespace rrgsim::wave::sph