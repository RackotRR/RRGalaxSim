#include "co_device_utils.cuh"
#include "co_device_structs.cuh"
#include "co_particles.h"

namespace rrgsim::common {

__constant__ ParticlesInfo particles_info_;

__constant__ GridInfo grid_info_;

__constant__ WaveParams wave_params_;


} // namespace rrgsim::common