#include "sm_p2mesh.cuh"
#include <co_device_utils.cuh>
#include <spdlog/spdlog.h>

namespace rrgsim::mesh::sph {

using rrgsim::common::sph_grid_info_;
using rrgsim::common::BLOCK_SIZE;
using rrgsim::common::detail::ParticleCellInfo;
using rrgsim::common::detail::CellInfo;
using rrgsim::common::OverInfo;
using RR::CUDA::CuDarray;

namespace {

    __global__ void assign_particles_to_cells_spherical_(
        const real3* __restrict__ particles_pos,
        ParticleCellInfo* __restrict__ particles_cell_info,
        CellInfo* __restrict__ cellInfo,
        int* __restrict__ cellParticleCount,
        const int numParticles)
    {
        const int i_part = threadIdx.x + blockIdx.x * blockDim.x;
        if (i_part >= numParticles) return;

        const real3 p = particles_pos[i_part];

        using rrgsim::common::sqr;
        using rrgsim::common::clamp;

        // ----- Декарт → сферические -----
        const real rxy2 = sqr(p.x) + sqr(p.y);
        const real r    = sqrt(rxy2 + sqr(p.z));

        // θ ∈ [0, π] — полярный угол (colatitude). atan2 устойчив и при r = 0.
        const real theta = atan2(sqrt(rxy2), p.z);

        // φ ∈ [0, 2π) - долгота
        real phi = atan2(p.y, p.x);
        if (phi < 0) {
            phi += 2 * PI;
        }

        // ----- Обратные отображения -----

        // r = R * x^p → x = (r/R)^(1/p)
        real xr = 0;
        if (r > 0) {
            xr = pow(r / sph_grid_info_.radius_max, 1 / sph_grid_info_.proj_r_degree);
        }
        int i = clamp(
            (int)(xr * sph_grid_info_.nr),
            0,
            sph_grid_info_.nr - 1
        );

        // θ = π/2 + arcsin(2x − 1)  →  x = (1 − cos θ)/2
        const real xt = 0.5f * (1.f - cosf(theta));
        int j = clamp(
            (int)(xt * sph_grid_info_.nt),
            0,
            sph_grid_info_.nt - 1
        );

        // φ = 2π k / Nφ → k = ⌊φ Nφ / 2π⌋,  с защитой от k == Nφ
        int k = clamp(
            (int)(phi * sph_grid_info_.np / (2 * PI)),
            0,
            sph_grid_info_.np - 1
        );

        const size_t i_cell = cell_index(i, j, k);

        // Сохраняем позицию внутри ячейки
        particles_cell_info[i_part].cell_id = (int)i_cell;

        // Атомарно увеличиваем счётчик частиц в ячейке
        const int pos_in_cell = atomicAdd(&cellParticleCount[i_cell], 1);

        // Сохраняем позицию внутри ячейки
        particles_cell_info[i_part].id_in_cell = pos_in_cell;

        // Атомарно обновляем количество частиц в ячейке
        atomicAdd(&cellInfo[i_cell].count, 1);
    }

} // namespace

void SphP2MeshSolver::assign_particles_to_cells(
    const rrgsim::common::OverInfo& over,
    const sParticlesContext_ particles_,
    sGridContext_ grid_
)
{
    spdlog::trace("assign_particles_to_cells_spherical_({}, {})", over.particles, over.blocks);
    RR::CUDA::CuCall(assign_particles_to_cells_spherical_, over.particles, over.blocks) (
        particles_->pos_,
        grid_->particles_cell_info_,
        grid_->cell_info_,
        grid_->cell_particles_count_
    );
}

void SphP2MeshSolver::project_particles_acceleration_on_grid(
    const sParticlesContext_ particles_context_,
    sGridContext_ grid_context_
)
{
    throw std::runtime_error{ "Not implemented" };
}

void SphP2MeshSolver::project_particles_grav_on_grid(
    const sParticlesContext_ particles_context_,
    sGridContext_ grid_context_
)
{
    throw std::runtime_error{ "Not implemented" };
}

void SphP2MeshSolver::calc_acceleration_field(
    const sGridContext_ grid_
)
{
    throw std::runtime_error{ "Not implemented" };
}

void SphP2MeshSolver::calc_grav_by_center_of_mass(
    rrgsim::common::sGridContext_ grid_,
    real3 center_of_mass,
    real total_mass
)
{
    throw std::runtime_error{ "Not implemented" };
}

const rrgsim::common::SphGridInfo&
SphP2MeshSolver::grid_info(
    const rrgsim::common::sGridContext_& grid_
)
{
    if (auto sph_grid_info = std::get_if<rrgsim::common::SphGridInfo>(&grid_->info); sph_grid_info) {
        return *sph_grid_info;
    }
    else {
        throw std::runtime_error{ "SphGridInfo::grid_info met unexpected grid info variant" };
    }
}

} // namespace rrgsim::mesh::sph