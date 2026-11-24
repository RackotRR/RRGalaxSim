#include <cuda_runtime.h>
#include <co_device_utils.cuh>
#include <co_device_structs.cuh>

#include "rm_mesh2p.cuh"

#include <spdlog/spdlog.h>

namespace rrgsim::mesh::rect {

using rrgsim::common::detail::ParticleCellInfo;
using rrgsim::common::detail::CellInfo;
using rrgsim::common::rect_grid_info_;

namespace {

    /// @note Запуск по частицам
    __global__ void projectCellMassOnParticles(
        const real* cell_rho,
        const ParticleCellInfo* particles_cell_info,
        const CellInfo* cell_info,
        real* particle_phi
    )
    {
        int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
        int i_cell = particles_cell_info[i_particle].cell_id;
        real V = rrgsim::common::cube(rect_grid_info_.dx);
        particle_phi[i_particle] = cell_rho[i_cell] * V / cell_info[i_cell].count;
    }

    /// @note Запуск по частицам
    __global__ void projectCellPhiOnParticles(
        const real* cell_phi,
        const ParticleCellInfo* particles_cell_info,
        real* particle_phi
    )
    {
        int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
        int i_cell = particles_cell_info[i_particle].cell_id;
        particle_phi[i_particle] = cell_phi[i_cell];
    }


    /**
     * @brief Ядро для интерполяции ускорения на частицы (трилинейная интерполяция)
     * @param particle_acc Выходные ускорения частиц
     * @param particle_pos Координаты частиц
     * @param field_acc Поле ускорения на сетке
     * @param n_particles Количество частиц
     */
    __global__ void interpolate_acceleration_to_particles_kernel(
        real3* particle_acc,
        const real3* particle_pos,
        const real3* field_acc,
        int n_particles
    ) {
        int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= n_particles) return;

        const int nx = rect_grid_info_.nx;
        const real dx = rect_grid_info_.dx;
        const real domain_min = rect_grid_info_.domain_min;

        real3 pos = particle_pos[p];

        // Индексы ячейки и веса для трилинейной интерполяции
        real fx = (pos.x - domain_min) / dx - 0.5f;
        real fy = (pos.y - domain_min) / dx - 0.5f;
        real fz = (pos.z - domain_min) / dx - 0.5f;

        int i0 = (int)floorf(fx);
        int j0 = (int)floorf(fy);
        int k0 = (int)floorf(fz);

        int i1 = i0 + 1;
        int j1 = j0 + 1;
        int k1 = k0 + 1;

        real wx = fx - i0;
        real wy = fy - j0;
        real wz = fz - k0;

        // Проверка границ
        i0 = max(0, min(i0, nx - 1));
        i1 = max(0, min(i1, nx - 1));
        j0 = max(0, min(j0, nx - 1));
        j1 = max(0, min(j1, nx - 1));
        k0 = max(0, min(k0, nx - 1));
        k1 = max(0, min(k1, nx - 1));

        // Трилинейная интерполяция
        real3 acc = make_real3(0.0f, 0.0f, 0.0f);

        for (int k = 0; k <= 1; k++) {
            int kk = (k == 0) ? k0 : k1;
            real wk = (k == 0) ? (1.0f - wz) : wz;

            for (int j = 0; j <= 1; j++) {
                int jj = (j == 0) ? j0 : j1;
                real wj = (j == 0) ? (1.0f - wy) : wy;

                for (int i = 0; i <= 1; i++) {
                    int ii = (i == 0) ? i0 : i1;
                    real wi = (i == 0) ? (1.0f - wx) : wx;

                    real weight = wi * wj * wk;
                    int idx = kk * nx * nx + jj * nx + ii;

                    acc.x += weight * field_acc[idx].x;
                    acc.y += weight * field_acc[idx].y;
                    acc.z += weight * field_acc[idx].z;
                }
            }
        }

        particle_acc[p] = acc;
    }

} // namespace

void RectMesh2PSolver::project_grid_acceleration_on_particles(
    const sGridContext_ grid_context_,
    sParticlesContext_ particles_context_
)
{
    spdlog::debug("project_grid_acceleration_on_particles");
    RR::CUDA::CuDeviceSync();

    auto over = rrgsim::common::OverInfo::calc(0, particles_context_->info.ntotal);

    RR::CUDA::CuCall(interpolate_acceleration_to_particles_kernel, over.particles, over.blocks) (
        particles_context_->acc_,
        particles_context_->pos_,
        grid_context_->acc_,
        over.num_particles
    );

}

void RectMesh2PSolver::project_grid_grav_on_particles(
    const sGridContext_ grid_context_,
    sParticlesContext_ particles_context_
)
{
    spdlog::debug("project_grid_grav_on_particles");
    RR::CUDA::CuDeviceSync();

    auto over = rrgsim::common::OverInfo::calc(0, particles_context_->info.ntotal);
    RR::CUDA::CuCall(projectCellPhiOnParticles, over.particles, over.blocks) (
        grid_context_->grav_curr_,
        grid_context_->particles_cell_info_,
        particles_context_->grav_
    );
}

GridProjectionOnParticles
RectMesh2PSolver::project_grid_onto_particles(
    sGridContext_ grid_context_,
    sParticlesContext_ particles_context_
)
{
    spdlog::debug("project_grid_onto_particles");
    RR::CUDA::CuDeviceSync();

    auto over = rrgsim::common::OverInfo::calc(0, particles_context_->info.ntotal);

    RR::CUDA::CuCall(projectCellPhiOnParticles, over.particles, over.blocks) (
        grid_context_->grav_curr_,
        grid_context_->particles_cell_info_,
        particles_context_->grav_
    );
    RR::CUDA::CuCall(projectCellMassOnParticles, over.particles, over.blocks) (
        grid_context_->rho_,
        grid_context_->particles_cell_info_,
        grid_context_->cell_info_,
        particles_context_->mass_
    );

    GridProjectionOnParticles projection;
    projection.grav = particles_context_->grav_.to_vector();
    projection.mass = particles_context_->mass_.to_vector();
    return projection;
}


} // namespace rrgsim::rect_mesh