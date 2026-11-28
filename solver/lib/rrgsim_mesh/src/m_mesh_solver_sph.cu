#include "m_mesh_solver_sph.cuh"
#include <co_device_utils.cuh>
#include <spdlog/spdlog.h>

namespace rrgsim::mesh::sph {

using rrgsim::common::cell_center_r_;
using rrgsim::common::cell_center_theta_;
using rrgsim::common::coord_rect_to_spherical;
using rrgsim::common::coord_spherical_to_uniform_;
using rrgsim::common::particles_info_;
using rrgsim::common::sph_grid_info_;
using rrgsim::common::BLOCK_SIZE;
using rrgsim::common::detail::ParticleCellInfo;
using rrgsim::common::detail::CellInfo;
using rrgsim::common::OverInfo;
using RR::CUDA::CuDarray;

namespace {

    __device__ real sph_cell_volume(int ri) {
        using rrgsim::common::cube;

        const real x_in  = (real)ri       / (real)sph_grid_info_.nr;
        const real x_out = (real)(ri + 1) / (real)sph_grid_info_.nr;
        // r³ = R³ · x^(3p)
        const real r3 = cube(sph_grid_info_.radius_max) *
                        (pow(x_out, 3. * sph_grid_info_.proj_r_degree) - pow(x_in, 3. * sph_grid_info_.proj_r_degree));
        const real dOmega = 4 * PI / (real)(sph_grid_info_.nt * sph_grid_info_.np);
        return r3 * dOmega * (1. / 3.);
    }

    __global__ void assign_particles_to_cells_spherical_(
        const real3* __restrict__ particles_pos,
        ParticleCellInfo* __restrict__ particles_cell_info,
        CellInfo* __restrict__ cellInfo,
        int* __restrict__ cellParticleCount
    )
    {
        const int i_part = threadIdx.x + blockIdx.x * blockDim.x;
        if (i_part >= particles_info_.ntotal) return;

        const real3 p = particles_pos[i_part];

        using rrgsim::common::sqr;
        using rrgsim::common::clamp;

        // ----- Декартовы -> Сферические -----
        real3 sph = coord_rect_to_spherical(p);

        // ----- Сферические -> Вычислительные [0..1] -----
        auto [xr, xt, xp] = coord_spherical_to_uniform_(sph);

        int i = clamp(
            (int)(xr * sph_grid_info_.nr),
            0,
            sph_grid_info_.nr - 1
        );

        int j = clamp(
            (int)(xt * sph_grid_info_.nt),
            0,
            sph_grid_info_.nt - 1
        );

        int k = (int)(xp * sph_grid_info_.np);
        if (k >= sph_grid_info_.np) {
            k -= sph_grid_info_.np;
        }
        else if (k < 0) {
            k += sph_grid_info_.np;
        }

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

    __device__ __forceinline__ DataCIC
    deposit_CIC_spherical_(const real3& pos) {
        using rrgsim::common::clamp;

        // ----- Декартовы -> Сферические -----
        real3 sph = coord_rect_to_spherical(pos);

        // ----- Сферические -> Вычислительные [0..1] -----
        auto [xr, xt, xp] = coord_spherical_to_uniform(sph, sph_grid_info_);

        // ----- Индексы центров ячеек -----
        //   r, θ, φ — со сдвигом -0.5    (центры в (i+1/2)/N)
        const real eps = 1e-6;

        real u = xr * (real)sph_grid_info_.nr - 0.5;
        real v = xt * (real)sph_grid_info_.nt - 0.5;
        real w = xp * (real)sph_grid_info_.np - 0.5;

        u = clamp(u, 0., (real)(sph_grid_info_.nr - 1) - eps);
        v = clamp(v, 0., (real)(sph_grid_info_.nt - 1) - eps);

        DataCIC data;

        const int i0 = (int)floor(u);
        const int j0 = (int)floor(v);
        const int k0 = ((int)floor(w) % sph_grid_info_.np + sph_grid_info_.np) % sph_grid_info_.np;
        const int i1 = i0 + 1;
        const int j1 = j0 + 1;
        const int k1 = (k0 + 1) % sph_grid_info_.np; // периодическая замотка

        const real fx = u - (real)i0;
        const real fy = v - (real)j0;
        const real fz = w - floor(w); // ∈ [0,1)

        // Веса CIC
        const real wx0 = 1 - fx, wx1 = fx;
        const real wy0 = 1 - fy, wy1 = fy;
        const real wz0 = 1 - fz, wz1 = fz;

        data.weights[0b000] = wx0*wy0*wz0;
        data.weights[0b100] = wx1*wy0*wz0;
        data.weights[0b010] = wx0*wy1*wz0;
        data.weights[0b110] = wx1*wy1*wz0;
        data.weights[0b001] = wx0*wy0*wz1;
        data.weights[0b101] = wx1*wy0*wz1;
        data.weights[0b011] = wx0*wy1*wz1;
        data.weights[0b111] = wx1*wy1*wz1;

        data.indices[0b000] = cell_index(i0, j0, k0);
        data.indices[0b100] = cell_index(i1, j0, k0);
        data.indices[0b010] = cell_index(i0, j1, k0);
        data.indices[0b110] = cell_index(i1, j1, k0);
        data.indices[0b001] = cell_index(i0, j0, k1);
        data.indices[0b101] = cell_index(i1, j0, k1);
        data.indices[0b011] = cell_index(i0, j1, k1);
        data.indices[0b111] = cell_index(i1, j1, k1);

        return data;
    }

    __global__ void deposit_real_CIC_spherical_(
        const real3* __restrict__ particle_pos,
        const real*  __restrict__ particle_real,
        real*        __restrict__ cell_real
    )
    {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= particles_info_.ntotal) return;

        const real3 pos = particle_pos[p];
        const real  v   = particle_real[p];

        auto data_CIC = deposit_CIC_spherical_(pos);

        #pragma unroll
        for (int i = 0; i < N_CIC; ++i) {
            size_t index = data_CIC.indices[i];
            real weight = data_CIC.weights[i];
            atomicAdd(&cell_real[index], v * weight);
        }
    }

    __global__ void deposit_real3_CIC_spherical_(
        const real3* __restrict__ particle_pos,
        const real3* __restrict__ particle_real3,
        real3*       __restrict__ cell_real3
    )
    {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= particles_info_.ntotal) return;

        const real3 pos = particle_pos[p];
        const real3 v = particle_real3[p];

        auto data_CIC = deposit_CIC_spherical_(pos);

        #pragma unroll
        for (int i = 0; i < N_CIC; ++i) {
            size_t index = data_CIC.indices[i];
            real weight = data_CIC.weights[i];
            atomicAdd(&cell_real3[index].x, v.x * weight);
            atomicAdd(&cell_real3[index].y, v.y * weight);
            atomicAdd(&cell_real3[index].z, v.z * weight);
        }
    }

    __global__ void project_real_CIC_spherical_(
        const real3* __restrict__ particle_pos,
        const real*  __restrict__ cell_real,
        real*        __restrict__ particle_real
    )
    {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= particles_info_.ntotal) return;

        auto data_CIC = deposit_CIC_spherical_(particle_pos[p]);

        real value = 0;

        #pragma unroll
        for (int i = 0; i < N_CIC; ++i) {
            size_t index = data_CIC.indices[i];
            real weight = data_CIC.weights[i];
            value += weight * cell_real[index];
        }

        particle_real[p] = value;
    }

    __global__ void project_real3_CIC_spherical_(
        const real3* __restrict__ particle_pos,
        const real3* __restrict__ cell_real3,
        real3*       __restrict__ particle_real3
    )
    {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= particles_info_.ntotal) return;

        auto data_CIC = deposit_CIC_spherical_(particle_pos[p]);

        real3 value = make_real3(0, 0, 0);

        #pragma unroll
        for (int i = 0; i < N_CIC; ++i) {
            size_t index = data_CIC.indices[i];
            real weight = data_CIC.weights[i];
            value.x += weight * cell_real3[index].x;
            value.y += weight * cell_real3[index].y;
            value.z += weight * cell_real3[index].z;
        }

        particle_real3[p] = value;
    }

    __global__ void mass_to_density_spherical_(real* cell_rho) {
        const size_t n = (size_t)sph_grid_info_.nr * sph_grid_info_.nt * sph_grid_info_.np;
        for (size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
            c < n;
            c += (size_t)gridDim.x * blockDim.x
        )
        {
            const int ri = (int)(c % (size_t)sph_grid_info_.nr);
            cell_rho[c] /= sph_cell_volume(ri);
        }
    }

    __global__ void calc_acceleration_field_sph_(
        real3* __restrict__ cell_acc,
        const real* __restrict__ cell_phi
    )
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        const int j = threadIdx.y + blockIdx.y * blockDim.y;
        const int k = threadIdx.z + blockIdx.z * blockDim.z;

        const int NR = sph_grid_info_.nr;
        const int NT = sph_grid_info_.nt;
        const int NP = sph_grid_info_.np;

        if (i >= NR || j >= NT || k >= NP) return;

        const size_t idx = cell_index(i, j, k);

        const real r_i    = cell_center_r_(i);
        const real th_i   = cell_center_theta_(j);
        const real sin_th = sin(th_i);

        // ------------------------------------------------------------
        // dgrav/dr
        // ------------------------------------------------------------
        real dgrav_dr;

        if (i == 0) {
            const real dr = cell_center_r_(i + 1) - r_i;
            dgrav_dr = (cell_phi[cell_index(i + 1, j, k)] - cell_phi[idx]) / dr;
        }
        else if (i == NR - 1) {
            const real dr = r_i - cell_center_r_(i - 1);
            dgrav_dr = (cell_phi[idx] - cell_phi[cell_index(i - 1, j, k)]) / dr;
        }
        else {
            const real hL = r_i - cell_center_r_(i - 1);
            const real hR = cell_center_r_(i + 1) - r_i;

            const real fm = cell_phi[cell_index(i - 1, j, k)];
            const real f0 = cell_phi[idx];
            const real fp = cell_phi[cell_index(i + 1, j, k)];

            // взвешенная центральная разность для неравномерной сетки
            dgrav_dr = (hR * (f0 - fm) / hL + hL * (fp - f0) / hR) / (hL + hR);
        }

        // ------------------------------------------------------------
        // dgrav/dtheta
        // ------------------------------------------------------------
        real dgrav_dtheta;

        if (j == 0) {
            const real dth = cell_center_theta_(j + 1) - th_i;
            dgrav_dtheta = (cell_phi[cell_index(i, j + 1, k)] - cell_phi[idx]) / dth;
        }
        else if (j == NT - 1) {
            const real dth = th_i - cell_center_theta_(j - 1);
            dgrav_dtheta = (cell_phi[idx] - cell_phi[cell_index(i, j - 1, k)]) / dth;
        }
        else {
            const real hL = th_i - cell_center_theta_(j - 1);
            const real hR = cell_center_theta_(j + 1) - th_i;

            const real fm = cell_phi[cell_index(i, j - 1, k)];
            const real f0 = cell_phi[idx];
            const real fp = cell_phi[cell_index(i, j + 1, k)];

            dgrav_dtheta = (hR * (f0 - fm) / hL + hL * (fp - f0) / hR) / (hL + hR);
        }

        // ------------------------------------------------------------
        // dgrav/dphi (долгота)
        // Периодическая центральная разность.
        // ------------------------------------------------------------
        real dgrav_dphi = 0;

        if (NP > 1) {
            const int kp = (k + 1) % NP;
            const int km = (k - 1 + NP) % NP;

            const real fp = cell_phi[cell_index(i, j, kp)];
            const real fm = cell_phi[cell_index(i, j, km)];

            real diff = fp - fm;

            // "разворачиваем" разность через 2*pi при переходе через границу
            if (km > kp) {
                diff += 2 * PI;
            }

            const real dphi = 2 * PI / NP;
            dgrav_dphi = diff / (2 * dphi);
        }

        // ------------------------------------------------------------
        // a = -grad(phi) в сферическом базисе
        // ------------------------------------------------------------
        const real inv_r      = 1 / r_i;
        const real inv_r_sin  = 1 / (r_i * sin_th);

        cell_acc[idx] = make_real3(
            -dgrav_dr,
            -inv_r * dgrav_dtheta,
            -inv_r_sin * dgrav_dphi
        );
    }
} // namespace

void SphMeshSolver::init(
    const sGridContext_ grid_
)
{
    RR::CUDA::CuCopyToSymbol(
        grid_info(grid_),
        rrgsim::common::sph_grid_info_,
        RR::CUDA::ToDevice
    );
}

void SphMeshSolver::assign_particles_to_cells(
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

void SphMeshSolver::deposit_mass(
    const rrgsim::common::OverInfo& over,
    const sParticlesContext_ particles_,
    sGridContext_ grid_
)
{
    spdlog::trace("deposit_mass({}, {})", over.particles, over.blocks);
	RR::CUDA::CuCall(deposit_real_CIC_spherical_, over.particles, over.blocks) (
		particles_->pos_,
		particles_->mass_,
		grid_->rho_
	);

    spdlog::trace("mass_to_density({}, {})", BLOCK_SIZE, BLOCK_SIZE);
    RR::CUDA::CuCall(mass_to_density_spherical_, BLOCK_SIZE, BLOCK_SIZE) (
        grid_->rho_
    );
}

void SphMeshSolver::project_particles_acceleration_on_grid(
    const sParticlesContext_ particles_,
    sGridContext_ grid_
)
{
    auto over = OverInfo::calc(
        rrgsim::common::get_grid_size(grid_->info),
        particles_->info.ntotal
    );

	grid_->acc_.set_zero();

	RR::CUDA::CuCall(deposit_real3_CIC_spherical_, over.particles, over.blocks) (
		particles_->pos_,
		particles_->acc_,
		grid_->acc_
	);
}

void SphMeshSolver::project_particles_grav_on_grid(
    const sParticlesContext_ particles_,
    sGridContext_ grid_
)
{
    auto over = OverInfo::calc(
        rrgsim::common::get_grid_size(grid_->info),
        particles_->info.ntotal
    );

	grid_->grav_curr_.set_zero();

	RR::CUDA::CuCall(deposit_real_CIC_spherical_, over.particles, over.blocks) (
		particles_->pos_,
		particles_->grav_,
		grid_->grav_curr_
	);
}

void SphMeshSolver::calc_acceleration_field(
    const sGridContext_ grid_
)
{
    spdlog::info("calc_acceleration_field");
    RR::CUDA::CuDeviceSync();

    using rrgsim::common::BLOCK_SIZE_IN_3D;

    dim3 threads_per_block{ BLOCK_SIZE_IN_3D, BLOCK_SIZE_IN_3D, BLOCK_SIZE_IN_3D };
    dim3 blocks_per_grid{
        static_cast<unsigned>(OverInfo::split_into_blocks(grid_info(grid_).nr, BLOCK_SIZE_IN_3D)),
        static_cast<unsigned>(OverInfo::split_into_blocks(grid_info(grid_).nt, BLOCK_SIZE_IN_3D)),
        static_cast<unsigned>(OverInfo::split_into_blocks(grid_info(grid_).np, BLOCK_SIZE_IN_3D))
    };

    RR::CUDA::CuCall(calc_acceleration_field_sph_, blocks_per_grid, threads_per_block) (
        grid_->acc_,
        grid_->grav_curr_
    );
}

void SphMeshSolver::calc_grav_by_center_of_mass(
    rrgsim::common::sGridContext_ grid_,
    real3 center_of_mass,
    real total_mass
)
{
    throw std::runtime_error{ "Not implemented" };
}

const rrgsim::common::SphGridInfo&
SphMeshSolver::grid_info(
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

void SphMeshSolver::project_grid_acceleration_on_particles(
    const sGridContext_ grid_,
    sParticlesContext_ particles_
)
{
    auto over = OverInfo::calc(0, particles_->info.ntotal);

	RR::CUDA::CuCall(project_real3_CIC_spherical_, over.particles, over.blocks) (
		particles_->pos_,
		grid_->acc_,
		particles_->acc_
	);
}

void SphMeshSolver::project_grid_grav_on_particles(
    const sGridContext_ grid_,
    sParticlesContext_ particles_
)
{
    auto over = OverInfo::calc(0, particles_->info.ntotal);

	RR::CUDA::CuCall(project_real_CIC_spherical_, over.particles, over.blocks) (
		particles_->pos_,
		grid_->grav_curr_,
		particles_->grav_
	);
}

} // namespace rrgsim::mesh::sph