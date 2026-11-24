#include "sm_p2mesh.cuh"
#include <co_device_utils.cuh>
#include <spdlog/spdlog.h>

namespace rrgsim::mesh::sph {

using rrgsim::common::particles_info_;
using rrgsim::common::sph_grid_info_;
using rrgsim::common::BLOCK_SIZE;
using rrgsim::common::detail::ParticleCellInfo;
using rrgsim::common::detail::CellInfo;
using rrgsim::common::OverInfo;
using RR::CUDA::CuDarray;

namespace {

    __device__ __forceinline__
    real normalize_two_pi(
        real phi
    )
    {
        static constexpr real TWO_PI = PI * 2;
        return phi - TWO_PI * std::floor(phi / TWO_PI);
    }

    // Декартовы координаты в сферические
    __device__ __forceinline__
    real3 coord_rect_to_spherical(
        const real3& p
    )
    {
        using rrgsim::common::sqr;

        const real rxy2 = sqr(p.x) + sqr(p.y);
        const real r    = sqrt(rxy2 + sqr(p.z));

        // θ ∈ [0, π] — полярный угол (colatitude). atan2 устойчив и при r = 0.
        const real theta = atan2(sqrt(rxy2), p.z);

        // φ ∈ [0, 2π) - долгота
        real phi = normalize_two_pi(atan2(p.y, p.x));

        return make_real3(
            r,
            theta,
            phi
        );
    }

    // обратное преобразование сферических координат в равномерные [0,1]
    __device__ __forceinline__
    real3 coord_spherical_to_uniform(
        const real3& sph
    )
    {
        const auto& [r, theta, phi] = sph;

        // r = R * x^p → x = (r/R)^(1/p)
        real xr = r > 0
            ? pow(r / sph_grid_info_.radius_max, 1 / sph_grid_info_.proj_r_degree)
            : real(0);

        // θ = π/2 + arcsin(2x − 1)  →  x = (1 − cos θ)/2
        const real xt = 0.5 * (1 - cos(theta));

        // φ = 2π k / Nφ → k = ⌊φ Nφ / 2π⌋,  с защитой от k == Nφ
        const real xp = phi / (2 * PI);

        return make_real3(
            xr,
            xt,
            xp
        );
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
        auto [xr, xt, xp] = coord_spherical_to_uniform(sph);

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

    __device__ DataCIC
    deposit_CIC_(const real3& pos) {
        using rrgsim::common::clamp;

        // ----- Декартовы -> Сферические -----
        real3 sph = coord_rect_to_spherical(pos);

        // ----- Сферические -> Вычислительные [0..1] -----
        auto [xr, xt, xp] = coord_spherical_to_uniform(sph);

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

        auto data_CIC = deposit_CIC_(pos);

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

        auto data_CIC = deposit_CIC_(pos);

        #pragma unroll
        for (int i = 0; i < N_CIC; ++i) {
            size_t index = data_CIC.indices[i];
            real weight = data_CIC.weights[i];
            atomicAdd(&cell_real3[index].x, v.x * weight);
            atomicAdd(&cell_real3[index].y, v.y * weight);
            atomicAdd(&cell_real3[index].z, v.z * weight);
        }
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

void SphP2MeshSolver::deposit_mass(
    const rrgsim::common::OverInfo& over,
    const sParticlesContext_ particles_,
    sGridContext_ grid_
)
{
    spdlog::trace("deposit_mass({}, {})", over.particles, over.blocks);
	RR::CUDA::CuCall(deposit_real_CIC_spherical_, over.particles, over.blocks) (
		particles_->pos_,
		particles_->mass_,
		grid_->mass_
	);
}

void SphP2MeshSolver::project_particles_acceleration_on_grid(
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

void SphP2MeshSolver::project_particles_grav_on_grid(
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