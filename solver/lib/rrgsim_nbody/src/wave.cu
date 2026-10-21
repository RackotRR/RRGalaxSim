#include "wave.cuh"
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <co_device_utils.cuh>
#include <co_device_structs.cuh>

#include <nbody.cuh>

namespace rrgsim::wave {
	using rrgsim::common::BLOCK_SIZE;
	using rrgsim::common::particles_info_;
	using rrgsim::common::grid_info_;
	using rrgsim::common::wave_params_;
	using rrgsim::common::bc_params_;
	using rrgsim::common::distance;
	using rrgsim::common::cube;
	using rrgsim::common::dot;

__device__ __forceinline__ real coord(int i) {
    return grid_info_.domain_min + i * grid_info_.dx;
}

__device__ __forceinline__ int wrap(int i, int n) {
    return (i < 0) ? i + n : ((i >= n) ? i - n : i);
}

/// Квадратичная диссипация в слое по одной оси
__device__ __forceinline__ real layer_diss(real x) {
    const real x_in = real(0.5) * grid_info_.sim_l; // внутренняя граница слоя
    const real d = fabs(x) - x_in;
    return (d > 0)
        ? wave_params_.diss_extra * d * d
        : real(0);
}

/// Асимптотический (монопольный) потенциал
__device__ __forceinline__ real monopole_phi(real3 p) {
    const real G = 1.;
    const real dist = distance(bc_params_.center_of_mass, p);
    const real r  = fmax(dist, bc_params_.r_min);
    return -G * bc_params_.mass / r;
}

/// Отклонение от асимптотики psi = phi - phi_mono в узле (i,j,k)
__device__ __forceinline__ real psi_at(const real* __restrict__ phi, int i, int j, int k) {
    const int NX = grid_info_.nx;
    return phi[AT(i, j, k)] - monopole_phi(
        make_real3(coord(i), coord(j), coord(k))
    );
}

/// Upwind-производная по одной оси вдоль направления распространения n_comp.
/// Сосед берётся «против потока», т.е. со стороны, откуда приходит волна.
__device__ __forceinline__ real upwind_axis(
    const real* __restrict__ phi,
    real psi0,
    real n_comp,
    int ix,
    int iy,
    int iz,
    int axis
)
{
    const int  NX = grid_info_.nx;
    const real DX = grid_info_.dx;

    int i0 = (axis == 0) ? ix : (axis == 1) ? iy : iz;
    int in = (n_comp > 0) ? i0 - 1 : i0 + 1;
    if (in < 0 || in >= NX) in = 2 * i0 - in;   // нет соседа — берём другую сторону

    int jx = ix, jy = iy, jz = iz;
    if (axis == 0) jx = in;
    else if (axis == 1) jy = in;
    else jz = in;

    const real d = (psi0 - psi_at(phi, jx, jy, jz)) / ((i0 - in) * DX);
    return n_comp * d;
}

/// Новое значение phi на граничном узле:
/// d_t psi + c (r^ . grad psi + psi / r) = 0, явная схема против потока
__device__ real radiative_boundary(const real* __restrict__ phi, int ix, int iy, int iz)
{
    const real C  = wave_params_.wave_speed;
    const real DT = wave_params_.dt;

    const real x = coord(ix), y = coord(iy), z = coord(iz);
    const real rx = x - bc_params_.center_of_mass.x;
    const real ry = y - bc_params_.center_of_mass.y;
    const real rz = z - bc_params_.center_of_mass.z;
    const real r  = fmax(sqrt(rx * rx + ry * ry + rz * rz), bc_params_.r_min);
    const real nx = rx / r, ny = ry / r, nz = rz / r;

    const real psi0 = psi_at(phi, ix, iy, iz);

    // Направленная производная по радиусу сферической волны,
    // а не по нормали к грани куба
    const real dpsi_dr =
          upwind_axis(phi, psi0, nx, ix, iy, iz, 0)
        + upwind_axis(phi, psi0, ny, ix, iy, iz, 1)
        + upwind_axis(phi, psi0, nz, ix, iy, iz, 2);

    const real psi_new = psi0 - C * DT * (dpsi_dr + psi0 / r);
    return psi_new + monopole_phi(make_real3(x, y, z));
}

__global__ void wave_diss_iteration(
    real* __restrict__ phi_new,
    const real* __restrict__ phi,
    const real* __restrict__ phi_old,
    const real* __restrict__ mass
)
{
    const int  NX         = grid_info_.nx;
    const real DX         = grid_info_.dx;
    const real C          = wave_params_.wave_speed;
    const real DT         = wave_params_.dt;
    const real DISS_BASE  = wave_params_.diss_base;
    const BoundaryType BT = bc_params_.type;

    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;
    if (ix >= NX || iy >= NX || iz >= NX) return;

    const bool periodic = (BT == BoundaryType::Periodic);
    const bool on_edge  = ix == 0 || ix == NX - 1
                       || iy == 0 || iy == NX - 1
                       || iz == 0 || iz == NX - 1;

    // ---------- Граничные узлы (непериодические режимы) ----------
    if (!periodic && on_edge) {
        real v;
        switch (BT) {
        case BoundaryType::MonopoleDirichlet:
            v = monopole_phi(make_real3(coord(ix), coord(iy), coord(iz)));
            break;
        case BoundaryType::MonopoleRadiative:
            v = radiative_boundary(phi, ix, iy, iz);
            break;
        case BoundaryType::ZeroDirichlet:
        default:
            v = 0.;
            break;
        }
        phi_new[AT(ix, iy, iz)] = v;
        return;
    }

    // ---------- Источник ----------
    const real G = 1.;
    real rho = mass[AT(ix, iy, iz)] / (DX * DX * DX);
    if (periodic) rho -= bc_params_.rho_mean;   // иначе нет установления
    const real f = 4 * PI * G * rho;

    // ---------- Диссипация ----------
    real diss = DISS_BASE;
    if (!periodic || bc_params_.periodic_keep_layer) {
        diss += layer_diss(coord(ix));
        diss += layer_diss(coord(iy));
        diss += layer_diss(coord(iz));
    }

    // ---------- Коэффициенты схемы (как раньше, без макросов) ----------
    const int  DIM      = 3;
    const real Q        = diss * C * DT * real(0.5);
    const real W        = 1 + Q;
    const real K        = C * C * DT * DT / (DX * DX);
    const real K_MAIN   = K / W;
    const real K_F      = -C * C * DT * DT / W;
    const real K_ACTUAL = 2 * (1 - DIM * K) / W;
    const real K_OLD    = (Q - 1) / W;

    // ---------- Соседи ----------
    int xm = ix - 1, xp = ix + 1;
    int ym = iy - 1, yp = iy + 1;
    int zm = iz - 1, zp = iz + 1;
    if (periodic) {
        xm = wrap(xm, NX); xp = wrap(xp, NX);
        ym = wrap(ym, NX); yp = wrap(yp, NX);
        zm = wrap(zm, NX); zp = wrap(zp, NX);
    }

    const real lap_sum =
          phi[AT(xm, iy, iz)] + phi[AT(xp, iy, iz)]
        + phi[AT(ix, ym, iz)] + phi[AT(ix, yp, iz)]
        + phi[AT(ix, iy, zm)] + phi[AT(ix, iy, zp)];

    phi_new[AT(ix, iy, iz)] =
          lap_sum * K_MAIN
        + phi[AT(ix, iy, iz)] * K_ACTUAL
        + phi_old[AT(ix, iy, iz)] * K_OLD
        + f * K_F;
}

/// Асимптотические ГУ: масса и центр масс подаются на вход
void set_boundary_monopole(
    real total_mass,
    const real3& center_of_mass,
    bool radiative,
    real dx
)
{
    BoundaryParams p{};
    p.type  = radiative
        ? BoundaryType::MonopoleRadiative
        : BoundaryType::MonopoleDirichlet;
    p.mass  = total_mass;
    p.center_of_mass = center_of_mass;
    p.r_min = real(0.5) * dx;
    RR::CUDA::CuCopyToSymbol(p, bc_params_, RR::CUDA::ToDevice);
}

/// Периодические ГУ: средняя плотность считается по сетке
void set_boundary_periodic(
    real total_mass,
    const rrgsim::common::GridInfo& g,
    bool keep_layer
)
{
    const real L = g.nx * g.dx;              // период

    BoundaryParams p{};
    p.type                = BoundaryType::Periodic;
    p.mass                = total_mass;
    p.rho_mean            = total_mass / (L * L * L);
    p.periodic_keep_layer = keep_layer ? 1 : 0;
    RR::CUDA::CuCopyToSymbol(p, bc_params_, RR::CUDA::ToDevice);
}

void solve_wave_equation(
    rrgsim::common::sGridContext_ grid_,
    rrgsim::common::WaveParams wave_params
)
{
    spdlog::debug("solve_wave_equation");
    RR::CUDA::CuDeviceSync();

	for (int iter = 0; iter < wave_params.setup_iterations; ++iter) {
		if (iter && iter % 500 == 0) {
            spdlog::debug("wave_diss_iteration step: {} / {}", iter, wave_params.setup_iterations);
		}

        using rrgsim::common::BLOCK_SIZE_IN_3D;
        using rrgsim::common::OverInfo;

        unsigned threads_count = BLOCK_SIZE_IN_3D;
        unsigned blocks_count = static_cast<unsigned>(OverInfo::split_into_blocks(grid_->info.nx, BLOCK_SIZE_IN_3D));
        dim3 threads_per_block{ threads_count, threads_count, threads_count };
        dim3 blocks_per_grid{ blocks_count, blocks_count, blocks_count };
		RR::CUDA::CuCall(wave_diss_iteration, blocks_per_grid, threads_per_block) (
            grid_->grav_next_,
            grid_->grav_curr_,
            grid_->grav_prev_,
            grid_->mass_
		);
		swap(grid_->grav_prev_, grid_->grav_curr_);
		swap(grid_->grav_curr_, grid_->grav_next_);
	}

    spdlog::debug("solve_wave_equation succeed");
}



__global__ void fill_in_initial_grav_(
    real* grav,
    real3 mass_center,
    real mass_value
)
{
	const int NX = grid_info_.nx;
	const real DX = grid_info_.dx;

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

    real3 pos = make_real3(
        grid_info_.domain_min + ix * DX,
        grid_info_.domain_min + iy * DX,
        grid_info_.domain_min + iz * DX
    );

    const real G = 1.;
    const real softening = 0.01;

    grav[AT(ix, iy, iz)] = -G * mass_value / (rrgsim::common::distance(pos, mass_center) + softening);
}

void fill_in_initial_grav(
    rrgsim::common::sGridContext_ grid_,
    real3 mass_center,
    real mass_value
)
{
    spdlog::debug("fill_in_initial_grav");

    using rrgsim::common::BLOCK_SIZE_IN_3D;
    using rrgsim::common::OverInfo;

    unsigned threads_count = BLOCK_SIZE_IN_3D;
    unsigned blocks_count = static_cast<unsigned>(OverInfo::split_into_blocks(grid_->info.nx, BLOCK_SIZE_IN_3D));
    dim3 threads_per_block{ threads_count, threads_count, threads_count };
    dim3 blocks_per_grid{ blocks_count, blocks_count, blocks_count };

    RR::CUDA::CuCall(fill_in_initial_grav_, blocks_per_grid, threads_per_block) (
        grid_->grav_curr_,
        mass_center,
        mass_value
    );

    RR::CUDA::CuCopyFromTo(
        grid_->grav_curr_,
        grid_->grav_prev_,
        RR::CUDA::DeviceToDevice
    );
}

} // namespace rrgsim::wave