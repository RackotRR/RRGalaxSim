#include "wave_rect.cuh"
#include <co_device_utils.cuh>
#include <co_device_structs.cuh>
#include <spdlog/spdlog.h>

namespace rrgsim::wave::rect {
	using rrgsim::common::rect_grid_info_;
	using rrgsim::common::wave_params_;
	using rrgsim::common::bc_params_;
	using rrgsim::common::distance;
    using rrgsim::common::BoundaryParams;
    using rrgsim::common::BoundaryType;

__device__ __forceinline__ real coord(int i) {
    return rect_grid_info_.domain_min + i * rect_grid_info_.dx;
}

__device__ __forceinline__ int wrap(int i, int n) {
    return (i < 0) ? i + n : ((i >= n) ? i - n : i);
}

/// Квадратичная диссипация в слое по одной оси
__device__ __forceinline__ real layer_diss(real x) {
    const real x_in = real(0.5) * rect_grid_info_.sim_l; // внутренняя граница слоя
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
    const int NX = rect_grid_info_.nx;
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
    const int  NX = rect_grid_info_.nx;
    const real DX = rect_grid_info_.dx;

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

__global__ void wave_diss_iteration_rect_(
    real* __restrict__ phi_new,
    const real* __restrict__ phi,
    const real* __restrict__ phi_old,
    const real* __restrict__ mass
)
{
    const int  NX         = rect_grid_info_.nx;
    const real DX         = rect_grid_info_.dx;
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

__global__ void fill_in_initial_grav_(
    real* grav,
    real3 mass_center,
    real mass_value
)
{
	const int NX = rect_grid_info_.nx;
	const real DX = rect_grid_info_.dx;

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;
    if (ix >= NX || iy >= NX || iz >= NX) return;

    real3 pos = make_real3(
        rect_grid_info_.domain_min + ix * DX,
        rect_grid_info_.domain_min + iy * DX,
        rect_grid_info_.domain_min + iz * DX
    );

    const real G = 1.;
    const real softening = 0.01;

    grav[AT(ix, iy, iz)] = -G * mass_value / (rrgsim::common::distance(pos, mass_center) + softening);
}


WaveRectSolver::WaveRectSolver(Params params)
    : WaveSolver{ std::move(params) }
{
    using rrgsim::common::BLOCK_SIZE_IN_3D;
    using rrgsim::common::OverInfo;

    const auto& rect_info = std::get<rrgsim::common::RectGridInfo>(params.grid_info);

    unsigned threads_count = BLOCK_SIZE_IN_3D;
    unsigned blocks_count = static_cast<unsigned>(OverInfo::split_into_blocks(rect_info.nx, BLOCK_SIZE_IN_3D));

    this->threads_per_block = dim3{ threads_count, threads_count, threads_count };
    this->blocks_per_grid = dim3{ blocks_count, blocks_count, blocks_count };
}

void WaveRectSolver::fill_in_initial_grav(
    rrgsim::common::sGridContext_ grid_,
    const NewtonianGravParams& initial_grav_params
) const {
    spdlog::debug("fill_in_initial_grav");

    RR::CUDA::CuCall(fill_in_initial_grav_, blocks_per_grid, threads_per_block) (
        grid_->grav_curr_,
        initial_grav_params.mass_center,
        initial_grav_params.mass_value
    );

    RR::CUDA::CuCopyFromTo(
        grid_->grav_curr_,
        grid_->grav_prev_,
        RR::CUDA::DeviceToDevice
    );
}

void WaveRectSolver::setup(
    rrgsim::common::sGridContext_ grid_
) {
    // выводим шаг по времени из условия CFL
    const auto& rect_grid_info = std::get<rrgsim::common::RectGridInfo>(params.grid_info);
    real dx = rect_grid_info.dx;
    real c = params.wave_params.wave_speed;
    constexpr real DIM = 3;
    params.wave_params.dt = 0.5 * dx / (c * std::sqrt(DIM));
    spdlog::trace("WaveRectSolver::setup wave_params.dt = {}", params.wave_params.dt);
    RR::CUDA::CuCopyToSymbol(params.wave_params, wave_params_, RR::CUDA::ToDevice);

    this->params.mb_initial_grav_params
        .transform(
            [this, grid_](const NewtonianGravParams& initial_grav_params) {
                this->fill_in_initial_grav(grid_, initial_grav_params);
            }
        );

}

void WaveRectSolver::step(rrgsim::common::sGridContext_ grid_) {
    RR::CUDA::CuCall(wave_diss_iteration_rect_, blocks_per_grid, threads_per_block) (
        grid_->grav_next_,
        grid_->grav_curr_,
        grid_->grav_prev_,
        grid_->mass_
    );
}


} // namespace rrgsim::wave::rect