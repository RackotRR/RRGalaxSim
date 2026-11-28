// wave_sph.cu
#include "wave_sph.cuh"
#include <co_device_utils.cuh>
#include <co_device_structs.cuh>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace rrgsim::wave::sph {

using rrgsim::common::SphGridInfo;
using rrgsim::common::WaveParams;
using rrgsim::common::BoundaryParams;
using rrgsim::common::BoundaryType;
using rrgsim::common::distance;

// -----------------------------------------------------------------------------
// Вспомогательные device-функции
// -----------------------------------------------------------------------------

__device__ __forceinline__ int wrap_idx(int i, int n) {
    return (i < 0) ? i + n : ((i >= n) ? i - n : i);
}

__device__ __forceinline__ int reflect_idx(int i, int n) {
    if (i < 0)  return -i;
    if (i >= n) return 2 * n - 2 - i;
    return i;
}

__device__ __forceinline__ real phi_center_(int k) {
    const int NP = sph_grid_info_.np;
    return (real(k) + real(0.5)) * (real(2) * PI / real(NP));
}

__device__ __forceinline__ real3 cell_rect_(int i, int j, int k) {
    const real r   = cell_center_r_(i);
    const real th  = cell_center_theta_(j);
    const real ph  = phi_center_(k);

    const real st = sin(th);
    const real ct = cos(th);
    const real sp = sin(ph);
    const real cp = cos(ph);

    return make_real3(r * st * cp, r * st * sp, r * ct);
}

__device__ __forceinline__ real monopole_phi_sph_(real3 p) {
    const real G = 1.;
    const real dist = distance(bc_params_.center_of_mass, p);
    const real r = fmax(dist, bc_params_.r_min);
    return -G * bc_params_.mass / r;
}

__device__ __forceinline__ real psi_at_sph_(
    const real* __restrict__ phi,
    int i, int j, int k
) {
    return phi[sph_index_(i, j, k)] - monopole_phi_sph_(cell_rect_(i, j, k));
}

// Upwind-производная по r вдоль направления nr.
__device__ __forceinline__ real upwind_r_sph_(
    const real* __restrict__ phi,
    real psi0,
    real nr,
    int i, int j, int k
) {
    if (fabs(nr) < real(1e-15)) return real(0);

    const int NR = sph_grid_info_.nr;

    int in = (nr > 0) ? i - 1 : i + 1;
    in = reflect_idx(in, NR);

    const real dr = cell_center_r_(i) - cell_center_r_(in);
    const real d  = (psi0 - psi_at_sph_(phi, in, j, k)) / dr;
    return nr * d;
}

// Upwind-производная по theta вдоль направления nth.
__device__ __forceinline__ real upwind_theta_sph_(
    const real* __restrict__ phi,
    real psi0,
    real nth,
    int i, int j, int k
) {
    if (fabs(nth) < real(1e-15)) return real(0);

    const int NT = sph_grid_info_.nt;

    int jn = (nth > 0) ? j - 1 : j + 1;
    jn = reflect_idx(jn, NT);

    const real r    = cell_center_r_(i);
    const real dth  = cell_center_theta_(j) - cell_center_theta_(jn);
    const real d    = (psi0 - psi_at_sph_(phi, i, jn, k)) / (r * dth);
    return nth * d;
}

// Upwind-производная по phi вдоль направления nph.
__device__ __forceinline__ real upwind_phi_sph_(
    const real* __restrict__ phi,
    real psi0,
    real nph,
    int i, int j, int k
) {
    if (fabs(nph) < real(1e-15)) return real(0);

    const int NP = sph_grid_info_.np;
    const real dphi = real(2) * PI / real(NP);

    const int kn_unwrapped = (nph > 0) ? k - 1 : k + 1;
    const int kn = wrap_idx(kn_unwrapped, NP);

    const real r     = cell_center_r_(i);
    const real sinth = fmax(sin(cell_center_theta_(j)), real(1e-12));
    const real denom = r * sinth * dphi * real(k - kn_unwrapped);

    const real d = (psi0 - psi_at_sph_(phi, i, j, kn)) / denom;
    return nph * d;
}

// Радиационное граничное условие для отклонения psi = phi - phi_mono:
// d_t psi + c (n . grad psi + psi / r_cm) = 0
__device__ real radiative_boundary_sph_(
    const real* __restrict__ phi,
    int i, int j, int k
) {
    const real C  = wave_params_.wave_speed;
    const real DT = wave_params_.dt;

    const real3 pos = cell_rect_(i, j, k);
    const real3 cm  = bc_params_.center_of_mass;

    const real rx = pos.x - cm.x;
    const real ry = pos.y - cm.y;
    const real rz = pos.z - cm.z;
    const real r_cm = fmax(sqrt(rx * rx + ry * ry + rz * rz), bc_params_.r_min);

    const real nx = rx / r_cm;
    const real ny = ry / r_cm;
    const real nz = rz / r_cm;

    const real r   = cell_center_r_(i);
    const real th  = cell_center_theta_(j);
    const real ph  = phi_center_(k);

    const real st = sin(th);
    const real ct = cos(th);
    const real sp = sin(ph);
    const real cp = cos(ph);

    // Базис сферических координат в декартовых компонентах.
    const real3 er  = make_real3(st * cp, st * sp, ct);
    const real3 eth = make_real3(ct * cp, ct * sp, -st);
    const real3 eph = make_real3(-sp, cp, 0.);

    const real nr  = nx * er.x  + ny * er.y  + nz * er.z;
    const real nth = nx * eth.x + ny * eth.y + nz * eth.z;
    const real nph = nx * eph.x + ny * eph.y + nz * eph.z;

    const real psi0 = psi_at_sph_(phi, i, j, k);

    const real dpsi_dr =
          upwind_r_sph_(phi, psi0, nr, i, j, k)
        + upwind_theta_sph_(phi, psi0, nth, i, j, k)
        + upwind_phi_sph_(phi, psi0, nph, i, j, k);

    const real psi_new = psi0 - C * DT * (dpsi_dr + psi0 / r_cm);
    return psi_new + monopole_phi_sph_(pos);
}

// -----------------------------------------------------------------------------
// Основное ядро шага
// -----------------------------------------------------------------------------

__global__ void wave_diss_iteration_sph_(
    real* __restrict__ phi_new,
    const real* __restrict__ phi,
    const real* __restrict__ phi_old,
    const real* __restrict__ cell_rho
) {
    const int NR = sph_grid_info_.nr;
    const int NT = sph_grid_info_.nt;
    const int NP = sph_grid_info_.np;

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= NR || j >= NT || k >= NP) return;

    const BoundaryType BT = bc_params_.type;
    const bool periodic = (BT == BoundaryType::Periodic);

    const bool on_boundary = (i == 0 || i == NR - 1 || j == 0 || j == NT - 1);

    const int idx = static_cast<int>(sph_index_(i, j, k));

    // ---------- Граничные ячейки по r/theta ----------
    if (on_boundary) {
        real v;
        switch (BT) {
        case BoundaryType::MonopoleDirichlet:
            v = monopole_phi_sph_(cell_rect_(i, j, k));
            break;

        case BoundaryType::MonopoleRadiative:
            v = radiative_boundary_sph_(phi, i, j, k);
            break;

        case BoundaryType::ZeroDirichlet:
        default:
            // Для Periodic по r/theta физической периодичности нет,
            // поэтому на этих границах оставляем нулевое Дирихле.
            v = real(0);
            break;
        }
        phi_new[idx] = v;
        return;
    }

    // ---------- Источник ----------
    const real G = 1.;
    real rho = cell_rho[idx];
    if (periodic) rho -= bc_params_.rho_mean;
    const real f = real(4) * PI * G * rho;

    const real phi0     = phi[idx];
    const real phi_old0 = phi_old[idx];

    const int im = i - 1;
    const int ip = i + 1;
    const int jm = j - 1;
    const int jp = j + 1;
    const int km = wrap_idx(k - 1, NP);
    const int kp = wrap_idx(k + 1, NP);

    const real r   = cell_center_r_(i);
    const real th  = cell_center_theta_(j);
    const real st  = fmax(sin(th), real(1e-12));

    const real dphi = real(2) * PI / real(NP);

    // ---------- Радиальная часть лапласиана ----------
    const real rim = cell_center_r_(im);
    const real rip = cell_center_r_(ip);

    const real rfm = real(0.5) * (rim + r);
    const real rfp = real(0.5) * (r + rip);

    const real drm = r - rim;
    const real drp = rip - r;
    const real drc = real(0.5) * (rip - rim);

    const real flux_r_plus  = rfp * rfp * (phi[sph_index_(ip, j, k)] - phi0) / drp;
    const real flux_r_minus = rfm * rfm * (phi0 - phi[sph_index_(im, j, k)]) / drm;

    const real lap_r = (flux_r_plus - flux_r_minus) / (r * r * drc);

    // ---------- Тета-часть лапласиана ----------
    const real thm = cell_center_theta_(jm);
    const real thp = cell_center_theta_(jp);

    const real stm = sin(real(0.5) * (thm + th));
    const real stp = sin(real(0.5) * (th + thp));

    const real dthm = th - thm;
    const real dthp = thp - th;
    const real dthc = real(0.5) * (thp - thm);

    const real flux_th_plus =
        stp * (phi[sph_index_(i, jp, k)] - phi0) / dthp;
    const real flux_th_minus =
        stm * (phi0 - phi[sph_index_(i, jm, k)]) / dthm;

    const real lap_th =
        (flux_th_plus - flux_th_minus) / (r * r * st * dthc);

    // ---------- Фи-часть лапласиана ----------
    const real lap_ph =
        (phi[sph_index_(i, j, kp)] - real(2) * phi0 + phi[sph_index_(i, j, km)])
        / (r * r * st * st * dphi * dphi);

    const real lap = lap_r + lap_th + lap_ph;

    // ---------- Диссипация ----------
    real diss = wave_params_.diss_base;

    if (!periodic || bc_params_.periodic_keep_layer) {
        const real r_in = real(0.5) * sph_grid_info_.radius_max;
        const real d = fabs(r) - r_in;
        if (d > 0) {
            diss += wave_params_.diss_extra * d * d;
        }
    }

    const real C  = wave_params_.wave_speed;
    const real DT = wave_params_.dt;

    const real Q = diss * C * DT * real(0.5);
    const real W = real(1) + Q;

    const real A = C * C * DT * DT / W;

    // (1+Q) phi_new - 2 phi + (1-Q) phi_old =
    //     C^2 DT^2 (lap - f)
    phi_new[idx] =
          A * lap
        + (real(2) / W) * phi0
        + ((Q - real(1)) / W) * phi_old0
        - A * f;
}

// -----------------------------------------------------------------------------
// Начальное гравитационное поле
// -----------------------------------------------------------------------------

__global__ void fill_in_initial_grav_sph_(
    real* grav,
    real3 mass_center,
    real mass_value
) {
    const int NR = sph_grid_info_.nr;
    const int NT = sph_grid_info_.nt;
    const int NP = sph_grid_info_.np;

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= NR || j >= NT || k >= NP) return;

    const real3 pos = cell_rect_(i, j, k);

    const real G = 1.;
    const real softening = real(0.01);

    grav[sph_index_(i, j, k)] =
        -G * mass_value / (distance(pos, mass_center) + softening);
}

// -----------------------------------------------------------------------------
// WaveSphSolver
// -----------------------------------------------------------------------------

WaveSphSolver::WaveSphSolver(Params params)
    : WaveSolver{ std::move(params) }
{
    using rrgsim::common::BLOCK_SIZE_IN_3D;
    using rrgsim::common::OverInfo;

    const auto& sph_info =
        std::get<SphGridInfo>(this->params.grid_info);

    const unsigned threads_count = BLOCK_SIZE_IN_3D;

    const unsigned bx = static_cast<unsigned>(
        OverInfo::split_into_blocks(sph_info.nr, BLOCK_SIZE_IN_3D)
    );
    const unsigned by = static_cast<unsigned>(
        OverInfo::split_into_blocks(sph_info.nt, BLOCK_SIZE_IN_3D)
    );
    const unsigned bz = static_cast<unsigned>(
        OverInfo::split_into_blocks(sph_info.np, BLOCK_SIZE_IN_3D)
    );

    this->threads_per_block = dim3{ threads_count, threads_count, threads_count };
    this->blocks_per_grid   = dim3{ bx, by, bz };
}

void WaveSphSolver::fill_in_initial_grav(
    rrgsim::common::sGridContext_ grid_,
    const NewtonianGravParams& initial_grav_params
) const {
    spdlog::debug("WaveSphSolver::fill_in_initial_grav");

    RR::CUDA::CuCall(fill_in_initial_grav_sph_, blocks_per_grid, threads_per_block)(
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

void WaveSphSolver::setup(
    rrgsim::common::sGridContext_ grid_
) {
    const auto& sph_info =
        std::get<SphGridInfo>(this->params.grid_info);

    const real C = this->params.wave_params.wave_speed;

    // Оценка CFL для неравномерной сферической сетки.
    real dr_min = std::numeric_limits<real>::max();
    for (int i = 0; i + 1 < sph_info.nr; ++i) {
        dr_min = std::min(
            dr_min,
            static_cast<real>(cell_center_r_(i + 1) - cell_center_r_(i))
        );
    }

    real dth_min = std::numeric_limits<real>::max();
    for (int j = 0; j + 1 < sph_info.nt; ++j) {
        dth_min = std::min(
            dth_min,
            static_cast<real>(
                cell_center_theta_(j + 1) - cell_center_theta_(j)
            )
        );
    }

    const real dphi = real(2) * PI / real(sph_info.np);

    real h_min = dr_min;

    for (int i = 0; i < sph_info.nr; ++i) {
        const real r = cell_center_r_(i);
        h_min = std::min(h_min, r * dth_min);

        for (int j = 0; j < sph_info.nt; ++j) {
            const real th = cell_center_theta_(j);
            const real st = std::sin(th);
            if (st > real(1e-6)) {
                h_min = std::min(h_min, r * st * dphi);
            }
        }
    }

    constexpr real DIM = real(3);
    this->params.wave_params.dt =
        real(0.5) * h_min / (C * std::sqrt(DIM));

    spdlog::trace(
        "WaveSphSolver::setup wave_params.dt = {}",
        this->params.wave_params.dt
    );

    RR::CUDA::CuCopyToSymbol(
        this->params.wave_params,
        wave_params_,
        RR::CUDA::ToDevice
    );

    // Если SphGridInfo не копируется в константную память снаружи,
    // раскомментируйте:
    // RR::CUDA::CuCopyToSymbol(sph_info, sph_grid_info_, RR::CUDA::ToDevice);

    this->params.mb_initial_grav_params.transform(
        [this, grid_](const NewtonianGravParams& initial_grav_params) {
            this->fill_in_initial_grav(grid_, initial_grav_params);
        }
    );
}

void WaveSphSolver::step(rrgsim::common::sGridContext_ grid_) {
    RR::CUDA::CuCall(wave_diss_iteration_sph_, blocks_per_grid, threads_per_block)(
        grid_->grav_next_,
        grid_->grav_curr_,
        grid_->grav_prev_,
        grid_->rho_
    );
}

} // namespace rrgsim::wave::sph