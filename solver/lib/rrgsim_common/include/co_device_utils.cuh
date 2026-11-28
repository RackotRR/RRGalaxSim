#pragma once
#include <cuda_runtime.h>
#include "co_types.h"
#include "co_device_structs.cuh"

namespace rrgsim::common {

template<typename T>
__host__ __device__
real dot(const T& vec1, const T& vec2);

template<>
inline __host__ __device__
real dot(const real2& vec1, const real2& vec2) {
	return
		vec1.x * vec2.x +
		vec1.y * vec2.y;
}

template<>
inline __host__ __device__
real dot(const real3& vec1, const real3& vec2) {
	return
		vec1.x * vec2.x +
		vec1.y * vec2.y +
		vec1.z * vec2.z;
}

template<typename T>
__host__ __device__
real norm(const T& vec) {
	return sqrt(dot(vec, vec));
}

inline __host__ __device__
real distance(const real3& v1, const real3& v2) {
	real3 r = make_real3(
		v2.x - v1.x,
		v2.y - v1.y,
		v2.z - v1.z
	);
	return sqrt(dot(r, r));
}

template<typename T>
__host__ __device__ T clamp(T val, T min_val, T max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

inline __host__ __device__
real sqr(real x) {
	return x * x;
}

inline __host__ __device__
real cube(real x) {
	return x * x * x;
}

inline __host__ __device__
real normalize_two_pi(
	real phi
)
{
	return phi - TWO_PI * std::floor(phi / TWO_PI);
}

// Декартовы координаты в сферические
inline __host__ __device__
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

// обратное преобразование сферических координат в вычислительные [0,1]
inline __host__ __device__
real3 coord_spherical_to_uniform(
	const real3& sph,
	const rrgsim::common::SphGridInfo& sph_grid_info
)
{
	const auto& [r, theta, phi] = sph;

	// r = R * x^p → x = (r/R)^(1/p)
	real xr = r > 0
		? pow(r / sph_grid_info.radius_max, 1 / sph_grid_info.proj_r_degree)
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
inline __device__
real3 coord_spherical_to_uniform_(const real3& sph) {
	return coord_spherical_to_uniform(sph, sph_grid_info_);
}

inline __host__ __device__
real cell_center_r(
	int i,
	const rrgsim::common::SphGridInfo& sph_grid_info
) {
	// x = (i + 1/2) / Nr,  r = R * x^p
	const real x = (i + 0.5) / static_cast<real>(sph_grid_info.nr);
	return sph_grid_info.radius_max * pow(x, sph_grid_info.proj_r_degree);
}
inline __device__
real cell_center_r_(int i) {
	return cell_center_r(i, sph_grid_info_);
}

inline __host__ __device__
real cell_center_theta(
	int j,
	const rrgsim::common::SphGridInfo& sph_grid_info
) {
	// x = (j + 1/2) / Nt
	// theta = pi/2 + arcsin(2*x - 1)
	const real x = (j + 0.5) / static_cast<real>(sph_grid_info.nt);
	return 0.5 * PI + asin(2 * x - 1);
}

inline __device__
real cell_center_theta_(int j) {
	return cell_center_theta(j, sph_grid_info_);
}


inline __host__ __device__
size_t sph_index(int i, int j, int k, const rrgsim::common::SphGridInfo& sph_grid_info) {
	return (size_t)i
		+  (size_t)sph_grid_info.nr * (size_t)j
		+  (size_t)sph_grid_info.nr * sph_grid_info.nt * (size_t)k;
}
inline __device__
size_t sph_index_(int i, int j, int k) {
	return sph_index(i, j, k, sph_grid_info_);
}


inline __host__ __device__
size_t rect_index(int i, int j, int k, const rrgsim::common::RectGridInfo& rect_grid_info) {
	return (size_t)i
		+  (size_t)rect_grid_info.nx * (size_t)j
		+  (size_t)rect_grid_info.nx * rect_grid_info.nx * (size_t)k;
}
inline __device__
size_t rect_index_(int i, int j, int k) {
	return rect_index(i, j, k, rect_grid_info_);
}

#define AT(x, y, z) (size_t(x) + size_t(y) * (NX) + size_t(z) * (NX) * (NX))

} // namespace rrgsim::common