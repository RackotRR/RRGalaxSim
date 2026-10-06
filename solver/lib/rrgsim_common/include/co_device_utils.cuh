#pragma once
#include <cuda_runtime.h>
#include <co_types.h>

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

#define AT(x, y, z) ((x) + (y) * (NX) + (z) * (NX) * (NX))

} // namespace rrgsim::common