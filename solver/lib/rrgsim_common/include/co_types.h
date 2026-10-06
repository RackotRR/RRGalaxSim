#pragma once

#define USE_FP32

#ifdef USE_FP32
#define real float
#define real2 float2
#define real3 float3
#define make_real2 make_float2
#define make_real3 make_float3
#else
#define real double
#define real2 double2
#define real3 double3
#define make_real2 make_double2
#define make_real3 make_double3
#endif
