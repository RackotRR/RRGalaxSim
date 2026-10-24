#pragma once
#include <RR/CUDA/CuCommon.cuh>
#include <co_types.h>

namespace rrgsim::wave {

    namespace detail {
        template <typename T>
        struct BitAtomic;

        template <>
        struct BitAtomic<float> {
            using T = unsigned int;
        };

        template <>
        struct BitAtomic<double> {
            using T = unsigned long long;
        };

        using BitsType = BitAtomic<real>::T;
    } // namespace detail


struct SteadyStateChecker {
    RR::CUDA::CuDarray<detail::BitsType> maxdiff_ = RR::CUDA::CuDarray<detail::BitsType>(1ull);
    RR::CUDA::CuDarray<detail::BitsType> maxabs_ = RR::CUDA::CuDarray<detail::BitsType>(1ull);
    static constexpr real EPS = 1.E-30;
    real tol = 1.E-6;
    int hold_need = 5;
    int hold_count = 0;

    SteadyStateChecker(real tolerance, int hold);

    double calc_max_diff(
        const RR::CUDA::CuDarray<real>& u_new_,
        const RR::CUDA::CuDarray<real>& u_old_
    );

    // true -> установление достигнуто
    bool check(
        const RR::CUDA::CuDarray<real>& u_new_,
        const RR::CUDA::CuDarray<real>& u_old_
    );

    bool check(double err);
};


} // namespace rrgsim::wave