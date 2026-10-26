#include "wave_steady_checker.cuh"
#include <co_info_structs.h>
#include <spdlog/spdlog.h>

namespace rrgsim::wave {

using rrgsim::common::BLOCK_SIZE;
using rrgsim::wave::detail::BitsType;

namespace {
    // val должен быть >= 0
    __device__ __forceinline__ void atomic_max_non_neg(
        BitsType* addr,
        real val
    )
    {
        BitsType bits;

        if constexpr (std::is_same_v<real, float>) {
            bits = __float_as_uint(val);
        }
        else {
            bits = __double_as_longlong(val);
        }

        atomicMax(addr, bits);
    }

    // ---------- Ядро редукции ----------
    __global__ void reduce_step_error(
        const real* __restrict__ u_new,
        const real* __restrict__ u_old,
        int N,
        BitsType* maxdiff,
        BitsType* maxabs
    )
    {
        int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

        real diff = real(0);
        real absu = real(0);
        if (i < N) {
            real un = u_new[i];
            real uo = u_old[i];
            diff = fabs(un - uo);
            absu = fabs(un);
        }

        // warp-редукция
        for (int off = 16; off > 0; off >>= 1) {
            diff = fmax(diff, __shfl_down_sync(0xffffffff, diff, off));
            absu = fmax(absu, __shfl_down_sync(0xffffffff, absu, off));
        }

        __shared__ real s_diff[32];
        __shared__ real s_absu[32];
        int lane = threadIdx.x & 31;
        int wid  = threadIdx.x >> 5;
        if (lane == 0) {
            s_diff[wid] = diff;
            s_absu[wid] = absu;
        }
        __syncthreads();

        constexpr int NWARPS = (BLOCK_SIZE + 31) / 32;
        if (wid == 0) {
            diff = (lane < NWARPS)
                ? s_diff[lane]
                : real(0);
            absu = (lane < NWARPS)
                ? s_absu[lane]
                : real(0);

            for (int off = 16; off > 0; off >>= 1) {
                diff = fmax(diff, __shfl_down_sync(0xffffffff, diff, off));
                absu = fmax(absu, __shfl_down_sync(0xffffffff, absu, off));
            }

            if (lane == 0) {
                atomic_max_non_neg(maxdiff, diff);
                atomic_max_non_neg(maxabs,  absu);
            }
        }
    }
} // namespace


SteadyStateChecker::SteadyStateChecker(real tolerance, int hold)
    : tol{ tolerance }
    , hold_need{ hold }
    , hold_count{ 0 }
{
}

double SteadyStateChecker::calc_max_diff(
    const RR::CUDA::CuDarray<real>& u_new_,
    const RR::CUDA::CuDarray<real>& u_old_
)
{
    int N = static_cast<int>(u_new_.size());
    if (u_new_.size() != u_old_.size()) {
        throw std::runtime_error("SteadyStateChecker: arrays size mismatch");
    }

    int blocks = rrgsim::common::OverInfo::split_into_blocks(N, BLOCK_SIZE);

    maxdiff_.set_zero();
    maxabs_.set_zero();

    RR::CUDA::CuCall(reduce_step_error, blocks, BLOCK_SIZE) (
        u_new_,
        u_old_,
        N,
        maxdiff_,
        maxabs_
    );


    BitsType diff_bits = maxdiff_.to_mem();
    BitsType abs_bits = maxabs_.to_mem();

    real diff = *reinterpret_cast<const real*>(&diff_bits);
    real abs  = *reinterpret_cast<const real*>(&abs_bits);;

    real err = diff / (abs + EPS);
    return err;
}


// true -> установление достигнуто
bool SteadyStateChecker::check(
    double err
)
{
    if (err < tol) {
        ++hold_count;
    }
    else {
        hold_count = 0;
    }

    return hold_count >= hold_need;
}

bool SteadyStateChecker::check(
    const RR::CUDA::CuDarray<real>& u_new_,
    const RR::CUDA::CuDarray<real>& u_old_
)
{
    double err = calc_max_diff(u_new_, u_old_);
    return check(err);
}


} // namespace rrgsim::wave