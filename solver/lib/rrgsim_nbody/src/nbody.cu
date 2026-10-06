#include "nbody_kernel.cuh"
#include "nbody.h"
#include <spdlog/spdlog.h>

namespace rrgsim::nbody {

void predict_step(
    sParticlesContext_ context_,
    const SimParams& sim_params
)
{
    spdlog::debug("NBody::predict_step (t={})", context_->time);

    int blocks_count = calc_blocks_count(context_->info.ntotal);
    RR::CUDA::CuCall(predict_step_, blocks_count, BLOCK_SIZE) (
        context_->acc_,
        context_->vel_,
        context_->pos_,
        context_->vel_predicted_,
        sim_params.dt_dynamics
    );
}

void nbody_acceleration(
    sParticlesContext_ context_,
    const SimParams& sim_params
)
{
    spdlog::debug("NBody::acceleration (t={})", context_->time);

    int blocks_count = calc_blocks_count(context_->info.ntotal);
    RR::CUDA::CuCall(acceleration_kernel_blocked_, blocks_count, BLOCK_SIZE) (
        context_->acc_,
        context_->pos_,
        context_->mass_,
        context_->soft2_
    );
}

void nbody_grav(
    sParticlesContext_ context_,
    const SimParams& sim_params
)
{
    spdlog::debug("NBody::grav (t={})", context_->time);

    int blocks_count = calc_blocks_count(context_->info.ntotal);
    RR::CUDA::CuCall(grav_kernel_blocked_, blocks_count, BLOCK_SIZE) (
        context_->grav_,
        context_->pos_,
        context_->mass_,
        context_->soft2_
    );
}

void correct_step(
    sParticlesContext_ context_,
    const SimParams& sim_params
)
{
    spdlog::debug("NBody::correct_step (t={})", context_->time);

    int blocks_count = calc_blocks_count(context_->info.ntotal);
    RR::CUDA::CuCall(correct_step_, blocks_count, BLOCK_SIZE) (
        context_->acc_,
        context_->vel_,
        context_->vel_predicted_,
        sim_params.dt_dynamics
    );
}

ConservationInfo calc_conservation(
    sParticlesContext context
)
{
    spdlog::info("NBody::conservation");
    const auto& pos = context->pos;
    const auto& vel = context->vel;
    const auto& mass = context->mass;
    const auto& grav = context->grav;
    if (vel.size() != mass.size()) {
        throw std::runtime_error("Conservation calculation error: velocity and mass arrays size mismatch");
    }
    const size_t N = vel.size();

    ConservationInfo info;
    info.momentum = real3(0., 0., 0.);
    info.angular = real3(0., 0., 0.);
    info.Ek = 0;
    info.Ep = 0;

    for (size_t i = 0; i < N; ++i) {
        const real x = pos[i].x;
        const real y = pos[i].y;
        const real z = pos[i].z;
        const real m = mass[i];
        const real vx = vel[i].x;
        const real vy = vel[i].y;
        const real vz = vel[i].z;

        info.momentum.x += m * vx;
        info.momentum.y += m * vy;
        info.momentum.z += m * vz;

        info.angular.x += m * (vz * y - vy * z);
        info.angular.y += m * (vx * z - vz * x);
        info.angular.z += m * (vy * x - vx * y);

        info.Ek += m * (vx * vx + vy * vy + vz * vz);
        info.Ep += m * grav[i];
    }

    info.Ek *= 0.5;
    info.Ep *= 0.5;
    info.time = context->time;
    return info;
}

} // namespace rrgsim::nbody