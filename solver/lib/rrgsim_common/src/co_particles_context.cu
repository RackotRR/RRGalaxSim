#include "co_particles_context.h"

#include <spdlog/spdlog.h>

namespace rrgsim::common {

void ParticlesContext::fill_device_data(
    const sParticlesContext_& context_
)
{
    context_->mass_.to_vector(this->mass);
    context_->vel_.to_vector(this->vel);
    context_->pos_.to_vector(this->pos);
    context_->grav_.to_vector(this->grav);
    this->time = context_->time;
}



sParticlesContext_
initialize_particles_context_(
    const common::ParticlesData& particles_data
)
{
    spdlog::info("Initialize device particles context");

    sParticlesContext_ context = std::make_shared<ParticlesContext_>();
    context->pos_ = particles_data.pos;
    context->vel_ = particles_data.vel;
    context->mass_ = particles_data.mass;
    context->soft2_ = particles_data.soft2;

    const size_t N = particles_data.info.ntotal;

    context->acc_ = CuDarray<double3>(N);
    context->vel_predicted_ = CuDarray<double3>(N);
    context->grav_ = CuDarray<double>(N);

    context->info = particles_data.info;
    RR::CUDA::CuCopyToSymbol(context->info, rrgsim::common::particles_info_, RR::CUDA::ToDevice);

    return context;
}

sParticlesContext
initialize_particles_context(
    common::ParticlesData particles_data
)
{
    spdlog::info("Initialize host particles context");

    auto particles_context = std::make_shared<rrgsim::common::ParticlesContext>();
    particles_context->mass = std::move(particles_data.mass);
    particles_context->pos = std::move(particles_data.pos);
    particles_context->vel = std::move(particles_data.vel);
    particles_context->grav = std::vector<double>(
        particles_context->pos.size(),
        std::numeric_limits<double>::quiet_NaN()
    );
    return particles_context;
}

} // namespace rrgsim::common