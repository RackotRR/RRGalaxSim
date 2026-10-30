#pragma once
#include "rrgsim_integration.cuh"
#include <rm_mesh2p.cuh>
#include <rm_p2mesh.cuh>
#include <nbody.cuh>
#include <wave.cuh>
#include <co_sim_params.h>
#include <co_info_structs.h>

namespace rrgsim::integration {


class RectWaveIntegrationSolver : public IntegrationSolver {
    void init(
        const IntegrationParams& params,
        rrgsim::common::sParticlesContext_ particles_context_,
        rrgsim::common::sGridContext_ grid_context_
    ) override {
        this->sim_params = params.sim_params;

        std::string for_error =  "for rect wave integration solver";

        if (false == params.mb_grid_info.has_value()) {
            throw std::runtime_error{ "No grid info " + for_error };
        }
        RR::CUDA::CuCopyToSymbol(
            grid_context_->info,
            rrgsim::common::grid_info_,
            RR::CUDA::ToDevice
        );

        if (false == params.mb_wave_params.has_value()) {
            throw std::runtime_error{ "No wave params " + for_error };
        }
        this->wave_params = params.mb_wave_params.value();
        RR::CUDA::CuCopyToSymbol(
            wave_params,
            rrgsim::common::wave_params_,
            RR::CUDA::ToDevice
        );

        if (nullptr == particles_context_) {
            throw std::runtime_error{ "No particles context " + for_error };
        }
        this->particles_context_ = particles_context_;

        if (nullptr == grid_context_) {
            throw std::runtime_error{ "No grid context " + for_error };
        }
        this->grid_context_ = grid_context_;
    }

    void precalc() override {
        // масса
        rrgsim::rect_mesh::convert_particles_to_grid(particles_context_, grid_context_);

        // real3 mass_center = rrgsim::nbody::calc_mass_center(particles_context_);
        // rrgsim::wave::set_boundary_monopole(
        //     1.,
        //     mass_center,
        //     false,
        //     grid_info.dx
        // );

        rrgsim::wave::set_boundary_periodic(
            1.,
            grid_context_->info,
            false
        );

        // rrgsim::wave::fill_in_initial_grav(
        //     grid_context_,
        //     rrgsim::nbody::calc_mass_center(particles_context_),
        //     1.
        // );

        rrgsim::wave::solve_wave_equation(grid_context_, wave_params, wave_params.setup_iterations);
        rrgsim::rect_mesh::calc_acceleration_field(grid_context_);

        // всё нужно синхронизировать
        particles_grav_is_actual = false;
        particles_acc_is_actual = false;
    }

    void sync_particles_and_grid() override {
        if (false == particles_grav_is_actual) {
            rrgsim::rect_mesh::project_grid_grav_on_particles(grid_context_, particles_context_);
        }

        if (false == particles_acc_is_actual) {
            rrgsim::rect_mesh::project_grid_acceleration_on_particles(grid_context_, particles_context_);
        }

        // всё здесь синхронизировалось
        particles_grav_is_actual = true;
        particles_acc_is_actual = true;
    }

    void calc_step() override {
        rrgsim::nbody::predict_step(particles_context_, sim_params);

        rrgsim::rect_mesh::convert_particles_to_grid(particles_context_, grid_context_);
        rrgsim::wave::solve_wave_equation(grid_context_, wave_params, wave_params.iterations);
        rrgsim::rect_mesh::calc_acceleration_field(grid_context_);

        rrgsim::rect_mesh::project_grid_acceleration_on_particles(grid_context_, particles_context_);
        particles_acc_is_actual = true; // ускорение на частицах актуально
        particles_grav_is_actual = false; // частицы сдвинулись, нужно пересчитать

        // тут уточняется скорость, координаты не меняются
        rrgsim::nbody::correct_step(particles_context_, sim_params);
    }

private:
    rrgsim::common::WaveParams wave_params;
    rrgsim::common::SimParams sim_params;

    bool particles_acc_is_actual = false;
    bool particles_grav_is_actual = false;

    rrgsim::common::sGridContext_ grid_context_;
    rrgsim::common::sParticlesContext_ particles_context_;
};

} // namespace rrgsim::integration