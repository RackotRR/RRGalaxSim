#pragma once
#include "rrgsim_integration.cuh"
#include <m_mesh_solver.cuh>
#include <nbody.cuh>
#include <wave.cuh>
#include <co_sim_params.h>
#include <co_info_structs.h>

namespace rrgsim::integration {


class WaveIntegrationSolver : public IntegrationSolver {
    void init(
        const IntegrationParams& params,
        rrgsim::common::sParticlesContext_ particles_context_,
        rrgsim::common::sGridContext_ grid_context_
    ) override {
        this->sim_params = params.sim_params;

        std::string for_error =  "for wave integration solver";

        if (false == params.mb_grid_info.has_value()) {
            throw std::runtime_error{ "No grid info " + for_error };
        }

        if (false == params.mb_wave_params.has_value()) {
            throw std::runtime_error{ "No wave params " + for_error };
        }
        this->wave_params = params.mb_wave_params.value();

        if (nullptr == particles_context_) {
            throw std::runtime_error{ "No particles context " + for_error };
        }
        this->particles_context_ = particles_context_;

        if (nullptr == grid_context_) {
            throw std::runtime_error{ "No grid context " + for_error };
        }
        this->grid_context_ = grid_context_;
        this->mesh_solver = rrgsim::mesh::make_mesh_solver(grid_context_->info);
    }

    void precalc() override {
        // масса
        mesh_solver->convert_particles_to_grid(particles_context_, grid_context_);

        // real3 mass_center = rrgsim::nbody::calc_mass_center(particles_context_);
        // rrgsim::wave::set_boundary_monopole(
        //     1.,
        //     mass_center,
        //     false,
        //     grid_info.dx
        // );


        wave::PeriodicBoundaryParams periodic_boundary;
        periodic_boundary.keep_extra_dissipative_layer = false;
        periodic_boundary.total_mass = 1.;

        wave::Params params;
        params.boundary_params = periodic_boundary;
        params.grid_info = grid_context_->info;
        params.mb_initial_grav_params = tl::nullopt;
        params.wave_params = wave_params;

        wave_solver = wave::make_wave_solver(params);
        wave_solver->setup(grid_context_);

        wave_solver->solve(grid_context_, wave_params.setup_iterations);
        mesh_solver->calc_acceleration_field(grid_context_);

        // всё нужно синхронизировать
        particles_grav_is_actual = false;
        particles_acc_is_actual = false;
    }

    void sync_particles_and_grid() override {
        if (false == particles_grav_is_actual) {
            mesh_solver->project_grid_grav_on_particles(grid_context_, particles_context_);
        }

        if (false == particles_acc_is_actual) {
            mesh_solver->project_grid_acceleration_on_particles(grid_context_, particles_context_);
        }

        // всё здесь синхронизировалось
        particles_grav_is_actual = true;
        particles_acc_is_actual = true;
    }

    void calc_step() override {
        rrgsim::nbody::predict_step(particles_context_, sim_params);

        mesh_solver->convert_particles_to_grid(particles_context_, grid_context_);
        wave_solver->solve(grid_context_, wave_params.iterations);
        mesh_solver->calc_acceleration_field(grid_context_);

        mesh_solver->project_grid_acceleration_on_particles(grid_context_, particles_context_);
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

    rrgsim::mesh::sMeshSolver mesh_solver;
    rrgsim::wave::sWaveSolver wave_solver;
};

} // namespace rrgsim::integration