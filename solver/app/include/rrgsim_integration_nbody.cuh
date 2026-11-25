#pragma once
#include "rrgsim_integration.cuh"
#include <co_sim_params.h>
#include <nbody.cuh>
#include <m_mesh_solver.cuh>

namespace rrgsim::integration {

class NBodyIntegrationSolver : public IntegrationSolver {
public:
    void init(
        const IntegrationParams& params,
        rrgsim::common::sParticlesContext_ particles_context_,
        rrgsim::common::sGridContext_ grid_context_
    ) override {
        this->sim_params = params.sim_params;
        this->particles_context_ = particles_context_;
        this->grid_context_ = grid_context_;

        if (grid_context_) {
            this->mesh_solver = rrgsim::mesh::make_mesh_solver(grid_context_->info);
        }

        if (nullptr == particles_context_) {
            throw std::runtime_error{ "No particles context NBody integration solver " };
        }
    }

    void precalc() override {
        rrgsim::nbody::nbody_grav(particles_context_);
        rrgsim::nbody::nbody_acceleration(particles_context_);
        particles_grav_is_actual = true;
    }

    void sync_particles_and_grid() override {
        if (nullptr == grid_context_) {
            return;
        }

        mesh_solver->convert_particles_to_grid(particles_context_, grid_context_);

        if (false == particles_grav_is_actual) {
            // в calc_step не рассчитывается потенциал
            rrgsim::nbody::nbody_grav(particles_context_);
        }

        mesh_solver->project_particles_grav_on_grid(particles_context_, grid_context_);
        mesh_solver->project_particles_acceleration_on_grid(particles_context_, grid_context_);

        particles_grav_is_actual = true;
    }

    void calc_step() override {
        rrgsim::nbody::predict_step(particles_context_, sim_params);
        rrgsim::nbody::nbody_acceleration(particles_context_);
        rrgsim::nbody::correct_step(particles_context_, sim_params);

        particles_grav_is_actual = false;
    }

private:
    rrgsim::common::sParticlesContext_ particles_context_;
    rrgsim::common::sGridContext_ grid_context_;

    bool particles_grav_is_actual = false;

    rrgsim::common::SimParams sim_params;

    rrgsim::mesh::sMeshSolver mesh_solver;
};

} // namespace rrgsim::integration