#pragma once
#include <cm_mesh2p.cuh>

namespace rrgsim::mesh::rect {

struct RectMesh2PSolver : public Mesh2PSolver {
    void project_grid_acceleration_on_particles(
        sGridContext_ grid_context_,
        sParticlesContext_ particles_context_
    ) override;

    void project_grid_grav_on_particles(
        const sGridContext_ grid_context_,
        sParticlesContext_ particles_context_
    ) override;

    GridProjectionOnParticles
    project_grid_onto_particles(
        sGridContext_ grid_context_,
        sParticlesContext_ particles_context_
    ) override;
};

} // namespace rrgsim::mesh::rect