#include <spdlog/spdlog.h>
#include "nbody_p2mesh.cuh"
#include "p2mesh.h"

namespace rrgsim::nbody {

void convert_particles_to_grid(
    const sParticlesContext_ particles_,
    sGridContext_ grid_
)
{
	RR::CUDA::CuDeviceSync();

    const int num_cells = rrgsim::common::cube(grid_->info.nx);
    const int num_particles = particles_->info.ntotal;
    const int num_cell_blocks = (num_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_particle_blocks = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int over_particles = num_particle_blocks;
    const int over_cells = num_cell_blocks;
    const int over_blocks = BLOCK_SIZE;

    if (nullptr == grid_->particles_cell_info_) {
        grid_->particles_cell_info_ = CuDarray<ParticleCellInfo>(num_particles);
    }
    if (nullptr == grid_->cell_info_) {
        grid_->cell_info_ = CuDarray<CellInfo>(num_cells);
    }
    if (nullptr == grid_->cell_particles_count_) {
        grid_->cell_particles_count_ = CuDarray<int>(num_cells);
    }
    if (nullptr == grid_->particles_in_block_) {
        grid_->particles_in_block_ = CuDarray<int>(num_cell_blocks);
    }

	grid_->particles_cell_info_.set_zero();
	grid_->cell_info_.set_zero();
	grid_->cell_particles_count_.set_zero();
	grid_->particles_in_block_.set_zero();
	grid_->mass_.set_zero();

    RR::CUDA::CuCall(assignParticlesToCells, over_particles, over_blocks) (
        particles_->pos_,
        grid_->particles_cell_info_,
        grid_->cell_info_,
        grid_->cell_particles_count_,
        num_particles
    );
    RR::CUDA::CuCall(computePrefixSums, over_cells, over_blocks) (
        grid_->cell_info_,
        grid_->particles_in_block_,
        num_cells
    );
    RR::CUDA::CuCall(adjustGlobalPrefixSums, over_cells, over_blocks) (
        grid_->cell_info_,
        grid_->particles_in_block_,
        num_cells
    );
	RR::CUDA::CuCall(computeCellMassesUnsorted, over_particles, over_blocks) (
		particles_->mass_,
		grid_->particles_cell_info_,
		grid_->mass_,
		num_particles
	);
}

} // namespace rrgsim::nbody