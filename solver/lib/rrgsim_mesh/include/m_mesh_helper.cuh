#pragma once
#include <memory>
#include <stdexcept>
#include <co_grid_context.cuh>
#include <co_device_utils.cuh>

namespace rrgsim::mesh {

    using rrgsim::common::GridInfoVariant;
    using rrgsim::common::RectGridInfo;
    using rrgsim::common::SphGridInfo;

    struct MeshHelper {
        // глобальный индекс ячейки
        virtual size_t cell_index(int i, int j, int k) const = 0;

        // моделируемый объём
        virtual real calc_volume() const = 0;

        // минимальный шаг по координате
        virtual real calc_min_dr() const = 0;

    };

    struct RectMeshHelper : public MeshHelper {
        RectMeshHelper(const RectGridInfo& info)
            : rect_info{ info }
        {
        }

        size_t cell_index(int i, int j, int k) const override {
            return (size_t)i
                + (size_t)rect_info.nx * (size_t)j
                + (size_t)(rect_info.nx * rect_info.nx) * (size_t)k;
        }

        real calc_volume() const override {
            return rrgsim::common::cube(rect_info.domain_l);
        }

        real calc_min_dr() const override {
            return rect_info.dx;
        }

        RectGridInfo rect_info;
    };

    class SphMeshHelper : public MeshHelper {
    public:
        SphMeshHelper(const SphGridInfo& info)
            : sph_info{ info }
        {
        }

        // линейный порядок:
        // i (радиус) — самый быстрый,
        // j (широта, colatitude),
        // k (долгота) — самый медленный.
        size_t cell_index(int i, int j, int k) const override {
            return (size_t)i
                + (size_t)sph_info.nr * (size_t)j
                + (size_t)(sph_info.nr * sph_info.nt) * (size_t)k;
        }

        real calc_volume() const override {
            double R3 = rrgsim::common::cube(sph_info.radius_max);
            return 4. / 3. * PI * R3;
        }

        real calc_min_dr() const override {
            throw std::runtime_error{ "Not implemented" };
        }

    private:
        SphGridInfo sph_info;
    };

    using sMeshHelper = std::shared_ptr<MeshHelper>;
    inline sMeshHelper make_mesh_helper(
        const rrgsim::common::GridInfoVariant& grid_info
    )
    {
        if (auto rect_grid_info = std::get_if<RectGridInfo>(&grid_info); rect_grid_info) {
            return std::make_shared<rrgsim::mesh::RectMeshHelper>(*rect_grid_info);
        }
        else if (auto sph_grid_info = std::get_if<SphGridInfo>(&grid_info); sph_grid_info) {
            return std::make_shared<rrgsim::mesh::SphMeshHelper>(*sph_grid_info);
        }
        else {
            throw std::runtime_error{ "make_mesh2p_solver error: invalid grid info variant" };
        }
    }

} // namespace rrgsim::mesh