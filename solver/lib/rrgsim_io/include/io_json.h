#pragma once
#include "io_common.h"
#include "io_ini_galaxy_data.h"

#include "co_sim_params.h"
#include "co_info_structs.h"

#include <nlohmann/json.hpp>

namespace rrgsim::common {
    void to_json(nlohmann::json& j, const SimParams& sim_params);
    void from_json(const nlohmann::json& j, SimParams& sim_params);

    void to_json(nlohmann::json& j, const RectGridInfo& grid_info);
    void from_json(const nlohmann::json& j, RectGridInfo& grid_info);

    void to_json(nlohmann::json& j, const SphGridInfo& grid_info);
    void from_json(const nlohmann::json& j, SphGridInfo& grid_info);

    void to_json(nlohmann::json& j, const WaveParams& wave_params);
    void from_json(const nlohmann::json& j, WaveParams& wave_params);
}

namespace nlohmann {
    template <>
    struct adl_serializer<rrgsim::common::GridInfoVariant> {
        static constexpr const char RECT_NAME[] = "rect";
        static constexpr const char SPH_NAME[] = "sphere";

        void to_json(json& j, const rrgsim::common::GridInfoVariant& s);
        void from_json(const json& j, rrgsim::common::GridInfoVariant& s);
    };
} // namespace nlohmann

namespace rrgsim::io {
    struct ParsedParams {
        IniGalaxyData galaxy_data;
        common::SimParams sim_params;
        tl::optional<common::GridInfoVariant> mb_grid_info;
        tl::optional<common::WaveParams> mb_wave_params;
    };

    void to_json(nlohmann::json& j, const IniGalaxyData& galaxy_data);
    void from_json(const nlohmann::json& j, IniGalaxyData& galaxy_data);

    tl::expected<ParsedParams, std::string>
    parse_params_json(const fs::path& path);
} // namespace rrgsim::io