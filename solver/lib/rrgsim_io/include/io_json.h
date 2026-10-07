#pragma once
#include "io_particles.h"

#include "co_sim_params.h"
#include "co_info_structs.h"

#include <nlohmann/json.hpp>
#include <rrgsim_tl.h>

namespace rrgsim::common {
    void to_json(nlohmann::json& j, const SimParams& sim_params);
    void from_json(const nlohmann::json& j, SimParams& sim_params);

    void to_json(nlohmann::json& j, const GridInfo& grid_info);
    void from_json(const nlohmann::json& j, GridInfo& grid_info);

    void to_json(nlohmann::json& j, const WaveParams& wave_params);
    void from_json(const nlohmann::json& j, WaveParams& wave_params);
}

namespace rrgsim::io {
    struct ParsedParams {
        IniGalaxyData galaxy_data;
        common::SimParams sim_params;
        tl::optional<common::GridInfo> mb_grid_info;
        tl::optional<common::WaveParams> mb_wave_params;
    };

    void to_json(nlohmann::json& j, const IniGalaxyData& galaxy_data);
    void from_json(const nlohmann::json& j, IniGalaxyData& galaxy_data);

    /// @brief Валидация и дополнение зависимых полей
    /// @param params Параметры из json
    void validate_dependent_fields(ParsedParams& params);

    tl::expected<ParsedParams, std::string>
    parse_params_json(const std::filesystem::path& path);
} // namespace rrgsim::io