#include <fstream>

#include <spdlog/spdlog.h>

#include "io_json.h"

namespace rrgsim::common {
    void to_json(nlohmann::json& j, const SimParams& sim_params) {
        j["time_max"] = sim_params.time_max;
        j["dt_save"] = sim_params.dt_save;
        j["dt_dynamics"] = sim_params.dt_dynamics;
    }
    void from_json(const nlohmann::json& j, SimParams& sim_params) {
        j.at("time_max").get_to(sim_params.time_max);
        j.at("dt_save").get_to(sim_params.dt_save);
        j.at("dt_dynamics").get_to(sim_params.dt_dynamics);
    }

    void to_json(nlohmann::json& j, const GridInfo& grid_info) {
        j.at("nx") = grid_info.nx;
        j.at("dx") = grid_info.dx;
        j.at("bc_frac") = 1. - grid_info.sim_l / grid_info.domain_l;
    }
    void from_json(const nlohmann::json& j, GridInfo& grid_info) {
        j.at("nx").get_to(grid_info.nx);
        j.at("dx").get_to(grid_info.dx);

        grid_info.domain_l = grid_info.dx * grid_info.nx;

        real bc_frac = j.at("bc_frac").get<real>();
        grid_info.sim_l = (1. - bc_frac) * grid_info.domain_l;
        grid_info.bc_l = 0.5 * bc_frac * grid_info.domain_l;

        grid_info.domain_max = 0.5 * grid_info.domain_l;
        grid_info.domain_min = -0.5 * grid_info.domain_l;
    }

    void to_json(nlohmann::json& j, const WaveParams& wave_params) {
        j.at("diss_base") = wave_params.diss_base;
        j.at("diss_extra") = wave_params.diss_extra;
        j.at("wave_speed") = wave_params.wave_speed;
        j.at("setup_iterations") = wave_params.setup_iterations;
    }
    void from_json(const nlohmann::json& j, WaveParams& wave_params) {
        j.at("diss_base").get_to(wave_params.diss_base);
        j.at("diss_extra").get_to(wave_params.diss_extra);
        j.at("wave_speed").get_to(wave_params.wave_speed);
        j.at("setup_iterations").get_to(wave_params.setup_iterations);
    }
} // namespace rrgsim::common

namespace rrgsim::io {
    void to_json(nlohmann::json& j, const IniGalaxyData& galaxy_data) {
        j["mass_star"] = galaxy_data.mass_star;
        j["mass_dark"] = galaxy_data.mass_dark;
        j["soft_star"] = galaxy_data.soft_star;
        j["soft_dark"] = galaxy_data.soft_dark;

        auto path_to_json = [](const fs::path& path) -> nlohmann::json {
            return path.string();
        };
        j["ini_file_star"] = galaxy_data.mb_ini_file_star.map_or(path_to_json, nlohmann::json{ nullptr });
        j["ini_file_dark"] = galaxy_data.mb_ini_file_dark.map_or(path_to_json, nlohmann::json{ nullptr });
    }
    void from_json(const nlohmann::json& j, IniGalaxyData& galaxy_data) {
        j.at("mass_star").get_to(galaxy_data.mass_star);
        j.at("mass_dark").get_to(galaxy_data.mass_dark);
        j.at("soft_star").get_to(galaxy_data.soft_star);
        j.at("soft_dark").get_to(galaxy_data.soft_dark);

        auto json_to_path = [&j](std::string_view key) -> tl::optional<fs::path> {
            if (auto iter = j.find(key); iter != j.end() && !iter->is_null()) {
                return iter->get<std::string>();
            }
            else {
                return tl::nullopt;
            }
        };
        galaxy_data.mb_ini_file_star = json_to_path("ini_file_star");
        galaxy_data.mb_ini_file_dark = json_to_path("ini_file_dark");
    }

    template<typename T>
    bool parse_opt_json_object(
        std::string_view key,
        tl::optional<T>& out_obj,
        const nlohmann::json& json
    )
    {
        if (json.contains(key)) {
            spdlog::debug("Parse {} section", key);
            out_obj = json.at(key).get<T>();
        }
        else {
            spdlog::debug("No {} section", key);
            out_obj = tl::nullopt;
        }

        return true;
    }

    template<typename T>
    bool parse_mandatory_json_object(
        std::string_view key,
        T& out_obj,
        const nlohmann::json& json
    )
    {
        if (json.contains(key)) {
            spdlog::debug("Parse {} section", key);
            out_obj = json.at(key).get<T>();
            return true;
        }
        else {
            spdlog::error("No {} section", key);
            return false;
        }
    }

    void validate_dependent_fields(ParsedParams& parsed) {
        if (parsed.mb_grid_info && parsed.mb_wave_params) {
            parsed.sim_params.use_wave_model = true;

            // выводим шаг по времени из условия CFL
            real dx = parsed.mb_grid_info->dx;
            real c = parsed.mb_wave_params->wave_speed;
            constexpr real DIM = 3;
            parsed.mb_wave_params->dt = 0.5 * dx / (c * std::sqrt(DIM));
        }

        spdlog::debug(
            parsed.sim_params.use_wave_model
                ? "Use wave model"
                : "Don't use wave model"
        );
    }

    tl::expected<ParsedParams, std::string>
    parse_params_json(const std::filesystem::path& path) {
        try {
            if (false == std::filesystem::exists(path)) {
                return tl::make_unexpected(
                    fmt::format("Expected params json: '{}'", path.string())
                );
            }

            std::ifstream stream{ path };
            nlohmann::json json; stream >> json;

            ParsedParams parsed;

            bool succeed =
                parse_opt_json_object(
                    "grid_params",
                    parsed.mb_grid_info,
                    json
                )
                &&
                parse_mandatory_json_object(
                    "sim_params",
                    parsed.sim_params,
                    json
                )
                &&
                parse_mandatory_json_object(
                    "ini_params",
                    parsed.galaxy_data,
                    json
                )
                && parse_opt_json_object(
                    "wave_params",
                    parsed.mb_wave_params,
                    json
                );

            if (succeed) {
                validate_dependent_fields(parsed);
                spdlog::info("Params json parsed successfully");
                return parsed;
            }
            else {
                return tl::make_unexpected(
                    fmt::format("Params json parsing failed")
                );
            }
        }
        catch (const std::exception& ex) {
            return tl::make_unexpected(
                fmt::format("Unexpected error on params json parsing: {}", ex.what())
            );
        }
    }
} // namespace rrgsim::io