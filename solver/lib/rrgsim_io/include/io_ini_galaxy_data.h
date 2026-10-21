#pragma once
#include "io_common.h"

namespace rrgsim::io {

    struct IniGalaxyComponentData {
        real mass = NAN_VALUE;
        real soft = NAN_VALUE;
        fs::path ini_file;
    };

    struct IniGalaxyData {
        real mass_star = NAN_VALUE;
        real mass_dark = NAN_VALUE;
        real soft_star = NAN_VALUE;
        real soft_dark = NAN_VALUE;
        tl::optional<fs::path> mb_ini_file_star;
        tl::optional<fs::path> mb_ini_file_dark;
    };

} // namespace rrgsim::io