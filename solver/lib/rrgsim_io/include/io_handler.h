#pragma once
#include "io_common.h"
#include <unordered_set>
#include <fstream>

namespace rrgsim::io {

    struct TagValue {
        std::string_view tag = "Unknown";
        double value = std::numeric_limits<double>::quiet_NaN();
        int precision = 8;
    };

    class IOHandler {
    public:
        static IOHandler& instance() {
            static IOHandler handler;
            return handler;
        }

        static void setup(fs::path work_dir);

        // TODO: мне не нравится, что файл открывается каждый раз
        void append_table(
            std::string tag,
            std::initializer_list<TagValue> items,
            const tl::optional<std::string>& mb_subdir = tl::nullopt
        ) const;

    private:
        IOHandler() = default;


        void fill_in_header(
            std::ofstream& stream,
            std::initializer_list<TagValue> items
        ) const;
        void fill_in_row(
            std::ofstream& stream,
            std::initializer_list<TagValue> items
        ) const;


        fs::path work_dir;

        mutable std::unordered_set<std::string> filled;
    };

} // namespace rrgsim::io