#pragma once
#include <filesystem>
#include <unordered_set>

namespace rrgsim::io {

    struct TagValue {
        std::string_view tag = "Unknown";
        double value = std::numeric_limits<double>::quiet_NaN();
    };

    class IOHandler {
    public:
        static IOHandler& instance() {
            static IOHandler handler;
            return handler;
        }

        static void setup(std::filesystem::path work_dir);

        void append_table(
            std::string_view tag,
            std::initializer_list<TagValue> items
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


        std::filesystem::path work_dir;

        mutable std::unordered_set<std::string_view> filled;
    };

} // namespace rrgsim::io