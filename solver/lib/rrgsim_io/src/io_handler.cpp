#include "io_handler.h"

#include <fstream>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <range/v3/view/drop_last.hpp>
#include <range/v3/view/take_last.hpp>
#include <range/v3/algorithm/for_each.hpp>

namespace rrgsim::io {

void IOHandler::setup(fs::path work_dir) {
    instance().work_dir = std::move(work_dir);
    fs::remove_all(instance().work_dir / "data");
}

fs::path IOHandler::get_data_dir() {
    auto data_dir = instance().work_dir / "data";
    if (false == fs::exists(data_dir)) {
        fs::create_directories(data_dir);
    }
    return data_dir;
}

fs::path IOHandler::get_dump_dir() {
    auto dump_dir = instance().work_dir / "dump";
    if (false == fs::exists(dump_dir)) {
        fs::create_directories(dump_dir);
    }
    return dump_dir;
}

void IOHandler::fill_in_header(
    std::ofstream& stream,
    std::initializer_list<TagValue> items
) const
{
    ranges::for_each(
        items | ranges::views::drop_last(1),
        [&stream](const TagValue& item) {
            stream << item.tag << ", ";
        }
    );

    ranges::for_each(
        items | ranges::views::take_last(1),
        [&stream](const TagValue& item) {
            stream << item.tag << std::endl;
        }
    );
}
void IOHandler::fill_in_row(
    std::ofstream& stream,
    std::initializer_list<TagValue> items
) const
{
    ranges::for_each(
        items | ranges::views::drop_last(1),
        [&stream](const TagValue& item) {
            stream << fmt::format("{:.{}f}, ", item.value, item.precision);
        }
    );

    ranges::for_each(
        items | ranges::views::take_last(1),
        [&stream](const TagValue& item) {
            stream << fmt::format("{:.{}f}", item.value, item.precision) << std::endl;
        }
    );
}
void IOHandler::append_table(
    std::string tag,
    std::initializer_list<TagValue> items,
    const tl::optional<std::string>& mb_subdir
) const
{
    fs::path target_dir = mb_subdir.map_or_else(
        [&](const std::string& subdir) {
            return get_data_dir() / subdir;
        },
        [&](void) {
            return get_data_dir();
        }
    );

    fs::path path = target_dir / fmt::format("{}.csv", tag);
    std::ofstream stream;

    if (filled.contains(tag)) {
        stream.open(path, std::ios::app);
    }
    else {
        stream.open(path);

        fill_in_header(stream, items);
        filled.insert(tag);
        spdlog::trace("Add data table '{}'", tag);
    }

    fill_in_row(stream, items);
}

} // namespace rrgsim::io