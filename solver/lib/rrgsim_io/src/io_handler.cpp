#include "io_handler.h"

#include <fstream>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <range/v3/view/drop_last.hpp>
#include <range/v3/view/take_last.hpp>
#include <range/v3/algorithm/for_each.hpp>

namespace rrgsim::io {

void IOHandler::setup(std::filesystem::path work_dir) {
    instance().work_dir = std::move(work_dir);
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
            stream << item.value << ", ";
        }
    );

    ranges::for_each(
        items | ranges::views::take_last(1),
        [&stream](const TagValue& item) {
            stream << item.value << std::endl;
        }
    );
}
void IOHandler::append_table(
    std::string_view tag,
    std::initializer_list<TagValue> items
) const
{
    std::filesystem::path data_dir = work_dir / "data";
    std::filesystem::path path = data_dir / fmt::format("{}.csv", tag);
    std::ofstream stream;

    if (filled.contains(tag)) {
        stream.open(path, std::ios::app);
    }
    else {
        std::filesystem::create_directories(data_dir);
        stream.open(path);

        fill_in_header(stream, items);
        filled.insert(tag);
        spdlog::info("Add data table '{}'", tag);
    }

    fill_in_row(stream, items);
}

} // namespace rrgsim::io