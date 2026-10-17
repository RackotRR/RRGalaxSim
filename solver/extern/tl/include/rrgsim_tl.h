#pragma once
#include <tl/expected.hpp>
#include <tl/optional.hpp>

#define EXPECTED_CHECK(expected) do \
    if (false == expected.has_value()) { return tl::make_unexpected(std::move(expected.error())); } \
    while(false)
