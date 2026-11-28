#pragma once

// Makes Voro++ throw instead of killing the process (#70).
//
// Voro++'s `voro_fatal_error()` (an inline function in common.hh) prints a
// message and calls `exit()`, which would terminate an embedding host (e.g.
// a Python interpreter) on internal errors / memory limits. This header
// redirects the `exit` token to `rvegen::detail::voro_exit`, which throws
// `std::runtime_error`. It MUST be applied identically to every translation
// unit that sees Voro++'s headers (the static library is force-included with
// this header by CMake; voronoi_backend_voropp.h includes it before
// <voro++.hh>), so the inline `voro_fatal_error` has one definition (no ODR
// violation). Consumer code `#undef exit`s again right after the Voro++
// include; the library build leaves it defined for the whole TU.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace rvegen::detail {
[[noreturn]] inline void voro_exit(int status) {
  throw std::runtime_error{"voro++ fatal error (status " +
                           std::to_string(status) +
                           "); see stderr for the message"};
}
} // namespace rvegen::detail

#define exit(status) ::rvegen::detail::voro_exit(status)
