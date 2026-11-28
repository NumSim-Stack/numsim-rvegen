#pragma once

// The library version, for run summaries and other output that should say
// which rvegen wrote it. Kept equal to project(rvegen VERSION ...) in
// CMakeLists.txt and to the Python package version (checked by a test).

namespace rvegen {

inline constexpr char const* version = "0.1.0";

} // namespace rvegen
