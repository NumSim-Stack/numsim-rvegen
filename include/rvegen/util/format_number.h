#pragma once

// The one place that turns a floating-point value into text for a geometry
// file (#135): .geo scripts, VTK spacing, the raw voxel header.
//
// Shortest decimal form that reads back to the same value (std::to_chars
// without a precision): 0.05 stays "0.05", a value that needs 17 digits
// gets them. It does not depend on the stream's precision, flags or
// locale, so a global locale with a decimal comma cannot corrupt a file.
//
// Display-only output (SVG coordinates, the three.js camera) keeps its own
// short formats: nothing reads those numbers back.

#include <array>
#include <charconv>
#include <concepts>
#include <string>

namespace rvegen {

template <std::floating_point T>
[[nodiscard]] inline std::string format_number(T v) {
  std::array<char, 64> buf{};
  auto const res = std::to_chars(buf.data(), buf.data() + buf.size(), v);
  return std::string(buf.data(), res.ptr);
}

} // namespace rvegen
