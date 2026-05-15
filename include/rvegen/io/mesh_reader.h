#pragma once

// Mesh-file reader with format dispatch by file extension.
//
// `mesh_inclusion_input` (and any other consumer reading a polygon
// mesh from a path) calls `read_mesh_file(path)` and receives a
// `std::vector<gte::Triangle3<T>>` regardless of whether the file is
// STL or PLY. The format is chosen from the path's lowercased
// extension; ASCII vs binary STL is then resolved by the STL reader's
// own header sniff. PLY's `format` line is the source of truth on the
// PLY side.
//
// Supported extensions today:
//   * `.stl` — ASCII or binary; dispatched via `read_stl_file`.
//   * `.ply` — ASCII or binary little/big endian; dispatched via
//     `read_ply_file` (which sniffs the format line in turn).
//
// Unknown extensions throw with a clear message rather than guessing.
// (Files with no extension at all are also rejected — the caller
// almost certainly meant something specific.)
//
// Paths are trimmed of leading/trailing whitespace before dispatch.
// Users feeding paths from sloppy JSON configs (e.g. trailing spaces
// after the closing quote on the previous line) get a sensible error
// pointing at the actual file rather than a confusing "no extension"
// for what is really a path-formatting bug.

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <Mathematics/Triangle.h>

#include "ply_reader.h"
#include "stl_reader.h"

namespace rvegen {

namespace detail {

inline std::string lowercased_extension(std::string const& path) {
  const auto ext = std::filesystem::path{path}.extension().string();
  std::string lower;
  lower.reserve(ext.size());
  for (char c : ext) lower.push_back(static_cast<char>(std::tolower(
      static_cast<unsigned char>(c))));
  return lower;
}

inline std::string trim_path(std::string const& path) {
  // Strip leading/trailing whitespace so a config with a sloppy
  // trailing space doesn't trip the "no extension" check on what is
  // really a path-formatting bug.
  const auto first = path.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = path.find_last_not_of(" \t\r\n");
  return path.substr(first, last - first + 1);
}

} // namespace detail

template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_mesh_file(
    std::string const& path) {
  const auto trimmed = detail::trim_path(path);
  const auto ext = detail::lowercased_extension(trimmed);
  if (ext == ".stl") return read_stl_file<T>(trimmed);
  if (ext == ".ply") return read_ply_file<T>(trimmed);
  if (ext.empty()) {
    throw std::runtime_error{
        "read_mesh_file: '" + path +
        "' has no extension; cannot infer format. Use read_stl_file / "
        "read_ply_file directly if the file lacks an extension."};
  }
  throw std::runtime_error{
      "read_mesh_file: '" + path + "' has unsupported extension '" +
      ext + "' — recognised extensions are .stl and .ply"};
}

} // namespace rvegen
