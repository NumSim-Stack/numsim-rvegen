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
//   * `.ply` — ASCII only at present; binary support lands when
//     `read_ply` learns binary payloads.
//
// Unknown extensions throw with a clear message rather than guessing.
// (Files with no extension at all are also rejected — the caller
// almost certainly meant something specific.)

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

} // namespace detail

template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_mesh_file(
    std::string const& path) {
  const auto ext = detail::lowercased_extension(path);
  if (ext == ".stl") return read_stl_file<T>(path);
  if (ext == ".ply") return read_ply_file<T>(path);
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
