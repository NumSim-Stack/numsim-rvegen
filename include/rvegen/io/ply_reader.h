#pragma once

// PLY (Stanford Polygon) reader, ASCII subset.
//
// PLY differs from STL in that it stores a deduplicated vertex table
// plus a face element list that references vertices by index. STL,
// by contrast, embeds three vertex copies per triangle. The PLY
// reader rebuilds full `gte::Triangle3<T>` records by looking up
// each face's indices in the parsed vertex table — that way callers
// (notably `mesh_inclusion`) consume the exact same data structure
// as the STL reader.
//
// API surface (phase 1, this header):
//   * `read_ply_ascii(istream)` / `read_ply_ascii_file(path)` —
//     ASCII PLY only. Binary PLY (little- or big-endian) is detected
//     in the header and refused with a clear "binary not yet
//     supported" error, mirroring the way the STL reader handled
//     binary in its first cut.
//   * `read_ply(istream)` / `read_ply_file(path)` — auto-detect
//     entry points that today only accept ASCII. They are exposed so
//     consumers (e.g. mesh_inclusion_input once it grows a `format
//     auto` mode) don't need to be edited when binary PLY support
//     lands.
//
// Subset of PLY this reader handles:
//   * Header lines: `ply`, `format ascii 1.0`, `comment ...`,
//     `element vertex N`, `element face N`, `property <type> <name>`,
//     `property list <count_type> <index_type> <name>`,
//     `end_header`.
//   * Vertex element MUST declare `x`, `y`, `z` properties of any
//     scalar type the reader can parse (float, double, int*, uint*).
//     Extra properties (nx, ny, nz, red, green, blue, alpha,
//     intensity, etc.) are silently skipped per-vertex.
//   * Face element MUST declare a single `vertex_indices` (or
//     `vertex_index`, the older name) list property. Other face
//     properties are rejected — see the format quirks note below.
//   * Faces with >3 indices are fan-triangulated around index 0.
//
// What it does NOT handle:
//   * Binary little/big endian PLY (header parses fine; payload is
//     refused — clear error message). Easy follow-up: same scaffolding
//     as the binary STL reader.
//   * Multi-element files beyond `vertex` + `face` (other elements
//     like `edge`, `material`, `tristrips` are rejected).
//   * Per-face properties beyond the index list. In real PLY files
//     these exist (e.g. `red green blue` per face), but parsing them
//     requires knowing each property's type at field-skipping time,
//     and the value isn't useful for our consumer (a polygon mesh
//     fed into `is_inside`).
//
// Tolerances:
//   Header keywords are lowercase by spec but the reader matches
//   case-insensitively. Extra whitespace, missing trailing newlines,
//   and comment lines anywhere in the header are accepted.

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Mathematics/Triangle.h>
#include <Mathematics/Vector3.h>

namespace rvegen {

namespace detail {

inline std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

// Read a single non-blank, non-comment header line, trimmed.
inline bool next_header_line(std::istream& in, std::string& out) {
  std::string line;
  while (std::getline(in, line)) {
    // Trim leading/trailing whitespace.
    auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos) continue;
    auto last = line.find_last_not_of(" \t\r");
    out = line.substr(first, last - first + 1);
    if (out.empty()) continue;
    if (to_lower(out).rfind("comment", 0) == 0) continue;
    if (to_lower(out).rfind("obj_info", 0) == 0) continue;
    return true;
  }
  return false;
}

enum class ply_format { ascii, binary_little_endian, binary_big_endian };

// One element + property declaration parsed from the header.
struct ply_element {
  std::string name;
  std::size_t count{0};
  // Order of declared scalar properties (name, type) — used to skip
  // unknown properties while reading the matching vertex / etc.
  // For face's `list` property, the entry's type holds the index
  // scalar type, and `is_list` is true.
  struct property {
    std::string name;
    std::string scalar_type;   // "float", "double", "int", "uchar", ...
    bool is_list{false};
    std::string list_count_type;
  };
  std::vector<property> props;
};

// Parse the header up to `end_header`. Reports the format, the
// element table, and the count of bytes consumed (we don't need that
// here, but exposed for symmetry with binary support).
inline std::pair<ply_format, std::vector<ply_element>>
read_ply_header(std::istream& in) {
  std::string first;
  if (!next_header_line(in, first) || to_lower(first) != "ply") {
    throw std::runtime_error{
        "read_ply: missing 'ply' magic on first line"};
  }
  std::string format_line;
  if (!next_header_line(in, format_line) ||
      to_lower(format_line).rfind("format", 0) != 0) {
    throw std::runtime_error{
        "read_ply: expected 'format ...' immediately after 'ply'"};
  }
  std::istringstream fs{format_line};
  std::string ignored, fmt, ver;
  fs >> ignored >> fmt >> ver;
  ply_format format;
  if (to_lower(fmt) == "ascii") {
    format = ply_format::ascii;
  } else if (to_lower(fmt) == "binary_little_endian") {
    format = ply_format::binary_little_endian;
  } else if (to_lower(fmt) == "binary_big_endian") {
    format = ply_format::binary_big_endian;
  } else {
    throw std::runtime_error{
        "read_ply: unknown format '" + fmt + "'"};
  }

  std::vector<ply_element> elems;
  std::string line;
  while (next_header_line(in, line)) {
    const auto lower = to_lower(line);
    if (lower == "end_header") break;
    if (lower.rfind("element", 0) == 0) {
      std::istringstream es{line};
      std::string kw, name;
      std::size_t count;
      es >> kw >> name >> count;
      elems.push_back({name, count, {}});
    } else if (lower.rfind("property", 0) == 0) {
      if (elems.empty()) {
        throw std::runtime_error{
            "read_ply: 'property' before any 'element'"};
      }
      std::istringstream ps{line};
      std::string kw, type_or_list;
      ps >> kw >> type_or_list;
      if (to_lower(type_or_list) == "list") {
        std::string count_type, index_type, name;
        ps >> count_type >> index_type >> name;
        elems.back().props.push_back(
            {name, index_type, true, count_type});
      } else {
        std::string name;
        ps >> name;
        elems.back().props.push_back({name, type_or_list, false, ""});
      }
    } else {
      throw std::runtime_error{
          "read_ply: unknown header line '" + line + "'"};
    }
  }
  return {format, elems};
}

// Read a single ASCII scalar value as double — covers every numeric
// PLY scalar type because we only need the value for x/y/z (cast to
// T at use site) and for the list count / index (cast to size_t).
inline double read_ascii_scalar(std::istream& in) {
  double v;
  if (!(in >> v)) {
    throw std::runtime_error{
        "read_ply_ascii: ran out of input mid-element"};
  }
  return v;
}

// Size in bytes of a PLY scalar type name. Returns 0 for unknown
// types (the caller throws a clearer error with file context).
inline std::size_t ply_scalar_size(std::string const& type_name) {
  const auto n = to_lower(type_name);
  if (n == "char"   || n == "int8"   || n == "uchar" || n == "uint8")  return 1;
  if (n == "short"  || n == "int16"  || n == "ushort"|| n == "uint16") return 2;
  if (n == "int"    || n == "int32"  || n == "uint"  || n == "uint32"
   || n == "float"  || n == "float32")                                 return 4;
  if (n == "double" || n == "float64")                                 return 8;
  return 0;
}

// Read a single binary scalar value as double. `format` is the file's
// declared binary byte order (LE or BE); on a LE host we memcpy
// straight for LE files and byte-reverse for BE; on a BE host (rare)
// the reverse. `std::endian::native` decides at compile time.
inline double read_binary_scalar(std::istream& in, ply_format format,
                                 std::string const& type_name) {
  const auto sz = ply_scalar_size(type_name);
  if (sz == 0) {
    throw std::runtime_error{
        "read_ply_binary: unknown scalar type '" + type_name + "'"};
  }
  std::array<unsigned char, 8> bytes{};
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(sz));
  if (static_cast<std::size_t>(in.gcount()) != sz) {
    throw std::runtime_error{
        "read_ply_binary: short read while parsing scalar of type '" +
        type_name + "'"};
  }

  // Normalise so `bytes` holds the value in HOST byte order. PLY's
  // binary scalars are stored in the format's declared endianness;
  // we byte-reverse iff that doesn't match the host.
  const bool file_is_le = format == ply_format::binary_little_endian;
  const bool host_is_le = std::endian::native == std::endian::little;
  if (file_is_le != host_is_le) {
    std::reverse(bytes.begin(), bytes.begin() + sz);
  }

  const auto n = to_lower(type_name);
  if (n == "char"   || n == "int8")   {
    std::int8_t v;  std::memcpy(&v, bytes.data(), sizeof(v));
    return static_cast<double>(v);
  }
  if (n == "uchar"  || n == "uint8")  {
    std::uint8_t v; std::memcpy(&v, bytes.data(), sizeof(v));
    return static_cast<double>(v);
  }
  if (n == "short"  || n == "int16")  {
    std::int16_t v; std::memcpy(&v, bytes.data(), sizeof(v));
    return static_cast<double>(v);
  }
  if (n == "ushort" || n == "uint16") {
    std::uint16_t v; std::memcpy(&v, bytes.data(), sizeof(v));
    return static_cast<double>(v);
  }
  if (n == "int"    || n == "int32")  {
    std::int32_t v; std::memcpy(&v, bytes.data(), sizeof(v));
    return static_cast<double>(v);
  }
  if (n == "uint"   || n == "uint32") {
    std::uint32_t v; std::memcpy(&v, bytes.data(), sizeof(v));
    return static_cast<double>(v);
  }
  if (n == "float"  || n == "float32") {
    float v; std::memcpy(&v, bytes.data(), sizeof(v));
    return static_cast<double>(v);
  }
  // double / float64.
  double v; std::memcpy(&v, bytes.data(), sizeof(v));
  return v;
}

// Locate the offsets of x/y/z within a vertex element's property
// list. Throws if any of the three is missing — the consumer needs
// them all to build a usable mesh.
inline std::array<std::size_t, 3> vertex_xyz_offsets(
    ply_element const& vertex) {
  std::array<std::size_t, 3> idx{
      std::numeric_limits<std::size_t>::max(),
      std::numeric_limits<std::size_t>::max(),
      std::numeric_limits<std::size_t>::max()};
  for (std::size_t i = 0; i < vertex.props.size(); ++i) {
    auto const& p = vertex.props[i];
    if (p.is_list) continue;
    if      (p.name == "x") idx[0] = i;
    else if (p.name == "y") idx[1] = i;
    else if (p.name == "z") idx[2] = i;
  }
  for (auto v : idx) {
    if (v == std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error{
          "read_ply: vertex element is missing one of x, y, z"};
    }
  }
  return idx;
}

inline std::size_t face_list_property_index(ply_element const& face) {
  for (std::size_t i = 0; i < face.props.size(); ++i) {
    auto const& p = face.props[i];
    if (!p.is_list) continue;
    if (p.name == "vertex_indices" || p.name == "vertex_index") return i;
  }
  throw std::runtime_error{
      "read_ply: face element is missing 'vertex_indices' list property"};
}

} // namespace detail

// ASCII PLY reader. Throws std::runtime_error on any parse mismatch,
// missing required property, or non-ASCII format declaration.
//
// Polygon faces (4+ indices) are fan-triangulated around the first
// vertex of the face. This matches what most viewers do and is the
// only sensible default for a polygon-soup consumer like
// `mesh_inclusion`.
template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_ply_ascii(std::istream& in) {
  auto [format, elems] = detail::read_ply_header(in);
  if (format != detail::ply_format::ascii) {
    throw std::runtime_error{
        "read_ply_ascii: file declares a binary PLY format. Use "
        "read_ply_binary (or the auto-detect read_ply / read_ply_file "
        "entry points) for binary PLY input."};
  }

  // Locate the vertex and face elements. Allow the file to declare
  // them in either order, though the spec recommends vertex first.
  detail::ply_element const* vertex = nullptr;
  detail::ply_element const* face   = nullptr;
  for (auto const& e : elems) {
    if (e.name == "vertex") vertex = &e;
    else if (e.name == "face") face = &e;
    else throw std::runtime_error{
        "read_ply_ascii: unsupported element '" + e.name + "'"};
  }
  if (!vertex) throw std::runtime_error{
      "read_ply_ascii: no 'vertex' element declared"};
  if (!face)   throw std::runtime_error{
      "read_ply_ascii: no 'face' element declared"};

  const auto xyz = detail::vertex_xyz_offsets(*vertex);
  const auto face_list_prop = detail::face_list_property_index(*face);

  // Read the vertex table, one row per vertex. Each row may carry
  // extra properties (normals, colours, etc.) — we read them as
  // doubles and discard.
  std::vector<std::array<T, 3>> verts;
  verts.reserve(vertex->count);
  for (std::size_t v = 0; v < vertex->count; ++v) {
    std::array<T, 3> p{};
    for (std::size_t i = 0; i < vertex->props.size(); ++i) {
      auto const& prop = vertex->props[i];
      if (prop.is_list) {
        // Read count, then skip that many scalars.
        const auto n = static_cast<std::size_t>(detail::read_ascii_scalar(in));
        for (std::size_t k = 0; k < n; ++k) (void)detail::read_ascii_scalar(in);
        continue;
      }
      const auto val = detail::read_ascii_scalar(in);
      if      (i == xyz[0]) p[0] = static_cast<T>(val);
      else if (i == xyz[1]) p[1] = static_cast<T>(val);
      else if (i == xyz[2]) p[2] = static_cast<T>(val);
      // else: silently drop the per-vertex extra property.
    }
    verts.push_back(p);
  }

  // Read the face table. Each face is a list of vertex indices, of
  // length given by the leading count scalar (so a triangle face line
  // looks like `3 0 1 2`). Fan-triangulate any face with >3 indices.
  // Faces in the file may also declare non-list properties after the
  // list (rare in practice but legal); we read and discard those.
  std::vector<gte::Triangle3<T>> triangles;
  triangles.reserve(face->count);
  for (std::size_t f = 0; f < face->count; ++f) {
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < face->props.size(); ++i) {
      auto const& prop = face->props[i];
      if (i == face_list_prop) {
        const auto n = static_cast<std::size_t>(detail::read_ascii_scalar(in));
        indices.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
          indices[k] = static_cast<std::size_t>(detail::read_ascii_scalar(in));
        }
      } else if (prop.is_list) {
        // Non-vertex-index list — skip it generically.
        const auto n = static_cast<std::size_t>(detail::read_ascii_scalar(in));
        for (std::size_t k = 0; k < n; ++k) (void)detail::read_ascii_scalar(in);
      } else {
        (void)detail::read_ascii_scalar(in);
      }
    }
    if (indices.size() < 3) {
      throw std::runtime_error{
          "read_ply_ascii: face with fewer than 3 vertices"};
    }
    for (auto i : indices) {
      if (i >= verts.size()) {
        throw std::runtime_error{
            "read_ply_ascii: face references out-of-range vertex index"};
      }
    }
    // Fan-triangulate around indices[0].
    for (std::size_t k = 1; k + 1 < indices.size(); ++k) {
      gte::Triangle3<T> tri;
      auto const& a = verts[indices[0]];
      auto const& b = verts[indices[k]];
      auto const& c = verts[indices[k + 1]];
      tri.v[0] = {a[0], a[1], a[2]};
      tri.v[1] = {b[0], b[1], b[2]};
      tri.v[2] = {c[0], c[1], c[2]};
      triangles.push_back(tri);
    }
  }
  return triangles;
}

template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_ply_ascii_file(
    std::string const& path) {
  std::ifstream in{path, std::ios::binary};
  if (!in) {
    throw std::runtime_error{"read_ply_ascii_file: cannot open '" + path + "'"};
  }
  return read_ply_ascii<T>(in);
}

// Binary PLY reader, little- or big-endian. The header parser is
// shared with the ASCII path — only the payload decoding differs.
//
// Stream MUST be opened in binary mode (`std::ios::binary`) when the
// source is a file; otherwise text-mode line-ending translation can
// corrupt the payload mid-element. `read_ply_binary_file` does this
// already; in-memory `stringstream` is fine either way.
template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_ply_binary(std::istream& in) {
  auto [format, elems] = detail::read_ply_header(in);
  if (format == detail::ply_format::ascii) {
    throw std::runtime_error{
        "read_ply_binary: file declares ASCII format. Use read_ply_ascii "
        "or the auto-detect read_ply entry point."};
  }

  detail::ply_element const* vertex = nullptr;
  detail::ply_element const* face   = nullptr;
  for (auto const& e : elems) {
    if (e.name == "vertex") vertex = &e;
    else if (e.name == "face") face = &e;
    else throw std::runtime_error{
        "read_ply_binary: unsupported element '" + e.name + "'"};
  }
  if (!vertex) throw std::runtime_error{
      "read_ply_binary: no 'vertex' element declared"};
  if (!face)   throw std::runtime_error{
      "read_ply_binary: no 'face' element declared"};

  const auto xyz = detail::vertex_xyz_offsets(*vertex);
  const auto face_list_prop = detail::face_list_property_index(*face);

  std::vector<std::array<T, 3>> verts;
  verts.reserve(vertex->count);
  for (std::size_t v = 0; v < vertex->count; ++v) {
    std::array<T, 3> p{};
    for (std::size_t i = 0; i < vertex->props.size(); ++i) {
      auto const& prop = vertex->props[i];
      if (prop.is_list) {
        // Vertex-level list property — rare. Read count via the
        // declared count type, then skip that many values of the
        // declared index/scalar type.
        const auto n = static_cast<std::size_t>(
            detail::read_binary_scalar(in, format, prop.list_count_type));
        for (std::size_t k = 0; k < n; ++k) {
          (void)detail::read_binary_scalar(in, format, prop.scalar_type);
        }
        continue;
      }
      const auto val = detail::read_binary_scalar(in, format, prop.scalar_type);
      if      (i == xyz[0]) p[0] = static_cast<T>(val);
      else if (i == xyz[1]) p[1] = static_cast<T>(val);
      else if (i == xyz[2]) p[2] = static_cast<T>(val);
      // else: silently drop the per-vertex extra property.
    }
    verts.push_back(p);
  }

  std::vector<gte::Triangle3<T>> triangles;
  triangles.reserve(face->count);
  for (std::size_t f = 0; f < face->count; ++f) {
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < face->props.size(); ++i) {
      auto const& prop = face->props[i];
      if (i == face_list_prop) {
        const auto n = static_cast<std::size_t>(
            detail::read_binary_scalar(in, format, prop.list_count_type));
        indices.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
          indices[k] = static_cast<std::size_t>(
              detail::read_binary_scalar(in, format, prop.scalar_type));
        }
      } else if (prop.is_list) {
        const auto n = static_cast<std::size_t>(
            detail::read_binary_scalar(in, format, prop.list_count_type));
        for (std::size_t k = 0; k < n; ++k) {
          (void)detail::read_binary_scalar(in, format, prop.scalar_type);
        }
      } else {
        (void)detail::read_binary_scalar(in, format, prop.scalar_type);
      }
    }
    if (indices.size() < 3) {
      throw std::runtime_error{
          "read_ply_binary: face with fewer than 3 vertices"};
    }
    for (auto i : indices) {
      if (i >= verts.size()) {
        throw std::runtime_error{
            "read_ply_binary: face references out-of-range vertex index"};
      }
    }
    for (std::size_t k = 1; k + 1 < indices.size(); ++k) {
      gte::Triangle3<T> tri;
      auto const& a = verts[indices[0]];
      auto const& b = verts[indices[k]];
      auto const& c = verts[indices[k + 1]];
      tri.v[0] = {a[0], a[1], a[2]};
      tri.v[1] = {b[0], b[1], b[2]};
      tri.v[2] = {c[0], c[1], c[2]};
      triangles.push_back(tri);
    }
  }
  return triangles;
}

template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_ply_binary_file(
    std::string const& path) {
  std::ifstream in{path, std::ios::binary};
  if (!in) {
    throw std::runtime_error{
        "read_ply_binary_file: cannot open '" + path + "'"};
  }
  return read_ply_binary<T>(in);
}

// Auto-detect entry point. Peeks at the `format` line in the header,
// then rewinds and dispatches to the right reader. Requires a
// seekable stream — `std::stringstream` and `std::ifstream` both
// qualify; pipes do not.
template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_ply(std::istream& in) {
  const auto start = in.tellg();
  if (start < 0) {
    // Non-seekable — fall back to ASCII, which will throw with a
    // clear error if the file turns out to be binary.
    return read_ply_ascii<T>(in);
  }
  // Read just enough of the header to find the format line. We
  // can't reuse `read_ply_header` here because it consumes element
  // declarations too; we need to rewind before that point.
  std::string line;
  detail::ply_format format = detail::ply_format::ascii;
  bool format_found = false;
  while (std::getline(in, line)) {
    auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos) continue;
    auto last = line.find_last_not_of(" \t\r");
    line = line.substr(first, last - first + 1);
    const auto lower = detail::to_lower(line);
    if (lower.rfind("comment", 0) == 0) continue;
    if (lower.rfind("obj_info", 0) == 0) continue;
    if (lower == "ply") continue;
    if (lower.rfind("format", 0) == 0) {
      std::istringstream fs{line};
      std::string ignored, fmt;
      fs >> ignored >> fmt;
      const auto f = detail::to_lower(fmt);
      if      (f == "ascii")                 format = detail::ply_format::ascii;
      else if (f == "binary_little_endian")  format = detail::ply_format::binary_little_endian;
      else if (f == "binary_big_endian")     format = detail::ply_format::binary_big_endian;
      else throw std::runtime_error{"read_ply: unknown format '" + fmt + "'"};
      format_found = true;
      break;
    }
    break;   // some other header line surfaced before `format` — bail
  }
  in.clear();
  in.seekg(start);
  if (!format_found) {
    // Let read_ply_ascii's own header parser surface the precise
    // "missing magic" / "missing format" error.
    return read_ply_ascii<T>(in);
  }
  if (format == detail::ply_format::ascii) return read_ply_ascii<T>(in);
  return read_ply_binary<T>(in);
}

template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_ply_file(
    std::string const& path) {
  std::ifstream in{path, std::ios::binary};
  if (!in) {
    throw std::runtime_error{"read_ply_file: cannot open '" + path + "'"};
  }
  return read_ply<T>(in);
}

} // namespace rvegen
