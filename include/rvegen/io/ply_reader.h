#pragma once

// PLY (Stanford Polygon) reader — ASCII and binary little/big endian.
//
// PLY differs from STL in that it stores a deduplicated vertex table
// plus a face element list that references vertices by index. STL,
// by contrast, embeds three vertex copies per triangle. The PLY
// reader rebuilds full `gte::Triangle3<T>` records by looking up
// each face's indices in the parsed vertex table — that way callers
// (notably `mesh_inclusion`) consume the exact same data structure
// as the STL reader.
//
// API surface:
//   * `read_ply_ascii(istream)` / `read_ply_ascii_file(path)` —
//     ASCII PLY only; a binary `format` line is refused with a clear
//     error pointing at the binary / auto-detect entry points.
//   * `read_ply_binary(istream)` / `read_ply_binary_file(path)` —
//     binary little- or big-endian PLY; an ASCII `format` line is
//     refused.
//   * `read_ply(istream)` / `read_ply_file(path)` — auto-detect:
//     sniff the `format` line, rewind, dispatch. Most callers want
//     this.
//
// Subset of PLY this reader handles:
//   * Header lines: `ply`, `format <ascii|binary_*> 1.0`,
//     `comment ...`, `obj_info ...`, `element vertex N`,
//     `element face N`, `property <type> <name>`,
//     `property list <count_type> <index_type> <name>`, `end_header`.
//     Unknown private header lines are skipped.
//   * Vertex element MUST declare `x`, `y`, `z` properties of any
//     scalar type the reader can parse (float, double, int*, uint*).
//     Extra properties (nx, ny, nz, red, green, blue, alpha,
//     intensity, etc.) are read and silently discarded per-vertex.
//   * Face element MUST declare a `vertex_indices` (or `vertex_index`,
//     the older name) list property. Other face properties — scalars
//     or lists, before or after the index list — are read and
//     discarded.
//   * `vertex` and `face` may be declared in either order; the
//     payload is read in declaration order and faces are resolved
//     against the vertex table once both are read.
//   * Faces with >3 indices are fan-triangulated around index 0.
//
// Validation (hostile / malformed input throws std::runtime_error):
//   * Element counts are capped (`max_ply_element_count`) and, on a
//     seekable stream, checked against the remaining payload size
//     BEFORE any reservation — every row needs at least one byte per
//     property (ASCII) or the sum of its fixed scalar sizes (binary).
//     Non-seekable streams reserve at most `non_seekable_reserve_cap`.
//   * List counts must be finite, non-negative integers no larger
//     than `max_ply_list_count`; long lists are also checked against
//     the remaining payload. Index storage grows only as data is
//     actually read, never from a declared count.
//   * The fan-triangulated output is capped at
//     `ply_read_limits::max_triangles` (default 10M; every stream entry
//     point takes an optional `ply_read_limits`) and, on seekable
//     streams, at one triangle per remaining index
//     scalar's worth of bytes. Triangles are emitted directly when the
//     vertex table precedes the faces; otherwise they are buffered as
//     uint32 index triples until the vertices have been read.
//   * Vertex indices must be finite, non-negative, integral and less
//     than the vertex count. No unchecked double → size_t casts.
//   * ASCII scalars must parse as a whole token ("1,5" is an error).
//
// What it does NOT handle:
//   * Multi-element files beyond `vertex` + `face` (other elements
//     like `edge`, `material`, `tristrips` are rejected).
//
// Tolerances:
//   Header keywords are lowercase by spec but the reader matches
//   case-insensitively (ASCII-only fold, locale-independent). A
//   leading UTF-8 BOM, extra whitespace, missing trailing newlines,
//   and comment lines anywhere in the header are accepted.

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Mathematics/Triangle.h>
#include <Mathematics/Vector3.h>

#include "stream_position.h"

namespace rvegen {

// Caller-tunable output limits for the PLY readers.
struct ply_read_limits {
  // Cap on the fan-triangulated output (default ~360 MB as
  // Triangle3<double>). Lower than the 1e8 element cap because fan
  // triangulation of long polygon lists amplifies a small file into
  // many triangles (tens of bytes out per byte in), and a
  // mesh_inclusion's is_inside is O(triangles) per query anyway. Raise
  // it for genuinely huge meshes: `read_ply_file` / `read_mesh_file`
  // take a `ply_read_limits`, and `mesh_inclusion_input` exposes it as
  // the optional `max_triangles` JSON key.
  std::uint64_t max_triangles = 5'000'000;
};

namespace detail {

// Absolute sanity caps, independent of the stream size. 100M rows
// matches the binary-STL cap; a single polygon with more than ~1M
// corners is certainly malformed.
inline constexpr std::uint64_t max_ply_element_count = 100'000'000;
// Maximum number of `element` declarations in one header. Real files
// have a handful; the cap keeps a hostile header from growing the
// element table without bound.
inline constexpr std::size_t max_ply_element_decls = 1024;
inline constexpr std::uint64_t max_ply_list_count = 1u << 20;
inline constexpr std::uint64_t max_ply_triangle_count =
    ply_read_limits{}.max_triangles;
// Upper bound on an up-front reserve() when the stream size is unknown.
inline constexpr std::size_t non_seekable_reserve_cap = 1u << 20;

// ASCII-only, locale-independent fold (see stream_position.h).
inline std::string to_lower(std::string s) { return ascii_to_lower(std::move(s)); }

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

// Map a PLY `format` keyword to the enum, throwing on unknown values.
// Shared between `read_ply_header` (full parse) and the auto-detect
// entry point so the format-name table lives in exactly one place.
inline ply_format parse_format_keyword(std::string const& fmt) {
  const auto f = to_lower(fmt);
  if (f == "ascii")                return ply_format::ascii;
  if (f == "binary_little_endian") return ply_format::binary_little_endian;
  if (f == "binary_big_endian")    return ply_format::binary_big_endian;
  throw std::runtime_error{"read_ply: unknown format '" + fmt + "'"};
}

// Validate a count (element row count or list length) that arrived
// as a double: finite, non-negative, integral, within `cap`.
inline std::size_t checked_count(double v, std::uint64_t cap,
                                 std::string const& what) {
  if (!std::isfinite(v) || v < 0.0 || v != std::floor(v)) {
    throw std::runtime_error{
        "read_ply: " + what + " must be a non-negative integer, got " +
        std::to_string(v)};
  }
  if (v > static_cast<double>(cap)) {
    throw std::runtime_error{
        "read_ply: " + what + " " + std::to_string(v) +
        " exceeds sanity limit (" + std::to_string(cap) +
        ") — probably a malformed file"};
  }
  return static_cast<std::size_t>(v);
}

// Parse the header up to `end_header`. Reports the format and the
// element table.
inline std::pair<ply_format, std::vector<ply_element>>
read_ply_header(std::istream& in) {
  skip_utf8_bom(in);
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
  const ply_format format = parse_format_keyword(fmt);

  std::vector<ply_element> elems;
  std::unordered_set<std::string> element_names;
  std::string line;
  bool saw_end_header = false;
  while (next_header_line(in, line)) {
    const auto lower = to_lower(line);
    if (lower == "end_header") { saw_end_header = true; break; }
    if (lower.rfind("element", 0) == 0) {
      std::istringstream es{line};
      std::string kw, name, count_token;
      double count_value = 0.0;
      if (!(es >> kw >> name >> count_token) ||
          !parse_real(count_token, count_value)) {
        throw std::runtime_error{
            "read_ply: malformed element line '" + line + "'"};
      }
      const auto count = checked_count(
          count_value, max_ply_element_count,
          "element '" + name + "' count");
      if (elems.size() >= max_ply_element_decls) {
        throw std::runtime_error{
            "read_ply: header declares more than " +
            std::to_string(max_ply_element_decls) + " elements"};
      }
      if (!element_names.insert(name).second) {
        throw std::runtime_error{
            "read_ply: element '" + name + "' declared twice"};
      }
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
        if (!(ps >> count_type >> index_type >> name)) {
          throw std::runtime_error{
              "read_ply: malformed list property line '" + line + "'"};
        }
        elems.back().props.push_back(
            {name, index_type, true, count_type});
      } else {
        std::string name;
        if (!(ps >> name)) {
          throw std::runtime_error{
              "read_ply: malformed property line '" + line + "'"};
        }
        elems.back().props.push_back({name, type_or_list, false, ""});
      }
    } else {
      // Unknown header line. The PLY spec allows producers to emit
      // private metadata lines (texturefile, materialfile, ply::group,
      // etc.); rejecting them would refuse otherwise-valid files from
      // tools like meshlab. Skip silently. The header structure we
      // care about — element + property declarations — is still
      // enforced; this only relaxes the "what other keywords may
      // appear" check.
    }
  }
  if (!saw_end_header) {
    throw std::runtime_error{"read_ply: header has no 'end_header' line"};
  }
  return {format, elems};
}

// Read a single ASCII scalar value as double — covers every numeric
// PLY scalar type because we only need the value for x/y/z (cast to
// T at use site) and for the list count / index (validated before
// any conversion to size_t). The whole token must be a number.
inline double read_ascii_scalar(std::istream& in) {
  std::string token;
  if (!(in >> token)) {
    throw std::runtime_error{
        "read_ply_ascii: ran out of input mid-element" +
        position_marker(in)};
  }
  double v = 0.0;
  if (!parse_real(token, v)) {
    throw std::runtime_error{
        "read_ply_ascii: malformed number '" + token + "'" +
        position_marker(in)};
  }
  return v;
}

// PLY scalar type, parsed once from the header. The binary reader's
// inner loop branches on this enum, not on a `to_lower`-ed string,
// so a 100k-vertex mesh doesn't allocate 300k+ temporary strings.
enum class scalar_kind {
  i8, u8, i16, u16, i32, u32, f32, f64
};

inline scalar_kind parse_scalar_kind(std::string const& type_name) {
  const auto n = to_lower(type_name);
  if (n == "char"   || n == "int8")    return scalar_kind::i8;
  if (n == "uchar"  || n == "uint8")   return scalar_kind::u8;
  if (n == "short"  || n == "int16")   return scalar_kind::i16;
  if (n == "ushort" || n == "uint16")  return scalar_kind::u16;
  if (n == "int"    || n == "int32")   return scalar_kind::i32;
  if (n == "uint"   || n == "uint32")  return scalar_kind::u32;
  if (n == "float"  || n == "float32") return scalar_kind::f32;
  if (n == "double" || n == "float64") return scalar_kind::f64;
  throw std::runtime_error{
      "read_ply: unknown scalar type '" + type_name + "'"};
}

inline std::size_t scalar_kind_size(scalar_kind k) noexcept {
  switch (k) {
    case scalar_kind::i8:  case scalar_kind::u8:                return 1;
    case scalar_kind::i16: case scalar_kind::u16:               return 2;
    case scalar_kind::i32: case scalar_kind::u32:
    case scalar_kind::f32:                                      return 4;
    case scalar_kind::f64:                                      return 8;
  }
  return 0;
}

// Read a single binary scalar value as double. `format` is the file's
// declared binary byte order (LE or BE); on a LE host we memcpy
// straight for LE files and byte-reverse for BE; on a BE host (rare)
// the reverse. `std::endian::native` decides at compile time.
inline double read_binary_scalar(std::istream& in, ply_format format,
                                 scalar_kind kind) {
  const auto sz = scalar_kind_size(kind);
  std::array<unsigned char, 8> bytes{};
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(sz));
  if (static_cast<std::size_t>(in.gcount()) != sz) {
    throw std::runtime_error{
        "read_ply_binary: short read while parsing scalar" +
        position_marker(in)};
  }

  // Normalise so `bytes` holds the value in HOST byte order. PLY's
  // binary scalars are stored in the format's declared endianness;
  // we byte-reverse iff that doesn't match the host.
  const bool file_is_le = format == ply_format::binary_little_endian;
  const bool host_is_le = std::endian::native == std::endian::little;
  if (file_is_le != host_is_le) {
    std::reverse(bytes.begin(), bytes.begin() + sz);
  }

  switch (kind) {
    case scalar_kind::i8:  { std::int8_t   v; std::memcpy(&v, bytes.data(), sizeof(v)); return static_cast<double>(v); }
    case scalar_kind::u8:  { std::uint8_t  v; std::memcpy(&v, bytes.data(), sizeof(v)); return static_cast<double>(v); }
    case scalar_kind::i16: { std::int16_t  v; std::memcpy(&v, bytes.data(), sizeof(v)); return static_cast<double>(v); }
    case scalar_kind::u16: { std::uint16_t v; std::memcpy(&v, bytes.data(), sizeof(v)); return static_cast<double>(v); }
    case scalar_kind::i32: { std::int32_t  v; std::memcpy(&v, bytes.data(), sizeof(v)); return static_cast<double>(v); }
    case scalar_kind::u32: { std::uint32_t v; std::memcpy(&v, bytes.data(), sizeof(v)); return static_cast<double>(v); }
    case scalar_kind::f32: { float          v; std::memcpy(&v, bytes.data(), sizeof(v)); return static_cast<double>(v); }
    case scalar_kind::f64: { double         v; std::memcpy(&v, bytes.data(), sizeof(v)); return v; }
  }
  return 0.0;   // unreachable; switch is exhaustive over the enum.
}

// Precomputed-kind view of an element's properties. Built once
// per element so the inner read loop branches on the enum value
// without ever touching the property's `std::string scalar_type`.
struct typed_property {
  scalar_kind kind;
  bool is_list;
  scalar_kind list_count_kind;   // unused unless is_list == true
};

inline std::vector<typed_property>
make_typed_props(ply_element const& e) {
  std::vector<typed_property> out;
  out.reserve(e.props.size());
  for (auto const& p : e.props) {
    if (p.is_list) {
      out.push_back({parse_scalar_kind(p.scalar_type), true,
                     parse_scalar_kind(p.list_count_type)});
    } else {
      out.push_back({parse_scalar_kind(p.scalar_type), false,
                     scalar_kind::u8 /*unused*/});
    }
  }
  return out;
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

// Validate a vertex index that arrived as a double before it is ever
// converted: finite, non-negative, integral, < n_verts.
inline std::size_t checked_vertex_index(double v, std::size_t n_verts,
                                        std::size_t face, char const* who) {
  if (!std::isfinite(v) || v < 0.0 || v != std::floor(v) ||
      v >= static_cast<double>(n_verts)) {
    throw std::runtime_error{
        std::string{who} + ": face " + std::to_string(face) +
        " references invalid vertex index " + std::to_string(v) +
        " (vertex count " + std::to_string(n_verts) + ")"};
  }
  return static_cast<std::size_t>(v);
}

// Shared payload reader for ASCII and binary PLY. `read_scalar(kind)`
// returns the next scalar as double; `min_bytes(kind)` is the fewest
// bytes a scalar of that kind can occupy in the payload (1 for ASCII,
// the fixed width for binary). Elements are read in header order, so
// `face` may precede `vertex`; triangles are assembled at the end.
template <typename T, typename ReadScalar, typename MinBytes>
std::vector<gte::Triangle3<T>> read_ply_payload(
    std::istream& in, std::vector<ply_element> const& elems,
    char const* who, ReadScalar&& read_scalar, MinBytes&& min_bytes,
    ply_read_limits const& limits) {
  ply_element const* vertex = nullptr;
  ply_element const* face   = nullptr;
  for (auto const& e : elems) {
    if (e.name == "vertex") vertex = &e;
    else if (e.name == "face") face = &e;
    else throw std::runtime_error{
        std::string{who} + ": unsupported element '" + e.name + "'"};
  }
  if (!vertex) throw std::runtime_error{
      std::string{who} + ": no 'vertex' element declared"};
  if (!face)   throw std::runtime_error{
      std::string{who} + ": no 'face' element declared"};

  const auto xyz = vertex_xyz_offsets(*vertex);
  const auto face_list_prop = face_list_property_index(*face);

  // Pre-resolve each property's scalar kind into a small enum so the
  // inner loops never touch strings.
  const auto vtyped = make_typed_props(*vertex);
  const auto ftyped = make_typed_props(*face);

  auto min_row_bytes = [&](std::vector<typed_property> const& props) {
    std::uint64_t b = 0;
    for (auto const& p : props) {
      b += min_bytes(p.is_list ? p.list_count_kind : p.kind);
    }
    return b;
  };

  // Bound header counts by what the stream can actually hold before
  // reserving anything. Counts are already capped at
  // max_ply_element_count, so the products cannot overflow.
  const auto remaining = remaining_bytes(in);
  std::optional<std::uint64_t> end_offset;
  if (remaining) {
    const std::uint64_t need =
        vertex->count * min_row_bytes(vtyped) +
        face->count * min_row_bytes(ftyped);
    if (need > *remaining) {
      throw std::runtime_error{
          std::string{who} + ": header declares " +
          std::to_string(vertex->count) + " vertices and " +
          std::to_string(face->count) + " faces (at least " +
          std::to_string(need) + " bytes) but only " +
          std::to_string(*remaining) + " bytes of payload remain"};
    }
    end_offset = static_cast<std::uint64_t>(in.tellg()) + *remaining;
  }
  auto reserve_hint = [&](std::size_t n) {
    return remaining ? n : std::min(n, non_seekable_reserve_cap);
  };

  // Read and validate a list count. Long lists are additionally
  // checked against the bytes left in the stream (cheap tellg only
  // when the list is suspiciously long).
  auto read_list_count = [&](typed_property const& prop,
                             char const* what) -> std::size_t {
    const auto n = checked_count(read_scalar(prop.list_count_kind),
                                 max_ply_list_count, what);
    if (n > 256 && end_offset) {
      const auto pos = in.tellg();
      const std::uint64_t left =
          pos >= 0 ? *end_offset - static_cast<std::uint64_t>(pos) : 0;
      if (static_cast<std::uint64_t>(n) * min_bytes(prop.kind) > left) {
        throw std::runtime_error{
            std::string{who} + ": " + what + " " + std::to_string(n) +
            " exceeds the remaining " + std::to_string(left) +
            " bytes of input" + position_marker(in)};
      }
    }
    return n;
  };

  std::vector<std::array<T, 3>> verts;
  std::vector<gte::Triangle3<T>> triangles;
  // Only used when `face` precedes `vertex`: fan triangles are held as
  // compact uint32 index triples (vertex count <= max_ply_element_count
  // < 2^32) until the vertex table has been read.
  std::vector<std::array<std::uint32_t, 3>> tri_idx;
  std::vector<std::uint32_t> indices;
  bool verts_read = false;

  auto emit = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
    gte::Triangle3<T> tri;
    std::uint32_t const idx[3] = {a, b, c};
    for (int k = 0; k < 3; ++k) {
      auto const& p = verts[idx[k]];
      tri.v[k] = {p[0], p[1], p[2]};
    }
    triangles.push_back(tri);
  };

  // Grow `vec` geometrically but never beyond `limit` elements, so a
  // hostile file that trips the triangle limit cannot first force a
  // doubling past it.
  auto grow_for = [](auto& vec, std::uint64_t extra, std::uint64_t limit) {
    const std::uint64_t want = vec.size() + extra;
    if (want <= vec.capacity()) return;
    const std::uint64_t doubled = std::max<std::uint64_t>(
        {want, 2 * static_cast<std::uint64_t>(vec.capacity()), 1024});
    vec.reserve(static_cast<std::size_t>(std::min(doubled, limit)));
  };

  for (auto const& e : elems) {
    if (&e == vertex) {
      verts_read = true;
      verts.reserve(reserve_hint(vertex->count));
      for (std::size_t v = 0; v < vertex->count; ++v) {
        std::array<T, 3> p{};
        for (std::size_t i = 0; i < vtyped.size(); ++i) {
          auto const& prop = vtyped[i];
          if (prop.is_list) {
            // Vertex-level list property — rare. Skip its values.
            const auto n = read_list_count(prop, "vertex list count");
            for (std::size_t k = 0; k < n; ++k) (void)read_scalar(prop.kind);
            continue;
          }
          const auto val = read_scalar(prop.kind);
          if      (i == xyz[0]) p[0] = static_cast<T>(val);
          else if (i == xyz[1]) p[1] = static_cast<T>(val);
          else if (i == xyz[2]) p[2] = static_cast<T>(val);
          // else: silently drop the per-vertex extra property.
        }
        verts.push_back(p);
      }
    } else {
      // Faces: indices are validated against the header's vertex
      // count, which is exact whether or not the vertex table has
      // been read yet. Extra face properties are read and discarded.
      //
      // Fan triangulation amplifies memory (a 255-gon of 1-byte
      // indices becomes 253 triangles of 72 bytes each), so the total
      // triangle count is bounded by `max_ply_triangle_count` and, on
      // seekable streams, by the payload left: every fan triangle
      // consumes at least one index scalar.
      std::uint64_t max_tris = limits.max_triangles;
      if (end_offset) {
        const auto pos = in.tellg();
        const std::uint64_t left =
            pos >= 0 ? *end_offset - static_cast<std::uint64_t>(pos) : 0;
        max_tris = std::min<std::uint64_t>(
            max_tris, left / min_bytes(ftyped[face_list_prop].kind));
      }
      const auto initial = static_cast<std::size_t>(
          std::min<std::uint64_t>(reserve_hint(face->count), max_tris));
      if (verts_read) triangles.reserve(initial);
      else            tri_idx.reserve(initial);
      std::uint64_t n_tris = 0;

      for (std::size_t f = 0; f < face->count; ++f) {
        indices.clear();
        for (std::size_t i = 0; i < ftyped.size(); ++i) {
          auto const& prop = ftyped[i];
          if (i == face_list_prop) {
            const auto n = read_list_count(prop, "face index count");
            for (std::size_t k = 0; k < n; ++k) {
              indices.push_back(static_cast<std::uint32_t>(
                  checked_vertex_index(read_scalar(prop.kind),
                                       vertex->count, f, who)));
            }
          } else if (prop.is_list) {
            const auto n = read_list_count(prop, "face list count");
            for (std::size_t k = 0; k < n; ++k) (void)read_scalar(prop.kind);
          } else {
            (void)read_scalar(prop.kind);
          }
        }
        if (indices.size() < 3) {
          throw std::runtime_error{
              std::string{who} + ": face " + std::to_string(f) +
              " has fewer than 3 vertices"};
        }
        const std::uint64_t add = indices.size() - 2;
        if (n_tris + add > max_tris) {
          throw std::runtime_error{
              std::string{who} + ": face " + std::to_string(f) +
              " pushes the fan-triangulated mesh past " +
              std::to_string(max_tris) + " triangles (limit " +
              std::to_string(limits.max_triangles) +
              ", or fewer when bounded by the remaining payload) — "
              "probably a malformed file"};
        }
        n_tris += add;
        // Fan-triangulate around indices[0]. Emit directly when the
        // vertex table is already known (the usual vertex-first order).
        if (verts_read) {
          grow_for(triangles, add, max_tris);
          for (std::size_t k = 1; k + 1 < indices.size(); ++k) {
            emit(indices[0], indices[k], indices[k + 1]);
          }
        } else {
          grow_for(tri_idx, add, max_tris);
          for (std::size_t k = 1; k + 1 < indices.size(); ++k) {
            tri_idx.push_back({indices[0], indices[k], indices[k + 1]});
          }
        }
      }
    }
  }

  // Face-before-vertex files: resolve the buffered index triples now.
  if (!tri_idx.empty()) {
    triangles.reserve(tri_idx.size());
    for (auto const& t : tri_idx) emit(t[0], t[1], t[2]);
    std::vector<std::array<std::uint32_t, 3>>{}.swap(tri_idx);
  }
  return triangles;
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
[[nodiscard]] std::vector<gte::Triangle3<T>> read_ply_ascii(
    std::istream& in, ply_read_limits const& limits = {}) {
  auto [format, elems] = detail::read_ply_header(in);
  if (format != detail::ply_format::ascii) {
    throw std::runtime_error{
        "read_ply_ascii: file declares a binary PLY format. Use "
        "read_ply_binary (or the auto-detect read_ply / read_ply_file "
        "entry points) for binary PLY input."};
  }
  return detail::read_ply_payload<T>(
      in, elems, "read_ply_ascii",
      [&](detail::scalar_kind) { return detail::read_ascii_scalar(in); },
      [](detail::scalar_kind) -> std::uint64_t { return 1; }, limits);
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

// Binary PLY reader, little- or big-endian. The header parser and
// payload walker are shared with the ASCII path — only the scalar
// decoding differs.
//
// Stream MUST be opened in binary mode (`std::ios::binary`) when the
// source is a file; otherwise text-mode line-ending translation can
// corrupt the payload mid-element. `read_ply_binary_file` does this
// already; in-memory `stringstream` is fine either way.
template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_ply_binary(
    std::istream& in, ply_read_limits const& limits = {}) {
  auto [format, elems] = detail::read_ply_header(in);
  if (format == detail::ply_format::ascii) {
    throw std::runtime_error{
        "read_ply_binary: file declares ASCII format. Use read_ply_ascii "
        "or the auto-detect read_ply entry point."};
  }
  const auto fmt = format;
  return detail::read_ply_payload<T>(
      in, elems, "read_ply_binary",
      [&](detail::scalar_kind k) {
        return detail::read_binary_scalar(in, fmt, k);
      },
      [](detail::scalar_kind k) -> std::uint64_t {
        return detail::scalar_kind_size(k);
      },
      limits);
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
[[nodiscard]] std::vector<gte::Triangle3<T>> read_ply(
    std::istream& in, ply_read_limits const& limits = {}) {
  const auto start = in.tellg();
  if (start < 0) {
    // Non-seekable — fall back to ASCII, which will throw with a
    // clear error if the file turns out to be binary.
    return read_ply_ascii<T>(in, limits);
  }
  // Read just enough of the header to find the format line. We
  // can't reuse `read_ply_header` here because it consumes element
  // declarations too; we need to rewind before that point.
  detail::skip_utf8_bom(in);
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
      format = detail::parse_format_keyword(fmt);
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
    return read_ply_ascii<T>(in, limits);
  }
  if (format == detail::ply_format::ascii) return read_ply_ascii<T>(in, limits);
  return read_ply_binary<T>(in, limits);
}

template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_ply_file(
    std::string const& path, ply_read_limits const& limits = {}) {
  std::ifstream in{path, std::ios::binary};
  if (!in) {
    throw std::runtime_error{"read_ply_file: cannot open '" + path + "'"};
  }
  return read_ply<T>(in, limits);
}

} // namespace rvegen
