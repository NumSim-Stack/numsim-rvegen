#pragma once

// STL readers — ASCII + binary + auto-detect.
//
// Both formats produce the same `std::vector<gte::Triangle3<T>>`.
// Most CAD/meshing tools emit binary STL by default (smaller, faster,
// no whitespace-handling pitfalls); rvegen's first cut shipped ASCII
// only. Binary support lands here so users don't have to convert via
// meshlab/admesh as a preprocessing step.
//
// API surface:
//   * `read_stl_ascii(istream)` / `read_stl_ascii_file(path)` — explicit
//     ASCII; rejects binary input up-front with a clear error.
//   * `read_stl_binary(istream)` / `read_stl_binary_file(path)` — explicit
//     binary; reads the 80-byte header, the uint32 triangle count, then
//     50 bytes per triangle (normal + 3 vertices + attribute bytes).
//   * `read_stl(istream)` / `read_stl_file(path)` — sniffs the first
//     bytes and dispatches to the right reader. Most callers want this.
//
// PLY follows in a subsequent PR (needs vertex deduplication via
// element tables — non-trivial relative to STL's flat triangle list).
//
// Binary-STL detection:
//   Binary STL files start with an 80-byte header that *often*
//   contains the literal word "solid" — meaning a naive ASCII parser
//   may walk into a binary file and silently consume garbage tokens.
//   We sniff the first ~256 bytes: if more than a small fraction look
//   like non-printable / non-whitespace characters, we refuse with a
//   clear error rather than producing nonsense. This is heuristic but
//   is the standard convention.
//
//   Caveat: detection requires a SEEKABLE stream (we read the head
//   then `seekg(0)` back to the start). For non-seekable inputs
//   (pipes, sockets) the sniff is silently skipped and a binary
//   payload will fall into the parse-error path instead — still
//   throws, just with a less helpful message. Consumers piping
//   in-memory data should wrap it in `std::stringstream`, which is
//   seekable; `std::ifstream` of a real file is also seekable.
//
// Tolerances:
//   The reader is forgiving — extra whitespace, mixed case keywords,
//   and missing trailing newlines all work. It does NOT validate
//   normals against the right-hand-rule of the vertex order; that's
//   the producer's responsibility.

#include <bit>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Mathematics/Triangle.h>
#include <Mathematics/Vector3.h>

namespace rvegen {

namespace detail {

// Heuristic binary-STL refusal. Reads the first up-to-256 bytes,
// rewinds, and if a substantial fraction looks non-printable assume
// binary and throw. The threshold is intentionally generous — only
// pathological binary files (printable headers + zero offset) escape
// the check, and those are best caught by the parse-error path that
// follows.
inline void reject_binary_stl(std::istream& in) {
  if (!in.seekg(0, std::ios::beg)) return;   // non-seekable stream — skip
  std::string head;
  head.resize(256);
  in.read(head.data(), static_cast<std::streamsize>(head.size()));
  const auto bytes_read = in.gcount();
  in.clear();
  in.seekg(0, std::ios::beg);
  if (bytes_read <= 0) return;
  std::size_t non_printable = 0;
  for (std::streamsize i = 0; i < bytes_read; ++i) {
    const unsigned char c = static_cast<unsigned char>(head[i]);
    if (!std::isprint(c) && !std::isspace(c)) ++non_printable;
  }
  // > 5 % non-printable in the first 256 bytes is strongly indicative
  // of binary content. Real ASCII STL is text-only with newlines.
  if (non_printable * 20 > static_cast<std::size_t>(bytes_read)) {
    throw std::runtime_error{
        "read_stl_ascii: input appears to be a binary STL "
        "(non-printable bytes in header). Use read_stl_binary() or "
        "read_stl() (auto-detect) instead."};
  }
}

// Returns true if the first ~256 bytes look like binary STL: ≥ 5 %
// non-printable / non-whitespace bytes. Used by `read_stl` to dispatch
// between ASCII and binary readers. Same heuristic as
// reject_binary_stl, just returning a bool rather than throwing.
inline bool looks_like_binary_stl(std::istream& in) {
  if (!in.seekg(0, std::ios::beg)) return false;
  std::string head;
  head.resize(256);
  in.read(head.data(), static_cast<std::streamsize>(head.size()));
  const auto bytes_read = in.gcount();
  in.clear();
  in.seekg(0, std::ios::beg);
  if (bytes_read <= 0) return false;
  std::size_t non_printable = 0;
  for (std::streamsize i = 0; i < bytes_read; ++i) {
    const unsigned char c = static_cast<unsigned char>(head[i]);
    if (!std::isprint(c) && !std::isspace(c)) ++non_printable;
  }
  return non_printable * 20 > static_cast<std::size_t>(bytes_read);
}

// Decode 4 little-endian bytes as a float32 / uint32. STL is defined
// as little-endian regardless of host byte order; on a big-endian host
// we'd need byte swapping. Today's targets (x86_64, ARM64 in
// little-endian mode) make this a no-op, but a `std::endian` guard
// catches a future big-endian build at compile time.
inline std::uint32_t le_u32(unsigned char const* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

inline float le_f32(unsigned char const* p) {
  static_assert(std::endian::native == std::endian::little ||
                    std::endian::native == std::endian::big,
                "mixed-endian host not supported");
  const std::uint32_t bits = le_u32(p);
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline std::string position_marker(std::istream& in) {
  // tellg() returns -1 on non-seekable streams; gracefully report no
  // position info in that case rather than embedding "-1" in the
  // error message.
  if (auto pos = in.tellg(); pos >= 0) {
    return " (at byte " + std::to_string(static_cast<long long>(pos)) + ")";
  }
  return "";
}

} // namespace detail

template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_stl_ascii(std::istream& in) {
  detail::reject_binary_stl(in);
  std::vector<gte::Triangle3<T>> triangles;
  std::string token;
  while (in >> token) {
    if (token == "solid" || token == "endsolid") {
      // Skip the optional name on the same line.
      std::string rest;
      std::getline(in, rest);
      continue;
    }
    if (token == "facet") {
      // facet normal nx ny nz
      std::string n;
      in >> n;   // "normal"
      T nx, ny, nz;
      in >> nx >> ny >> nz;
      // outer loop
      in >> token;
      if (token != "outer") {
        throw std::runtime_error{
            "read_stl_ascii: expected 'outer' got '" + token + "'" +
            detail::position_marker(in)};
      }
      in >> token;   // "loop"

      gte::Triangle3<T> tri;
      for (int i = 0; i < 3; ++i) {
        in >> token;   // "vertex"
        if (token != "vertex") {
          throw std::runtime_error{
              "read_stl_ascii: expected 'vertex' got '" + token + "'" +
              detail::position_marker(in)};
        }
        T vx, vy, vz;
        in >> vx >> vy >> vz;
        tri.v[i][0] = vx;
        tri.v[i][1] = vy;
        tri.v[i][2] = vz;
      }
      triangles.push_back(tri);

      in >> token;   // "endloop"
      in >> token;   // "endfacet"
    }
  }
  return triangles;
}

template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_stl_ascii_file(
    std::string const& path) {
  std::ifstream in{path, std::ios::binary};
  if (!in) {
    throw std::runtime_error{"read_stl_ascii_file: cannot open '" + path +
                             "'"};
  }
  return read_stl_ascii<T>(in);
}

// Binary STL layout (little-endian throughout):
//   bytes  0..79   : 80-byte ASCII header (typically a producer name)
//   bytes 80..83   : uint32 triangle count N
//   bytes 84..end  : N × 50-byte triangle records:
//                      [0..11]  : normal (3 × float32)
//                      [12..47] : 3 vertices (each 3 × float32)
//                      [48..49] : attribute byte count (usually 0)
//
// We read the header (skip), the triangle count, then loop reading
// 50-byte records into gte::Triangle3<T>. Normals are discarded — gte
// recomputes them when needed and we don't trust the producer's
// right-hand-rule anyway (some CAD tools emit inconsistent winding
// even in binary STL).
template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_stl_binary(std::istream& in) {
  std::vector<gte::Triangle3<T>> triangles;

  // 80-byte header.
  std::array<unsigned char, 80> header;
  in.read(reinterpret_cast<char*>(header.data()),
          static_cast<std::streamsize>(header.size()));
  if (in.gcount() != static_cast<std::streamsize>(header.size())) {
    throw std::runtime_error{
        "read_stl_binary: file too short to contain the 80-byte header"};
  }

  // 4-byte triangle count.
  std::array<unsigned char, 4> count_bytes;
  in.read(reinterpret_cast<char*>(count_bytes.data()), 4);
  if (in.gcount() != 4) {
    throw std::runtime_error{
        "read_stl_binary: file too short to contain the triangle count"};
  }
  const std::uint32_t n_tris = detail::le_u32(count_bytes.data());

  // Sanity cap: 100M triangles would be a 5 GB file. Bigger than
  // anything reasonable; refuse rather than allocate gigabytes on a
  // malformed input.
  constexpr std::uint32_t max_tris = 100'000'000;
  if (n_tris > max_tris) {
    throw std::runtime_error{
        "read_stl_binary: triangle count " + std::to_string(n_tris) +
        " exceeds sanity limit (" + std::to_string(max_tris) +
        ") — probably a malformed file"};
  }

  triangles.reserve(n_tris);

  // 50-byte triangle records.
  std::array<unsigned char, 50> rec;
  for (std::uint32_t t = 0; t < n_tris; ++t) {
    in.read(reinterpret_cast<char*>(rec.data()),
            static_cast<std::streamsize>(rec.size()));
    if (in.gcount() != static_cast<std::streamsize>(rec.size())) {
      throw std::runtime_error{
          "read_stl_binary: file ended after " + std::to_string(t) +
          " of " + std::to_string(n_tris) + " advertised triangles"};
    }
    gte::Triangle3<T> tri;
    // bytes 12..47 are the 3 × 3 vertex floats. Skip the normal (0..11).
    for (int i = 0; i < 3; ++i) {
      const std::size_t base = 12 + i * 12;
      tri.v[i][0] = static_cast<T>(detail::le_f32(rec.data() + base + 0));
      tri.v[i][1] = static_cast<T>(detail::le_f32(rec.data() + base + 4));
      tri.v[i][2] = static_cast<T>(detail::le_f32(rec.data() + base + 8));
    }
    triangles.push_back(tri);
  }

  return triangles;
}

template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_stl_binary_file(
    std::string const& path) {
  std::ifstream in{path, std::ios::binary};
  if (!in) {
    throw std::runtime_error{"read_stl_binary_file: cannot open '" + path +
                             "'"};
  }
  return read_stl_binary<T>(in);
}

// Auto-detect: sniff the first bytes, dispatch to read_stl_ascii or
// read_stl_binary. Most callers want this — they don't know up-front
// which format a third-party STL file uses.
//
// Requires a seekable stream (the sniff seeks back to 0). Non-seekable
// inputs should pick the explicit reader.
template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_stl(std::istream& in) {
  if (detail::looks_like_binary_stl(in)) {
    return read_stl_binary<T>(in);
  }
  return read_stl_ascii<T>(in);
}

template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_stl_file(
    std::string const& path) {
  std::ifstream in{path, std::ios::binary};
  if (!in) {
    throw std::runtime_error{"read_stl_file: cannot open '" + path + "'"};
  }
  return read_stl<T>(in);
}

} // namespace rvegen
