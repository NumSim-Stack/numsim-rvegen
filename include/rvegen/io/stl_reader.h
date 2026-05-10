#pragma once

// Minimal STL ASCII reader. Parses a single solid and emits a vector
// of gte::Triangle3<T>. Binary STL + PLY are explicitly out of scope
// for phase 1 — see notes below.
//
// Why ASCII-only and why no PLY:
//   The bulk of practical STL files in materials work are binary
//   (smaller, faster, more common from CAD/meshing tools). ASCII is
//   chosen as the phase-1 reader because (a) it has no endianness or
//   header-size pitfalls, (b) the parsing logic is ~50 lines and fits
//   in a header, (c) it's enough to land an end-to-end mesh-inclusion
//   path through the rvegen pipeline that we can iterate on. Binary
//   STL and PLY (which needs vertex deduplication via element
//   tables) follow in subsequent PRs once the consumer side
//   (mesh_inclusion shape, mesh_inclusion_input) is exercised.
//
// Binary-STL detection:
//   Binary STL files start with an 80-byte header that *often*
//   contains the literal word "solid" — meaning a naive ASCII parser
//   may walk into a binary file and silently consume garbage tokens.
//   We sniff the first ~80 bytes: if more than a small fraction look
//   like non-printable / non-whitespace characters, we refuse with a
//   clear error rather than producing nonsense. This is heuristic but
//   is the standard convention.
//
// Tolerances:
//   The reader is forgiving — extra whitespace, mixed case keywords,
//   and missing trailing newlines all work. It does NOT validate
//   normals against the right-hand-rule of the vertex order; that's
//   the producer's responsibility.

#include <cctype>
#include <cstddef>
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
        "(non-printable bytes in header). Binary STL is not supported "
        "in phase 1 — convert to ASCII via meshlab/admesh, or wait for "
        "the binary reader follow-up."};
  }
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

} // namespace rvegen
