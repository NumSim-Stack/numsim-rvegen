#pragma once

// Minimal STL ASCII reader. Parses a single solid and emits a vector
// of gte::Triangle3<T>. Binary STL + PLY are explicitly out of scope
// for phase 1 — see notes below.
//
// Why ASCII-only and why no PLY:
//   The bulk of practical STL files in materials work are binary
//   (smaller, faster, more common from CAD/meshing tools). ASCII is
//   chosen as the phase-1 reader because (a) it has no endianness or
//   header-size pitfalls, (b) the parsing logic is ~30 lines and fits
//   in a header, (c) it's enough to land an end-to-end mesh-inclusion
//   path through the rvegen pipeline that we can iterate on. Binary
//   STL and PLY (which needs vertex deduplication via element
//   tables) follow in subsequent PRs once the consumer side
//   (triangle_mesh shape, mesh_inclusion_input) is exercised.
//
// Tolerances:
//   The reader is forgiving — extra whitespace, mixed case keywords,
//   and missing trailing newlines all work. It does NOT validate
//   normals against the right-hand-rule of the vertex order; that's
//   the producer's responsibility.

#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Mathematics/Triangle.h>
#include <Mathematics/Vector3.h>

namespace rvegen {

template <typename T = double>
[[nodiscard]] std::vector<gte::Triangle3<T>> read_stl_ascii(std::istream& in) {
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
        throw std::runtime_error{"read_stl_ascii: expected 'outer' got '" +
                                 token + "'"};
      }
      in >> token;   // "loop"

      gte::Triangle3<T> tri;
      for (int i = 0; i < 3; ++i) {
        in >> token;   // "vertex"
        if (token != "vertex") {
          throw std::runtime_error{
              "read_stl_ascii: expected 'vertex' got '" + token + "'"};
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
  std::ifstream in{path};
  if (!in) {
    throw std::runtime_error{"read_stl_ascii_file: cannot open '" + path +
                             "'"};
  }
  return read_stl_ascii<T>(in);
}

} // namespace rvegen
