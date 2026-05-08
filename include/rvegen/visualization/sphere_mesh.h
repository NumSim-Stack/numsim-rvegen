#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "../shapes/sphere.h"
#include "triangle_mesh.h"

namespace rvegen {

namespace detail {

// Subdivide an icosahedron `subdivisions` times and project every vertex back
// onto the unit sphere. Returns positions on the unit sphere (caller scales
// and translates). Triangle counts:
//   level 0 → 20 tris (raw icosahedron)
//   level 1 → 80 tris
//   level 2 → 320 tris (Tessera preview default)
//   level 3 → 1280 tris
template <typename T>
struct icosphere_unit {
  std::vector<std::array<T, 3>> verts;
  std::vector<std::array<std::size_t, 3>> tris;
};

template <typename T>
[[nodiscard]] inline icosphere_unit<T>
build_unit_icosphere(unsigned subdivisions) {
  // Golden-ratio icosahedron: 12 vertices forming three orthogonal golden
  // rectangles. Coordinates are scaled to put each vertex on the unit sphere.
  const T t = (T{1} + std::sqrt(T{5})) / T{2};   // golden ratio
  const T s = T{1} / std::sqrt(T{1} + t * t);    // unit-length scale
  const T a = s;
  const T b = s * t;

  icosphere_unit<T> ico;
  ico.verts = {
    {-a,  b,  0}, { a,  b,  0}, {-a, -b,  0}, { a, -b,  0},
    { 0, -a,  b}, { 0,  a,  b}, { 0, -a, -b}, { 0,  a, -b},
    { b,  0, -a}, { b,  0,  a}, {-b,  0, -a}, {-b,  0,  a},
  };
  ico.tris = {
    {0,11,5}, {0,5,1}, {0,1,7}, {0,7,10}, {0,10,11},
    {1,5,9},  {5,11,4},{11,10,2},{10,7,6}, {7,1,8},
    {3,9,4},  {3,4,2}, {3,2,6},  {3,6,8},  {3,8,9},
    {4,9,5},  {2,4,11},{6,2,10}, {8,6,7},  {9,8,1},
  };

  // Subdivide: split each triangle into 4 by inserting midpoints on its
  // three edges. Cache midpoints in a map keyed by (lo, hi) vertex indices
  // so each shared edge contributes one new vertex, not two.
  for (unsigned step = 0; step < subdivisions; ++step) {
    std::unordered_map<std::size_t, std::size_t> midpoint_cache;
    auto edge_key = [](std::size_t i, std::size_t j) {
      const auto lo = std::min(i, j);
      const auto hi = std::max(i, j);
      return (lo << 32) | hi;          // safe for vertex counts well under 2^32
    };
    auto midpoint_index = [&](std::size_t i, std::size_t j) {
      const auto key = edge_key(i, j);
      if (auto it = midpoint_cache.find(key); it != midpoint_cache.end())
        return it->second;
      const auto& a_pt = ico.verts[i];
      const auto& b_pt = ico.verts[j];
      std::array<T, 3> mid{
        (a_pt[0] + b_pt[0]) * T{0.5},
        (a_pt[1] + b_pt[1]) * T{0.5},
        (a_pt[2] + b_pt[2]) * T{0.5},
      };
      // Project the midpoint back onto the unit sphere; otherwise the mesh
      // is just a faceted icosahedron at higher resolution.
      const T len = std::sqrt(mid[0]*mid[0] + mid[1]*mid[1] + mid[2]*mid[2]);
      mid[0] /= len; mid[1] /= len; mid[2] /= len;
      const auto idx = ico.verts.size();
      ico.verts.push_back(mid);
      midpoint_cache[key] = idx;
      return idx;
    };

    std::vector<std::array<std::size_t, 3>> new_tris;
    new_tris.reserve(ico.tris.size() * 4);
    for (auto const& tri : ico.tris) {
      const auto a01 = midpoint_index(tri[0], tri[1]);
      const auto a12 = midpoint_index(tri[1], tri[2]);
      const auto a20 = midpoint_index(tri[2], tri[0]);
      new_tris.push_back({tri[0], a01, a20});
      new_tris.push_back({tri[1], a12, a01});
      new_tris.push_back({tri[2], a20, a12});
      new_tris.push_back({a01,    a12, a20});
    }
    ico.tris = std::move(new_tris);
  }

  return ico;
}

} // namespace detail

// Convert a sphere to a triangle mesh. Default subdivision level 2 yields 320
// triangles — preview quality, what Tessera renders in the viewport. Per-vertex
// normals are populated (smooth shading is what users expect for spheres).
template <typename T>
[[nodiscard]] inline triangle_mesh<T>
to_mesh(sphere<T> const& s, unsigned subdivisions = 2) {
  auto ico = detail::build_unit_icosphere<T>(subdivisions);

  triangle_mesh<T> m;
  m.tris = std::move(ico.tris);
  m.verts.reserve(ico.verts.size());
  m.normals.reserve(ico.verts.size());

  for (auto const& v : ico.verts) {
    m.verts.push_back({
      s.center[0] + s.radius * v[0],
      s.center[1] + s.radius * v[1],
      s.center[2] + s.radius * v[2],
    });
    // Normal is the unit-sphere direction — already unit-length by construction.
    m.normals.push_back({v[0], v[1], v[2]});
  }

  return m;
}

} // namespace rvegen
