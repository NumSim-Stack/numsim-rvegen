#pragma once

#include <array>
#include <cstddef>
#include <map>

#include "../shapes/mesh_inclusion.h"
#include "triangle_mesh.h"

namespace rvegen {

// Convert a placed mesh_inclusion (STL / PLY inclusion) to a triangle
// mesh in world coordinates: every base vertex goes through the
// shape's translation · rotation · scale (`mesh_inclusion::to_world`).
//
// Bit-identical base vertices are shared (the readers emit triangle
// soups, so this welds them back), keeping the output compact for the
// three.js / Tessera renderers. The base mesh is validated and
// outward-wound at construction, and the placement is a proper
// rotation with positive scale, so the triangles stay CCW from outside.
// Degenerate (zero-area) triangles are passed through untouched.
// Normals are left empty — consumers compute flat or smooth normals.
template <typename T>
[[nodiscard]] inline triangle_mesh<T>
to_mesh(mesh_inclusion<T> const& shape) {
  triangle_mesh<T> m;
  auto const base = shape.base_triangles();
  if (!base || base->empty()) return m;

  std::map<std::array<T, 3>, std::size_t> ids;
  m.tris.reserve(base->size());
  for (auto const& tri : *base) {
    std::array<std::size_t, 3> t{};
    for (int c = 0; c < 3; ++c) {
      auto const& v = tri.v[c];
      auto [it, inserted] = ids.try_emplace({v[0], v[1], v[2]}, m.verts.size());
      if (inserted) m.verts.push_back(shape.to_world(v));
      t[c] = it->second;
    }
    m.tris.push_back(t);
  }
  return m;
}

} // namespace rvegen
