#pragma once

#include <array>
#include <cassert>
#include <cstddef>

#include "../shapes/voronoi_cell.h"
#include "triangle_mesh.h"

namespace rvegen {

// Convert a 3D voronoi_cell (vertex/face polyhedron) to a triangle mesh.
//
// Each face is a planar convex polygon stored as a CCW vertex-index list
// when viewed from outside the cell (voronoi_generator_3d orders them
// against the outward normal). Triangulate each face as a fan from its
// first vertex — N face vertices → N − 2 triangles, preserving the
// face's winding so the resulting mesh has consistent outward normals.
//
// Vertices are emitted once each (the cell already deduplicates them);
// per-vertex normals are intentionally left empty — the cell has hard
// edges between faces, and flat shading from per-face normals is the
// right look. Renderers wanting hard edges either consume `normals`
// empty and compute flat normals, or duplicate vertices per-face.
template <typename T>
[[nodiscard]] inline triangle_mesh<T>
to_mesh(voronoi_cell<T> const& cell) {
  triangle_mesh<T> m;
  auto const& vs = cell.vertices();
  auto const& fs = cell.faces();
  if (vs.empty() || fs.empty()) return m;

  m.verts.reserve(vs.size());
  for (auto const& v : vs) {
    m.verts.push_back({v[0], v[1], v[2]});
  }

  // Sum of (face_size - 2) over faces with ≥ 3 vertices.
  std::size_t n_tris = 0;
  for (auto const& f : fs) if (f.size() >= 3) n_tris += f.size() - 2;
  m.tris.reserve(n_tris);

  for (auto const& face : fs) {
    if (face.size() < 3) continue;   // degenerate
    const std::size_t v0 = face[0];
    assert(v0 < vs.size() &&
           "voronoi_cell::to_mesh: face vertex index out of range");
    for (std::size_t i = 1; i + 1 < face.size(); ++i) {
      assert(face[i] < vs.size() && face[i + 1] < vs.size() &&
             "voronoi_cell::to_mesh: face vertex index out of range");
      m.tris.push_back({v0, face[i], face[i + 1]});
    }
  }

  return m;
}

} // namespace rvegen
