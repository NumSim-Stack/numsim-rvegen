#pragma once

#include <array>
#include <cstddef>

#include "../shapes/convex_polygon.h"
#include "triangle_mesh.h"

namespace rvegen {

// Convert a 2D convex polygon to a flat triangle fan in the z=0 plane.
// The polygon's vertex list is CCW by construction (convex_polygon's
// invariant); triangulating as a fan from vertex 0 is correct for any
// convex polygon and uses N - 2 triangles for N vertices — same vertex
// count as the input, no extra centre point.
template <typename T>
[[nodiscard]] inline triangle_mesh<T>
to_mesh(convex_polygon<T> const& poly) {
  triangle_mesh<T> m;
  auto const& vs = poly.vertices();
  if (vs.size() < 3) return m;

  m.verts.reserve(vs.size());
  for (auto const& v : vs) {
    m.verts.push_back({v[0], v[1], T{0}});
  }

  m.tris.reserve(vs.size() - 2);
  for (std::size_t i = 1; i + 1 < vs.size(); ++i) {
    m.tris.push_back({0, i, i + 1});
  }

  return m;
}

} // namespace rvegen
