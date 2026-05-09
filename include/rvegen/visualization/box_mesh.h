#pragma once

#include <array>
#include <cstddef>

#include "../shapes/box.h"
#include "triangle_mesh.h"

namespace rvegen {

// Convert a box to a 12-triangle surface mesh (2 triangles per face × 6 faces,
// 8 vertices). Triangle winding is counter-clockwise when viewed from outside;
// renderers can derive flat outward normals via the right-hand rule. The
// `normals` field is left empty so the consumer (VTK, etc.) can auto-generate
// per-face flat normals — what users expect from a box.
template <typename T>
[[nodiscard]] inline triangle_mesh<T> to_mesh(box<T> const& b) {
  triangle_mesh<T> m;

  const T x0 = b.min[0], y0 = b.min[1], z0 = b.min[2];
  const T x1 = b.max[0], y1 = b.max[1], z1 = b.max[2];

  // 8 corners.
  m.verts = {
    {x0, y0, z0}, {x1, y0, z0}, {x1, y1, z0}, {x0, y1, z0},
    {x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1},
  };

  // 6 faces, 2 triangles each, CCW from outside.
  // bottom (z = z0, normal -z): looking from below, CCW = 0,3,2,1
  // top    (z = z1, normal +z): looking from above, CCW = 4,5,6,7
  // front  (y = y0, normal -y): looking from -y, CCW = 0,1,5,4
  // back   (y = y1, normal +y): looking from +y, CCW = 2,3,7,6
  // left   (x = x0, normal -x): looking from -x, CCW = 0,4,7,3
  // right  (x = x1, normal +x): looking from +x, CCW = 1,2,6,5
  m.tris = {
    {0, 3, 2}, {0, 2, 1},   // bottom
    {4, 5, 6}, {4, 6, 7},   // top
    {0, 1, 5}, {0, 5, 4},   // front
    {2, 3, 7}, {2, 7, 6},   // back
    {0, 4, 7}, {0, 7, 3},   // left
    {1, 2, 6}, {1, 6, 5},   // right
  };

  return m;
}

} // namespace rvegen
