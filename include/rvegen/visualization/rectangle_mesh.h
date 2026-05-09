#pragma once

#include <array>
#include <cstddef>

#include "../shapes/rectangle.h"
#include "triangle_mesh.h"

namespace rvegen {

// Convert a 2D axis-aligned rectangle to a flat 2-triangle quad in the z=0
// plane. CCW from +z so the quad faces the camera.
template <typename T>
[[nodiscard]] inline triangle_mesh<T> to_mesh(rectangle<T> const& r) {
  triangle_mesh<T> m;

  const T x0 = r.min[0], y0 = r.min[1];
  const T x1 = r.max[0], y1 = r.max[1];

  m.verts = {
    {x0, y0, T{0}},
    {x1, y0, T{0}},
    {x1, y1, T{0}},
    {x0, y1, T{0}},
  };
  m.tris = {
    {0, 1, 2},
    {0, 2, 3},
  };

  return m;
}

} // namespace rvegen
