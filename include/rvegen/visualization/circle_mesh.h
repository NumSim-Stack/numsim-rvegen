#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

#include "../shapes/circle.h"
#include "triangle_mesh.h"

namespace rvegen {

// Convert a 2D circle to a flat-disk triangle mesh in the z=0 plane.
//
// 2D shapes were originally skipped in the mesh dispatcher (the assumption
// being voxel rendering covers them), but the in-app viewport and any
// shape-rendering 3D view still want a polygonal representation. The mesh
// is a triangle fan: one centre vertex plus N rim vertices, N triangles.
//
// `segments` defaults to 64 — round enough at typical circle sizes that
// the silhouette doesn't reveal facets. Increase for export quality.
template <typename T>
[[nodiscard]] inline triangle_mesh<T>
to_mesh(circle<T> const& c, std::size_t segments = 64) {
  triangle_mesh<T> m;
  if (segments < 3) segments = 3;
  m.verts.reserve(segments + 1);
  m.tris.reserve(segments);

  const T cx = c.center[0];
  const T cy = c.center[1];
  const T r  = c.radius;
  const T two_pi = T{2} * std::numbers::pi_v<T>;

  // Centre vertex first; rim vertices follow.
  m.verts.push_back({cx, cy, T{0}});
  for (std::size_t i = 0; i < segments; ++i) {
    const T angle = two_pi * static_cast<T>(i) / static_cast<T>(segments);
    m.verts.push_back({cx + r * std::cos(angle),
                       cy + r * std::sin(angle),
                       T{0}});
  }

  // Triangle fan, CCW from +z so the disk faces the camera.
  for (std::size_t i = 0; i < segments; ++i) {
    const std::size_t a = i + 1;
    const std::size_t b = (i + 1) % segments + 1;
    m.tris.push_back({0, a, b});
  }

  return m;
}

} // namespace rvegen
