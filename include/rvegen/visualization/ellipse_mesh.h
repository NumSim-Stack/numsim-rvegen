#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

#include "../shapes/ellipse.h"
#include "triangle_mesh.h"

namespace rvegen {

// Convert a 2D rotated ellipse to a flat triangle-fan mesh in the z=0
// plane. The rim is parameterised in the ellipse-local frame
// (extent[0]·cos(θ), extent[1]·sin(θ)) and rotated into world space via
// the stored `axis` basis — same transform GTE uses internally.
template <typename T>
[[nodiscard]] inline triangle_mesh<T>
to_mesh(ellipse<T> const& e, std::size_t segments = 64) {
  triangle_mesh<T> m;
  if (segments < 3) segments = 3;
  m.verts.reserve(segments + 1);
  m.tris.reserve(segments);

  const T cx = e.center[0];
  const T cy = e.center[1];
  const T ra = e.extent[0];
  const T rb = e.extent[1];
  const T a0x = e.axis[0][0], a0y = e.axis[0][1];
  const T a1x = e.axis[1][0], a1y = e.axis[1][1];
  const T two_pi = T{2} * std::numbers::pi_v<T>;

  m.verts.push_back({cx, cy, T{0}});
  for (std::size_t i = 0; i < segments; ++i) {
    const T angle = two_pi * static_cast<T>(i) / static_cast<T>(segments);
    const T u = ra * std::cos(angle);
    const T v = rb * std::sin(angle);
    // Local (u, v) → world via axis basis.
    m.verts.push_back({cx + u * a0x + v * a1x,
                       cy + u * a0y + v * a1y,
                       T{0}});
  }

  for (std::size_t i = 0; i < segments; ++i) {
    const std::size_t a = i + 1;
    const std::size_t b = (i + 1) % segments + 1;
    m.tris.push_back({0, a, b});
  }

  return m;
}

} // namespace rvegen
