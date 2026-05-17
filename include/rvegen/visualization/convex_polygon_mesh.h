#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "../shapes/convex_polygon.h"
#include "triangle_mesh.h"

namespace rvegen {

// Convert a 2D convex polygon to a flat triangle fan in the z=0 plane.
// The polygon's vertex list is CCW by construction (convex_polygon's
// invariant); triangulating as a fan from vertex 0 is correct for any
// convex polygon and uses N - 2 triangles for N vertices — same vertex
// count as the input, no extra centre point.
//
// Degenerate triangles (zero or near-zero signed area, which happens
// when the input polygon has collinear vertices — the convex_polygon
// ctor only checks `size() >= 3`, not non-collinearity) are skipped so
// downstream renderers don't receive invisible-but-counted geometry.
// Tolerance is scale-relative: any triangle whose signed area falls
// below `eps * bbox_diag²` is treated as degenerate.
template <typename T>
[[nodiscard]] inline triangle_mesh<T>
to_mesh(convex_polygon<T> const& poly) {
  triangle_mesh<T> m;
  auto const& vs = poly.vertices();
  if (vs.size() < 3) return m;

  m.verts.reserve(vs.size());
  T min_x = vs[0][0], max_x = vs[0][0];
  T min_y = vs[0][1], max_y = vs[0][1];
  for (auto const& v : vs) {
    m.verts.push_back({v[0], v[1], T{0}});
    if (v[0] < min_x) min_x = v[0]; else if (v[0] > max_x) max_x = v[0];
    if (v[1] < min_y) min_y = v[1]; else if (v[1] > max_y) max_y = v[1];
  }
  const T dx = max_x - min_x;
  const T dy = max_y - min_y;
  const T area_eps =
      std::numeric_limits<T>::epsilon() * (dx * dx + dy * dy);

  m.tris.reserve(vs.size() - 2);
  for (std::size_t i = 1; i + 1 < vs.size(); ++i) {
    auto const& a = vs[0];
    auto const& b = vs[i];
    auto const& c = vs[i + 1];
    // 2× signed area; positive iff CCW.
    const T cross = (b[0] - a[0]) * (c[1] - a[1])
                  - (b[1] - a[1]) * (c[0] - a[0]);
    if (std::abs(cross) <= area_eps) continue;   // collinear → skip
    m.tris.push_back({0, i, i + 1});
  }

  return m;
}

} // namespace rvegen
