#pragma once

// 2D bounded Voronoi tessellation by seed points (#114 phase 1).
//
// Given a rectangular domain and N seed points in it, produces N
// convex polygons — one per seed — that partition the domain. Each
// cell is the set of points closer to its seed than to any other.
// The standard textbook Voronoi diagram, clipped to a bounding box.
//
// Why not use Voro++ / CGAL?
//   For 2D Voronoi in a rectangle the algorithm is well-bounded:
//   Sutherland-Hodgman half-plane clipping. Each cell starts as
//   the bounding rectangle and is clipped against the perpendicular
//   bisector of the seed against every other seed (keeping the
//   "closer-to-our-seed" side). ~80 lines, no external dependency,
//   robust to all non-degenerate inputs.
//
//   3D Voronoi via half-space intersection needs vertex enumeration
//   on a polytope, which is a chunk of code (~400 lines) plus face
//   reconstruction. Deferred to a follow-up.
//
// Output type:
//   `build()` returns `std::vector<polygon_type>` where
//   `polygon_type = std::vector<std::array<T, 2>>` — one CCW vertex
//   list per cell, in seed-order (cell `i` corresponds to seed `i`).
//   Empty polygons (degenerate clipping) appear in the output but
//   are typically only produced when two seeds coincide. Wrapping
//   the polygons into a shape_base subclass (a future 2D
//   `convex_polygon` shape, or extruded 3D voronoi_cell prisms) is
//   deliberately separate — this header is the algorithm primitive.
//
// Numerical considerations:
//   * `T` should be a floating-point type. Integer / fixed-point
//     specialisations would need an exact-arithmetic kernel.
//   * Seeds on the bounding-box edge or coincident with each other
//     produce degenerate (zero-area) cells without throwing. The
//     caller can detect this by `cell.size() < 3` if they care.
//   * Edge-on-bisector intersections are handled by the half-plane
//     test `A·x + B·y ≤ C`: a point exactly on the bisector lands
//     on BOTH sides' boundary and is kept in both cells. For floats
//     this is robust enough at engineering scales.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace rvegen {

template <typename T = double>
class voronoi_generator_2d {
public:
  using value_type = T;
  using point_type = std::array<T, 2>;
  using polygon_type = std::vector<point_type>;

  // Domain is the rectangle `[0, Lx] × [0, Ly]`. Seeds must lie in
  // the open domain (boundary-coincident seeds produce degenerate
  // cells but the routine doesn't refuse them).
  voronoi_generator_2d(T Lx, T Ly, std::vector<point_type> seeds)
      : _Lx{Lx}, _Ly{Ly}, _seeds{std::move(seeds)} {
    if (Lx <= T{0} || Ly <= T{0}) {
      throw std::runtime_error{
          "voronoi_generator_2d: Lx and Ly must both be positive"};
    }
    if (_seeds.empty()) {
      throw std::runtime_error{
          "voronoi_generator_2d: at least one seed required"};
    }
  }

  [[nodiscard]] std::size_t seed_count() const noexcept {
    return _seeds.size();
  }

  // Build all cells. Cell `i` corresponds to seed `i`. CCW vertex
  // order in each cell. For one seed, the cell is the full bounding
  // rectangle.
  [[nodiscard]] std::vector<polygon_type> build() const {
    std::vector<polygon_type> out;
    out.reserve(_seeds.size());
    const polygon_type bbox = make_bounding_rectangle();

    for (std::size_t i = 0; i < _seeds.size(); ++i) {
      polygon_type cell = bbox;
      auto const& p_i = _seeds[i];
      for (std::size_t j = 0; j < _seeds.size(); ++j) {
        if (i == j) continue;
        auto const& p_j = _seeds[j];
        // Perpendicular-bisector half-plane: keep the points x such
        // that |x - p_i|² ≤ |x - p_j|². Expanding gives
        //   2·(p_j - p_i)·x ≤ |p_j|² - |p_i|²
        // i.e. A·x + B·y ≤ C with
        //   A = 2(p_j.x - p_i.x)
        //   B = 2(p_j.y - p_i.y)
        //   C = |p_j|² - |p_i|²
        const T A = T{2} * (p_j[0] - p_i[0]);
        const T B = T{2} * (p_j[1] - p_i[1]);
        const T C = (p_j[0] * p_j[0] + p_j[1] * p_j[1]) -
                    (p_i[0] * p_i[0] + p_i[1] * p_i[1]);
        cell = clip_polygon(cell, A, B, C);
        if (cell.size() < 3) break;   // collapsed; further clipping won't recover
      }
      out.push_back(std::move(cell));
    }
    return out;
  }

private:
  T _Lx;
  T _Ly;
  std::vector<point_type> _seeds;

  // Bounding rectangle vertices in CCW order, starting bottom-left.
  [[nodiscard]] polygon_type make_bounding_rectangle() const {
    return {{T{0}, T{0}},
            {_Lx,  T{0}},
            {_Lx,  _Ly },
            {T{0}, _Ly }};
  }

  // Sutherland-Hodgman clipping by the half-plane `A·x + B·y ≤ C`.
  // Walks the polygon edges and for each edge keeps:
  //   * The destination vertex if it's inside.
  //   * The intersection point if the edge crosses the boundary.
  // Edge cases on the boundary are treated as inside (≤, not <), so
  // a vertex exactly on the bisector is preserved.
  static polygon_type clip_polygon(polygon_type const& polygon,
                                    T A, T B, T C) {
    polygon_type out;
    out.reserve(polygon.size() + 1);
    const auto n = polygon.size();
    if (n == 0) return out;

    auto inside = [A, B, C](point_type const& p) {
      return A * p[0] + B * p[1] <= C;
    };
    auto intersect = [A, B, C](point_type const& a, point_type const& b) {
      // Edge parameterised as a + t · (b - a), 0 ≤ t ≤ 1. Solve
      // A·(a.x + t·(b.x - a.x)) + B·(a.y + t·(b.y - a.y)) = C for t.
      const T denom = A * (b[0] - a[0]) + B * (b[1] - a[1]);
      // denom == 0 means the edge is parallel to the boundary; the
      // calling pattern guarantees one endpoint is inside and the
      // other outside (strict inequalities), so parallel-to-boundary
      // can only happen for both-on-boundary, which never reaches
      // this branch (both inside, both kept). Defensive return of
      // the midpoint just in case.
      if (std::abs(denom) < std::numeric_limits<T>::epsilon() * T{1024}) {
        return point_type{(a[0] + b[0]) * T{0.5}, (a[1] + b[1]) * T{0.5}};
      }
      const T t = (C - A * a[0] - B * a[1]) / denom;
      return point_type{a[0] + t * (b[0] - a[0]),
                        a[1] + t * (b[1] - a[1])};
    };

    for (std::size_t k = 0; k < n; ++k) {
      auto const& start = polygon[k];
      auto const& end   = polygon[(k + 1) % n];
      const bool start_in = inside(start);
      const bool end_in   = inside(end);
      if (start_in && end_in) {
        out.push_back(end);
      } else if (start_in && !end_in) {
        out.push_back(intersect(start, end));
      } else if (!start_in && end_in) {
        out.push_back(intersect(start, end));
        out.push_back(end);
      }
      // else (both outside): drop both.
    }
    return out;
  }
};

// Convenience helper: shoelace formula for the signed area of a
// CCW polygon (returns positive area for a correctly-wound cell).
// Exposed so tests and callers can verify the area-covers-domain
// invariant without redoing the math.
template <typename T>
[[nodiscard]] T polygon_area(std::vector<std::array<T, 2>> const& poly) {
  const auto n = poly.size();
  if (n < 3) return T{0};
  T sum{0};
  for (std::size_t k = 0; k < n; ++k) {
    auto const& a = poly[k];
    auto const& b = poly[(k + 1) % n];
    sum += a[0] * b[1] - b[0] * a[1];
  }
  return T{0.5} * std::abs(sum);
}

} // namespace rvegen

// One-line bridge from voronoi_generator_2d into the rvegen shape
// pipeline: build the cells, wrap each as a convex_polygon, return
// the shape_vector that the standard writers (voxel, gmsh, svg)
// consume. Skips degenerate (< 3 vertex) cells silently — those
// only arise from coincident seeds, and a writer would refuse them
// anyway via the convex_polygon ctor's vertex-count guard.
//
// Lives in its own header so callers can pull in the helper without
// dragging in shape_base / convex_polygon transitively when they
// only want the raw polygon output.

#include "../shapes/convex_polygon.h"
#include "../shapes/shape_base.h"

namespace rvegen {

template <typename T = double>
[[nodiscard]] inline std::vector<std::unique_ptr<shape_base<T>>>
voronoi_to_shapes(voronoi_generator_2d<T> const& gen) {
  auto cells = gen.build();
  std::vector<std::unique_ptr<shape_base<T>>> out;
  out.reserve(cells.size());
  for (auto& v : cells) {
    if (v.size() < 3) continue;   // degenerate (coincident seeds); skip
    out.push_back(std::make_unique<convex_polygon<T>>(std::move(v)));
  }
  return out;
}

} // namespace rvegen
