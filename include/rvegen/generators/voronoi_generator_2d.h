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
//   The 3D counterpart lives in voronoi_generator_3d.h.
//
// Output type:
//   `build()` returns `std::vector<polygon_type>` where
//   `polygon_type = std::vector<std::array<T, 2>>` — one CCW vertex
//   list per cell, in seed-order (cell `i` corresponds to seed `i`).
//   Empty polygons (numerically collapsed cells) keep their slot in
//   the output so index == seed index always holds. Wrapping
//   the polygons into a shape_base subclass (a future 2D
//   `convex_polygon` shape, or extruded 3D voronoi_cell prisms) is
//   deliberately separate — this header is the algorithm primitive.
//
// Numerical considerations:
//   * `T` should be a floating-point type. Integer / fixed-point
//     specialisations would need an exact-arithmetic kernel.
//   * Scale-invariant (#79): bisectors are normalised to unit normals,
//     so the clip test is a signed distance and every tolerance is
//     relative to the domain size. Works unchanged from 1e-6 to 1e3.
//   * Seeds on the bounding-box edge are valid (their cells stay
//     non-degenerate).
//   * Coincident seeds (closer than 1e-9 of the domain size) and seeds
//     outside the domain are rejected with std::runtime_error.
//   * Edge-on-bisector intersections are handled by the half-plane
//     test `dist ≤ 0`: a point exactly on the bisector lands on BOTH
//     sides' boundary and is kept in both cells.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "voronoi_seed_validation.h"

namespace rvegen {

template <typename T = double>
class voronoi_generator_2d {
public:
  using value_type = T;
  using point_type = std::array<T, 2>;
  using polygon_type = std::vector<point_type>;

  // Domain is the rectangle `[0, Lx] × [0, Ly]`. Seeds must lie in
  // the closed domain (boundary seeds are fine) and must be pairwise
  // distinct; violations throw std::runtime_error.
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
    detail::validate_voronoi_seeds<T, 2>({Lx, Ly}, _seeds,
                                         "voronoi_generator_2d");
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
    const T merge_tol =
        std::numeric_limits<T>::epsilon() * T{16} * std::min(_Lx, _Ly);

    for (std::size_t i = 0; i < _seeds.size(); ++i) {
      polygon_type cell = bbox;
      auto const& p_i = _seeds[i];
      for (std::size_t j = 0; j < _seeds.size(); ++j) {
        if (i == j) continue;
        auto const& p_j = _seeds[j];
        // Perpendicular-bisector half-plane, unit normal n toward p_j:
        //   n · x ≤ n · (p_i + p_j) / 2
        const T dx = p_j[0] - p_i[0];
        const T dy = p_j[1] - p_i[1];
        const T len = std::sqrt(dx * dx + dy * dy);
        const T A = dx / len;
        const T B = dy / len;
        const T C = A * (p_i[0] + p_j[0]) * T{0.5} +
                    B * (p_i[1] + p_j[1]) * T{0.5};
        cell = clip_polygon(cell, A, B, C, merge_tol);
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

  // Sutherland-Hodgman clipping by the half-plane `A·x + B·y ≤ C`
  // with (A, B) a UNIT normal, so `A·x + B·y - C` is a signed
  // distance. Walks the polygon edges and for each edge keeps:
  //   * The destination vertex if it's inside.
  //   * The intersection point if the edge crosses the boundary.
  // Boundary points count as inside (≤, not <). Consecutive vertices
  // closer than `merge_tol` are merged so on-bisector vertices are not
  // duplicated.
  static polygon_type clip_polygon(polygon_type const& polygon,
                                    T A, T B, T C, T merge_tol) {
    polygon_type out;
    out.reserve(polygon.size() + 1);
    const auto n = polygon.size();
    if (n == 0) return out;

    auto dist = [A, B, C](point_type const& p) {
      return A * p[0] + B * p[1] - C;
    };
    auto push = [&out, merge_tol](point_type const& p) {
      if (!out.empty() &&
          std::abs(out.back()[0] - p[0]) <= merge_tol &&
          std::abs(out.back()[1] - p[1]) <= merge_tol) {
        return;
      }
      out.push_back(p);
    };

    for (std::size_t k = 0; k < n; ++k) {
      auto const& start = polygon[k];
      auto const& end   = polygon[(k + 1) % n];
      const T ds = dist(start);
      const T de = dist(end);
      const bool start_in = ds <= T{0};
      const bool end_in   = de <= T{0};
      if (start_in != end_in) {
        // Endpoints are on strictly opposite sides (or one exactly on
        // the line), so ds - de != 0 and t lies in [0, 1].
        const T t = ds / (ds - de);
        push(point_type{start[0] + t * (end[0] - start[0]),
                        start[1] + t * (end[1] - start[1])});
      }
      if (end_in) push(end);
      // else (both outside): drop both.
    }
    if (out.size() > 1 &&
        std::abs(out.front()[0] - out.back()[0]) <= merge_tol &&
        std::abs(out.front()[1] - out.back()[1]) <= merge_tol) {
      out.pop_back();
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
// consume. Always returns one shape per seed, in seed order (#79):
// a numerically collapsed (< 3 vertex) cell becomes an empty
// convex_polygon placeholder (area 0, contains nothing) so downstream
// per-grain data indexed by seed stays aligned.
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
    if (v.size() < 3) {
      out.push_back(std::make_unique<convex_polygon<T>>());   // placeholder
      continue;
    }
    out.push_back(std::make_unique<convex_polygon<T>>(std::move(v)));
  }
  return out;
}

} // namespace rvegen
