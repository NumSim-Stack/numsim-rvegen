#pragma once

// 2D convex polygon shape. Stores a CCW-ordered list of vertices
// and implements the shape_base contract: area (shoelace), is_inside
// (left-of-every-edge for convex polygons), AABB.
//
// Motivating use case: wrap the per-cell output of
// `voronoi_generator_2d` as `shape_base` instances so the standard
// rvegen pipeline (voxel_writer, gmsh_geo_writer, etc.) can consume
// a polycrystal-style RVE without custom plumbing.
//
// Convex-only by design:
//   * `is_inside` reduces to "the query point is on the same side
//     (left, for CCW) of every edge" — O(N) per query, no
//     special-cases. This is correct only for convex polygons; a
//     general polygon needs ray-casting.
//   * Voronoi cells (the primary input) are always convex, so the
//     restriction matches the use case. Non-convex polygon support
//     would be a separate shape type with its own `is_inside`.
//
// Winding-order convention:
//   Vertices MUST be in CCW order. The ctor doesn't reorder — it
//   trusts the caller (the Voronoi generator already produces CCW).
//   `area()` returns abs(signed area) so it's robust to a mis-wound
//   input, but `is_inside` is winding-sensitive and will reject ALL
//   queries for a CW polygon. Callers passing hand-rolled vertex
//   lists should ensure CCW.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <numsim-core/static_indexing.h>

#include "../types.h"
#include "rectangle_bounding.h"
#include "shape_base.h"

namespace rvegen {

template <typename T = double>
class convex_polygon
    : public numsim_core::static_indexing<convex_polygon<T>, shape_base<T>> {
public:
  using value_type = T;
  using point_type = std::array<T, 2>;

  convex_polygon() = default;

  // Vertices in CCW order; at least 3 required for a non-degenerate
  // polygon. Stored as-is; centroid is computed once and cached for
  // O(1) `get_middle_point` and `set_middle_point`.
  explicit convex_polygon(std::vector<point_type> vertices)
      : _vertices{std::move(vertices)} {
    if (_vertices.size() < 3) {
      throw std::runtime_error{
          "convex_polygon: at least 3 vertices required"};
    }
    _centroid = compute_centroid(_vertices);
  }

  // ---- shape_base contract ----------------------------------------------

  [[nodiscard]] T area() const override {
    // Shoelace formula; abs() so the result is robust to winding.
    if (_vertices.size() < 3) return T{0};
    T sum{0};
    const std::size_t n = _vertices.size();
    for (std::size_t k = 0; k < n; ++k) {
      auto const& a = _vertices[k];
      auto const& b = _vertices[(k + 1) % n];
      sum += a[0] * b[1] - b[0] * a[1];
    }
    return std::abs(sum) * T{0.5};
  }

  [[nodiscard]] int dimension() const noexcept override { return 2; }
  // Width of a convex polygon: it is reached with one side lying on the
  // slab, so the smallest, over the sides, of the largest vertex distance
  // from the side's line.
  [[nodiscard]] T min_width() const override {
    const std::size_t n = _vertices.size();
    T width = std::numeric_limits<T>::infinity();
    for (std::size_t k = 0; k < n; ++k) {
      auto const& a = _vertices[k];
      auto const& b = _vertices[(k + 1) % n];
      const T ex = b[0] - a[0], ey = b[1] - a[1];
      const T len = std::hypot(ex, ey);
      if (!(len > T{0})) continue;
      T far = T{0};
      for (auto const& v : _vertices)
        far = std::max(far, std::abs(ex * (v[1] - a[1]) - ey * (v[0] - a[0])));
      width = std::min(width, far / len);
    }
    return std::isfinite(width) ? width : T{0};
  }
  [[nodiscard]] T volume() const override { return T{0}; }

  [[nodiscard]] std::array<T, 3> max_expansion() const override {
    // Per-axis largest one-sided extent from get_middle_point() (the
    // area centroid), so middle +/- max_expansion() always encloses the
    // polygon even when the centroid is off the AABB centre (#79).
    if (_vertices.empty()) return {T{0}, T{0}, T{0}};
    auto [mn, mx] = extents();
    return {std::max(mx[0] - _centroid[0], _centroid[0] - mn[0]),
            std::max(mx[1] - _centroid[1], _centroid[1] - mn[1]), T{0}};
  }

  [[nodiscard]] std::array<T, 3> get_middle_point() const override {
    return {_centroid[0], _centroid[1], T{0}};
  }
  void set_middle_point(std::array<T, 3> middle_point) override {
    const T dx = middle_point[0] - _centroid[0];
    const T dy = middle_point[1] - _centroid[1];
    for (auto& v : _vertices) { v[0] += dx; v[1] += dy; }
    _centroid = {middle_point[0], middle_point[1]};
  }

  [[nodiscard]] bool is_inside(std::array<T, 3> const& point) const override {
    // For a CCW convex polygon, a point is inside iff it lies on the
    // left of (or on) every edge. The cross product (b - a) × (p - a)
    // must be ≥ 0 for every consecutive pair (a, b). Tolerance for
    // on-edge classification is scale-relative: the cross product has
    // dimension length², so the tolerance is eps·1024·|edge|·(|edge| +
    // |p - a|) (#79) and behaves identically at 1e-6 m and 1e3 m.
    if (_vertices.size() < 3) return false;
    const std::size_t n = _vertices.size();
    for (std::size_t k = 0; k < n; ++k) {
      auto const& a = _vertices[k];
      auto const& b = _vertices[(k + 1) % n];
      const T ex = b[0] - a[0], ey = b[1] - a[1];
      const T px = point[0] - a[0], py = point[1] - a[1];
      const T cross = ex * py - ey * px;
      const T e_len = std::sqrt(ex * ex + ey * ey);
      const T p_len = std::sqrt(px * px + py * py);
      const T tol = std::numeric_limits<T>::epsilon() * T{1024} * e_len *
                    (e_len + p_len);
      if (cross < -tol) return false;
    }
    return true;
  }

  void make_bounding_box() override {
    // A default-constructed (placeholder) polygon has no vertices: its
    // box collapses to the origin instead of reading _vertices[0].
    auto [mn, mx] = extents();
    auto box = std::make_unique<rectangle_bounding<T>>();
    box->top_point()    = {mx[0], mx[1], T{0}};
    box->bottom_point() = {mn[0], mn[1], T{0}};
    this->_bounding_box = std::move(box);
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> clone() const override {
    return std::make_unique<convex_polygon<T>>(*this);
  }

  [[nodiscard]] numsim_core::type_id shape_id() const noexcept override {
    return convex_polygon::m_id;
  }

  // ---- convex_polygon accessors -----------------------------------------

  [[nodiscard]] std::vector<point_type> const& vertices() const noexcept {
    return _vertices;
  }
  [[nodiscard]] std::size_t vertex_count() const noexcept {
    return _vertices.size();
  }

private:
  // AABB corners (min, max); both the origin for an empty polygon.
  [[nodiscard]] std::pair<point_type, point_type> extents() const {
    if (_vertices.empty()) return {point_type{}, point_type{}};
    point_type mn = _vertices[0], mx = _vertices[0];
    for (auto const& v : _vertices) {
      mn[0] = std::min(mn[0], v[0]); mx[0] = std::max(mx[0], v[0]);
      mn[1] = std::min(mn[1], v[1]); mx[1] = std::max(mx[1], v[1]);
    }
    return {mn, mx};
  }

  static point_type compute_centroid(std::vector<point_type> const& v) {
    // Use the area-weighted polygon centroid (Wikipedia, "Centroid
    // of a polygon"). For a degenerate polygon (sum of signed
    // sub-areas = 0) fall back to the arithmetic mean of the
    // vertices — robust enough for the practical input cases.
    const std::size_t n = v.size();
    T cx{0}, cy{0}, sa{0};
    for (std::size_t k = 0; k < n; ++k) {
      auto const& a = v[k];
      auto const& b = v[(k + 1) % n];
      const T cross = a[0] * b[1] - b[0] * a[1];
      sa += cross;
      cx += (a[0] + b[0]) * cross;
      cy += (a[1] + b[1]) * cross;
    }
    // Degenerate test is scale-relative: sa has dimension length².
    T lo_x = v[0][0], hi_x = v[0][0], lo_y = v[0][1], hi_y = v[0][1];
    for (auto const& p : v) {
      lo_x = std::min(lo_x, p[0]); hi_x = std::max(hi_x, p[0]);
      lo_y = std::min(lo_y, p[1]); hi_y = std::max(hi_y, p[1]);
    }
    const T ext = std::max(hi_x - lo_x, hi_y - lo_y);
    if (std::abs(sa) <
        std::numeric_limits<T>::epsilon() * T{1024} * ext * ext) {
      // Degenerate — average vertex positions.
      T ax{0}, ay{0};
      for (auto const& p : v) { ax += p[0]; ay += p[1]; }
      return {ax / static_cast<T>(n), ay / static_cast<T>(n)};
    }
    const T factor = T{1} / (T{3} * sa);
    return {cx * factor, cy * factor};
  }

  std::vector<point_type> _vertices;
  point_type _centroid{};
};

} // namespace rvegen
