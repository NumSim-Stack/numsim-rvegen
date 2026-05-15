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

  [[nodiscard]] T volume() const override { return T{0}; }

  [[nodiscard]] std::array<T, 3> max_expansion() const override {
    // Largest distance from the centroid to any vertex. Used by
    // only_inside placement to test whether the polygon's bounding
    // radius fits in the RVE box.
    T max_d2{0};
    for (auto const& v : _vertices) {
      const T dx = v[0] - _centroid[0];
      const T dy = v[1] - _centroid[1];
      const T d2 = dx * dx + dy * dy;
      if (d2 > max_d2) max_d2 = d2;
    }
    const T r = std::sqrt(max_d2);
    return {r, r, T{0}};
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
    // on-edge classification matches the Voronoi clip helper.
    if (_vertices.size() < 3) return false;
    const std::size_t n = _vertices.size();
    for (std::size_t k = 0; k < n; ++k) {
      auto const& a = _vertices[k];
      auto const& b = _vertices[(k + 1) % n];
      const T cross = (b[0] - a[0]) * (point[1] - a[1])
                    - (b[1] - a[1]) * (point[0] - a[0]);
      if (cross < -std::numeric_limits<T>::epsilon() * T{1024}) return false;
    }
    return true;
  }

  void make_bounding_box() override {
    T min_x = _vertices[0][0], max_x = _vertices[0][0];
    T min_y = _vertices[0][1], max_y = _vertices[0][1];
    for (auto const& v : _vertices) {
      if (v[0] < min_x) min_x = v[0];
      if (v[0] > max_x) max_x = v[0];
      if (v[1] < min_y) min_y = v[1];
      if (v[1] > max_y) max_y = v[1];
    }
    auto box = std::make_unique<rectangle_bounding<T>>();
    box->top_point()    = {max_x, max_y, T{0}};
    box->bottom_point() = {min_x, min_y, T{0}};
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
    if (std::abs(sa) < std::numeric_limits<T>::epsilon() * T{1024}) {
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
