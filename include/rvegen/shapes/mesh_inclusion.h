#pragma once

// mesh_inclusion — a 3D shape defined by a closed triangle mesh,
// typically loaded from STL via `read_stl_ascii`.
//
// What this header lands today (phase 1 of #9):
//   * `mesh_inclusion<T>` — stores a vector of `gte::Triangle3<T>`.
//   * `is_inside(p)` — ray-cast from p along +x; odd intersection
//     count = inside. Uses `gte::TIQuery<T, Ray3<T>, Triangle3<T>>`.
//   * `volume()` — divergence theorem (signed-tetrahedra sum).
//   * `make_bounding_box()`, `get_middle_point()`,
//     `set_middle_point()` (translate), `clone()`, `shape_id()`.
//   * Direct C++ ctor from a vector of triangles.
//
// Out of scope here, ships in follow-up PRs against #9:
//   * Binary STL + PLY readers (ASCII STL only today).
//   * `mesh_inclusion_input` — JSON-driven input that loads a file +
//     applies translation/rotation/scale per shape, registered into
//     the input pipeline.
//   * `to_mesh(mesh_inclusion)` — VTK rendering for Tessera (it's the
//     identity, but needs the dispatch entry).
//   * Robust ray-casting on degenerate edges (today we cast along +x;
//     a coplanar ray-edge intersection biases the parity. Real
//     consumers should ensure their meshes are well-formed.)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <numsim-core/static_indexing.h>

#include <Mathematics/IntrRay3Triangle3.h>
#include <Mathematics/Ray.h>
#include <Mathematics/Triangle.h>
#include <Mathematics/Vector3.h>

#include "../types.h"
#include "box_bounding.h"
#include "shape_base.h"

namespace rvegen {

template <typename T = double>
class mesh_inclusion
    : public numsim_core::static_indexing<mesh_inclusion<T>, shape_base<T>> {
public:
  using value_type = T;
  using triangle_type = gte::Triangle3<T>;

  mesh_inclusion() = default;

  explicit mesh_inclusion(std::vector<triangle_type> triangles)
      : _triangles{std::move(triangles)} {}

  // ---- shape_base contract ----------------------------------------------
  [[nodiscard]] T area() const override { return T{0}; }   // 3D shape

  // Volume via divergence theorem: V = (1/6) Σ (v0 · (v1 × v2)) over
  // outward-oriented triangles. Returns the absolute value so callers
  // get a positive volume regardless of mesh winding.
  [[nodiscard]] T volume() const override {
    T sum = T{0};
    for (auto const& t : _triangles) {
      const auto& v0 = t.v[0];
      const auto& v1 = t.v[1];
      const auto& v2 = t.v[2];
      sum += (v0[0] * (v1[1] * v2[2] - v1[2] * v2[1]) +
              v0[1] * (v1[2] * v2[0] - v1[0] * v2[2]) +
              v0[2] * (v1[0] * v2[1] - v1[1] * v2[0]));
    }
    return std::abs(sum) / T{6};
  }

  [[nodiscard]] std::array<T, 3> max_expansion() const override {
    auto bb = compute_bb_extents();
    return {(bb[1][0] - bb[0][0]) * T{0.5},
            (bb[1][1] - bb[0][1]) * T{0.5},
            (bb[1][2] - bb[0][2]) * T{0.5}};
  }

  [[nodiscard]] std::array<T, 3> get_middle_point() const override {
    auto bb = compute_bb_extents();
    return {(bb[0][0] + bb[1][0]) * T{0.5},
            (bb[0][1] + bb[1][1]) * T{0.5},
            (bb[0][2] + bb[1][2]) * T{0.5}};
  }

  void set_middle_point(std::array<T, 3> middle_point) override {
    const auto cur = get_middle_point();
    const T dx = middle_point[0] - cur[0];
    const T dy = middle_point[1] - cur[1];
    const T dz = middle_point[2] - cur[2];
    for (auto& tri : _triangles) {
      for (int i = 0; i < 3; ++i) {
        tri.v[i][0] += dx;
        tri.v[i][1] += dy;
        tri.v[i][2] += dz;
      }
    }
  }

  // Inside test by ray-casting from p along +x. Uses `gte::TIQuery` for
  // each triangle; counts intersections; odd = inside. O(N) in the
  // triangle count — acceptable for moderate meshes during voxelization;
  // a BVH (gte::BVTreeOfTriangles) is the obvious follow-up.
  [[nodiscard]] bool is_inside(std::array<T, 3> const& point) const override {
    if (_triangles.empty()) return false;
    gte::Ray3<T> ray;
    ray.origin[0] = point[0]; ray.origin[1] = point[1]; ray.origin[2] = point[2];
    // Slightly off-axis direction to dodge the parity bug when the ray
    // grazes a shared edge between two triangles of an axis-aligned face
    // (both triangles report intersection ⇒ count is even ⇒ wrong parity).
    ray.direction[0] = T{1};
    ray.direction[1] = T{0.000721};
    ray.direction[2] = T{0.000349};
    gte::TIQuery<T, gte::Ray3<T>, gte::Triangle3<T>> query;
    int hits = 0;
    for (auto const& tri : _triangles) {
      if (query(ray, tri).intersect) ++hits;
    }
    return (hits % 2) == 1;
  }

  void make_bounding_box() override {
    auto bb = compute_bb_extents();
    auto box = std::make_unique<box_bounding<T>>();
    box->top_point()    = bb[1];
    box->bottom_point() = bb[0];
    this->_bounding_box = std::move(box);
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> clone() const override {
    return std::make_unique<mesh_inclusion<T>>(*this);
  }

  [[nodiscard]] numsim_core::type_id shape_id() const noexcept override {
    return mesh_inclusion::m_id;
  }

  // ---- accessors --------------------------------------------------------
  [[nodiscard]] std::vector<triangle_type> const& triangles() const noexcept {
    return _triangles;
  }
  [[nodiscard]] std::size_t triangle_count() const noexcept {
    return _triangles.size();
  }

private:
  [[nodiscard]] std::array<std::array<T, 3>, 2> compute_bb_extents() const {
    if (_triangles.empty()) {
      return {{{T{0}, T{0}, T{0}}, {T{0}, T{0}, T{0}}}};
    }
    std::array<T, 3> mn{_triangles[0].v[0][0], _triangles[0].v[0][1],
                        _triangles[0].v[0][2]};
    std::array<T, 3> mx = mn;
    for (auto const& tri : _triangles) {
      for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
          mn[k] = std::min(mn[k], tri.v[i][k]);
          mx[k] = std::max(mx[k], tri.v[i][k]);
        }
      }
    }
    return {{mn, mx}};
  }

  std::vector<triangle_type> _triangles;
};

} // namespace rvegen
