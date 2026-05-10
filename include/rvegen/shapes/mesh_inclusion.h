#pragma once

// mesh_inclusion — a 3D shape defined by a closed triangle mesh,
// typically loaded from STL via `read_stl_ascii`.
//
// What this header lands today (phase 1 of #9):
//   * `mesh_inclusion<T>` — stores a SHARED base triangle list +
//     a per-shape translation offset. Many placed shapes spawned
//     from the same STL share one triangle vector; per-shape state
//     is just the offset (24 bytes for double) instead of a full
//     triangle copy. Loading 1000 placements of a 10k-triangle mesh
//     allocates the triangles ONCE, not 1000 times.
//   * `is_inside(p)` — ray-cast from `(p − offset)` against the
//     shared base triangles via `gte::TIQuery<T, Ray3<T>,
//     Triangle3<T>>`. Translation is linear, so testing the offset
//     point against base triangles is equivalent to testing the
//     original point against translated triangles.
//   * `volume()` — translate-invariant; computed once over the base
//     mesh via the divergence theorem.
//   * `make_bounding_box()`, `get_middle_point()`,
//     `set_middle_point()` (translate via offset; O(1)), `clone()`,
//     `shape_id()`.
//   * Direct C++ ctor from a vector of triangles (wraps in shared_ptr)
//     OR from an existing shared_ptr (zero-copy).
//
// Why offset + shared base, not per-shape vector copy:
//   The previous design copied the triangle vector into every shape.
//   For a generator that places N copies of the same STL mesh, that's
//   O(N · M) memory and time on the inner loop. Offset-based storage
//   collapses this to O(M) (shared) + O(N) (per-shape offsets).
//   set_middle_point is O(1) instead of O(M).
//
// Out of scope here, ships in follow-up PRs against #9:
//   * Binary STL + PLY readers.
//   * `mesh_inclusion_input` JSON-driven loader (already PR #27).
//   * `to_mesh(mesh_inclusion)` — VTK rendering for Tessera.
//   * Per-shape rotation. Today the offset is translation-only;
//     rotation needs a quaternion or rotation matrix on top.
//   * Robust ray-casting on degenerate edges (today a slightly
//     off-axis direction dodges most parity bugs but pathological
//     inputs can still trip it).

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
  using triangle_list = std::vector<triangle_type>;
  using shared_triangles = std::shared_ptr<triangle_list const>;

  mesh_inclusion() = default;

  // Wraps a freshly-loaded triangle list in a shared_ptr. Used by
  // STL/PLY readers and by tests.
  explicit mesh_inclusion(triangle_list triangles)
      : _base{std::make_shared<triangle_list>(std::move(triangles))},
        _base_centre{compute_base_centre(*_base)} {}

  // Zero-copy ctor for the load-once-place-many case (mesh_inclusion_input).
  // The shared_ptr is captured by value; subsequent mutations to the
  // underlying vector via other shared owners would race, but the
  // input model treats the base as immutable after first load.
  explicit mesh_inclusion(shared_triangles base)
      : _base{std::move(base)},
        _base_centre{_base ? compute_base_centre(*_base)
                           : std::array<T, 3>{}} {}

  // ---- shape_base contract ----------------------------------------------
  [[nodiscard]] T area() const override { return T{0}; }   // 3D shape

  // Volume is translate-invariant — computed once over the base
  // triangle list via the divergence theorem. With consistent face
  // winding the sum is signed; abs at the end makes the result
  // direction-agnostic.
  [[nodiscard]] T volume() const override {
    if (!_base || _base->empty()) return T{0};
    T sum = T{0};
    for (auto const& t : *_base) {
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
    auto bb = compute_translated_bb();
    return {(bb[1][0] - bb[0][0]) * T{0.5},
            (bb[1][1] - bb[0][1]) * T{0.5},
            (bb[1][2] - bb[0][2]) * T{0.5}};
  }

  [[nodiscard]] std::array<T, 3> get_middle_point() const override {
    return {_base_centre[0] + _offset[0],
            _base_centre[1] + _offset[1],
            _base_centre[2] + _offset[2]};
  }

  // O(1) — store the new offset relative to the base's AABB centre.
  // No triangle data is touched.
  void set_middle_point(std::array<T, 3> middle_point) override {
    _offset[0] = middle_point[0] - _base_centre[0];
    _offset[1] = middle_point[1] - _base_centre[1];
    _offset[2] = middle_point[2] - _base_centre[2];
  }

  // Inside test by ray-casting from (point − offset) along a slightly
  // off-axis +x direction. Translation is linear, so testing the
  // offset point against the base mesh is equivalent to testing the
  // original point against the translated mesh — and avoids any
  // per-shape mutation of triangle data.
  [[nodiscard]] bool is_inside(std::array<T, 3> const& point) const override {
    if (!_base || _base->empty()) return false;
    gte::Ray3<T> ray;
    ray.origin[0] = point[0] - _offset[0];
    ray.origin[1] = point[1] - _offset[1];
    ray.origin[2] = point[2] - _offset[2];
    // Slightly off-axis direction to dodge the parity bug when the ray
    // grazes a shared edge between two triangles of an axis-aligned face.
    ray.direction[0] = T{1};
    ray.direction[1] = T{0.000721};
    ray.direction[2] = T{0.000349};
    gte::TIQuery<T, gte::Ray3<T>, gte::Triangle3<T>> query;
    int hits = 0;
    for (auto const& tri : *_base) {
      if (query(ray, tri).intersect) ++hits;
    }
    return (hits % 2) == 1;
  }

  void make_bounding_box() override {
    auto bb = compute_translated_bb();
    auto box = std::make_unique<box_bounding<T>>();
    box->top_point()    = bb[1];
    box->bottom_point() = bb[0];
    this->_bounding_box = std::move(box);
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> clone() const override {
    // Default copy-ctor copies the shared_ptr (refcount bump only) +
    // the offset + base_centre. No triangle copy.
    return std::make_unique<mesh_inclusion<T>>(*this);
  }

  [[nodiscard]] numsim_core::type_id shape_id() const noexcept override {
    return mesh_inclusion::m_id;
  }

  // ---- accessors --------------------------------------------------------
  // Triangles are exposed via the shared base; callers needing the
  // translated form must add `_offset` to each vertex themselves.
  [[nodiscard]] shared_triangles base_triangles() const noexcept {
    return _base;
  }
  [[nodiscard]] std::size_t triangle_count() const noexcept {
    return _base ? _base->size() : 0;
  }
  [[nodiscard]] std::array<T, 3> const& offset() const noexcept {
    return _offset;
  }

private:
  // AABB-centre of the untranslated base mesh — cached at ctor time.
  // Used as the reference point for the offset (so set_middle_point
  // and get_middle_point have stable semantics regardless of how
  // the mesh was originally positioned in the STL file).
  static std::array<T, 3> compute_base_centre(triangle_list const& tris) {
    if (tris.empty()) return {T{0}, T{0}, T{0}};
    std::array<T, 3> mn{tris[0].v[0][0], tris[0].v[0][1], tris[0].v[0][2]};
    std::array<T, 3> mx = mn;
    for (auto const& tri : tris) {
      for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
          mn[k] = std::min(mn[k], tri.v[i][k]);
          mx[k] = std::max(mx[k], tri.v[i][k]);
        }
      }
    }
    return {(mn[0] + mx[0]) * T{0.5},
            (mn[1] + mx[1]) * T{0.5},
            (mn[2] + mx[2]) * T{0.5}};
  }

  [[nodiscard]] std::array<std::array<T, 3>, 2> compute_translated_bb() const {
    if (!_base || _base->empty()) {
      return {{{T{0}, T{0}, T{0}}, {T{0}, T{0}, T{0}}}};
    }
    auto const& tris = *_base;
    std::array<T, 3> mn{tris[0].v[0][0], tris[0].v[0][1], tris[0].v[0][2]};
    std::array<T, 3> mx = mn;
    for (auto const& tri : tris) {
      for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
          mn[k] = std::min(mn[k], tri.v[i][k]);
          mx[k] = std::max(mx[k], tri.v[i][k]);
        }
      }
    }
    for (int k = 0; k < 3; ++k) { mn[k] += _offset[k]; mx[k] += _offset[k]; }
    return {{mn, mx}};
  }

  shared_triangles _base;
  std::array<T, 3> _base_centre{};
  std::array<T, 3> _offset{};
};

} // namespace rvegen
