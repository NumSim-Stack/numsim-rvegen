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
//     mesh via the divergence theorem and cached.
//   * `make_bounding_box()`, `max_expansion()`, `get_middle_point()`,
//     `set_middle_point()` (translate via offset; O(1) — the base
//     AABB is cached), `clone()`, `shape_id()`.
//   * Direct C++ ctor from a vector of triangles (wraps in shared_ptr)
//     OR from an existing shared_ptr (zero-copy unless re-orienting).
//   * Construction validates the mesh: finite vertices, closed and
//     manifold (each welded edge shared by exactly two triangles),
//     orientable. Inconsistently wound triangles are re-oriented so
//     `volume()` is correct for CAD output with mixed winding; an
//     open / non-manifold / non-finite mesh throws std::runtime_error.
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
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
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

namespace detail {

// Validate a triangle soup as a closed, orientable surface and compute
// the per-triangle flips (swap v[1] <-> v[2]) that give it consistent
// OUTWARD winding. Throws std::runtime_error when:
//   * any vertex coordinate is NaN / inf;
//   * the mesh is open or non-manifold — after welding bit-identical
//     vertices, some undirected edge is not shared by exactly two
//     (non-degenerate) triangles;
//   * the surface is non-orientable (no consistent winding exists).
//
// Algorithm: weld vertices, collect the three undirected edges of each
// non-degenerate triangle, sort, require groups of exactly two, then
// BFS over the resulting face adjacency so that every shared edge is
// traversed in opposite directions by its two faces. Each connected
// component is then flipped as a whole so its signed volume is
// positive — or negative if it is nested inside an odd number of other
// components (a cavity), matching the parity semantics of `is_inside`.
// Degenerate triangles (two welded corners coincide) carry no area and
// are left out of the topology check and never flipped.
//
// Returns an empty vector for an empty mesh.
template <typename T>
[[nodiscard]] std::vector<char> mesh_inclusion_orientation_flips(
    std::vector<gte::Triangle3<T>> const& tris) {
  const std::size_t m = tris.size();
  if (m == 0) return {};

  for (std::size_t f = 0; f < m; ++f) {
    for (int c = 0; c < 3; ++c) {
      for (int k = 0; k < 3; ++k) {
        if (!std::isfinite(tris[f].v[c][k])) {
          throw std::runtime_error{
              "mesh_inclusion: triangle " + std::to_string(f) +
              " has a non-finite (NaN/inf) vertex coordinate"};
        }
      }
    }
  }

  // Weld bit-identical vertices into ids.
  std::map<std::array<T, 3>, std::size_t> ids;
  std::vector<gte::Vector3<T>> pos;
  std::vector<std::array<std::size_t, 3>> fv(m);
  for (std::size_t f = 0; f < m; ++f) {
    for (int c = 0; c < 3; ++c) {
      auto const& v = tris[f].v[c];
      auto [it, inserted] = ids.try_emplace({v[0], v[1], v[2]}, pos.size());
      if (inserted) pos.push_back(v);
      fv[f][c] = it->second;
    }
  }

  struct half_edge {
    std::size_t lo, hi, face;
    bool forward;   // face traverses lo -> hi
  };
  std::vector<half_edge> halves;
  halves.reserve(3 * m);
  std::vector<char> degenerate(m, 0);
  for (std::size_t f = 0; f < m; ++f) {
    auto const& t = fv[f];
    if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) {
      degenerate[f] = 1;
      continue;
    }
    for (int k = 0; k < 3; ++k) {
      const auto u = t[k];
      const auto w = t[(k + 1) % 3];
      halves.push_back({std::min(u, w), std::max(u, w), f, u < w});
    }
  }
  std::sort(halves.begin(), halves.end(),
            [](half_edge const& a, half_edge const& b) {
              return a.lo != b.lo ? a.lo < b.lo : a.hi < b.hi;
            });

  auto fmt_point = [&](std::size_t id) {
    return "(" + std::to_string(pos[id][0]) + ", " +
           std::to_string(pos[id][1]) + ", " +
           std::to_string(pos[id][2]) + ")";
  };

  // Every non-degenerate face has exactly three neighbours (one per
  // edge) once each edge is known to be shared by exactly two faces.
  // `same` = both faces traverse the edge in the same direction, i.e.
  // exactly one of them must be flipped.
  struct neighbour { std::size_t face; bool same; };
  std::vector<std::array<neighbour, 3>> adj(m);
  std::vector<unsigned char> n_adj(m, 0);
  for (std::size_t i = 0; i < halves.size();) {
    std::size_t j = i + 1;
    while (j < halves.size() && halves[j].lo == halves[i].lo &&
           halves[j].hi == halves[i].hi) {
      ++j;
    }
    if (j - i != 2) {
      throw std::runtime_error{
          "mesh_inclusion: mesh is not closed (or not manifold): edge " +
          fmt_point(halves[i].lo) + " - " + fmt_point(halves[i].hi) +
          " is shared by " + std::to_string(j - i) +
          " triangle(s), expected exactly 2"};
    }
    auto const& a = halves[i];
    auto const& b = halves[i + 1];
    const bool same = a.forward == b.forward;
    adj[a.face][n_adj[a.face]++] = {b.face, same};
    adj[b.face][n_adj[b.face]++] = {a.face, same};
    i = j;
  }

  // BFS: propagate a consistent winding through each component.
  constexpr std::size_t unvisited = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> comp(m, unvisited);
  std::vector<char> flip(m, 0);
  std::vector<std::size_t> seeds;
  std::vector<std::size_t> queue;
  for (std::size_t s = 0; s < m; ++s) {
    if (degenerate[s] || comp[s] != unvisited) continue;
    const std::size_t c = seeds.size();
    seeds.push_back(s);
    comp[s] = c;
    queue.assign(1, s);
    while (!queue.empty()) {
      const auto f = queue.back();
      queue.pop_back();
      for (auto const& nb : adj[f]) {
        const char want = static_cast<char>(flip[f] ^ (nb.same ? 1 : 0));
        if (comp[nb.face] == unvisited) {
          comp[nb.face] = c;
          flip[nb.face] = want;
          queue.push_back(nb.face);
        } else if (flip[nb.face] != want) {
          throw std::runtime_error{
              "mesh_inclusion: mesh is not orientable (no consistent "
              "triangle winding exists)"};
        }
      }
    }
  }
  const std::size_t n_comp = seeds.size();

  // Per-component signed volume and AABB under the current flips.
  std::vector<T> comp_vol(n_comp, T{0});
  std::vector<std::array<T, 3>> bb_min(n_comp), bb_max(n_comp);
  for (std::size_t c = 0; c < n_comp; ++c) {
    for (int k = 0; k < 3; ++k) {
      bb_min[c][k] = std::numeric_limits<T>::max();
      bb_max[c][k] = std::numeric_limits<T>::lowest();
    }
  }
  for (std::size_t f = 0; f < m; ++f) {
    if (degenerate[f]) continue;
    const auto c = comp[f];
    auto const& t = tris[f];
    const T dv = gte::DotCross(t.v[0], t.v[1], t.v[2]);
    comp_vol[c] += flip[f] ? -dv : dv;
    for (int i = 0; i < 3; ++i) {
      for (int k = 0; k < 3; ++k) {
        bb_min[c][k] = std::min(bb_min[c][k], t.v[i][k]);
        bb_max[c][k] = std::max(bb_max[c][k], t.v[i][k]);
      }
    }
  }

  // Desired sign per component: + for a solid, - for a cavity (nested
  // inside an odd number of other components). Nesting depth is found
  // by casting the same ray `is_inside` uses from one of the
  // component's vertices against the other components' triangles,
  // pre-filtered by AABB.
  std::vector<char> cavity(n_comp, 0);
  if (n_comp > 1) {
    gte::TIQuery<T, gte::Ray3<T>, gte::Triangle3<T>> query;
    for (std::size_t c = 0; c < n_comp; ++c) {
      const auto& p = tris[seeds[c]].v[0];
      gte::Ray3<T> ray;
      ray.origin = p;
      ray.direction[0] = T{1};
      ray.direction[1] = T{0.000721};
      ray.direction[2] = T{0.000349};
      std::vector<char> candidate(n_comp, 0);
      bool any = false;
      for (std::size_t o = 0; o < n_comp; ++o) {
        if (o == c) continue;
        bool in_box = true;
        for (int k = 0; k < 3; ++k) {
          in_box = in_box && p[k] >= bb_min[o][k] && p[k] <= bb_max[o][k];
        }
        candidate[o] = in_box ? 1 : 0;
        any = any || in_box;
      }
      if (!any) continue;
      int hits = 0;
      for (std::size_t f = 0; f < m; ++f) {
        if (degenerate[f] || !candidate[comp[f]]) continue;
        if (query(ray, tris[f]).intersect) ++hits;
      }
      cavity[c] = static_cast<char>(hits % 2);
    }
  }

  for (std::size_t f = 0; f < m; ++f) {
    if (degenerate[f]) continue;
    const auto c = comp[f];
    const bool positive = comp_vol[c] >= T{0};
    if (positive == static_cast<bool>(cavity[c])) flip[f] ^= 1;
  }
  return flip;
}

} // namespace detail

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
  // STL/PLY readers and by tests. The mesh is validated (finite,
  // closed, manifold, orientable — see
  // `detail::mesh_inclusion_orientation_flips`) and inconsistently
  // wound triangles are re-oriented in place before sharing. Throws
  // std::runtime_error on an invalid mesh.
  explicit mesh_inclusion(triangle_list triangles) {
    apply_flips(triangles, detail::mesh_inclusion_orientation_flips(triangles));
    _base = std::make_shared<triangle_list const>(std::move(triangles));
    init_cache();
  }

  // Shares an existing triangle list. The list is validated like the
  // by-value ctor; if any triangle needs re-orienting, a corrected
  // private copy is made (the shared list itself is never mutated).
  // This ctor costs O(M log M) — for the load-once-place-many case,
  // build ONE mesh_inclusion and copy / clone() it per placement
  // (refcount bump + cached AABB/volume; no validation), as
  // `mesh_inclusion_input` does.
  explicit mesh_inclusion(shared_triangles base) : _base{std::move(base)} {
    if (_base && !_base->empty()) {
      const auto flips = detail::mesh_inclusion_orientation_flips(*_base);
      if (std::find(flips.begin(), flips.end(), char{1}) != flips.end()) {
        triangle_list fixed = *_base;
        apply_flips(fixed, flips);
        _base = std::make_shared<triangle_list const>(std::move(fixed));
      }
    }
    init_cache();
  }

  // ---- shape_base contract ----------------------------------------------
  [[nodiscard]] T area() const override { return T{0}; }   // 3D shape

  // Volume is translate-invariant — computed once at construction via
  // the divergence theorem over the validated, consistently
  // outward-wound base mesh (cavities contribute negatively).
  [[nodiscard]] T volume() const override { return _volume; }

  [[nodiscard]] std::array<T, 3> max_expansion() const override {
    return {(_base_max[0] - _base_min[0]) * T{0.5},
            (_base_max[1] - _base_min[1]) * T{0.5},
            (_base_max[2] - _base_min[2]) * T{0.5}};
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
    auto box = std::make_unique<box_bounding<T>>();
    for (int k = 0; k < 3; ++k) {
      box->top_point()[k]    = _base_max[k] + _offset[k];
      box->bottom_point()[k] = _base_min[k] + _offset[k];
    }
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
  static void apply_flips(triangle_list& tris, std::vector<char> const& flips) {
    for (std::size_t f = 0; f < flips.size(); ++f) {
      if (flips[f]) std::swap(tris[f].v[1], tris[f].v[2]);
    }
  }

  // Cache the untranslated base AABB, its centre and the volume. The
  // base is shared and immutable, so these never change; copies /
  // clones carry them along, making max_expansion(),
  // make_bounding_box() and volume() O(1). The AABB centre is the
  // reference point for the offset (so set_middle_point and
  // get_middle_point have stable semantics regardless of how the mesh
  // was originally positioned in the source file).
  void init_cache() {
    if (!_base || _base->empty()) return;
    auto const& tris = *_base;
    _base_min = {tris[0].v[0][0], tris[0].v[0][1], tris[0].v[0][2]};
    _base_max = _base_min;
    T sum = T{0};
    for (auto const& tri : tris) {
      for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
          _base_min[k] = std::min(_base_min[k], tri.v[i][k]);
          _base_max[k] = std::max(_base_max[k], tri.v[i][k]);
        }
      }
      sum += gte::DotCross(tri.v[0], tri.v[1], tri.v[2]);
    }
    for (int k = 0; k < 3; ++k) {
      _base_centre[k] = (_base_min[k] + _base_max[k]) * T{0.5};
    }
    _volume = sum / T{6};
  }

  shared_triangles _base;
  std::array<T, 3> _base_min{};
  std::array<T, 3> _base_max{};
  std::array<T, 3> _base_centre{};
  std::array<T, 3> _offset{};
  T _volume{0};
};

} // namespace rvegen
