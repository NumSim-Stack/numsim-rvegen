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
#include <tuple>
#include <utility>
#include <vector>

#include <numsim-core/static_indexing.h>

#include <Mathematics/DistPointTriangle.h>
#include <Mathematics/IntrRay3Triangle3.h>
#include <Mathematics/Ray.h>
#include <Mathematics/Triangle.h>
#include <Mathematics/Vector3.h>

#include "../types.h"
#include "box_bounding.h"
#include "shape_base.h"

namespace rvegen {

namespace detail {

// Uniform-grid cell index of coordinate `x` along an axis spanning
// [lo, hi] split into `n` cells. Monotone in `x` and clamped, so a
// point inside an AABB always lands in one of the AABB's cells.
template <typename T>
[[nodiscard]] std::size_t mesh_grid_cell(T x, T lo, T hi, std::size_t n) {
  const T ext = hi - lo;
  if (!(ext > T{0})) return 0;
  const T r = (x - lo) / ext * static_cast<T>(n);
  if (!(r > T{0})) return 0;
  return std::min(static_cast<std::size_t>(r), n - 1);
}

// Slightly off-axis +x ray direction shared by `mesh_inclusion::is_inside`
// and the component interpenetration check. The tiny y/z slope dodges the
// parity bug when the ray grazes a shared edge of an axis-aligned face.
inline constexpr double mesh_ray_slope_y = 0.000721;
inline constexpr double mesh_ray_slope_z = 0.000349;

// Candidate-triangle lookup for +x rays (direction (1, sy, sz)) cast from
// points inside the AABB of a triangle set. Triangles are binned in a
// uniform 2D grid over (y, z). A ray from origin o reaches y in
// [o.y, o.y + sy * extent_x] (and likewise z), so a triangle can only be
// hit if o.y lies in [ymin - sy * extent_x, ymax]; registering each
// triangle over that padded range lets a query read ONE bin and still see
// every triangle the ray could hit, with no duplicates. Pure pre-filter:
// the exact intersection test stays with the caller.
template <typename T>
class mesh_ray_grid {
public:
  mesh_ray_grid(std::vector<gte::Triangle3<T>> const& tris,
                std::vector<std::size_t> const& ids,
                std::array<T, 3> const& lo, std::array<T, 3> const& hi)
      : _lo{lo}, _hi{hi} {
    const std::size_t n = ids.size();
    _r = std::clamp<std::size_t>(
        static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n)))),
        1, 256);
    const T ext = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
    const T pad = ext * T{1e-6};
    const T dy = static_cast<T>(mesh_ray_slope_y) * (hi[0] - lo[0]) * T{1.01} + pad;
    const T dz = static_cast<T>(mesh_ray_slope_z) * (hi[0] - lo[0]) * T{1.01} + pad;
    struct range { std::size_t y0, y1, z0, z1; };
    std::vector<range> rg(n);
    // A mesh that is very long along x relative to its y/z size makes the
    // padded footprints cover most bins; then the grid would not pay off
    // (and would blow up memory), so `usable()` is false.
    const std::size_t budget = 16 * n + 1024;
    std::size_t total = 0;
    for (std::size_t i = 0; i < n; ++i) {
      auto const& t = tris[ids[i]];
      std::array<T, 3> mn = {t.v[0][0], t.v[0][1], t.v[0][2]}, mx = mn;
      for (int c = 1; c < 3; ++c) {
        for (int k = 0; k < 3; ++k) {
          mn[k] = std::min(mn[k], t.v[c][k]);
          mx[k] = std::max(mx[k], t.v[c][k]);
        }
      }
      rg[i] = {mesh_grid_cell(mn[1] - dy, lo[1], hi[1], _r),
               mesh_grid_cell(mx[1] + pad, lo[1], hi[1], _r),
               mesh_grid_cell(mn[2] - dz, lo[2], hi[2], _r),
               mesh_grid_cell(mx[2] + pad, lo[2], hi[2], _r)};
      total += (rg[i].y1 - rg[i].y0 + 1) * (rg[i].z1 - rg[i].z0 + 1);
      if (total > budget) return;
    }
    _usable = true;
    _start.assign(_r * _r + 1, 0);
    for (std::size_t i = 0; i < n; ++i) {
      for (auto y = rg[i].y0; y <= rg[i].y1; ++y)
        for (auto z = rg[i].z0; z <= rg[i].z1; ++z) ++_start[y * _r + z + 1];
    }
    for (std::size_t b = 0; b < _r * _r; ++b) _start[b + 1] += _start[b];
    _ids.resize(_start.back());
    std::vector<std::size_t> cursor(_start.begin(), _start.end() - 1);
    for (std::size_t i = 0; i < n; ++i) {
      for (auto y = rg[i].y0; y <= rg[i].y1; ++y)
        for (auto z = rg[i].z0; z <= rg[i].z1; ++z)
          _ids[cursor[y * _r + z]++] = ids[i];
    }
  }

  [[nodiscard]] bool usable() const noexcept { return _usable; }

  // Triangle ids that a +x ray from (any x, y, z) may hit.
  [[nodiscard]] std::pair<std::size_t const*, std::size_t const*>
  candidates(T y, T z) const {
    const auto b = mesh_grid_cell(y, _lo[1], _hi[1], _r) * _r +
                   mesh_grid_cell(z, _lo[2], _hi[2], _r);
    return {_ids.data() + _start[b], _ids.data() + _start[b + 1]};
  }

private:
  std::array<T, 3> _lo, _hi;
  std::size_t _r{1};
  bool _usable{false};
  std::vector<std::size_t> _start, _ids;
};

// Reject components that interpenetrate. For every pair of components
// whose AABBs overlap, each vertex of one is classified against the
// other as inside / outside / on-surface (within 1e-7 of the container's
// largest AABB extent). The pair is rejected when it has vertices of
// both kinds inside and outside: fully disjoint and fully nested pairs
// pass, and components that merely touch (the shared vertices / edges
// classify as on-surface and are ignored) keep working. Known limit:
// crossing bodies none of whose vertices lie inside the other (a "+"
// of two slabs) are not detected.
template <typename T>
void mesh_inclusion_reject_interpenetration(
    std::vector<gte::Triangle3<T>> const& tris,
    std::vector<std::size_t> const& comp,
    std::vector<char> const& degenerate, std::size_t n_comp,
    std::vector<std::array<T, 3>> const& bb_min,
    std::vector<std::array<T, 3>> const& bb_max) {
  if (n_comp < 2) return;
  const std::size_t m = tris.size();

  // Sweep along x for overlapping AABB pairs.
  std::vector<std::size_t> order(n_comp);
  for (std::size_t c = 0; c < n_comp; ++c) order[c] = c;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return bb_min[a][0] < bb_min[b][0];
  });
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  for (std::size_t i = 0; i < n_comp; ++i) {
    const auto a = order[i];
    for (std::size_t j = i + 1; j < n_comp; ++j) {
      const auto b = order[j];
      if (bb_min[b][0] > bb_max[a][0]) break;
      bool overlap = true;
      for (int k = 1; k < 3; ++k) {
        if (bb_min[b][k] > bb_max[a][k] || bb_min[a][k] > bb_max[b][k]) {
          overlap = false;
          break;
        }
      }
      if (overlap) pairs.emplace_back(std::min(a, b), std::max(a, b));
    }
  }
  if (pairs.empty()) return;

  // Faces per component (CSR).
  std::vector<std::size_t> start(n_comp + 1, 0);
  for (std::size_t f = 0; f < m; ++f) {
    if (!degenerate[f]) ++start[comp[f] + 1];
  }
  for (std::size_t c = 0; c < n_comp; ++c) start[c + 1] += start[c];
  std::vector<std::size_t> faces(start[n_comp]);
  {
    std::vector<std::size_t> cursor(start.begin(), start.end() - 1);
    for (std::size_t f = 0; f < m; ++f) {
      if (!degenerate[f]) faces[cursor[comp[f]]++] = f;
    }
  }

  std::vector<std::unique_ptr<mesh_ray_grid<T>>> grids(n_comp);
  gte::TIQuery<T, gte::Ray3<T>, gte::Triangle3<T>> query;
  gte::DCPQuery<T, gte::Vector3<T>, gte::Triangle3<T>> dist_query;
  std::vector<std::size_t> own;

  // true when some vertex of `a` is inside `b` and another is outside.
  auto mixed = [&](std::size_t a, std::size_t b) {
    std::vector<std::size_t> b_faces(faces.begin() + start[b],
                                     faces.begin() + start[b + 1]);
    if (!grids[b] && b_faces.size() > 32) {
      grids[b] = std::make_unique<mesh_ray_grid<T>>(tris, b_faces, bb_min[b], bb_max[b]);
    }
    auto const* grid = (grids[b] && grids[b]->usable()) ? grids[b].get() : nullptr;
    T extent = T{0};
    for (int k = 0; k < 3; ++k) extent = std::max(extent, bb_max[b][k] - bb_min[b][k]);
    const T eps2 = (extent * T{1e-7}) * (extent * T{1e-7});
    gte::Ray3<T> ray;
    ray.direction = {T{1}, static_cast<T>(mesh_ray_slope_y),
                     static_cast<T>(mesh_ray_slope_z)};
    bool any_in = false, any_out = false;
    for (std::size_t i = start[a]; i < start[a + 1] && !(any_in && any_out); ++i) {
      auto const& t = tris[faces[i]];
      for (int c = 0; c < 3; ++c) {
        auto const& p = t.v[c];
        bool in_box = true;
        for (int k = 0; k < 3; ++k) {
          if (p[k] < bb_min[b][k] || p[k] > bb_max[b][k]) in_box = false;
        }
        if (!in_box) { any_out = true; continue; }
        ray.origin = p;
        std::size_t const* first = b_faces.data();
        std::size_t const* last = first + b_faces.size();
        if (grid) std::tie(first, last) = grid->candidates(p[1], p[2]);
        int hits = 0;
        bool on_surface = false;
        for (auto it = first; it != last; ++it) {
          if (dist_query(p, tris[*it]).sqrDistance <= eps2) { on_surface = true; break; }
          if (query(ray, tris[*it]).intersect) ++hits;
        }
        if (on_surface) continue;
        (hits % 2 ? any_in : any_out) = true;
      }
    }
    return any_in && any_out;
  };

  for (auto const& [a, b] : pairs) {
    if (mixed(a, b) || mixed(b, a)) {
      throw std::runtime_error{
          "mesh_inclusion: connected components " + std::to_string(a) +
          " and " + std::to_string(b) +
          " interpenetrate (some vertices of one lie inside the other, "
          "others outside); components must be disjoint or fully nested"};
    }
  }
}

// For each connected component c of a closed mesh, decide whether its
// probe point `probes[c]` (a point on c's own surface, away from its
// edges) lies inside an odd number of OTHER components — i.e. whether
// c bounds a cavity. Returns one parity flag per component.
//
// Cost is kept near-linear for meshes with many components:
//   * Candidate containers are found by point-stabbing a uniform 3D
//     grid of component AABBs (components covering many cells go to a
//     short "large" list checked for every probe).
//   * Probes are grouped per candidate container. Each probe is tested
//     with three generic ray directions and a majority vote, so a ray
//     that grazes an edge or vertex (miscounting the parity) is
//     outvoted. For a container with many triangles and many probes,
//     its triangles are binned in a 2D grid in the plane perpendicular
//     to each ray direction, and a probe only tests its own bin.
template <typename T>
[[nodiscard]] std::vector<char> mesh_inclusion_nesting_parity(
    std::vector<gte::Triangle3<T>> const& tris,
    std::vector<std::size_t> const& comp,
    std::vector<char> const& degenerate, std::size_t n_comp,
    std::vector<std::array<T, 3>> const& bb_min,
    std::vector<std::array<T, 3>> const& bb_max,
    std::vector<gte::Vector3<T>> const& probes) {
  const std::size_t m = tris.size();

  // Faces of each component, CSR layout.
  std::vector<std::size_t> start(n_comp + 1, 0);
  for (std::size_t f = 0; f < m; ++f) {
    if (!degenerate[f]) ++start[comp[f] + 1];
  }
  for (std::size_t c = 0; c < n_comp; ++c) start[c + 1] += start[c];
  std::vector<std::size_t> faces(start[n_comp]);
  {
    std::vector<std::size_t> cursor(start.begin(), start.end() - 1);
    for (std::size_t f = 0; f < m; ++f) {
      if (!degenerate[f]) faces[cursor[comp[f]]++] = f;
    }
  }

  auto box_contains = [&](std::size_t o, gte::Vector3<T> const& p) {
    for (int k = 0; k < 3; ++k) {
      if (p[k] < bb_min[o][k] || p[k] > bb_max[o][k]) return false;
    }
    return true;
  };

  // ---- Step A: (container, probe) candidate pairs via an AABB grid.
  std::array<T, 3> gmin = bb_min[0], gmax = bb_max[0];
  for (std::size_t c = 1; c < n_comp; ++c) {
    for (int k = 0; k < 3; ++k) {
      gmin[k] = std::min(gmin[k], bb_min[c][k]);
      gmax[k] = std::max(gmax[k], bb_max[c][k]);
    }
  }
  // Cell size per axis ~ the median component extent along that axis,
  // so a typical component covers a handful of cells whatever the
  // layout (a long row of parts, a slab, a cloud). The total cell
  // count is then scaled down to at most ~4 cells per component.
  std::array<std::size_t, 3> res{1, 1, 1};
  {
    std::vector<T> ext(n_comp);
    std::array<double, 3> r{};
    for (int k = 0; k < 3; ++k) {
      for (std::size_t c = 0; c < n_comp; ++c) ext[c] = bb_max[c][k] - bb_min[c][k];
      std::nth_element(ext.begin(), ext.begin() + n_comp / 2, ext.end());
      const double med = static_cast<double>(ext[n_comp / 2]);
      const double span = static_cast<double>(gmax[k] - gmin[k]);
      r[k] = (med > 0.0 && span > 0.0) ? std::ceil(span / med) : 1.0;
      r[k] = std::clamp(r[k], 1.0, static_cast<double>(4 * n_comp + 64));
    }
    const double budget = static_cast<double>(4 * n_comp + 64);
    for (int iter = 0; iter < 8 && r[0] * r[1] * r[2] > budget; ++iter) {
      const double f = std::cbrt(budget / (r[0] * r[1] * r[2]));
      for (auto& rk : r) rk = std::max(1.0, std::floor(rk * f));
    }
    for (int k = 0; k < 3; ++k) res[k] = static_cast<std::size_t>(r[k]);
    // If clamping to 1 left the product over budget, cap the largest axis.
    while (res[0] * res[1] * res[2] > static_cast<std::size_t>(budget)) {
      auto& big = *std::max_element(res.begin(), res.end());
      big = std::max<std::size_t>(1, big / 2);
    }
  }
  auto cell3 = [&](std::array<std::size_t, 3> const& i) {
    return (i[0] * res[1] + i[1]) * res[2] + i[2];
  };
  std::vector<std::vector<std::size_t>> cells(res[0] * res[1] * res[2]);
  std::vector<std::size_t> large;
  constexpr std::size_t large_cells = 512;
  for (std::size_t o = 0; o < n_comp; ++o) {
    std::array<std::size_t, 3> lo{}, hi{};
    std::size_t count = 1;
    for (int k = 0; k < 3; ++k) {
      lo[k] = mesh_grid_cell(bb_min[o][k], gmin[k], gmax[k], res[k]);
      hi[k] = mesh_grid_cell(bb_max[o][k], gmin[k], gmax[k], res[k]);
      count *= hi[k] - lo[k] + 1;
    }
    if (count > large_cells) { large.push_back(o); continue; }
    for (auto x = lo[0]; x <= hi[0]; ++x)
      for (auto y = lo[1]; y <= hi[1]; ++y)
        for (auto z = lo[2]; z <= hi[2]; ++z) cells[cell3({x, y, z})].push_back(o);
  }
  std::vector<std::pair<std::size_t, std::size_t>> pairs;   // (container, probe)
  for (std::size_t c = 0; c < n_comp; ++c) {
    auto const& p = probes[c];
    std::array<std::size_t, 3> i{};
    for (int k = 0; k < 3; ++k) i[k] = mesh_grid_cell(p[k], gmin[k], gmax[k], res[k]);
    for (auto const* list : {&cells[cell3(i)], &large}) {
      for (auto o : *list) {
        if (o != c && box_contains(o, p)) pairs.emplace_back(o, c);
      }
    }
  }
  std::sort(pairs.begin(), pairs.end());

  // ---- Step B: parity of each probe against each candidate container.
  //
  // A probe that lies ON the container's surface (a component touching
  // its container) gives an arbitrary ray parity. While testing the
  // first ray direction every triangle a probe could touch is also
  // checked by point-triangle distance; a probe found on the surface
  // is retried with another candidate point of the same component.
  std::array<gte::Vector3<T>, 3> dirs{{
      {T{0.5773}, T{0.4987}, T{0.6467}},
      {T{-0.6321}, T{0.7071}, T{-0.3175}},
      {T{0.2113}, T{-0.8660}, T{0.4532}}}};
  for (auto& d : dirs) gte::Normalize(d);

  // Candidate probes per component: `probes[c]` first, then centroids
  // of faces spread evenly over the component.
  constexpr std::size_t max_candidates = 8;
  auto candidate = [&](std::size_t c, std::size_t k) -> gte::Vector3<T> {
    if (k == 0) return probes[c];
    const std::size_t nfc = start[c + 1] - start[c];
    auto const& t = tris[faces[start[c] + (k * nfc) / max_candidates]];
    return (t.v[0] + t.v[1] + t.v[2]) / T{3};
  };

  gte::TIQuery<T, gte::Ray3<T>, gte::Triangle3<T>> query;
  gte::DCPQuery<T, gte::Vector3<T>, gte::Triangle3<T>> dist_query;
  std::vector<char> parity(n_comp, 0);
  std::vector<int> votes;
  std::vector<char> on_surface;
  std::vector<gte::Vector3<T>> qp;           // probe point per pending query
  std::vector<std::size_t> pending, next_pending, bin_start, bin_faces;
  std::vector<std::array<T, 4>> proj_box;   // per face: umin, umax, wmin, wmax
  for (std::size_t g = 0; g < pairs.size();) {
    const std::size_t o = pairs[g].first;
    std::size_t g_end = g;
    while (g_end < pairs.size() && pairs[g_end].first == o) ++g_end;
    const std::size_t f0 = start[o], nf = start[o + 1] - start[o];
    T extent = T{0};
    for (int k = 0; k < 3; ++k) extent = std::max(extent, bb_max[o][k] - bb_min[o][k]);
    const T surf_eps2 = (extent * T{1e-7}) * (extent * T{1e-7});

    pending.clear();
    for (std::size_t q = g; q < g_end; ++q) pending.push_back(q);
    for (std::size_t round = 0; round < max_candidates && !pending.empty(); ++round) {
    const std::size_t nq = pending.size();
    votes.assign(nq, 0);
    on_surface.assign(nq, 0);
    qp.resize(nq);
    for (std::size_t q = 0; q < nq; ++q) {
      qp[q] = candidate(pairs[pending[q]].second, round);
    }

    for (std::size_t di = 0; di < dirs.size(); ++di) {
      auto const& d = dirs[di];
      gte::Ray3<T> ray;
      ray.direction = d;
      auto hits_in = [&](std::size_t q, std::size_t const* first,
                         std::size_t const* last) {
        int hits = 0;
        for (auto it = first; it != last; ++it) {
          if (query(ray, tris[*it]).intersect) ++hits;
          if (di == 0 && !on_surface[q] &&
              dist_query(qp[q], tris[*it]).sqrDistance <= surf_eps2) {
            on_surface[q] = 1;
          }
        }
        return hits;
      };
      if (nf * nq <= 4096) {
        for (std::size_t q = 0; q < nq; ++q) {
          ray.origin = qp[q];
          votes[q] += hits_in(q, faces.data() + f0, faces.data() + f0 + nf) % 2;
        }
        continue;
      }
      // Bin the container's triangles in the plane perpendicular to d.
      gte::Vector3<T> a{T{1}, T{0}, T{0}};
      if (std::abs(d[0]) > T{0.9}) a = {T{0}, T{1}, T{0}};
      auto u = gte::Cross(d, a);
      gte::Normalize(u);
      const auto w = gte::Cross(d, u);
      proj_box.resize(nf);
      T umin = std::numeric_limits<T>::max(), umax = std::numeric_limits<T>::lowest();
      T wmin = umin, wmax = umax;
      for (std::size_t i = 0; i < nf; ++i) {
        auto const& t = tris[faces[f0 + i]];
        std::array<T, 4> b{std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest(),
                           std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()};
        for (int k = 0; k < 3; ++k) {
          const T pu = gte::Dot(t.v[k], u), pw = gte::Dot(t.v[k], w);
          b[0] = std::min(b[0], pu); b[1] = std::max(b[1], pu);
          b[2] = std::min(b[2], pw); b[3] = std::max(b[3], pw);
        }
        proj_box[i] = b;
        umin = std::min(umin, b[0]); umax = std::max(umax, b[1]);
        wmin = std::min(wmin, b[2]); wmax = std::max(wmax, b[3]);
      }
      const std::size_t r2 = std::clamp<std::size_t>(
          static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(nf)))),
          1, 1024);
      bin_start.assign(r2 * r2 + 1, 0);
      auto for_each_bin = [&](std::array<T, 4> const& b, auto&& fn) {
        const auto u0 = mesh_grid_cell(b[0], umin, umax, r2);
        const auto u1 = mesh_grid_cell(b[1], umin, umax, r2);
        const auto w0 = mesh_grid_cell(b[2], wmin, wmax, r2);
        const auto w1 = mesh_grid_cell(b[3], wmin, wmax, r2);
        for (auto x = u0; x <= u1; ++x)
          for (auto y = w0; y <= w1; ++y) fn(x * r2 + y);
      };
      for (std::size_t i = 0; i < nf; ++i) {
        for_each_bin(proj_box[i], [&](std::size_t b) { ++bin_start[b + 1]; });
      }
      for (std::size_t b = 0; b < r2 * r2; ++b) bin_start[b + 1] += bin_start[b];
      bin_faces.resize(bin_start[r2 * r2]);
      {
        std::vector<std::size_t> cursor(bin_start.begin(), bin_start.end() - 1);
        for (std::size_t i = 0; i < nf; ++i) {
          for_each_bin(proj_box[i], [&](std::size_t b) {
            bin_faces[cursor[b]++] = faces[f0 + i];
          });
        }
      }
      for (std::size_t q = 0; q < nq; ++q) {
        auto const& p = qp[q];
        const T pu = gte::Dot(p, u), pw = gte::Dot(p, w);
        if (pu < umin || pu > umax || pw < wmin || pw > wmax) continue;
        const auto b = mesh_grid_cell(pu, umin, umax, r2) * r2 +
                       mesh_grid_cell(pw, wmin, wmax, r2);
        ray.origin = p;
        votes[q] += hits_in(q, bin_faces.data() + bin_start[b],
                            bin_faces.data() + bin_start[b + 1]) % 2;
      }
    }
    next_pending.clear();
    for (std::size_t q = 0; q < nq; ++q) {
      // Retry an on-surface probe with the next candidate while one is
      // left; the last candidate's vote is used as-is.
      if (on_surface[q] && round + 1 < max_candidates) {
        next_pending.push_back(pending[q]);
      } else if (votes[q] >= 2) {
        parity[pairs[pending[q]].second] ^= 1;
      }
    }
    pending.swap(next_pending);
    }
    g = g_end;
  }
  return parity;
}

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

  // Per-component signed volume, AABB and probe face (the largest-area
  // triangle; its centroid is a point strictly inside the component's
  // surface patch, never on another component's corner) under the
  // current flips.
  std::vector<T> comp_vol(n_comp, T{0});
  std::vector<std::array<T, 3>> bb_min(n_comp), bb_max(n_comp);
  std::vector<T> probe_area(n_comp, T{-1});
  std::vector<std::size_t> probe_face(seeds);
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
    const auto n = gte::Cross(t.v[1] - t.v[0], t.v[2] - t.v[0]);
    const T area2 = gte::Dot(n, n);
    if (area2 > probe_area[c]) { probe_area[c] = area2; probe_face[c] = f; }
    for (int i = 0; i < 3; ++i) {
      for (int k = 0; k < 3; ++k) {
        bb_min[c][k] = std::min(bb_min[c][k], t.v[i][k]);
        bb_max[c][k] = std::max(bb_max[c][k], t.v[i][k]);
      }
    }
  }

  mesh_inclusion_reject_interpenetration<T>(tris, comp, degenerate,
                                                    n_comp, bb_min, bb_max);

  // Desired sign per component: + for a solid, - for a cavity (nested
  // inside an odd number of other components). See
  // `mesh_inclusion_nesting_parity` for how nesting is detected.
  std::vector<char> cavity;
  if (n_comp > 1) {
    std::vector<gte::Vector3<T>> probes(n_comp);
    for (std::size_t c = 0; c < n_comp; ++c) {
      auto const& t = tris[probe_face[c]];
      probes[c] = (t.v[0] + t.v[1] + t.v[2]) / T{3};
    }
    cavity = mesh_inclusion_nesting_parity(tris, comp, degenerate, n_comp,
                                           bb_min, bb_max, probes);
  } else {
    cavity.assign(n_comp, 0);
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
  // std::runtime_error on an invalid mesh (including an empty list).
  explicit mesh_inclusion(triangle_list triangles) {
    if (triangles.empty()) throw_empty();
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
    if (!_base || _base->empty()) throw_empty();
    const auto flips = detail::mesh_inclusion_orientation_flips(*_base);
    if (std::find(flips.begin(), flips.end(), char{1}) != flips.end()) {
      triangle_list fixed = *_base;
      apply_flips(fixed, flips);
      _base = std::make_shared<triangle_list const>(std::move(fixed));
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
    // A point outside the cached base AABB is outside the closed mesh.
    for (int k = 0; k < 3; ++k) {
      if (!(ray.origin[k] >= _base_min[k] && ray.origin[k] <= _base_max[k])) {
        return false;
      }
    }
    // Slightly off-axis direction to dodge the parity bug when the ray
    // grazes a shared edge between two triangles of an axis-aligned face.
    ray.direction[0] = T{1};
    ray.direction[1] = static_cast<T>(detail::mesh_ray_slope_y);
    ray.direction[2] = static_cast<T>(detail::mesh_ray_slope_z);
    gte::TIQuery<T, gte::Ray3<T>, gte::Triangle3<T>> query;
    int hits = 0;
    if (_grid) {
      // Only triangles whose (y, z) footprint the ray can reach.
      auto const [first, last] = _grid->candidates(ray.origin[1], ray.origin[2]);
      for (auto it = first; it != last; ++it) {
        if (query(ray, (*_base)[*it]).intersect) ++hits;
      }
    } else {
      for (auto const& tri : *_base) {
        if (query(ray, tri).intersect) ++hits;
      }
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
  [[noreturn]] static void throw_empty() {
    throw std::runtime_error{
        "mesh_inclusion: empty triangle list (a mesh inclusion needs at "
        "least one closed surface)"};
  }

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
    if (!_base || _base->empty()) throw_empty();
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
    // A closed surface that encloses no volume (e.g. a triangle and
    // its reverse) passes the topology check but is useless as an
    // inclusion: reject |V| <= 1000·eps·L³, L = largest AABB extent.
    const T extent = std::max({_base_max[0] - _base_min[0],
                               _base_max[1] - _base_min[1],
                               _base_max[2] - _base_min[2]});
    const T tol = T{1000} * std::numeric_limits<T>::epsilon() *
                  extent * extent * extent;
    if (!(std::abs(_volume) > tol)) {
      throw std::runtime_error{
          "mesh_inclusion: closed mesh encloses (near-)zero volume (" +
          std::to_string(_volume) + ", bounding-box extent " +
          std::to_string(extent) + ") — degenerate or doubled-up surface"};
    }
    // Large meshes: bin the triangles so is_inside reads one small bucket.
    if (tris.size() > 32) {
      std::vector<std::size_t> ids(tris.size());
      for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = i;
      auto grid = std::make_shared<detail::mesh_ray_grid<T> const>(
          tris, ids, _base_min, _base_max);
      if (grid->usable()) _grid = std::move(grid);
    }
  }

  shared_triangles _base;
  std::array<T, 3> _base_min{};
  std::array<T, 3> _base_max{};
  std::array<T, 3> _base_centre{};
  std::array<T, 3> _offset{};
  T _volume{0};
  std::shared_ptr<detail::mesh_ray_grid<T> const> _grid;   // null for small meshes
};

} // namespace rvegen
