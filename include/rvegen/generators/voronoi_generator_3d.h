#pragma once

// 3D bounded Voronoi tessellation (#114 phase 2). Given a box
// `[0, Lx] × [0, Ly] × [0, Lz]` and N seed points in it, produces N
// `voronoi_cell` shapes — one per seed — that partition the box.
// Each cell is the convex polytope of points closer to its seed than
// to any other seed, clipped to the domain.
//
// Algorithm: vertex enumeration via brute-force half-space-triplet
// intersection.
//
//   1. Collect 6 + (N - 1) half-spaces for cell i:
//      * 6 from the bounding box (x ≥ 0, x ≤ Lx, ...).
//      * N - 1 perpendicular bisectors of seed i against each other
//        seed (keeping the "closer-to-seed-i" side).
//   2. For each unordered triplet of half-spaces, solve the 3x3
//      linear system to find the plane intersection point. Skip
//      degenerate triplets (parallel or coplanar planes).
//   3. Check the intersection point against ALL half-spaces:
//      if it satisfies every constraint within a small tolerance,
//      it IS a polytope vertex.
//   4. Group vertices by which half-space they lie on → faces.
//      Each face has ≥ 3 vertices; we order them CCW around the
//      half-space's outward normal.
//   5. Construct a `voronoi_cell` from the (vertices, faces) pair.
//
// Complexity: O(N · (N + 5)³) vertex-triplets, each O(N) feasibility
// test ≈ O(N⁵) in the worst case (measured ~N⁴ for random seeds). For
// N ≤ 50 seeds this is well under a second; the constructor rejects more
// than voronoi_in_house_max_seeds(periodic) seeds (see there for timings):
// switch to Voro++ for larger N.
//
// Backends (#70):
//   * `voronoi_backend::in_house` (default) — the enumeration above.
//   * `voronoi_backend::voro_pp` — Voro++ (roughly linear in N). Only
//     available when the build defines RVEGEN_WITH_VORO_PP (CMake
//     option of the same name); requesting it otherwise throws
//     std::runtime_error. Output types and the validation policy are
//     identical for both backends.
//
// Periodic mode (#8): `periodic = true` tessellates the box as a 3-torus.
// Each primary cell is the Voronoi cell of its seed against all 3x3x3 image
// seeds (shifted by -L, 0, +L per axis), with NO box clipping. A cell that
// crosses a box face is therefore returned as ONE whole convex polytope
// that sticks out of [0,L]^3 (same convention as periodic inclusions with
// wraps: the caller sees the cell around its seed, translation images are
// implicit). Cell volumes sum to the box volume, and neighbouring faces
// match across opposite boundaries up to a translation by L. The in-house
// backend prunes the 27N image planes to those that can touch the cell
// (two passes), but stays O(N^5)-ish: use Voro++ for large periodic N.
//
// Weights (#8, Laguerre / power diagram): optional per-seed weights w_i
// [length^2] replace |x-P_i|^2 by |x-P_i|^2 - w_i, so cell volumes can be
// tuned (see grain_size_control.h). A weighted cell may be empty (zero
// volume placeholder) and need not contain its seed.
//
// Numerical robustness (scale-invariant, #79):
//   * Every half-space is normalised to a unit normal, so the 3x3
//     determinant is dimensionless and the feasibility / merge
//     tolerances are distances relative to the shortest domain edge
//     (1e-9 · scale). Domains of 1e-6 m and 1e3 m behave identically.
//   * Bisector offsets are computed as n · midpoint (not from the
//     difference of squared norms), avoiding cancellation for
//     closely spaced seeds.
//   * Coincident vertices (multiple triplets producing the same
//     point at degeneracies) are deduplicated by L2 distance.
//   * Coincident seeds (closer than 1e-9 of the domain size) and seeds
//     outside the domain are rejected with std::runtime_error. Seeds on
//     the domain boundary are valid.
//   * `build()` always returns exactly one shape per seed, in seed
//     order; a cell that collapses numerically is an empty
//     `voronoi_cell` (volume() == 0) placeholder.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "../shapes/periodic_wraps.h"
#include "../shapes/shape_base.h"
#include "../shapes/voronoi_cell.h"
#include "rve_generator_base.h"   // progress_options (cancel hook)
#include "voronoi_backend_voropp.h"
#include "voronoi_seed_validation.h"

namespace rvegen {

// Tessellation backend for voronoi_generator_3d (#70).
enum class voronoi_backend { in_house, voro_pp };

// True iff this build was configured with RVEGEN_WITH_VORO_PP=ON.
[[nodiscard]] constexpr bool voro_pp_available() noexcept {
#ifdef RVEGEN_WITH_VORO_PP
  return true;
#else
  return false;
#endif
}

// Largest seed count accepted by the in-house backend. Measured cost of one
// tessellation of random seeds in a unit box:
//   bounded:  N = 100 ~ 1 s, N = 200 ~ 17 s, N = 300 ~ 85 s (grows ~N^4);
//   periodic: N = 400 ~ 3 s, N = 1600 ~ 13 s, N = 3200 ~ 67 s (the image
//             pruning keeps it roughly N^1.5).
// The caps sit near one minute per build; the grain-size iteration repeats
// the tessellation up to 1000 times, so use the Voro++ backend for more
// seeds.
[[nodiscard]] constexpr std::size_t
voronoi_in_house_max_seeds(bool periodic) noexcept {
  return periodic ? 3000 : 300;
}

// Thrown by build() when progress_options::cancel returned true.
class voronoi_cancelled : public std::runtime_error {
public:
  voronoi_cancelled() : std::runtime_error{"voronoi tessellation cancelled"} {}
};

// Relative tolerance of the volume-sum check that every build() performs:
// |sum of cell volumes - box volume| <= this * box volume.
inline constexpr double voronoi_volume_sum_tol = 1e-6;

namespace detail {

// Throws std::runtime_error (naming the backend and the mismatch) unless the
// cell volumes sum to `box_volume` within voronoi_volume_sum_tol (relative).
template <typename T>
void check_voronoi_volume_sum(
    std::vector<std::unique_ptr<shape_base<T>>> const& cells, T box_volume,
    char const* backend, bool periodic, bool weighted) {
  long double sum = 0;
  for (auto const& c : cells) sum += c->volume();
  const long double box = box_volume;
  const long double rel = std::abs(sum - box) / box;
  if (!(rel <= static_cast<long double>(voronoi_volume_sum_tol))) {
    throw std::runtime_error{
        std::string{"voronoi_generator_3d ("} + backend +
        (periodic ? ", periodic" : "") + (weighted ? ", weighted" : "") +
        "): cell volumes sum to " + std::to_string(static_cast<double>(sum)) +
        " but the box volume is " + std::to_string(static_cast<double>(box)) +
        " (relative mismatch " + std::to_string(static_cast<double>(rel)) +
        ", tolerance " + std::to_string(voronoi_volume_sum_tol) +
        "); the tessellation is not trustworthy"};
  }
}

} // namespace detail

template <typename T = double>
class voronoi_generator_3d {
public:
  using value_type = T;
  using seed_type = std::array<T, 3>;
  using shape_vector = std::vector<std::unique_ptr<shape_base<T>>>;

  voronoi_generator_3d(T Lx, T Ly, T Lz, std::vector<seed_type> seeds,
                       voronoi_backend backend = voronoi_backend::in_house,
                       bool periodic = false, std::vector<T> weights = {})
      : _Lx{Lx}, _Ly{Ly}, _Lz{Lz}, _seeds{std::move(seeds)},
        _backend{backend}, _periodic{periodic}, _weights{std::move(weights)} {
    if (!_weights.empty() && _weights.size() != _seeds.size()) {
      throw std::runtime_error{
          "voronoi_generator_3d: weights must be empty or one per seed"};
    }
    if (backend == voronoi_backend::voro_pp && !voro_pp_available()) {
      throw std::runtime_error{
          "voronoi_generator_3d: Voro++ backend requested but rvegen was "
          "built without RVEGEN_WITH_VORO_PP"};
    }
    const std::array<std::pair<char const*, T>, 3> extents{
        {{"Lx", Lx}, {"Ly", Ly}, {"Lz", Lz}}};
    for (auto const& [name, v] : extents) {
      if (!std::isfinite(v) || !(v > T{0})) {
        throw std::invalid_argument{
            std::string{"voronoi_generator_3d: "} + name +
            " must be finite and positive (got " + std::to_string(v) + ")"};
      }
    }
    if (_seeds.empty()) {
      throw std::runtime_error{
          "voronoi_generator_3d: at least one seed required"};
    }
    for (std::size_t i = 0; i < _seeds.size(); ++i) {
      for (std::size_t k = 0; k < 3; ++k) {
        if (!std::isfinite(_seeds[i][k])) {
          throw std::invalid_argument{
              "voronoi_generator_3d: seeds[" + std::to_string(i) +
              "] has a non-finite coordinate"};
        }
      }
    }
    for (std::size_t i = 0; i < _weights.size(); ++i) {
      if (!std::isfinite(_weights[i])) {
        throw std::invalid_argument{"voronoi_generator_3d: weights[" +
                                    std::to_string(i) + "] is not finite"};
      }
    }
    if (backend == voronoi_backend::in_house &&
        _seeds.size() > voronoi_in_house_max_seeds(periodic)) {
      throw std::invalid_argument{
          "voronoi_generator_3d: " + std::to_string(_seeds.size()) +
          " seeds exceed the limit of " +
          std::to_string(voronoi_in_house_max_seeds(periodic)) +
          (periodic ? " (periodic)" : " (bounded)") +
          " for the in-house backend (its cost grows steeply); use the Voro++ "
          "backend (voronoi_backend::voro_pp, RVEGEN_WITH_VORO_PP=ON)"};
    }
    detail::validate_voronoi_seeds<T, 3>({Lx, Ly, Lz}, _seeds,
                                         "voronoi_generator_3d", periodic);
  }

  [[nodiscard]] bool periodic() const noexcept { return _periodic; }

  [[nodiscard]] std::array<T, 3> box() const noexcept {
    return {_Lx, _Ly, _Lz};
  }

  [[nodiscard]] std::size_t seed_count() const noexcept {
    return _seeds.size();
  }

  // Indices of cells with zero volume. Only possible in weighted mode (a
  // plain Voronoi cell always contains its seed): a power-diagram weight set
  // that is too spread out leaves some seeds with an empty cell.
  [[nodiscard]] static std::vector<std::size_t>
  empty_cell_indices(shape_vector const& cells) {
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < cells.size(); ++i) {
      if (!(cells[i]->volume() > T{0})) out.push_back(i);
    }
    return out;
  }

  // build() that refuses silent empty cells: throws std::runtime_error
  // naming the empty seed indices (weighted mode) instead of returning
  // zero-volume placeholders.
  [[nodiscard]] shape_vector build_checked() const {
    auto cells = build();
    const auto empty = empty_cell_indices(cells);
    if (!empty.empty()) {
      std::string msg = "voronoi_generator_3d: " + std::to_string(empty.size()) +
                        " seed(s) have an empty cell (weights too spread out"
                        " or seed dropped); first index " +
                        std::to_string(empty.front());
      throw std::runtime_error{msg};
    }
    return cells;
  }

  // Build one voronoi_cell per seed. Cell index matches seed index.
  // In weighted mode some cells may be empty placeholders; check with
  // empty_cell_indices() or use build_checked(). After every tessellation
  // (either backend, periodic or not, weighted or not) the cell volumes
  // must sum to the box volume within voronoi_volume_sum_tol (relative);
  // otherwise std::runtime_error names the backend and the mismatch
  // instead of returning overlapping or missing cells. `opts.cancel` is
  // polled once per cell (in-house) / before the call (Voro++); a true
  // answer throws voronoi_cancelled.
  [[nodiscard]] shape_vector build(progress_options const& opts = {}) const {
    shape_vector out;
    if (opts.cancel && opts.cancel()) throw voronoi_cancelled{};
#ifdef RVEGEN_WITH_VORO_PP
    if (_backend == voronoi_backend::voro_pp) {
      out = detail::build_voronoi_cells_voro_pp<T>(_Lx, _Ly, _Lz, _seeds,
                                                   _periodic, _weights);
      check_volume_sum(out);
      return out;
    }
#endif
    out.reserve(_seeds.size());
    for (std::size_t i = 0; i < _seeds.size(); ++i) {
      if (opts.cancel && opts.cancel()) throw voronoi_cancelled{};
      out.push_back(build_cell(i));
    }
    check_volume_sum(out);
    return out;
  }

private:
  T _Lx, _Ly, _Lz;
  std::vector<seed_type> _seeds;
  voronoi_backend _backend;
  bool _periodic;
  std::vector<T> _weights;   // empty = plain Voronoi

  void check_volume_sum(shape_vector const& cells) const {
    detail::check_voronoi_volume_sum<T>(
        cells, _Lx * _Ly * _Lz,
        _backend == voronoi_backend::voro_pp ? "voro_pp" : "in_house",
        _periodic, !_weights.empty());
  }

  // Half-space: A·x + B·y + C·z ≤ D with a UNIT normal (A, B, C), so D is
  // a signed distance and all tolerances are plain lengths.
  struct half_space {
    T A, B, C, D;
  };

  // Tolerance scale: the SHORTEST domain edge, so a thin box (e.g.
  // 1 x 1 x 1e-4) is not resolved with a tolerance coarser than its own
  // thickness (that lost ~5 % of the periodic volume).
  [[nodiscard]] T scale() const noexcept {
    return std::min({_Lx, _Ly, _Lz});
  }

  // Distance tolerance for feasibility / vertex merging. Relative to the
  // SHORTEST edge (so a thin slab keeps its resolution) and deliberately
  // far below the seed-coincidence threshold: near-degenerate vertex
  // clusters (jittered lattices) only resolve correctly when the merge
  // tolerance is much smaller than the feature size. The second term is
  // the roundoff floor of the 3x3 solves, which grows with the domain
  // aspect ratio (nearly parallel bisectors).
  [[nodiscard]] T geometric_tol() const noexcept {
    const T lmin = scale();
    const T lmax = std::max({_Lx, _Ly, _Lz});
    return std::max(T{1e-12} * lmin,
                    T{64} * std::numeric_limits<T>::epsilon() * lmax *
                        (lmax / lmin));
  }

  // Collect the 6 box + (N - 1) bisector half-spaces for cell `i`.
  [[nodiscard]] std::vector<half_space> half_spaces_for(std::size_t i) const {
    std::vector<half_space> hs;
    hs.reserve(_seeds.size() + 5);
    // Box: x ≥ 0 → -x ≤ 0 ; x ≤ Lx ; same for y, z.
    hs.push_back({T{-1}, T{ 0}, T{ 0}, T{0 }});
    hs.push_back({T{ 1}, T{ 0}, T{ 0}, _Lx });
    hs.push_back({T{ 0}, T{-1}, T{ 0}, T{0 }});
    hs.push_back({T{ 0}, T{ 1}, T{ 0}, _Ly });
    hs.push_back({T{ 0}, T{ 0}, T{-1}, T{0 }});
    hs.push_back({T{ 0}, T{ 0}, T{ 1}, _Lz });
    // Bisector toward seed j: n · x ≤ n · (P_i + P_j)/2 with
    // n = (P_j − P_i)/|P_j − P_i|.
    for (std::size_t j = 0; j < _seeds.size(); ++j) {
      if (j == i) continue;
      hs.push_back(bisector(i, _seeds[j], weight(j)));
    }
    return hs;
  }

  [[nodiscard]] T weight(std::size_t k) const {
    return _weights.empty() ? T{0} : _weights[k];
  }

  // Power-diagram plane between seed i and a (possibly image) seed Pj with
  // weight wj, oriented so the cell of i is the n·x <= D side:
  //   n = (Pj-Pi)/|Pj-Pi|,  D = n·(Pi+Pj)/2 - (wj-wi)/(2|Pj-Pi|).
  [[nodiscard]] half_space bisector(std::size_t i, seed_type const& Pj,
                                    T wj) const {
    auto const& Pi = _seeds[i];
    const T dx = Pj[0] - Pi[0], dy = Pj[1] - Pi[1], dz = Pj[2] - Pi[2];
    const T len = std::sqrt(dx * dx + dy * dy + dz * dz);
    const T nx = dx / len, ny = dy / len, nz = dz / len;
    return {nx, ny, nz,
            nx * (Pi[0] + Pj[0]) * T{0.5} + ny * (Pi[1] + Pj[1]) * T{0.5} +
                nz * (Pi[2] + Pj[2]) * T{0.5} -
                (wj - weight(i)) / (T{2} * len)};
  }

  // Periodic cell of seed i: planes against the 3x3x3 images, pruned in two
  // passes (see header comment). No box planes: the 6 own-image planes
  // already confine the cell to [Pi - L/2, Pi + L/2] per axis.
  [[nodiscard]] std::unique_ptr<shape_base<T>>
  build_periodic_cell(std::size_t i) const {
    struct image { half_space h; T dist2; bool own_axis; };
    std::vector<image> imgs;
    imgs.reserve(27 * _seeds.size());
    auto const& Pi = _seeds[i];
    for (std::size_t j = 0; j < _seeds.size(); ++j) {
      for (int sx = -1; sx <= 1; ++sx)
        for (int sy = -1; sy <= 1; ++sy)
          for (int sz = -1; sz <= 1; ++sz) {
            const int nz_shift = (sx != 0) + (sy != 0) + (sz != 0);
            if (j == i && nz_shift == 0) continue;
            const seed_type Pj{_seeds[j][0] + sx * _Lx, _seeds[j][1] + sy * _Ly,
                               _seeds[j][2] + sz * _Lz};
            const T dx = Pj[0] - Pi[0], dy = Pj[1] - Pi[1], dz = Pj[2] - Pi[2];
            imgs.push_back({bisector(i, Pj, weight(j)),
                            dx * dx + dy * dy + dz * dz,
                            j == i && nz_shift == 1});
          }
    }
    // Pass 1: own axis images + the nearest images.
    constexpr std::size_t nearest = 30;
    std::vector<std::size_t> idx(imgs.size());
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    const std::size_t m = std::min(nearest, idx.size());
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(m),
                      idx.end(), [&](std::size_t a, std::size_t b) {
                        return imgs[a].dist2 < imgs[b].dist2;
                      });
    std::vector<half_space> hs;
    for (std::size_t k = 0; k < idx.size(); ++k) {
      if (k < m || imgs[idx[k]].own_axis) hs.push_back(imgs[idx[k]].h);
    }
    auto first = build_cell_from(hs);
    auto const* c1 = static_cast<voronoi_cell<T> const*>(first.get());
    if (c1->vertices().empty()) return first;
    // Pass 2: every plane that can still touch the cell. All of cell 1 lies
    // within R of Pi, so planes farther than R from Pi are redundant.
    T R{0};
    for (auto const& v : c1->vertices()) {
      const T dx = v[0] - Pi[0], dy = v[1] - Pi[1], dz = v[2] - Pi[2];
      R = std::max(R, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    const T slack = R * T{1e-6} + static_cast<T>(detail::voronoi_rel_tol) * scale();
    hs.clear();
    for (auto const& im : imgs) {
      const T dist = im.h.D - (im.h.A * Pi[0] + im.h.B * Pi[1] + im.h.C * Pi[2]);
      if (dist <= R + slack) hs.push_back(im.h);
    }
    return build_cell_from(hs);
  }

  // Build cell i: enumerate vertices, group by face, order CCW,
  // wrap in a voronoi_cell.
  [[nodiscard]] std::unique_ptr<shape_base<T>> build_cell(std::size_t i) const {
    if (_periodic) return build_periodic_cell(i);
    return build_cell_from(half_spaces_for(i));
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>>
  build_cell_from(std::vector<half_space> const& hs) const {
    const std::size_t H = hs.size();

    // For each candidate vertex, remember WHICH half-spaces it
    // lies on (so we can build faces by grouping).
    struct candidate {
      Eigen::Vector3<T> point;
      std::array<std::size_t, 3> on_planes;
      T err;   // roundoff bound of the 3x3 solve, ~eps*L/|det|
    };
    std::vector<candidate> cands;
    // Distance tolerance, relative to the domain size.
    const T tol = geometric_tol();

    for (std::size_t a = 0; a < H; ++a) {
      for (std::size_t b = a + 1; b < H; ++b) {
        for (std::size_t c = b + 1; c < H; ++c) {
          Eigen::Matrix3<T> M;
          M << hs[a].A, hs[a].B, hs[a].C,
               hs[b].A, hs[b].B, hs[b].C,
               hs[c].A, hs[c].B, hs[c].C;
          Eigen::Vector3<T> rhs;
          rhs << hs[a].D, hs[b].D, hs[c].D;
          Eigen::PartialPivLU<Eigen::Matrix3<T>> lu{M};
          // Unit normals → |det| is a dimensionless sine-volume.
          const T det = std::abs(M.determinant());
          if (det < T{1e-10}) {
            continue;   // parallel / coplanar — no unique intersection
          }
          const Eigen::Vector3<T> x = lu.solve(rhs);
          const T err = T{64} * std::numeric_limits<T>::epsilon() *
                        std::max({_Lx, _Ly, _Lz}) / det;
          // Feasibility: all half-spaces satisfied within tolerance.
          bool feasible = true;
          for (std::size_t k = 0; k < H; ++k) {
            if (k == a || k == b || k == c) continue;
            const T lhs = hs[k].A * x[0] + hs[k].B * x[1] + hs[k].C * x[2];
            if (lhs > hs[k].D + tol) { feasible = false; break; }
          }
          if (!feasible) continue;
          cands.push_back({x, {a, b, c}, err});
        }
      }
    }

    if (cands.empty()) {
      // Pathological (every plane triplet degenerate). Produce an
      // empty placeholder so cell i still matches seed i; caller can
      // detect via `volume() == 0`.
      return std::make_unique<voronoi_cell<T>>();
    }

    // Deduplicate vertices that came from different triplets but
    // landed at the same point (a vertex on 4+ half-spaces, e.g.
    // a corner of the bounding box). Use L2 distance with a small
    // tolerance PLUS the roundoff bound of the two solves: a triple of
    // nearly parallel bisectors (two close seeds) is solved with error
    // ~eps/|det|, far above the fixed tolerance, so its copies of one
    // vertex would otherwise stay apart and corrupt the face loops (the
    // periodic cell of a third seed lost up to 10 % of its volume).
    std::vector<Eigen::Vector3<T>> verts;
    std::vector<std::vector<std::size_t>> vert_planes;   // which planes pass through each unique vertex
    std::vector<T> vert_err;
    const T merge_tol = tol;
    for (auto const& c : cands) {
      bool dup = false;
      for (std::size_t v = 0; v < verts.size(); ++v) {
        if ((verts[v] - c.point).norm() < merge_tol + c.err + vert_err[v]) {
          // Add this triplet's planes to the existing vertex.
          for (auto p : c.on_planes) {
            if (std::find(vert_planes[v].begin(), vert_planes[v].end(), p)
                == vert_planes[v].end()) {
              vert_planes[v].push_back(p);
            }
          }
          dup = true;
          break;
        }
      }
      if (!dup) {
        verts.push_back(c.point);
        vert_planes.emplace_back(c.on_planes.begin(), c.on_planes.end());
        vert_err.push_back(c.err);
      }
    }

    // Build faces: for each half-space, gather the vertices that
    // lie on it (i.e. have it in their vert_planes list). A face
    // needs ≥ 3 vertices; skip degenerate cases.
    std::vector<std::vector<std::size_t>> faces;
    for (std::size_t h = 0; h < H; ++h) {
      std::vector<std::size_t> face_verts;
      for (std::size_t v = 0; v < verts.size(); ++v) {
        if (std::find(vert_planes[v].begin(), vert_planes[v].end(), h)
            != vert_planes[v].end()) {
          face_verts.push_back(v);
        }
      }
      if (face_verts.size() < 3) continue;   // degenerate face

      // Order CCW around the half-space's outward normal (away from
      // the cell interior — the half-space's stored (A, B, C) IS the
      // outward normal direction, since the half-space inequality
      // says "we're on the ≤ side").
      const Eigen::Vector3<T> n_out{hs[h].A, hs[h].B, hs[h].C};
      Eigen::Vector3<T> centroid{T{0}, T{0}, T{0}};
      for (auto v : face_verts) centroid += verts[v];
      centroid /= static_cast<T>(face_verts.size());

      // Pick two in-plane basis vectors orthogonal to n_out.
      Eigen::Vector3<T> tangent;
      Eigen::Vector3<T> ref_x{T{1}, T{0}, T{0}};
      if (std::abs(n_out.dot(ref_x)) > T{0.9} * n_out.norm()) {
        ref_x = Eigen::Vector3<T>{T{0}, T{1}, T{0}};
      }
      tangent = n_out.cross(ref_x).normalized();
      const Eigen::Vector3<T> bitangent =
          n_out.cross(tangent).normalized();

      // Sort by angle around the centroid. The angle of every vertex is
      // computed ONCE and the sort compares stored numbers: recomputing it
      // inside the comparator lets a compiler that contracts a*b+c (FMA,
      // -march=native) evaluate the same vertex differently in different
      // comparisons. A vertex on the atan2 branch cut (angle +-pi, typical
      // for symmetric seed sets) then flips sign between comparisons, the
      // comparator stops being a strict weak order and the face loop comes
      // out scrambled (periodic cells lost ~0.5 % of their volume).
      std::vector<std::pair<T, std::size_t>> keyed;
      keyed.reserve(face_verts.size());
      for (auto v : face_verts) {
        const auto d = verts[v] - centroid;
        keyed.emplace_back(std::atan2(d.dot(bitangent), d.dot(tangent)), v);
      }
      std::sort(keyed.begin(), keyed.end());
      for (std::size_t k = 0; k < keyed.size(); ++k) face_verts[k] = keyed[k].second;
      faces.push_back(std::move(face_verts));
    }

    // Build the voronoi_cell. Vertices need to be `gte::Vector<3, T>`.
    std::vector<gte::Vector<3, T>> gte_verts;
    gte_verts.reserve(verts.size());
    for (auto const& v : verts) {
      gte::Vector<3, T> g;
      g[0] = v[0]; g[1] = v[1]; g[2] = v[2];
      gte_verts.push_back(g);
    }
    return std::make_unique<voronoi_cell<T>>(std::move(gte_verts),
                                              std::move(faces));
  }
};

// Cells as a shape_vector ready for the standard writers (voxel, gmsh, ...).
// The first seed_count() entries are the cells in seed order. For a periodic
// generator a cell that sticks out of the box is returned whole by build(),
// so its wrapped pieces (translated copies, see generate_periodic_wraps) are
// appended after them: without them part of the box has no owner. Every
// piece carries info "grain_id" = its seed index, so all pieces of one grain
// share the grain's id (sample_voxel_grid honours it).
template <typename T = double>
[[nodiscard]] inline std::vector<std::unique_ptr<shape_base<T>>>
voronoi_to_shapes(voronoi_generator_3d<T> const& gen) {
  auto shapes = gen.build();
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    shapes[i]->info().set("grain_id", i);
  }
  if (gen.periodic()) {
    const std::size_t n = shapes.size();
    for (std::size_t i = 0; i < n; ++i) {
      auto wraps = generate_periodic_wraps<T>(*shapes[i], gen.box());
      for (auto& w : wraps) shapes.push_back(std::move(w));
    }
  }
  return shapes;
}

} // namespace rvegen
