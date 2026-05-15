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
// test ≈ O(N⁵). For N ≤ 50 seeds this is well under a second; beyond
// that, switch to a real Voronoi library (Voro++, CGAL).
//
// Numerical robustness:
//   * The 3x3 solve uses Eigen's PartialPivLU.
//   * Feasibility test uses a relative tolerance (1e-9 of each
//     half-space coefficient norm) — generous enough for typical
//     engineering-scale inputs.
//   * Coincident vertices (multiple triplets producing the same
//     point at degeneracies) are deduplicated by L2 distance.
//   * Coincident seeds are not handled — the output will contain a
//     pathologically thin cell. Caller should filter coincident
//     seeds upstream.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "../shapes/shape_base.h"
#include "../shapes/voronoi_cell.h"

namespace rvegen {

template <typename T = double>
class voronoi_generator_3d {
public:
  using value_type = T;
  using seed_type = std::array<T, 3>;
  using shape_vector = std::vector<std::unique_ptr<shape_base<T>>>;

  voronoi_generator_3d(T Lx, T Ly, T Lz, std::vector<seed_type> seeds)
      : _Lx{Lx}, _Ly{Ly}, _Lz{Lz}, _seeds{std::move(seeds)} {
    if (Lx <= T{0} || Ly <= T{0} || Lz <= T{0}) {
      throw std::runtime_error{
          "voronoi_generator_3d: Lx, Ly, Lz must all be positive"};
    }
    if (_seeds.empty()) {
      throw std::runtime_error{
          "voronoi_generator_3d: at least one seed required"};
    }
  }

  [[nodiscard]] std::size_t seed_count() const noexcept {
    return _seeds.size();
  }

  // Build one voronoi_cell per seed. Cell index matches seed index.
  [[nodiscard]] shape_vector build() const {
    shape_vector out;
    out.reserve(_seeds.size());

    for (std::size_t i = 0; i < _seeds.size(); ++i) {
      out.push_back(build_cell(i));
    }
    return out;
  }

private:
  T _Lx, _Ly, _Lz;
  std::vector<seed_type> _seeds;

  // Half-space: A·x + B·y + C·z ≤ D.
  struct half_space {
    T A, B, C, D;
  };

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
    // Bisectors: 2·(P_j − P_i) · x ≤ |P_j|² − |P_i|².
    auto const& Pi = _seeds[i];
    const T mag_i = Pi[0] * Pi[0] + Pi[1] * Pi[1] + Pi[2] * Pi[2];
    for (std::size_t j = 0; j < _seeds.size(); ++j) {
      if (j == i) continue;
      auto const& Pj = _seeds[j];
      hs.push_back({
          T{2} * (Pj[0] - Pi[0]),
          T{2} * (Pj[1] - Pi[1]),
          T{2} * (Pj[2] - Pi[2]),
          (Pj[0] * Pj[0] + Pj[1] * Pj[1] + Pj[2] * Pj[2]) - mag_i
      });
    }
    return hs;
  }

  // Build cell i: enumerate vertices, group by face, order CCW,
  // wrap in a voronoi_cell.
  [[nodiscard]] std::unique_ptr<shape_base<T>> build_cell(std::size_t i) const {
    const auto hs = half_spaces_for(i);
    const std::size_t H = hs.size();

    // For each candidate vertex, remember WHICH half-spaces it
    // lies on (so we can build faces by grouping).
    struct candidate {
      Eigen::Vector3<T> point;
      std::array<std::size_t, 3> on_planes;
    };
    std::vector<candidate> cands;

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
          if (std::abs(M.determinant())
              < std::numeric_limits<T>::epsilon() * T{1024}) {
            continue;   // parallel / coplanar — no unique intersection
          }
          const Eigen::Vector3<T> x = lu.solve(rhs);
          // Feasibility: all half-spaces satisfied within tolerance.
          bool feasible = true;
          for (std::size_t k = 0; k < H; ++k) {
            if (k == a || k == b || k == c) continue;
            const T lhs = hs[k].A * x[0] + hs[k].B * x[1] + hs[k].C * x[2];
            // Tolerance: 1e-9 relative to the coefficient norm.
            const T tol = T{1e-9} * (T{1} + std::abs(hs[k].D));
            if (lhs > hs[k].D + tol) { feasible = false; break; }
          }
          if (!feasible) continue;
          cands.push_back({x, {a, b, c}});
        }
      }
    }

    if (cands.empty()) {
      // Pathological — likely the seed sits outside the box, or
      // every plane triplet is degenerate. Produce an empty cell;
      // caller can detect via `volume() == 0`.
      return std::make_unique<voronoi_cell<T>>();
    }

    // Deduplicate vertices that came from different triplets but
    // landed at the same point (a vertex on 4+ half-spaces, e.g.
    // a corner of the bounding box). Use L2 distance with a small
    // tolerance.
    std::vector<Eigen::Vector3<T>> verts;
    std::vector<std::vector<std::size_t>> vert_planes;   // which planes pass through each unique vertex
    const T merge_tol = T{1e-9} * std::max({_Lx, _Ly, _Lz});
    for (auto const& c : cands) {
      bool dup = false;
      for (std::size_t v = 0; v < verts.size(); ++v) {
        if ((verts[v] - c.point).norm() < merge_tol) {
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

      // Sort by angle around the centroid.
      std::sort(face_verts.begin(), face_verts.end(),
                [&](std::size_t a, std::size_t b) {
                  const auto da = verts[a] - centroid;
                  const auto db = verts[b] - centroid;
                  const T ang_a = std::atan2(da.dot(bitangent),
                                             da.dot(tangent));
                  const T ang_b = std::atan2(db.dot(bitangent),
                                             db.dot(tangent));
                  return ang_a < ang_b;
                });
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

} // namespace rvegen
