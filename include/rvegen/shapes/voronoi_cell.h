#pragma once

// voronoi_cell — a convex polyhedral inclusion defined by vertices and
// faces. Foundation for #8 (Voronoi-based polycrystal generator).
//
// Why this scope and why no CGAL dependency yet:
//   The full polycrystal generator wants to (a) sample N seed points
//   in the RVE box, (b) compute a periodic 3D Voronoi tessellation,
//   and (c) emit each cell as a shape. Step (b) needs CGAL or Voro++
//   — both are heavyweight dependencies (CGAL is ~200 MB installed).
//   Adding them as hard rvegen deps to land an unused phase-1 cell
//   shape is premature.
//
//   Instead, this PR ships the *consumer* side: a convex polyhedron
//   shape that can be constructed from pre-computed cell geometry.
//   External tools (Voro++, qhull, CGAL outside rvegen, even SciPy
//   `scipy.spatial.Voronoi` from Python) can produce the cell vertex
//   / face data; rvegen ingests it. The follow-up PR adds the
//   in-tree tessellator.
//
// What this header lands today (phase 1 of #8):
//   * `voronoi_cell<T>` — vertices (`std::vector<gte::Vector3<T>>`)
//     + faces (`std::vector<std::vector<std::size_t>>` — per-face
//     vertex-index lists). Assumes the cell is convex.
//   * `is_inside(p)` — half-space test against each face (outward
//     normals derived once via the centroid trick).
//   * `volume()` — divergence theorem via signed tetrahedra from
//     centroid.
//   * Standard `shape_base` overrides + `static_indexing` shape_id.
//
// Out of scope here, ships in follow-up PRs against #8:
//   * `voronoi_polycrystal_generator` — produces N tessellated cells
//     for a given seed-point distribution. Needs Voro++ or CGAL; pick
//     one and add it as an optional dependency.
//   * Periodic Voronoi (cells that wrap across the RVE box). Trivial
//     once the tessellator is wired in.
//   * `voronoi_cell_input` — JSON-driven loader (cell vertex / face
//     data is heterogeneous; needs careful schema design).
//   * Precise `collision_details(voronoi_cell, X)` overloads — falls
//     back to AABB until then.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <numsim-core/static_indexing.h>

#include <Mathematics/Vector3.h>

#include "../types.h"
#include "box_bounding.h"
#include "shape_base.h"

namespace rvegen {

template <typename T = double>
class voronoi_cell
    : public numsim_core::static_indexing<voronoi_cell<T>, shape_base<T>> {
public:
  using value_type = T;
  using vector_type = gte::Vector<3, T>;
  using face_type = std::vector<std::size_t>;

  voronoi_cell() = default;

  voronoi_cell(std::vector<vector_type> vertices,
               std::vector<face_type> faces)
      : _vertices{std::move(vertices)}, _faces{std::move(faces)} {
    rebuild_face_planes();
  }

  // ---- shape_base contract ----------------------------------------------
  [[nodiscard]] T area() const override { return T{0}; }   // 3D shape

  // Volume via signed tetrahedra from the cell centroid. For each face,
  // form a fan of triangles by tessellating the face polygon with the
  // first vertex of the face as a fan apex; sum (1/6) · |a · (b × c)|
  // for each triangle, where a/b/c are vertex offsets from centroid.
  // Returns the sum without an absolute value at the end — convex
  // polyhedra with consistent face winding produce a positive sum.
  [[nodiscard]] T volume() const override {
    if (_vertices.empty() || _faces.empty()) return T{0};
    const auto c = _centroid;
    T sum = T{0};
    for (auto const& face : _faces) {
      if (face.size() < 3) continue;
      const vector_type apex = _vertices[face[0]] - c;
      for (std::size_t i = 1; i + 1 < face.size(); ++i) {
        const vector_type b = _vertices[face[i]] - c;
        const vector_type d = _vertices[face[i + 1]] - c;
        // Tetrahedron volume = (1/6) |apex · (b × d)|.
        const T cx = b[1] * d[2] - b[2] * d[1];
        const T cy = b[2] * d[0] - b[0] * d[2];
        const T cz = b[0] * d[1] - b[1] * d[0];
        sum += std::abs(apex[0] * cx + apex[1] * cy + apex[2] * cz);
      }
    }
    return sum / T{6};
  }

  [[nodiscard]] std::array<T, 3> max_expansion() const override {
    auto bb = compute_bb_extents();
    return {(bb[1][0] - bb[0][0]) * T{0.5},
            (bb[1][1] - bb[0][1]) * T{0.5},
            (bb[1][2] - bb[0][2]) * T{0.5}};
  }

  [[nodiscard]] std::array<T, 3> get_middle_point() const override {
    return {_centroid[0], _centroid[1], _centroid[2]};
  }

  void set_middle_point(std::array<T, 3> middle_point) override {
    const T dx = middle_point[0] - _centroid[0];
    const T dy = middle_point[1] - _centroid[1];
    const T dz = middle_point[2] - _centroid[2];
    for (auto& v : _vertices) {
      v[0] += dx; v[1] += dy; v[2] += dz;
    }
    _centroid[0] += dx; _centroid[1] += dy; _centroid[2] += dz;
    // Face planes translate with the cell — re-derive offsets only.
    for (auto& plane : _face_planes) {
      plane.offset += plane.normal[0] * dx + plane.normal[1] * dy +
                       plane.normal[2] * dz;
    }
  }

  // Inside iff the point is on the inward side of every face plane.
  // Outward normals are derived (once, in rebuild_face_planes) by
  // ensuring the centroid is on the "negative" side; then "inside" is
  // `(p · n - offset) ≤ 0`.
  [[nodiscard]] bool is_inside(std::array<T, 3> const& point) const override {
    if (_face_planes.empty()) return false;
    for (auto const& plane : _face_planes) {
      const T side = plane.normal[0] * point[0] + plane.normal[1] * point[1] +
                      plane.normal[2] * point[2] - plane.offset;
      if (side > T{0}) return false;
    }
    return true;
  }

  void make_bounding_box() override {
    auto bb = compute_bb_extents();
    auto box = std::make_unique<box_bounding<T>>();
    box->top_point()    = bb[1];
    box->bottom_point() = bb[0];
    this->_bounding_box = std::move(box);
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> clone() const override {
    return std::make_unique<voronoi_cell<T>>(*this);
  }

  [[nodiscard]] numsim_core::type_id shape_id() const noexcept override {
    return voronoi_cell::m_id;
  }

  // ---- accessors --------------------------------------------------------
  [[nodiscard]] std::vector<vector_type> const& vertices() const noexcept {
    return _vertices;
  }
  [[nodiscard]] std::vector<face_type> const& faces() const noexcept {
    return _faces;
  }

private:
  struct face_plane {
    vector_type normal;
    T offset;   // normal · v[0] for any face vertex v[0]
  };

  // Builds outward-pointing normals for every face. The face polygon
  // is assumed planar and convex; normal direction is fixed by
  // requiring the centroid to lie on the negative side. The centroid
  // itself is the mean of all vertices — fine for the convex-cell
  // case here.
  void rebuild_face_planes() {
    _centroid = vector_type::Zero();
    if (_vertices.empty()) return;
    for (auto const& v : _vertices) _centroid += v;
    _centroid /= static_cast<T>(_vertices.size());

    _face_planes.clear();
    _face_planes.reserve(_faces.size());
    for (auto const& face : _faces) {
      if (face.size() < 3) continue;
      const auto& a = _vertices[face[0]];
      const auto& b = _vertices[face[1]];
      const auto& c = _vertices[face[2]];
      const vector_type e1 = b - a;
      const vector_type e2 = c - a;
      vector_type n;
      n[0] = e1[1] * e2[2] - e1[2] * e2[1];
      n[1] = e1[2] * e2[0] - e1[0] * e2[2];
      n[2] = e1[0] * e2[1] - e1[1] * e2[0];
      const T mag = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      if (mag > T{0}) n /= mag;
      // Flip outward if the centroid is on the positive side.
      const T offset0 = n[0] * a[0] + n[1] * a[1] + n[2] * a[2];
      const T centroid_side = n[0] * _centroid[0] + n[1] * _centroid[1] +
                               n[2] * _centroid[2] - offset0;
      if (centroid_side > T{0}) {
        n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2];
        _face_planes.push_back({n, -offset0});
      } else {
        _face_planes.push_back({n, offset0});
      }
    }
  }

  [[nodiscard]] std::array<std::array<T, 3>, 2> compute_bb_extents() const {
    if (_vertices.empty()) {
      return {{{T{0}, T{0}, T{0}}, {T{0}, T{0}, T{0}}}};
    }
    std::array<T, 3> mn{_vertices[0][0], _vertices[0][1], _vertices[0][2]};
    std::array<T, 3> mx = mn;
    for (auto const& v : _vertices) {
      for (int k = 0; k < 3; ++k) {
        mn[k] = std::min(mn[k], v[k]);
        mx[k] = std::max(mx[k], v[k]);
      }
    }
    return {{mn, mx}};
  }

  std::vector<vector_type> _vertices;
  std::vector<face_type> _faces;
  std::vector<face_plane> _face_planes;
  vector_type _centroid{vector_type::Zero()};
};

} // namespace rvegen
