#pragma once

// Polyline tube — a swept tube of constant radius along a 3D centerline.
// Foundation for the woven-fabric / textile-composite work in #3:
// individual fibres are polyline tubes; a future weave_generator emits
// N of them with interlocked over/under topology.
//
// What this header provides today:
//   * polyline_tube<T> shape with direct C++ ctor.
//   * is_inside via closest-point-to-segment over each centerline edge.
//   * AABB bounding box from centerline extents ± radius.
//   * static_indexing id so the collision dispatcher can address it
//     (fallback to AABB-only collision until a polyline-tube-vs-X
//     precise dispatcher entry is registered — fine for the inner
//     generator loop).
//   * clone(), get_middle_point(), set_middle_point().
//
// Out of scope here, ships in follow-up PRs against #3:
//   * polyline_tube_input + JSON schema (a centerline can't be
//     expressed as a flat parameter_handler; needs either
//     vector<double> packing or a separate config form).
//   * weave_generator that produces N interlocked tubes.
//   * to_mesh(polyline_tube) — Frenet-frame ring extrusion for
//     Tessera VTK rendering.
//   * Precise collision_details(polyline_tube, X) overloads — falls
//     back to AABB overlap until then.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <vector>

#include <numsim-core/static_indexing.h>

#include "../types.h"
#include "box_bounding.h"
#include "shape_base.h"

namespace rvegen {

template <typename T = double>
class polyline_tube
    : public numsim_core::static_indexing<polyline_tube<T>, shape_base<T>> {
public:
  using value_type = T;

  polyline_tube() = default;

  polyline_tube(std::vector<std::array<T, 3>> centerline, T radius)
      : _centerline{std::move(centerline)}, _radius{radius} {}

  // ---- shape_base contract ----------------------------------------------
  [[nodiscard]] T area() const override { return T{0}; }   // 3D shape

  // Tube volume = π · r² · total_centerline_length. Approximation:
  // ignores the spherical end-caps and the curvature-weighted surplus
  // at sharp corners (Steiner-tube-like correction), but accurate to
  // a few percent for smooth centerlines and good enough for the
  // generator's volume_fraction termination.
  [[nodiscard]] T volume() const override {
    if (_centerline.size() < 2) return T{0};
    T length = T{0};
    for (std::size_t i = 1; i < _centerline.size(); ++i) {
      const auto& a = _centerline[i - 1];
      const auto& b = _centerline[i];
      const T dx = b[0] - a[0];
      const T dy = b[1] - a[1];
      const T dz = b[2] - a[2];
      length += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    constexpr T pi = T{3.14159265358979323846};
    return pi * _radius * _radius * length;
  }

  [[nodiscard]] std::array<T, 3> max_expansion() const override {
    auto bb_extents = compute_bb_extents();
    return {(bb_extents[1][0] - bb_extents[0][0]) * T{0.5},
            (bb_extents[1][1] - bb_extents[0][1]) * T{0.5},
            (bb_extents[1][2] - bb_extents[0][2]) * T{0.5}};
  }

  // Centroid: mean of centerline points. Cheap and physically sensible
  // for the centre-in-domain filter.
  [[nodiscard]] std::array<T, 3> get_middle_point() const override {
    if (_centerline.empty()) return {T{0}, T{0}, T{0}};
    std::array<T, 3> sum{T{0}, T{0}, T{0}};
    for (auto const& p : _centerline) {
      sum[0] += p[0]; sum[1] += p[1]; sum[2] += p[2];
    }
    const T n = static_cast<T>(_centerline.size());
    return {sum[0] / n, sum[1] / n, sum[2] / n};
  }

  void set_middle_point(std::array<T, 3> middle_point) override {
    const auto cur = get_middle_point();
    const T dx = middle_point[0] - cur[0];
    const T dy = middle_point[1] - cur[1];
    const T dz = middle_point[2] - cur[2];
    for (auto& p : _centerline) {
      p[0] += dx; p[1] += dy; p[2] += dz;
    }
  }

  // Inside iff closest distance from `point` to ANY centerline segment
  // is ≤ radius. O(N) in centerline length; voxelization at moderate
  // grids stays fast because most voxels reject early via bounding box.
  [[nodiscard]] bool is_inside(std::array<T, 3> const& point) const override {
    if (_centerline.size() < 2) return false;
    const T r2 = _radius * _radius;
    for (std::size_t i = 1; i < _centerline.size(); ++i) {
      if (point_segment_distance_sq(point,
                                     _centerline[i - 1],
                                     _centerline[i]) <= r2) {
        return true;
      }
    }
    return false;
  }

  void make_bounding_box() override {
    auto bb_extents = compute_bb_extents();
    auto box = std::make_unique<box_bounding<T>>();
    box->top_point()    = bb_extents[1];
    box->bottom_point() = bb_extents[0];
    this->_bounding_box = std::move(box);
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> clone() const override {
    return std::make_unique<polyline_tube<T>>(*this);
  }

  [[nodiscard]] numsim_core::type_id shape_id() const noexcept override {
    return polyline_tube::m_id;
  }

  // ---- accessors --------------------------------------------------------
  [[nodiscard]] std::vector<std::array<T, 3>> const& centerline() const noexcept {
    return _centerline;
  }
  [[nodiscard]] T radius() const noexcept { return _radius; }

  void set_centerline(std::vector<std::array<T, 3>> centerline) {
    _centerline = std::move(centerline);
  }
  void set_radius(T r) noexcept { _radius = r; }

private:
  // Returns {min_corner, max_corner} of the bounding box in 3D.
  [[nodiscard]] std::array<std::array<T, 3>, 2> compute_bb_extents() const {
    if (_centerline.empty()) return {{{T{0}, T{0}, T{0}}, {T{0}, T{0}, T{0}}}};
    std::array<T, 3> mn = _centerline[0];
    std::array<T, 3> mx = _centerline[0];
    for (auto const& p : _centerline) {
      for (int k = 0; k < 3; ++k) {
        mn[k] = std::min(mn[k], p[k]);
        mx[k] = std::max(mx[k], p[k]);
      }
    }
    for (int k = 0; k < 3; ++k) {
      mn[k] -= _radius;
      mx[k] += _radius;
    }
    return {{mn, mx}};
  }

  // Standard parameterised closest-point-on-segment. Returns the
  // squared 3D distance from `p` to the segment [a, b]. Squared form
  // avoids a sqrt per call inside is_inside.
  static T point_segment_distance_sq(std::array<T, 3> const& p,
                                      std::array<T, 3> const& a,
                                      std::array<T, 3> const& b) {
    const T abx = b[0] - a[0];
    const T aby = b[1] - a[1];
    const T abz = b[2] - a[2];
    const T apx = p[0] - a[0];
    const T apy = p[1] - a[1];
    const T apz = p[2] - a[2];
    const T ab_len_sq = abx * abx + aby * aby + abz * abz;
    if (ab_len_sq <= T{0}) {
      // Degenerate zero-length segment: distance to point a.
      return apx * apx + apy * apy + apz * apz;
    }
    T t = (apx * abx + apy * aby + apz * abz) / ab_len_sq;
    t = std::max(T{0}, std::min(T{1}, t));
    const T qx = a[0] + t * abx;
    const T qy = a[1] + t * aby;
    const T qz = a[2] + t * abz;
    const T dx = p[0] - qx;
    const T dy = p[1] - qy;
    const T dz = p[2] - qz;
    return dx * dx + dy * dy + dz * dz;
  }

  std::vector<std::array<T, 3>> _centerline;
  T _radius{T{0}};
};

} // namespace rvegen
