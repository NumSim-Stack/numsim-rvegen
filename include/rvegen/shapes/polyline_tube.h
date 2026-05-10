#pragma once

// Polyline tube — a swept tube of constant radius along a 3D centerline.
// Foundation for the woven-fabric / textile-composite work in #3:
// individual fibres are polyline tubes; a future weave_generator emits
// N of them with interlocked over/under topology.
//
// What this header provides today:
//   * polyline_tube<T> shape with direct C++ ctors (std::array or
//     gte::Vector3 centerline points).
//   * is_inside via GTE's DCPQuery point-to-segment for each centerline edge.
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
#include <vector>

#include <numsim-core/static_indexing.h>

#include <Mathematics/DistPointSegment.h>
#include <Mathematics/Segment.h>
#include <Mathematics/Vector.h>

#include "../types.h"
#include "box_bounding.h"
#include "shape_base.h"

namespace rvegen {

template <typename T = double>
class polyline_tube
    : public numsim_core::static_indexing<polyline_tube<T>, shape_base<T>> {
public:
  using value_type = T;
  using point_type = gte::Vector<3, T>;

  polyline_tube() = default;

  polyline_tube(std::vector<point_type> centerline, T radius)
      : _centerline{std::move(centerline)}, _radius{radius} {}

  // std::array overload — accepts {{x, y, z}, ...}-style initializers
  // without forcing callers to construct gte::Vector3 explicitly.
  polyline_tube(std::vector<std::array<T, 3>> const& centerline, T radius)
      : _radius{radius} {
    _centerline.reserve(centerline.size());
    for (auto const& p : centerline) {
      _centerline.push_back(make_vec(p));
    }
  }

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
      const auto d = _centerline[i] - _centerline[i - 1];
      length += std::sqrt(gte::Dot(d, d));
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
    point_type sum = point_type::Zero();
    for (auto const& p : _centerline) sum += p;
    const T n = static_cast<T>(_centerline.size());
    return {sum[0] / n, sum[1] / n, sum[2] / n};
  }

  void set_middle_point(std::array<T, 3> middle_point) override {
    const auto cur = get_middle_point();
    point_type delta;
    delta[0] = middle_point[0] - cur[0];
    delta[1] = middle_point[1] - cur[1];
    delta[2] = middle_point[2] - cur[2];
    for (auto& p : _centerline) p += delta;
  }

  // Inside iff GTE's point-to-segment squared distance to ANY centerline
  // edge is ≤ radius². O(N) in centerline length; voxelization at
  // moderate grids stays fast because most voxels reject early via
  // the bounding box check upstream.
  [[nodiscard]] bool is_inside(std::array<T, 3> const& point) const override {
    if (_centerline.size() < 2) return false;
    const T r2 = _radius * _radius;
    const auto p = make_vec(point);
    gte::DCPQuery<T, point_type, gte::Segment<3, T>> query;
    for (std::size_t i = 1; i < _centerline.size(); ++i) {
      gte::Segment<3, T> seg{_centerline[i - 1], _centerline[i]};
      if (query(p, seg).sqrDistance <= r2) return true;
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
  [[nodiscard]] std::vector<point_type> const& centerline() const noexcept {
    return _centerline;
  }
  [[nodiscard]] T radius() const noexcept { return _radius; }

  void set_centerline(std::vector<point_type> centerline) {
    _centerline = std::move(centerline);
  }
  void set_radius(T r) noexcept { _radius = r; }

private:
  static point_type make_vec(std::array<T, 3> const& a) {
    point_type v;
    v[0] = a[0]; v[1] = a[1]; v[2] = a[2];
    return v;
  }

  // Returns {min_corner, max_corner} of the bounding box in 3D.
  [[nodiscard]] std::array<std::array<T, 3>, 2> compute_bb_extents() const {
    if (_centerline.empty()) return {{{T{0}, T{0}, T{0}}, {T{0}, T{0}, T{0}}}};
    std::array<T, 3> mn{_centerline[0][0], _centerline[0][1], _centerline[0][2]};
    std::array<T, 3> mx = mn;
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

  std::vector<point_type> _centerline;
  T _radius{T{0}};
};

} // namespace rvegen
