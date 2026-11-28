#pragma once

// Minimum gaps for the placement generators (#111).
//
// Rejecting overlaps alone lets two inclusions end up 1e-6 apart, and an
// inclusion a hair away from a domain face or poking through it by a
// sliver. A mesher then produces degenerate elements or no mesh at all, and
// no writer can repair that afterwards. The generators `only_inside`,
// `random` and `periodic` therefore take three optional keys:
//
//   "min_gap"           smallest distance between two inclusions, in the
//                       units of the box (default 0)
//   "min_gap_relative"  the same as a fraction of the equivalent diameter
//                       of the smaller of the two inclusions (default 0);
//                       the larger of the two requirements applies
//   "wall_gap"          clearance to the boundary of the domain (default
//                       0): an inclusion either stays at least this far
//                       from a face or, where the generator lets
//                       inclusions cross a face (`random`, `periodic`),
//                       reaches at least this far through it; the same
//                       for the corners of a 2D box and the edges and
//                       corners of a 3D one (boundary_clear() below).
//                       `only_inside` never crosses.
//
// All three default to 0, and with 0 a generator takes exactly the code
// path it took before these keys existed: same candidates, same accepted
// list.
//
// The equivalent diameter of a shape is the diameter of the circle (2D) or
// sphere (3D) with the same area or volume; it does not depend on the
// orientation, unlike a bounding-box extent. The cube root is the portable
// one (portable_math.h), so a relative gap is the same number everywhere.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <numsim-core/input_parameter_controller.h>

#include "../distributions/portable_math.h"
#include "../domain.h"
#include "../intersection/aabb_grid.h"
#include "../intersection/collision_dispatcher.h"
#include "../shapes/bounding_box_base.h"
#include "../shapes/circle.h"
#include "../shapes/polyline_tube.h"
#include "../shapes/shape_base.h"
#include "../shapes/sphere.h"
#include "../types.h"

namespace rvegen {

template <typename T = double>
struct placement_gap {
  T min_gap{0};
  T min_gap_relative{0};
  T wall_gap{0};

  constexpr placement_gap() = default;
  placement_gap(T absolute, T relative, T wall)
      : min_gap{absolute}, min_gap_relative{relative}, wall_gap{wall} {
    check("min_gap", min_gap);
    check("min_gap_relative", min_gap_relative);
    check("wall_gap", wall_gap);
  }
  explicit placement_gap(parameter_handler_t const& handler)
      : placement_gap{handler.template get<T>("min_gap"),
                      handler.template get<T>("min_gap_relative"),
                      handler.template get<T>("wall_gap")} {}

  // Adds the three keys to a generator's schema.
  static void declare(parameter_controller_t& p) {
    p.template optional<T>("min_gap", T{0})
        .template add<min_only<T{0}>>()
        .template add<numsim_core::description_label<"smallest distance between two inclusions, in box units (default 0: touching is allowed to within rounding)">>();
    p.template optional<T>("min_gap_relative", T{0})
        .template add<min_only<T{0}>>()
        .template add<numsim_core::description_label<"smallest distance between two inclusions as a fraction of the equivalent diameter of the smaller one (default 0); the larger of min_gap and this applies">>();
    p.template optional<T>("wall_gap", T{0})
        .template add<min_only<T{0}>>()
        .template add<numsim_core::description_label<"clearance to the domain faces (default 0): an inclusion stays this far from a face or, where it may cross one, reaches this far through it">>();
  }

  [[nodiscard]] constexpr bool between_shapes() const noexcept {
    return min_gap > T{0} || min_gap_relative > T{0};
  }

  // The size a relative gap refers to; 0 when no relative gap is set (the
  // measure is then never computed).
  [[nodiscard]] T size_of(shape_base<T> const& shape) const {
    if (!(min_gap_relative > T{0})) return T{0};
    const T m = shape.measure();
    if (!(m > T{0})) return T{0};
    return shape.dimension() == 3
               ? portable_cbrt(T{6} * m / std::numbers::pi_v<T>)
               : T{2} * std::sqrt(m / std::numbers::pi_v<T>);
  }

  // Required distance between two shapes of the given sizes.
  [[nodiscard]] constexpr T between(T size_a, T size_b) const noexcept {
    return std::max(min_gap, min_gap_relative * std::min(size_a, size_b));
  }
  // Largest distance a shape of this size can be asked to keep from any
  // other shape: the padding of the neighbour search.
  [[nodiscard]] constexpr T reach(T size) const noexcept {
    return std::max(min_gap, min_gap_relative * size);
  }

private:
  static void check(char const* name, T v) {
    if (!(v >= T{0}) || !(v <= std::numeric_limits<T>::max())) {
      throw std::invalid_argument{std::string{name} +
                                  " must be finite and >= 0, got " +
                                  std::to_string(v)};
    }
  }
};

// True when the box keeps `gap` from every face of the domain: on each
// axis of the domain and for both faces it either stays at least `gap`
// away from the face plane or, if `may_cross`, extends at least `gap`
// beyond it on both sides. For a convex shape the bounding box touches the
// shape in every axis direction, so this is the shape's own clearance and
// the depth of the part that sticks through the face.
template <typename T>
[[nodiscard]] inline bool
faces_clear(bounding_box_base<T> const& bb,
            std::type_identity_t<domain<T>> const& domain_box, T gap,
            bool may_cross) noexcept {
  auto const lo = bb.bottom_point();
  auto const hi = bb.top_point();
  for (int axis = 0; axis < domain_box.dimension(); ++axis) {
    const auto k = static_cast<std::size_t>(axis);
    for (const T plane : {T{0}, domain_box[k]}) {
      const bool clear = lo[k] >= plane + gap || hi[k] <= plane - gap;
      const bool through = lo[k] <= plane - gap && hi[k] >= plane + gap;
      if (!(clear || (may_cross && through))) return false;
    }
  }
  return true;
}

namespace detail {

// Is `shape` within `gap` of the point p? The point is probed as a circle
// (2D) or sphere (3D) of radius 0 through the collision dispatcher, so
// every shape type with a registered pair against those is measured
// exactly.
template <typename T>
[[nodiscard]] inline bool near_point(shape_base<T> const& shape,
                                     std::array<T, 3> const& p, T gap,
                                     bool three_d) {
  auto const& dispatch = collision_dispatcher<T>::instance();
  if (three_d) {
    sphere<T> probe{p[0], p[1], p[2], T{0}};
    probe.make_bounding_box();
    return dispatch(shape, probe, gap);
  }
  circle<T> probe{p[0], p[1], T{0}};
  probe.make_bounding_box();
  return dispatch(shape, probe, gap);
}

// Does the shape contain the axis-aligned square (2D) or cube (3D) of
// half-width `gap` around p? Its corners are probed: for a convex shape
// that is the whole cube, hence every point within `gap` of p.
template <typename T>
[[nodiscard]] inline bool deep_inside(shape_base<T> const& shape,
                                      std::array<T, 3> const& p, T gap,
                                      bool three_d) {
  const T tiny = gap * T{1e-6};   // "in the shape", decided by the gap test
  for (int corner = 0; corner < (three_d ? 8 : 4); ++corner) {
    const std::array<T, 3> q{p[0] + ((corner & 1) ? gap : -gap),
                             p[1] + ((corner & 2) ? gap : -gap),
                             three_d ? p[2] + ((corner & 4) ? gap : -gap)
                                     : p[2]};
    if (!near_point(shape, q, tiny, three_d)) return false;
  }
  return true;
}

} // namespace detail

// The full wall_gap rule for one shape (bounding box made): no part of
// its surface runs within `gap` of the boundary of the domain without
// crossing it properly.
//
//   faces    faces_clear() above.
//   corners  (2D, and 3D where three crossed faces meet) the shape is at
//            least `gap` away from the corner, or the square / cube of
//            half-width `gap` around the corner lies inside it.
//   edges    (3D, where two crossed faces meet) the shape is at least
//            `gap` away from the edge line, or one of 33 equally spaced
//            points of the line within the shape's extent has such a cube
//            inside the shape.
//
// Without this an inclusion that just covers a corner leaves, in the
// periodic copy at the opposite corner, a piece as small as one likes.
// The probes are conservative: a shape that crosses an edge at a shallow
// angle (a long fibre) may be refused although it would have been fine.
// The edge line is probed over the whole extent of the shape, not only
// inside the box: in a periodic tiling the edge continues, and for `random`
// a shape that grazes the edge inside the box while crossing it outside is
// caught at the corner in between, where the distance to a convex shape is
// smaller still. All of this holds for convex shapes; a bent tube or a
// non-convex mesh is judged by its bounding box at the faces and by the
// probes at edges and corners, which is a good approximation, not a proof.
// `may_cross` = false is the `only_inside` rule: faces only, since a shape
// that stays `gap` inside every face is clear of edges and corners too.
template <typename T>
[[nodiscard]] inline bool
boundary_clear(shape_base<T> const& shape,
               std::type_identity_t<domain<T>> const& domain_box, T gap,
               bool may_cross) {
  auto const& bb = *shape.bounding_box();
  if (!faces_clear(bb, domain_box, gap, may_cross)) return false;
  if (!may_cross) return true;

  const int dim = domain_box.dimension();
  const bool three_d = dim == 3;
  auto const lo = bb.bottom_point();
  auto const hi = bb.top_point();
  // Planes the shape crosses, per axis: 0 = the face at 0, 1 = the face
  // at L. (After faces_clear, "not crossed" means at least gap away.)
  std::array<std::array<bool, 2>, 3> crossed{};
  int axes_crossed = 0;
  for (int axis = 0; axis < dim; ++axis) {
    const auto k = static_cast<std::size_t>(axis);
    crossed[k][0] = lo[k] < T{0} && hi[k] > T{0};
    crossed[k][1] = lo[k] < domain_box[k] && hi[k] > domain_box[k];
    if (crossed[k][0] || crossed[k][1]) ++axes_crossed;
  }
  if (axes_crossed < 2) return true;

  auto corner_ok = [&](std::array<T, 3> const& c) {
    return !detail::near_point(shape, c, gap, three_d) ||
           detail::deep_inside(shape, c, gap, three_d);
  };

  if (!three_d) {
    for (int sx = 0; sx < 2; ++sx) {
      for (int sy = 0; sy < 2; ++sy) {
        if (!crossed[0][static_cast<std::size_t>(sx)] ||
            !crossed[1][static_cast<std::size_t>(sy)])
          continue;
        if (!corner_ok({sx ? domain_box[0] : T{0}, sy ? domain_box[1] : T{0},
                        T{0}}))
          return false;
      }
    }
    return true;
  }

  auto const& dispatch = collision_dispatcher<T>::instance();
  // Edges: the line where the faces (axis i, side si) and (axis j, side sj)
  // meet, running along axis k over the extent of the shape.
  for (std::size_t k = 0; k < 3; ++k) {
    const std::size_t i = (k + 1) % 3, j = (k + 2) % 3;
    for (std::size_t si = 0; si < 2; ++si) {
      for (std::size_t sj = 0; sj < 2; ++sj) {
        if (!crossed[i][si] || !crossed[j][sj]) continue;
        std::array<T, 3> a{}, b{};
        a[i] = b[i] = si ? domain_box[i] : T{0};
        a[j] = b[j] = sj ? domain_box[j] : T{0};
        a[k] = lo[k] - gap;
        b[k] = hi[k] + gap;
        polyline_tube<T> line{std::vector<std::array<T, 3>>{a, b}, T{0}};
        line.make_bounding_box();
        if (!dispatch(shape, line, gap)) continue;   // at least gap away
        bool through = false;
        constexpr int probes = 33;
        for (int n = 0; n < probes && !through; ++n) {
          std::array<T, 3> p = a;
          p[k] = lo[k] + (hi[k] - lo[k]) * static_cast<T>(n) /
                             static_cast<T>(probes - 1);
          through = detail::deep_inside(shape, p, gap, true);
        }
        if (!through) return false;
      }
    }
  }
  // Corners where three crossed faces meet.
  for (std::size_t sx = 0; sx < 2; ++sx) {
    for (std::size_t sy = 0; sy < 2; ++sy) {
      for (std::size_t sz = 0; sz < 2; ++sz) {
        if (!crossed[0][sx] || !crossed[1][sy] || !crossed[2][sz]) continue;
        if (!corner_ok({sx ? domain_box[0] : T{0}, sy ? domain_box[1] : T{0},
                        sz ? domain_box[2] : T{0}}))
          return false;
      }
    }
  }
  return true;
}

// The accepted shapes as the placement loop sees them: their boxes in a
// grid and, when a relative gap is set, their sizes. Index i is
// accepted[i] of the generator.
template <typename T>
class gap_neighbourhood {
public:
  using shape_vector = std::vector<std::unique_ptr<shape_base<T>>>;

  explicit gap_neighbourhood(placement_gap<T> const& gap) : _gap{gap} {}

  // True when `shape` overlaps an accepted shape or is closer to one than
  // the required gap. Without a gap this is the plain two-stage collision
  // test (box overlap, then the pair test).
  [[nodiscard]] bool too_close(shape_base<T> const& shape, T size,
                               shape_vector const& accepted) {
    auto const& bb = *shape.bounding_box();
    auto const bound = collision_dispatcher<T>::instance().bind_lhs(shape);
    const T reach = _gap.reach(size);
    if (!(reach > T{0})) {
      return _grid.any_overlap(bb, [&](std::size_t i) {
        return bound(shape, *accepted[i]);
      });
    }
    auto lo = bb.bottom_point();
    auto hi = bb.top_point();
    for (std::size_t k = 0; k < 3; ++k) {
      lo[k] -= reach;
      hi[k] += reach;
    }
    return _grid.any_overlap(lo, hi, [&](std::size_t i) {
      return bound(shape, *accepted[i], _gap.between(size, _sizes[i]));
    });
  }

  // Call for every shape appended to `accepted`, in the same order.
  void insert(shape_base<T> const& shape, T size) {
    _grid.insert(*shape.bounding_box());
    if (_gap.between_shapes()) _sizes.push_back(size);
  }

private:
  placement_gap<T> _gap;
  aabb_grid<T> _grid;
  std::vector<T> _sizes;
};

} // namespace rvegen
