#pragma once

// Pair tests with a minimum gap (#111).
//
// collision_details(a, b, gap) answers "are a and b closer than gap?":
// true when their distance is <= gap. With gap = 0 that is the plain
// collision test (touching counts), which stays in collision_details.h /
// collision_details_extra.h and is what the dispatcher calls when no gap
// is asked for, so runs without a gap are unchanged.
//
// A generator that accepts a candidate only when this returns false for
// every neighbour keeps every pair at least `gap` apart, which is what a
// mesher needs: two inclusions 1e-6 apart give sliver elements or no mesh
// at all.
//
// Exact and conservative pairs:
//   * Exact (true iff distance <= gap): every pair of circle, rectangle,
//     sphere, box, polyline_tube and mesh_inclusion, including mixed ones.
//   * Conservative (never accepts a pair closer than gap, may refuse one
//     that is a little farther): pairs with an ellipse. The ellipse is
//     replaced by the smallest ellipse with the same axes that contains
//     everything within `gap` of it (detail::inflated); along the minor
//     axis that is up to about 0.2 * gap more than needed for an aspect
//     ratio of 2 and gap = 0.1 * minor axis.

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <vector>

#include <Mathematics/DistPointSegment.h>
#include <Mathematics/DistPointTriangle.h>
#include <Mathematics/DistSegment3AlignedBox3.h>
#include <Mathematics/DistSegment3Triangle3.h>
#include <Mathematics/DistSegmentSegment.h>
#include <Mathematics/DistTriangle3AlignedBox3.h>
#include <Mathematics/DistTriangle3Triangle3.h>

#include "collision_details.h"
#include "collision_details_extra.h"

namespace rvegen {

namespace detail {

// Smallest ellipse with the axes of `e` that contains every point within
// `gap` of `e`. With semi-axes a >= b the result has a + gap and
// sqrt(b^2 + gap^2 + gap (a^2 + b^2) / a). (Support functions: the
// requirement A^2 s + B^2 (1 - s) >= (h(s) + gap)^2 with
// h(s) = sqrt(a^2 s + b^2 (1 - s)), s in [0, 1], is tight at s = 1 for
// A = a + gap; h is concave, so the smallest admissible B follows from the
// tangent of h at s = 1.) Simply adding gap to both semi-axes is not
// enough: the offset curve leaves that ellipse between the axes.
template <typename T>
[[nodiscard]] inline gte::Ellipse2<T> inflated(gte::Ellipse2<T> e, T gap) {
  const bool first_is_major = e.extent[0] >= e.extent[1];
  const T a = first_is_major ? e.extent[0] : e.extent[1];
  const T b = first_is_major ? e.extent[1] : e.extent[0];
  const T major = a + gap;
  const T minor = std::sqrt(b * b + gap * gap + gap * (a * a + b * b) / a);
  e.extent[0] = first_is_major ? major : minor;
  e.extent[1] = first_is_major ? minor : major;
  return e;
}

// Squared distance between two axis-aligned boxes given by min / max
// corners (anything indexable: std::array, gte::Vector).
template <typename V>
[[nodiscard]] constexpr auto aabb_sqr_distance(V const& amin, V const& amax,
                                               V const& bmin, V const& bmax,
                                               int dims) noexcept {
  using T = std::remove_cvref_t<decltype(amin[0])>;
  T d2{0};
  for (int k = 0; k < dims; ++k) {
    const T s = std::max({T{0}, amin[k] - bmax[k], bmin[k] - amax[k]});
    d2 += s * s;
  }
  return d2;
}

template <typename T>
[[nodiscard]] constexpr bounds3<T> padded(bounds3<T> b, T pad) noexcept {
  for (int k = 0; k < 3; ++k) {
    b.lo[k] -= pad;
    b.hi[k] += pad;
  }
  return b;
}

// The rectangle x ellipse test of collision_details_extra.h for a plain
// GTE ellipse (the inflated one).
template <typename T>
[[nodiscard]] inline bool rectangle_ellipse_collide(rectangle<T> const& r,
                                                     gte::Ellipse2<T> const& e) {
  if (e.center[0] >= r.min[0] && e.center[0] <= r.max[0] &&
      e.center[1] >= r.min[1] && e.center[1] <= r.max[1]) {
    return true;
  }
  const std::array<std::array<T, 2>, 4> corners{{
      {r.min[0], r.min[1]}, {r.max[0], r.min[1]},
      {r.max[0], r.max[1]}, {r.min[0], r.max[1]}}};
  std::array<gte::Vector2<T>, 4> q;
  for (int i = 0; i < 4; ++i) {
    const T dx = corners[i][0] - e.center[0];
    const T dy = corners[i][1] - e.center[1];
    q[i][0] = (dx * e.axis[0][0] + dy * e.axis[0][1]) / e.extent[0];
    q[i][1] = (dx * e.axis[1][0] + dy * e.axis[1][1]) / e.extent[1];
  }
  gte::Vector2<T> origin;
  origin[0] = T{0};
  origin[1] = T{0};
  gte::DCPQuery<T, gte::Vector2<T>, gte::Segment2<T>> query;
  for (int i = 0; i < 4; ++i) {
    gte::Segment2<T> edge{q[i], q[(i + 1) % 4]};
    if (query(origin, edge).sqrDistance <= T{1}) return true;
  }
  return false;
}

} // namespace detail

// ---- 2D ---------------------------------------------------------------------
template <typename T>
[[nodiscard]] constexpr bool collision_details(circle<T> const& lhs,
                                               circle<T> const& rhs,
                                               T gap) noexcept {
  const auto reach = lhs.radius + rhs.radius + gap;
  const auto dx = lhs(0) - rhs(0);
  const auto dy = lhs(1) - rhs(1);
  return reach * reach >= (dx * dx + dy * dy);
}

template <typename T>
[[nodiscard]] constexpr bool collision_details(rectangle<T> const& lhs,
                                               rectangle<T> const& rhs,
                                               T gap) noexcept {
  return detail::aabb_sqr_distance(lhs.min, lhs.max, rhs.min, rhs.max, 2) <=
         gap * gap;
}

template <typename T>
[[nodiscard]] constexpr bool collision_details(circle<T> const& c,
                                               rectangle<T> const& r,
                                               T gap) noexcept {
  const auto closest_x = detail::clamp(c(0), r.min[0], r.max[0]);
  const auto closest_y = detail::clamp(c(1), r.min[1], r.max[1]);
  const auto dx = c(0) - closest_x;
  const auto dy = c(1) - closest_y;
  const auto reach = c.radius + gap;
  return reach * reach >= (dx * dx + dy * dy);
}
template <typename T>
[[nodiscard]] constexpr bool collision_details(rectangle<T> const& r,
                                               circle<T> const& c,
                                               T gap) noexcept {
  return collision_details(c, r, gap);
}

template <typename T>
[[nodiscard]] bool collision_details(ellipse<T> const& lhs,
                                     ellipse<T> const& rhs, T gap) {
  return detail::gte_ellipses_collide<T>(detail::inflated<T>(lhs, gap), rhs);
}

template <typename T>
[[nodiscard]] bool collision_details(circle<T> const& c, ellipse<T> const& e,
                                     T gap) {
  // The circle is the one that grows: exact for the circle, and the
  // ellipse is taken as it is.
  gte::Ellipse2<T> circle_as_ellipse;
  circle_as_ellipse.center = c.center;
  circle_as_ellipse.axis[0] = {T{1}, T{0}};
  circle_as_ellipse.axis[1] = {T{0}, T{1}};
  circle_as_ellipse.extent = {c.radius + gap, c.radius + gap};
  return detail::gte_ellipses_collide<T>(circle_as_ellipse, e);
}
template <typename T>
[[nodiscard]] bool collision_details(ellipse<T> const& e, circle<T> const& c,
                                     T gap) {
  return collision_details(c, e, gap);
}

template <typename T>
[[nodiscard]] bool collision_details(rectangle<T> const& r,
                                     ellipse<T> const& e, T gap) {
  return detail::rectangle_ellipse_collide<T>(r, detail::inflated<T>(e, gap));
}
template <typename T>
[[nodiscard]] bool collision_details(ellipse<T> const& e,
                                     rectangle<T> const& r, T gap) {
  return collision_details(r, e, gap);
}

// ---- 3D primitives -----------------------------------------------------------
template <typename T>
[[nodiscard]] constexpr bool collision_details(sphere<T> const& lhs,
                                               sphere<T> const& rhs,
                                               T gap) noexcept {
  const auto reach = lhs.radius + rhs.radius + gap;
  const auto dx = lhs(0) - rhs(0);
  const auto dy = lhs(1) - rhs(1);
  const auto dz = lhs(2) - rhs(2);
  return reach * reach >= (dx * dx + dy * dy + dz * dz);
}

template <typename T>
[[nodiscard]] constexpr bool collision_details(box<T> const& lhs,
                                               box<T> const& rhs,
                                               T gap) noexcept {
  return detail::aabb_sqr_distance(lhs.min, lhs.max, rhs.min, rhs.max, 3) <=
         gap * gap;
}

template <typename T>
[[nodiscard]] constexpr bool collision_details(sphere<T> const& s,
                                               box<T> const& b,
                                               T gap) noexcept {
  const auto closest_x = detail::clamp(s(0), b.min[0], b.max[0]);
  const auto closest_y = detail::clamp(s(1), b.min[1], b.max[1]);
  const auto closest_z = detail::clamp(s(2), b.min[2], b.max[2]);
  const auto dx = s(0) - closest_x;
  const auto dy = s(1) - closest_y;
  const auto dz = s(2) - closest_z;
  const auto reach = s.radius + gap;
  return reach * reach >= (dx * dx + dy * dy + dz * dz);
}
template <typename T>
[[nodiscard]] constexpr bool collision_details(box<T> const& b,
                                               sphere<T> const& s,
                                               T gap) noexcept {
  return collision_details(s, b, gap);
}

// ---- tubes --------------------------------------------------------------------
template <typename T>
[[nodiscard]] bool collision_details(sphere<T> const& s,
                                     polyline_tube<T> const& t, T gap) {
  const auto c = detail::to_vec3<T>({s.center[0], s.center[1], s.center[2]});
  const T reach = s.radius + t.radius() + gap;
  gte::DCPQuery<T, detail::vec3<T>, gte::Segment3<T>> query;
  return detail::any_tube_segment(t, [&](gte::Segment3<T> const& seg) {
    return query(c, seg).distance <= reach;
  });
}
template <typename T>
[[nodiscard]] bool collision_details(polyline_tube<T> const& t,
                                     sphere<T> const& s, T gap) {
  return collision_details(s, t, gap);
}

template <typename T>
[[nodiscard]] bool collision_details(box<T> const& b,
                                     polyline_tube<T> const& t, T gap) {
  gte::DCPQuery<T, gte::Segment3<T>, gte::AlignedBox3<T>> query;
  const T reach = t.radius() + gap;
  return detail::any_tube_segment(t, [&](gte::Segment3<T> const& seg) {
    return query(seg, b).distance <= reach;
  });
}
template <typename T>
[[nodiscard]] bool collision_details(polyline_tube<T> const& t,
                                     box<T> const& b, T gap) {
  return collision_details(b, t, gap);
}

template <typename T>
[[nodiscard]] bool collision_details(polyline_tube<T> const& a,
                                     polyline_tube<T> const& b, T gap) {
  const T reach = a.radius() + b.radius() + gap;
  gte::DCPQuery<T, gte::Segment3<T>, gte::Segment3<T>> query;
  auto seg_box = [](gte::Segment3<T> const& s, T pad) {
    detail::bounds3<T> r;
    for (int k = 0; k < 3; ++k) {
      r.lo[k] = std::min(s.p[0][k], s.p[1][k]) - pad;
      r.hi[k] = std::max(s.p[0][k], s.p[1][k]) + pad;
    }
    return r;
  };
  return detail::any_tube_segment(a, [&](gte::Segment3<T> const& sa) {
    const auto abox = seg_box(sa, reach);
    return detail::any_tube_segment(b, [&](gte::Segment3<T> const& sb) {
      if (!detail::overlap(abox, seg_box(sb, T{0}))) return false;
      return query(sa, sb).distance <= reach;
    });
  });
}

// ---- meshes -------------------------------------------------------------------
template <typename T>
[[nodiscard]] bool collision_details(mesh_inclusion<T> const& m,
                                     sphere<T> const& s, T gap) {
  const std::array<T, 3> c{s.center[0], s.center[1], s.center[2]};
  const T reach = s.radius + gap;
  detail::bounds3<T> region;
  for (int k = 0; k < 3; ++k) {
    region.lo[k] = c[k] - reach;
    region.hi[k] = c[k] + reach;
  }
  const auto cv = detail::to_vec3<T>(c);
  gte::DCPQuery<T, detail::vec3<T>, gte::Triangle3<T>> query;
  if (detail::any_mesh_triangle_in(m, region, [&](gte::Triangle3<T> const& tri) {
        return query(cv, tri).distance <= reach;
      })) {
    return true;
  }
  if (!detail::overlap(detail::mesh_bounds(m), region)) return false;
  return m.is_inside(c);   // sphere centre inside the solid
}
template <typename T>
[[nodiscard]] bool collision_details(sphere<T> const& s,
                                     mesh_inclusion<T> const& m, T gap) {
  return collision_details(m, s, gap);
}

template <typename T>
[[nodiscard]] bool collision_details(mesh_inclusion<T> const& m,
                                     box<T> const& b, T gap) {
  detail::bounds3<T> region;
  for (int k = 0; k < 3; ++k) {
    region.lo[k] = b.min[k] - gap;
    region.hi[k] = b.max[k] + gap;
  }
  gte::DCPQuery<T, gte::Triangle3<T>, gte::AlignedBox3<T>> query;
  if (detail::any_mesh_triangle_in(m, region, [&](gte::Triangle3<T> const& tri) {
        return query(tri, b).distance <= gap;
      })) {
    return true;
  }
  if (!detail::overlap(detail::mesh_bounds(m), region)) return false;
  // Box entirely inside the mesh solid.
  return m.is_inside({(b.min[0] + b.max[0]) / T{2},
                      (b.min[1] + b.max[1]) / T{2},
                      (b.min[2] + b.max[2]) / T{2}});
}
template <typename T>
[[nodiscard]] bool collision_details(box<T> const& b,
                                     mesh_inclusion<T> const& m, T gap) {
  return collision_details(m, b, gap);
}

template <typename T>
[[nodiscard]] bool collision_details(mesh_inclusion<T> const& m,
                                     polyline_tube<T> const& t, T gap) {
  if (t.centerline().empty()) return false;
  const auto region = detail::padded(detail::tube_bounds(t), gap);
  const auto tris = detail::mesh_triangles_in(m, region);
  gte::DCPQuery<T, gte::Segment3<T>, gte::Triangle3<T>> query;
  const T reach = t.radius() + gap;
  if (detail::any_tube_segment(t, [&](gte::Segment3<T> const& seg) {
        for (auto const& tb : tris) {
          if (query(seg, tb.tri).distance <= reach) return true;
        }
        return false;
      })) {
    return true;
  }
  if (!detail::overlap(detail::mesh_bounds(m), region)) return false;
  auto const& p = t.centerline().front();
  return m.is_inside({p[0], p[1], p[2]});   // tube entirely inside the mesh
}
template <typename T>
[[nodiscard]] bool collision_details(polyline_tube<T> const& t,
                                     mesh_inclusion<T> const& m, T gap) {
  return collision_details(m, t, gap);
}

// Triangle-triangle distance (GTE) over the triangles of both meshes that
// lie within `gap` of the other mesh's bounding box; a pair is measured
// only when its boxes are within `gap`. Containment (one mesh inside the
// other, surfaces farther apart than gap) is caught by the vertex tests.
template <typename T>
[[nodiscard]] bool collision_details(mesh_inclusion<T> const& a,
                                     mesh_inclusion<T> const& b, T gap) {
  const auto ba = detail::mesh_bounds(a);
  const auto bb = detail::mesh_bounds(b);
  if (!detail::overlap(detail::padded(ba, gap), bb)) return false;
  const auto ta = detail::mesh_triangles_in(a, detail::padded(bb, gap));
  const auto tb = detail::mesh_triangles_in(b, detail::padded(ba, gap));
  constexpr std::size_t samples = 2;
  auto vertex_inside = [&](std::vector<detail::tri_box<T>> const& tris,
                           mesh_inclusion<T> const& other) {
    if (tris.empty()) return false;
    const std::size_t step = std::max<std::size_t>(1, tris.size() / samples);
    for (std::size_t i = 0; i < tris.size(); i += step) {
      auto const& v = tris[i].tri.v[0];
      if (other.is_inside({v[0], v[1], v[2]})) return true;
    }
    return false;
  };
  if (vertex_inside(tb, a) || vertex_inside(ta, b)) return true;
  gte::DCPQuery<T, gte::Triangle3<T>, gte::Triangle3<T>> query;
  for (auto const& x : ta) {
    const auto xbox = detail::padded(x.box, gap);
    for (auto const& y : tb) {
      if (!detail::overlap(xbox, y.box)) continue;
      if (query(x.tri, y.tri).distance <= gap) return true;
    }
  }
  return a.is_inside(detail::mesh_first_vertex(b)) ||
         b.is_inside(detail::mesh_first_vertex(a));
}

} // namespace rvegen
