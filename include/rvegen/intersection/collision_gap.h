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
// Every pair is exact (true iff the distance is <= gap, to within
// rounding): circle, rectangle, sphere, box, polyline_tube and
// mesh_inclusion pairs from closed-form or GTE distance queries. Pairs with
// an ellipse have no closed form: two ellipses that bound the offset
// region from inside and outside settle most calls, and the band in
// between is decided by the distance itself, minimised over the boundary
// of one shape (detail::min_over_ellipse).

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <type_traits>
#include <vector>

#include <Mathematics/DistPointHyperellipsoid.h>
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

// The ellipse with both semi-axes grown by `gap`: contained in the region
// within `gap` of `e` (which bulges beyond it between the axes), so a hit
// on it proves "closer than gap".
template <typename T>
[[nodiscard]] inline gte::Ellipse2<T> grown(gte::Ellipse2<T> e, T gap) {
  e.extent[0] += gap;
  e.extent[1] += gap;
  return e;
}

// Smallest value of f(point) over the boundary of `e`, for f the distance
// to a convex set that does not meet `e`. Such an f is convex in the
// plane, so along the boundary it falls to the nearest point and rises
// again on both sides (a second, higher dip can exist on the far side):
// 64 samples find the right valley, a golden-section search finishes it.
// The result is never below the true minimum and within about 1e-12 of it
// relative to the size of the ellipse.
template <typename T, typename F>
[[nodiscard]] inline T min_over_ellipse(gte::Ellipse2<T> const& e, F&& f) {
  constexpr int samples = 64;
  constexpr T two_pi = T{2} * std::numbers::pi_v<T>;
  auto at = [&](T t) {
    const T u = e.extent[0] * std::cos(t);
    const T v = e.extent[1] * std::sin(t);
    gte::Vector2<T> p;
    p[0] = e.center[0] + u * e.axis[0][0] + v * e.axis[1][0];
    p[1] = e.center[1] + u * e.axis[0][1] + v * e.axis[1][1];
    return f(p);
  };
  T best = at(T{0});
  int best_i = 0;
  for (int i = 1; i < samples; ++i) {
    const T v = at(two_pi * static_cast<T>(i) / samples);
    if (v < best) {
      best = v;
      best_i = i;
    }
  }
  T lo = two_pi * static_cast<T>(best_i - 1) / samples;
  T hi = two_pi * static_cast<T>(best_i + 1) / samples;
  const T ratio = (std::sqrt(T{5}) - T{1}) / T{2};
  T x1 = hi - ratio * (hi - lo);
  T x2 = lo + ratio * (hi - lo);
  T f1 = at(x1);
  T f2 = at(x2);
  for (int it = 0; it < 80 && hi - lo > T{1e-12}; ++it) {
    if (f1 < f2) {
      hi = x2;
      x2 = x1;
      f2 = f1;
      x1 = hi - ratio * (hi - lo);
      f1 = at(x1);
    } else {
      lo = x1;
      x1 = x2;
      f1 = f2;
      x2 = lo + ratio * (hi - lo);
      f2 = at(x2);
    }
  }
  return std::min({best, f1, f2});
}

// Distance from a point outside `e` to `e` (GTE).
template <typename T>
[[nodiscard]] inline T point_ellipse_distance(gte::Vector2<T> const& p,
                                              gte::Ellipse2<T> const& e) {
  gte::DCPQuery<T, gte::Vector2<T>, gte::Hyperellipsoid<2, T>> query;
  return query(p, e).distance;
}

// The computed distance is an upper bound that can exceed the true one by
// rounding: a pair this close to the gap counts as too close.
template <typename T>
[[nodiscard]] constexpr bool within_gap(T distance, T gap) noexcept {
  return distance <= gap * (T{1} + T{1e-9});
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
  // Outside the outer bound: farther than gap. Inside the inner bound:
  // closer. In between, the distance decides.
  if (!detail::gte_ellipses_collide<T>(detail::inflated<T>(lhs, gap), rhs))
    return false;
  if (detail::gte_ellipses_collide<T>(detail::grown<T>(lhs, gap), rhs))
    return true;
  gte::Ellipse2<T> const& other = lhs;
  const T distance = detail::min_over_ellipse<T>(
      rhs, [&](gte::Vector2<T> const& p) {
        return detail::point_ellipse_distance<T>(p, other);
      });
  return detail::within_gap(distance, gap);
}

template <typename T>
[[nodiscard]] bool collision_details(circle<T> const& c, ellipse<T> const& e,
                                     T gap) {
  // Centre in the ellipse, or the ellipse within radius + gap of the
  // centre (GTE point-ellipse distance). Also right for a radius of 0,
  // which is how a single point is probed (placement_gap.h).
  const T dx = c.center[0] - e.center[0];
  const T dy = c.center[1] - e.center[1];
  const T u = (dx * e.axis[0][0] + dy * e.axis[0][1]) / e.extent[0];
  const T v = (dx * e.axis[1][0] + dy * e.axis[1][1]) / e.extent[1];
  if (u * u + v * v <= T{1}) return true;
  gte::Ellipse2<T> const& ellipse_only = e;
  return detail::point_ellipse_distance<T>(c.center, ellipse_only) <=
         c.radius + gap;
}
template <typename T>
[[nodiscard]] bool collision_details(ellipse<T> const& e, circle<T> const& c,
                                     T gap) {
  return collision_details(c, e, gap);
}

template <typename T>
[[nodiscard]] bool collision_details(rectangle<T> const& r,
                                     ellipse<T> const& e, T gap) {
  if (!detail::rectangle_ellipse_collide<T>(r, detail::inflated<T>(e, gap)))
    return false;
  if (detail::rectangle_ellipse_collide<T>(r, detail::grown<T>(e, gap)))
    return true;
  const T distance = detail::min_over_ellipse<T>(
      e, [&](gte::Vector2<T> const& p) {
        const T dx = std::max({T{0}, r.min[0] - p[0], p[0] - r.max[0]});
        const T dy = std::max({T{0}, r.min[1] - p[1], p[1] - r.max[1]});
        return std::sqrt(dx * dx + dy * dy);
      });
  return detail::within_gap(distance, gap);
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
