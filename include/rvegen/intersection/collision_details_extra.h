#pragma once

// Precise collision tests for the pairs that used to fall back to AABB
// overlap (#83): rectangle x ellipse, polyline_tube x {sphere, box, tube},
// mesh_inclusion x {sphere, box, tube, mesh}. All distance / intersection
// primitives come from GTE; this file only composes them.
//
// Conventions: touching counts as colliding (<=), matching the existing
// circle/sphere/box tests.
//
// polyline_tube is treated as a swept sphere (capsule chain): a tube with
// one centerline point is a sphere, an empty tube never collides.
//
// mesh_inclusion is treated as a SOLID bounded by a closed triangle mesh:
// two shapes collide if a triangle of one touches the other, or one is
// entirely inside the mesh (checked with the mesh's ray-cast is_inside on
// a representative point). On an open / non-manifold mesh the containment
// part is unreliable (same caveat as mesh_inclusion::is_inside); the
// surface-contact part is always exact. Triangles whose AABB cannot reach
// the other shape's AABB are skipped up front.

#include <algorithm>
#include <array>
#include <vector>

#include <Mathematics/DistPointSegment.h>
#include <Mathematics/DistPointTriangle.h>
#include <Mathematics/DistSegment3AlignedBox3.h>
#include <Mathematics/DistSegment3Triangle3.h>
#include <Mathematics/DistSegmentSegment.h>
#include <Mathematics/DistTriangle3AlignedBox3.h>
#include <Mathematics/IntrTriangle3Triangle3.h>

#include "../shapes/box.h"
#include "../shapes/ellipse.h"
#include "../shapes/mesh_inclusion.h"
#include "../shapes/polyline_tube.h"
#include "../shapes/rectangle.h"
#include "../shapes/sphere.h"

namespace rvegen {

namespace detail {

template <typename T>
using vec3 = gte::Vector3<T>;

template <typename T>
[[nodiscard]] inline vec3<T> to_vec3(std::array<T, 3> const& a) {
  vec3<T> v;
  v[0] = a[0]; v[1] = a[1]; v[2] = a[2];
  return v;
}

template <typename T>
struct bounds3 {
  std::array<T, 3> lo{};
  std::array<T, 3> hi{};
};

template <typename T>
[[nodiscard]] constexpr bool overlap(bounds3<T> const& a,
                                      bounds3<T> const& b) noexcept {
  for (int k = 0; k < 3; ++k) {
    if (a.lo[k] > b.hi[k] || a.hi[k] < b.lo[k]) return false;
  }
  return true;
}

// Visit the centerline as segments; a single point becomes a zero-length
// segment so it behaves as a sphere.
template <typename T, typename F>
[[nodiscard]] inline bool any_tube_segment(polyline_tube<T> const& t, F&& f) {
  auto const& c = t.centerline();
  if (c.empty()) return false;
  if (c.size() == 1) return f(gte::Segment3<T>{c[0], c[0]});
  for (std::size_t i = 1; i < c.size(); ++i) {
    if (f(gte::Segment3<T>{c[i - 1], c[i]})) return true;
  }
  return false;
}

template <typename T>
[[nodiscard]] inline bounds3<T> tube_bounds(polyline_tube<T> const& t) {
  bounds3<T> b;
  auto const& c = t.centerline();
  if (c.empty()) return b;
  for (int k = 0; k < 3; ++k) b.lo[k] = b.hi[k] = c[0][k];
  for (auto const& p : c) {
    for (int k = 0; k < 3; ++k) {
      b.lo[k] = std::min(b.lo[k], p[k]);
      b.hi[k] = std::max(b.hi[k], p[k]);
    }
  }
  for (int k = 0; k < 3; ++k) { b.lo[k] -= t.radius(); b.hi[k] += t.radius(); }
  return b;
}

template <typename T>
[[nodiscard]] inline bounds3<T> mesh_bounds(mesh_inclusion<T> const& m) {
  mesh_inclusion<T> copy{m};   // shares the triangle list: O(1)
  copy.make_bounding_box();
  auto const* bb = copy.bounding_box();
  return {bb->bottom_point(), bb->top_point()};
}

// Translated triangles of `m` whose AABB overlaps `region`.
template <typename T>
[[nodiscard]] inline std::vector<gte::Triangle3<T>>
mesh_triangles_in(mesh_inclusion<T> const& m, bounds3<T> const& region) {
  std::vector<gte::Triangle3<T>> out;
  auto const base = m.base_triangles();
  if (!base) return out;
  auto const& off = m.offset();
  for (auto const& tri : *base) {
    gte::Triangle3<T> t = tri;
    bounds3<T> tb;
    for (int k = 0; k < 3; ++k) {
      for (int i = 0; i < 3; ++i) t.v[i][k] += off[k];
      tb.lo[k] = std::min({t.v[0][k], t.v[1][k], t.v[2][k]});
      tb.hi[k] = std::max({t.v[0][k], t.v[1][k], t.v[2][k]});
    }
    if (overlap(tb, region)) out.push_back(t);
  }
  return out;
}

template <typename T>
[[nodiscard]] inline std::array<T, 3> mesh_first_vertex(mesh_inclusion<T> const& m) {
  auto const base = m.base_triangles();
  if (!base || base->empty()) return {};
  auto const& v = (*base)[0].v[0];
  auto const& off = m.offset();
  return {v[0] + off[0], v[1] + off[1], v[2] + off[2]};
}

} // namespace detail

// ---- rectangle x ellipse ---------------------------------------------------
// Map the ellipse to the unit circle (translate, rotate onto its axes,
// scale by 1/extent): an affine map preserves intersection, the rectangle
// becomes a parallelogram. They intersect iff the circle centre (origin) is
// inside the parallelogram, or a parallelogram edge is within distance 1 of
// the origin (GTE point-segment distance).
template <typename T>
[[nodiscard]] bool collision_details(rectangle<T> const& r,
                                      ellipse<T> const& e) {
  // Ellipse centre inside the rectangle.
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
  origin[0] = T{0}; origin[1] = T{0};
  gte::DCPQuery<T, gte::Vector2<T>, gte::Segment2<T>> query;
  for (int i = 0; i < 4; ++i) {
    gte::Segment2<T> edge{q[i], q[(i + 1) % 4]};
    if (query(origin, edge).sqrDistance <= T{1}) return true;
  }
  return false;
}

template <typename T>
[[nodiscard]] bool collision_details(ellipse<T> const& e,
                                      rectangle<T> const& r) {
  return collision_details(r, e);
}

// ---- sphere x tube ----------------------------------------------------------
template <typename T>
[[nodiscard]] bool collision_details(sphere<T> const& s,
                                      polyline_tube<T> const& t) {
  const auto c = detail::to_vec3<T>({s.center[0], s.center[1], s.center[2]});
  const T reach = s.radius + t.radius();
  gte::DCPQuery<T, detail::vec3<T>, gte::Segment3<T>> query;
  return detail::any_tube_segment(t, [&](gte::Segment3<T> const& seg) {
    return query(c, seg).distance <= reach;
  });
}

template <typename T>
[[nodiscard]] bool collision_details(polyline_tube<T> const& t,
                                      sphere<T> const& s) {
  return collision_details(s, t);
}

// ---- box x tube -------------------------------------------------------------
template <typename T>
[[nodiscard]] bool collision_details(box<T> const& b,
                                      polyline_tube<T> const& t) {
  gte::DCPQuery<T, gte::Segment3<T>, gte::AlignedBox3<T>> query;
  const T r = t.radius();
  return detail::any_tube_segment(t, [&](gte::Segment3<T> const& seg) {
    return query(seg, b).distance <= r;
  });
}

template <typename T>
[[nodiscard]] bool collision_details(polyline_tube<T> const& t,
                                      box<T> const& b) {
  return collision_details(b, t);
}

// ---- tube x tube ------------------------------------------------------------
// Segment-segment distance (GTE) against the radius sum, with a per-segment
// AABB reject so long polylines do not pay O(n*m) distance queries.
template <typename T>
[[nodiscard]] bool collision_details(polyline_tube<T> const& a,
                                      polyline_tube<T> const& b) {
  const T reach = a.radius() + b.radius();
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

// ---- mesh x sphere ----------------------------------------------------------
template <typename T>
[[nodiscard]] bool collision_details(mesh_inclusion<T> const& m,
                                      sphere<T> const& s) {
  const std::array<T, 3> c{s.center[0], s.center[1], s.center[2]};
  detail::bounds3<T> region;
  for (int k = 0; k < 3; ++k) {
    region.lo[k] = c[k] - s.radius;
    region.hi[k] = c[k] + s.radius;
  }
  const auto cv = detail::to_vec3<T>(c);
  gte::DCPQuery<T, detail::vec3<T>, gte::Triangle3<T>> query;
  for (auto const& tri : detail::mesh_triangles_in(m, region)) {
    if (query(cv, tri).distance <= s.radius) return true;
  }
  return m.is_inside(c);   // sphere centre inside the solid
}

template <typename T>
[[nodiscard]] bool collision_details(sphere<T> const& s,
                                      mesh_inclusion<T> const& m) {
  return collision_details(m, s);
}

// ---- mesh x box -------------------------------------------------------------
template <typename T>
[[nodiscard]] bool collision_details(mesh_inclusion<T> const& m,
                                      box<T> const& b) {
  detail::bounds3<T> region;
  for (int k = 0; k < 3; ++k) { region.lo[k] = b.min[k]; region.hi[k] = b.max[k]; }
  gte::DCPQuery<T, gte::Triangle3<T>, gte::AlignedBox3<T>> query;
  for (auto const& tri : detail::mesh_triangles_in(m, region)) {
    if (query(tri, b).distance <= T{0}) return true;
  }
  // Box entirely inside the mesh solid.
  return m.is_inside({(b.min[0] + b.max[0]) / T{2},
                      (b.min[1] + b.max[1]) / T{2},
                      (b.min[2] + b.max[2]) / T{2}});
}

template <typename T>
[[nodiscard]] bool collision_details(box<T> const& b,
                                      mesh_inclusion<T> const& m) {
  return collision_details(m, b);
}

// ---- mesh x tube ------------------------------------------------------------
template <typename T>
[[nodiscard]] bool collision_details(mesh_inclusion<T> const& m,
                                      polyline_tube<T> const& t) {
  if (t.centerline().empty()) return false;
  const auto region = detail::tube_bounds(t);
  const auto tris = detail::mesh_triangles_in(m, region);
  gte::DCPQuery<T, gte::Segment3<T>, gte::Triangle3<T>> query;
  const T r = t.radius();
  if (detail::any_tube_segment(t, [&](gte::Segment3<T> const& seg) {
        for (auto const& tri : tris) {
          if (query(seg, tri).distance <= r) return true;
        }
        return false;
      })) {
    return true;
  }
  auto const& p = t.centerline().front();
  return m.is_inside({p[0], p[1], p[2]});   // tube entirely inside the mesh
}

template <typename T>
[[nodiscard]] bool collision_details(polyline_tube<T> const& t,
                                      mesh_inclusion<T> const& m) {
  return collision_details(m, t);
}

// ---- mesh x mesh ------------------------------------------------------------
// Triangle-triangle intersection (GTE) restricted to the AABB overlap
// region, then mutual containment via one representative vertex.
template <typename T>
[[nodiscard]] bool collision_details(mesh_inclusion<T> const& a,
                                      mesh_inclusion<T> const& b) {
  const auto ba = detail::mesh_bounds(a);
  const auto bb = detail::mesh_bounds(b);
  if (!detail::overlap(ba, bb)) return false;
  detail::bounds3<T> region;
  for (int k = 0; k < 3; ++k) {
    region.lo[k] = std::max(ba.lo[k], bb.lo[k]);
    region.hi[k] = std::min(ba.hi[k], bb.hi[k]);
  }
  const auto ta = detail::mesh_triangles_in(a, region);
  const auto tb = detail::mesh_triangles_in(b, region);
  gte::TIQuery<T, gte::Triangle3<T>, gte::Triangle3<T>> query;
  for (auto const& x : ta) {
    for (auto const& y : tb) {
      if (query(x, y).intersect) return true;
    }
  }
  return a.is_inside(detail::mesh_first_vertex(b)) ||
         b.is_inside(detail::mesh_first_vertex(a));
}

} // namespace rvegen
