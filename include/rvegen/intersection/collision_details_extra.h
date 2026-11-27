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

// Bounds of the placed mesh (rotation, scale and translation applied),
// read from the shape's cache: O(1).
template <typename T>
[[nodiscard]] inline bounds3<T> mesh_bounds(mesh_inclusion<T> const& m) {
  bounds3<T> b;
  b.lo = m.placed_lo();
  b.hi = m.placed_hi();
  return b;
}

// Placed (world-space) triangle together with its AABB.
template <typename T>
struct tri_box {
  gte::Triangle3<T> tri;
  bounds3<T> box;
};

// Maps a base triangle of `m` through the shape's placement
// (mesh_inclusion::to_world: rotation, scale, translation).
template <typename T>
[[nodiscard]] inline tri_box<T> placed(gte::Triangle3<T> const& src,
                                        mesh_inclusion<T> const& m) {
  tri_box<T> r{src, {}};
  for (int i = 0; i < 3; ++i) {
    const auto w = m.to_world(src.v[i]);
    for (int k = 0; k < 3; ++k) r.tri.v[i][k] = w[k];
  }
  for (int k = 0; k < 3; ++k) {
    r.box.lo[k] = std::min({r.tri.v[0][k], r.tri.v[1][k], r.tri.v[2][k]});
    r.box.hi[k] = std::max({r.tri.v[0][k], r.tri.v[1][k], r.tri.v[2][k]});
  }
  return r;
}

// Visit placed triangles of `m` whose AABB overlaps `region`, without
// materialising them; stops at the first `f` that returns true.
template <typename T, typename F>
[[nodiscard]] inline bool any_mesh_triangle_in(mesh_inclusion<T> const& m,
                                                bounds3<T> const& region,
                                                F&& f) {
  auto const base = m.base_triangles();
  if (!base || !overlap(mesh_bounds(m), region)) return false;
  for (auto const& src : *base) {
    auto const t = placed(src, m);
    if (overlap(t.box, region) && f(t.tri)) return true;
  }
  return false;
}

// Placed triangles of `m` whose AABB overlaps `region`.
template <typename T>
[[nodiscard]] inline std::vector<tri_box<T>>
mesh_triangles_in(mesh_inclusion<T> const& m, bounds3<T> const& region) {
  std::vector<tri_box<T>> out;
  auto const base = m.base_triangles();
  if (!base || !overlap(mesh_bounds(m), region)) return out;
  for (auto const& src : *base) {
    auto t = placed(src, m);
    if (overlap(t.box, region)) out.push_back(t);
  }
  return out;
}

template <typename T>
[[nodiscard]] inline std::array<T, 3> mesh_first_vertex(mesh_inclusion<T> const& m) {
  auto const base = m.base_triangles();
  if (!base || base->empty()) return {};
  return m.to_world((*base)[0].v[0]);
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
  if (detail::any_mesh_triangle_in(m, region, [&](gte::Triangle3<T> const& tri) {
        return query(cv, tri).distance <= s.radius;
      })) {
    return true;
  }
  if (!detail::overlap(detail::mesh_bounds(m), region)) return false;
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
  if (detail::any_mesh_triangle_in(m, region, [&](gte::Triangle3<T> const& tri) {
        return query(tri, b).distance <= T{0};
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
        for (auto const& tb : tris) {
          if (query(seg, tb.tri).distance <= r) return true;
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
                                      mesh_inclusion<T> const& m) {
  return collision_details(m, t);
}

// ---- mesh x mesh ------------------------------------------------------------
// Cheap accepts first (a few vertices of one mesh inside the other), then
// triangle-triangle intersection (GTE) restricted to the AABB overlap
// region: in-region triangles of both meshes are cached with their AABBs
// (triangles outside the region are never copied), a pair is only tested
// when the triangle AABBs overlap, and the loop exits on the first hit. Finally mutual containment via one vertex.
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
  const auto tb = detail::mesh_triangles_in(b, region);

  // Sampled vertices of the in-region triangles of one mesh inside the other.
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
  if (vertex_inside(tb, a)) return true;
  const auto ta = detail::mesh_triangles_in(a, region);
  if (vertex_inside(ta, b)) return true;

  gte::TIQuery<T, gte::Triangle3<T>, gte::Triangle3<T>> query;
  for (auto const& x : ta) {
    for (auto const& y : tb) {
      if (!detail::overlap(x.box, y.box)) continue;
      if (query(x.tri, y.tri).intersect) return true;
    }
  }
  return a.is_inside(detail::mesh_first_vertex(b)) ||
         b.is_inside(detail::mesh_first_vertex(a));
}

} // namespace rvegen
