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
#include <limits>
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

template <typename T>
[[nodiscard]] constexpr bounds3<T> padded(bounds3<T> b, T pad) noexcept {
  for (int k = 0; k < 3; ++k) {
    b.lo[k] -= pad;
    b.hi[k] += pad;
  }
  return b;
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

// Bounds of the tube, from the shape itself (polyline_tube::extents).
template <typename T>
[[nodiscard]] inline bounds3<T> tube_bounds(polyline_tube<T> const& t) {
  const auto e = t.extents();
  return bounds3<T>{e[0], e[1]};
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
// materialising them; stops at the first `f` that returns true. The
// candidates come from the triangle tree of the mesh: every triangle with a
// point in `region` is among them, the order is the tree's, and a triangle
// whose box overlaps the region while the triangle itself stays outside may
// be left out. `f` must therefore be false for a triangle that does not
// reach into `region`, which holds for all distance tests below (the region
// is the other shape's box, padded by the reach of the test).
template <typename T, typename F>
[[nodiscard]] inline bool any_mesh_triangle_in(mesh_inclusion<T> const& m,
                                                bounds3<T> const& region,
                                                F&& f) {
  auto const base = m.base_triangles();
  if (!base || !overlap(mesh_bounds(m), region)) return false;
  std::vector<std::size_t> near;
  m.triangles_near(region.lo, region.hi, near);
  for (auto i : near) {
    auto const t = placed((*base)[i], m);
    if (overlap(t.box, region) && f(t.tri)) return true;
  }
  return false;
}

// Placed triangles of `m` whose AABB overlaps `region`; candidates as in
// any_mesh_triangle_in.
template <typename T>
[[nodiscard]] inline std::vector<tri_box<T>>
mesh_triangles_in(mesh_inclusion<T> const& m, bounds3<T> const& region) {
  std::vector<tri_box<T>> out;
  auto const base = m.base_triangles();
  if (!base || !overlap(mesh_bounds(m), region)) return out;
  std::vector<std::size_t> near;
  m.triangles_near(region.lo, region.hi, near);
  out.reserve(near.size());
  for (auto i : near) {
    auto t = placed((*base)[i], m);
    if (overlap(t.box, region)) out.push_back(t);
  }
  return out;
}

// Whether a whole connected component of `inner`'s surface lies inside the
// solid `outer`: one point per component is enough once the caller knows
// that no triangles of the two cross. Testing only the first vertex of the
// mesh would miss a second component (a mesh may be several closed
// surfaces) sitting inside `outer` while the first one is outside.
template <typename T>
[[nodiscard]] inline bool mesh_component_inside(mesh_inclusion<T> const& inner,
                                                mesh_inclusion<T> const& outer) {
  for (auto const& p : inner.component_points()) {
    if (outer.is_inside(p)) return true;
  }
  return false;
}

// World-space bounds of node `i` of the triangle tree of `m`: the box of
// the eight placed corners of the node box, which contains every placed
// triangle of the node. Widened by a rounding margin, since a placed vertex
// and a placed corner are rounded independently.
template <typename T>
[[nodiscard]] inline bounds3<T> mesh_node_bounds(
    mesh_inclusion<T> const& m, mesh_triangle_tree<T> const& tree,
    std::size_t i) {
  auto const& lo = tree.lo(i);
  auto const& hi = tree.hi(i);
  bounds3<T> b;
  b.lo.fill(std::numeric_limits<T>::max());
  b.hi.fill(std::numeric_limits<T>::lowest());
  T reach = T{0};
  for (int corner = 0; corner < 8; ++corner) {
    const auto w = m.to_world(vec3<T>{(corner & 1) ? hi[0] : lo[0],
                                      (corner & 2) ? hi[1] : lo[1],
                                      (corner & 4) ? hi[2] : lo[2]});
    for (int k = 0; k < 3; ++k) {
      b.lo[k] = std::min(b.lo[k], w[k]);
      b.hi[k] = std::max(b.hi[k], w[k]);
      reach = std::max(reach, std::abs(w[k]));
    }
  }
  return padded(b, reach * mesh_rounding_margin<T>);
}

// Whether `f(x, y)` holds for a placed triangle x of `a` and y of `b`
// whose boxes are within `pad` of each other; stops at the first hit.
//
// The two triangle trees are walked against each other: a pair of nodes is
// dropped as soon as their world boxes are farther apart than `pad`, the
// larger node is split otherwise, and two leaves have their triangles
// tested pair by pair. Every pair of triangles within `pad` is reached (a
// node box contains its triangles), so the answer is that of the test over
// all pairs; the cost follows the number of node pairs that are close, which
// for two surfaces that cross or pass each other is a small part of each.
template <typename T, typename F>
[[nodiscard]] inline bool any_mesh_triangle_pair(mesh_inclusion<T> const& a,
                                                 mesh_inclusion<T> const& b,
                                                 T pad, F&& f) {
  auto const* ta = a.triangle_tree();
  auto const* tb = b.triangle_tree();
  auto const base_a = a.base_triangles();
  auto const base_b = b.base_triangles();
  if (!ta || !tb || ta->empty() || tb->empty() || !base_a || !base_b) {
    return false;
  }
  struct entry {
    std::size_t ia, ib;
    bounds3<T> ba;   // already padded
    bounds3<T> bb;
  };
  auto size = [](bounds3<T> const& x) {
    return (x.hi[0] - x.lo[0]) + (x.hi[1] - x.lo[1]) + (x.hi[2] - x.lo[2]);
  };
  auto apart = [](bounds3<T> const& x, bounds3<T> const& y) {   // centres
    T sum = T{0};
    for (int k = 0; k < 3; ++k) {
      const T d = (x.lo[k] + x.hi[k]) - (y.lo[k] + y.hi[k]);
      sum += d * d;
    }
    return sum;
  };
  std::vector<entry> stack;
  stack.reserve(128);
  stack.push_back({0, 0, padded(mesh_node_bounds(a, *ta, 0), pad),
                   mesh_node_bounds(b, *tb, 0)});
  std::vector<tri_box<T>> la, lb;
  while (!stack.empty()) {
    const entry e = stack.back();
    stack.pop_back();
    if (!overlap(e.ba, e.bb)) continue;
    const bool leaf_a = ta->leaf(e.ia);
    const bool leaf_b = tb->leaf(e.ib);
    if (leaf_a && leaf_b) {
      la.clear();
      lb.clear();
      for (auto i : ta->triangles(e.ia)) {
        la.push_back(placed((*base_a)[i], a));
        la.back().box = padded(la.back().box, pad);
      }
      for (auto i : tb->triangles(e.ib)) lb.push_back(placed((*base_b)[i], b));
      for (auto const& x : la) {
        for (auto const& y : lb) {
          if (overlap(x.box, y.box) && f(x.tri, y.tri)) return true;
        }
      }
      continue;
    }
    // The child whose box centre is nearer to the other node goes on the
    // stack last and is looked at first: a hit, if there is one, is then
    // found after a few node pairs instead of after most of them.
    if (!leaf_a && (leaf_b || size(e.ba) >= size(e.bb))) {
      const entry l{2 * e.ia + 1, e.ib,
                    padded(mesh_node_bounds(a, *ta, 2 * e.ia + 1), pad), e.bb};
      const entry r{2 * e.ia + 2, e.ib,
                    padded(mesh_node_bounds(a, *ta, 2 * e.ia + 2), pad), e.bb};
      const bool left_first = apart(l.ba, l.bb) <= apart(r.ba, r.bb);
      stack.push_back(left_first ? r : l);
      stack.push_back(left_first ? l : r);
    } else {
      const entry l{e.ia, 2 * e.ib + 1, e.ba,
                    mesh_node_bounds(b, *tb, 2 * e.ib + 1)};
      const entry r{e.ia, 2 * e.ib + 2, e.ba,
                    mesh_node_bounds(b, *tb, 2 * e.ib + 2)};
      const bool left_first = apart(l.ba, l.bb) <= apart(r.ba, r.bb);
      stack.push_back(left_first ? r : l);
      stack.push_back(left_first ? l : r);
    }
  }
  return false;
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
// Two solids overlap exactly when their surfaces cross or a whole connected
// component of one lies inside the other. The second is one point-in-mesh
// query per component and comes first, because it also settles most deep
// overlaps at once (a point of a component inside the other solid means an
// overlap either way). The first is triangle-triangle intersection (GTE)
// over the pairs the two triangle trees bring together.
template <typename T>
[[nodiscard]] bool collision_details(mesh_inclusion<T> const& a,
                                      mesh_inclusion<T> const& b) {
  if (!detail::overlap(detail::mesh_bounds(a), detail::mesh_bounds(b))) {
    return false;
  }
  if (detail::mesh_component_inside(b, a) ||
      detail::mesh_component_inside(a, b)) {
    return true;
  }
  gte::TIQuery<T, gte::Triangle3<T>, gte::Triangle3<T>> query;
  return detail::any_mesh_triangle_pair(
      a, b, T{0},
      [&](gte::Triangle3<T> const& x, gte::Triangle3<T> const& y) {
        return query(x, y).intersect;
      });
}

} // namespace rvegen
