// Triangle tree of mesh inclusions (#126): the tree-backed pair tests,
// point queries and rotated bounds give the answers of a pass over all
// triangles, the work no longer grows with the triangle count, and all
// copies of a mesh share one tree.
//
// The references in this file are written without the tree and without the
// ray grid: point-in-mesh by the solid-angle (winding number) sum, pair
// tests by GTE distance / intersection queries over every triangle or every
// pair of triangles.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Mathematics/DistPointTriangle.h>
#include <Mathematics/DistSegment3Triangle3.h>
#include <Mathematics/DistTriangle3AlignedBox3.h>
#include <Mathematics/DistTriangle3Triangle3.h>
#include <Mathematics/IntrTriangle3Triangle3.h>

#include "rvegen/rvegen.h"

namespace {

int failures = 0;

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "REQUIRE failed: " #cond " at " __FILE__ ":" << __LINE__    \
                << "\n";                                                       \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

using D = double;
using V = gte::Vector3<D>;
using tri = gte::Triangle3<D>;
using tri_list = std::vector<tri>;
using mesh_t = rvegen::mesh_inclusion<D>;
using P = std::array<D, 3>;
constexpr D pi = std::numbers::pi;

// ---- meshes -----------------------------------------------------------------

// Sphere of radius r around c with 20 * 4^level triangles.
tri_list icosphere(int level, D r, V const& c = V{0, 0, 0}) {
  const D t = (1 + std::sqrt(5.0)) / 2;
  std::vector<V> v{{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
                   {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
                   {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  for (auto& p : v) gte::Normalize(p);
  std::vector<std::array<int, 3>> f{
      {0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
      {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
      {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
      {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1}};
  for (int l = 0; l < level; ++l) {
    std::map<std::pair<int, int>, int> mid;
    std::vector<std::array<int, 3>> next;
    auto middle = [&](int a, int b) {
      const auto key = std::minmax(a, b);
      if (auto it = mid.find(key); it != mid.end()) return it->second;
      V p = v[a] + v[b];
      gte::Normalize(p);
      v.push_back(p);
      return mid[key] = static_cast<int>(v.size()) - 1;
    };
    for (auto const& t3 : f) {
      const int a = middle(t3[0], t3[1]);
      const int b = middle(t3[1], t3[2]);
      const int c2 = middle(t3[2], t3[0]);
      next.push_back({t3[0], a, c2});
      next.push_back({t3[1], b, a});
      next.push_back({t3[2], c2, b});
      next.push_back({a, b, c2});
    }
    f = std::move(next);
  }
  tri_list out;
  out.reserve(f.size());
  for (auto const& t3 : f) {
    tri x;
    for (int k = 0; k < 3; ++k) x.v[k] = c + r * v[t3[k]];
    out.push_back(x);
  }
  return out;
}

// Torus around the z axis: not convex, one component.
tri_list torus(int nu, int nv, D big, D small) {
  auto point = [&](int i, int j) {
    const D u = 2 * pi * (i % nu) / nu;
    const D w = 2 * pi * (j % nv) / nv;
    return V{(big + small * std::cos(w)) * std::cos(u),
             (big + small * std::cos(w)) * std::sin(u), small * std::sin(w)};
  };
  tri_list out;
  for (int i = 0; i < nu; ++i) {
    for (int j = 0; j < nv; ++j) {
      out.push_back({point(i, j), point(i + 1, j), point(i + 1, j + 1)});
      out.push_back({point(i, j), point(i + 1, j + 1), point(i, j + 1)});
    }
  }
  return out;
}

// Thin tetrahedron along z, 0.02 x 0.02 x 1.5: four triangles, a tip that
// goes through a single face of another mesh.
tri_list needle() {
  const std::array<V, 4> p{{{0, 0, 0.75}, {0.01, -0.01, -0.75},
                            {-0.01, -0.01, -0.75}, {0, 0.01, -0.75}}};
  constexpr std::array<std::array<int, 3>, 4> f{{
      {0, 1, 3}, {0, 3, 2}, {0, 2, 1}, {1, 2, 3}}};
  tri_list out;
  for (auto const& t : f) out.push_back({p[t[0]], p[t[1]], p[t[2]]});
  return out;
}

tri_list joined(tri_list a, tri_list const& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

// Two separate balls: two components.
tri_list two_balls(int level) {
  return joined(icosphere(level, 0.6, V{-1.5, 0, 0}),
                icosphere(level, 0.6, V{1.5, 0, 0}));
}

// A ball with a spherical hole: the inner surface bounds a cavity.
tri_list shell(int level) {
  return joined(icosphere(level, 1.0), icosphere(level, 0.5));
}

// ---- placement ----------------------------------------------------------------

struct rng {
  std::mt19937_64 engine;
  D uniform(D a, D b) {
    return a + (b - a) * (static_cast<D>(engine() >> 11) * 0x1.0p-53);
  }
  P direction() {
    for (;;) {
      const P d{uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)};
      const D n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if (n > 0.1 && n <= 1) return {d[0] / n, d[1] / n, d[2] / n};
    }
  }
};

mesh_t placed_copy(mesh_t const& proto, rng& r, P const& where, bool rotate,
                   D scale) {
  mesh_t m{proto};
  if (rotate) m.set_rotation_axis_angle(r.direction(), r.uniform(0, 2 * pi));
  m.set_scale(scale);
  m.set_middle_point(where);
  m.make_bounding_box();
  return m;
}

tri_list world_triangles(mesh_t const& m) {
  tri_list out;
  for (auto const& src : *m.base_triangles()) {
    tri t;
    for (int i = 0; i < 3; ++i) {
      const auto w = m.to_world(src.v[i]);
      t.v[i] = V{w[0], w[1], w[2]};
    }
    out.push_back(t);
  }
  return out;
}

// ---- references ---------------------------------------------------------------

// Winding number of the surface around p: sum of the signed solid angles of
// all triangles (Van Oosterom and Strackee) over 4 pi. 1 inside a solid, 0
// outside and in a cavity.
bool brute_inside(tri_list const& tris, V const& p) {
  D total = 0;
  for (auto const& t : tris) {
    const V a = t.v[0] - p, b = t.v[1] - p, c = t.v[2] - p;
    const D la = gte::Length(a), lb = gte::Length(b), lc = gte::Length(c);
    const D num = gte::DotCross(a, b, c);
    const D den = la * lb * lc + gte::Dot(a, b) * lc + gte::Dot(b, c) * la +
                  gte::Dot(c, a) * lb;
    total += 2 * std::atan2(num, den);
  }
  return std::abs(total) > 2 * pi;
}

struct box3 {
  P lo, hi;
};
box3 bounds_of(tri_list const& tris) {
  box3 b{{1e300, 1e300, 1e300}, {-1e300, -1e300, -1e300}};
  for (auto const& t : tris)
    for (auto const& v : t.v)
      for (int k = 0; k < 3; ++k) {
        b.lo[k] = std::min(b.lo[k], v[k]);
        b.hi[k] = std::max(b.hi[k], v[k]);
      }
  return b;
}

bool brute(tri_list const& m, rvegen::sphere<D> const& s, D gap) {
  gte::DCPQuery<D, V, tri> query;
  const V c{s.center[0], s.center[1], s.center[2]};
  for (auto const& t : m)
    if (query(c, t).distance <= s.radius + gap) return true;
  return brute_inside(m, c);
}

bool brute(tri_list const& m, rvegen::box<D> const& b, D gap) {
  gte::DCPQuery<D, tri, gte::AlignedBox3<D>> query;
  for (auto const& t : m)
    if (query(t, b).distance <= gap) return true;
  return brute_inside(m, V{(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2,
                           (b.min[2] + b.max[2]) / 2});
}

bool brute(tri_list const& m, rvegen::polyline_tube<D> const& t, D gap) {
  gte::DCPQuery<D, gte::Segment3<D>, tri> query;
  auto const& line = t.centerline();
  for (std::size_t i = 0; i + 1 < std::max<std::size_t>(line.size(), 2); ++i) {
    const gte::Segment3<D> seg{line[i], line[std::min(i + 1, line.size() - 1)]};
    for (auto const& x : m)
      if (query(seg, x).distance <= t.radius() + gap) return true;
  }
  return brute_inside(m, line.front());
}

// All pairs of triangles, then every vertex of each mesh against the other.
bool brute(tri_list const& a, tri_list const& b, D gap) {
  gte::TIQuery<D, tri, tri> cross;
  gte::DCPQuery<D, tri, tri> distance;
  for (auto const& x : a)
    for (auto const& y : b) {
      if (gap > 0 ? distance(x, y).distance <= gap : cross(x, y).intersect)
        return true;
    }
  for (auto const& y : b)
    if (brute_inside(a, y.v[0])) return true;
  for (auto const& x : a)
    if (brute_inside(b, x.v[0])) return true;
  return false;
}

// ---- tests --------------------------------------------------------------------

struct tally {
  int yes = 0, no = 0;
  void add(bool v) { v ? ++yes : ++no; }
  // Both answers must occur often, or the comparison proves little.
  [[nodiscard]] bool mixed() const {
    return yes * 6 >= yes + no && no * 6 >= yes + no;
  }
};

std::vector<mesh_t> prototypes() {
  std::vector<mesh_t> out;
  out.emplace_back(icosphere(2, 1.0));       // 320 triangles
  out.emplace_back(torus(16, 10, 0.8, 0.3)); // 320, not convex
  out.emplace_back(two_balls(1));            // 160, two components
  out.emplace_back(shell(1));                // 160, cavity
  out.emplace_back(icosphere(0, 1.0));       // 20: a tree of one leaf pair
  return out;
}

void test_point_queries() {
  rng r{std::mt19937_64{11}};
  for (auto const& proto : prototypes()) {
    for (int placement = 0; placement < 4; ++placement) {
      const auto m = placed_copy(proto, r, {r.uniform(-3, 3), 0.2, -0.4},
                                 placement > 0, r.uniform(0.3, 2.5));
      const auto world = world_triangles(m);
      const auto bb = bounds_of(world);
      tally t;
      for (int i = 0; i < 400; ++i) {
        const P p{r.uniform(bb.lo[0] - 0.1, bb.hi[0] + 0.1),
                  r.uniform(bb.lo[1] - 0.1, bb.hi[1] + 0.1),
                  r.uniform(bb.lo[2] - 0.1, bb.hi[2] + 0.1)};
        const bool got = m.is_inside(p);
        REQUIRE(got == brute_inside(world, V{p[0], p[1], p[2]}));
        t.add(got);
      }
      REQUIRE(t.yes > 20 && t.no > 20);
    }
  }
}

// Sphere, box and tube against a mesh, with and without a gap, through the
// dispatcher in both argument orders.
void test_pairs_with_primitives() {
  auto& dispatch = rvegen::collision_dispatcher<D>::instance();
  rng r{std::mt19937_64{12}};
  tally spheres, boxes, tubes;
  for (auto const& proto : prototypes()) {
    for (int probe = 0; probe < 150; ++probe) {
      const D scale = r.uniform(0.4, 2.0);
      const auto m =
          placed_copy(proto, r, {0.3, -0.2, 0.1}, probe % 3 != 0, scale);
      const auto world = world_triangles(m);
      const auto bb = bounds_of(world);
      const D extent = bb.hi[0] - bb.lo[0];
      const D gap = probe % 2 == 0 ? D{0} : r.uniform(0.01, 0.3) * scale;
      auto const dir = r.direction();
      const D away = r.uniform(0, 0.9) * extent;
      const P at{0.3 + dir[0] * away, -0.2 + dir[1] * away, 0.1 + dir[2] * away};
      auto check = [&](rvegen::shape_base<D>& other, bool expected, tally& t) {
        other.make_bounding_box();
        const bool got = gap > 0 ? dispatch(m, other, gap) : dispatch(m, other);
        const bool swapped =
            gap > 0 ? dispatch(other, m, gap) : dispatch(other, m);
        REQUIRE(got == expected);
        REQUIRE(swapped == expected);
        t.add(got);
      };
      {
        rvegen::sphere<D> s{at[0], at[1], at[2], r.uniform(0.02, 0.3) * extent};
        check(s, brute(world, s, gap), spheres);
      }
      {
        const D hx = r.uniform(0.02, 0.3) * extent;
        const D hy = r.uniform(0.02, 0.3) * extent;
        const D hz = r.uniform(0.02, 0.3) * extent;
        rvegen::box<D> b{at[0], at[1], at[2], 2 * hx, 2 * hy, 2 * hz};
        check(b, brute(world, b, gap), boxes);
      }
      {
        std::vector<P> line{at};
        for (int k = 0; k < probe % 3; ++k) {
          auto const d = r.direction();
          const D step = r.uniform(0.1, 0.4) * extent;
          line.push_back({line.back()[0] + d[0] * step,
                          line.back()[1] + d[1] * step,
                          line.back()[2] + d[2] * step});
        }
        rvegen::polyline_tube<D> t{line, r.uniform(0.01, 0.15) * extent};
        check(t, brute(world, t, gap), tubes);
      }
    }
  }
  REQUIRE(spheres.mixed());
  REQUIRE(boxes.mixed());
  REQUIRE(tubes.mixed());
}

void test_mesh_pairs() {
  auto& dispatch = rvegen::collision_dispatcher<D>::instance();
  rng r{std::mt19937_64{13}};
  const auto protos = prototypes();
  tally plain, gapped;
  for (std::size_t ia = 0; ia < protos.size(); ++ia) {
    for (std::size_t ib = ia; ib < protos.size(); ++ib) {
      for (int probe = 0; probe < 18; ++probe) {
        const auto a = placed_copy(protos[ia], r, {0, 0, 0}, probe % 4 != 0,
                                   r.uniform(0.5, 1.5));
        const auto wa = world_triangles(a);
        const auto ba = bounds_of(wa);
        const D ra = (ba.hi[0] - ba.lo[0]) / 2;
        const D sb = r.uniform(0.3, 1.5);
        auto const dir = r.direction();
        // From concentric (one inside the other, or inside its cavity) to
        // clearly apart.
        const D away = r.uniform(0, 1.6) * (ra + sb);
        const auto b = placed_copy(
            protos[ib], r, {dir[0] * away, dir[1] * away, dir[2] * away},
            probe % 5 != 0, sb);
        const auto wb = world_triangles(b);
        {
          const bool expected = brute(wa, wb, D{0});
          REQUIRE(dispatch(a, b) == expected);
          REQUIRE(dispatch(b, a) == expected);
          plain.add(expected);
        }
        if (probe % 3 == 0) {
          const D gap = r.uniform(0.02, 0.4) * sb;
          const bool expected = brute(wa, wb, gap);
          REQUIRE(dispatch(a, b, gap) == expected);
          REQUIRE(dispatch(b, a, gap) == expected);
          gapped.add(expected);
        }
      }
    }
  }
  REQUIRE(plain.mixed());
  REQUIRE(gapped.mixed());
}

// A contact through one or two triangles: a needle whose tip pokes through
// the surface of a larger mesh, or stops just short of it. A walk that
// loses a single triangle of a leaf shows up here.
void test_single_triangle_contacts() {
  auto& dispatch = rvegen::collision_dispatcher<D>::instance();
  rng r{std::mt19937_64{18}};
  const mesh_t thin{needle()};
  std::vector<mesh_t> targets;
  targets.emplace_back(icosphere(3, 1.0));      // 1280 triangles
  targets.emplace_back(torus(24, 12, 0.8, 0.3)); // 576
  tally plain, gapped;
  for (auto const& target : targets) {
    for (int probe = 0; probe < 700; ++probe) {
      const auto a = placed_copy(target, r, {0, 0, 0}, probe % 2 == 0, 1.0);
      auto const dir = r.direction();
      // Mostly near the surface, where the tip decides.
      const D away = r.uniform(0.9, 2.1);
      const auto b = placed_copy(
          thin, r, {dir[0] * away, dir[1] * away, dir[2] * away}, true,
          r.uniform(0.5, 1.5));
      const auto wa = world_triangles(a);
      const auto wb = world_triangles(b);
      const bool expected = brute(wa, wb, D{0});
      REQUIRE(dispatch(a, b) == expected);
      REQUIRE(dispatch(b, a) == expected);
      plain.add(expected);
      if (probe % 4 == 0) {
        const D gap = r.uniform(0.01, 0.1);
        const bool close = brute(wa, wb, gap);
        REQUIRE(dispatch(a, b, gap) == close);
        REQUIRE(dispatch(b, a, gap) == close);
        gapped.add(close);
      }
    }
  }
  REQUIRE(plain.mixed());
  REQUIRE(gapped.mixed());
}

// A mesh may be several closed surfaces. One of them inside another solid
// is an overlap even when the surfaces never cross and the first vertex of
// the mesh is far away; a solid sitting in the cavity of a shell is not.
void test_components_and_cavities() {
  auto& dispatch = rvegen::collision_dispatcher<D>::instance();
  mesh_t pair{two_balls(2)};
  pair.set_middle_point({0, 0, 0});
  pair.make_bounding_box();
  REQUIRE(pair.component_count() == 2);
  for (D x : {-1.5, 1.5}) {
    // Radius 0.8 around a ball of radius 0.6: nothing crosses.
    mesh_t around{icosphere(2, 0.8)};
    around.set_middle_point({x, 0, 0});
    around.make_bounding_box();
    REQUIRE(dispatch(pair, around));
    REQUIRE(dispatch(around, pair));
    REQUIRE(dispatch(pair, around, 0.01));
    REQUIRE(dispatch(around, pair, 0.01));
  }
  {
    mesh_t between{icosphere(2, 0.5)};   // in the 1.8 wide space between them
    between.set_middle_point({0, 0, 0});
    between.make_bounding_box();
    REQUIRE(!dispatch(pair, between));
    REQUIRE(!dispatch(between, pair));
    REQUIRE(!dispatch(pair, between, 0.3));
    REQUIRE(dispatch(pair, between, 0.45));   // 0.4 apart
  }
  mesh_t hollow{shell(2)};
  hollow.set_middle_point({0, 0, 0});
  hollow.make_bounding_box();
  REQUIRE(hollow.component_count() == 2);
  mesh_t seed{icosphere(2, 0.3)};   // the cavity has radius 0.5
  seed.set_middle_point({0.05, 0, 0});
  seed.make_bounding_box();
  REQUIRE(!dispatch(hollow, seed));
  REQUIRE(!dispatch(seed, hollow));
  REQUIRE(!dispatch(hollow, seed, 0.1));
  REQUIRE(dispatch(hollow, seed, 0.2));   // 0.15 from the cavity wall
  seed.set_middle_point({0.75, 0, 0});    // inside the wall of the shell
  seed.make_bounding_box();
  REQUIRE(dispatch(hollow, seed));
  REQUIRE(dispatch(seed, hollow));
}

// The candidate list of a box holds every triangle that has a point in it.
void test_candidates_are_a_superset() {
  rng r{std::mt19937_64{14}};
  gte::DCPQuery<D, tri, gte::AlignedBox3<D>> query;
  for (auto const& proto : prototypes()) {
    for (int placement = 0; placement < 6; ++placement) {
      const auto m = placed_copy(proto, r, {r.uniform(-50, 50), 3, -7},
                                 placement % 2 == 1, r.uniform(0.01, 30));
      const auto world = world_triangles(m);
      const auto bb = bounds_of(world);
      for (int probe = 0; probe < 40; ++probe) {
        P lo, hi;
        for (int k = 0; k < 3; ++k) {
          const D a = r.uniform(bb.lo[k], bb.hi[k]);
          const D w = r.uniform(0, 0.4) * (bb.hi[k] - bb.lo[k]);
          lo[k] = a - w;
          hi[k] = a + w;
        }
        std::vector<std::size_t> near;
        m.triangles_near(lo, hi, near);
        std::vector<char> listed(world.size(), 0);
        for (auto i : near) {
          REQUIRE(i < world.size());
          REQUIRE(!listed[i]);   // each triangle at most once
          listed[i] = 1;
        }
        const gte::AlignedBox3<D> box{V{lo[0], lo[1], lo[2]},
                                      V{hi[0], hi[1], hi[2]}};
        for (std::size_t i = 0; i < world.size(); ++i) {
          if (query(world[i], box).distance == 0) REQUIRE(listed[i]);
        }
      }
      // A box that is a single point, placed on a vertex: the query box
      // touches the triangles around that vertex and nothing else, so only
      // the rounding margin keeps them in the list.
      for (std::size_t i = 0; i < world.size(); i += 3) {
        for (auto const& v : world[i].v) {
          const P at{v[0], v[1], v[2]};
          std::vector<std::size_t> near;
          m.triangles_near(at, at, near);
          std::size_t expected = 0, found = 0;
          for (std::size_t j = 0; j < world.size(); ++j) {
            const bool touches = world[j].v[0] == v || world[j].v[1] == v ||
                                 world[j].v[2] == v;
            if (!touches) continue;
            ++expected;
            found += std::count(near.begin(), near.end(), j);
          }
          REQUIRE(found == expected);
        }
      }
    }
  }
}

// The bounding box of a rotated copy comes from the tree; it has to be the
// box of the placed vertices, neither larger nor smaller.
void test_rotated_bounds() {
  rng r{std::mt19937_64{15}};
  auto protos = prototypes();
  protos.emplace_back(icosphere(4, 1.0));   // 5120
  for (auto const& proto : protos) {
    for (int placement = 0; placement < 25; ++placement) {
      const auto m = placed_copy(proto, r, {r.uniform(-5, 5), 1, 2}, true,
                                 r.uniform(0.2, 4));
      const auto bb = bounds_of(world_triangles(m));
      const auto lo = m.placed_lo();
      const auto hi = m.placed_hi();
      for (int k = 0; k < 3; ++k) {
        const D tol = 1e-12 * (1 + std::abs(lo[k]) + std::abs(hi[k]));
        REQUIRE(std::abs(lo[k] - bb.lo[k]) <= tol);
        REQUIRE(std::abs(hi[k] - bb.hi[k]) <= tol);
      }
    }
  }
}

// One tree and one triangle list for any number of copies, and a tree
// smaller than the mesh.
void test_copies_share_the_tree() {
  rng r{std::mt19937_64{16}};
  const mesh_t proto{icosphere(4, 1.0)};
  REQUIRE(proto.triangle_tree() != nullptr);
  REQUIRE(proto.tree_memory() > 0);
  REQUIRE(proto.tree_memory() < proto.triangle_count() * sizeof(tri));
  std::vector<std::unique_ptr<rvegen::shape_base<D>>> copies;
  for (int i = 0; i < 200; ++i) {
    auto m = placed_copy(proto, r, {r.uniform(0, 9), 1, 2}, true, 0.5);
    REQUIRE(m.triangle_tree() == proto.triangle_tree());
    REQUIRE(m.base_triangles() == proto.base_triangles());
    copies.push_back(m.clone());
  }
  for (auto const& c : copies) {
    auto const& m = dynamic_cast<mesh_t const&>(*c);
    REQUIRE(m.triangle_tree() == proto.triangle_tree());
    REQUIRE(m.base_triangles() == proto.base_triangles());
  }
  REQUIRE(proto.base_triangles().use_count() >= 200);
}

// The work of a query on a large mesh, counted, not timed: a small box
// brings a small share of the triangles, and two crossing surfaces hand a
// number of triangle pairs to the exact test that grows with the mesh, not
// with its square.
void test_work_on_a_large_mesh() {
  using clock = std::chrono::steady_clock;
  auto ms = [](clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  };
  auto t0 = clock::now();
  const mesh_t proto{icosphere(6, 1.0)};   // 81920 triangles
  const std::size_t n = proto.triangle_count();
  REQUIRE(n == 81920);
  std::cout << "  81920 triangles: built in " << ms(t0) << " ms, tree "
            << proto.tree_memory() / 1024 << " KiB, triangles "
            << n * sizeof(tri) / 1024 << " KiB\n";

  rng r{std::mt19937_64{17}};
  auto a = placed_copy(proto, r, {0, 0, 0}, true, 1.0);
  {
    std::vector<std::size_t> near;
    a.triangles_near({0.95, -0.05, -0.05}, {1.05, 0.05, 0.05}, near);
    REQUIRE(!near.empty());
    REQUIRE(near.size() < n / 100);
    near.clear();
    a.triangles_near({-0.2, -0.2, -0.2}, {0.2, 0.2, 0.2}, near);   // interior
    REQUIRE(near.empty());
  }
  // Against the analytic sphere: the mesh is inscribed, its faces lie
  // within 2e-3 of the unit sphere.
  auto& dispatch = rvegen::collision_dispatcher<D>::instance();
  t0 = clock::now();
  int checks = 0;
  for (int i = 0; i < 300; ++i) {
    auto const d = r.direction();
    const D rs = r.uniform(0.05, 0.4);
    const D dist = r.uniform(0, 1.8);
    rvegen::sphere<D> s{d[0] * dist, d[1] * dist, d[2] * dist, rs};
    s.make_bounding_box();
    if (std::abs(dist - (1 + rs)) < 4e-3) continue;
    REQUIRE(dispatch(a, s) == (dist < 1 + rs));
    const D gap = 0.1;
    if (std::abs(dist - (1 + rs + gap)) < 4e-3) continue;
    REQUIRE(dispatch(a, s, gap) == (dist < 1 + rs + gap));
    checks += 2;
  }
  std::cout << "  mesh x sphere: " << ms(t0) * 1000 / checks << " us per test\n";

  t0 = clock::now();
  checks = 0;
  std::size_t pairs = 0;
  for (int i = 0; i < 60; ++i) {
    auto const d = r.direction();
    const D dist = r.uniform(0, 2.6);
    const auto b =
        placed_copy(proto, r, {d[0] * dist, d[1] * dist, d[2] * dist}, true, 1.0);
    // The exhaustive walk (a test that never stops early) bounds the work
    // of any pair test on these two meshes.
    std::size_t handed = 0;
    REQUIRE(!rvegen::detail::any_mesh_triangle_pair(
        a, b, D{0}, [&](tri const&, tri const&) {
          ++handed;
          return false;
        }));
    pairs = std::max(pairs, handed);
    if (std::abs(dist - 2) < 6e-3) continue;
    REQUIRE(dispatch(a, b) == (dist < 2));
    const D gap = 0.1;
    if (std::abs(dist - 2 - gap) < 6e-3) continue;
    REQUIRE(dispatch(a, b, gap) == (dist < 2 + gap));
    checks += 2;
  }
  std::cout << "  mesh x mesh: " << ms(t0) / checks
            << " ms per test (the exhaustive walk included), at most " << pairs
            << " triangle pairs of " << static_cast<double>(n) * n << "\n";
  REQUIRE(pairs < 64 * n);

  t0 = clock::now();
  int inside = 0;
  const int queries = 20000;
  for (int i = 0; i < queries; ++i)
    inside += a.is_inside({r.uniform(-1.1, 1.1), r.uniform(-1.1, 1.1),
                           r.uniform(-1.1, 1.1)});
  std::cout << "  is_inside: " << ms(t0) * 1000 / queries << " us per query\n";
  // Volume share of the unit ball in the cube of side 2.2: 0.3934.
  REQUIRE(std::abs(static_cast<D>(inside) / queries - 0.3934) < 0.02);

  t0 = clock::now();
  for (int i = 0; i < 50; ++i)
    a.set_rotation_axis_angle(r.direction(), r.uniform(0, 2 * pi));
  std::cout << "  set_rotation: " << ms(t0) * 1000 / 50 << " us\n";
}

} // namespace

int main() {
  rvegen::register_all_collision_pairs<D>();
  test_point_queries();
  test_pairs_with_primitives();
  test_mesh_pairs();
  test_single_triangle_contacts();
  test_components_and_cavities();
  test_candidates_are_a_superset();
  test_rotated_bounds();
  test_copies_share_the_tree();
  test_work_on_a_large_mesh();
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "mesh_acceleration_test: all checks passed\n";
  return EXIT_SUCCESS;
}
