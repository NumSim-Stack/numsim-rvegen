// Minimum gap between inclusions and to the domain faces (#111).
//
//  1. Pair tests: for every registered pair, collision_details(a, b, gap)
//     is true exactly when the two shapes are closer than gap (configurations
//     with a distance known in closed form); ellipse pairs are conservative
//     and the test pins how conservative.
//  2. Generated packs: every pair of accepted shapes (periodic copies and
//     all lattice images included) is at least min_gap apart. The distance
//     is measured here with points sampled on the shapes, not with the
//     library's pair tests, and the same config without a gap is shown to
//     violate the bound, so the check can fail.
//  3. min_gap_relative and wall_gap, checked exactly on circles and spheres.
//  4. All gaps 0: the accepted list is bit-identical to a config without
//     the keys.
//  5. The gaps reach the report, the run summary and the gmsh / voxel
//     headers; a run without gaps writes no such line.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rvegen/pipeline.h"
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

using namespace rvegen;
using nlohmann::json;
using D = double;
using P3 = std::array<D, 3>;
using shape_vector = std::vector<std::unique_ptr<shape_base<D>>>;

constexpr D pi = std::numbers::pi;

// ---------------------------------------------------------------------------
// 1. Pair tests
// ---------------------------------------------------------------------------

// The shapes are `distance` apart: the gap test must say "too close" for a
// gap just above the distance and "fine" just below, in both argument
// orders, and agree with the plain test at gap 0.
template <typename A, typename B>
void exact(A const& a, B const& b, D distance, char const* what) {
  const bool below = collision_details(a, b, distance * (1 - 1e-9));
  const bool above = collision_details(a, b, distance * (1 + 1e-9));
  const bool below_r = collision_details(b, a, distance * (1 - 1e-9));
  const bool above_r = collision_details(b, a, distance * (1 + 1e-9));
  if (below || !above || below_r || !above_r) {
    std::cerr << "pair " << what << " at distance " << distance << ": gap test "
              << below << above << below_r << above_r << " (want 0101)\n";
    ++failures;
  }
  REQUIRE(collision_details(a, b, D{0}) == collision_details(a, b));
  REQUIRE(!collision_details(a, b));
}

// Closed 12-triangle cube mesh of half-size h.
mesh_inclusion<D> cube_mesh(D h, D cx, D cy, D cz) {
  std::array<P3, 8> v;
  for (int i = 0; i < 8; ++i)
    v[i] = {(i & 1) ? h : -h, (i & 2) ? h : -h, (i & 4) ? h : -h};
  constexpr int faces[12][3] = {
      {0, 1, 3}, {0, 3, 2}, {4, 5, 7}, {4, 7, 6}, {0, 1, 5}, {0, 5, 4},
      {2, 3, 7}, {2, 7, 6}, {0, 2, 6}, {0, 6, 4}, {1, 3, 7}, {1, 7, 5}};
  typename mesh_inclusion<D>::triangle_list tris;
  for (auto const& f : faces) {
    gte::Triangle3<D> t;
    for (int k = 0; k < 3; ++k)
      for (int c = 0; c < 3; ++c) t.v[k][c] = v[f[k]][c];
    tris.push_back(t);
  }
  mesh_inclusion<D> m{std::move(tris)};
  m.set_middle_point({cx, cy, cz});
  return m;
}

void test_pairs_exact() {
  // 2D
  exact(circle<D>{0, 0, 0.3}, circle<D>{1, 0, 0.2}, 0.5, "circle-circle");
  exact(circle<D>{0, 0, 0.3}, circle<D>{0.6, 0.8, 0.2}, 0.5, "circle-circle diagonal");
  exact(rectangle<D>{0, 0, 1, 1}, rectangle<D>{2, 0, 1, 1}, 1.0, "rect-rect face");
  exact(rectangle<D>{0, 0, 1, 1}, rectangle<D>{1.3, 1.4, 1, 1}, 0.5, "rect-rect corner");
  exact(circle<D>{2, 0, 0.5}, rectangle<D>{0, 0, 1, 1}, 1.0, "circle-rect face");
  exact(circle<D>{1.1, 1.3, 0.25}, rectangle<D>{0, 0, 1, 1}, 0.75, "circle-rect corner");
  // 3D
  exact(sphere<D>{0, 0, 0, 0.3}, sphere<D>{0, 0, 1, 0.2}, 0.5, "sphere-sphere");
  exact(box<D>{0, 0, 0, 1, 1, 1}, box<D>{0, 0, 2.5, 1, 1, 1}, 1.5, "box-box face");
  exact(box<D>{0, 0, 0, 1, 1, 1}, box<D>{1.2, 1.3, 2.2, 1, 1, 1},
        std::sqrt(0.04 + 0.09 + 1.44), "box-box corner");
  exact(sphere<D>{0, 0, 2, 0.5}, box<D>{0, 0, 0, 1, 1, 1}, 1.0, "sphere-box face");
  exact(sphere<D>{0.8, 0.9, 1.7, 0.1}, box<D>{0, 0, 0, 1, 1, 1},
        std::sqrt(0.09 + 0.16 + 1.44) - 0.1, "sphere-box corner");
  // Tubes: straight along x from 0 to 4, radius 0.2.
  polyline_tube<D> tube{std::vector<P3>{{0, 0, 0}, {4, 0, 0}}, 0.2};
  exact(sphere<D>{2, 1, 0, 0.3}, tube, 0.5, "sphere-tube side");
  exact(sphere<D>{5, 0, 0, 0.3}, tube, 0.5, "sphere-tube end");
  exact(box<D>{2, 0, 1.5, 1, 1, 1}, tube, 0.8, "box-tube");
  exact(polyline_tube<D>{std::vector<P3>{{0, 1, 0}, {4, 1, 0}}, 0.3}, tube, 0.5,
        "tube-tube parallel");
  exact(polyline_tube<D>{std::vector<P3>{{2, -3, 1.5}, {2, 3, 1.5}}, 0.3}, tube,
        1.0, "tube-tube crossing");
  // Meshes: cube [-0.5, 0.5]^3.
  auto cube = cube_mesh(0.5, 0, 0, 0);
  exact(cube, sphere<D>{0, 0, 1.5, 0.3}, 0.7, "mesh-sphere face");
  exact(cube, sphere<D>{1.1, 1.3, 0, 0.25}, 0.75, "mesh-sphere edge");
  exact(cube, box<D>{2, 0, 0, 1, 1, 1}, 1.0, "mesh-box face");
  exact(cube, box<D>{1.3, 1.4, 0, 1, 1, 1}, 0.5, "mesh-box edge");
  exact(cube, polyline_tube<D>{std::vector<P3>{{-2, 0, 1.5}, {2, 0, 1.5}}, 0.2},
        0.8, "mesh-tube");
  exact(cube, cube_mesh(0.5, 2, 0, 0), 1.0, "mesh-mesh face");
  exact(cube, cube_mesh(0.5, 1.3, 1.4, 0), 0.5, "mesh-mesh edge");
  exact(cube, cube_mesh(0.5, 1.2, 1.3, 2.2), std::sqrt(0.04 + 0.09 + 1.44),
        "mesh-mesh corner");
}

// Reference distances are measured by sampling the boundaries: a coarse
// pass finds the nearest pair of samples, a fine pass resamples both
// curves around it. The result is an upper bound within 1e-6 of the
// distance for the sizes used here.
P3 ellipse_point(ellipse<D> const& e, D t) {
  const D u = e.extent[0] * std::cos(t), v = e.extent[1] * std::sin(t);
  return {e.center[0] + u * e.axis[0][0] + v * e.axis[1][0],
          e.center[1] + u * e.axis[0][1] + v * e.axis[1][1], 0};
}
std::vector<P3> ellipse_points(ellipse<D> const& e, int n) {
  std::vector<P3> out;
  for (int i = 0; i < n; ++i) out.push_back(ellipse_point(e, 2 * pi * i / n));
  return out;
}
// Smallest distance between curve points a(s), b(t), s and t in [0, 1).
template <typename A, typename B>
D curve_distance(A&& a, B&& b) {
  constexpr int coarse = 300, fine = 300;
  auto d2 = [](P3 const& p, P3 const& q) {
    return (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]);
  };
  std::vector<P3> pa, pb;
  for (int i = 0; i < coarse; ++i) {
    pa.push_back(a(static_cast<D>(i) / coarse));
    pb.push_back(b(static_cast<D>(i) / coarse));
  }
  D best = std::numeric_limits<D>::max();
  int bi = 0, bj = 0;
  for (int i = 0; i < coarse; ++i)
    for (int j = 0; j < coarse; ++j)
      if (const D v = d2(pa[i], pb[j]); v < best) {
        best = v;
        bi = i;
        bj = j;
      }
  pa.clear();
  pb.clear();
  for (int k = 0; k <= fine; ++k) {
    const D w = (static_cast<D>(k) / fine - 0.5) * 4.0 / coarse;
    pa.push_back(a(static_cast<D>(bi) / coarse + w));
    pb.push_back(b(static_cast<D>(bj) / coarse + w));
  }
  for (auto const& p : pa)
    for (auto const& q : pb) best = std::min(best, d2(p, q));
  return std::sqrt(best);
}
D ellipse_distance(ellipse<D> const& a, ellipse<D> const& b) {
  return curve_distance([&](D s) { return ellipse_point(a, 2 * pi * s); },
                        [&](D t) { return ellipse_point(b, 2 * pi * t); });
}
// Unit square [0, 1]^2 boundary, parameter in [0, 1) (periodic).
P3 unit_square_point(D t) {
  t = (t - std::floor(t)) * 4;
  if (t < 1) return {t, 0, 0};
  if (t < 2) return {1, t - 1, 0};
  if (t < 3) return {3 - t, 1, 0};
  return {0, 4 - t, 0};
}
void test_pairs_ellipse() {
  // Along the major axes the inflated ellipse is exact.
  exact(ellipse<D>{0, 0, 0.5, 0.2, 0}, ellipse<D>{2, 0, 0.5, 0.2, 0}, 1.0,
        "ellipse-ellipse major");
  exact(circle<D>{2, 0, 0.5}, ellipse<D>{0, 0, 0.5, 0.2, 0}, 1.0,
        "circle-ellipse major");
  exact(circle<D>{0, 1, 0.3}, ellipse<D>{0, 0, 0.5, 0.2, 0}, 0.5,
        "circle-ellipse minor");
  exact(rectangle<D>{2.5, 0, 1, 1}, ellipse<D>{0, 0, 0.5, 0.2, 0}, 1.5,
        "rect-ellipse major");

  // Everywhere else the distance is measured here by sampling both
  // boundaries (within 1e-5 of the true distance): the test must say "too
  // close" for a gap just above it and "fine" just below. Aspect ratios up
  // to 8, all relative rotations and directions.
  int n = 0;
  for (D aspect : {1.5, 2.5, 4.0, 8.0}) {
    for (D rot_a : {0.0, 0.4, 1.1}) {
      for (D rot_b : {0.0, 0.7, 1.5}) {
        for (D dir : {0.0, 0.5, 1.0, 1.57, 2.3}) {
          const D b = 0.1, a = aspect * b;
          ellipse<D> ea{0, 0, a, b, rot_a};
          for (D centre_dist : {2.05 * a, 1.6 * a, 1.2 * a, 0.9 * a}) {
            ellipse<D> eb{centre_dist * std::cos(dir), centre_dist * std::sin(dir),
                          a, b, rot_b};
            if (collision_details(ea, eb)) continue;   // overlapping
            const D dist = ellipse_distance(ea, eb);
            if (dist > 3 * b) continue;
            ++n;
            REQUIRE(collision_details(ea, eb, dist * 1.0001 + 3e-6));
            REQUIRE(collision_details(eb, ea, dist * 1.0001 + 3e-6));
            REQUIRE(!collision_details(ea, eb, dist * 0.9999 - 3e-6));
            REQUIRE(!collision_details(eb, ea, dist * 0.9999 - 3e-6));
          }
        }
      }
    }
  }
  REQUIRE(n > 100);

  // Rectangle x ellipse, the ellipse rotated towards a corner.
  int n_rect = 0;
  rectangle<D> r{0.5, 0.5, 1.0, 1.0};
  for (D rot : {0.0, 0.6, pi / 4, 1.2}) {
    for (D off : {1.35, 1.5, 1.8}) {
      ellipse<D> e{off, off, 0.3, 0.1, rot};
      if (collision_details(r, e)) continue;
      const D dist = curve_distance(
          unit_square_point, [&](D t) { return ellipse_point(e, 2 * pi * t); });
      REQUIRE(collision_details(r, e, dist * 1.0001 + 3e-6));
      REQUIRE(!collision_details(e, r, dist * 0.9999 - 3e-6));
      ++n_rect;
    }
  }
  REQUIRE(n_rect >= 6);
}

void test_dispatcher() {
  auto& d = collision_dispatcher<D>::instance();
  d.clear();
  register_all_collision_pairs<D>();
  // Every registered pair has its gap-aware test.
  REQUIRE(d.size() > 0 && d.size() == d.gap_size());

  circle<D> a{0, 0, 0.3}, b{1, 0, 0.2};
  a.make_bounding_box();
  b.make_bounding_box();
  REQUIRE(!d(a, b) && !d(a, b, 0.0) && !d(a, b, -1.0));
  REQUIRE(!d(a, b, 0.499) && d(a, b, 0.501));
  auto bound = d.bind_lhs(a);
  REQUIRE(!bound(a, b, 0.499) && bound(a, b, 0.501));

  // A pair without a registered test falls back to the distance between
  // the bounding boxes: [-0.3, 0.3]^2 and [0.8, 1.2] x [-0.2, 0.2] here.
  d.clear();
  REQUIRE(!d(a, b, 0.499) && d(a, b, 0.501));
  circle<D> c{1, 1, 0.2};                    // boxes sqrt(2) * 0.5 apart
  c.make_bounding_box();
  REQUIRE(!d(a, c, 0.70) && d(a, c, 0.71));
  register_all_collision_pairs<D>();
  REQUIRE(!d(a, c, 0.91) && d(a, c, 0.92));  // the circles: sqrt(2) - 0.5
}

// ---------------------------------------------------------------------------
// 2. Generated packs, measured by sampling
// ---------------------------------------------------------------------------

void sphere_points(P3 const& c, D r, int n, std::vector<P3>& out) {
  const D golden = pi * (3 - std::sqrt(5.0));
  for (int i = 0; i < n; ++i) {
    const D z = 1 - 2 * (i + 0.5) / n;
    const D s = std::sqrt(1 - z * z);
    out.push_back({c[0] + r * s * std::cos(golden * i),
                   c[1] + r * s * std::sin(golden * i), c[2] + r * z});
  }
}

// Points of the solid (on its surface wherever that matters), dense enough
// that the nearest pair of two shapes is within ~2 % of a 0.05 feature.
std::vector<P3> sample(shape_base<D> const& s) {
  std::vector<P3> out;
  if (auto const* c = dynamic_cast<circle<D> const*>(&s)) {
    for (int i = 0; i < 720; ++i) {
      const D t = 2 * pi * i / 720;
      out.push_back({c->center[0] + c->radius * std::cos(t),
                     c->center[1] + c->radius * std::sin(t), 0});
    }
  } else if (auto const* e = dynamic_cast<ellipse<D> const*>(&s)) {
    out = ellipse_points(*e, 720);
  } else if (auto const* r = dynamic_cast<rectangle<D> const*>(&s)) {
    for (int i = 0; i <= 200; ++i) {
      const D t = static_cast<D>(i) / 200;
      const D x = r->min[0] + t * (r->max[0] - r->min[0]);
      const D y = r->min[1] + t * (r->max[1] - r->min[1]);
      out.push_back({x, r->min[1], 0});
      out.push_back({x, r->max[1], 0});
      out.push_back({r->min[0], y, 0});
      out.push_back({r->max[0], y, 0});
    }
  } else if (auto const* sp = dynamic_cast<sphere<D> const*>(&s)) {
    sphere_points({sp->center[0], sp->center[1], sp->center[2]}, sp->radius,
                  1500, out);
  } else if (auto const* b = dynamic_cast<box<D> const*>(&s)) {
    constexpr int n = 16;
    for (int axis = 0; axis < 3; ++axis) {
      const int u = (axis + 1) % 3, v = (axis + 2) % 3;
      for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
          P3 p{};
          p[u] = b->min[u] + (b->max[u] - b->min[u]) * i / n;
          p[v] = b->min[v] + (b->max[v] - b->min[v]) * j / n;
          p[axis] = b->min[axis];
          out.push_back(p);
          p[axis] = b->max[axis];
          out.push_back(p);
        }
      }
    }
  } else if (auto const* t = dynamic_cast<polyline_tube<D> const*>(&s)) {
    auto const& line = t->centerline();
    for (std::size_t i = 1; i < line.size(); ++i) {
      const P3 p0{line[i - 1][0], line[i - 1][1], line[i - 1][2]};
      const P3 p1{line[i][0], line[i][1], line[i][2]};
      const D len = std::sqrt((p1[0] - p0[0]) * (p1[0] - p0[0]) +
                              (p1[1] - p0[1]) * (p1[1] - p0[1]) +
                              (p1[2] - p0[2]) * (p1[2] - p0[2]));
      const int steps = std::max(2, static_cast<int>(len / (0.2 * t->radius())));
      for (int k = 0; k <= steps; ++k) {
        const D w = static_cast<D>(k) / steps;
        sphere_points({p0[0] + w * (p1[0] - p0[0]), p0[1] + w * (p1[1] - p0[1]),
                       p0[2] + w * (p1[2] - p0[2])},
                      t->radius(), 90, out);
      }
    }
  } else if (auto const* m = dynamic_cast<mesh_inclusion<D> const*>(&s)) {
    auto const base = m->base_triangles();
    constexpr int n = 10;
    for (auto const& tri : *base) {
      std::array<P3, 3> w;
      for (int k = 0; k < 3; ++k) w[k] = m->to_world(tri.v[k]);
      for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n - i; ++j) {
          const D a = static_cast<D>(i) / n, b2 = static_cast<D>(j) / n;
          const D c = 1 - a - b2;
          out.push_back({a * w[0][0] + b2 * w[1][0] + c * w[2][0],
                         a * w[0][1] + b2 * w[1][1] + c * w[2][1],
                         a * w[0][2] + b2 * w[1][2] + c * w[2][2]});
        }
      }
    }
  } else {
    std::cerr << "sample: unknown shape type\n";
    ++failures;
  }
  return out;
}

struct sampled {
  std::vector<P3> points;
  P3 lo{}, hi{};
  bool primary{true};
};

std::vector<sampled> sample_all(shape_vector const& shapes) {
  std::vector<sampled> out;
  for (auto const& s : shapes) {
    sampled x;
    x.points = sample(*s);
    x.primary = s->is_primary();
    x.lo = {1e300, 1e300, 1e300};
    x.hi = {-1e300, -1e300, -1e300};
    for (auto const& p : x.points) {
      for (int k = 0; k < 3; ++k) {
        x.lo[k] = std::min(x.lo[k], p[k]);
        x.hi[k] = std::max(x.hi[k], p[k]);
      }
    }
    out.push_back(std::move(x));
  }
  return out;
}

// Distance between a and b shifted by `shift`; pairs whose boxes are
// farther apart than `cutoff` are skipped (returns a large number).
D sampled_distance(sampled const& a, sampled const& b, P3 const& shift, D cutoff) {
  D box2 = 0;
  for (int k = 0; k < 3; ++k) {
    const D s = std::max({0.0, a.lo[k] - (b.hi[k] + shift[k]),
                          (b.lo[k] + shift[k]) - a.hi[k]});
    box2 += s * s;
  }
  if (box2 > cutoff * cutoff) return std::numeric_limits<D>::max();
  D best = std::numeric_limits<D>::max();
  for (auto const& p : a.points) {
    // only points of a near b's box can be nearest
    D pb = 0;
    for (int k = 0; k < 3; ++k) {
      const D s = std::max({0.0, (b.lo[k] + shift[k]) - p[k],
                            p[k] - (b.hi[k] + shift[k])});
      pb += s * s;
    }
    if (pb > cutoff * cutoff) continue;
    for (auto const& q : b.points) {
      const D dx = p[0] - q[0] - shift[0], dy = p[1] - q[1] - shift[1],
              dz = p[2] - q[2] - shift[2];
      best = std::min(best, dx * dx + dy * dy + dz * dz);
    }
  }
  return best == std::numeric_limits<D>::max() ? best : std::sqrt(best);
}

// Smallest sampled distance between two different inclusions of a pack.
// Non-periodic: all pairs of shapes. Periodic: all pairs of primaries over
// all 3^d lattice shifts (an inclusion against its own images included),
// which is every pair the tiling contains, with or without a stored copy;
// and the stored copies against all other stored shapes, which is what the
// writers see.
D smallest_distance(shape_vector const& shapes, domain<D> const& box,
                    bool periodic, D cutoff) {
  auto const all = sample_all(shapes);
  D best = std::numeric_limits<D>::max();
  const P3 none{0, 0, 0};
  for (std::size_t i = 0; i < all.size(); ++i) {
    for (std::size_t j = i + 1; j < all.size(); ++j) {
      auto const ii = shapes[i]->inclusion_id();
      if (ii && ii == shapes[j]->inclusion_id()) continue;   // one inclusion
      best = std::min(best, sampled_distance(all[i], all[j], none, cutoff));
    }
  }
  if (!periodic) return best;
  const int zmax = box.is_3d() ? 1 : 0;
  for (std::size_t i = 0; i < all.size(); ++i) {
    if (!all[i].primary) continue;
    for (std::size_t j = i; j < all.size(); ++j) {
      if (!all[j].primary) continue;
      for (int sx = -1; sx <= 1; ++sx)
        for (int sy = -1; sy <= 1; ++sy)
          for (int sz = -zmax; sz <= zmax; ++sz) {
            if (i == j && sx == 0 && sy == 0 && sz == 0) continue;
            best = std::min(best, sampled_distance(all[i], all[j],
                                                   {sx * box[0], sy * box[1],
                                                    sz * box[2]},
                                                   cutoff));
          }
    }
  }
  return best;
}

json base_2d() {
  return json::parse(R"({
    "domain_box": [1.0, 1.0],
    "seed": 11,
    "distributions": {
      "any": {"type": "uniform_real", "a": 0.0, "b": 1.0},
      "in": {"type": "uniform_real", "a": 0.12, "b": 0.88},
      "r": {"type": "uniform_real", "a": 0.03, "b": 0.05},
      "ea": {"type": "uniform_real", "a": 0.05, "b": 0.08},
      "eb": {"type": "uniform_real", "a": 0.02, "b": 0.035},
      "angle": {"type": "uniform_real", "a": 0.0, "b": 3.14159},
      "w": {"type": "uniform_real", "a": 0.05, "b": 0.09}
    },
    "shapes": [
      {"type": "circle_input", "pos_x_dist": "any", "pos_y_dist": "any",
       "radius_dist": "r"},
      {"type": "ellipse_input", "pos_x_dist": "any", "pos_y_dist": "any",
       "radius_a_dist": "ea", "radius_b_dist": "eb", "rotation_dist": "angle"},
      {"type": "rectangle_input", "pos_x_dist": "any", "pos_y_dist": "any",
       "width_dist": "w", "height_dist": "w"}
    ],
    "generator": {"type": "periodic", "max_attempts": 200000},
    "termination": {"type": "number_of_inclusions", "target": 45}
  })");
}

json base_3d() {
  auto cfg = json::parse(R"({
    "domain_box": [1.0, 1.0, 1.0],
    "seed": 12,
    "distributions": {
      "any": {"type": "uniform_real", "a": 0.0, "b": 1.0},
      "r": {"type": "uniform_real", "a": 0.05, "b": 0.08},
      "w": {"type": "uniform_real", "a": 0.08, "b": 0.14},
      "tr": {"type": "uniform_real", "a": 0.03, "b": 0.04}
    },
    "shapes": [
      {"type": "sphere_input", "pos_x_dist": "any", "pos_y_dist": "any",
       "pos_z_dist": "any", "radius_dist": "r"},
      {"type": "box_input", "pos_x_dist": "any", "pos_y_dist": "any",
       "pos_z_dist": "any", "width_dist": "w", "height_dist": "w",
       "depth_dist": "w"},
      {"type": "polyline_tube_input", "start_x_dist": "any",
       "start_y_dist": "any", "start_z_dist": "any", "end_x_dist": "any",
       "end_y_dist": "any", "end_z_dist": "any", "radius_dist": "tr"},
      {"type": "mesh_inclusion_input", "mesh_path": "",
       "position_x_dist": "any", "position_y_dist": "any",
       "position_z_dist": "any"}
    ],
    "generator": {"type": "periodic", "max_attempts": 200000},
    "termination": {"type": "number_of_inclusions", "target": 60}
  })");
  cfg["shapes"][3]["mesh_path"] =
      std::string{RVEGEN_EXAMPLES_DIR} + "/data/octahedron.stl";
  return cfg;
}

void test_generated_packs() {
  struct run {
    char const* name;
    json config;
    D gap;
  };
  std::vector<run> runs;
  for (char const* gen : {"only_inside", "random", "periodic"}) {
    auto c2 = base_2d();
    c2["generator"]["type"] = gen;
    auto c3 = base_3d();
    c3["generator"]["type"] = gen;
    if (std::string{gen} == "only_inside") {
      // inside the box only: keep the centres away from the faces
      for (auto& s : c2["shapes"])
        for (auto& [k, v] : s.items())
          if (v == "any" && k.find("pos") != std::string::npos) v = "in";
      c3["distributions"]["any"] = {{"type", "uniform_real"}, {"a", 0.2}, {"b", 0.8}};
      c3["termination"]["target"] = 40;   // the inner 0.6^3 holds fewer
    }
    runs.push_back({gen, c2, 0.02});
    runs.push_back({gen, c3, 0.03});
  }
  for (auto& r : runs) {
    const bool periodic = r.config["generator"]["type"] == "periodic";
    auto with = r.config;
    with["generator"]["min_gap"] = r.gap;
    auto const packed = run_pipeline(with);
    auto const loose = run_pipeline(r.config);
    REQUIRE(!packed.shapes.empty() && !loose.shapes.empty());
    REQUIRE(!packed.report.target_missed());
    const D d_with = smallest_distance(packed.shapes, packed.domain_box,
                                       periodic, 4 * r.gap);
    const D d_without = smallest_distance(loose.shapes, loose.domain_box,
                                          periodic, 4 * r.gap);
    const bool three_d = packed.domain_box.is_3d();
    std::cout << "  " << r.name << (three_d ? " 3D" : " 2D") << ": "
              << packed.shapes.size() << " shapes, smallest distance "
              << d_with << " (min_gap " << r.gap << "), without a gap "
              << d_without << "\n";
    // Sampled distances are never below the true distance, so this holds
    // whenever the generator kept its promise...
    REQUIRE(d_with >= r.gap);
    // ...and it is not vacuous: the same config without a gap packs closer.
    REQUIRE(d_without < 0.5 * r.gap);
    REQUIRE(packed.report.min_gap == r.gap);
    REQUIRE(loose.report.min_gap == 0.0);
  }
}

// ---------------------------------------------------------------------------
// 3. Relative gap and wall gap, exact on circles and spheres
// ---------------------------------------------------------------------------

json circles(char const* generator) {
  auto cfg = json::parse(R"({
    "domain_box": [1.0, 1.0],
    "seed": 3,
    "distributions": {
      "x": {"type": "uniform_real", "a": 0.0, "b": 1.0},
      "r": {"type": "uniform_real", "a": 0.02, "b": 0.07}
    },
    "shapes": [{"type": "circle_input", "pos_x_dist": "x",
                "pos_y_dist": "x", "radius_dist": "r"}],
    "generator": {"type": "periodic", "max_attempts": 300000},
    "termination": {"type": "number_of_inclusions", "target": 40}
  })");
  cfg["generator"]["type"] = generator;
  return cfg;
}

json spheres(char const* generator) {
  auto cfg = json::parse(R"({
    "domain_box": [1.0, 1.0, 1.0],
    "seed": 4,
    "distributions": {
      "x": {"type": "uniform_real", "a": 0.0, "b": 1.0},
      "r": {"type": "uniform_real", "a": 0.05, "b": 0.1}
    },
    "shapes": [{"type": "sphere_input", "pos_x_dist": "x", "pos_y_dist": "x",
                "pos_z_dist": "x", "radius_dist": "r"}],
    "generator": {"type": "periodic", "max_attempts": 300000},
    "termination": {"type": "number_of_inclusions", "target": 50}
  })");
  cfg["generator"]["type"] = generator;
  return cfg;
}

struct ball {
  P3 c;
  D r;
  bool primary;
};

std::vector<ball> balls(shape_vector const& shapes) {
  std::vector<ball> out;
  for (auto const& s : shapes) {
    if (auto const* c = dynamic_cast<circle<D> const*>(s.get()))
      out.push_back({{c->center[0], c->center[1], 0}, c->radius, s->is_primary()});
    else if (auto const* sp = dynamic_cast<sphere<D> const*>(s.get()))
      out.push_back({{sp->center[0], sp->center[1], sp->center[2]}, sp->radius,
                     s->is_primary()});
  }
  return out;
}

void test_relative_gap() {
  constexpr D rel = 0.5;
  for (auto const& cfg0 : {circles("periodic"), circles("random"),
                           spheres("periodic"), spheres("only_inside")}) {
    auto cfg = cfg0;
    if (cfg["generator"]["type"] == "only_inside")
      cfg["distributions"]["x"] = {{"type", "uniform_real"}, {"a", 0.15}, {"b", 0.85}};
    cfg["generator"]["min_gap_relative"] = rel;
    cfg["generator"]["min_gap"] = 0.004;       // the larger of the two applies
    auto const res = run_pipeline(cfg);
    auto const b = balls(res.shapes);
    REQUIRE(b.size() == res.shapes.size() && b.size() >= 40);
    const bool periodic = cfg["generator"]["type"] == "periodic";
    const int zmax = res.domain_box.is_3d() ? 1 : 0;
    D worst = std::numeric_limits<D>::max();   // smallest distance / required
    D tightest = std::numeric_limits<D>::max();
    for (std::size_t i = 0; i < b.size(); ++i) {
      for (std::size_t j = i; j < b.size(); ++j) {
        if (periodic && (!b[i].primary || !b[j].primary)) continue;
        for (int sx = -1; sx <= 1; ++sx)
          for (int sy = -1; sy <= 1; ++sy)
            for (int sz = -zmax; sz <= zmax; ++sz) {
              const bool same = sx == 0 && sy == 0 && sz == 0;
              if (!periodic && !same) continue;
              if (i == j && same) continue;
              const D dx = b[i].c[0] - b[j].c[0] - sx * res.domain_box[0];
              const D dy = b[i].c[1] - b[j].c[1] - sy * res.domain_box[1];
              const D dz = b[i].c[2] - b[j].c[2] - sz * res.domain_box[2];
              const D dist = std::sqrt(dx * dx + dy * dy + dz * dz) - b[i].r - b[j].r;
              const D need = std::max(0.004, rel * 2 * std::min(b[i].r, b[j].r));
              worst = std::min(worst, dist / need);
              tightest = std::min(tightest, dist);
            }
      }
    }
    REQUIRE(worst >= 1.0 - 1e-12);
    // The relative part is what binds: every pair is farther apart than
    // the absolute 0.004 alone would ask for.
    REQUIRE(tightest > 0.019);
    REQUIRE(res.report.min_gap_relative == rel && res.report.min_gap == 0.004);
  }
}

// Two inclusions that both stay inside the box have no stored copies, yet
// are neighbours through a periodic face. Hand-placed, so the outcome does
// not depend on what a seed happens to produce.
void test_periodic_images() {
  // A at x = 0.03 and B at x = 0.97, radius 0.02, same y: 0.9 apart inside
  // the box, 0.02 apart through the face x = 0 / 1. Neither crosses a face.
  auto two = json::parse(R"({
    "domain_box": [1.0, 1.0],
    "seed": 1,
    "distributions": {
      "xa": {"type": "constant", "value": 0.03},
      "xb": {"type": "constant", "value": 0.97},
      "y": {"type": "constant", "value": 0.5},
      "r": {"type": "constant", "value": 0.02}
    },
    "shapes": [
      {"type": "circle_input", "pos_x_dist": "xa", "pos_y_dist": "y",
       "radius_dist": "r"},
      {"type": "circle_input", "pos_x_dist": "xb", "pos_y_dist": "y",
       "radius_dist": "r"}
    ],
    "generator": {"type": "periodic", "max_attempts": 6},
    "termination": {"type": "number_of_inclusions", "target": 2}
  })");
  pipeline_options quiet;
  quiet.warn = [](std::string const&) {};
  REQUIRE(run_pipeline(two, quiet).shapes.size() == 2);       // no gap: both fit
  two["generator"]["min_gap"] = 0.019;
  REQUIRE(run_pipeline(two, quiet).shapes.size() == 2);       // 0.02 apart
  two["generator"]["min_gap"] = 0.021;
  auto const refused = run_pipeline(two, quiet);
  REQUIRE(refused.shapes.size() == 1);                        // B is too close
  REQUIRE(refused.report.target_missed());
  // The same through a corner: A near (0, 0), B near (1, 1).
  two["distributions"]["y"] = {{"type", "constant"}, {"value", 0.03}};
  two["distributions"]["yb"] = {{"type", "constant"}, {"value", 0.97}};
  two["shapes"][1]["pos_y_dist"] = "yb";                      // 0.0449 apart
  two["generator"]["min_gap"] = 0.044;
  REQUIRE(run_pipeline(two, quiet).shapes.size() == 2);
  two["generator"]["min_gap"] = 0.046;
  REQUIRE(run_pipeline(two, quiet).shapes.size() == 1);
  // In the non-periodic generators the two are 0.9 apart and both fit.
  two["generator"]["type"] = "only_inside";
  REQUIRE(run_pipeline(two, quiet).shapes.size() == 2);

  // An inclusion and its own image: radius 0.49 in the unit box leaves
  // 0.02 between the circle and its copy one period away.
  auto one = json::parse(R"({
    "domain_box": [1.0, 1.0],
    "seed": 1,
    "distributions": {
      "c": {"type": "constant", "value": 0.5},
      "r": {"type": "constant", "value": 0.49}
    },
    "shapes": [{"type": "circle_input", "pos_x_dist": "c", "pos_y_dist": "c",
                "radius_dist": "r"}],
    "generator": {"type": "periodic", "max_attempts": 4},
    "termination": {"type": "number_of_inclusions", "target": 1}
  })");
  REQUIRE(run_pipeline(one, quiet).shapes.size() == 1);
  one["generator"]["min_gap"] = 0.019;
  REQUIRE(run_pipeline(one, quiet).shapes.size() == 1);
  one["generator"]["min_gap"] = 0.021;
  REQUIRE(run_pipeline(one, quiet).shapes.empty());
  one["generator"]["min_gap"] = 0.0;
  one["generator"]["min_gap_relative"] = 0.03;                // 0.03 * 0.98
  REQUIRE(run_pipeline(one, quiet).shapes.empty());

  // Dense periodic packs over several seeds, every pair of the tiling
  // measured exactly: a gap this large against circles this size makes
  // neighbours through a face the rule, not the exception.
  int through_a_face = 0;
  for (int seed = 1; seed <= 8; ++seed) {
    auto cfg = circles("periodic");
    cfg["seed"] = seed;
    cfg["distributions"]["r"] = {{"type", "uniform_real"}, {"a", 0.03}, {"b", 0.08}};
    cfg["termination"]["target"] = 30;
    cfg["generator"]["min_gap"] = 0.04;
    auto const res = run_pipeline(cfg, quiet);
    auto const b = balls(res.shapes);
    for (std::size_t i = 0; i < b.size(); ++i) {
      if (!b[i].primary) continue;
      for (std::size_t j = i; j < b.size(); ++j) {
        if (!b[j].primary) continue;
        for (int sx = -1; sx <= 1; ++sx)
          for (int sy = -1; sy <= 1; ++sy) {
            if (i == j && sx == 0 && sy == 0) continue;
            const D dist = std::hypot(b[i].c[0] - b[j].c[0] - sx,
                                      b[i].c[1] - b[j].c[1] - sy) -
                           b[i].r - b[j].r;
            REQUIRE(dist >= 0.04 * (1 - 1e-12));
            // A pair this close only through a face, with both circles
            // clear of that face: the case only the image test catches.
            if (dist < 0.08 && (sx != 0 || sy != 0)) ++through_a_face;
          }
      }
    }
  }
  REQUIRE(through_a_face > 10);
}

void test_wall_gap() {
  constexpr D g = 0.03;
  for (auto const& cfg0 : {circles("only_inside"), circles("random"),
                           circles("periodic"), spheres("only_inside"),
                           spheres("random"), spheres("periodic")}) {
    auto cfg = cfg0;
    const std::string gen = cfg["generator"]["type"];
    if (gen == "only_inside")
      cfg["distributions"]["x"] = {{"type", "uniform_real"}, {"a", 0.05}, {"b", 0.95}};
    auto count_bad = [&](pipeline_result const& res, int& crossing) {
      int bad = 0;
      crossing = 0;
      for (auto const& b : balls(res.shapes)) {
        if (!b.primary) continue;
        for (int k = 0; k < res.domain_box.dimension(); ++k) {
          for (const D plane : {0.0, res.domain_box[k]}) {
            const D d = std::abs(b.c[k] - plane);     // centre to face plane
            const bool clear = d >= b.r + g - 1e-12;  // stays g away
            const bool through = d <= b.r - g + 1e-12; // reaches g through
            if (through) ++crossing;
            if (gen == "only_inside" ? !clear : !(clear || through)) ++bad;
          }
        }
        // Corners (2D) and edge lines (3D): the surface of the ball must
        // not run within g of them. In 3D the edge along axis k through
        // the corner (ci, cj) of the other two axes; in 2D k is z and the
        // "edge" is the corner point. Corners of a 3D box likewise.
        const int dim = res.domain_box.dimension();
        for (int k = (dim == 3 ? 0 : 2); k < 3; ++k) {
          const int i = (k + 1) % 3, j = (k + 2) % 3;
          for (const D pi_ : {0.0, res.domain_box[i]})
            for (const D pj : {0.0, res.domain_box[j]}) {
              const D d = std::hypot(b.c[i] - pi_, b.c[j] - pj);
              if (d > b.r - g + 1e-12 && d < b.r + g - 1e-12) ++bad;
            }
        }
        if (dim == 3) {
          for (const D px : {0.0, res.domain_box[0]})
            for (const D py : {0.0, res.domain_box[1]})
              for (const D pz : {0.0, res.domain_box[2]}) {
                const D d = std::sqrt((b.c[0] - px) * (b.c[0] - px) +
                                      (b.c[1] - py) * (b.c[1] - py) +
                                      (b.c[2] - pz) * (b.c[2] - pz));
                if (d > b.r - g + 1e-12 && d < b.r + g - 1e-12) ++bad;
              }
        }
      }
      return bad;
    };
    int crossing = 0, crossing0 = 0;
    auto with = cfg;
    with["generator"]["wall_gap"] = g;
    auto const res = run_pipeline(with);
    REQUIRE(res.shapes.size() >= 40);
    REQUIRE(count_bad(res, crossing) == 0);
    // Shapes still cross faces where the generator allows it...
    REQUIRE(gen == "only_inside" ? crossing == 0 : crossing > 0);
    // ...and without the key the same config does come too close to a face.
    REQUIRE(count_bad(run_pipeline(cfg), crossing0) > 0);
    REQUIRE(res.report.wall_gap == g);
  }
}

// boundary_clear() on hand-placed shapes: faces, corners and edges.
void test_boundary_rule() {
  const domain<D> sq{1.0, 1.0};
  const domain<D> cube{1.0, 1.0, 1.0};
  constexpr D g = 0.05;
  auto ok = [&](auto shape, domain<D> const& box, bool may_cross) {
    shape.make_bounding_box();
    return boundary_clear<D>(shape, box, g, may_cross);
  };
  // Faces.
  REQUIRE(ok(circle<D>{0.5, 0.5, 0.2}, sq, false));
  REQUIRE(!ok(circle<D>{0.24, 0.5, 0.2}, sq, false));    // 0.04 from x = 0
  REQUIRE(ok(circle<D>{0.26, 0.5, 0.2}, sq, false));
  REQUIRE(!ok(circle<D>{0.1, 0.5, 0.2}, sq, false));     // crosses: not allowed
  REQUIRE(ok(circle<D>{0.1, 0.5, 0.2}, sq, true));       // 0.1 through
  REQUIRE(!ok(circle<D>{0.17, 0.5, 0.2}, sq, true));     // 0.03 through
  REQUIRE(!ok(circle<D>{-0.17, 0.5, 0.2}, sq, true));    // cap of 0.03 inside
  REQUIRE(ok(rectangle<D>{0.0, 0.5, 0.2, 0.2}, sq, true));
  REQUIRE(!ok(rectangle<D>{0.07, 0.5, 0.2, 0.2}, sq, true));
  // Corner of the square: the circle crosses both faces by 0.1 or more in
  // all four cases; what differs is where its boundary passes the corner.
  REQUIRE(ok(circle<D>{0.05, 0.05, 0.2}, sq, true));     // corner 0.13 inside
  REQUIRE(!ok(circle<D>{0.13, 0.13, 0.2}, sq, true));    // corner 0.016 inside
  REQUIRE(!ok(circle<D>{0.15, 0.15, 0.2}, sq, true));    // corner 0.012 outside
  REQUIRE(ok(circle<D>{0.05, 0.95, 0.2}, sq, true));     // corner (0, 1)
  // The corner rule probes the square of half-width g, not the disc: a
  // corner 0.059 inside, but with a square corner outside, is refused.
  REQUIRE(!ok(circle<D>{0.1, 0.9, 0.2}, sq, true));
  // An ellipse and a rectangle through the same corner.
  REQUIRE(ok(ellipse<D>{0.0, 0.0, 0.3, 0.2, 0.4}, sq, true));
  REQUIRE(!ok(ellipse<D>{0.2, 0.1, 0.25, 0.12, 0.0}, sq, true));
  REQUIRE(ok(rectangle<D>{0.0, 0.0, 0.3, 0.3}, sq, true));
  REQUIRE(!ok(rectangle<D>{0.13, 0.13, 0.3, 0.3}, sq, true));   // corner 0.02 in
  // 3D: a sphere through the edge x = y = 0 (well inside in z).
  REQUIRE(ok(sphere<D>{0.05, 0.05, 0.5, 0.2}, cube, true));
  REQUIRE(!ok(sphere<D>{0.13, 0.13, 0.5, 0.2}, cube, true));    // edge 0.016 in
  REQUIRE(!ok(sphere<D>{0.15, 0.15, 0.5, 0.2}, cube, true));    // edge 0.012 out
  REQUIRE(ok(sphere<D>{0.1, 0.5, 0.5, 0.2}, cube, true));       // one face only
  // ...and through the corner at the origin.
  REQUIRE(ok(sphere<D>{0.03, 0.03, 0.03, 0.2}, cube, true));
  REQUIRE(!ok(sphere<D>{0.1, 0.1, 0.1, 0.2}, cube, true));      // corner 0.027 in
  REQUIRE(ok(box<D>{0.0, 0.0, 0.5, 0.3, 0.3, 0.3}, cube, true));
  REQUIRE(!ok(box<D>{0.13, 0.13, 0.5, 0.3, 0.3, 0.3}, cube, true));
}

// ---------------------------------------------------------------------------
// 4. Zero gaps change nothing
// ---------------------------------------------------------------------------

std::uint64_t fingerprint(shape_vector const& shapes) {
  std::uint64_t h = 1469598103934665603ull;
  auto add = [&](D v) {
    std::uint64_t bits;
    std::memcpy(&bits, &v, sizeof bits);
    for (int i = 0; i < 8; ++i) {
      h ^= (bits >> (8 * i)) & 0xff;
      h *= 1099511628211ull;
    }
  };
  for (auto const& s : shapes) {
    auto const c = s->get_middle_point();
    add(c[0]);
    add(c[1]);
    add(c[2]);
    add(s->measure());
    add(s->is_primary() ? 1.0 : 0.0);
    auto const* bb = s->bounding_box();
    if (bb) {
      add(bb->bottom_point()[0]);
      add(bb->top_point()[1]);
    }
  }
  return h;
}

void test_zero_gap_is_unchanged() {
  for (char const* gen : {"only_inside", "random", "periodic"}) {
    for (auto cfg : {base_2d(), base_3d()}) {
      cfg["generator"]["type"] = gen;
      if (std::string{gen} == "only_inside")
        cfg["distributions"]["any"] = {{"type", "uniform_real"}, {"a", 0.2}, {"b", 0.8}};
      auto zero = cfg;
      zero["generator"]["min_gap"] = 0.0;
      zero["generator"]["min_gap_relative"] = 0.0;
      zero["generator"]["wall_gap"] = 0.0;
      auto const a = run_pipeline(cfg);
      auto const b = run_pipeline(zero);
      REQUIRE(a.shapes.size() == b.shapes.size() && !a.shapes.empty());
      REQUIRE(fingerprint(a.shapes) == fingerprint(b.shapes));
      REQUIRE(a.report.attempts == b.report.attempts);
    }
  }
}

// ---------------------------------------------------------------------------
// 5. Schema, report, headers
// ---------------------------------------------------------------------------

std::string slurp(std::filesystem::path const& p) {
  std::ifstream in{p};
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void test_schema_and_outputs() {
  for (char const* key : {"min_gap", "min_gap_relative", "wall_gap"}) {
    for (char const* gen : {"only_inside", "random", "periodic"}) {
      auto cfg = circles(gen);
      cfg["generator"][key] = -0.01;
      bool threw = false;
      try {
        (void)run_pipeline(cfg);
      } catch (std::exception const& e) {
        threw = std::string{e.what()}.find(key) != std::string::npos;
      }
      REQUIRE(threw);
    }
  }
  REQUIRE(placement_gap<D>{}.between_shapes() == false);
  REQUIRE(placement_gap<D>(0.1, 0.5, 0.0).between(0.4, 0.1) == 0.1);
  REQUIRE(placement_gap<D>(0.1, 0.5, 0.0).between(0.4, 0.6) == 0.2);
  REQUIRE(placement_gap<D>(0.1, 0.5, 0.0).reach(0.6) == 0.3);
  bool threw = false;
  try {
    placement_gap<D> bad{std::numeric_limits<D>::quiet_NaN(), 0.0, 0.0};
    (void)bad;
  } catch (std::invalid_argument const&) {
    threw = true;
  }
  REQUIRE(threw);

  auto const dir = std::filesystem::temp_directory_path() / "rvegen_min_gap_test";
  std::filesystem::create_directories(dir);
  auto cfg = circles("periodic");
  cfg["termination"]["target"] = 12;
  cfg["postprocess"] = json::array(
      {{{"type", "gmsh_geo"}, {"output_path", (dir / "a.geo").string()},
        {"periodic", true}},
       {{"type", "voxel"}, {"output_path", (dir / "a.vox").string()},
        {"nx", 16}, {"ny", 16}, {"nz", 1}},
       {{"type", "run_summary"}, {"output_path", (dir / "a.json").string()}}});
  auto with = cfg;
  with["generator"]["min_gap"] = 0.02;
  with["generator"]["wall_gap"] = 0.01;
  (void)run_pipeline(with);
  REQUIRE(slurp(dir / "a.geo").find("// placement: min_gap = 0.02, wall_gap = 0.01\n") !=
          std::string::npos);
  REQUIRE(slurp(dir / "a.vox").find("# placement: min_gap = 0.02, wall_gap = 0.01\n") !=
          std::string::npos);
  auto const summary = json::parse(slurp(dir / "a.json"));
  REQUIRE(summary.at("report").at("min_gap") == 0.02);
  REQUIRE(summary.at("report").at("min_gap_relative") == 0.0);
  REQUIRE(summary.at("report").at("wall_gap") == 0.01);

  (void)run_pipeline(cfg);                       // no gaps: no extra line
  REQUIRE(slurp(dir / "a.geo").find("placement:") == std::string::npos);
  REQUIRE(slurp(dir / "a.vox").find("placement:") == std::string::npos);
  REQUIRE(json::parse(slurp(dir / "a.json")).at("report").at("min_gap") == 0.0);
  std::filesystem::remove_all(dir);
}

} // namespace

int main() {
  register_all_collision_pairs<D>();
  test_pairs_exact();
  test_pairs_ellipse();
  test_dispatcher();
  test_generated_packs();
  test_relative_gap();
  test_periodic_images();
  test_wall_gap();
  test_boundary_rule();
  test_zero_gap_is_unchanged();
  test_schema_and_outputs();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "min_gap_test: all passed\n";
  return 0;
}
