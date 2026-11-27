// Precise collision pairs added for #83: rectangle x ellipse,
// polyline_tube x {sphere, box, tube}, mesh_inclusion x {sphere, box, tube,
// mesh}. Each pair is tested touching, separated, overlapping and (where
// the geometry allows) rotated / diagonal. The "diagonal" cases are the
// ones where the AABBs overlap but the shapes do not: they would have been
// false positives under the old AABB fallback; the test asserts the AABBs
// DO overlap so the case stays meaningful.

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

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
using D = double;
using P3 = std::array<D, 3>;

constexpr D pi = 3.14159265358979323846;

// Typed check in both argument orders.
template <typename A, typename B>
bool hit(A const& a, B const& b) {
  const bool ab = collision_details(a, b);
  const bool ba = collision_details(b, a);
  REQUIRE(ab == ba);
  return ab;
}

// True if the shapes' AABBs overlap (the old fallback's answer).
bool aabb_hit(shape_base<D> const& a, shape_base<D> const& b) {
  auto ca = a.clone();
  auto cb = b.clone();
  ca->make_bounding_box();
  cb->make_bounding_box();
  return aabb_overlap(*ca->bounding_box(), *cb->bounding_box());
}

// ---- rectangle x ellipse --------------------------------------------------
void test_rectangle_ellipse() {
  rectangle<D> r{0.5, 0.5, 1.0, 1.0};   // [0,1]^2
  REQUIRE(hit(r, ellipse<D>{2.0, 0.5, 1.0, 0.3, 0.0}));      // touching
  REQUIRE(!hit(r, ellipse<D>{2.1, 0.5, 1.0, 0.3, 0.0}));     // separated
  REQUIRE(hit(r, ellipse<D>{1.5, 0.5, 1.0, 0.3, 0.0}));      // overlapping
  REQUIRE(hit(r, ellipse<D>{0.5, 0.5, 0.1, 0.05, 0.3}));     // ellipse inside
  REQUIRE(hit(r, ellipse<D>{0.5, 0.5, 5.0, 4.0, 0.3}));      // rect inside
  // Ellipse centre outside, but the ellipse swallows the rectangle.
  REQUIRE(hit(r, ellipse<D>{3.0, 0.5, 4.0, 4.0, 0.0}));

  // Rotated: long thin ellipse at (1.6, 1.6) pointing at the corner (1,1).
  ellipse<D> toward{1.6, 1.6, 1.0, 0.1, pi / 4};
  REQUIRE(hit(r, toward));
  // Same ellipse turned 90 degrees: tangential to the corner, AABBs overlap
  // but the shapes are 0.75 apart.
  ellipse<D> tangent{1.6, 1.6, 1.0, 0.1, -pi / 4};
  REQUIRE(aabb_hit(r, tangent));
  REQUIRE(!hit(r, tangent));
  // ...and with a longer minor axis it reaches the corner.
  REQUIRE(hit(r, ellipse<D>{1.6, 1.6, 1.0, 0.9, -pi / 4}));
}

// ---- tube x sphere / box ----------------------------------------------------
polyline_tube<D> x_tube(D radius) {
  return polyline_tube<D>{std::vector<P3>{{0, 0, 0}, {10, 0, 0}}, radius};
}
polyline_tube<D> diag_tube(D radius) {
  return polyline_tube<D>{std::vector<P3>{{0, 0, 0}, {10, 10, 0}}, radius};
}

void test_tube_sphere() {
  auto t = x_tube(0.5);
  REQUIRE(hit(t, sphere<D>{5, 1.5, 0, 1.0}));    // touching
  REQUIRE(!hit(t, sphere<D>{5, 1.6, 0, 1.0}));   // separated
  REQUIRE(hit(t, sphere<D>{5, 1.0, 0, 1.0}));    // overlapping
  REQUIRE(hit(t, sphere<D>{-0.9, 0, 0, 0.5}));   // beyond the end cap, close
  REQUIRE(!hit(t, sphere<D>{-1.1, 0, 0, 0.5}));  // beyond the end cap, far

  auto d = diag_tube(0.5);
  sphere<D> corner{10, 0, 0, 1.0};
  REQUIRE(aabb_hit(d, corner));
  REQUIRE(!hit(d, corner));                      // 7.07 from the line
  REQUIRE(hit(d, sphere<D>{6, 4, 0, 1.0}));      // 1.41 <= 1.5

  // Bent polyline: second leg runs along x = 5.
  polyline_tube<D> bent{std::vector<P3>{{0, 0, 0}, {5, 0, 0}, {5, 5, 0}}, 0.5};
  REQUIRE(hit(bent, sphere<D>{4, 4, 0, 0.6}));
  REQUIRE(!hit(bent, sphere<D>{4, 4, 0, 0.4}));

  // Degenerate tubes: one point behaves as a sphere, no points never hit.
  polyline_tube<D> dot{std::vector<P3>{{0, 0, 0}}, 0.5};
  REQUIRE(hit(dot, sphere<D>{1.0, 0, 0, 0.5}));
  REQUIRE(!hit(dot, sphere<D>{1.2, 0, 0, 0.5}));
  polyline_tube<D> none{std::vector<P3>{}, 0.5};
  REQUIRE(!hit(none, sphere<D>{0, 0, 0, 5.0}));
}

void test_tube_box() {
  auto t = x_tube(0.5);
  REQUIRE(!hit(t, box<D>{5, 2, 0, 2, 2, 2}));    // y in [1,3], gap 0.5
  REQUIRE(hit(x_tube(1.0), box<D>{5, 2, 0, 2, 2, 2}));   // touching
  REQUIRE(hit(x_tube(1.5), box<D>{5, 2, 0, 2, 2, 2}));   // overlapping
  REQUIRE(hit(t, box<D>{5, 0, 0, 1, 1, 1}));     // tube through the box

  auto d = diag_tube(0.5);
  box<D> off_corner{5, 1, 0, 2, 2, 2};           // [4,6]x[0,2]: corner (4,2)
  REQUIRE(aabb_hit(d, off_corner));
  REQUIRE(!hit(d, off_corner));                  // 1.41 from the line
  REQUIRE(hit(diag_tube(1.5), off_corner));
  REQUIRE(hit(d, box<D>{4, 3, 0, 2, 2, 2}));     // line passes through
}

void test_tube_tube() {
  // Skew crossing: x-axis tube at z = 0, y-axis tube at z = 1.
  auto a = x_tube(0.5);
  auto b = [](D z, D r) {
    return polyline_tube<D>{std::vector<P3>{{5, -5, z}, {5, 5, z}}, r};
  };
  REQUIRE(hit(a, b(1.0, 0.5)));      // touching: distance 1 == 0.5 + 0.5
  REQUIRE(!hit(a, b(1.1, 0.5)));     // separated
  REQUIRE(hit(a, b(0.5, 0.5)));      // overlapping
  REQUIRE(hit(a, b(0.0, 0.1)));      // intersecting axes

  // Parallel diagonals 1.41 apart (AABBs overlap).
  polyline_tube<D> d1 = diag_tube(0.5);
  polyline_tube<D> d2{std::vector<P3>{{0, 2, 0}, {8, 10, 0}}, 0.5};
  REQUIRE(aabb_hit(d1, d2));
  REQUIRE(!hit(d1, d2));
  polyline_tube<D> d3{std::vector<P3>{{0, 2, 0}, {8, 10, 0}}, 0.8};
  polyline_tube<D> d4 = diag_tube(0.8);
  REQUIRE(hit(d3, d4));

  // Anti-diagonals crossing in the middle at different heights.
  polyline_tube<D> up{std::vector<P3>{{0, 10, 3}, {10, 0, 3}}, 0.5};
  REQUIRE(!hit(d1, up));             // 3 apart in z
  polyline_tube<D> level{std::vector<P3>{{0, 10, 0.9}, {10, 0, 0.9}}, 0.5};
  REQUIRE(hit(d1, level));

  // Multi-segment: zig-zag next to a straight tube, only one leg close.
  polyline_tube<D> zig{
      std::vector<P3>{{0, 3, 0}, {4, 3, 0}, {4, 0.7, 0}, {8, 0.7, 0}}, 0.3};
  REQUIRE(hit(a, zig));              // last legs 0.7 above <= 0.5 + 0.3
  polyline_tube<D> zag{
      std::vector<P3>{{0, 3, 0}, {4, 3, 0}, {4, 0.9, 0}, {8, 0.9, 0}}, 0.3};
  REQUIRE(!hit(a, zag));
}

// ---- mesh_inclusion ---------------------------------------------------------
// Axis-aligned or z-rotated cube of half-size h around the origin, as a
// 12-triangle closed mesh.
mesh_inclusion<D> cube_mesh(D h, D centre_x, D centre_y, D centre_z,
                            D rot_z = 0.0) {
  const D c = std::cos(rot_z), s = std::sin(rot_z);
  std::array<std::array<D, 3>, 8> v;
  for (int i = 0; i < 8; ++i) {
    const D x = (i & 1) ? h : -h;
    const D y = (i & 2) ? h : -h;
    const D z = (i & 4) ? h : -h;
    v[i] = {c * x - s * y, s * x + c * y, z};
  }
  constexpr int faces[12][3] = {
      {0, 1, 3}, {0, 3, 2}, {4, 5, 7}, {4, 7, 6}, {0, 1, 5}, {0, 5, 4},
      {2, 3, 7}, {2, 7, 6}, {0, 2, 6}, {0, 6, 4}, {1, 3, 7}, {1, 7, 5}};
  typename mesh_inclusion<D>::triangle_list tris;
  for (auto const& f : faces) {
    gte::Triangle3<D> t;
    for (int k = 0; k < 3; ++k) {
      t.v[k][0] = v[f[k]][0];
      t.v[k][1] = v[f[k]][1];
      t.v[k][2] = v[f[k]][2];
    }
    tris.push_back(t);
  }
  mesh_inclusion<D> m{std::move(tris)};
  m.set_middle_point({centre_x, centre_y, centre_z});
  return m;
}

// Closed box mesh with half extents (hx, hy, hz) around the origin.
mesh_inclusion<D> bar_mesh(D hx, D hy, D hz) {
  std::array<std::array<D, 3>, 8> v;
  for (int i = 0; i < 8; ++i) {
    v[i] = {(i & 1) ? hx : -hx, (i & 2) ? hy : -hy, (i & 4) ? hz : -hz};
  }
  constexpr int faces[12][3] = {
      {0, 1, 3}, {0, 3, 2}, {4, 5, 7}, {4, 7, 6}, {0, 1, 5}, {0, 5, 4},
      {2, 3, 7}, {2, 7, 6}, {0, 2, 6}, {0, 6, 4}, {1, 3, 7}, {1, 7, 5}};
  typename mesh_inclusion<D>::triangle_list tris;
  for (auto const& f : faces) {
    gte::Triangle3<D> t;
    for (int k = 0; k < 3; ++k) {
      for (int c = 0; c < 3; ++c) t.v[k][c] = v[f[k]][c];
    }
    tris.push_back(t);
  }
  return mesh_inclusion<D>{std::move(tris)};
}

// The pair tests must see the placed mesh (rotation, scale, translation),
// not the base triangles shifted by the offset. Every case below gives the
// opposite answer when the rotation or the scale is ignored.
void test_mesh_placement_transform() {
  // Bar along x, turned by 90 degrees about z: now along y.
  auto bar = bar_mesh(2.0, 0.25, 0.25);
  bar.set_rotation_axis_angle({0, 0, 1}, pi / 2);
  REQUIRE(hit(bar, sphere<D>{0.0, 2.05, 0.0, 0.1}));    // touches the y end cap
  REQUIRE(!hit(bar, sphere<D>{2.05, 0.0, 0.0, 0.1}));   // old x end cap is gone
  REQUIRE(hit(bar, box<D>{0.0, 2.1, 0.0, 0.4, 0.4, 0.4}));
  REQUIRE(!hit(bar, box<D>{2.1, 0.0, 0.0, 0.4, 0.4, 0.4}));
  polyline_tube<D> across{std::vector<P3>{{-1, 1.5, 0.3}, {1, 1.5, 0.3}}, 0.1};
  polyline_tube<D> beside{std::vector<P3>{{1.5, -1, 0.3}, {1.5, 1, 0.3}}, 0.1};
  REQUIRE(hit(bar, across));                             // grazes the top face
  REQUIRE(!hit(bar, beside));
  REQUIRE(hit(bar, cube_mesh(0.1, 0.0, 2.05, 0.0)));    // straddles the y end cap
  REQUIRE(!hit(bar, cube_mesh(0.1, 2.05, 0.0, 0.0)));

  // Rotation and translation together.
  bar.set_middle_point({10, 0, 0});
  REQUIRE(hit(bar, sphere<D>{10.0, 2.05, 0.0, 0.1}));
  REQUIRE(!hit(bar, sphere<D>{12.05, 0.0, 0.0, 0.1}));

  // Unit cube scaled by 3: the surface is at |x| = 3.
  auto big = bar_mesh(1.0, 1.0, 1.0);
  big.set_scale(3.0);
  REQUIRE(hit(big, sphere<D>{3.05, 0.0, 0.0, 0.1}));
  REQUIRE(!hit(big, sphere<D>{3.2, 0.0, 0.0, 0.1}));
  REQUIRE(hit(big, cube_mesh(0.1, 3.05, 0.0, 0.0)));
  // Scaled down: nothing at the old surface any more.
  auto small = bar_mesh(1.0, 1.0, 1.0);
  small.set_scale(0.25);
  REQUIRE(!hit(small, sphere<D>{1.05, 0.0, 0.0, 0.1}));
  REQUIRE(hit(small, sphere<D>{0.3, 0.0, 0.0, 0.1}));
}

void test_mesh_sphere() {
  auto m = cube_mesh(1.0, 1, 1, 1);     // [0,2]^3
  REQUIRE(hit(m, sphere<D>{2.5, 1, 1, 0.5}));     // touching a face
  REQUIRE(!hit(m, sphere<D>{2.6, 1, 1, 0.5}));    // separated
  REQUIRE(hit(m, sphere<D>{2.3, 1, 1, 0.5}));     // overlapping
  REQUIRE(hit(m, sphere<D>{1, 1, 1, 0.1}));       // fully inside
  REQUIRE(hit(m, sphere<D>{1, 1, 1, 10.0}));      // swallows the mesh
  // Near an edge: distance to the edge (2,2,z) is 0.707.
  REQUIRE(!hit(m, sphere<D>{2.5, 2.5, 1, 0.7}));
  REQUIRE(hit(m, sphere<D>{2.5, 2.5, 1, 0.75}));
}

void test_mesh_box() {
  auto m = cube_mesh(1.0, 1, 1, 1);
  REQUIRE(hit(m, box<D>{2.5, 1, 1, 1, 1, 1}));    // touching: x in [2,3]
  REQUIRE(!hit(m, box<D>{2.6, 1, 1, 1, 1, 1}));   // separated
  REQUIRE(hit(m, box<D>{2.2, 1, 1, 1, 1, 1}));    // overlapping
  REQUIRE(hit(m, box<D>{1, 1, 1, 0.4, 0.4, 0.4}));// box inside the mesh
  REQUIRE(hit(m, box<D>{1, 1, 1, 9, 9, 9}));      // mesh inside the box
  // Rotated cube vs box in the gap its AABB cannot see.
  auto r = cube_mesh(1.0, 0, 0, 0, pi / 4);       // diamond, |x|+|y| <= 1.414
  box<D> in_aabb_gap{1.5, 1.4, 0, 1.0, 1.0, 1.0}; // [1,2]x[0.9,1.9]
  REQUIRE(aabb_hit(r, in_aabb_gap));
  REQUIRE(!hit(r, in_aabb_gap));
}

void test_mesh_tube() {
  auto m = cube_mesh(1.0, 1, 1, 1);
  auto along_x = [](D z, D r) {
    return polyline_tube<D>{std::vector<P3>{{-5, 1, z}, {5, 1, z}}, r};
  };
  REQUIRE(hit(m, along_x(-0.5, 0.5)));            // touching the bottom face
  REQUIRE(!hit(m, along_x(-0.6, 0.5)));           // separated
  REQUIRE(hit(m, along_x(1.0, 0.1)));             // through the mesh
  // Tube entirely inside the mesh.
  polyline_tube<D> inner{std::vector<P3>{{0.8, 1, 1}, {1.2, 1, 1}}, 0.1};
  REQUIRE(hit(m, inner));
  // Diagonal tube passing the cube corner (2,2): 1.41 away.
  polyline_tube<D> diag{std::vector<P3>{{1, 5, 1}, {5, 1, 1}}, 0.5};
  REQUIRE(aabb_hit(m, diag));
  REQUIRE(!hit(m, diag));
  polyline_tube<D> fat{std::vector<P3>{{1, 5, 1}, {5, 1, 1}}, 1.5};
  REQUIRE(hit(m, fat));
}

void test_mesh_mesh() {
  auto a = cube_mesh(1.0, 1, 1, 1);               // [0,2]^3
  REQUIRE(hit(a, cube_mesh(1.0, 3, 1, 1)));       // face touching at x = 2
  REQUIRE(!hit(a, cube_mesh(1.0, 3.2, 1, 1)));    // separated
  REQUIRE(hit(a, cube_mesh(1.0, 2.5, 1, 1)));     // overlapping
  REQUIRE(hit(a, cube_mesh(0.2, 1, 1, 1)));       // nested, no surface contact
  REQUIRE(hit(a, cube_mesh(5.0, 1, 1, 1)));       // enclosing
  // Rotated cube vs axis-aligned cube across an AABB-only overlap.
  auto r = cube_mesh(1.0, 0, 0, 0, pi / 4);
  auto n = cube_mesh(0.5, 1.5, 1.4, 0);           // [1,2]x[0.9,1.9]
  REQUIRE(aabb_hit(r, n));
  REQUIRE(!hit(r, n));
  REQUIRE(hit(r, cube_mesh(0.5, 1.0, 0.8, 0)));   // pokes into the diamond
}

// ---- dispatcher round-trip ----------------------------------------------------
void test_dispatcher() {
  auto& d = collision_dispatcher<D>::instance();
  d.clear();
  register_all_collision_pairs<D>();

  polyline_tube<D> d1 = diag_tube(0.5);
  polyline_tube<D> d2{std::vector<P3>{{0, 2, 0}, {8, 10, 0}}, 0.5};
  d1.make_bounding_box();
  d2.make_bounding_box();
  REQUIRE(!d(d1, d2));                       // precise, not the AABB fallback

  rectangle<D> r{0.5, 0.5, 1.0, 1.0};
  ellipse<D> e{1.6, 1.6, 1.0, 0.1, -pi / 4};
  r.make_bounding_box();
  e.make_bounding_box();
  REQUIRE(!d(r, e));
  REQUIRE(!d(e, r));

  auto m = cube_mesh(1.0, 1, 1, 1);
  sphere<D> s{3.0, 1, 1, 0.5};
  m.make_bounding_box();
  s.make_bounding_box();
  REQUIRE(!d(m, s));
  sphere<D> s2{2.4, 1, 1, 0.5};
  s2.make_bounding_box();
  REQUIRE(d(s2, m));
}

} // namespace

int main() {
  test_rectangle_ellipse();
  test_tube_sphere();
  test_tube_box();
  test_tube_tube();
  test_mesh_sphere();
  test_mesh_box();
  test_mesh_tube();
  test_mesh_mesh();
  test_mesh_placement_transform();
  test_dispatcher();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "collision_pairs_test OK\n";
  return 0;
}
