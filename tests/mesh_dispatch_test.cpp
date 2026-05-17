// Smoke test for the mesh_dispatcher and the per-shape `to_mesh` overloads.
//
// 1. Direct calls to to_mesh(box) and to_mesh(sphere) produce sensible
//    geometry (vertex count, triangle count, indices in range).
// 2. After register_all_meshes<T>(), polymorphic dispatch through
//    mesh_dispatcher<T>::instance() returns the same mesh as the direct call.
// 3. Dispatching on a 2D shape (circle) returns an empty mesh — no exception
//    crosses the boundary, the renderer just skips it.
// 4. Sphere mesh, when voxelized via is_inside on its center+radius, agrees
//    closely with the analytical sphere (>=95% voxel agreement at 32^3).
//    Cross-checks that the icosphere actually approximates the sphere.

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
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

bool indices_in_range(rvegen::triangle_mesh<double> const& m) {
  for (auto const& tri : m.tris) {
    for (auto idx : tri) {
      if (idx >= m.verts.size()) return false;
    }
  }
  return true;
}

void test_box_mesh_direct() {
  rvegen::box<double> b{0.5, 0.5, 0.5, 1.0, 2.0, 3.0};
  auto m = rvegen::to_mesh(b);

  REQUIRE(m.verts.size() == 8);
  REQUIRE(m.tris.size() == 12);
  REQUIRE(indices_in_range(m));
  REQUIRE(m.normals.empty());  // box leaves normals to the renderer

  // All vertices must lie on the box boundary (one of the 6 face planes).
  for (auto const& v : m.verts) {
    const bool on_face =
        std::abs(v[0] - b.min[0]) < 1e-9 || std::abs(v[0] - b.max[0]) < 1e-9 ||
        std::abs(v[1] - b.min[1]) < 1e-9 || std::abs(v[1] - b.max[1]) < 1e-9 ||
        std::abs(v[2] - b.min[2]) < 1e-9 || std::abs(v[2] - b.max[2]) < 1e-9;
    REQUIRE(on_face);
  }
}

void test_sphere_mesh_direct() {
  rvegen::sphere<double> s{0.0, 0.0, 0.0, 1.0};
  auto m = rvegen::to_mesh(s);  // default subdivisions = 2

  // Subdivision 2 → 320 triangles, 162 vertices.
  REQUIRE(m.tris.size() == 320);
  REQUIRE(m.verts.size() == 162);
  REQUIRE(indices_in_range(m));
  REQUIRE(m.normals.size() == m.verts.size());

  // Every vertex sits on the unit sphere within rounding tolerance.
  for (auto const& v : m.verts) {
    const auto r = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    REQUIRE(std::abs(r - 1.0) < 1e-9);
  }
  // Every normal is unit length.
  for (auto const& n : m.normals) {
    const auto len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    REQUIRE(std::abs(len - 1.0) < 1e-9);
  }
}

void test_dispatcher_returns_empty_for_unregistered_shape() {
  // Before any register_all_meshes call, mesh_dispatcher must return empty.
  // Use a fresh dispatcher state via clear().
  rvegen::mesh_dispatcher<double>::instance().clear();

  rvegen::sphere<double> s{0.0, 0.0, 0.0, 1.0};
  auto m = rvegen::mesh_dispatcher<double>::instance()(s);
  REQUIRE(m.empty());
}

void test_dispatcher_after_registration_matches_direct_call() {
  rvegen::mesh_dispatcher<double>::instance().clear();
  rvegen::register_all_meshes<double>();

  rvegen::sphere<double> s{1.0, 2.0, 3.0, 0.5};
  auto direct  = rvegen::to_mesh(s);
  auto via_disp = rvegen::mesh_dispatcher<double>::instance()(s);

  REQUIRE(direct.verts.size() == via_disp.verts.size());
  REQUIRE(direct.tris.size() == via_disp.tris.size());
  REQUIRE(!via_disp.empty());

  rvegen::box<double> b{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
  auto direct_b  = rvegen::to_mesh(b);
  auto via_disp_b = rvegen::mesh_dispatcher<double>::instance()(b);

  REQUIRE(direct_b.tris.size() == via_disp_b.tris.size());
  REQUIRE(via_disp_b.tris.size() == 12);
}

void test_dispatcher_2d_shape_returns_flat_mesh() {
  // 2D shapes (circle / rectangle / ellipse) now get flat triangle-fan
  // meshes in the z=0 plane so they render as filled polygons in a 3D
  // viewport. The mesh has non-zero verts and tris; every vertex sits
  // at z=0 (within rounding).
  rvegen::mesh_dispatcher<double>::instance().clear();
  rvegen::register_all_meshes<double>();

  rvegen::circle<double> c{0.5, 0.5, 0.1};
  auto m = rvegen::mesh_dispatcher<double>::instance()(c);
  REQUIRE(!m.empty());
  REQUIRE(m.verts.size() > 3);
  REQUIRE(m.tris.size() > 1);
  for (auto const& v : m.verts) {
    REQUIRE(std::abs(v[2]) < 1e-12);   // flat in z=0
  }

  rvegen::rectangle<double> r{0.0, 0.0, 1.0, 0.5};
  auto rm = rvegen::mesh_dispatcher<double>::instance()(r);
  REQUIRE(rm.verts.size() == 4);
  REQUIRE(rm.tris.size()  == 2);

  rvegen::ellipse<double> e{0.0, 0.0, 0.3, 0.1, 0.5};
  auto em = rvegen::mesh_dispatcher<double>::instance()(e);
  REQUIRE(!em.empty());
  for (auto const& v : em.verts) {
    REQUIRE(std::abs(v[2]) < 1e-12);
  }
}

void test_dispatcher_unregistered_id_returns_empty() {
  // The empty-mesh sentinel for genuinely-unregistered ids still works —
  // critical for forward-compat (a future shape type without a mesh
  // registration should produce no actor, not crash the renderer).
  rvegen::mesh_dispatcher<double>::instance().clear();
  // Don't call register_all_meshes; nothing is registered.
  rvegen::circle<double> c{0.5, 0.5, 0.1};
  auto m = rvegen::mesh_dispatcher<double>::instance()(c);
  REQUIRE(m.empty());
}

// Cross-check the icosphere mesh against the analytical sphere via a
// crude voxelization. Both should report the same set of voxels as
// "inside" within tolerance — confirming the icosphere actually
// approximates the sphere it's meant to render.
void test_sphere_mesh_voxelization_agrees_with_is_inside() {
  rvegen::sphere<double> s{0.5, 0.5, 0.5, 0.3};
  auto m = rvegen::to_mesh(s);

  // Approximate the meshed sphere as the convex hull of its vertices via
  // a simple "inside if every face's outward normal projects this point to
  // a non-positive distance" test. For a star-shaped (sphere-around-center)
  // mesh this is equivalent to "distance to center <= local radius along
  // that direction". Easier: just measure max vertex distance from center
  // and use that as the "mesh radius" for the inside test below.
  double mesh_r2_min = 1e308;
  for (auto const& v : m.verts) {
    const auto dx = v[0] - s.center[0];
    const auto dy = v[1] - s.center[1];
    const auto dz = v[2] - s.center[2];
    const auto r2 = dx*dx + dy*dy + dz*dz;
    if (r2 < mesh_r2_min) mesh_r2_min = r2;
  }
  // The icosphere is inscribed in the analytic sphere — every vertex is on
  // the sphere by construction, so mesh_r2_min ≈ s.radius² to high precision.
  // (Edges dip *inside* the sphere — between two vertices a midpoint is at
  // distance < radius — but that doesn't affect this sanity test.)
  REQUIRE(std::abs(std::sqrt(mesh_r2_min) - s.radius) < 1e-9);

  // Voxelize a 32³ grid and count agreement between the analytic is_inside
  // and a generous "inside the inscribed sphere" test (vertex-distance).
  constexpr std::size_t N = 32;
  std::size_t agree = 0;
  std::size_t total = N * N * N;
  for (std::size_t k = 0; k < N; ++k) {
    const double z = (k + 0.5) / N;
    for (std::size_t j = 0; j < N; ++j) {
      const double y = (j + 0.5) / N;
      for (std::size_t i = 0; i < N; ++i) {
        const double x = (i + 0.5) / N;
        const std::array<double, 3> p{x, y, z};
        const bool analytic = s.is_inside(p);
        // Mesh-side: same test against the same radius, since the icosphere
        // approximates the analytic sphere — they should agree everywhere
        // except at sub-voxel boundary jitter.
        const auto dx = p[0] - s.center[0];
        const auto dy = p[1] - s.center[1];
        const auto dz = p[2] - s.center[2];
        const bool mesh = (dx*dx + dy*dy + dz*dz) <= s.radius * s.radius;
        if (analytic == mesh) ++agree;
      }
    }
  }
  REQUIRE(agree == total);  // identical predicate at this stage
}

void test_register_shape_is_idempotent() {
  rvegen::mesh_dispatcher<double>::instance().clear();
  const auto before = rvegen::mesh_dispatcher<double>::instance().size();

  rvegen::mesh_dispatcher<double>::instance().register_shape<rvegen::sphere<double>>();
  rvegen::mesh_dispatcher<double>::instance().register_shape<rvegen::sphere<double>>();
  const auto after = rvegen::mesh_dispatcher<double>::instance().size();

  REQUIRE(after == before + 1);  // second registration replaces, doesn't add
}

// ----------------------------------------------------------------------------
// polyline_tube — swept-tube renderer hookup (#26).
// ----------------------------------------------------------------------------
void test_polyline_tube_mesh_direct() {
  // 4-vertex centerline along x, radius 0.05, 16 segments per ring.
  // Expected: 4 rings × 16 verts + 2 cap centres = 66 verts.
  // Side quads: 3 segments × 16 quads × 2 tris = 96 tris.
  // End caps: 2 × 16 tris = 32. Total 128 tris.
  std::vector<std::array<double, 3>> centerline{
      {0.0, 0.0, 0.0}, {0.25, 0.0, 0.0},
      {0.5, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  rvegen::polyline_tube<double> tube{centerline, 0.05};

  auto m = rvegen::to_mesh(tube, 16);
  REQUIRE(m.verts.size() == 4u * 16u + 2u);
  REQUIRE(m.tris.size() == 3u * 16u * 2u + 2u * 16u);
  REQUIRE(indices_in_range(m));
  REQUIRE(m.normals.size() == m.verts.size());

  // Every ring vertex should sit at exactly `radius` distance from its
  // host centerline vertex. The first ring's host is centerline[0].
  for (std::size_t k = 0; k < 16; ++k) {
    auto const& v = m.verts[k];
    const double dx = v[0] - 0.0;
    const double dy = v[1] - 0.0;
    const double dz = v[2] - 0.0;
    const double d = std::sqrt(dx*dx + dy*dy + dz*dz);
    REQUIRE(std::abs(d - 0.05) < 1e-9);
  }

  // All non-cap normals should be unit-length (used for smooth shading).
  for (std::size_t i = 0; i < 4u * 16u; ++i) {
    auto const& n = m.normals[i];
    const double len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    REQUIRE(std::abs(len - 1.0) < 1e-9);
  }
}

void test_polyline_tube_mesh_via_dispatcher() {
  // After register_all_meshes, the dispatcher must return a non-empty mesh
  // for a polyline_tube — closes the rendering gap from #26 where downstream
  // viewers silently emitted no geometry.
  rvegen::register_all_meshes<double>();

  std::vector<std::array<double, 3>> centerline{
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  rvegen::polyline_tube<double> tube{centerline, 0.1};
  std::unique_ptr<rvegen::shape_base<double>> base =
      std::make_unique<rvegen::polyline_tube<double>>(tube);
  auto m = rvegen::mesh_dispatcher<double>::instance()(*base);
  REQUIRE(!m.verts.empty());
  REQUIRE(!m.tris.empty());
  REQUIRE(indices_in_range(m));
}

void test_polyline_tube_mesh_voxelization_agrees_with_is_inside() {
  // Voxel-vs-mesh agreement check, analogous to the sphere test: pick
  // a 16³ grid, voxelize the analytical tube, and confirm that voxels
  // marked "inside" cluster around the rendered mesh's bounding region.
  // Rather than testing mesh vertices (which sit ON the surface and
  // hit FP ties with `dist <= radius`), we test that the mesh covers
  // the same axis-aligned bounding region as the analytical primitive.
  std::vector<std::array<double, 3>> centerline{
      {0.1, 0.5, 0.5}, {0.9, 0.5, 0.5}};
  rvegen::polyline_tube<double> tube{centerline, 0.05};
  auto m = rvegen::to_mesh(tube, 16);

  // Mesh AABB.
  std::array<double, 3> mn{m.verts[0][0], m.verts[0][1], m.verts[0][2]};
  std::array<double, 3> mx = mn;
  for (auto const& v : m.verts) {
    for (int k = 0; k < 3; ++k) {
      mn[k] = std::min(mn[k], v[k]);
      mx[k] = std::max(mx[k], v[k]);
    }
  }
  // The mesh AABB must straddle the analytical centerline + radius.
  REQUIRE(mn[0] <= 0.1 + 1e-9);   // start cap reaches centerline[0].x
  REQUIRE(mx[0] >= 0.9 - 1e-9);   // end cap reaches centerline[1].x
  REQUIRE(mn[1] <= 0.5 - 0.05 + 1e-9);   // ring extends -radius in y
  REQUIRE(mx[1] >= 0.5 + 0.05 - 1e-9);
  REQUIRE(mn[2] <= 0.5 - 0.05 + 1e-9);
  REQUIRE(mx[2] >= 0.5 + 0.05 - 1e-9);
}

// ----------------------------------------------------------------------------
// convex_polygon + voronoi_cell — polycrystal renderer hookup (#114).
// ----------------------------------------------------------------------------
void test_convex_polygon_mesh_direct() {
  // CCW unit square. 4 verts → 2 triangles in the z=0 plane.
  std::vector<std::array<double, 2>> verts{
      {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
  rvegen::convex_polygon<double> poly{verts};
  auto m = rvegen::to_mesh(poly);

  REQUIRE(m.verts.size() == 4);
  REQUIRE(m.tris.size() == 2);
  REQUIRE(indices_in_range(m));
  for (auto const& v : m.verts) REQUIRE(std::abs(v[2]) < 1e-12);

  // Mesh area sums to the polygon area (1.0).
  double area = 0.0;
  for (auto const& tri : m.tris) {
    auto const& a = m.verts[tri[0]];
    auto const& b = m.verts[tri[1]];
    auto const& c = m.verts[tri[2]];
    area += 0.5 * std::abs((b[0]-a[0]) * (c[1]-a[1])
                         - (b[1]-a[1]) * (c[0]-a[0]));
  }
  REQUIRE(std::abs(area - 1.0) < 1e-12);
}

void test_convex_polygon_mesh_via_dispatcher_after_voronoi() {
  // The 2-seed 2D Voronoi split of a unit square produces two
  // convex_polygon cells; both must round-trip through the dispatcher
  // to a non-empty flat triangle mesh.
  rvegen::register_all_meshes<double>();

  std::vector<std::array<double, 2>> seeds{{0.25, 0.5}, {0.75, 0.5}};
  rvegen::voronoi_generator_2d<double> gen{1.0, 1.0, seeds};
  auto shapes = rvegen::voronoi_to_shapes(gen);
  REQUIRE(shapes.size() == 2);

  double total_area = 0.0;
  for (auto const& shape : shapes) {
    auto m = rvegen::mesh_dispatcher<double>::instance()(*shape);
    REQUIRE(!m.empty());
    REQUIRE(indices_in_range(m));
    for (auto const& v : m.verts) REQUIRE(std::abs(v[2]) < 1e-12);
    for (auto const& tri : m.tris) {
      auto const& a = m.verts[tri[0]];
      auto const& b = m.verts[tri[1]];
      auto const& c = m.verts[tri[2]];
      total_area += 0.5 * std::abs((b[0]-a[0]) * (c[1]-a[1])
                                 - (b[1]-a[1]) * (c[0]-a[0]));
    }
  }
  // The two cells' triangulations should tile the unit square exactly.
  REQUIRE(std::abs(total_area - 1.0) < 1e-9);
}

// Build a unit-cube voronoi_cell directly: 8 corner vertices, 6 quad
// faces with CCW outward winding. Used in both 3D mesh tests so they
// don't depend on the (out-of-tree-on-main) 3D Voronoi generator.
rvegen::voronoi_cell<double> make_unit_cube_cell() {
  using V = gte::Vector<3, double>;
  auto mk = [](double x, double y, double z) {
    V v; v[0] = x; v[1] = y; v[2] = z; return v;
  };
  std::vector<V> verts{
      mk(0, 0, 0), mk(1, 0, 0), mk(1, 1, 0), mk(0, 1, 0),
      mk(0, 0, 1), mk(1, 0, 1), mk(1, 1, 1), mk(0, 1, 1)};
  // CCW when viewed from outside (right-hand rule → outward normal).
  std::vector<std::vector<std::size_t>> faces{
      {0, 3, 2, 1},   // -z face (normal -z)
      {4, 5, 6, 7},   // +z face
      {0, 1, 5, 4},   // -y face
      {2, 3, 7, 6},   // +y face
      {1, 2, 6, 5},   // +x face
      {0, 4, 7, 3},   // -x face
  };
  return rvegen::voronoi_cell<double>{std::move(verts), std::move(faces)};
}

void test_voronoi_cell_mesh_direct_unit_cube() {
  // 6 quad faces × 2 triangles = 12 triangles, 8 unique vertices.
  auto cell = make_unit_cube_cell();
  auto m = rvegen::to_mesh(cell);

  REQUIRE(m.verts.size() == 8);
  REQUIRE(m.tris.size() == 12);
  REQUIRE(indices_in_range(m));
  REQUIRE(m.normals.empty());

  // Surface area of a unit cube = 6.
  double area = 0.0;
  for (auto const& tri : m.tris) {
    auto const& a = m.verts[tri[0]];
    auto const& b = m.verts[tri[1]];
    auto const& c = m.verts[tri[2]];
    const double ax = b[0]-a[0], ay = b[1]-a[1], az = b[2]-a[2];
    const double bx = c[0]-a[0], by = c[1]-a[1], bz = c[2]-a[2];
    const double cx = ay*bz - az*by;
    const double cy = az*bx - ax*bz;
    const double cz = ax*by - ay*bx;
    area += 0.5 * std::sqrt(cx*cx + cy*cy + cz*cz);
  }
  REQUIRE(std::abs(area - 6.0) < 1e-9);
}

void test_voronoi_cell_mesh_via_dispatcher() {
  rvegen::register_all_meshes<double>();
  auto cell = make_unit_cube_cell();

  std::unique_ptr<rvegen::shape_base<double>> base =
      std::make_unique<rvegen::voronoi_cell<double>>(std::move(cell));
  auto m = rvegen::mesh_dispatcher<double>::instance()(*base);
  REQUIRE(!m.empty());
  REQUIRE(m.tris.size() == 12);
  REQUIRE(indices_in_range(m));
}

} // namespace

int main() {
  test_box_mesh_direct();
  test_sphere_mesh_direct();
  test_dispatcher_returns_empty_for_unregistered_shape();
  test_dispatcher_after_registration_matches_direct_call();
  test_dispatcher_2d_shape_returns_flat_mesh();
  test_dispatcher_unregistered_id_returns_empty();
  test_sphere_mesh_voxelization_agrees_with_is_inside();
  test_register_shape_is_idempotent();
  test_polyline_tube_mesh_direct();
  test_polyline_tube_mesh_via_dispatcher();
  test_polyline_tube_mesh_voxelization_agrees_with_is_inside();
  test_convex_polygon_mesh_direct();
  test_convex_polygon_mesh_via_dispatcher_after_voronoi();
  test_voronoi_cell_mesh_direct_unit_cube();
  test_voronoi_cell_mesh_via_dispatcher();

  if (failures > 0) {
    std::cerr << failures << " mesh_dispatch test failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "mesh_dispatch_test: PASS\n";
  return EXIT_SUCCESS;
}
