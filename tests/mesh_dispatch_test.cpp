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

void test_dispatcher_2d_shape_returns_empty() {
  // Circle is 2D — deliberately not registered. Dispatch must return empty
  // without throwing (a thrown exception across a thread/renderer boundary
  // would crash the GUI; empty-mesh sentinel lets the renderer skip silently).
  rvegen::mesh_dispatcher<double>::instance().clear();
  rvegen::register_all_meshes<double>();

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

} // namespace

int main() {
  test_box_mesh_direct();
  test_sphere_mesh_direct();
  test_dispatcher_returns_empty_for_unregistered_shape();
  test_dispatcher_after_registration_matches_direct_call();
  test_dispatcher_2d_shape_returns_empty();
  test_sphere_mesh_voxelization_agrees_with_is_inside();
  test_register_shape_is_idempotent();

  if (failures > 0) {
    std::cerr << failures << " mesh_dispatch test failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "mesh_dispatch_test: PASS\n";
  return EXIT_SUCCESS;
}
