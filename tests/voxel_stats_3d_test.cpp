// Tests for issue #81: 3D voxel sampling, 3D microstructure statistics and
// the VTK cell-data layout.
//
// 1. A 3D domain sampled with nz = 1 hits the mid-plane z = Lz/2 (it used
//    to sample the z = 0 face), while 2D sampling is unchanged.
// 2. two_point_correlation_3d: S2(r → 0) equals the voxel volume fraction
//    and S2(r large) tends to φ² on a random periodic sphere packing.
// 3. radial_distribution_3d: g(r) ≈ 1 at mid range for a Poisson point
//    set, and periodic copies (centre outside the box) are ignored.
// 4. vtk_legacy_writer v3: CELL_DATA on an (n+1)-point lattice with
//    ORIGIN 0 0 0 and SPACING L/n, values in (k, j, i) order.
// 5. statistics_post_process (built from JSON) picks the 3D estimators
//    for a 3D domain and still accepts a 2D config without "nz".

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rvegen/rvegen.h"
#include "rvegen/postprocess/statistics_post_process.h"
#include "rvegen/postprocess/voxel_grid.h"
#include "rvegen/postprocess/vtk_legacy_writer.h"
#include "rvegen/shapes/circle.h"
#include "rvegen/shapes/sphere.h"
#include "rvegen/statistics/microstructure_stats.h"

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

using shape_vector = std::vector<std::unique_ptr<rvegen::shape_base<double>>>;

double voxel_fraction(std::vector<rvegen::phase_id> const& grid) {
  std::size_t n = 0;
  for (auto v : grid) n += (v != 0);
  return double(n) / double(grid.size());
}

// Hard-sphere random sequential addition in the unit cube with periodic
// minimum-image overlap checks; every sphere that crosses a face gets its
// periodic images so the voxelized field is truly periodic.
shape_vector random_periodic_spheres(std::size_t target, double radius,
                                     std::uint64_t seed) {
  std::mt19937_64 engine{seed};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  std::vector<std::array<double, 3>> centres;
  for (std::size_t attempt = 0; attempt < 200'000 && centres.size() < target;
       ++attempt) {
    const std::array<double, 3> c{u(engine), u(engine), u(engine)};
    bool ok = true;
    for (auto const& o : centres) {
      double d2 = 0.0;
      for (int a = 0; a < 3; ++a) {
        double d = std::abs(c[a] - o[a]);
        if (d > 0.5) d = 1.0 - d;
        d2 += d * d;
      }
      if (d2 < 4.0 * radius * radius) { ok = false; break; }
    }
    if (ok) centres.push_back(c);
  }
  shape_vector shapes;
  for (auto const& c : centres) {
    for (int sx = -1; sx <= 1; ++sx)
      for (int sy = -1; sy <= 1; ++sy)
        for (int sz = -1; sz <= 1; ++sz) {
          const std::array<double, 3> p{c[0] + sx, c[1] + sy, c[2] + sz};
          bool touches = true;
          for (int a = 0; a < 3; ++a) {
            if (p[a] + radius < 0.0 || p[a] - radius > 1.0) touches = false;
          }
          if (touches) {
            shapes.emplace_back(
                std::make_unique<rvegen::sphere<double>>(p[0], p[1], p[2], radius));
          }
        }
  }
  return shapes;
}

void test_nz1_samples_mid_plane() {
  shape_vector shapes;
  // Sphere centred at Lz/2 that does not reach the z = 0 face.
  shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(0.5, 0.5, 1.0, 0.3));
  const std::array<double, 3> box{1.0, 1.0, 2.0};
  const auto grid = rvegen::sample_voxel_grid(shapes, box, 8, 8, 1);
  REQUIRE(grid.size() == 64);
  // Voxel (4, 4) has centre (0.5625, 0.5625, 1.0): inside the sphere.
  REQUIRE(grid[4 * 8 + 4] == 1);
  REQUIRE(voxel_fraction(grid) > 0.0);

  // 2D behaviour is unchanged: z = 0 for a flat domain.
  shape_vector circles;
  circles.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.3));
  const auto grid2d = rvegen::sample_voxel_grid(circles, {1.0, 1.0, 0.0}, 8, 8, 5);
  REQUIRE(grid2d.size() == 64);
  REQUIRE(grid2d[4 * 8 + 4] == 1);
}

void test_s2_3d_limits() {
  const auto shapes = random_periodic_spheres(60, 0.08, 7);
  const std::array<double, 3> box{1.0, 1.0, 1.0};
  constexpr std::size_t n = 32;
  const double phi_v = voxel_fraction(rvegen::sample_voxel_grid(shapes, box, n, n, n));
  REQUIRE(phi_v > 0.05);

  // r → 0: with 256 bins the first bin sits at r ≈ 0.001, far below the
  // voxel size 1/32, so S2 is the voxel volume fraction up to noise.
  const auto s2_fine = rvegen::two_point_correlation_3d(shapes, box, n, n, n,
                                                        256, 20'000);
  REQUIRE(s2_fine.size() == 256);
  REQUIRE(std::abs(s2_fine.front().s2 - phi_v) < 0.01);

  // r large (≥ ~4R): S2 → φ².
  const auto s2 = rvegen::two_point_correlation_3d(shapes, box, n, n, n,
                                                   16, 40'000);
  REQUIRE(s2.size() == 16);
  REQUIRE(std::abs(s2.back().r - 0.5 * 15.5 / 16.0) < 1e-12);
  double tail = 0.0;
  std::size_t count = 0;
  for (auto const& s : s2) {
    if (s.r >= 0.3) { tail += s.s2; ++count; }
  }
  tail /= double(count);
  const double phi2 = phi_v * phi_v;
  std::cout << "  S2_3d: phi_v=" << phi_v << " S2(0)=" << s2_fine.front().s2
            << " tail=" << tail << " phi^2=" << phi2 << '\n';
  REQUIRE(std::abs(tail - phi2) < 0.15 * phi2);
}

void test_rdf_3d_poisson() {
  std::mt19937_64 engine{11};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  shape_vector shapes;
  constexpr std::size_t N = 1500;
  for (std::size_t i = 0; i < N; ++i) {
    const double x = u(engine), y = u(engine), z = u(engine);
    shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(x, y, z, 1e-3));
    // Periodic copies (centre outside the box) must be ignored; counted,
    // they would put zero-distance pairs into the first bin.
    if (i % 10 == 0) {
      shapes.emplace_back(
          std::make_unique<rvegen::sphere<double>>(x + 1.0, y, z, 1e-3));
    }
  }
  const auto g = rvegen::radial_distribution_3d(shapes, {1.0, 1.0, 1.0}, 10);
  REQUIRE(g.size() == 10);
  for (auto const& s : g) {
    if (s.r > 0.1 && s.r < 0.4) {
      REQUIRE(std::abs(s.g - 1.0) < 0.05);
    }
  }
  REQUIRE(std::abs(g.front().g - 1.0) < 0.15);
}

std::vector<int> vtk_body(std::string const& text) {
  std::vector<int> values;
  const std::string marker = "LOOKUP_TABLE default\n";
  const auto pos = text.find(marker);
  if (pos == std::string::npos) return values;
  std::istringstream in{text.substr(pos + marker.size())};
  int v = 0;
  while (in >> v) values.push_back(v);
  return values;
}

void test_vtk_cell_data_layout() {
  // 3D: one small sphere at the centre of cell (i=1, j=0, k=1).
  shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(0.75, 0.25, 0.5, 0.1));
  rvegen::vtk_legacy_writer<double> w{4, 2, 3};
  std::stringstream out;
  w.write(out, shapes, {2.0, 1.0, 1.0});
  const auto text = out.str();
  REQUIRE(text.find("[rvegen-vtk-format: v3]") != std::string::npos);
  REQUIRE(text.find("DATASET STRUCTURED_POINTS\n") != std::string::npos);
  REQUIRE(text.find("DIMENSIONS 5 3 4\n") != std::string::npos);
  REQUIRE(text.find("ORIGIN 0 0 0\n") != std::string::npos);
  REQUIRE(text.find("SPACING 0.5 0.5 0.333333\n") != std::string::npos);
  REQUIRE(text.find("CELL_DATA 24\n") != std::string::npos);
  REQUIRE(text.find("POINT_DATA") == std::string::npos);
  const auto body = vtk_body(text);
  REQUIRE(body.size() == 24);
  for (std::size_t idx = 0; idx < body.size(); ++idx) {
    REQUIRE(body[idx] == (idx == (1 * 2 + 0) * 4 + 1 ? 1 : 0));
  }

  // 3D with nz = 1: one slab of cells spanning the full thickness.
  rvegen::vtk_legacy_writer<double> slab{4, 2, 1};
  std::stringstream out_slab;
  slab.write(out_slab, shapes, {2.0, 1.0, 1.0});
  const auto slab_text = out_slab.str();
  REQUIRE(slab_text.find("DIMENSIONS 5 3 2\n") != std::string::npos);
  REQUIRE(slab_text.find("SPACING 0.5 0.5 1\n") != std::string::npos);
  REQUIRE(slab_text.find("CELL_DATA 8\n") != std::string::npos);
  const auto slab_body = vtk_body(slab_text);
  REQUIRE(slab_body.size() == 8 && slab_body[1] == 1);

  // 2D: flat lattice with a single z point.
  shape_vector circles;
  circles.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.2));
  rvegen::vtk_legacy_writer<double> flat{8, 8, 4};
  std::stringstream out2d;
  flat.write(out2d, circles, {1.0, 1.0, 0.0});
  const auto text2d = out2d.str();
  REQUIRE(text2d.find("DIMENSIONS 9 9 1\n") != std::string::npos);
  REQUIRE(text2d.find("SPACING 0.125 0.125 1\n") != std::string::npos);
  REQUIRE(text2d.find("CELL_DATA 64\n") != std::string::npos);
  REQUIRE(vtk_body(text2d).size() == 64);
}

// Build "statistics" from JSON; 3D domain dispatches to the 3D estimators
// (explicit nz) and a 2D config without nz stays valid.
std::string run_statistics(std::string const& json, shape_vector const& shapes,
                           std::array<double, 3> const& box,
                           std::string const& path) {
  rvegen::register_all_post_processes<>();
  auto& registry = rvegen::post_process_registry_t<>::instance();
  auto pp = rvegen::build_from_json(registry, nlohmann::json::parse(json));
  REQUIRE(pp != nullptr);
  if (!pp) return {};
  pp->run(shapes, box);
  std::ifstream in{path};
  std::stringstream buf;
  buf << in.rdbuf();
  std::remove(path.c_str());
  return buf.str();
}

void test_statistics_post_process_3d() {
  const auto shapes = random_periodic_spheres(30, 0.08, 3);
  const std::string path3 = "voxel_stats_3d_test_stats3d.csv";
  const auto text = run_statistics(R"({
    "type": "statistics", "output_path": ")" + path3 + R"(",
    "nx": 16, "ny": 16, "nz": 16, "nbins": 8 })",
                                   shapes, {1.0, 1.0, 1.0}, path3);
  REQUIRE(text.find("# dimension = 3D\n") != std::string::npos);
  const auto s2 = rvegen::two_point_correlation_3d(shapes, {1.0, 1.0, 1.0},
                                                   16, 16, 16, 8);
  const auto g = rvegen::radial_distribution_3d(shapes, {1.0, 1.0, 1.0}, 8);
  std::ostringstream first_row;
  first_row << '\n' << s2[0].r << ',' << s2[0].s2 << ',' << g[0].g << '\n';
  REQUIRE(text.find(first_row.str()) != std::string::npos);

  shape_vector circles;
  circles.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.2));
  const std::string path2 = "voxel_stats_3d_test_stats2d.csv";
  const auto text2d = run_statistics(R"({
    "type": "statistics", "output_path": ")" + path2 + R"(",
    "nx": 16, "ny": 16, "nbins": 8 })",
                                     circles, {1.0, 1.0, 0.0}, path2);
  REQUIRE(text2d.find("# dimension = 2D\n") != std::string::npos);
}

} // namespace

int main() {
  test_nz1_samples_mid_plane();
  test_s2_3d_limits();
  test_rdf_3d_poisson();
  test_vtk_cell_data_layout();
  test_statistics_post_process_3d();

  if (failures > 0) {
    std::cerr << failures << " voxel/stats 3D test failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "voxel_stats_3d_test: PASS\n";
  return EXIT_SUCCESS;
}
