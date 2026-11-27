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
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rvegen/rvegen.h"
#include "rvegen/postprocess/statistics_post_process.h"
#include "rvegen/postprocess/voxel_grid.h"
#include "rvegen/postprocess/voxel_writer.h"
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

// Same check in 2D, which shares the N(N−1)/V pair normalisation.
void test_rdf_2d_poisson() {
  std::mt19937_64 engine{13};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  shape_vector shapes;
  for (std::size_t i = 0; i < 1500; ++i) {
    shapes.emplace_back(
        std::make_unique<rvegen::circle<double>>(u(engine), u(engine), 1e-3));
  }
  const auto g = rvegen::radial_distribution_2d(shapes, {1.0, 1.0, 0.0}, 10);
  REQUIRE(g.size() == 10);
  double mid = 0.0;
  std::size_t count = 0;
  for (auto const& s : g) {
    if (s.r > 0.1 && s.r < 0.4) {
      REQUIRE(std::abs(s.g - 1.0) < 0.05);
      mid += s.g;
      ++count;
    }
  }
  REQUIRE(std::abs(mid / double(count) - 1.0) < 0.02);
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
  REQUIRE(text.find("SPACING 0.5 0.5 0.33333333333333331\n") != std::string::npos);
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

// SPACING round-trips at full precision: Lx = 1, nx = 3 must read back as
// exactly 1.0 / 3.0, not the 6-digit 0.333333.
void test_vtk_spacing_round_trips() {
  shape_vector shapes;
  rvegen::vtk_legacy_writer<double> w{3, 3, 3};
  std::stringstream out;
  w.write(out, shapes, {1.0, 1.0, 1.0});
  const auto text = out.str();
  const std::string key = "SPACING ";
  const auto pos = text.find(key);
  REQUIRE(pos != std::string::npos);
  if (pos == std::string::npos) return;
  std::istringstream line{text.substr(pos + key.size())};
  double sx = 0, sy = 0, sz = 0;
  line >> sx >> sy >> sz;
  REQUIRE(sx == 1.0 / 3.0);
  REQUIRE(sy == 1.0 / 3.0);
  REQUIRE(sz == 1.0 / 3.0);
  REQUIRE(3.0 * sx == 1.0);
  // The caller's stream precision is restored after the header.
  REQUIRE(out.precision() == std::stringstream{}.precision());
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

  // 3D domain without "nz" must throw with a clear message, not pick a
  // potentially expensive default.
  rvegen::register_all_post_processes<>();
  auto& registry = rvegen::post_process_registry_t<>::instance();
  auto pp = rvegen::build_from_json(registry, nlohmann::json::parse(R"({
    "type": "statistics", "output_path": "voxel_stats_3d_test_nonz.csv",
    "nx": 8, "ny": 8, "nbins": 4 })"));
  REQUIRE(pp != nullptr);
  bool threw = false;
  if (pp) {
    try {
      pp->run(shapes, {1.0, 1.0, 1.0});
    } catch (std::runtime_error const& e) {
      threw = std::string{e.what()}.find("\"nz\"") != std::string::npos;
    }
  }
  REQUIRE(threw);
  std::remove("voxel_stats_3d_test_nonz.csv");
}

// The AABB-pruned scan must reproduce the brute-force first-match result,
// including overlaps (lowest index wins) and shapes sticking out of the box.
void test_voxel_matches_brute_force() {
  std::mt19937 gen{7};
  std::uniform_real_distribution<double> u(-0.2, 1.2), ur(0.02, 0.25);
  shape_vector shapes;
  for (int i = 0; i < 40; ++i) {
    shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(
        u(gen), u(gen), u(gen), ur(gen)));
    // Odd shapes keep a null box (full-scan fallback), the rest are pruned.
    if (i % 5 != 0) shapes.back()->make_bounding_box();
  }
  // A stale box (shape moved afterwards) must not hide voxels.
  shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(0.2, 0.2, 0.2, 0.1));
  shapes.back()->make_bounding_box();
  shapes.back()->set_middle_point({0.8, 0.8, 0.8});
  const std::array<double, 3> box{1.0, 1.0, 1.0};
  const std::size_t n = 24;
  auto grid = rvegen::sample_voxel_grid(shapes, box, n, n, n);
  std::size_t mismatches = 0;
  for (std::size_t k = 0; k < n; ++k)
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = 0; i < n; ++i) {
        const std::array<double, 3> p{(i + 0.5) / n, (j + 0.5) / n,
                                      (k + 0.5) / n};
        rvegen::phase_id expect = 0;
        for (std::size_t s = 0; s < shapes.size(); ++s) {
          if (shapes[s]->is_inside(p)) { expect = rvegen::phase_id(s + 1); break; }
        }
        if (grid[(k * n + j) * n + i] != expect) ++mismatches;
      }
  REQUIRE(mismatches == 0);
}

// nbins == 0 used to index hist[nbins - 1] out of bounds.
void test_rdf_zero_bins() {
  shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(0.3, 0.3, 0.3, 0.1));
  shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(0.6, 0.6, 0.6, 0.1));
  REQUIRE(rvegen::radial_distribution_3d(shapes, {1.0, 1.0, 1.0}, 0).empty());
  shape_vector c;
  c.emplace_back(std::make_unique<rvegen::circle<double>>(0.3, 0.3, 0.1));
  REQUIRE(rvegen::radial_distribution_2d(c, {1.0, 1.0, 0.0}, 0).empty());
}

// Fewer than two inclusions: g(r) is undefined but S2 is not, so the CSV
// still has one row per S2 bin (g column empty) plus an explanatory header
// line. With two inclusions the layout is the plain three-column one.
void test_statistics_csv_with_few_inclusions() {
  auto data_rows = [](std::string const& text) {
    std::vector<std::string> rows;
    std::istringstream in{text};
    for (std::string line; std::getline(in, line);) {
      if (!line.empty() && line[0] != '#') rows.push_back(line);
    }
    return rows;
  };
  for (bool is3d : {false, true}) {
    for (int count : {0, 1, 2}) {
      shape_vector shapes;
      for (int c = 0; c < count; ++c) {
        const double x = 0.3 + 0.4 * c;
        if (is3d)
          shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(x, 0.5, 0.5, 0.2));
        else
          shapes.emplace_back(std::make_unique<rvegen::circle<double>>(x, 0.5, 0.2));
      }
      const std::string path = "voxel_stats_3d_test_few.csv";
      const std::string cfg = std::string{R"({"type":"statistics","output_path":")"} +
          path + R"(","nx":16,"ny":16,"nz":16,"nbins":4})";
      const auto text = run_statistics(cfg, shapes,
                                       {1.0, 1.0, is3d ? 1.0 : 0.0}, path);
      const auto rows = data_rows(text);
      REQUIRE(rows.size() == 4);
      const bool has_note =
          text.find("# g(r) undefined") != std::string::npos;
      REQUIRE(has_note == (count < 2));
      for (auto const& r : rows) {
        const auto c1 = r.find(',');
        const auto c2 = r.find(',', c1 + 1);
        REQUIRE(c1 != std::string::npos && c2 != std::string::npos);
        REQUIRE(r.find(',', c2 + 1) == std::string::npos);
        if (c2 == std::string::npos) continue;
        REQUIRE((c2 + 1 == r.size()) == (count < 2));
      }
      if (count == 1 && !rows.empty()) {
        // S2 is kept: the first bin (r ~ 0) equals the voxel fraction > 0.
        REQUIRE(std::stod(rows[0].substr(rows[0].find(',') + 1)) > 0.0);
      }
    }
  }
}

// nx*ny*nz used to be an unchecked size_t product: it wrapped to a tiny or
// zero grid and the sampling loops then hung or wrote out of bounds.
void test_voxel_count_overflow_and_cap() {
  auto throws_invalid = [](std::size_t nx, std::size_t ny, std::size_t nz) {
    try {
      (void)rvegen::checked_voxel_count(nx, ny, nz);
    } catch (std::invalid_argument const& e) {
      const std::string w = e.what();
      return w.find("nx=") != std::string::npos &&
             w.find("ny=") != std::string::npos &&
             w.find("nz=") != std::string::npos;
    } catch (...) {
    }
    return false;
  };
  const std::size_t two62p1 = (std::size_t{1} << 62) + 1;
  REQUIRE(throws_invalid(4, two62p1, 1));                 // wraps to 4
  REQUIRE(throws_invalid(std::size_t{1} << 32, std::size_t{1} << 32, 1));  // wraps to 0
  REQUIRE(throws_invalid(std::size_t{1} << 22, std::size_t{1} << 22,
                         std::size_t{1} << 20));          // wraps to 0
  REQUIRE(throws_invalid(std::size_t{1} << 20, std::size_t{1} << 20, 8));  // over cap
  REQUIRE(throws_invalid(0, 4, 4));
  REQUIRE(throws_invalid(4, 0, 4));
  REQUIRE(throws_invalid(4, 4, 0));
  REQUIRE(rvegen::checked_voxel_count(3, 5, 7) == 105);
  REQUIRE(rvegen::checked_voxel_count(std::size_t{1} << 16, std::size_t{1} << 16, 1) ==
          (std::size_t{1} << 32));

  // Every consumer rejects the grid before allocating or looping.
  shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(0.5, 0.5, 0.5, 0.2));
  const std::array<double, 3> box{1.0, 1.0, 1.0};
  auto rejects = [](auto&& f) {
    try { f(); } catch (std::invalid_argument const&) { return true; } catch (...) {}
    return false;
  };
  REQUIRE(rejects([&] { (void)rvegen::sample_voxel_grid(shapes, box, 4, two62p1, 1); }));
  REQUIRE(rejects([&] {
    (void)rvegen::two_point_correlation_3d(shapes, box, std::size_t{1} << 32,
                                           std::size_t{1} << 32, 1, 4);
  }));
  std::stringstream sink;
  REQUIRE(rejects([&] {
    rvegen::vtk_legacy_writer<double>{4, two62p1, 1}.write(sink, shapes, box);
  }));
  REQUIRE(rejects([&] {
    rvegen::voxel_writer<double>{std::size_t{1} << 22, std::size_t{1} << 22,
                                 std::size_t{1} << 20}.write(sink, shapes, box);
  }));
  REQUIRE(rejects([&] {
    rvegen::voxel_writer<double>{4, 4, 0}.write(sink, shapes, box);
  }));
  // 2D ignores nz, so a huge nz must not trip the cap there.
  shape_vector c;
  c.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.2));
  REQUIRE(rvegen::sample_voxel_grid(c, {1.0, 1.0, 0.0}, 4, 4,
                                    std::size_t{1} << 40).size() == 16);
}

// The raw voxel header must carry the box extents at round-trip precision.
void test_voxel_header_extent_round_trips() {
  const std::array<double, 3> box{0.1234567890123, 1.0 / 3.0, 0.0};
  shape_vector c;
  c.emplace_back(std::make_unique<rvegen::circle<double>>(0.05, 0.1, 0.02));
  std::stringstream out;
  rvegen::voxel_writer<double>{4, 4, 1}.write(out, c, box);
  std::istringstream in{out.str()};
  std::string line;
  bool found = false;
  while (std::getline(in, line)) {
    if (line.rfind("# Lx Ly Lz:", 0) == 0) {
      std::istringstream ls{line.substr(11)};
      double a = 0, b = 0, d = -1;
      ls >> a >> b >> d;
      REQUIRE(a == box[0]);
      REQUIRE(b == box[1]);
      REQUIRE(d == box[2]);
      found = true;
    }
  }
  REQUIRE(found);
  REQUIRE(out.str().find("[rvegen-voxel-format: v3]") != std::string::npos);
}

} // namespace

int main() {
  test_voxel_matches_brute_force();
  test_rdf_zero_bins();
  test_nz1_samples_mid_plane();
  test_s2_3d_limits();
  test_rdf_3d_poisson();
  test_rdf_2d_poisson();
  test_vtk_cell_data_layout();
  test_vtk_spacing_round_trips();
  test_statistics_post_process_3d();
  test_statistics_csv_with_few_inclusions();
  test_voxel_count_overflow_and_cap();
  test_voxel_header_extent_round_trips();

  if (failures > 0) {
    std::cerr << failures << " voxel/stats 3D test failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "voxel_stats_3d_test: PASS\n";
  return EXIT_SUCCESS;
}
