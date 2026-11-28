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
// 6. Shape-index mode: a periodic primary and all its wraps share one id,
//    and the number of distinct ids equals the number of primaries.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <set>
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

  // r = 0: S2 is the voxel volume fraction, exactly (the pair counts are
  // integers, #127). With 256 bins the bins are narrower than a voxel, and
  // the first ones hold no lag at all.
  const auto fine = rvegen::two_point_correlations(shapes, box, n, n, n, 256);
  REQUIRE(fine.inclusion.zero == phi_v);
  const auto s2_fine = rvegen::two_point_correlation_3d(shapes, box, n, n, n, 256);
  REQUIRE(s2_fine.size() == 256);
  REQUIRE(std::isnan(s2_fine.front().s2));
  // The first lag is one voxel, 1/32: bin 16 of 256 up to 0.5.
  REQUIRE(s2_fine[16].s2 < phi_v && s2_fine[16].s2 > 0.7 * phi_v);

  // r large (≥ ~4R): S2 → φ².
  const auto s2 = rvegen::two_point_correlation_3d(shapes, box, n, n, n, 16);
  REQUIRE(s2.size() == 16);
  REQUIRE(std::abs(s2.back().r - 0.5 * 15.5 / 16.0) < 1e-12);
  double tail = 0.0;
  std::size_t count = 0;
  for (auto const& s : s2) {
    if (s.r >= 0.3) { tail += s.s2; ++count; }
  }
  tail /= double(count);
  const double phi2 = phi_v * phi_v;
  std::cout << "  S2_3d: phi_v=" << phi_v << " S2(0)=" << fine.inclusion.zero
            << " tail=" << tail << " phi^2=" << phi2 << '\n';
  REQUIRE(std::abs(tail - phi2) < 0.05 * phi2);
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
  // Shortest text that reads back to exactly 1/3 (format_number); the 17
  // digit form 0.33333333333333331 is the same double.
  REQUIRE(text.find("SPACING 0.5 0.5 0.3333333333333333\n") != std::string::npos);
  REQUIRE(std::stod("0.3333333333333333") == 1.0 / 3.0);
  REQUIRE(std::stod("0.33333333333333331") == 1.0 / 3.0);
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

// Issue #81 item 3: in shape-index mode a periodic primary and its wraps
// form one inclusion and must share one id. Place a shape across the
// corner plus random shapes with periodic_generator, then check that
// every corner voxel carries the same id and that the number of distinct
// ids equals the number of primaries (centre inside the box).
std::set<rvegen::phase_id> nonzero_ids(std::vector<rvegen::phase_id> const& g) {
  std::set<rvegen::phase_id> ids;
  for (auto v : g) if (v != 0) ids.insert(v);
  return ids;
}

std::size_t count_primaries(shape_vector const& shapes,
                            std::array<double, 3> const& box) {
  std::size_t n = 0;
  for (auto const& s : shapes) {
    if (rvegen::is_centre_in_domain(s->get_middle_point(), box)) ++n;
  }
  return n;
}

void test_shape_index_wraps_share_primary_id_2d() {
  std::mt19937 engine{5};
  rvegen::constant_distribution<double> corner{0.95};
  rvegen::constant_distribution<double> r_corner{0.1};
  rvegen::uniform_real_distribution<double> pos{0.0, 1.0, engine};
  rvegen::constant_distribution<double> r_small{0.04};
  rvegen::periodic_generator<double>::input_vector inputs;
  inputs.push_back(
      std::make_unique<rvegen::circle_input<double>>(corner, corner, r_corner));
  inputs.push_back(
      std::make_unique<rvegen::circle_input<double>>(pos, pos, r_small));

  const std::array<double, 3> box{1.0, 1.0, 0.0};
  rvegen::periodic_generator<double> generator{2000};
  rvegen::number_of_inclusions<double> termination{40};  // entries, incl. wraps
  const auto shapes = generator.compute(inputs, termination, box);
  const std::size_t primaries = count_primaries(shapes, box);
  REQUIRE(shapes.size() > primaries + 3);  // corner circle alone adds 3 wraps

  constexpr std::size_t n = 64;
  const auto grid = rvegen::sample_voxel_grid(shapes, box, n, n, 1);
  const auto corner_id = grid[0];
  REQUIRE(corner_id != 0);
  REQUIRE(grid[n - 1] == corner_id);
  REQUIRE(grid[(n - 1) * n] == corner_id);
  REQUIRE(grid[(n - 1) * n + n - 1] == corner_id);
  // The corner circle is the first primary placed, so it is inclusion 1.
  REQUIRE(corner_id == 1);

  const auto ids = nonzero_ids(grid);
  REQUIRE(ids.size() == primaries);
  REQUIRE(*ids.rbegin() == primaries);  // ids are compact: 1..primaries

  // The writers emit the same ids.
  rvegen::vtk_legacy_writer<double> vtk{n, n, 1};
  std::stringstream out;
  vtk.write(out, shapes, box);
  const auto body = vtk_body(out.str());
  REQUIRE(body.size() == n * n);
  REQUIRE(body.front() == int(corner_id) && body.back() == int(corner_id));
  std::set<int> vtk_ids;
  for (int v : body) if (v != 0) vtk_ids.insert(v);
  REQUIRE(vtk_ids.size() == primaries);
  rvegen::voxel_writer<double> vox{n, n, 1};
  REQUIRE(vox.sample(shapes, box) == grid);
}

void test_shape_index_wraps_share_primary_id_3d() {
  std::mt19937 engine{9};
  rvegen::constant_distribution<double> corner{0.95};
  rvegen::constant_distribution<double> r_corner{0.12};
  rvegen::uniform_real_distribution<double> pos{0.0, 1.0, engine};
  rvegen::constant_distribution<double> r_small{0.08};
  rvegen::periodic_generator<double>::input_vector inputs;
  inputs.push_back(std::make_unique<rvegen::sphere_input<double>>(
      corner, corner, corner, r_corner));
  inputs.push_back(std::make_unique<rvegen::sphere_input<double>>(
      pos, pos, pos, r_small));

  const std::array<double, 3> box{1.0, 1.0, 1.0};
  rvegen::periodic_generator<double> generator{2000};
  rvegen::number_of_inclusions<double> termination{40};
  const auto shapes = generator.compute(inputs, termination, box);
  const std::size_t primaries = count_primaries(shapes, box);
  REQUIRE(shapes.size() > primaries + 7);  // corner sphere alone adds 7 wraps

  constexpr std::size_t n = 32;
  const auto grid = rvegen::sample_voxel_grid(shapes, box, n, n, n);
  auto at = [&](std::size_t i, std::size_t j, std::size_t k) {
    return grid[(k * n + j) * n + i];
  };
  const auto corner_id = at(0, 0, 0);
  REQUIRE(corner_id == 1);
  for (std::size_t c = 0; c < 8; ++c) {
    REQUIRE(at((c & 1) ? n - 1 : 0, (c & 2) ? n - 1 : 0,
               (c & 4) ? n - 1 : 0) == corner_id);
  }
  const auto ids = nonzero_ids(grid);
  REQUIRE(ids.size() == primaries);
  REQUIRE(*ids.rbegin() == primaries);
}

// Shapes without a periodic link keep the plain 1-based shape index.
void test_shape_index_without_wraps_unchanged() {
  shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.25, 0.5, 0.1));
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.75, 0.5, 0.1));
  const auto grid = rvegen::sample_voxel_grid(shapes, {1.0, 1.0, 0.0}, 8, 8, 1);
  REQUIRE(grid[3 * 8 + 1] == 1);  // centre (0.1875, 0.4375)
  REQUIRE(grid[3 * 8 + 5] == 2);  // centre (0.6875, 0.4375)
}

void test_statistics_post_process_3d() {
  const auto shapes = random_periodic_spheres(30, 0.08, 3);
  const std::string path3 = "voxel_stats_3d_test_stats3d.csv";
  const auto text = run_statistics(R"({
    "type": "statistics", "output_path": ")" + path3 + R"(",
    "nx": 16, "ny": 16, "nz": 16, "nbins": 8 })",
                                   shapes, {1.0, 1.0, 1.0}, path3);
  REQUIRE(text.find("# dimension = 3D\n") != std::string::npos);
  // Outside the pipeline nothing says the box is periodic: pairs inside
  // the box only. The row of r = 0 comes first, then the bins.
  const auto s2 = rvegen::two_point_correlation_3d(
      shapes, {1.0, 1.0, 1.0}, 16, 16, 16, 8,
      rvegen::two_point_boundary::bounded);
  const auto g = rvegen::radial_distribution_3d(shapes, {1.0, 1.0, 1.0}, 8);
  // The first bin ends where the first lag (one voxel, 1/16) begins: it
  // holds no lag and its S2 field is empty. The second one has a value.
  REQUIRE(std::isnan(s2[0].s2));
  std::ostringstream bins;
  bins << std::setprecision(12) << '\n'
       << s2[0].r << ",," << g[0].g << ",,,\n"
       << s2[1].r << ',' << s2[1].s2 << ',' << g[1].g << ',';
  REQUIRE(text.find(bins.str()) != std::string::npos);
  REQUIRE(text.find("\n0,") != std::string::npos);
  REQUIRE(text.find("# columns: r, S2(r), g(r), S2_x(r), S2_y(r), S2_z(r)\n") !=
          std::string::npos);

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

// Inclusion ids live outside info(): a user "rvegen_inclusion_id" metadata
// key (even a non-integer one) is harmless, wrap-less shapes carry no id,
// and ids stay unique across generate() calls.
void test_inclusion_id_isolation() {
  shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.25, 0.5, 0.1));
  shapes.back()->info().set("rvegen_inclusion_id", std::string{"user"});
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.75, 0.5, 0.1));
  REQUIRE(!rvegen::inclusion_id(*shapes[0]).has_value());
  const auto grid = rvegen::sample_voxel_grid(shapes, {1.0, 1.0, 0.0}, 8, 8, 1);
  REQUIRE(grid[3 * 8 + 1] == 1 && grid[3 * 8 + 5] == 2);

  auto run = [](unsigned seed) {
    std::mt19937 engine{seed};
    rvegen::constant_distribution<double> corner{0.95};
    rvegen::constant_distribution<double> r{0.1};
    rvegen::periodic_generator<double>::input_vector inputs;
    inputs.push_back(
        std::make_unique<rvegen::circle_input<double>>(corner, corner, r));
    rvegen::periodic_generator<double> g{200};
    rvegen::number_of_inclusions<double> term{4};
    return g.compute(inputs, term, {1.0, 1.0, 0.0});
  };
  const auto a = run(1), b = run(2);
  std::set<std::size_t> ida, idb;
  for (auto const& s : a) if (auto i = rvegen::inclusion_id(*s)) ida.insert(*i);
  for (auto const& s : b) if (auto i = rvegen::inclusion_id(*s)) idb.insert(*i);
  REQUIRE(!ida.empty() && !idb.empty());
  for (auto i : ida) REQUIRE(idb.count(i) == 0);
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
      REQUIRE(rows.size() == 5);   // r = 0 and the four bins
      const bool has_note =
          text.find("# g(r) undefined") != std::string::npos;
      REQUIRE(has_note == (count < 2));
      // r, S2, g and one S2 column per axis; g is the third field and is
      // empty at r = 0 and wherever it is undefined.
      for (std::size_t n = 0; n < rows.size(); ++n) {
        auto const& r = rows[n];
        const auto c1 = r.find(',');
        const auto c2 = r.find(',', c1 + 1);
        const auto c3 = r.find(',', c2 + 1);
        REQUIRE(c1 != std::string::npos && c2 != std::string::npos &&
                c3 != std::string::npos);
        REQUIRE(std::count(r.begin(), r.end(), ',') == (is3d ? 5 : 4));
        if (c3 == std::string::npos) continue;
        REQUIRE((c3 == c2 + 1) == (count < 2 || n == 0));
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
  test_inclusion_id_isolation();
  test_voxel_matches_brute_force();
  test_rdf_zero_bins();
  test_nz1_samples_mid_plane();
  test_s2_3d_limits();
  test_rdf_3d_poisson();
  test_rdf_2d_poisson();
  test_vtk_cell_data_layout();
  test_vtk_spacing_round_trips();
  test_shape_index_wraps_share_primary_id_2d();
  test_shape_index_wraps_share_primary_id_3d();
  test_shape_index_without_wraps_unchanged();
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
