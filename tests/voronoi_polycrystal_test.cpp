// Periodic Voronoi + grain-size control tests (#8).

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "rvegen/rvegen.h"
#include "rvegen/generators/grain_size_control.h"

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

using seeds3_t = std::vector<std::array<double, 3>>;

seeds3_t make_seeds3(std::size_t n, std::array<double, 3> L, unsigned seed) {
  std::mt19937 eng{seed};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  seeds3_t s;
  for (std::size_t i = 0; i < n; ++i)
    s.push_back({u(eng) * L[0], u(eng) * L[1], u(eng) * L[2]});
  return s;
}

// ---- 2D -------------------------------------------------------------------
void test_periodic_2d() {
  const double Lx = 2.0, Ly = 1.5;
  std::mt19937 eng{3};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  std::vector<std::array<double, 2>> s;
  for (int i = 0; i < 15; ++i) s.push_back({u(eng) * Lx, u(eng) * Ly});
  rvegen::voronoi_generator_2d<double> gen{Lx, Ly, s, true};
  auto polys = gen.build();
  double total = 0;
  std::vector<rvegen::convex_polygon<double>> shapes;
  for (auto const& p : polys) {
    total += rvegen::polygon_area(p);
    shapes.emplace_back(p);
  }
  REQUIRE(std::abs(total - Lx * Ly) < 1e-10);

  // Every box point belongs to exactly one cell modulo translation by L:
  // this is the "neighbour faces match across opposite boundaries" check.
  for (int k = 0; k < 3000; ++k) {
    const std::array<double, 3> q{u(eng) * Lx, u(eng) * Ly, 0.0};
    int hits = 0;
    for (auto const& c : shapes)
      for (int sx = -1; sx <= 1; ++sx)
        for (int sy = -1; sy <= 1; ++sy)
          if (c.is_inside({q[0] + sx * Lx, q[1] + sy * Ly, 0.0})) ++hits;
    REQUIRE(hits == 1);
  }

  // Congruence: translating all seeds (wrapped into the box) translates
  // every cell with its seed.
  auto s2 = s;
  for (auto& p : s2) {
    p[0] = std::fmod(p[0] + 0.7 * Lx, Lx);
    p[1] = std::fmod(p[1] + 0.3 * Ly, Ly);
  }
  rvegen::voronoi_generator_2d<double> gen2{Lx, Ly, s2, true};
  auto polys2 = gen2.build();
  for (std::size_t i = 0; i < s.size(); ++i) {
    REQUIRE(std::abs(rvegen::polygon_area(polys[i]) -
                     rvegen::polygon_area(polys2[i])) < 1e-9);
  }

  // Seeds on opposite faces coincide modulo the box: rejected.
  bool threw = false;
  try {
    rvegen::voronoi_generator_2d<double> g{1, 1, {{0, .5}, {1, .5}}, true};
  } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

// ---- 3D ---------------------------------------------------------------------
void check_periodic_3d(rvegen::voronoi_backend backend, std::size_t n,
                       std::array<double, 3> L, unsigned rng) {
  const double V = L[0] * L[1] * L[2];
  const auto s = make_seeds3(n, L, rng);
  rvegen::voronoi_generator_3d<double> gen{L[0], L[1], L[2], s, backend, true};
  auto cells = gen.build();
  double total = 0;
  for (auto const& c : cells) total += c->volume();
  REQUIRE(std::abs(total - V) < 1e-9 * V);

  // Exactly one cell contains each box point modulo L.
  std::mt19937 eng{99};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  for (int k = 0; k < 600; ++k) {
    const double qx = u(eng) * L[0], qy = u(eng) * L[1], qz = u(eng) * L[2];
    int hits = 0;
    for (auto const& c : cells)
      for (int sx = -1; sx <= 1; ++sx)
        for (int sy = -1; sy <= 1; ++sy)
          for (int sz = -1; sz <= 1; ++sz)
            if (c->is_inside({qx + sx * L[0], qy + sy * L[1], qz + sz * L[2]}))
              ++hits;
    REQUIRE(hits == 1);
  }

  // Congruence under a wrapped common translation: same volume per seed and
  // same sorted vertex distances from the seed.
  auto s2 = s;
  for (auto& p : s2)
    for (int k = 0; k < 3; ++k) p[k] = std::fmod(p[k] + 0.37 * L[k], L[k]);
  rvegen::voronoi_generator_3d<double> gen2{L[0], L[1], L[2], s2, backend, true};
  auto cells2 = gen2.build();
  auto radii = [](rvegen::shape_base<double> const& sh,
                  std::array<double, 3> const& p) {
    auto const& vc = static_cast<rvegen::voronoi_cell<double> const&>(sh);
    std::vector<double> r;
    for (auto const& v : vc.vertices())
      r.push_back(std::hypot(v[0] - p[0], v[1] - p[1], v[2] - p[2]));
    std::sort(r.begin(), r.end());
    return r;
  };
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(std::abs(cells[i]->volume() - cells2[i]->volume()) < 1e-9 * V);
    auto a = radii(*cells[i], s[i]);
    auto b = radii(*cells2[i], s2[i]);
    REQUIRE(a.size() == b.size());
    for (std::size_t k = 0; k < std::min(a.size(), b.size()); ++k)
      REQUIRE(std::abs(a[k] - b[k]) < 1e-9 * L[0]);
  }
}

void test_periodic_3d_in_house() {
  check_periodic_3d(rvegen::voronoi_backend::in_house, 1, {1, 1, 1}, 1);
  check_periodic_3d(rvegen::voronoi_backend::in_house, 10, {1.0, 1.5, 2.0}, 2);
  check_periodic_3d(rvegen::voronoi_backend::in_house, 10, {1e-6, 1e-6, 1e-6}, 4);
}

void test_weighted_2d_3d_partition() {
  const std::vector<double> w3{0.0, 0.02, -0.01, 0.015, 0.0, 0.03};
  const auto s = make_seeds3(6, {1, 1, 1}, 8);
  for (bool per : {false, true}) {
    rvegen::voronoi_generator_3d<double> g{1, 1, 1, s,
        rvegen::voronoi_backend::in_house, per, w3};
    double t = 0;
    for (auto const& c : g.build()) t += c->volume();
    REQUIRE(std::abs(t - 1.0) < 1e-8);
  }
  std::vector<std::array<double, 2>> s2{{.2, .3}, {.7, .6}, {.5, .9}, {.1, .8}};
  for (bool per : {false, true}) {
    rvegen::voronoi_generator_2d<double> g{1, 1, s2, per, {0.0, .02, -.01, .03}};
    double t = 0;
    for (auto const& p : g.build()) t += rvegen::polygon_area(p);
    REQUIRE(std::abs(t - 1.0) < 1e-10);
  }
}

#ifdef RVEGEN_WITH_VORO_PP
void test_periodic_3d_voro_pp() {
  check_periodic_3d(rvegen::voronoi_backend::voro_pp, 40, {1.0, 1.5, 2.0}, 5);
  check_periodic_3d(rvegen::voronoi_backend::voro_pp, 3, {1, 1, 1}, 6);
  // Backend agreement, periodic.
  const auto s = make_seeds3(12, {1, 1, 1}, 7);
  rvegen::voronoi_generator_3d<double> a{1, 1, 1, s, rvegen::voronoi_backend::in_house, true};
  rvegen::voronoi_generator_3d<double> b{1, 1, 1, s, rvegen::voronoi_backend::voro_pp, true};
  auto ca = a.build(), cb = b.build();
  for (std::size_t i = 0; i < s.size(); ++i)
    REQUIRE(std::abs(ca[i]->volume() - cb[i]->volume()) < 1e-9);
  // Backend agreement, weighted + periodic.
  std::vector<double> w(12);
  std::mt19937 eng{2};
  for (auto& x : w) x = std::uniform_real_distribution<double>{0, 0.02}(eng);
  rvegen::voronoi_generator_3d<double> aw{1, 1, 1, s, rvegen::voronoi_backend::in_house, true, w};
  rvegen::voronoi_generator_3d<double> bw{1, 1, 1, s, rvegen::voronoi_backend::voro_pp, true, w};
  auto cw = aw.build(), dw = bw.build();
  for (std::size_t i = 0; i < s.size(); ++i)
    REQUIRE(std::abs(cw[i]->volume() - dw[i]->volume()) < 1e-9);
}
#endif

// ---- grain-size control -------------------------------------------------
struct stats { double mean, sd; };
stats mean_sd(std::vector<double> const& v) {
  double m = 0;
  for (double x : v) m += x;
  m /= static_cast<double>(v.size());
  double q = 0;
  for (double x : v) q += (x - m) * (x - m);
  return {m, std::sqrt(q / static_cast<double>(v.size() - 1))};
}

void check_grain_sizes(rvegen::voronoi_backend backend, double L, double mean_d,
                       double sigma, double tol_mean, double tol_sd_rel) {
  std::mt19937 eng{123};
  const auto t0 = std::chrono::steady_clock::now();
  auto r = rvegen::lognormal_grain_seeds_3d<double>(L, L, L, mean_d, sigma, eng,
                                                    backend);
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  const auto st = mean_sd(r.diameters);
  // Analytic lognormal moments of D.
  const double sd_target = mean_d * std::sqrt(std::exp(sigma * sigma) - 1.0);
  std::cout << "grains=" << r.seeds.size() << " iters=" << r.iterations
            << " vol_err=" << r.max_rel_volume_error << " mean=" << st.mean
            << " sd=" << st.sd << " (target " << mean_d << ", " << sd_target
            << ") " << secs << " s\n";
  // Achieved diameters track the drawn targets themselves much tighter than
  // the analytic moments (which also carry the finite-N sampling error).
  const auto tg = mean_sd(r.target_diameters);
  REQUIRE(std::abs(st.mean - tg.mean) < 0.01 * tg.mean);
  REQUIRE(std::abs(st.sd - tg.sd) < 0.05 * tg.sd);
  REQUIRE(r.max_rel_volume_error <= 0.01);
  REQUIRE(std::abs(st.mean - mean_d) < tol_mean * mean_d);
  REQUIRE(std::abs(st.sd - sd_target) < tol_sd_rel * sd_target);
}

void test_grain_size_small_in_house() {
  check_grain_sizes(rvegen::voronoi_backend::in_house, 2.2, 1.0, 0.25, 0.05, 0.35);
}

#ifdef RVEGEN_WITH_VORO_PP
void test_grain_size_300_voro_pp() {
  // ~340 grains: stated tolerance = mean within 3 %, std within 15 %.
  check_grain_sizes(rvegen::voronoi_backend::voro_pp, 6.0, 1.0, 0.25, 0.03, 0.15);
}
#endif

// Multi-seed robustness (regression): with sigma = 0.8 the old iteration
// left empty cells and no flag on most engine seeds.
void check_grain_robust(rvegen::voronoi_backend backend, double L, double sigma,
                        unsigned first_seed, unsigned count) {
  for (unsigned sd = first_seed; sd < first_seed + count; ++sd) {
    std::mt19937 eng{sd};
    auto r = rvegen::lognormal_grain_seeds_3d<double>(L, L, L, 1.0, sigma, eng,
                                                      backend);
    std::cout << "robust seed=" << sd << " N=" << r.seeds.size()
              << " it=" << r.iterations << " restarts=" << r.restarts
              << " err=" << r.max_rel_volume_error
              << " empty=" << r.empty_cells << "\n";
    REQUIRE(r.converged);
    REQUIRE(r.empty_cells == 0);
    REQUIRE(r.max_rel_volume_error <= 0.01);
    for (double d : r.diameters) REQUIRE(d > 0.0);
    // Returned seeds/weights reproduce the reported volumes.
    rvegen::voronoi_generator_3d<double> gen{L, L, L, r.seeds, backend, true,
                                             r.weights};
    auto cells = gen.build();
    for (std::size_t i = 0; i < cells.size(); ++i)
      REQUIRE(std::abs(rvegen::equivalent_diameter(cells[i]->volume()) -
                       r.diameters[i]) < 1e-6 * r.diameters[i]);
  }
}

void test_grain_robust_in_house() {
  check_grain_robust(rvegen::voronoi_backend::in_house, 3.5, 0.8, 1, 6);
}

// A non-convergent configuration must be flagged, not silent.
void test_grain_nonconvergence_flagged() {
  std::mt19937 eng{1};
  auto r = rvegen::lognormal_grain_seeds_3d<double>(
      3.5, 3.5, 3.5, 1.0, 0.8, eng, rvegen::voronoi_backend::in_house, true,
      1e-12, 5, 0);
  REQUIRE(!r.converged);
  REQUIRE(r.max_rel_volume_error > 1e-12);
}

#ifdef RVEGEN_WITH_VORO_PP
void test_grain_robust_voro_pp() {
  check_grain_robust(rvegen::voronoi_backend::voro_pp, 6.0, 0.8, 1, 6);
  check_grain_robust(rvegen::voronoi_backend::voro_pp, 6.0, 0.5, 1, 3);
}
#endif

// ---- periodic seed validation -------------------------------------------
void test_validation_scales_and_wraps() {
  // 200k seeds must validate in well under a second (was O(N^2): 80 s).
  const auto t0 = std::chrono::steady_clock::now();
  rvegen::detail::validate_voronoi_seeds<double, 3>(
      {1.0, 1.0, 1.0}, make_seeds3(200000, {1, 1, 1}, 9), "t", true);
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  std::cout << "validate 200k periodic: " << secs << " s\n";
  REQUIRE(secs < 5.0);

  auto throws = [](seeds3_t s, std::array<double, 3> L, bool per) {
    try {
      rvegen::detail::validate_voronoi_seeds<double, 3>(L, s, "t", per);
    } catch (std::runtime_error const&) { return true; }
    return false;
  };
  // x = 0 and x = L are the same point modulo the box.
  REQUIRE(throws({{0.0, 0.5, 0.5}, {1.0, 0.5, 0.5}}, {1, 1, 1}, true));
  REQUIRE(!throws({{0.0, 0.5, 0.5}, {1.0, 0.5, 0.5}}, {1, 1, 1}, false));
  // Within tolerance across the wrap, other axes too.
  REQUIRE(throws({{1e-12, 0.5, 0.5}, {1.0 - 1e-12, 0.5, 0.5}}, {1, 1, 1}, true));
  REQUIRE(throws({{0.5, 0.5, 1e-12}, {0.5, 0.5, 1.0}}, {1, 1, 1}, true));
  REQUIRE(throws({{0.3, 0.3, 0.3}, {0.3, 0.3, 0.3 + 1e-12}}, {1, 1, 1}, false));
  REQUIRE(!throws({{0.3, 0.3, 0.3}, {0.3, 0.3, 0.3 + 1e-6}}, {1, 1, 1}, true));
  // Degenerate (planar) distribution must stay fast.
  seeds3_t planar;
  for (int i = 0; i < 100000; ++i)
    planar.push_back({0.5, (i % 320) / 320.0, (i / 320) / 320.0});
  const auto t1 = std::chrono::steady_clock::now();
  rvegen::detail::validate_voronoi_seeds<double, 3>({1, 1, 1}, planar, "t",
                                                    true);
  REQUIRE(std::chrono::duration<double>(std::chrono::steady_clock::now() - t1)
              .count() < 5.0);
}

// Generator-built cells carry no bounding box; wraps must still be found.
void test_periodic_wraps_without_bbox() {
  rvegen::voronoi_generator_3d<double> gen{
      1.0, 1.0, 1.0, {{0.05, 0.5, 0.5}, {0.55, 0.5, 0.5}},
      rvegen::voronoi_backend::in_house, true};
  auto cells = gen.build();
  std::size_t total = 0;
  for (auto const& c : cells) {
    REQUIRE(c->bounding_box() == nullptr);
    total += rvegen::generate_periodic_wraps<double>(*c, {1.0, 1.0, 1.0}).size();
  }
  REQUIRE(total > 0);
}

// ---- weighted empties, thin periodic box, gmsh clipping -----------------
void check_weighted_empty(rvegen::voronoi_backend backend) {
  seeds3_t seeds{{0.25, 0.5, 0.5}, {0.75, 0.5, 0.5}, {0.5, 0.2, 0.8}};
  // Seed 0's weight is far below its neighbours': its power cell is empty.
  std::vector<double> w{-10.0, 0.0, 0.0};
  rvegen::voronoi_generator_3d<double> gen{1, 1, 1, seeds, backend, true, w};
  auto cells = gen.build();
  const auto empty = gen.empty_cell_indices(cells);
  REQUIRE(empty.size() == 1 && empty[0] == 0);
  bool threw = false;
  try { (void)gen.build_checked(); } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
  // Zero weights: never empty, build_checked succeeds.
  rvegen::voronoi_generator_3d<double> ok{1, 1, 1, seeds, backend, true,
                                          std::vector<double>(3, 0.0)};
  REQUIRE(ok.build_checked().size() == 3);
}

void check_thin_periodic(rvegen::voronoi_backend backend) {
  const std::array<double, 3> L{1.0, 1.0, 1e-4};
  auto s = make_seeds3(12, L, 21);
  rvegen::voronoi_generator_3d<double> gen{L[0], L[1], L[2], s, backend, true};
  double total = 0;
  for (auto const& c : gen.build()) total += c->volume();
  std::cout << "thin periodic volume ratio " << total / (L[0] * L[1] * L[2])
            << "\n";
  REQUIRE(std::abs(total / (L[0] * L[1] * L[2]) - 1.0) < 1e-6);
}

void test_weighted_empty_and_thin_periodic() {
  check_weighted_empty(rvegen::voronoi_backend::in_house);
  check_thin_periodic(rvegen::voronoi_backend::in_house);
#ifdef RVEGEN_WITH_VORO_PP
  check_weighted_empty(rvegen::voronoi_backend::voro_pp);
  check_thin_periodic(rvegen::voronoi_backend::voro_pp);
#endif
}

void test_gmsh_clips_periodic_polygons() {
  const double Lx = 2.0, Ly = 1.5;
  std::mt19937 eng{5};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  std::vector<std::array<double, 2>> seeds;
  for (int i = 0; i < 8; ++i) seeds.push_back({u(eng) * Lx, u(eng) * Ly});
  rvegen::voronoi_generator_2d<double> gen{Lx, Ly, seeds, true};
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  for (auto const& p : gen.build()) {
    auto c = std::make_unique<rvegen::convex_polygon<double>>(p);
    c->make_bounding_box();
    for (auto& w : rvegen::generate_periodic_wraps<double>(*c, {Lx, Ly, 0.0}))
      shapes.push_back(std::move(w));
    shapes.push_back(std::move(c));
  }
  std::ostringstream out;
  rvegen::gmsh_geo_writer<double>{}.write(out, shapes, {Lx, Ly, 0.0});
  std::istringstream in{out.str()};
  std::string line;
  std::size_t points = 0;
  while (std::getline(in, line)) {
    // Plain `Point(N) = {x, y, 0};` or `_pK = newp; Point(_pK) = {x, y, 0};`.
    const auto pp = line.find("Point(");
    if (pp == std::string::npos) continue;
    const auto b = line.find('{', pp);
    double x, y;
    char c1;
    std::istringstream ls{line.substr(b + 1)};
    ls >> x >> c1 >> y;
    ++points;
    REQUIRE(x >= -1e-12 && x <= Lx + 1e-12);
    REQUIRE(y >= -1e-12 && y <= Ly + 1e-12);
  }
  REQUIRE(points > 8 * 3);
  if (const char* o = std::getenv("RVEGEN_TEST_GEO_OUT")) std::ofstream(o) << out.str();
}


// ---- review fixes (#8): volume check, near pairs, limits, wraps ---------
template <typename E, typename F>
bool throws_with(F&& f, std::string const& needle) {
  try { f(); } catch (E const& e) {
    return std::string{e.what()}.find(needle) != std::string::npos;
  } catch (...) {}
  return false;
}

// The volume-sum check names the backend and the mismatch.
void test_volume_sum_check() {
  using shapes_t = std::vector<std::unique_ptr<rvegen::shape_base<double>>>;
  auto box_cell = [](double x0, double x1) {
    using vec = gte::Vector<3, double>;
    auto mk = [](double x, double y, double z) { vec v; v[0] = x; v[1] = y; v[2] = z; return v; };
    std::vector<vec> v{mk(x0, 0, 0), mk(x1, 0, 0), mk(x1, 1, 0), mk(x0, 1, 0),
                       mk(x0, 0, 1), mk(x1, 0, 1), mk(x1, 1, 1), mk(x0, 1, 1)};
    return std::make_unique<rvegen::voronoi_cell<double>>(
        v, std::vector<std::vector<std::size_t>>{{0, 3, 2, 1}, {4, 5, 6, 7},
            {0, 1, 5, 4}, {2, 3, 7, 6}, {0, 4, 7, 3}, {1, 2, 6, 5}});
  };
  shapes_t good, overlapping;
  good.push_back(box_cell(0, 0.5)); good.push_back(box_cell(0.5, 1));
  overlapping.push_back(box_cell(0, 0.7)); overlapping.push_back(box_cell(0.4, 1));
  rvegen::detail::check_voronoi_volume_sum<double>(good, 1.0, "voro_pp", true, true);
  REQUIRE((throws_with<std::runtime_error>([&] {
    rvegen::detail::check_voronoi_volume_sum<double>(overlapping, 1.0, "voro_pp", true, true);
  }, "voro_pp, periodic, weighted): cell volumes sum to 1.300000")));
  REQUIRE((throws_with<std::runtime_error>([&] {
    rvegen::detail::check_voronoi_volume_sum<double>(overlapping, 1.0, "in_house", false, false);
  }, "in_house): cell volumes sum")));
}

// Close seed pairs in a periodic box: no volume may be lost (#8).
void check_close_pairs(rvegen::voronoi_backend backend) {
  for (double d : {1e-3, 1e-6, 1.5e-9}) {   // 1e-9 is the coincidence limit
    seeds3_t s{{0.5, 0.5, 0.5}, {0.5 + d, 0.5, 0.5}, {0.2, 0.3, 0.7}};
    rvegen::voronoi_generator_3d<double> gen{1, 1, 1, s, backend, true};
    double total = 0;
    auto cells = gen.build();   // also runs the volume-sum check
    for (auto const& c : cells) total += c->volume();
    REQUIRE(std::abs(total - 1.0) < 1e-9);
    REQUIRE(cells[2]->volume() > 0.49);
  }
  // Random near-pair sweep (18 of 200 lost up to 10 % before the fix).
  std::mt19937 e{3};
  std::uniform_real_distribution<double> u{0, 1};
  for (int rep = 0; rep < 60; ++rep) {
    seeds3_t s;
    for (int i = 0; i < 2 + rep % 6; ++i) s.push_back({u(e), u(e), u(e)});
    const double d = std::pow(10.0, -3 - 5.5 * u(e));
    s.push_back({std::min(1.0, s[0][0] + d), s[0][1], s[0][2]});
    rvegen::voronoi_generator_3d<double> gen{1, 1, 1, s, backend, true};
    double total = 0;
    for (auto const& c : gen.build()) total += c->volume();
    REQUIRE(std::abs(total - 1.0) < 1e-9);
  }
}

void test_periodic_close_pairs() {
  check_close_pairs(rvegen::voronoi_backend::in_house);
#ifdef RVEGEN_WITH_VORO_PP
  check_close_pairs(rvegen::voronoi_backend::voro_pp);
#endif
}

// A request the box cannot hold throws instead of clamping to 2 grains.
void test_grain_request_too_large_for_box() {
  std::mt19937 eng{1};
  REQUIRE((throws_with<std::invalid_argument>([&] {
    (void)rvegen::lognormal_grain_seeds_3d<double>(1, 1, 1, 1.0, 0.25, eng);
  }, "implies 1.5")));
  REQUIRE((throws_with<std::invalid_argument>([&] {
    (void)rvegen::lognormal_grain_seeds_3d<double>(0.5, 0.5, 0.5, 1.0, 0.25, eng);
  }, "mean diameter 1.0")));
  REQUIRE((throws_with<std::invalid_argument>([&] {
    (void)rvegen::lognormal_grain_seeds_3d<double>(
        1, 1, std::nan(""), 0.1, 0.25, eng);
  }, "finite")));
}

// Grain-count caps of the in-house backend, with a pointer to Voro++.
void test_in_house_grain_cap() {
  std::mt19937 eng{1};
  REQUIRE((throws_with<std::invalid_argument>([&] {
    (void)rvegen::lognormal_grain_seeds_3d<double>(
        10, 10, 10, 1.0, 0.0, eng, rvegen::voronoi_backend::in_house, false);
  }, "Voro++")));
  REQUIRE((throws_with<std::invalid_argument>([&] {
    (void)rvegen::lognormal_grain_seeds_3d<double>(
        20, 20, 20, 1.0, 0.0, eng, rvegen::voronoi_backend::in_house, true);
  }, "limit of 3000")));
  REQUIRE((throws_with<std::invalid_argument>([] {
    rvegen::voronoi_generator_3d<double> g{
        1, 1, 1, make_seeds3(301, {1, 1, 1}, 1), rvegen::voronoi_backend::in_house};
  }, "limit of 300 (bounded)")));
  // At the cap it is still accepted.
  rvegen::voronoi_generator_3d<double> ok{
      1, 1, 1, make_seeds3(300, {1, 1, 1}, 1), rvegen::voronoi_backend::in_house,
      true};
  REQUIRE(ok.seed_count() == 300);
}

// progress_options::cancel stops the tessellation and the weight iteration.
void test_cancel_hook() {
  rvegen::voronoi_generator_3d<double> gen{
      1, 1, 1, make_seeds3(40, {1, 1, 1}, 2), rvegen::voronoi_backend::in_house};
  rvegen::progress_options now;
  now.cancel = [] { return true; };
  bool threw = false;
  try { (void)gen.build(now); } catch (rvegen::voronoi_cancelled const&) { threw = true; }
  REQUIRE(threw);
  // Polled once per cell: cancel after 5 polls stops long before 40 cells.
  int polls = 0;
  rvegen::progress_options later;
  later.cancel = [&polls] { return ++polls > 5; };
  threw = false;
  try { (void)gen.build(later); } catch (rvegen::voronoi_cancelled const&) { threw = true; }
  REQUIRE(threw && polls == 6);

  // Grain-size iteration: stops with cancelled = true, converged = false.
  std::mt19937 eng{7};
  int it_polls = 0;
  rvegen::progress_options opts;
  opts.cancel = [&it_polls] { return ++it_polls > 400; };   // some iterations in
  const auto t0 = std::chrono::steady_clock::now();
  auto r = rvegen::lognormal_grain_seeds_3d<double>(
      3.5, 3.5, 3.5, 1.0, 0.8, eng, rvegen::voronoi_backend::in_house, true,
      1e-15, 100000, 5, opts);
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  REQUIRE(r.cancelled && !r.converged);
  REQUIRE(!r.seeds.empty());
  REQUIRE(secs < 60.0);
}

// Symmetric / lattice-like seed sets put many vertices on exactly degenerate
// positions (and on the atan2 branch cut of the face ordering). With a
// compiler that contracts a*b+c into FMA the in-house backend used to lose
// ~0.5 % of the volume on some of them; the volume-sum check in build()
// throws then, so merely building them is the assertion. Unweighted lattices
// additionally must give equal cells.
void test_symmetric_periodic_seed_sets() {
  using rvegen::voronoi_backend;
  auto run = [](seeds3_t const& s, std::vector<double> w, bool per) {
    rvegen::voronoi_generator_3d<double> g{1, 1, 1, s, voronoi_backend::in_house,
                                           per, std::move(w)};
    return g.build();   // throws std::runtime_error on a volume mismatch
  };
  seeds3_t three{{0.25, 0.5, 0.5}, {0.75, 0.5, 0.5}, {0.5, 0.2, 0.8}};
  for (bool per : {true, false}) {
    run(three, {}, per);
    run(three, {0, 0, 0}, per);
    run(three, {-10, 0, 0}, per);   // empties seed 0
    run(three, {0, -10, 0}, per);
    run(three, {0, 0, -10}, per);
    for (int n : {2, 3}) {
      seeds3_t lat, corner;
      for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) {
        lat.push_back({(i + .5) / n, (j + .5) / n, (k + .5) / n});
        corner.push_back({double(i) / n, double(j) / n, double(k) / n});
      }
      const double each = 1.0 / double(lat.size());
      if (per) {
        for (auto const& c : run(lat, {}, true)) REQUIRE(std::abs(c->volume() - each) < 1e-9);
        for (auto const& c : run(corner, {}, true)) REQUIRE(std::abs(c->volume() - each) < 1e-9);
      } else {
        run(lat, {}, false); run(corner, {}, false);
      }
      run(lat, std::vector<double>(lat.size(), 0.0), per);
      std::vector<double> w(lat.size(), 0.0);
      w[0] = -10;                       // a weight that empties a cell
      run(lat, w, per);
      w.assign(lat.size(), 0.0); w[1] = 0.01; w[2] = -0.01;
      run(lat, w, per);
    }
    seeds3_t bcc{{0.25, 0.25, 0.25}, {0.75, 0.75, 0.75}};
    run(bcc, {}, per); run(bcc, {0, 0.02}, per);
    seeds3_t half{{0.25, 0.25, 0.25}, {0.75, 0.25, 0.25}, {0.25, 0.75, 0.25}, {0.75, 0.75, 0.75}};
    run(half, {}, per); run(half, {-10, 0, 0, 0}, per);
    seeds3_t fcc{{0, 0, 0}, {.5, .5, 0}, {.5, 0, .5}, {0, .5, .5}};
    run(fcc, {}, per); run(fcc, {0, 0, 0, 0}, per);
    std::mt19937 e{5};
    std::uniform_int_distribution<int> q{0, 7};
    for (int rep = 0; rep < 60; ++rep) {   // random points of an 8^3 grid
      seeds3_t u;
      for (int i = 0; i < 3 + rep % 6; ++i) {
        std::array<double, 3> p{q(e) / 8.0, q(e) / 8.0, q(e) / 8.0};
        if (std::find(u.begin(), u.end(), p) == u.end()) u.push_back(p);
      }
      std::vector<double> w;
      if (rep % 4 == 1) for (std::size_t i = 0; i < u.size(); ++i) w.push_back((q(e) - 4) * 0.01);
      run(u, w, per);
    }
  }
}

// Periodic cells returned by voronoi_to_shapes must cover the whole box.
void test_voronoi_to_shapes_periodic_coverage_2d() {
  const double Lx = 2.0, Ly = 1.5;
  std::mt19937 eng{5};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  std::vector<std::array<double, 2>> seeds;
  for (int i = 0; i < 8; ++i) seeds.push_back({u(eng) * Lx, u(eng) * Ly});
  rvegen::voronoi_generator_2d<double> gen{Lx, Ly, seeds, true};
  auto shapes = rvegen::voronoi_to_shapes(gen);
  REQUIRE(shapes.size() > seeds.size());   // wrap pieces appended
  const std::size_t nx = 200, ny = 150;
  auto grid = rvegen::sample_voxel_grid<double>(shapes, {Lx, Ly, 0.0}, nx, ny, 1);
  std::vector<std::size_t> count(seeds.size() + 1, 0);
  for (auto id : grid) { REQUIRE(id <= seeds.size()); ++count[id]; }
  REQUIRE(count[0] == 0);   // every voxel assigned
  const double dV = Lx / nx * Ly / ny;
  auto polys = gen.build();
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    const double a = rvegen::polygon_area(polys[i]);
    REQUIRE(std::abs(count[i + 1] * dV - a) < 0.03 * a + 4 * dV);
  }
  // Every piece of grain i carries grain id i.
  for (std::size_t s = 0; s < shapes.size(); ++s)
    REQUIRE(shapes[s]->info().contains("grain_id"));
  for (std::size_t i = 0; i < seeds.size(); ++i)
    REQUIRE(shapes[i]->info().get<std::size_t>("grain_id") == i);
}

void check_voronoi_to_shapes_periodic_3d(rvegen::voronoi_backend backend) {
  const std::array<double, 3> L{1.0, 1.2, 0.9};
  auto seeds = make_seeds3(9, L, 11);
  rvegen::voronoi_generator_3d<double> gen{L[0], L[1], L[2], seeds, backend, true};
  auto shapes = rvegen::voronoi_to_shapes(gen);
  REQUIRE(shapes.size() > seeds.size());
  const std::size_t nx = 40, ny = 48, nz = 36;
  auto grid = rvegen::sample_voxel_grid<double>(shapes, L, nx, ny, nz);
  std::vector<std::size_t> count(seeds.size() + 1, 0);
  for (auto id : grid) { REQUIRE(id <= seeds.size()); ++count[id]; }
  REQUIRE(count[0] == 0);
  const double dV = L[0] / nx * L[1] / ny * L[2] / nz;
  auto cells = gen.build();
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    const double v = cells[i]->volume();
    REQUIRE(std::abs(count[i + 1] * dV - v) < 0.1 * v);
  }
}

void test_voronoi_to_shapes_periodic_coverage_3d() {
  check_voronoi_to_shapes_periodic_3d(rvegen::voronoi_backend::in_house);
#ifdef RVEGEN_WITH_VORO_PP
  check_voronoi_to_shapes_periodic_3d(rvegen::voronoi_backend::voro_pp);
#endif
}

// Non-finite or non-positive inputs name the offending argument.
void test_input_validation() {
  const double nan = std::nan(""), inf = std::numeric_limits<double>::infinity();
  seeds3_t s{{0.25, 0.5, 0.5}, {0.75, 0.5, 0.5}, {0.5, 0.2, 0.8}};
  using G3 = rvegen::voronoi_generator_3d<double>;
  REQUIRE((throws_with<std::invalid_argument>([&] { G3 g{1, 1, 1, s, rvegen::voronoi_backend::in_house, false, {0, nan, 0}}; }, "weights[1]")));
  REQUIRE((throws_with<std::invalid_argument>([&] { G3 g{1, 1, 1, s, rvegen::voronoi_backend::in_house, true, {0, 0, inf}}; }, "weights[2]")));
  REQUIRE((throws_with<std::invalid_argument>([&] { G3 g{nan, 1, 1, s}; }, "Lx")));
  REQUIRE((throws_with<std::invalid_argument>([&] { G3 g{1, inf, 1, s}; }, "Ly")));
  REQUIRE((throws_with<std::invalid_argument>([&] { G3 g{1, 1, 0, s}; }, "Lz")));
  REQUIRE((throws_with<std::invalid_argument>([&] { G3 g{1, 1, -1, s}; }, "Lz")));
  auto bad = s; bad[1][2] = nan;
  REQUIRE((throws_with<std::invalid_argument>([&] { G3 g{1, 1, 1, bad}; }, "seeds[1]")));
  bad = s; bad[0][0] = -inf;
  REQUIRE((throws_with<std::invalid_argument>([&] { G3 g{1, 1, 1, bad}; }, "seeds[0]")));
  using G2 = rvegen::voronoi_generator_2d<double>;
  std::vector<std::array<double, 2>> s2{{0.25, 0.5}, {0.75, 0.5}};
  REQUIRE((throws_with<std::invalid_argument>([&] { G2 g{1, 1, s2, false, {nan, 0}}; }, "weights[0]")));
  REQUIRE((throws_with<std::invalid_argument>([&] { G2 g{nan, 1, s2}; }, "Lx")));
  REQUIRE((throws_with<std::invalid_argument>([&] { G2 g{1, 0, s2}; }, "Ly")));
  auto bad2 = s2; bad2[1][0] = inf;
  REQUIRE((throws_with<std::invalid_argument>([&] { G2 g{1, 1, bad2}; }, "seeds[1]")));
}

#ifdef RVEGEN_WITH_VORO_PP
// A weighted seed whose power cell vanishes is reported as an empty cell
// (volume 0); a seed Voro++ cannot store is an error. A non-finite
// coordinate cannot be stored, so it takes the "dropped" path.
void test_voro_pp_dropped_vs_vanished() {
  seeds3_t seeds{{0.25, 0.5, 0.5}, {0.75, 0.5, 0.5}, {0.5, 0.2, 0.8}};
  rvegen::voronoi_generator_3d<double> gen{
      1, 1, 1, seeds, rvegen::voronoi_backend::voro_pp, false, {-10.0, 0.0, 0.0}};
  auto cells = gen.build();   // vanished cell: no exception
  REQUIRE(cells.size() == 3 && cells[0]->volume() == 0.0);
  seeds[1][0] = std::nan("");
  REQUIRE((throws_with<std::runtime_error>([&] {
    (void)rvegen::detail::build_voronoi_cells_voro_pp<double>(
        1, 1, 1, seeds, false, {0.0, 0.0, 0.0});
  }, "dropped")));
}
#endif

} // namespace

int main() {
  test_periodic_2d();
  test_periodic_3d_in_house();
  test_weighted_2d_3d_partition();
  test_grain_size_small_in_house();
  test_grain_robust_in_house();
  test_grain_nonconvergence_flagged();
  test_validation_scales_and_wraps();
  test_weighted_empty_and_thin_periodic();
  test_gmsh_clips_periodic_polygons();
  test_periodic_wraps_without_bbox();
  test_volume_sum_check();
  test_periodic_close_pairs();
  test_symmetric_periodic_seed_sets();
  test_grain_request_too_large_for_box();
  test_in_house_grain_cap();
  test_cancel_hook();
  test_voronoi_to_shapes_periodic_coverage_2d();
  test_voronoi_to_shapes_periodic_coverage_3d();
  test_input_validation();
#ifdef RVEGEN_WITH_VORO_PP
  test_voro_pp_dropped_vs_vanished();
  test_periodic_3d_voro_pp();
  test_grain_size_300_voro_pp();
  test_grain_robust_voro_pp();
#endif
  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "voronoi_polycrystal_test OK\n";
  return 0;
}
