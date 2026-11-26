// Periodic Voronoi + grain-size control tests (#8).

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
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

} // namespace

int main() {
  test_periodic_2d();
  test_periodic_3d_in_house();
  test_weighted_2d_3d_partition();
  test_grain_size_small_in_house();
#ifdef RVEGEN_WITH_VORO_PP
  test_periodic_3d_voro_pp();
  test_grain_size_300_voro_pp();
#endif
  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "voronoi_polycrystal_test OK\n";
  return 0;
}
