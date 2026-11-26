// Voronoi backend selection tests (#70). With RVEGEN_WITH_VORO_PP=OFF only
// the "requesting Voro++ throws" path runs; with it ON the Voro++ backend is
// checked against the in-house one and against the partition-of-unity
// invariant, plus a 200+ seed timing regression.

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
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

using seeds_t = std::vector<std::array<double, 3>>;

seeds_t make_seeds(std::size_t n, double L, unsigned seed) {
  std::mt19937 eng{seed};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  seeds_t s;
  for (std::size_t i = 0; i < n; ++i)
    s.push_back({u(eng) * L, u(eng) * L, u(eng) * L});
  return s;
}

std::vector<double> volumes(seeds_t const& s, double L,
                            rvegen::voronoi_backend b) {
  rvegen::voronoi_generator_3d<double> gen{L, L, L, s, b};
  std::vector<double> v;
  for (auto const& c : gen.build()) v.push_back(c->volume());
  return v;
}

double sum(std::vector<double> const& v) {
  double t = 0;
  for (double x : v) t += x;
  return t;
}

#ifndef RVEGEN_WITH_VORO_PP

void test_voro_pp_unavailable_throws() {
  REQUIRE(!rvegen::voro_pp_available());
  bool threw = false;
  try {
    rvegen::voronoi_generator_3d<double> g{
        1, 1, 1, {{.5, .5, .5}}, rvegen::voronoi_backend::voro_pp};
  } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

#else

void test_voro_pp_partition_of_unity_across_scales() {
  for (double L : {1e-6, 1.0, 1e3}) {
    auto v = volumes(make_seeds(40, L, 9), L, rvegen::voronoi_backend::voro_pp);
    REQUIRE(std::abs(sum(v) / (L * L * L) - 1.0) < 1e-9);
  }
}

void test_voro_pp_matches_in_house() {
  for (double L : {1e-6, 1.0, 1e3}) {
    const auto s = make_seeds(30, L, 13);
    const auto a = volumes(s, L, rvegen::voronoi_backend::in_house);
    const auto b = volumes(s, L, rvegen::voronoi_backend::voro_pp);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      REQUIRE(std::abs(a[i] - b[i]) < 1e-9 * L * L * L);
  }
}

void test_voro_pp_boundary_seeds_and_validation() {
  const double L = 2.0;
  seeds_t s{{0, 1, 1}, {L, 1, 1}, {1, 0, 0}, {1, L, L},
            {0, 0, 0}, {L, L, L}, {.8, 1.1, 1.3}};
  const auto a = volumes(s, L, rvegen::voronoi_backend::in_house);
  const auto b = volumes(s, L, rvegen::voronoi_backend::voro_pp);
  REQUIRE(std::abs(sum(b) - L * L * L) < 1e-9);
  for (std::size_t i = 0; i < a.size(); ++i) REQUIRE(std::abs(a[i] - b[i]) < 1e-9);

  bool threw = false;   // validation is shared by both backends
  try {
    rvegen::voronoi_generator_3d<double> g{
        1, 1, 1, {{.5, .5, .5}, {.5, .5, .5}}, rvegen::voronoi_backend::voro_pp};
  } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_voro_pp_cells_contain_their_seed() {
  const auto s = make_seeds(25, 1.0, 5);
  rvegen::voronoi_generator_3d<double> gen{1, 1, 1, s,
                                           rvegen::voronoi_backend::voro_pp};
  auto cells = gen.build();
  REQUIRE(cells.size() == s.size());
  for (std::size_t i = 0; i < s.size(); ++i) REQUIRE(cells[i]->is_inside(s[i]));
}

void test_voro_pp_200_plus_seeds_fast() {
  const auto s = make_seeds(300, 1.0, 17);
  const auto t0 = std::chrono::steady_clock::now();
  auto v = volumes(s, 1.0, rvegen::voronoi_backend::voro_pp);
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  std::cout << "voro++ 300 seeds: " << secs << " s\n";
  REQUIRE(secs < 1.0);
  REQUIRE(std::abs(sum(v) - 1.0) < 1e-9);
}

#endif

void test_default_backend_is_in_house() {
  const auto s = make_seeds(12, 1.0, 1);
  rvegen::voronoi_generator_3d<double> gen{1, 1, 1, s};   // default
  double t = 0;
  for (auto const& c : gen.build()) t += c->volume();
  REQUIRE(std::abs(t - 1.0) < 1e-9);
}

} // namespace

int main() {
  test_default_backend_is_in_house();
#ifndef RVEGEN_WITH_VORO_PP
  test_voro_pp_unavailable_throws();
#else
  test_voro_pp_partition_of_unity_across_scales();
  test_voro_pp_matches_in_house();
  test_voro_pp_boundary_seeds_and_validation();
  test_voro_pp_cells_contain_their_seed();
  test_voro_pp_200_plus_seeds_fast();
#endif
  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "voronoi_backend_test OK\n";
  return 0;
}
