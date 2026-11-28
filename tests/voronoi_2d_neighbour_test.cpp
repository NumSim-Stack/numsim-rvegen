// voronoi_generator_2d skips the clips that cannot cut a cell (#154). The
// cells must be the ones the full sequence of clips produces: every cell
// clipped by every other seed in list order. That sequence is kept here as
// the reference, and the comparison is exact (same vertices, same bits,
// same start vertex).

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
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

using D = double;
using point = std::array<D, 2>;
using polygon = std::vector<point>;

// ---- the full sequence, as the generator computed it before #154 -------------

polygon reference_clip(polygon const& poly, D A, D B, D C, D merge_tol) {
  polygon out;
  out.reserve(poly.size() + 1);
  const auto n = poly.size();
  if (n == 0) return out;
  auto dist = [A, B, C](point const& p) { return A * p[0] + B * p[1] - C; };
  auto push = [&out, merge_tol](point const& p) {
    if (!out.empty() && std::abs(out.back()[0] - p[0]) <= merge_tol &&
        std::abs(out.back()[1] - p[1]) <= merge_tol) {
      return;
    }
    out.push_back(p);
  };
  for (std::size_t k = 0; k < n; ++k) {
    auto const& start = poly[k];
    auto const& end = poly[(k + 1) % n];
    const D ds = dist(start);
    const D de = dist(end);
    const bool start_in = ds <= D{0};
    const bool end_in = de <= D{0};
    if (start_in != end_in) {
      const D t = ds / (ds - de);
      push(point{start[0] + t * (end[0] - start[0]),
                 start[1] + t * (end[1] - start[1])});
    }
    if (end_in) push(end);
  }
  if (out.size() > 1 && std::abs(out.front()[0] - out.back()[0]) <= merge_tol &&
      std::abs(out.front()[1] - out.back()[1]) <= merge_tol) {
    out.pop_back();
  }
  return out;
}

std::vector<polygon> reference_build(D Lx, D Ly, std::vector<point> const& seeds,
                                     bool periodic,
                                     std::vector<D> const& weights) {
  std::vector<polygon> out;
  const polygon bbox{{0, 0}, {Lx, 0}, {Lx, Ly}, {0, Ly}};
  const int range = periodic ? 1 : 0;
  const D merge_tol =
      std::numeric_limits<D>::epsilon() * D{16} * std::min(Lx, Ly);
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    polygon cell = bbox;
    auto const& p_i = seeds[i];
    if (periodic) {
      cell = {{p_i[0] - Lx / 2, p_i[1] - Ly / 2},
              {p_i[0] + Lx / 2, p_i[1] - Ly / 2},
              {p_i[0] + Lx / 2, p_i[1] + Ly / 2},
              {p_i[0] - Lx / 2, p_i[1] + Ly / 2}};
    }
    const D w_i = weights.empty() ? D{0} : weights[i];
    for (std::size_t j = 0; j < seeds.size() && cell.size() >= 3; ++j)
      for (int sx = -range; sx <= range && cell.size() >= 3; ++sx)
        for (int sy = -range; sy <= range && cell.size() >= 3; ++sy) {
          if (i == j && sx == 0 && sy == 0) continue;
          const point p_j{seeds[j][0] + sx * Lx, seeds[j][1] + sy * Ly};
          const D w_j = weights.empty() ? D{0} : weights[j];
          const D dx = p_j[0] - p_i[0];
          const D dy = p_j[1] - p_i[1];
          const D len = std::sqrt(dx * dx + dy * dy);
          const D A = dx / len;
          const D B = dy / len;
          const D C = A * (p_i[0] + p_j[0]) * D{0.5} +
                      B * (p_i[1] + p_j[1]) * D{0.5} -
                      (w_j - w_i) / (D{2} * len);
          cell = reference_clip(cell, A, B, C, merge_tol);
        }
    out.push_back(std::move(cell));
  }
  return out;
}

// ---- seeds ----------------------------------------------------------------------

struct rng {
  std::mt19937_64 engine;
  D uniform(D a, D b) {
    return a + (b - a) * (static_cast<D>(engine() >> 11) * 0x1.0p-53);
  }
};

std::vector<point> random_seeds(std::size_t n, D Lx, D Ly, rng& r) {
  std::vector<point> s;
  for (std::size_t i = 0; i < n; ++i)
    s.push_back({r.uniform(0, Lx), r.uniform(0, Ly)});
  return s;
}

// All seeds in a small patch, the rest of the box empty: cells are huge on
// one side and tiny on the other.
std::vector<point> clustered_seeds(std::size_t n, D Lx, D Ly, rng& r) {
  std::vector<point> s;
  for (std::size_t i = 0; i < n; ++i)
    s.push_back({r.uniform(0.6 * Lx, 0.7 * Lx), r.uniform(0.1 * Ly, 0.2 * Ly)});
  return s;
}

// Square lattice in shuffled order: four cells meet in every corner, every
// bisector of a diagonal neighbour passes through a vertex.
std::vector<point> lattice_seeds(std::size_t m, D Lx, D Ly, rng& r) {
  std::vector<point> s;
  for (std::size_t a = 0; a < m; ++a)
    for (std::size_t b = 0; b < m; ++b)
      s.push_back({(a + D{0.5}) * Lx / m, (b + D{0.5}) * Ly / m});
  std::shuffle(s.begin(), s.end(), r.engine);
  return s;
}

std::vector<point> line_seeds(std::size_t n, D Lx, D Ly) {
  std::vector<point> s;
  for (std::size_t i = 0; i < n; ++i)
    s.push_back({(i + D{0.5}) * Lx / n, Ly / 2});
  return s;
}

// ---- comparison -----------------------------------------------------------------

int compared = 0;

void same(std::string const& what, D Lx, D Ly, std::vector<point> const& seeds,
          bool periodic, std::vector<D> const& weights = {}) {
  const auto got =
      rvegen::voronoi_generator_2d<D>{Lx, Ly, seeds, periodic, weights}.build();
  const auto want = reference_build(Lx, Ly, seeds, periodic, weights);
  ++compared;
  if (got == want) return;
  std::size_t differing = 0, first = seeds.size();
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    if (got[i] != want[i]) {
      ++differing;
      first = std::min(first, i);
    }
  }
  std::cerr << "cells differ: " << what << ", " << seeds.size() << " seeds, "
            << (periodic ? "periodic" : "bounded") << ": " << differing
            << " cells, first " << first << " (" << got[first].size() << " vs "
            << want[first].size() << " vertices)\n";
  ++failures;
}

void test_identical_cells() {
  rng r{std::mt19937_64{154}};
  for (bool periodic : {false, true}) {
    for (std::size_t n : {1, 2, 3, 5, 15, 16, 17, 31, 32, 33, 64, 100, 257,
                          700, 1500}) {
      same("random, unit box", 1, 1, random_seeds(n, 1, 1, r), periodic);
      same("random, flat box", 3, 0.5, random_seeds(n, 3, 0.5, r), periodic);
    }
    same("random, tall box", 0.02, 7, random_seeds(900, 0.02, 7, r), periodic);
    same("random, tiny box", 1e-6, 2e-6, random_seeds(600, 1e-6, 2e-6, r),
         periodic);
    same("random, large box", 1e3, 4e2, random_seeds(600, 1e3, 4e2, r),
         periodic);
    for (std::size_t n : {20, 200, 1200})
      same("clustered", 1, 1, clustered_seeds(n, 1, 1, r), periodic);
    for (std::size_t m : {3, 4, 5, 9, 24})
      same("lattice", 1, 1.5, lattice_seeds(m, 1, 1.5, r), periodic);
    for (std::size_t n : {17, 80, 400})
      same("line", 2, 1, line_seeds(n, 2, 1), periodic);
    // Half the seeds on a lattice, half random: exact ties next to ordinary
    // cells.
    {
      auto s = lattice_seeds(12, 1, 1, r);
      const auto extra = random_seeds(150, 1, 1, r);
      s.insert(s.end(), extra.begin(), extra.end());
      std::shuffle(s.begin(), s.end(), r.engine);
      same("lattice and random", 1, 1, s, periodic);
    }
  }
}

// Power diagrams: weights move the bisectors, cells may be empty and need
// not contain their seed.
void test_identical_weighted_cells() {
  rng r{std::mt19937_64{155}};
  for (bool periodic : {false, true}) {
    for (std::size_t n : {4, 16, 17, 40, 300, 1200}) {
      const auto seeds = random_seeds(n, 1, 1, r);
      const D spacing2 = D{1} / static_cast<D>(n);   // mean spacing squared
      for (D strength : {0.05, 0.5, 3.0, 40.0}) {
        std::vector<D> w;
        for (std::size_t i = 0; i < n; ++i)
          w.push_back(r.uniform(-strength, strength) * spacing2);
        same("weighted " + std::to_string(strength), 1, 1, seeds, periodic, w);
      }
      // One dominant seed late in the list, and one early.
      std::vector<D> w(n, D{0});
      w[n - 1] = 30 * spacing2;
      same("one heavy seed, last", 1, 1, seeds, periodic, w);
      w[n - 1] = 0;
      w[0] = 30 * spacing2;
      same("one heavy seed, first", 1, 1, seeds, periodic, w);
    }
  }
  // The strong weights must have produced empty cells, or the case proves
  // little.
  const auto seeds = random_seeds(300, 1, 1, r);
  std::vector<D> w;
  for (std::size_t i = 0; i < seeds.size(); ++i)
    w.push_back(r.uniform(-40.0, 40.0) / 300.0);
  const auto cells = rvegen::voronoi_generator_2d<D>{1, 1, seeds, true, w}.build();
  const auto empty = std::count_if(cells.begin(), cells.end(),
                                   [](polygon const& c) { return c.size() < 3; });
  REQUIRE(empty > 20 && empty < 290);
}

// A size the full sequence would need about a quarter of an hour for; the
// test has a time limit in ctest. The cells still tile the box.
void test_many_seeds() {
  using clock = std::chrono::steady_clock;
  rng r{std::mt19937_64{156}};
  for (bool periodic : {false, true}) {
    const std::size_t n = 40000;
    const auto seeds = random_seeds(n, 2, 1, r);
    const auto t0 = clock::now();
    const auto cells =
        rvegen::voronoi_generator_2d<D>{2, 1, seeds, periodic}.build();
    const double s = std::chrono::duration<double>(clock::now() - t0).count();
    D area = 0;
    std::size_t vertices = 0;
    for (auto const& c : cells) {
      area += rvegen::polygon_area(c);
      vertices += c.size();
    }
    std::cout << "  " << n << " seeds, " << (periodic ? "periodic" : "bounded")
              << ": " << s << " s, " << static_cast<double>(vertices) / n
              << " vertices per cell\n";
    REQUIRE(cells.size() == n);
    REQUIRE(std::abs(area - 2.0) < 1e-9);
    // Euler: a cell of a random tessellation has six neighbours on average.
    REQUIRE(std::abs(static_cast<double>(vertices) / n - 6.0) < 0.1);
  }
}

} // namespace

int main() {
  test_identical_cells();
  test_identical_weighted_cells();
  test_many_seeds();
  std::cout << "  " << compared << " tessellations compared with the full "
            << "sequence of clips\n";
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "voronoi_2d_neighbour_test: all checks passed\n";
  return EXIT_SUCCESS;
}
