// voronoi_generator_2d skips the clips that cannot cut a cell (#154). The
// cells must be the ones the full sequence of clips produces: every cell
// clipped by every other seed, the seeds taken in clip_order(). That
// sequence is kept here as the reference, and the comparison is exact (same
// vertices, same bits, same start vertex), for double and for float.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
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

// FNV-1a over clip_order(1000): pins the order across standard libraries.
#define CLIP_ORDER_HASH 0x76cf3e6954878b6bull

using D = double;
template <typename T> using point_t = std::array<T, 2>;
template <typename T> using polygon_t = std::vector<point_t<T>>;
using point = point_t<D>;
using polygon = polygon_t<D>;

// ---- the full sequence of clips ----------------------------------------------

template <typename T>
polygon_t<T> reference_clip(polygon_t<T> const& poly, T A, T B, T C,
                            T merge_tol) {
  polygon_t<T> out;
  out.reserve(poly.size() + 1);
  const auto n = poly.size();
  if (n == 0) return out;
  auto dist = [A, B, C](point_t<T> const& p) {
    return A * p[0] + B * p[1] - C;
  };
  auto push = [&out, merge_tol](point_t<T> const& p) {
    if (!out.empty() && std::abs(out.back()[0] - p[0]) <= merge_tol &&
        std::abs(out.back()[1] - p[1]) <= merge_tol) {
      return;
    }
    out.push_back(p);
  };
  for (std::size_t k = 0; k < n; ++k) {
    auto const& start = poly[k];
    auto const& end = poly[(k + 1) % n];
    const T ds = dist(start);
    const T de = dist(end);
    const bool start_in = ds <= T{0};
    const bool end_in = de <= T{0};
    if (start_in != end_in) {
      const T t = ds / (ds - de);
      push(point_t<T>{start[0] + t * (end[0] - start[0]),
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

template <typename T>
std::vector<polygon_t<T>> reference_build(T Lx, T Ly,
                                          std::vector<point_t<T>> const& seeds,
                                          bool periodic,
                                          std::vector<T> const& weights) {
  std::vector<polygon_t<T>> out;
  const polygon_t<T> bbox{{0, 0}, {Lx, 0}, {Lx, Ly}, {0, Ly}};
  const int range = periodic ? 1 : 0;
  const T merge_tol =
      std::numeric_limits<T>::epsilon() * T{16} * std::min(Lx, Ly);
  const auto order =
      rvegen::voronoi_generator_2d<T>::clip_order(seeds.size());
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    polygon_t<T> cell = bbox;
    auto const& p_i = seeds[i];
    if (periodic) {
      cell = {{p_i[0] - Lx / 2, p_i[1] - Ly / 2},
              {p_i[0] + Lx / 2, p_i[1] - Ly / 2},
              {p_i[0] + Lx / 2, p_i[1] + Ly / 2},
              {p_i[0] - Lx / 2, p_i[1] + Ly / 2}};
    }
    const T w_i = weights.empty() ? T{0} : weights[i];
    for (std::size_t r = 0; r < seeds.size() && cell.size() >= 3; ++r) {
      const std::size_t j = order[r];
      for (int sx = -range; sx <= range && cell.size() >= 3; ++sx)
        for (int sy = -range; sy <= range && cell.size() >= 3; ++sy) {
          if (i == j && sx == 0 && sy == 0) continue;
          const point_t<T> p_j{seeds[j][0] + sx * Lx, seeds[j][1] + sy * Ly};
          const T w_j = weights.empty() ? T{0} : weights[j];
          const T dx = p_j[0] - p_i[0];
          const T dy = p_j[1] - p_i[1];
          const T len = std::sqrt(dx * dx + dy * dy);
          const T A = dx / len;
          const T B = dy / len;
          const T C = A * (p_i[0] + p_j[0]) * T{0.5} +
                      B * (p_i[1] + p_j[1]) * T{0.5} -
                      (w_j - w_i) / (T{2} * len);
          cell = reference_clip(cell, A, B, C, merge_tol);
        }
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

std::vector<point> sorted_by_x(std::vector<point> s) {
  std::sort(s.begin(), s.end());
  return s;
}

// Lattice in row-major order, each seed moved a little: the order a
// structured seed list has.
std::vector<point> row_major_seeds(std::size_t m, D Lx, D Ly, D jitter, rng& r) {
  std::vector<point> s;
  for (std::size_t a = 0; a < m; ++a)
    for (std::size_t b = 0; b < m; ++b)
      s.push_back({(a + D{0.5} + r.uniform(-jitter, jitter)) * Lx / m,
                   (b + D{0.5} + r.uniform(-jitter, jitter)) * Ly / m});
  return s;
}

// ---- comparison -----------------------------------------------------------------

int compared = 0;

template <typename T>
void same_as(std::string const& what, D Lx, D Ly,
             std::vector<point> const& seeds_d, bool periodic,
             std::vector<D> const& weights_d) {
  std::vector<point_t<T>> seeds;
  for (auto const& q : seeds_d)
    seeds.push_back({static_cast<T>(q[0]), static_cast<T>(q[1])});
  const std::vector<T> weights(weights_d.begin(), weights_d.end());
  const T lx = static_cast<T>(Lx), ly = static_cast<T>(Ly);
  std::vector<polygon_t<T>> got;
  try {
    got = rvegen::voronoi_generator_2d<T>{lx, ly, seeds, periodic, weights}
              .build();
  } catch (std::runtime_error const&) {
    return;   // seeds that coincide in this precision: nothing to compare
  }
  const auto want = reference_build<T>(lx, ly, seeds, periodic, weights);
  ++compared;
  if (got == want) return;
  std::size_t differing = 0, first = seeds.size();
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    if (got[i] != want[i]) {
      ++differing;
      first = std::min(first, i);
    }
  }
  std::cerr << "cells differ (" << (sizeof(T) == 4 ? "float" : "double")
            << "): " << what << ", " << seeds.size() << " seeds, "
            << (periodic ? "periodic" : "bounded") << ": " << differing
            << " cells, first " << first << " (" << got[first].size() << " vs "
            << want[first].size() << " vertices)\n";
  ++failures;
}

void same(std::string const& what, D Lx, D Ly, std::vector<point> const& seeds,
          bool periodic, std::vector<D> const& weights = {}) {
  same_as<D>(what, Lx, Ly, seeds, periodic, weights);
  // The reference costs the square of the number of seeds: float for the
  // sizes that stay cheap.
  if (seeds.size() <= 600)
    same_as<float>(what, Lx, Ly, seeds, periodic, weights);
}

void test_identical_cells() {
  rng r{std::mt19937_64{154}};
  for (bool periodic : {false, true}) {
    for (std::size_t n : {1, 2, 3, 5, 15, 16, 17, 31, 32, 33, 64, 100, 257,
                          600, 1000}) {
      same("random, unit box", 1, 1, random_seeds(n, 1, 1, r), periodic);
      same("random, flat box", 3, 0.5, random_seeds(n, 3, 0.5, r), periodic);
    }
    same("random, tall box", 0.02, 7, random_seeds(900, 0.02, 7, r), periodic);
    same("random, tiny box", 1e-6, 2e-6, random_seeds(600, 1e-6, 2e-6, r),
         periodic);
    same("random, large box", 1e3, 4e2, random_seeds(600, 1e3, 4e2, r),
         periodic);
    for (std::size_t n : {20, 200, 900})
      same("clustered", 1, 1, clustered_seeds(n, 1, 1, r), periodic);
    for (std::size_t m : {3, 4, 5, 9, 24})
      same("lattice", 1, 1.5, lattice_seeds(m, 1, 1.5, r), periodic);
    for (std::size_t n : {17, 80, 400})
      same("line", 2, 1, line_seeds(n, 2, 1), periodic);
    // Lists in an order: sorted, and row by row through a lattice.
    for (std::size_t n : {40, 300, 1000})
      same("sorted by x", 1, 1, sorted_by_x(random_seeds(n, 1, 1, r)),
           periodic);
    for (std::size_t m : {5, 12, 30})
      same("row-major lattice", 1, 2, row_major_seeds(m, 1, 2, 0.0, r),
           periodic);
    same("row-major lattice, jittered", 1, 1,
         row_major_seeds(30, 1, 1, 0.3, r), periodic);
    // Seeds on the edges and in the corners of the box.
    {
      auto s = random_seeds(200, 1, 1, r);
      for (std::size_t k = 0; k < 40; ++k) {
        s.push_back({r.uniform(0, 1), k % 2 == 0 ? D{0} : D{1}});
        s.push_back({k % 2 == 0 ? D{0} : D{1}, r.uniform(0, 1)});
      }
      s.push_back({0, 0});
      s.push_back({1, 1});
      std::shuffle(s.begin(), s.end(), r.engine);
      same("seeds on the box edges", 1, 1, s, periodic);
    }
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
    for (std::size_t n : {4, 16, 17, 40, 300, 800}) {
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

// Sizes the full sequence would need many minutes for. The cells still tile
// the box, and a list in an order (sorted, row by row through a lattice,
// bunched in a corner) costs about what a random list costs: without the
// clip order every later seed of such a list cuts every cell.
void test_many_seeds() {
  using clock = std::chrono::steady_clock;
  rng r{std::mt19937_64{156}};
  const std::size_t n = 30000;
  auto run = [&](std::string const& what, std::vector<point> const& seeds,
                 bool periodic) {
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
    const double per_cell = static_cast<double>(vertices) / seeds.size();
    std::cout << "  " << seeds.size() << " seeds, " << what << ", "
              << (periodic ? "periodic" : "bounded") << ": " << s << " s, "
              << per_cell << " vertices per cell\n";
    REQUIRE(cells.size() == seeds.size());
    REQUIRE(std::abs(area - 2.0) < 1e-9);
    // Euler: a cell of a tessellation has six neighbours on average (a
    // little under in a bounded box, four in an exact square lattice).
    REQUIRE(per_cell > 3.9 && per_cell < 6.1);
    return s;
  };
  for (bool periodic : {false, true}) {
    const auto random = random_seeds(n, 2, 1, r);
    const double base = run("random", random, periodic);
    const double limit = 10.0 * base + 1.0;
    REQUIRE(run("sorted by x", sorted_by_x(random), periodic) < limit);
    REQUIRE(run("row-major lattice", row_major_seeds(173, 2, 1, 0.3, r),
                periodic) < limit);
    REQUIRE(run("bunched in a corner", clustered_seeds(n, 2, 1, r), periodic) <
            limit);
  }
}

// The clip order is a permutation, the same wherever it is computed.
void test_clip_order() {
  using gen = rvegen::voronoi_generator_2d<D>;
  REQUIRE(gen::clip_order(0).empty());
  REQUIRE(gen::clip_order(1) == std::vector<std::size_t>{0});
  for (std::size_t n : {2u, 17u, 1000u}) {
    auto order = gen::clip_order(n);
    REQUIRE(order == gen::clip_order(n));
    auto sorted = order;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t k = 0; k < n; ++k) REQUIRE(sorted[k] == k);
  }
  std::size_t h = 1469598103934665603ull;
  for (auto v : gen::clip_order(1000)) h = (h ^ v) * 1099511628211ull;
  std::cout << "  clip_order(1000) hash 0x" << std::hex << h << std::dec << "\n";
  REQUIRE(h == CLIP_ORDER_HASH);
}

} // namespace

int main() {
  test_identical_cells();
  test_identical_weighted_cells();
  test_many_seeds();
  test_clip_order();
  std::cout << "  " << compared << " tessellations compared with the full "
            << "sequence of clips\n";
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "voronoi_2d_neighbour_test: all checks passed\n";
  return EXIT_SUCCESS;
}
