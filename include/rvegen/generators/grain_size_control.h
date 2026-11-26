#pragma once

// Grain-size control for polycrystal RVEs (#8): seeds + Laguerre weights
// that make a 3D tessellation reproduce a target lognormal distribution of
// equivalent grain diameters.
//
// Why a capacity-constrained power (Laguerre) diagram, and not "random
// sequential addition of lognormal spheres, then plain Voronoi":
//   * With N grains in a box of volume V the cell volumes must sum to V, so
//     a plain Voronoi of ANY seed set has mean cell volume V/N. RSA
//     spheres only shape the seed positions; the resulting cell-size
//     spread is whatever the geometry gives, not the requested one (and
//     RSA stalls near ~38 % packing, which limits polydispersity).
//   * A power diagram has one free parameter per cell (the weight w_i,
//     length^2) that moves faces without moving seeds, and cell volume is
//     monotone in it. Iterating w_i until every cell volume equals its
//     target is the standard capacity-constrained power diagram
//     construction; it hits the requested volumes (hence the distribution)
//     to a chosen tolerance instead of statistically.
//
// Inputs: mean equivalent diameter and the log-standard-deviation sigma of
// ln(D). The grain count follows from the volume constraint:
//   mu = ln(mean) - sigma^2/2,  E[V_grain] = pi/6 * exp(3 mu + 4.5 sigma^2),
//   N = round(V_box / E[V_grain]).
// Target diameters are drawn from LogNormal(mu, sigma) and their volumes
// rescaled by a common factor (~1) so they sum to V_box exactly. The
// equivalent diameter of a cell is D = (6 V / pi)^(1/3).
//
// Usage: pass `seeds` and `weights` to voronoi_generator_3d (periodic by
// default here, which removes boundary truncation and makes the volumes
// exactly controllable). Use backend voro_pp for N beyond a few dozen.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <vector>

#include "voronoi_generator_3d.h"

namespace rvegen {

template <typename T = double>
struct grain_size_result {
  std::vector<std::array<T, 3>> seeds;
  std::vector<T> weights;           // Laguerre weights [length^2]
  std::vector<T> target_diameters;  // requested equivalent diameters
  std::vector<T> diameters;         // achieved equivalent diameters
  T max_rel_volume_error{};         // max_i |V_i - Vt_i| / Vt_i
  std::size_t iterations{};
};

// Equivalent sphere diameter of volume v.
template <typename T>
[[nodiscard]] inline T equivalent_diameter(T v) {
  return std::cbrt(T{6} * v / std::numbers::pi_v<T>);
}

template <typename T = double, typename Engine>
[[nodiscard]] grain_size_result<T> lognormal_grain_seeds_3d(
    T Lx, T Ly, T Lz, T mean_diameter, T sigma_ln, Engine& engine,
    voronoi_backend backend = voronoi_backend::in_house, bool periodic = true,
    T tolerance = T{0.01}, std::size_t max_iterations = 200) {
  if (mean_diameter <= T{0} || sigma_ln < T{0}) {
    throw std::runtime_error{
        "lognormal_grain_seeds_3d: mean_diameter > 0 and sigma_ln >= 0 required"};
  }
  constexpr T pi = std::numbers::pi_v<T>;
  const T V = Lx * Ly * Lz;
  const T mu = std::log(mean_diameter) - sigma_ln * sigma_ln / T{2};
  const T mean_v = pi / T{6} * std::exp(T{3} * mu + T{4.5} * sigma_ln * sigma_ln);
  const auto n = static_cast<std::size_t>(std::max(T{2}, std::round(V / mean_v)));

  grain_size_result<T> res;
  std::lognormal_distribution<T> dist{mu, sigma_ln};
  std::uniform_real_distribution<T> u{T{0}, T{1}};
  std::vector<T> vt(n);
  T vsum{0};
  for (auto& v : vt) {
    const T d = dist(engine);
    v = pi / T{6} * d * d * d;
    vsum += v;
  }
  for (auto& v : vt) v *= V / vsum;
  for (auto v : vt) res.target_diameters.push_back(equivalent_diameter(v));

  for (std::size_t i = 0; i < n; ++i) {
    res.seeds.push_back({u(engine) * Lx, u(engine) * Ly, u(engine) * Lz});
  }
  res.weights.resize(n);
  for (std::size_t i = 0; i < n; ++i) {   // start from w = r^2
    const T r = res.target_diameters[i] / T{2};
    res.weights[i] = r * r;
  }

  std::vector<T> vol(n);
  for (std::size_t it = 0; it < max_iterations; ++it) {
    voronoi_generator_3d<T> gen{Lx, Ly, Lz, res.seeds, backend, periodic,
                                res.weights};
    auto cells = gen.build();
    T err{0};
    for (std::size_t i = 0; i < n; ++i) {
      vol[i] = cells[i]->volume();
      err = std::max(err, std::abs(vol[i] - vt[i]) / vt[i]);
    }
    res.max_rel_volume_error = err;
    res.iterations = it;
    if (err <= tolerance) break;
    // Volume responds to a weight change roughly as dV = 3 s dw, s = cell size.
    for (std::size_t i = 0; i < n; ++i) {
      const T s = std::cbrt(vt[i]);
      res.weights[i] += T{0.7} * (vt[i] - vol[i]) / (T{3} * s);
    }
    // Lloyd relaxation (power-diagram flavour): early on, pull each seed
    // toward its cell's volume centroid. Random seeds can leave a cell
    // pinned between neighbours so that no weight can reach its target;
    // relaxing the seeds removes those traps. Stopped in the second half so
    // the weights can settle on fixed seeds.
    if (it < max_iterations / 2) {
      for (std::size_t i = 0; i < n; ++i) {
        if (vol[i] <= T{0}) continue;
        const auto c = cells[i]->get_middle_point();
        const std::array<T, 3> L{Lx, Ly, Lz};
        for (int k = 0; k < 3; ++k) {
          T x = res.seeds[i][k] + T{0.5} * (c[k] - res.seeds[i][k]);
          if (periodic) x -= L[k] * std::floor(x / L[k]);
          res.seeds[i][k] = std::clamp(x, T{0}, L[k]);
        }
      }
    }
  }
  res.diameters.resize(n);
  for (std::size_t i = 0; i < n; ++i) res.diameters[i] = equivalent_diameter(vol[i]);
  return res;
}

} // namespace rvegen
