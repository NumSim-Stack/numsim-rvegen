#pragma once

// Seed validation shared by the 2D and 3D Voronoi generators (#79).
//
// Policy:
//   * Seeds must lie in the closed domain `[0, L_k]` (a relative slack of
//     `rel_tol * scale` is tolerated). Seeds ON the boundary are valid.
//   * Coincident seeds are REJECTED with a clear `std::runtime_error`.
//     Two seeds closer than `rel_tol * scale` (scale = longest domain
//     edge, `rel_tol = 1e-9`) would produce duplicate or zero-volume
//     cells, and merging silently would break the "cell i <-> seed i"
//     index contract; the caller decides how to dedupe.
// All tolerances are relative to the domain size, so the check behaves
// identically for a 1e-6 m or a 1e3 m domain.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace rvegen::detail {

inline constexpr double voronoi_rel_tol = 1e-9;

template <typename T, std::size_t D>
void validate_voronoi_seeds(std::array<T, D> const& extent,
                            std::vector<std::array<T, D>> const& seeds,
                            char const* who) {
  T scale{0};
  for (auto e : extent) scale = std::max(scale, e);
  const T tol = static_cast<T>(voronoi_rel_tol) * scale;

  for (std::size_t i = 0; i < seeds.size(); ++i) {
    for (std::size_t k = 0; k < D; ++k) {
      const T x = seeds[i][k];
      if (!std::isfinite(x) || x < -tol || x > extent[k] + tol) {
        throw std::runtime_error{std::string{who} + ": seed " +
                                 std::to_string(i) +
                                 " lies outside the domain"};
      }
    }
  }

  // Sweep along axis 0: only neighbours within `tol` in x can coincide.
  std::vector<std::size_t> order(seeds.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return seeds[a][0] < seeds[b][0];
  });
  for (std::size_t p = 0; p < order.size(); ++p) {
    for (std::size_t q = p + 1; q < order.size(); ++q) {
      const auto& a = seeds[order[p]];
      const auto& b = seeds[order[q]];
      if (b[0] - a[0] > tol) break;
      T d2{0};
      for (std::size_t k = 0; k < D; ++k) d2 += (a[k] - b[k]) * (a[k] - b[k]);
      if (d2 <= tol * tol) {
        throw std::runtime_error{
            std::string{who} + ": seeds " +
            std::to_string(std::min(order[p], order[q])) + " and " +
            std::to_string(std::max(order[p], order[q])) +
            " are coincident (closer than 1e-9 of the domain size)"};
      }
    }
  }
}

} // namespace rvegen::detail
