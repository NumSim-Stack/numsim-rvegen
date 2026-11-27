#pragma once

// Seed validation shared by the 2D and 3D Voronoi generators (#79).
//
// Policy:
//   * Seeds must lie in the closed domain `[0, L_k]` (a relative slack of
//     `rel_tol * scale` is tolerated). Seeds ON the boundary are valid.
//   * Coincident seeds are REJECTED with a clear `std::runtime_error`.
//     Two seeds closer than `rel_tol * scale` (scale = shortest domain
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
#include <unordered_map>
#include <vector>

namespace rvegen::detail {

inline constexpr double voronoi_rel_tol = 1e-9;

template <typename T, std::size_t D>
void validate_voronoi_seeds(std::array<T, D> const& extent,
                            std::vector<std::array<T, D>> const& seeds,
                            char const* who, bool periodic = false) {
  // Smallest extent: a thin slab must not lose resolution to its long edges.
  T scale = extent[0];
  for (auto e : extent) scale = std::min(scale, e);
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

  // Coincidence check via a hash grid with cell size `tol`: two seeds closer
  // than `tol` fall in the same or adjacent cells, so only the 3^D
  // neighbourhood is scanned. O(N) expected for any seed distribution
  // (including planar/degenerate ones that defeat a sort-and-sweep). In
  // periodic mode seeds on opposite faces (x = 0 and x = L) are the same
  // point modulo the box: cell indices wrap and the minimum-image distance
  // is used.
  using key_type = std::array<long long, D>;
  struct key_hash {
    std::size_t operator()(key_type const& k) const noexcept {
      std::size_t h = 1469598103934665603ull;
      for (auto v : k) {
        h ^= static_cast<std::size_t>(v) + 0x9e3779b97f4a7c15ull + (h << 6) +
             (h >> 2);
      }
      return h;
    }
  };
  std::array<long long, D> ncell{};
  for (std::size_t k = 0; k < D; ++k) {
    ncell[k] = std::max(1LL, static_cast<long long>(std::floor(
                                 static_cast<double>(extent[k] / tol))));
  }
  auto index_of = [&](std::array<T, D> const& p) {
    key_type key{};
    for (std::size_t k = 0; k < D; ++k) {
      T x = p[k];
      if (periodic) {
        x -= extent[k] * std::floor(x / extent[k]);   // wrap into [0, L)
      }
      long long c = static_cast<long long>(std::floor(x / tol));
      if (periodic) c = std::clamp(c, 0LL, ncell[k] - 1);
      key[k] = c;
    }
    return key;
  };

  std::unordered_map<key_type, std::vector<std::size_t>, key_hash> grid;
  grid.reserve(seeds.size() * 2);
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    const key_type key = index_of(seeds[i]);
    // Scan the 3^D neighbouring cells for earlier seeds.
    std::size_t combos = 1;
    for (std::size_t k = 0; k < D; ++k) combos *= 3;
    for (std::size_t c = 0; c < combos; ++c) {
      key_type nk = key;
      std::size_t cc = c;
      for (std::size_t k = 0; k < D; ++k) {
        nk[k] += static_cast<long long>(cc % 3) - 1;
        cc /= 3;
        if (periodic) nk[k] = ((nk[k] % ncell[k]) + ncell[k]) % ncell[k];
      }
      auto it = grid.find(nk);
      if (it == grid.end()) continue;
      for (std::size_t j : it->second) {
        T d2{0}, pd2{0};
        for (std::size_t k = 0; k < D; ++k) {
          const T d = std::abs(seeds[i][k] - seeds[j][k]);
          d2 += d * d;
          const T pd = std::min(d, extent[k] - d);
          pd2 += pd * pd;
        }
        const bool plain = d2 <= tol * tol;
        if (plain || (periodic && pd2 <= tol * tol)) {
          throw std::runtime_error{
              std::string{who} + ": seeds " + std::to_string(j) + " and " +
              std::to_string(i) +
              (plain ? " are coincident (closer than 1e-9 of the domain size)"
                     : " are coincident modulo the periodic box")};
        }
      }
    }
    grid[key].push_back(i);
  }
}

} // namespace rvegen::detail

