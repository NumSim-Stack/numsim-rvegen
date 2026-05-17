#pragma once

// Generate a clustered 2D seed-point distribution. Useful as input
// for `voronoi_generator_2d` when a polycrystal RVE should have
// non-uniform grain spacing — e.g. modelling dual-phase steels
// where one phase forms isolated colonies inside a matrix.
//
// Algorithm:
//   1. Pick `n_clusters` cluster-centre points uniformly at random
//      inside the domain rectangle.
//   2. For each cluster, sample `n_per_cluster` seed points from a
//      Gaussian around the centre with std-dev `cluster_spread`.
//   3. Reject (and resample) any point that falls outside the
//      domain — the resulting distribution stays inside the box
//      but is no longer perfectly Gaussian near the edges.
//
// Returns `n_clusters · n_per_cluster` total seeds. Order is
// cluster-by-cluster; seeds inside one cluster come together in the
// returned vector.

#include <array>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace rvegen {

template <typename T = double, typename Engine = std::mt19937>
[[nodiscard]] std::vector<std::array<T, 2>> cluster_seeds_2d(
    T Lx, T Ly,
    std::size_t n_clusters,
    std::size_t n_per_cluster,
    T cluster_spread,
    Engine& engine,
    std::size_t max_attempts_per_seed = 100) {
  if (Lx <= T{0} || Ly <= T{0}) {
    throw std::runtime_error{
        "cluster_seeds_2d: Lx and Ly must both be positive"};
  }
  if (n_clusters == 0 || n_per_cluster == 0) {
    throw std::runtime_error{
        "cluster_seeds_2d: n_clusters and n_per_cluster must both be ≥ 1"};
  }
  if (cluster_spread <= T{0}) {
    throw std::runtime_error{
        "cluster_seeds_2d: cluster_spread must be positive"};
  }

  std::uniform_real_distribution<T> ux{T{0}, Lx};
  std::uniform_real_distribution<T> uy{T{0}, Ly};
  std::normal_distribution<T> gauss{T{0}, cluster_spread};

  std::vector<std::array<T, 2>> seeds;
  seeds.reserve(n_clusters * n_per_cluster);

  for (std::size_t c = 0; c < n_clusters; ++c) {
    const T cx = ux(engine);
    const T cy = uy(engine);
    for (std::size_t k = 0; k < n_per_cluster; ++k) {
      // Sample-and-reject loop: keep drawing offsets until the
      // resulting point lies in the domain. Bail with an error if
      // we exceed max_attempts (e.g. very large spread relative to
      // domain — caller error).
      bool accepted = false;
      for (std::size_t attempt = 0; attempt < max_attempts_per_seed; ++attempt) {
        const T x = cx + gauss(engine);
        const T y = cy + gauss(engine);
        if (x >= T{0} && x <= Lx && y >= T{0} && y <= Ly) {
          seeds.push_back({x, y});
          accepted = true;
          break;
        }
      }
      if (!accepted) {
        throw std::runtime_error{
            "cluster_seeds_2d: exceeded max_attempts_per_seed; "
            "cluster_spread may be too large relative to the domain"};
      }
    }
  }
  return seeds;
}

} // namespace rvegen
