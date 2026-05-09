#pragma once

// Microstructure statistics — free functions for the standard quantitative
// descriptors that homogenization papers report alongside their RVEs:
//
//   * volume_fraction_in_domain — accepts shapes, returns φ.
//   * two_point_correlation_2d — S2(r), the probability that two points
//     separated by r are both in the inclusion phase. S2(0) = φ;
//     S2(∞) = φ². Computed on a voxel grid via random-pair sampling.
//   * radial_distribution_2d — g(r), the pair-correlation function for
//     particle centres. Useful for periodic packings.
//
// All implementations work on the existing shape_base / voxel_grid types.
// 2D for now; 3D follows the same pattern (sample over a 3D grid /
// 3-component shifts) — extend when needed.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "../postprocess/voxel_grid.h"
#include "../shapes/shape_base.h"

namespace rvegen {

// ---------------------------------------------------------------------------
// Volume fraction over the unit cell, filtering wrap copies via the
// centre-in-domain test. Same definition as the volume_fraction termination
// uses internally — exposed here as a public helper so post-processes,
// audits, and tests can compute it without duplicating the logic.
// ---------------------------------------------------------------------------
template <typename T>
[[nodiscard]] double volume_fraction_in_domain(
    std::vector<std::unique_ptr<shape_base<T>>> const& accepted,
    std::array<T, 3> const& domain_box) noexcept {
  const bool is_3d = domain_box[2] > T{0};
  const double dom_extent = is_3d
      ? double(domain_box[0]) * double(domain_box[1]) * double(domain_box[2])
      : double(domain_box[0]) * double(domain_box[1]);
  if (dom_extent <= 0.0) return 0.0;

  double sum = 0.0;
  for (auto const& shape : accepted) {
    if (!shape) continue;
    const auto centre = shape->get_middle_point();
    const bool in_domain =
        centre[0] >= T{0} && centre[0] < domain_box[0] &&
        centre[1] >= T{0} && centre[1] < domain_box[1] &&
        (!is_3d || (centre[2] >= T{0} && centre[2] < domain_box[2]));
    if (!in_domain) continue;
    const auto v = shape->volume();
    sum += double((v > T{0}) ? v : shape->area());
  }
  return sum / dom_extent;
}

// ---------------------------------------------------------------------------
// Two-point correlation S2(r) on a voxelized 2D RVE.
//
// Discretization: voxelize at (nx, ny), then for each radius bin sample
// `samples_per_bin` random voxel pairs at that separation and count how
// many have both endpoints in the inclusion phase. Periodic wrap via
// modulo on the indices.
//
// Returns: vector of (r, S2(r)) pairs, r ∈ [0, max_r] in `nbins` steps.
// max_r defaults to half the smaller domain edge — beyond that, periodic
// images dominate and S2 is no longer informative.
// ---------------------------------------------------------------------------
struct s2_sample {
  double r;       // radius in domain units
  double s2;      // probability that two points separated by r are both in inclusion
};

template <typename T>
[[nodiscard]] std::vector<s2_sample>
two_point_correlation_2d(
    std::vector<std::unique_ptr<shape_base<T>>> const& accepted,
    std::array<T, 3> const& domain_box,
    std::size_t nx, std::size_t ny,
    std::size_t nbins = 32,
    std::size_t samples_per_bin = 4'000,
    std::uint64_t seed = 2026) {
  auto grid = sample_voxel_grid(accepted, domain_box, nx, ny, 1);
  const double Lx = double(domain_box[0]);
  const double Ly = double(domain_box[1]);
  const double dx = Lx / double(nx);
  const double dy = Ly / double(ny);
  const double max_r = std::min(Lx, Ly) * 0.5;

  std::vector<s2_sample> out;
  out.reserve(nbins);

  std::mt19937_64 engine{seed};
  std::uniform_int_distribution<std::size_t> ix{0, nx - 1};
  std::uniform_int_distribution<std::size_t> iy{0, ny - 1};
  std::uniform_real_distribution<double> theta{0.0, 2.0 * 3.14159265358979323846};

  for (std::size_t b = 0; b < nbins; ++b) {
    const double r = max_r * (double(b) + 0.5) / double(nbins);
    std::size_t both_in = 0;
    for (std::size_t s = 0; s < samples_per_bin; ++s) {
      const std::size_t i0 = ix(engine);
      const std::size_t j0 = iy(engine);
      const double a = theta(engine);
      const double x1 = (double(i0) + 0.5) * dx + r * std::cos(a);
      const double y1 = (double(j0) + 0.5) * dy + r * std::sin(a);
      // Periodic wrap into [0, L*).
      auto wrap = [](double v, double L) {
        v = std::fmod(v, L);
        return v < 0 ? v + L : v;
      };
      const std::size_t i1 = std::min(std::size_t(wrap(x1, Lx) / dx), nx - 1);
      const std::size_t j1 = std::min(std::size_t(wrap(y1, Ly) / dy), ny - 1);
      if (grid[j0 * nx + i0] != 0 && grid[j1 * nx + i1] != 0) ++both_in;
    }
    out.push_back({r, double(both_in) / double(samples_per_bin)});
  }
  return out;
}

// ---------------------------------------------------------------------------
// Radial distribution function g(r) for shape *centres* (not voxel pairs).
//
// Standard particle-physics definition: g(r) = (number of pairs with
// centres in [r, r+dr]) / (expected number under uniform random density).
// For an ergodic, well-mixed packing g(r) → 1 at large r; the small-r
// region encodes excluded-volume effects (g(r) = 0 below the closest
// approach, peak just above).
// ---------------------------------------------------------------------------
struct rdf_sample {
  double r;       // bin centre
  double g;       // pair density / expected density
};

template <typename T>
[[nodiscard]] std::vector<rdf_sample>
radial_distribution_2d(
    std::vector<std::unique_ptr<shape_base<T>>> const& accepted,
    std::array<T, 3> const& domain_box,
    std::size_t nbins = 32) {
  // Collect in-domain centres (filter wrap copies).
  std::vector<std::array<double, 2>> centres;
  centres.reserve(accepted.size());
  for (auto const& shape : accepted) {
    if (!shape) continue;
    const auto c = shape->get_middle_point();
    if (c[0] >= T{0} && c[0] < domain_box[0] &&
        c[1] >= T{0} && c[1] < domain_box[1]) {
      centres.push_back({double(c[0]), double(c[1])});
    }
  }
  const std::size_t N = centres.size();
  const double Lx = double(domain_box[0]);
  const double Ly = double(domain_box[1]);
  const double area = Lx * Ly;
  const double max_r = std::min(Lx, Ly) * 0.5;
  const double dr = max_r / double(nbins);

  // Periodic minimum-image distance.
  auto pdist = [&](auto const& a, auto const& b) {
    double dx = std::abs(a[0] - b[0]);
    double dy = std::abs(a[1] - b[1]);
    if (dx > Lx * 0.5) dx = Lx - dx;
    if (dy > Ly * 0.5) dy = Ly - dy;
    return std::sqrt(dx * dx + dy * dy);
  };

  std::vector<std::size_t> hist(nbins, 0);
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = i + 1; j < N; ++j) {
      const double r = pdist(centres[i], centres[j]);
      if (r >= max_r) continue;
      const std::size_t b = std::min(std::size_t(r / dr), nbins - 1);
      hist[b] += 2;  // each pair contributes to both i→j and j→i
    }
  }

  std::vector<rdf_sample> out;
  out.reserve(nbins);
  if (N < 2) return out;
  const double rho = double(N) / area;
  for (std::size_t b = 0; b < nbins; ++b) {
    const double r_inner = double(b) * dr;
    const double r_outer = r_inner + dr;
    const double r_centre = (r_inner + r_outer) * 0.5;
    const double shell_area = 3.14159265358979323846 *
                              (r_outer * r_outer - r_inner * r_inner);
    const double expected = rho * shell_area * double(N);
    out.push_back({r_centre, expected > 0 ? double(hist[b]) / expected : 0.0});
  }
  return out;
}

} // namespace rvegen
