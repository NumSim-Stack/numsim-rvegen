#pragma once

// Microstructure statistics — free functions for the standard geometric
// descriptors of an RVE:
//
//   * volume_fraction_in_domain — accepts shapes, returns φ.
//   * two_point_correlation_2d — S2(r), the probability that two points
//     separated by r are both in the inclusion phase. S2(0) = φ;
//     S2(∞) = φ². Computed on a voxel grid via random-pair sampling.
//   * radial_distribution_2d — g(r), the pair-correlation function for
//     particle centres. Useful for periodic packings.
//   * two_point_correlation_3d / radial_distribution_3d — the 3D
//     counterparts: S2 over an nx × ny × nz voxel grid with random
//     directions on the unit sphere, and g(r) with spherical-shell
//     normalisation against the number density N / (Lx·Ly·Lz).
//
// All implementations work on the existing shape_base / voxel_grid types.
// The *_2d variants are planar: on a 3D domain, two_point_correlation_2d
// gives the S2 of the mid-plane cross-section and radial_distribution_2d
// uses xy-projected distances. Use the *_3d variants (or
// statistics_post_process, which dispatches on the domain dimension) for 3D.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <random>
#include <type_traits>
#include <vector>

#include "../domain.h"
#include "../distributions/portable_random.h"
#include "../postprocess/voxel_grid.h"
#include "../generators/rve_generator_base.h"
#include "../shapes/periodic_wraps.h"
#include "../shapes/shape_base.h"
#include "../util/wrap_to_box.h"

namespace rvegen {

// ---------------------------------------------------------------------------
// Volume fraction over the unit cell. Same definition as the
// volume_fraction termination and the generators' progress reports
// (current_volume_fraction: periodic copies are not counted again) —
// exposed here under the statistics name so post-processes, audits and
// tests need not know where it lives.
// ---------------------------------------------------------------------------
template <typename T>
[[nodiscard]] double volume_fraction_in_domain(
    std::vector<std::unique_ptr<shape_base<T>>> const& accepted,
    std::type_identity_t<domain<T>> const& domain_box) noexcept {
  return current_volume_fraction<T>(accepted, domain_box);
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
    std::type_identity_t<domain<T>> const& domain_box,
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
  // Portable transforms (see portable_random.h) so the sampled pairs do
  // not depend on the standard library.
  portable_uniform_int<std::size_t> ix{0, nx - 1};
  portable_uniform_int<std::size_t> iy{0, ny - 1};
  portable_uniform_real<double> theta{0.0, 2.0 * 3.14159265358979323846};

  for (std::size_t b = 0; b < nbins; ++b) {
    const double r = max_r * (double(b) + 0.5) / double(nbins);
    std::size_t both_in = 0;
    for (std::size_t s = 0; s < samples_per_bin; ++s) {
      const std::size_t i0 = ix(engine);
      const std::size_t j0 = iy(engine);
      const double a = theta(engine);
      const double x1 = (double(i0) + 0.5) * dx + r * std::cos(a);
      const double y1 = (double(j0) + 0.5) * dy + r * std::sin(a);
      const std::size_t i1 = std::min(std::size_t(wrap_to_box(x1, Lx) / dx), nx - 1);
      const std::size_t j1 = std::min(std::size_t(wrap_to_box(y1, Ly) / dy), ny - 1);
      if (grid[j0 * nx + i0] != 0 && grid[j1 * nx + i1] != 0) ++both_in;
    }
    out.push_back({r, double(both_in) / double(samples_per_bin)});
  }
  return out;
}

// ---------------------------------------------------------------------------
// Two-point correlation S2(r) on a voxelized 3D RVE.
//
// Same definition as two_point_correlation_2d, extended to three axes:
// voxelize at (nx, ny, nz), and for each radius bin draw `samples_per_bin`
// random voxel centres plus a direction uniform on the unit sphere; the
// endpoint is wrapped periodically along x, y and z. Bin b sits at
// r = max_r · (b + 0.5) / nbins with max_r = min(Lx, Ly, Lz) / 2.
//
// Cost: voxelization is O(nx·ny·nz · shapes) is_inside calls and
// nx·ny·nz · 4 bytes of memory (128³ ≈ 2.1 M voxels, 8 MiB); the pair
// sampling is O(nbins · samples_per_bin), independent of the grid. The
// voxelization dominates, so pick nz (and nx, ny) to suit the time budget.
// ---------------------------------------------------------------------------
template <typename T>
[[nodiscard]] std::vector<s2_sample>
two_point_correlation_3d(
    std::vector<std::unique_ptr<shape_base<T>>> const& accepted,
    std::type_identity_t<domain<T>> const& domain_box,
    std::size_t nx, std::size_t ny, std::size_t nz,
    std::size_t nbins = 32,
    std::size_t samples_per_bin = 4'000,
    std::uint64_t seed = 2026) {
  auto grid = sample_voxel_grid(accepted, domain_box, nx, ny, nz);
  const double Lx = double(domain_box[0]);
  const double Ly = double(domain_box[1]);
  const double Lz = double(domain_box[2]);
  const double dx = Lx / double(nx);
  const double dy = Ly / double(ny);
  const double dz = Lz / double(nz);
  const double max_r = std::min({Lx, Ly, Lz}) * 0.5;

  std::vector<s2_sample> out;
  out.reserve(nbins);

  std::mt19937_64 engine{seed};
  std::uniform_int_distribution<std::size_t> ix{0, nx - 1};
  std::uniform_int_distribution<std::size_t> iy{0, ny - 1};
  std::uniform_int_distribution<std::size_t> iz{0, nz - 1};
  std::uniform_real_distribution<double> cos_theta{-1.0, 1.0};
  std::uniform_real_distribution<double> phi{0.0, 2.0 * std::numbers::pi};

  auto at = [&](std::size_t i, std::size_t j, std::size_t k) {
    return grid[(k * ny + j) * nx + i];
  };

  for (std::size_t b = 0; b < nbins; ++b) {
    const double r = max_r * (double(b) + 0.5) / double(nbins);
    std::size_t both_in = 0;
    for (std::size_t s = 0; s < samples_per_bin; ++s) {
      const std::size_t i0 = ix(engine);
      const std::size_t j0 = iy(engine);
      const std::size_t k0 = iz(engine);
      const double ct = cos_theta(engine);
      const double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
      const double a = phi(engine);
      const double x1 = (double(i0) + 0.5) * dx + r * st * std::cos(a);
      const double y1 = (double(j0) + 0.5) * dy + r * st * std::sin(a);
      const double z1 = (double(k0) + 0.5) * dz + r * ct;
      const std::size_t i1 = std::min(std::size_t(wrap_to_box(x1, Lx) / dx), nx - 1);
      const std::size_t j1 = std::min(std::size_t(wrap_to_box(y1, Ly) / dy), ny - 1);
      const std::size_t k1 = std::min(std::size_t(wrap_to_box(z1, Lz) / dz), nz - 1);
      if (at(i0, j0, k0) != 0 && at(i1, j1, k1) != 0) ++both_in;
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
    std::type_identity_t<domain<T>> const& domain_box,
    std::size_t nbins = 32) {
  // Collect one centre per inclusion (no periodic copies).
  std::vector<std::array<double, 2>> centres;
  if (!(domain_box[0] > T{0}) || !(domain_box[1] > T{0})) return {};
  centres.reserve(accepted.size());
  for (auto const& shape : accepted) {
    if (!shape) continue;
    // One centre per inclusion, taken into the box: a primary whose
    // periodic copies hold the part inside may itself be centred outside.
    if (!is_counted_in_rve(*shape, domain_box)) continue;
    const auto c = shape->get_middle_point();
    centres.push_back({double(wrap_to_box(c[0], domain_box[0])),
                       double(wrap_to_box(c[1], domain_box[1]))});
  }
  if (nbins == 0) return {};
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
  // Expected ordered-pair count in a shell is N(N−1)/A · shell, so an
  // uncorrelated set gives g = 1 without a (N−1)/N bias.
  const double rho = double(N) / area;
  for (std::size_t b = 0; b < nbins; ++b) {
    const double r_inner = double(b) * dr;
    const double r_outer = r_inner + dr;
    const double r_centre = (r_inner + r_outer) * 0.5;
    const double shell_area = 3.14159265358979323846 *
                              (r_outer * r_outer - r_inner * r_inner);
    const double expected = rho * shell_area * double(N - 1);
    out.push_back({r_centre, expected > 0 ? double(hist[b]) / expected : 0.0});
  }
  return out;
}

// ---------------------------------------------------------------------------
// Radial distribution function g(r) in 3D: same estimator as
// radial_distribution_2d with periodic minimum-image distances along all
// three axes, spherical shells of volume 4/3·π·(r_outer³ − r_inner³) and
// number density N / (Lx·Ly·Lz). Each inclusion counts once
// (is_counted_in_rve), so periodic wrap copies are filtered out.
// max_r = min(Lx, Ly, Lz) / 2.
// ---------------------------------------------------------------------------
template <typename T>
[[nodiscard]] std::vector<rdf_sample>
radial_distribution_3d(
    std::vector<std::unique_ptr<shape_base<T>>> const& accepted,
    std::type_identity_t<domain<T>> const& domain_box,
    std::size_t nbins = 32) {
  std::vector<std::array<double, 3>> centres;
  centres.reserve(accepted.size());
  for (auto const& shape : accepted) {
    if (!shape) continue;
    if (!domain_box.is_3d()) break;   // not a 3D box: no centres
    if (!is_counted_in_rve(*shape, domain_box)) continue;
    const auto c = shape->get_middle_point();
    centres.push_back({double(wrap_to_box(c[0], domain_box[0])),
                       double(wrap_to_box(c[1], domain_box[1])),
                       double(wrap_to_box(c[2], domain_box[2]))});
  }
  if (nbins == 0) return {};
  const std::size_t N = centres.size();
  const double Lx = double(domain_box[0]);
  const double Ly = double(domain_box[1]);
  const double Lz = double(domain_box[2]);
  const double volume = Lx * Ly * Lz;
  const double max_r = std::min({Lx, Ly, Lz}) * 0.5;
  const double dr = max_r / double(nbins);

  // Periodic minimum-image distance.
  auto pdist = [&](auto const& a, auto const& b) {
    double dx = std::abs(a[0] - b[0]);
    double dy = std::abs(a[1] - b[1]);
    double dz = std::abs(a[2] - b[2]);
    if (dx > Lx * 0.5) dx = Lx - dx;
    if (dy > Ly * 0.5) dy = Ly - dy;
    if (dz > Lz * 0.5) dz = Lz - dz;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
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
  // Expected ordered-pair count in a shell is N(N−1)/V · shell, so an
  // uncorrelated set gives g = 1 without a (N−1)/N bias.
  const double rho = double(N) / volume;
  for (std::size_t b = 0; b < nbins; ++b) {
    const double r_inner = double(b) * dr;
    const double r_outer = r_inner + dr;
    const double r_centre = (r_inner + r_outer) * 0.5;
    const double shell_volume = 4.0 / 3.0 * std::numbers::pi *
        (r_outer * r_outer * r_outer - r_inner * r_inner * r_inner);
    const double expected = rho * shell_volume * double(N - 1);
    out.push_back({r_centre, expected > 0 ? double(hist[b]) / expected : 0.0});
  }
  return out;
}

} // namespace rvegen
