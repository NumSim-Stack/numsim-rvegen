#pragma once

// Microstructure statistics — free functions for the standard geometric
// descriptors of an RVE:
//
//   * volume_fraction_in_domain — accepts shapes, returns φ.
//   * two_point_correlations — S2, the probability that two points a lag
//     apart are both in an inclusion: exact on a voxel grid by FFT, as a
//     radial average and along x, y, z; per phase and between phases.
//     two_point_correlation_2d / _3d give the radial average alone.
//   * radial_distribution_2d — g(r), the pair-correlation function for
//     particle centres. Useful for periodic packings.
//   * radial_distribution_3d — the 3D counterpart: g(r) with
//     spherical-shell normalisation against the number density
//     N / (Lx·Ly·Lz).
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
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../domain.h"
#include "../phase/phase.h"
#include "../postprocess/voxel_grid.h"
#include "../generators/rve_generator_base.h"
#include "../shapes/periodic_wraps.h"
#include "../shapes/shape_base.h"
#include "two_point_fft.h"
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
// Two-point statistics on the voxel grid, exact (#127).
//
// S2(l) is the probability that two points a lag l apart both lie in an
// inclusion. On an nx x ny (x nz) grid of voxel centres it is a count of
// voxel pairs, which the FFT gives for every lag at once
// (two_point_fft.h): no random sampling, and the same numbers on every
// platform. Reported are
//   * the value at lag 0, which is the volume fraction of the grid,
//   * the radial average: all lags of a length within a bin,
//   * the profiles along x, y and z: the lags along that axis only. For an
//     isotropic structure they agree with the radial average; aligned
//     fibres or a weave show up as profiles that differ.
// Bin b of `nbins` covers lag lengths in [b, b + 1) * max_r / nbins, with
// max_r half the smallest edge of the sampled box; r in the results is the
// bin centre. A bin narrower than the voxel can hold no lag: its value is
// NaN (the profiles along an axis have lags only at multiples of the voxel
// size, so they run empty first).
//
// With phases, the same for every phase (the points of the phase,
// including the matrix) and for every pair of phases (first point in one,
// second in the other; the average of both orders).
//
// Cost: the voxelization, O(nx*ny*nz * shapes), plus a few FFTs of the
// grid. A bounded box is padded by its largest lag, to about 1.5 times the
// voxels per axis; 16 bytes per padded voxel and stored phase.
// ---------------------------------------------------------------------------
struct s2_sample {
  double r;       // bin centre, in domain units
  double s2;      // probability that two points r apart are both in an inclusion
};

struct two_point_statistics {
  struct correlation {
    std::string a, b;            // phase names; a == b: the phase itself
    two_point_profile profile;
  };
  two_point_boundary boundary{two_point_boundary::periodic};
  std::array<std::size_t, 3> grid{};     // voxels per axis (nz = 1 in 2D)
  std::size_t nbins{0};
  double max_r{0.0};
  two_point_profile inclusion;           // points covered by any shape
  std::vector<correlation> phases;       // empty without phases

  [[nodiscard]] double r(std::size_t bin) const noexcept {
    return two_point_bin_centre(bin, nbins, max_r);
  }
  [[nodiscard]] std::vector<s2_sample> radial() const {
    std::vector<s2_sample> out;
    out.reserve(nbins);
    for (std::size_t b = 0; b < nbins; ++b)
      out.push_back({r(b), inclusion.radial[b]});
    return out;
  }
};

template <typename T>
[[nodiscard]] two_point_statistics two_point_correlations(
    std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
    std::type_identity_t<domain<T>> const& domain_box,
    std::size_t nx, std::size_t ny, std::size_t nz, std::size_t nbins = 32,
    two_point_boundary boundary = two_point_boundary::periodic,
    phase_collection<T> const* phases = nullptr,
    cancel_fn const* cancel = nullptr) {
  two_point_statistics out;
  out.boundary = boundary;
  out.nbins = nbins;
  out.grid = {nx, ny, effective_nz(domain_box, nz)};
  const std::array<double, 3> extent{double(domain_box[0]), double(domain_box[1]),
                                     domain_box.is_3d() ? double(domain_box[2]) : 0.0};
  // Half the smallest edge among the axes that have lags at all.
  double smallest = 0.0;
  for (int pass = 0; pass < 2 && !(smallest > 0.0); ++pass) {
    for (std::size_t a = 0; a < 3; ++a) {
      if (!(extent[a] > 0.0) || (pass == 0 && out.grid[a] < 2)) continue;
      smallest = smallest > 0.0 ? std::min(smallest, extent[a]) : extent[a];
    }
  }
  if (!(smallest > 0.0) || !std::isfinite(smallest)) {
    throw std::invalid_argument{
        "two-point statistics: the domain box has no positive finite edge"};
  }
  out.max_r = 0.5 * smallest;
  const bool wrap = boundary == two_point_boundary::periodic;
  const voxel_correlator correlator{out.grid, extent, {wrap, wrap, wrap},
                                    out.max_r, nbins};

  // One voxelization serves both questions: a voxel holds 0 where no shape
  // is, else 1 + the phase id of the shape that covers it (1 + 0 for a
  // shape without a phase).
  std::vector<phase_id> shape_to_id(shapes.size(), 1);
  std::size_t matrix_id = 0;
  if (phases != nullptr) {
    matrix_id = phases->matrix_id();
    for (std::size_t s = 0; s < shapes.size(); ++s) {
      const auto name = shapes[s]->phase_name();
      const std::size_t id = name.empty() ? std::size_t{0} : phases->id_of(name);
      if (id >= std::numeric_limits<phase_id>::max() ||
          matrix_id >= std::numeric_limits<phase_id>::max()) {
        throw std::runtime_error{
            "two-point statistics: phase id exceeds phase_id range (uint32)"};
      }
      shape_to_id[s] = static_cast<phase_id>(id + 1);
    }
  }
  const auto grid = detail::sample_voxel_grid_with_ids(
      shapes, domain_box, nx, ny, nz, shape_to_id, 0, cancel);

  {
    const auto covered = correlator.transform(
        grid, [](phase_id v) { return v != 0; }, cancel);
    out.inclusion = correlator.correlate(covered, covered, cancel);
  }
  if (phases == nullptr) return out;

  // The phases, in the order of their ids; id 0 without a phase of that id
  // is what belongs to no phase.
  std::vector<std::pair<std::size_t, std::string>> classes;
  bool zero_declared = false;
  for (auto const* p : phases->ordered()) {
    classes.emplace_back(p->id, p->name);
    zero_declared = zero_declared || p->id == 0;
  }
  std::sort(classes.begin(), classes.end());
  const auto phase_of = [matrix_id](phase_id v) {
    return v == 0 ? matrix_id : std::size_t{v} - 1;
  };
  if (!zero_declared &&
      std::any_of(grid.begin(), grid.end(),
                  [&](phase_id v) { return phase_of(v) == 0; })) {
    classes.insert(classes.begin(), {std::size_t{0}, std::string{"unassigned"}});
  }
  std::vector<voxel_correlator::spectrum> spectra;
  spectra.reserve(classes.size());
  for (auto const& c : classes) {
    spectra.push_back(correlator.transform(
        grid, [&, id = c.first](phase_id v) { return phase_of(v) == id; }, cancel));
  }
  for (std::size_t i = 0; i < classes.size(); ++i) {
    for (std::size_t j = i; j < classes.size(); ++j) {
      out.phases.push_back({classes[i].second, classes[j].second,
                            correlator.correlate(spectra[i], spectra[j], cancel)});
    }
  }
  return out;
}

// S2(r), radial average, of a 2D RVE (on a 3D domain: of its mid-plane
// cross-section) and of a 3D RVE. See two_point_correlations for the
// profiles along the axes and for phases.
template <typename T>
[[nodiscard]] std::vector<s2_sample>
two_point_correlation_2d(
    std::vector<std::unique_ptr<shape_base<T>>> const& accepted,
    std::type_identity_t<domain<T>> const& domain_box,
    std::size_t nx, std::size_t ny,
    std::size_t nbins = 32,
    two_point_boundary boundary = two_point_boundary::periodic) {
  return two_point_correlations(accepted, domain_box, nx, ny, 1, nbins, boundary)
      .radial();
}

template <typename T>
[[nodiscard]] std::vector<s2_sample>
two_point_correlation_3d(
    std::vector<std::unique_ptr<shape_base<T>>> const& accepted,
    std::type_identity_t<domain<T>> const& domain_box,
    std::size_t nx, std::size_t ny, std::size_t nz,
    std::size_t nbins = 32,
    two_point_boundary boundary = two_point_boundary::periodic) {
  return two_point_correlations(accepted, domain_box, nx, ny, nz, nbins, boundary)
      .radial();
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
