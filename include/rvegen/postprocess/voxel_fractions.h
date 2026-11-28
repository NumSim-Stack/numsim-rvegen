#pragma once

// Partial-volume voxels and resolution guidance for voxel grids (#115).
//
//   * sample_voxel_fractions — instead of one sample at the voxel centre,
//     k sample points per axis in every voxel; the result is, per voxel,
//     the id with the largest share and, for voxels that hold more than
//     one id, the share of each (composite voxels).
//   * check_voxel_resolution — how many voxels the thinnest shape is
//     across, and the grid that would give it a wanted number.
//   * report_voxel_sampling — the two lines the voxel writers print: the
//     volume fraction the grid ended up with next to the one of the
//     shapes, and the resolution warning.
//
// The ids are those of sample_voxel_grid (voxel_grid.h): phase ids with a
// phase_collection, 1-based inclusion indices without.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <numsim-core/input_parameter_controller.h>

#include "../domain.h"
#include "../generators/cancel_poll.h"
#include "../phase/phase.h"
#include "../shapes/periodic_wraps.h"
#include "../shapes/shape_base.h"
#include "../types.h"
#include "voxel_grid.h"

namespace rvegen {

// Largest number of sample points per voxel and axis: 16^3 = 4096 points
// per voxel is far beyond what a volume fraction needs, and the cost of
// sampling grows with the number of points.
inline constexpr std::size_t max_voxel_subsamples = 16;
// Sample points the sampler holds at a time (4 bytes each): it works
// through a layer of voxels in bands of rows that fit this, one row of
// voxels at the least, and refuses a grid whose single row is beyond
// max_voxel_row_samples.
inline constexpr std::size_t voxel_sample_buffer_entries = std::size_t{1} << 24;
inline constexpr std::uint64_t max_voxel_row_samples = std::uint64_t{1} << 28;

// The part of one voxel that belongs to one id: `samples` of the voxel's
// sample points.
struct voxel_share {
  phase_id id{0};
  std::uint32_t samples{0};

  friend bool operator==(voxel_share const&, voxel_share const&) = default;
};

// A voxel grid with partial volumes. `dominant` has one value per voxel in
// the order of sample_voxel_grid ((k * ny + j) * nx + i). Only voxels that
// hold more than one id are listed in `mixed` (ascending); the shares of
// mixed[m] are shares[first[m]] .. shares[first[m + 1] - 1], ascending by
// id, and their samples add up to samples_per_voxel. Every other voxel
// belongs to its dominant id entirely.
struct voxel_fractions {
  std::size_t nx{0}, ny{0}, nz{0};      // voxels; nz is the effective count
  std::size_t subsamples{1};            // sample points per voxel and axis
  std::size_t samples_per_voxel{1};     // subsamples^2 (2D) or ^3 (3D)
  std::vector<phase_id> dominant;       // largest share; smallest id on a tie
  std::vector<std::size_t> mixed;
  std::vector<std::size_t> first;       // mixed.size() + 1 entries
  std::vector<voxel_share> shares;
  // Sample points per id over the whole grid (index = id).
  std::vector<std::uint64_t> samples_of_id;

  [[nodiscard]] std::uint64_t total_samples() const noexcept {
    return std::uint64_t{dominant.size()} * samples_per_voxel;
  }
  // Volume fraction of one id on the grid.
  [[nodiscard]] double fraction_of(phase_id id) const noexcept {
    if (id >= samples_of_id.size() || dominant.empty()) return 0.0;
    return static_cast<double>(samples_of_id[id]) /
           static_cast<double>(total_samples());
  }
};

namespace detail {

// Number of ids a table indexed by id needs: one more than the largest.
[[nodiscard]] inline std::size_t
id_table_size(std::vector<phase_id> const& shape_to_id, phase_id fill) {
  phase_id largest = fill;
  for (const phase_id id : shape_to_id) largest = std::max(largest, id);
  return static_cast<std::size_t>(largest) + 1;
}

// The sampling loop behind both sample_voxel_fractions overloads. One layer
// of voxels at a time is rasterised on the fine lattice (the voxels'
// sample points) exactly as sample_voxel_grid_with_ids rasterises the
// whole grid, then reduced to shares, so the memory beyond the result is
// one fine layer.
// Sample points of one row of voxels (along x): what the sampler holds at
// the least. Throws when that alone is beyond `max_voxel_row_samples`.
[[nodiscard]] inline std::size_t
checked_voxel_row_samples(std::size_t nx, std::size_t subsamples,
                          std::size_t subsamples_z) {
  const std::uint64_t per_voxel =
      std::uint64_t{subsamples} * subsamples * subsamples_z;   // <= 16^3
  if (std::uint64_t{nx} > max_voxel_row_samples / per_voxel) {
    throw std::invalid_argument{
        "voxel grid: nx * subsamples^" +
        std::string{subsamples_z > 1 ? "3" : "2"} +
        " sample points in a row of voxels exceed the 2^28 limit (nx=" +
        std::to_string(nx) + ", subsamples=" + std::to_string(subsamples) + ")"};
  }
  return static_cast<std::size_t>(nx * per_voxel);
}

template <typename T>
[[nodiscard]] voxel_fractions sample_voxel_fractions_with_ids(
    std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
    std::type_identity_t<domain<T>> const& domain_box,
    std::size_t nx, std::size_t ny, std::size_t nz, std::size_t subsamples,
    std::vector<phase_id> const& shape_to_id, phase_id fill,
    cancel_fn const* cancel,
    std::size_t buffer_entries = voxel_sample_buffer_entries) {
  if (subsamples == 0 || subsamples > max_voxel_subsamples) {
    throw std::invalid_argument{
        "voxel grid: subsamples must be between 1 and " +
        std::to_string(max_voxel_subsamples) + ", got " +
        std::to_string(subsamples)};
  }
  const bool is_3d = domain_box.is_3d();
  const std::size_t nz_eff = is_3d ? nz : 1;
  const std::size_t voxels = checked_voxel_count(nx, ny, nz_eff);
  const std::size_t k = subsamples;
  const std::size_t kz = is_3d ? k : 1;
  const std::size_t fx = nx * k, fy = ny * k, fz = nz_eff * kz;

  voxel_fractions out;
  out.nx = nx;
  out.ny = ny;
  out.nz = nz_eff;
  out.subsamples = k;
  out.samples_per_voxel = k * k * kz;
  out.dominant.assign(voxels, fill);
  out.samples_of_id.assign(id_table_size(shape_to_id, fill), 0);

  const auto dx = domain_box[0] / static_cast<T>(fx);
  const auto dy = domain_box[1] / static_cast<T>(fy);
  const auto dz = is_3d ? domain_box[2] / static_cast<T>(fz) : T{0};

  // Index ranges of every shape on the fine lattice (as in
  // sample_voxel_grid_with_ids: the bounding box padded by one cell, the
  // whole lattice for a shape without a usable box).
  auto range = [](T lo, T hi, T d, std::size_t n) {
    const double flo = std::floor(double(lo / d)) - 1.0;
    const double fhi = std::ceil(double(hi / d)) + 1.0;
    const std::size_t i0 =
        flo <= 0.0 ? 0 : (flo >= double(n) ? n : std::size_t(flo));
    const std::size_t i1 =
        fhi <= 0.0 ? 0 : (fhi >= double(n) ? n : std::size_t(fhi));
    return std::pair<std::size_t, std::size_t>{i0, i1};
  };
  struct span {
    std::size_t i0, i1, j0, j1, k0, k1;
  };
  std::vector<span> spans(shapes.size());
  // Shapes per layer of voxels, highest index first: painted in that
  // order the lowest-index shape wins on overlap (first match).
  std::vector<std::vector<std::size_t>> in_layer(nz_eff);
  for (std::size_t si = shapes.size(); si-- > 0;) {
    auto const& shape = shapes[si];
    span s{0, fx, 0, fy, 0, fz};
    if (auto const* bb = shape->bounding_box()) {
      auto const& lo = bb->bottom_point();
      auto const& hi = bb->top_point();
      const auto mp = shape->get_middle_point();
      bool usable = true;
      for (int a = 0; a < 3; ++a) {
        usable = usable && std::isfinite(double(lo[a])) &&
                 std::isfinite(double(hi[a])) && lo[a] <= mp[a] &&
                 mp[a] <= hi[a];
      }
      if (usable) {
        std::tie(s.i0, s.i1) = range(lo[0], hi[0], dx, fx);
        std::tie(s.j0, s.j1) = range(lo[1], hi[1], dy, fy);
        if (is_3d) std::tie(s.k0, s.k1) = range(lo[2], hi[2], dz, fz);
      }
    }
    spans[si] = s;
    if (s.i0 >= s.i1 || s.j0 >= s.j1 || s.k0 >= s.k1) continue;
    for (std::size_t layer = s.k0 / kz; layer <= (s.k1 - 1) / kz; ++layer)
      in_layer[layer].push_back(si);
  }

  // The sample points are held for a band of voxel rows of one layer at a
  // time: as many rows as fit `buffer_entries`, at least one.
  const std::size_t layer_voxels = nx * ny;
  const std::size_t row_entries = checked_voxel_row_samples(nx, k, kz);
  const std::size_t band_rows =
      std::min(ny, std::max<std::size_t>(1, buffer_entries / row_entries));
  const std::size_t by = band_rows * k;   // rows of sample points in a band
  std::vector<phase_id> fine(band_rows * row_entries);
  std::vector<phase_id> cell(out.samples_per_voxel);
  for (std::size_t layer = 0; layer < nz_eff; ++layer) {
    if (in_layer[layer].empty()) {   // nothing but `fill`, as initialised
      poll_cancel(cancel);
      out.samples_of_id[fill] +=
          std::uint64_t{layer_voxels} * out.samples_per_voxel;
      continue;
    }
    const std::size_t z_begin = layer * kz, z_end = z_begin + kz;
    for (std::size_t j_begin = 0; j_begin < ny; j_begin += band_rows) {
    const std::size_t j_end = std::min(ny, j_begin + band_rows);
    const std::size_t y_begin = j_begin * k, y_end = j_end * k;
    bool painted = false;
    for (const std::size_t si : in_layer[layer]) {
      auto const& shape = shapes[si];
      auto const& s = spans[si];
      const std::size_t j0 = std::max(s.j0, y_begin);
      const std::size_t j1 = std::min(s.j1, y_end);
      if (j0 >= j1) continue;
      if (!painted) {
        std::fill(fine.begin(), fine.end(), fill);
        painted = true;
      }
      const phase_id id = shape_to_id[si];
      for (std::size_t kk = std::max(s.k0, z_begin);
           kk < std::min(s.k1, z_end); ++kk) {
        const auto z = is_3d ? (static_cast<T>(kk) + T{0.5}) * dz : T{0};
        for (std::size_t jj = j0; jj < j1; ++jj) {
          poll_cancel(cancel);   // once per row of sample points
          const auto y = (static_cast<T>(jj) + T{0.5}) * dy;
          phase_id* const row =
              fine.data() + ((kk - z_begin) * by + (jj - y_begin)) * fx;
          for (std::size_t ii = s.i0; ii < s.i1; ++ii) {
            const auto x = (static_cast<T>(ii) + T{0.5}) * dx;
            if (shape->is_inside(std::array<T, 3>{x, y, z})) row[ii] = id;
          }
        }
      }
    }
    if (!painted) {   // no shape reaches this band
      poll_cancel(cancel);
      out.samples_of_id[fill] +=
          std::uint64_t{(j_end - j_begin) * nx} * out.samples_per_voxel;
      continue;
    }
    // Reduce the band to one entry per voxel.
    for (std::size_t j = j_begin; j < j_end; ++j) {
      poll_cancel(cancel);
      for (std::size_t i = 0; i < nx; ++i) {
        std::size_t n = 0;
        bool pure = true;
        for (std::size_t c = 0; c < kz; ++c) {
          for (std::size_t b = 0; b < k; ++b) {
            phase_id const* const row =
                fine.data() + (c * by + (j - j_begin) * k + b) * fx + i * k;
            for (std::size_t a = 0; a < k; ++a) {
              cell[n] = row[a];
              pure = pure && cell[n] == cell[0];
              ++n;
            }
          }
        }
        const std::size_t voxel = layer * layer_voxels + j * nx + i;
        if (pure) {
          out.dominant[voxel] = cell[0];
          out.samples_of_id[cell[0]] += out.samples_per_voxel;
          continue;
        }
        std::sort(cell.begin(), cell.end());
        out.mixed.push_back(voxel);
        out.first.push_back(out.shares.size());
        std::uint32_t best = 0;
        for (std::size_t a = 0; a < n;) {
          std::size_t e = a;
          while (e < n && cell[e] == cell[a]) ++e;
          const auto count = static_cast<std::uint32_t>(e - a);
          out.shares.push_back({cell[a], count});
          out.samples_of_id[cell[a]] += count;
          if (count > best) {   // ascending ids: the smallest wins a tie
            best = count;
            out.dominant[voxel] = cell[a];
          }
          a = e;
        }
      }
    }
    }   // band
  }
  out.first.push_back(out.shares.size());
  return out;
}

// "0.4012": a volume fraction or a length in a message.
[[nodiscard]] inline std::string short_number(double value, int digits = 4) {
  char buffer[40];
  std::snprintf(buffer, sizeof buffer, "%.*g", digits, value);
  return buffer;
}

} // namespace detail

// Partial-volume voxel grid without phases: the ids are the inclusion
// indices of sample_voxel_grid (0 = no shape). `subsamples` sample points
// per axis lie at the centres of the voxel's subsamples x subsamples
// (x subsamples) equal parts; with 1 the dominant ids are exactly
// sample_voxel_grid's and no voxel is mixed. A 2D domain is sampled in
// the plane only; a 3D domain also along z, so a 3D grid with nz = 1
// averages over the whole thickness.
template <typename T>
[[nodiscard]] voxel_fractions sample_voxel_fractions(
    std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
    std::type_identity_t<domain<T>> const& domain_box,
    std::size_t nx, std::size_t ny, std::size_t nz, std::size_t subsamples,
    cancel_fn const* cancel = nullptr) {
  return detail::sample_voxel_fractions_with_ids(
      shapes, domain_box, nx, ny, nz, subsamples,
      detail::inclusion_index_of_shapes(shapes), 0, cancel);
}

// Partial-volume voxel grid with phases: the ids are phase ids, as in the
// phase-aware sample_voxel_grid (same rules for the matrix phase, shapes
// without a phase name and `strict`).
template <typename T>
[[nodiscard]] voxel_fractions sample_voxel_fractions(
    std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
    std::type_identity_t<domain<T>> const& domain_box,
    std::size_t nx, std::size_t ny, std::size_t nz, std::size_t subsamples,
    phase_collection<T> const& phases, bool strict = false,
    cancel_fn const* cancel = nullptr) {
  return detail::sample_voxel_fractions_with_ids(
      shapes, domain_box, nx, ny, nz, subsamples,
      detail::phase_id_of_shapes(shapes, phases, strict),
      detail::matrix_phase_id(phases), cancel);
}

// Sample points per id of a plain grid (one sample per voxel).
[[nodiscard]] inline std::vector<std::uint64_t>
count_voxel_ids(std::vector<phase_id> const& grid) {
  std::vector<std::uint64_t> counts;
  for (const phase_id id : grid) {
    if (id >= counts.size()) counts.resize(std::size_t{id} + 1, 0);
    ++counts[id];
  }
  return counts;
}

// The volume fraction of every id from the sizes of the shapes, for
// comparison with the one a grid ended up with: the measure of each shape
// that counts as material of the RVE (is_counted_in_rve: a periodic
// inclusion once) goes to the shape's id, the rest of the domain to `fill`.
// Where shapes overlap or leave the box the sum is not the covered part,
// and the two numbers differ for that reason too; the fractions add up to
// more than 1 when the sizes exceed the box.
template <typename T>
[[nodiscard]] std::vector<double> shape_size_fractions(
    std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
    std::type_identity_t<domain<T>> const& domain_box,
    std::vector<phase_id> const& shape_to_id, phase_id fill) {
  std::vector<double> fraction(detail::id_table_size(shape_to_id, fill), 0.0);
  const double whole = static_cast<double>(domain_box.measure());
  double covered = 0.0;
  for (std::size_t s = 0; s < shapes.size(); ++s) {
    if (!is_counted_in_rve(*shapes[s], domain_box)) continue;
    const double part = static_cast<double>(shapes[s]->measure()) / whole;
    fraction[shape_to_id[s]] += part;
    covered += part;
  }
  // Shapes that overlap or leave the box add up to more than the box:
  // nothing is left for `fill` then, and the fractions sum to more than 1.
  fraction[fill] += std::max(0.0, 1.0 - covered);
  return fraction;
}

// How well a grid resolves the shapes: voxels across the thinnest one.
struct voxel_resolution {
  std::size_t shapes{0};          // shapes looked at (periodic copies not)
  std::size_t under{0};           // of them, fewer than `wanted` across
  double smallest_width{0.0};     // shape_base::min_width of the thinnest
  double voxel{0.0};              // largest voxel edge of the sampled axes
  double wanted{0.0};
  // The counts that give the thinnest shape `wanted` voxels (never below
  // the given ones; an axis that is not sampled keeps its count).
  std::array<std::size_t, 3> suggested{0, 0, 0};
  // false: the suggested grid has more than the 2^32 voxels a grid may have.
  bool suggestion_fits{true};
  bool sampled_z{false};

  [[nodiscard]] double voxels_across() const noexcept {
    return voxel > 0.0 ? smallest_width / voxel : 0.0;
  }
  [[nodiscard]] bool resolved() const noexcept { return under == 0; }

  // "12 of 15 shapes are less than 8 voxels across: ..." (empty when
  // resolved()).
  [[nodiscard]] std::string message() const {
    if (resolved()) return {};
    std::string text =
        std::to_string(under) + " of " + std::to_string(shapes) +
        (shapes == 1 ? " shape is" : " shapes are") + " less than " +
        detail::short_number(wanted) + " voxels across: ";
    if (!(smallest_width > 0.0))   // a degenerate shape: no grid resolves it
      return text + "the thinnest has no width";
    text += "the thinnest is " +
        detail::short_number(smallest_width) + " wide, " +
        detail::short_number(voxels_across(), 3) + " voxels of " +
        detail::short_number(voxel) + ". nx >= " +
        std::to_string(suggested[0]) + ", ny >= " +
        std::to_string(suggested[1]);
    if (sampled_z) text += ", nz >= " + std::to_string(suggested[2]);
    text += " gives it " + detail::short_number(wanted) + " voxels";
    if (!suggestion_fits)
      text += ", which is more than the 2^32 voxels a grid may have";
    return text;
  }
};

// `wanted` voxels across the thinnest shape is the rule of thumb for a
// grid that an FFT solver can use; under it interfaces are staircases and
// thin shapes break up. The width of a shape is shape_base::min_width().
// The z axis counts for a 3D domain sampled with more than one layer.
template <typename T>
[[nodiscard]] voxel_resolution check_voxel_resolution(
    std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
    std::type_identity_t<domain<T>> const& domain_box,
    std::size_t nx, std::size_t ny, std::size_t nz, double wanted) {
  const std::size_t nz_eff = effective_nz(domain_box, nz);
  (void)checked_voxel_count(nx, ny, nz_eff);
  voxel_resolution r;
  r.wanted = wanted;
  r.sampled_z = domain_box.is_3d() && nz_eff > 1;
  r.suggested = {nx, ny, nz_eff};
  const std::array<double, 3> length{static_cast<double>(domain_box[0]),
                                     static_cast<double>(domain_box[1]),
                                     static_cast<double>(domain_box[2])};
  const std::array<std::size_t, 3> count{nx, ny, nz_eff};
  const int axes = r.sampled_z ? 3 : 2;
  for (int a = 0; a < axes; ++a)
    r.voxel = std::max(r.voxel, length[a] / static_cast<double>(count[a]));
  if (!(wanted > 0.0) || !(r.voxel > 0.0)) return r;

  double smallest = std::numeric_limits<double>::infinity();
  for (auto const& shape : shapes) {
    if (!shape->is_primary()) continue;
    const double width = static_cast<double>(shape->min_width());
    if (!(width >= 0.0) || !std::isfinite(width)) continue;   // unknown
    ++r.shapes;
    smallest = std::min(smallest, width);
    // A hair under the threshold from rounding is not under-resolved.
    if (width / r.voxel < wanted * (1.0 - 1e-9)) ++r.under;
  }
  if (r.shapes == 0) return r;
  r.smallest_width = smallest;
  if (r.under == 0 || !(smallest > 0.0)) return r;
  const double cap = static_cast<double>(max_voxel_count);
  double product = 1.0;
  for (int a = 0; a < axes; ++a) {
    const double n = std::ceil(wanted * length[a] / smallest * (1.0 - 1e-9));
    r.suggested[a] = std::max(
        count[a], static_cast<std::size_t>(std::min(n, cap)));
  }
  for (const auto n : r.suggested) product *= static_cast<double>(n);
  r.suggestion_fits = product <= cap;
  return r;
}

// The two parameters every voxel writer has on top of its grid (#115).
template <typename T>
void add_voxel_sampling_parameters(parameter_controller_t& s) {
  s.template optional<std::size_t>("subsamples", std::size_t{1})
      .template add<numsim_core::range<std::size_t{1}, max_voxel_subsamples>>()
      .template add<numsim_core::description_label<"sample points per voxel and axis: 1 samples the voxel centre, k > 1 writes partial volumes (the share of each id in a voxel) from k x k (x k) points">>();
  s.template optional<T>("min_voxels_per_feature", T{8})
      .template add<min_only<T{0}>>()
      .template add<numsim_core::unit_label<"cells">>()
      .template add<numsim_core::description_label<"warn when the thinnest shape is fewer voxels across than this (0: no check)">>();
}

// Reject a grid or a subsample count that cannot be sampled, before any
// file is created: the voxel count, and the sample points of one row of
// voxels, which is the least the sampler holds in memory.
template <typename T>
void validate_voxel_sampling(domain<T> const& domain_box, std::size_t nx,
                             std::size_t ny, std::size_t nz,
                             std::size_t subsamples) {
  (void)checked_voxel_count(nx, ny, effective_nz(domain_box, nz));
  if (subsamples == 0 || subsamples > max_voxel_subsamples) {
    throw std::invalid_argument{
        "voxel grid: subsamples must be between 1 and " +
        std::to_string(max_voxel_subsamples) + ", got " +
        std::to_string(subsamples)};
  }
  (void)detail::checked_voxel_row_samples(
      nx, subsamples, domain_box.is_3d() ? subsamples : 1);
}

// What a voxel writer says about the grid it wrote (#115):
//   info  "volume fraction ...": the grid's against the shapes';
//   warn  the resolution message, when a shape is under `min_voxels`
//         voxels across (0 switches the check off).
// `samples_of_id` are the sample points per id of the grid as written
// (voxel_fractions::samples_of_id or count_voxel_ids); `phases` may be
// null (inclusion indices: every id but 0 is an inclusion).
template <typename T>
void report_voxel_sampling(
    std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
    std::type_identity_t<domain<T>> const& domain_box,
    std::size_t nx, std::size_t ny, std::size_t nz, std::size_t subsamples,
    double min_voxels, phase_collection<T> const* phases, bool strict,
    std::vector<std::uint64_t> const& samples_of_id,
    std::function<void(std::string const&)> const& info,
    std::function<void(std::string const&)> const& warn) {
  const std::size_t nz_eff = effective_nz(domain_box, nz);
  const bool is_3d = domain_box.is_3d();
  if (info) {
    const auto shape_to_id =
        phases ? detail::phase_id_of_shapes(shapes, *phases, strict)
               : detail::inclusion_index_of_shapes(shapes);
    const phase_id fill = phases ? detail::matrix_phase_id(*phases) : 0;
    const auto of_shapes =
        shape_size_fractions(shapes, domain_box, shape_to_id, fill);
    std::uint64_t total = 0;
    for (const auto n : samples_of_id) total += n;
    auto on_grid = [&](std::size_t id) {
      return id < samples_of_id.size() && total > 0
                 ? static_cast<double>(samples_of_id[id]) /
                       static_cast<double>(total)
                 : 0.0;
    };
    std::string grid = std::to_string(nx) + " x " + std::to_string(ny);
    if (is_3d) grid += " x " + std::to_string(nz_eff);
    grid += " voxels";
    if (subsamples > 1) {
      const std::string k = std::to_string(subsamples);
      grid += ", " + k + " x " + k + (is_3d ? " x " + k : std::string{}) +
              " sample points each";
    }
    double sum_of_shapes = 0.0;
    for (const double f : of_shapes) sum_of_shapes += f;
    // More than the box: the sizes count overlaps (or parts outside) twice.
    const std::string excess =
        sum_of_shapes > 1.0 + 1e-9
            ? "; the shapes overlap or leave the box, their sizes add up "
              "to more than the box"
            : "";
    std::string line;
    if (phases == nullptr) {
      line = "volume fraction of the inclusions " +
             detail::short_number(1.0 - on_grid(0)) + " on the grid, " +
             detail::short_number(sum_of_shapes - of_shapes[0]) +
             " from the shape sizes (" + grid + ")" + excess;
    } else {
      line = "volume fractions on the grid (from the shape sizes): ";
      std::vector<std::pair<std::size_t, std::string>> names;
      for (auto const* p : phases->ordered())
        names.emplace_back(static_cast<std::size_t>(p->id), p->name);
      // Id 0 is no phase's id (they start at 1): shapes without a phase
      // name and, when no matrix phase is declared, what no shape covers.
      const double zero_grid = on_grid(0);
      const double zero_shapes = of_shapes.empty() ? 0.0 : of_shapes[0];
      if (zero_grid > 0.0 || std::abs(zero_shapes) > 1e-12)
        names.emplace_back(0, phases->has_matrix() ? "untagged (id 0)"
                                                    : "unassigned (id 0)");
      constexpr std::size_t listed = 12;
      for (std::size_t n = 0; n < names.size() && n < listed; ++n) {
        const std::size_t id = names[n].first;
        if (n > 0) line += ", ";
        line += names[n].second + " " + detail::short_number(on_grid(id)) +
                " (" +
                detail::short_number(id < of_shapes.size() ? of_shapes[id]
                                                           : 0.0) +
                ")";
      }
      if (names.size() > listed)
        line += ", and " + std::to_string(names.size() - listed) + " more";
      line += " (" + grid + ")" + excess;
    }
    info(line);
  }
  if (warn && min_voxels > 0.0) {
    const auto resolution =
        check_voxel_resolution(shapes, domain_box, nx, ny, nz, min_voxels);
    if (!resolution.resolved())
      warn(resolution.message() +
           " (\"min_voxels_per_feature\" sets the threshold, 0 switches the "
           "check off)");
  }
}

} // namespace rvegen
