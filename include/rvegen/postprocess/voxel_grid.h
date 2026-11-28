#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <utility>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <vector>

#include "../phase/phase.h"
#include "../shapes/periodic_wraps.h"
#include "../shapes/shape_base.h"

namespace rvegen {

using phase_id = std::uint32_t;

// Upper bound on the number of voxels of one grid (2^32; the id array alone
// is then 16 GiB). Larger grids are rejected up front instead of wrapping
// the size_t product or failing deep inside the allocator.
inline constexpr std::uint64_t max_voxel_count = std::uint64_t{1} << 32;

// The ONE place that turns (nx, ny, nz) into a voxel count. `nz` must
// already be the effective z count (see `effective_nz`). Throws
// std::invalid_argument naming nx/ny/nz for a zero count, for a product that
// would overflow, and for a product above `max_voxel_count`.
[[nodiscard]] inline std::size_t
checked_voxel_count(std::size_t nx, std::size_t ny, std::size_t nz) {
  auto fail = [&](char const* why) {
    throw std::invalid_argument{
        std::string{"voxel grid: "} + why + " (nx=" + std::to_string(nx) +
        ", ny=" + std::to_string(ny) + ", nz=" + std::to_string(nz) + ")"};
  };
  if (nx == 0 || ny == 0 || nz == 0) fail("nx, ny and nz must be >= 1");
  // Divide instead of multiplying so no intermediate product can wrap.
  const std::uint64_t cap = max_voxel_count;
  if (std::uint64_t{ny} > cap / nx) {
    fail("nx * ny * nz exceeds the 2^32 voxel limit");
  }
  const std::uint64_t xy = std::uint64_t{nx} * ny;  // <= 2^32
  if (std::uint64_t{nz} > cap / xy) {
    fail("nx * ny * nz exceeds the 2^32 voxel limit");
  }
  return static_cast<std::size_t>(xy * nz);
}

namespace detail {

// Core scanning loop shared by both `sample_voxel_grid` overloads.
// `shape_to_id[s]` is the value written to grid voxels that the s-th
// shape contains. First-match wins; voxels outside every shape keep
// `fill` (0 unless a matrix phase is declared).
template <typename T>
[[nodiscard]] std::vector<phase_id>
sample_voxel_grid_with_ids(
    std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
    std::array<T, 3> const& domain_box,
    std::size_t nx, std::size_t ny, std::size_t nz,
    std::vector<phase_id> const& shape_to_id,
    phase_id fill = 0) {
  // Key z on "is the domain 3D", not on nz > 1: a 3D domain sampled with
  // nz = 1 must hit the mid-plane z = Lz/2, not the bottom face z = 0.
  const bool is_3d = domain_box[2] > T{0};
  const std::size_t nz_eff = is_3d ? nz : 1;
  std::vector<phase_id> grid(checked_voxel_count(nx, ny, nz_eff), fill);

  const auto dx = domain_box[0] / static_cast<T>(nx);
  const auto dy = domain_box[1] / static_cast<T>(ny);
  const auto dz = is_3d ? domain_box[2] / static_cast<T>(nz_eff) : T{0};

  // Paint shapes in reverse order so the lowest-index shape wins on
  // overlap (first-match semantics), and visit only the voxels inside each
  // shape's AABB (padded by one voxel) instead of testing every shape at
  // every voxel. Shapes without a bounding box scan the full grid.
  auto range = [](T lo, T hi, T d, std::size_t n) {
    const double flo = std::floor(double(lo / d)) - 1.0;
    const double fhi = std::ceil(double(hi / d)) + 1.0;
    const std::size_t i0 =
        flo <= 0.0 ? 0 : (flo >= double(n) ? n : std::size_t(flo));
    const std::size_t i1 =
        fhi <= 0.0 ? 0 : (fhi >= double(n) ? n : std::size_t(fhi));
    return std::pair<std::size_t, std::size_t>{i0, i1};
  };
  for (std::size_t si = shapes.size(); si-- > 0;) {
    auto const& shape = shapes[si];
    std::size_t i0 = 0, i1 = nx, j0 = 0, j1 = ny, k0 = 0, k1 = nz_eff;
    if (auto const* bb = shape->bounding_box()) {
      auto const& lo = bb->bottom_point();
      auto const& hi = bb->top_point();
      // Use the box only when it is finite and still encloses the shape's
      // reference point; a stale box (shape moved without
      // make_bounding_box()) falls back to the full scan.
      const auto mp = shape->get_middle_point();
      bool usable = true;
      for (int a = 0; a < 3; ++a) {
        usable = usable && std::isfinite(double(lo[a])) &&
                 std::isfinite(double(hi[a])) && lo[a] <= mp[a] &&
                 mp[a] <= hi[a];
      }
      if (usable) {
        std::tie(i0, i1) = range(lo[0], hi[0], dx, nx);
        std::tie(j0, j1) = range(lo[1], hi[1], dy, ny);
        if (is_3d) std::tie(k0, k1) = range(lo[2], hi[2], dz, nz_eff);
      }
    }
    for (std::size_t k = k0; k < k1; ++k) {
      const auto z = is_3d ? (static_cast<T>(k) + T{0.5}) * dz : T{0};
      for (std::size_t j = j0; j < j1; ++j) {
        const auto y = (static_cast<T>(j) + T{0.5}) * dy;
        for (std::size_t i = i0; i < i1; ++i) {
          const auto x = (static_cast<T>(i) + T{0.5}) * dx;
          if (shape->is_inside(std::array<T, 3>{x, y, z})) {
            grid[(k * ny + j) * nx + i] = shape_to_id[si];
          }
        }
      }
    }
  }
  return grid;
}

} // namespace detail

// Sample a regular voxel grid: for each voxel centre point, ask each shape
// in turn whether the point is inside; mark with the (1-based) inclusion
// index or 0 if no shape contains it. First-match wins; later shapes don't
// overwrite earlier ones.
//
// Inclusion index: shapes are numbered 1, 2, ... in order of first
// appearance, except that shapes sharing an `inclusion_id` (a periodic
// primary and its wraps, linked by periodic_generator) share one number.
// Without periodic wraps this is the plain 1-based shape index.
//
// Voronoi grains: a shape whose info() has an integer "grain_id" (set by
// voronoi_to_shapes) is marked with grain_id + 1 instead, so the periodic
// wrap pieces of one grain share that grain's value and the value matches
// the grain numbering of the generator. If a shape list mixes grains with
// other shapes, the other shapes are numbered after the largest grain
// value, so the two numberings never collide.
//
// 2D RVEs (domain_box[2] == 0) collapse to a single z-slice (nz_eff = 1)
// at z = 0. 3D RVEs sample voxel centres along z as well, so nz = 1 gives
// the mid-plane slice z = Lz/2.
// Returns the flat array in (k, j, i) order — k is the slowest-varying axis.
template <typename T>
[[nodiscard]] std::vector<phase_id>
sample_voxel_grid(std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
                  std::array<T, 3> const& domain_box,
                  std::size_t nx, std::size_t ny, std::size_t nz) {
  std::vector<phase_id> shape_to_id(shapes.size());
  std::unordered_map<std::size_t, phase_id> id_of_inclusion;
  // Grain value (grain_id + 1) per shape, 0 for a shape without grain_id;
  // other shapes are numbered from the largest grain value on.
  std::vector<phase_id> grain_value(shapes.size(), 0);
  phase_id next_id = 0;
  for (std::size_t s = 0; s < shapes.size(); ++s) {
    auto const& info = shapes[s]->info();
    if (info.contains("grain_id")) {
      grain_value[s] =
          static_cast<phase_id>(info.template get<std::size_t>("grain_id") + 1);
      next_id = std::max(next_id, grain_value[s]);
    }
  }
  for (std::size_t s = 0; s < shapes.size(); ++s) {
    if (grain_value[s] != 0) {
      shape_to_id[s] = grain_value[s];
    } else if (auto const inc = inclusion_id(*shapes[s])) {
      auto const [it, inserted] =
          id_of_inclusion.try_emplace(*inc, next_id + 1);
      if (inserted) ++next_id;
      shape_to_id[s] = it->second;
    } else {
      shape_to_id[s] = ++next_id;
    }
  }
  return detail::sample_voxel_grid_with_ids(
      shapes, domain_box, nx, ny, nz, shape_to_id);
}

// Phase-aware overload: each voxel gets the phase id resolved from
// `shape.phase_name()` via the supplied `phase_collection`, instead of the
// 1-based shape index. Voxels outside every shape (and voxels whose
// containing shape has an empty `phase_name`, i.e. no phase was tagged
// in the input JSON) stay at 0 — except that voxels no shape covers take
// the id of the phase declared as matrix (`phases.set_matrix`), if any.
//
// Throws if a shape carries a `phase_name` that does not appear in the
// `phase_collection` — a likely-typo failure mode worth catching early
// rather than silently mapping to 0. (Untagged shapes — empty name — are
// not an error by default; they just keep the matrix id.)
//
// `strict` flips the empty-name behaviour: with strict=true, an
// untagged shape throws instead of falling through to the matrix id.
// Matches `gmsh_geo_writer::set_phase_strict()`.
template <typename T>
[[nodiscard]] std::vector<phase_id>
sample_voxel_grid(std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
                  std::array<T, 3> const& domain_box,
                  std::size_t nx, std::size_t ny, std::size_t nz,
                  phase_collection<T> const& phases,
                  bool strict = false) {
  // Pre-resolve phase ids once per shape — avoids re-hashing the name
  // on every voxel and lets us throw early on unknown names.
  std::vector<phase_id> shape_to_id(shapes.size(), 0);
  for (std::size_t s = 0; s < shapes.size(); ++s) {
    const auto name = shapes[s]->phase_name();
    if (name.empty()) {
      if (strict) {
        throw std::runtime_error{
            "sample_voxel_grid: shape has empty phase_name and strict "
            "mode is enabled. Tag every shape's phase, or disable strict mode."};
      }
      continue;
    }
    const auto id = phases.id_of(name);
    if (id > std::numeric_limits<phase_id>::max()) {
      throw std::runtime_error{
          "sample_voxel_grid: phase id exceeds phase_id range (uint32)"};
    }
    shape_to_id[s] = static_cast<phase_id>(id);
  }
  const auto matrix_id = phases.matrix_id();
  if (matrix_id > std::numeric_limits<phase_id>::max()) {
    throw std::runtime_error{
        "sample_voxel_grid: phase id exceeds phase_id range (uint32)"};
  }
  return detail::sample_voxel_grid_with_ids(
      shapes, domain_box, nx, ny, nz, shape_to_id,
      static_cast<phase_id>(matrix_id));
}

// Effective z-dimension of the grid given the domain (collapses to 1 in 2D).
template <typename T>
[[nodiscard]] constexpr std::size_t
effective_nz(std::array<T, 3> const& domain_box, std::size_t nz) noexcept {
  return (domain_box[2] > T{0}) ? nz : std::size_t{1};
}

} // namespace rvegen
