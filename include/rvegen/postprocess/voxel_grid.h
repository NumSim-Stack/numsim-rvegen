#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "../phase/phase.h"
#include "../shapes/shape_base.h"

namespace rvegen {

using phase_id = std::uint32_t;

// Sample a regular voxel grid: for each voxel centre point, ask each shape
// in turn whether the point is inside; mark with the (1-based) shape index
// or 0 if no shape contains it. First-match wins; later shapes don't
// overwrite earlier ones.
//
// 2D RVEs (domain_box[2] == 0) collapse to a single z-slice (nz_eff = 1).
// Returns the flat array in (k, j, i) order — k is the slowest-varying axis.
template <typename T>
[[nodiscard]] std::vector<phase_id>
sample_voxel_grid(std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
                  std::array<T, 3> const& domain_box,
                  std::size_t nx, std::size_t ny, std::size_t nz) {
  const std::size_t nz_eff = (domain_box[2] > T{0}) ? nz : 1;
  std::vector<phase_id> grid(nx * ny * nz_eff, 0);

  const auto dx = domain_box[0] / static_cast<T>(nx);
  const auto dy = domain_box[1] / static_cast<T>(ny);
  const auto dz = (nz_eff > 1) ? domain_box[2] / static_cast<T>(nz_eff)
                               : T{0};

  for (std::size_t k = 0; k < nz_eff; ++k) {
    const auto z = (nz_eff > 1)
        ? (static_cast<T>(k) + T{0.5}) * dz
        : T{0};
    for (std::size_t j = 0; j < ny; ++j) {
      const auto y = (static_cast<T>(j) + T{0.5}) * dy;
      for (std::size_t i = 0; i < nx; ++i) {
        const auto x = (static_cast<T>(i) + T{0.5}) * dx;
        const std::array<T, 3> p{x, y, z};

        for (std::size_t s = 0; s < shapes.size(); ++s) {
          if (shapes[s]->is_inside(p)) {
            grid[(k * ny + j) * nx + i] = static_cast<phase_id>(s + 1);
            break;
          }
        }
      }
    }
  }
  return grid;
}

// Phase-aware overload: each voxel gets the phase id resolved from
// `shape.phase_name()` via the supplied `phase_collection`, instead of the
// 1-based shape index. Voxels outside every shape (and voxels whose
// containing shape has an empty `phase_name`, i.e. no phase was tagged
// in the input JSON) stay at 0.
//
// Throws if a shape carries a `phase_name` that does not appear in the
// `phase_collection` — a likely-typo failure mode worth catching early
// rather than silently mapping to 0. (Untagged shapes — empty name — are
// not an error; they just keep the matrix id.)
template <typename T>
[[nodiscard]] std::vector<phase_id>
sample_voxel_grid(std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
                  std::array<T, 3> const& domain_box,
                  std::size_t nx, std::size_t ny, std::size_t nz,
                  phase_collection<T> const& phases) {
  const std::size_t nz_eff = (domain_box[2] > T{0}) ? nz : 1;
  std::vector<phase_id> grid(nx * ny * nz_eff, 0);

  // Pre-resolve phase ids once per shape — avoids re-hashing the name
  // on every voxel and lets us throw early on unknown names.
  std::vector<phase_id> shape_phase_id(shapes.size(), 0);
  for (std::size_t s = 0; s < shapes.size(); ++s) {
    const auto name = shapes[s]->phase_name();
    if (name.empty()) continue;     // untagged → matrix (0)
    shape_phase_id[s] = static_cast<phase_id>(phases.id_of(name));
  }

  const auto dx = domain_box[0] / static_cast<T>(nx);
  const auto dy = domain_box[1] / static_cast<T>(ny);
  const auto dz = (nz_eff > 1) ? domain_box[2] / static_cast<T>(nz_eff)
                               : T{0};

  for (std::size_t k = 0; k < nz_eff; ++k) {
    const auto z = (nz_eff > 1)
        ? (static_cast<T>(k) + T{0.5}) * dz
        : T{0};
    for (std::size_t j = 0; j < ny; ++j) {
      const auto y = (static_cast<T>(j) + T{0.5}) * dy;
      for (std::size_t i = 0; i < nx; ++i) {
        const auto x = (static_cast<T>(i) + T{0.5}) * dx;
        const std::array<T, 3> p{x, y, z};

        for (std::size_t s = 0; s < shapes.size(); ++s) {
          if (shapes[s]->is_inside(p)) {
            grid[(k * ny + j) * nx + i] = shape_phase_id[s];
            break;
          }
        }
      }
    }
  }
  return grid;
}

// Effective z-dimension of the grid given the domain (collapses to 1 in 2D).
template <typename T>
[[nodiscard]] constexpr std::size_t
effective_nz(std::array<T, 3> const& domain_box, std::size_t nz) noexcept {
  return (domain_box[2] > T{0}) ? nz : std::size_t{1};
}

} // namespace rvegen
