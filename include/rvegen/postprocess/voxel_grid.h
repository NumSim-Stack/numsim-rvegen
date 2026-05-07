#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

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

// Effective z-dimension of the grid given the domain (collapses to 1 in 2D).
template <typename T>
[[nodiscard]] constexpr std::size_t
effective_nz(std::array<T, 3> const& domain_box, std::size_t nz) noexcept {
  return (domain_box[2] > T{0}) ? nz : std::size_t{1};
}

} // namespace rvegen
