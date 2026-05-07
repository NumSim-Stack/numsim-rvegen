#pragma once

// generate_periodic_wraps(primary, domain_box) — clone the primary shape
// once for each face it pokes through (and for each pokes-through-corner
// combination), translating by ±domain_box[axis] so the wrap appears on
// the opposite face.
//
// The primary itself is NOT included in the returned vector; only the
// wraps. A shape that fits cleanly inside the domain returns an empty
// vector. A shape pokes through k axes returns 2^k − 1 wraps (e.g. a
// shape poking through +x AND +y returns 3 wraps: −x, −y, and −x−y).
//
// Used by periodic_generator to populate the accepted vector with the
// full equivalence class so subsequent collision checks can use the
// non-periodic (faster) dispatch.

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "shape_base.h"

namespace rvegen {

template <typename T>
[[nodiscard]] std::vector<std::unique_ptr<shape_base<T>>>
generate_periodic_wraps(shape_base<T> const& primary,
                         std::array<T, 3> const& domain_box) {
  std::vector<std::unique_ptr<shape_base<T>>> wraps;
  auto const* bb = primary.bounding_box();
  if (bb == nullptr) return wraps;

  // For each axis, decide if the bb pokes through:
  //   shift_dirs[axis] = -1  →  wrap to the −L side (bb pokes high face)
  //   shift_dirs[axis] = +1  →  wrap to the +L side (bb pokes low face)
  //   shift_dirs[axis] = 0   →  no wrap on this axis
  std::array<int, 3> shift_dirs{0, 0, 0};
  bool const is_3d = domain_box[2] > T{0};
  for (int axis = 0; axis < (is_3d ? 3 : 2); ++axis) {
    if (bb->bottom_point()[axis] < T{0}) {
      shift_dirs[axis] = +1;
    } else if (bb->top_point()[axis] > domain_box[axis]) {
      shift_dirs[axis] = -1;
    }
  }

  // Collect the poking axes; each contributes one bit to the combination
  // mask. With k poking axes there are 2^k subsets; subset 0 is the
  // primary (skipped here). Each non-zero subset becomes one wrap.
  std::array<int, 3> poking_axes{};
  std::size_t n_poking = 0;
  for (int axis = 0; axis < (is_3d ? 3 : 2); ++axis) {
    if (shift_dirs[axis] != 0) poking_axes[n_poking++] = axis;
  }
  if (n_poking == 0) return wraps;

  std::size_t const n_combinations = std::size_t{1} << n_poking;
  for (std::size_t mask = 1; mask < n_combinations; ++mask) {
    std::array<T, 3> shift{T{0}, T{0}, T{0}};
    for (std::size_t i = 0; i < n_poking; ++i) {
      if (mask & (std::size_t{1} << i)) {
        int const axis = poking_axes[i];
        shift[axis] = static_cast<T>(shift_dirs[axis]) * domain_box[axis];
      }
    }
    auto wrap = primary.clone();
    auto mp = wrap->get_middle_point();
    wrap->set_middle_point({mp[0] + shift[0],
                             mp[1] + shift[1],
                             mp[2] + shift[2]});
    wrap->make_bounding_box();
    wraps.push_back(std::move(wrap));
  }
  return wraps;
}

} // namespace rvegen
