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
// A bounding box that pokes through BOTH faces of an axis (e.g. a weave
// yarn spanning [−r, L + r]) gets a wrap in both directions on that
// axis, so each axis contributes 1, 2 or 3 shift options {0, +L, −L}
// and the shape gets (Π options) − 1 wraps.
//
// Used by periodic_generator to populate the accepted vector with the
// full equivalence class so subsequent collision checks can use the
// non-periodic (faster) dispatch.
//
// Wraps are clones, so they copy the primary's `info`. periodic_generator
// stamps the primary with an inclusion id (set_inclusion_id) before
// generating its wraps, which gives every member of the equivalence
// class the same id. Shape-index voxel output uses that id to number one
// physical inclusion once, however many periodic pieces it has.

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "shape_base.h"

namespace rvegen {

// `info` key linking a primary and its periodic wraps. Namespaced with the
// `rvegen_` prefix so it cannot collide with user metadata keys.
inline constexpr char const* inclusion_id_key = "rvegen_inclusion_id";

template <typename T>
void set_inclusion_id(shape_base<T>& shape, std::size_t id) {
  shape.info().set(inclusion_id_key, id);
}

// Inclusion id shared by a primary and its wraps, or nullopt for shapes
// that were not produced by a periodic placement.
template <typename T>
[[nodiscard]] std::optional<std::size_t>
inclusion_id(shape_base<T> const& shape) {
  if (!shape.info().contains(inclusion_id_key)) return std::nullopt;
  return shape.info().template get<std::size_t>(inclusion_id_key);
}

template <typename T>
[[nodiscard]] std::vector<std::unique_ptr<shape_base<T>>>
generate_periodic_wraps(shape_base<T> const& primary,
                         std::array<T, 3> const& domain_box) {
  std::vector<std::unique_ptr<shape_base<T>>> wraps;
  auto const* bb = primary.bounding_box();
  if (bb == nullptr) return wraps;

  // For each axis, collect the shift options. Option 0 is always "no
  // shift" (the primary's own position on this axis); then
  //   +1  →  wrap to the +L side (bb pokes the low face)
  //   −1  →  wrap to the −L side (bb pokes the high face)
  // Both can apply when the bb sticks out of both faces.
  std::array<std::array<int, 3>, 3> shift_opts{};
  std::array<std::size_t, 3> n_opts{1, 1, 1};
  bool const is_3d = domain_box[2] > T{0};
  for (int axis = 0; axis < (is_3d ? 3 : 2); ++axis) {
    if (bb->bottom_point()[axis] < T{0}) {
      shift_opts[axis][n_opts[axis]++] = +1;
    }
    if (bb->top_point()[axis] > domain_box[axis]) {
      shift_opts[axis][n_opts[axis]++] = -1;
    }
  }

  // Enumerate every combination of per-axis options as a mixed-radix
  // counter (axis 0 is the least-significant digit). Combination 0 is
  // the primary (skipped here); each other combination becomes one
  // wrap. With at most one option per axis this reduces to the 2^k − 1
  // bitmask subsets, in the same order.
  std::size_t const n_combinations = n_opts[0] * n_opts[1] * n_opts[2];
  for (std::size_t combo = 1; combo < n_combinations; ++combo) {
    std::array<T, 3> shift{T{0}, T{0}, T{0}};
    std::size_t rest = combo;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      std::size_t const digit = rest % n_opts[axis];
      rest /= n_opts[axis];
      shift[axis] =
          static_cast<T>(shift_opts[axis][digit]) * domain_box[axis];
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
