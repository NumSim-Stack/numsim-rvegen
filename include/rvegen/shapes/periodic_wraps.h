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
// Linking (#121): every wrap is marked as an image of the primary. It
// carries the integer lattice shift (shape_base::image_shift, relative to
// the primary's own shift, normally {0,0,0}) and shares the primary's
// inclusion id. When wraps exist and the primary has no id yet, the primary
// is given a fresh one, which is why it is taken by non-const reference.
// A primary without wraps is left untouched. Consumers that must count an
// inclusion once use is_primary() / is_counted_in_rve() below; shape-index
// voxel output numbers one physical inclusion once through the shared id.

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "../domain.h"
#include "../intersection/aabb_overlap.h"
#include "shape_base.h"

namespace rvegen {

// Process-wide unique id for a periodic equivalence class. Unique across
// generate() calls so results from several runs can be concatenated
// without unrelated inclusions being merged. Voxel numbering depends only
// on first appearance, never on the id value, so output stays deterministic.
[[nodiscard]] inline std::size_t next_inclusion_id() noexcept {
  static std::atomic<std::size_t> counter{0};
  return counter.fetch_add(1, std::memory_order_relaxed);
}

// Inclusion id shared by a primary and its wraps, or nullopt for shapes
// that were not part of a multi-piece periodic inclusion.
template <typename T>
[[nodiscard]] std::optional<std::size_t>
inclusion_id(shape_base<T> const& shape) {
  return shape.inclusion_id();
}

// Does this shape count as material of the RVE, once? The single rule
// behind the volume fraction (termination, progress, statistics) and the
// centre-based statistics:
//   * a periodic copy never counts: its primary does;
//   * a primary that has periodic copies counts in full, wherever its
//     centre lies: the copies bring the parts that left the box back in;
//   * a primary without copies counts when its centre is inside the box.
//     For a shape fully inside that is always true. It only decides for a
//     shape that crosses a face WITHOUT being wrapped (random_generator
//     clips at the box): counting the shapes centred inside in full and
//     the ones centred outside not at all is the unbiased estimate of the
//     clipped volume, and has nothing to do with periodic images.
template <typename T>
[[nodiscard]] inline bool
is_counted_in_rve(shape_base<T> const& shape,
                  std::type_identity_t<domain<T>> const& domain_box) noexcept {
  if (!shape.is_primary()) return false;
  if (shape.inclusion_id().has_value()) return true;
  return is_centre_in_domain(shape.get_middle_point(), domain_box);
}

template <typename T>
[[nodiscard]] std::vector<std::unique_ptr<shape_base<T>>>
generate_periodic_wraps(shape_base<T>& primary,
                         std::type_identity_t<domain<T>> const& domain_box) {
  std::vector<std::unique_ptr<shape_base<T>>> wraps;
  auto const* bb = primary.bounding_box();
  // A shape without a computed bounding box (e.g. generator-built Voronoi
  // cells) must not silently get zero wraps: derive the box from a clone.
  std::unique_ptr<shape_base<T>> bb_holder;
  if (bb == nullptr) {
    bb_holder = primary.clone();
    bb_holder->make_bounding_box();
    bb = bb_holder->bounding_box();
    if (bb == nullptr) return wraps;
  }

  // For each axis, collect the shift options. Option 0 is always "no
  // shift" (the primary's own position on this axis); then
  //   +1  →  wrap to the +L side (bb pokes the low face)
  //   −1  →  wrap to the −L side (bb pokes the high face)
  // Both can apply when the bb sticks out of both faces.
  std::array<std::array<int, 3>, 3> shift_opts{};
  std::array<std::size_t, 3> n_opts{1, 1, 1};
  for (int axis = 0; axis < domain_box.dimension(); ++axis) {
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
    std::array<int, 3> lattice = primary.image_shift();
    std::size_t rest = combo;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      std::size_t const digit = rest % n_opts[axis];
      rest /= n_opts[axis];
      shift[axis] =
          static_cast<T>(shift_opts[axis][digit]) * domain_box[axis];
      lattice[axis] += shift_opts[axis][digit];
    }
    auto wrap = primary.clone();
    auto mp = wrap->get_middle_point();
    wrap->set_middle_point({mp[0] + shift[0],
                             mp[1] + shift[1],
                             mp[2] + shift[2]});
    wrap->set_image_shift(lattice);
    wrap->make_bounding_box();
    wraps.push_back(std::move(wrap));
  }
  // One id for the whole class. Only a multi-piece class gets one: a
  // primary without wraps stays an ordinary single shape.
  if (!wraps.empty()) {
    if (!primary.inclusion_id()) primary.set_inclusion_id(next_inclusion_id());
    for (auto& w : wraps) w->set_inclusion_id(*primary.inclusion_id());
  }
  return wraps;
}

} // namespace rvegen
