#pragma once

#include <array>

#include "../shapes/bounding_box_base.h"

namespace rvegen {

// AABB overlap test on the polymorphic bounding_box_base interface.
//
// Used by the generator's heterogeneous-shape collision pre-check: shapes
// stored as unique_ptr<shape_base<T>> can't easily reach into the
// collision_details overload set without dynamic_cast. AABB overlap is a
// safe (conservative) fallback — true intersections are always caught;
// false positives just reduce achievable packing density.
//
// A precise pairwise dispatcher (registry of typeid-keyed function pointers
// over collision_details) replaces this in a later phase.
template <typename T>
[[nodiscard]] constexpr bool aabb_overlap(bounding_box_base<T> const& lhs,
                                           bounding_box_base<T> const& rhs) noexcept {
  auto const& lt = lhs.top_point();
  auto const& lb = lhs.bottom_point();
  auto const& rt = rhs.top_point();
  auto const& rb = rhs.bottom_point();
  return lb[0] <= rt[0] && lt[0] >= rb[0] &&
         lb[1] <= rt[1] && lt[1] >= rb[1] &&
         lb[2] <= rt[2] && lt[2] >= rb[2];
}

// Is the shape's bounding box fully contained in [0, box[0]] × [0, box[1]] × [0, box[2]] ?
template <typename T>
[[nodiscard]] constexpr bool aabb_inside(bounding_box_base<T> const& bb,
                                          T box_x, T box_y, T box_z) noexcept {
  auto const& t = bb.top_point();
  auto const& b = bb.bottom_point();
  return b[0] >= T{0} && t[0] <= box_x &&
         b[1] >= T{0} && t[1] <= box_y &&
         b[2] >= T{0} && t[2] <= box_z;
}

// AABB overlap with `rhs` shifted by the given delta. Used by
// periodic_aabb_overlap to test against periodic images of `rhs`.
template <typename T>
[[nodiscard]] constexpr bool
aabb_overlap_shifted(bounding_box_base<T> const& lhs,
                     bounding_box_base<T> const& rhs,
                     std::array<T, 3> const& shift) noexcept {
  auto const& lt = lhs.top_point();
  auto const& lb = lhs.bottom_point();
  auto const& rt = rhs.top_point();
  auto const& rb = rhs.bottom_point();
  const T rb0 = rb[0] + shift[0], rt0 = rt[0] + shift[0];
  const T rb1 = rb[1] + shift[1], rt1 = rt[1] + shift[1];
  const T rb2 = rb[2] + shift[2], rt2 = rt[2] + shift[2];
  return lb[0] <= rt0 && lt[0] >= rb0 &&
         lb[1] <= rt1 && lt[1] >= rb1 &&
         lb[2] <= rt2 && lt[2] >= rb2;
}

// Periodic AABB overlap test: do `lhs` and any periodic image of `rhs`
// (translated by integer multiples of the domain box dimensions) overlap?
//
// Iterates 3×3 (2D, when box[2] == 0) or 3×3×3 (3D) shifts including
// (0,0,0). Brute force is acceptable here — RVE inner loops typically have
// O(N) accepted shapes, and periodic checks add a 9× / 27× constant factor.
template <typename T>
[[nodiscard]] bool periodic_aabb_overlap(
    bounding_box_base<T> const& lhs,
    bounding_box_base<T> const& rhs,
    std::array<T, 3> const& domain_box) noexcept {
  const bool is_3d = domain_box[2] > T{0};
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (!is_3d && dz != 0) continue;
        const std::array<T, 3> shift{
            static_cast<T>(dx) * domain_box[0],
            static_cast<T>(dy) * domain_box[1],
            static_cast<T>(dz) * domain_box[2]};
        if (aabb_overlap_shifted(lhs, rhs, shift)) return true;
      }
    }
  }
  return false;
}

// True if the shape's middle point sits in [0, Lx) × [0, Ly) × [0, Lz).
// The half-open intervals match the periodic-wrap convention: a shape at
// exactly x = Lx is the wrap of a shape at x = 0; only one of them is the
// "primary". 2D RVEs (domain_box[2] == 0) ignore the z component.
//
// Used by periodic accounting (e.g. volume_fraction) to count primaries
// only, not periodic wrap copies that share storage.
template <typename T>
[[nodiscard]] constexpr bool
is_centre_in_domain(std::array<T, 3> const& middle_point,
                    std::array<T, 3> const& domain_box) noexcept {
  bool const is_3d = domain_box[2] > T{0};
  return middle_point[0] >= T{0} && middle_point[0] < domain_box[0] &&
         middle_point[1] >= T{0} && middle_point[1] < domain_box[1] &&
         (!is_3d || (middle_point[2] >= T{0} && middle_point[2] < domain_box[2]));
}

} // namespace rvegen
