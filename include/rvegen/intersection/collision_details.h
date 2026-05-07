#pragma once

// Compile-time collision dispatch for the rvegen library.
//
// Each shape pair has its own free-function template overload of
// collision_details(A, B). Overload resolution picks the right one at the
// call site — no runtime type lookup, no virtual dispatch, no warehouse.
//
// Because rvegen shapes inherit from their corresponding GTE primitives
// (rvegen::circle is gte::Circle2, etc.), the overloads can pass shapes
// directly to gte::TIQuery without any conversion or adapter type.

#include <algorithm>

#include <Mathematics/Hyperellipsoid.h>
#include <Mathematics/Hypersphere.h>
#include <Mathematics/IntrEllipse2Ellipse2.h>

#include "../shapes/box.h"
#include "../shapes/circle.h"
#include "../shapes/ellipse.h"
#include "../shapes/rectangle.h"
#include "../shapes/sphere.h"

namespace rvegen {

// ---- circle × circle ------------------------------------------------------
template <typename T>
[[nodiscard]] constexpr bool collision_details(circle<T> const& lhs,
                                                circle<T> const& rhs) noexcept {
  const auto sum_r = lhs.radius + rhs.radius;
  const auto dx = lhs(0) - rhs(0);
  const auto dy = lhs(1) - rhs(1);
  return sum_r * sum_r >= (dx * dx + dy * dy);
}

// ---- sphere × sphere ------------------------------------------------------
template <typename T>
[[nodiscard]] constexpr bool collision_details(sphere<T> const& lhs,
                                                sphere<T> const& rhs) noexcept {
  const auto sum_r = lhs.radius + rhs.radius;
  const auto dx = lhs(0) - rhs(0);
  const auto dy = lhs(1) - rhs(1);
  const auto dz = lhs(2) - rhs(2);
  return sum_r * sum_r >= (dx * dx + dy * dy + dz * dz);
}

// ---- rectangle × rectangle ------------------------------------------------
// AlignedBox vs AlignedBox: simple min/max overlap.
template <typename T>
[[nodiscard]] constexpr bool collision_details(rectangle<T> const& lhs,
                                                rectangle<T> const& rhs) noexcept {
  return lhs.min[0] <= rhs.max[0] && lhs.max[0] >= rhs.min[0] &&
         lhs.min[1] <= rhs.max[1] && lhs.max[1] >= rhs.min[1];
}

// ---- box × box ------------------------------------------------------------
template <typename T>
[[nodiscard]] constexpr bool collision_details(box<T> const& lhs,
                                                box<T> const& rhs) noexcept {
  return lhs.min[0] <= rhs.max[0] && lhs.max[0] >= rhs.min[0] &&
         lhs.min[1] <= rhs.max[1] && lhs.max[1] >= rhs.min[1] &&
         lhs.min[2] <= rhs.max[2] && lhs.max[2] >= rhs.min[2];
}

namespace detail {
template <typename T>
constexpr T clamp(T v, T lo, T hi) noexcept {
  return v < lo ? lo : (v > hi ? hi : v);
}
} // namespace detail

// ---- circle × rectangle (and reverse) -------------------------------------
template <typename T>
[[nodiscard]] constexpr bool collision_details(circle<T> const& c,
                                                rectangle<T> const& r) noexcept {
  const auto closest_x = detail::clamp(c(0), r.min[0], r.max[0]);
  const auto closest_y = detail::clamp(c(1), r.min[1], r.max[1]);
  const auto dx = c(0) - closest_x;
  const auto dy = c(1) - closest_y;
  return c.radius * c.radius >= (dx * dx + dy * dy);
}

template <typename T>
[[nodiscard]] constexpr bool collision_details(rectangle<T> const& r,
                                                circle<T> const& c) noexcept {
  return collision_details(c, r);
}

// ---- sphere × box (and reverse) -------------------------------------------
template <typename T>
[[nodiscard]] constexpr bool collision_details(sphere<T> const& s,
                                                box<T> const& b) noexcept {
  const auto closest_x = detail::clamp(s(0), b.min[0], b.max[0]);
  const auto closest_y = detail::clamp(s(1), b.min[1], b.max[1]);
  const auto closest_z = detail::clamp(s(2), b.min[2], b.max[2]);
  const auto dx = s(0) - closest_x;
  const auto dy = s(1) - closest_y;
  const auto dz = s(2) - closest_z;
  return s.radius * s.radius >= (dx * dx + dy * dy + dz * dz);
}

template <typename T>
[[nodiscard]] constexpr bool collision_details(box<T> const& b,
                                                sphere<T> const& s) noexcept {
  return collision_details(s, b);
}

// ---- ellipse × ellipse ----------------------------------------------------
// Precise rotated intersection via gte::TIQuery on the underlying
// gte::Ellipse2<T>. rvegen::ellipse IS-A gte::Ellipse2, so the upcast is
// implicit — no adapter needed.
namespace detail {
template <typename T>
[[nodiscard]] inline bool gte_ellipses_collide(gte::Ellipse2<T> const& a,
                                                gte::Ellipse2<T> const& b) {
  using Q = gte::TIQuery<T, gte::Ellipse2<T>, gte::Ellipse2<T>>;
  Q query;
  return query(a, b) != Q::Classification::ELLIPSES_SEPARATED;
}
} // namespace detail

template <typename T>
[[nodiscard]] bool collision_details(ellipse<T> const& lhs,
                                      ellipse<T> const& rhs) {
  return detail::gte_ellipses_collide<T>(lhs, rhs);
}

// ---- circle × ellipse (and reverse) --------------------------------------
// A circle is a degenerate axis-aligned ellipse. Build a temporary Ellipse2
// in-place to drive the same TIQuery; allocation-free.
template <typename T>
[[nodiscard]] bool collision_details(circle<T> const& c,
                                      ellipse<T> const& e) {
  gte::Ellipse2<T> circle_as_ellipse;
  circle_as_ellipse.center = c.center;
  circle_as_ellipse.axis[0] = {T{1}, T{0}};
  circle_as_ellipse.axis[1] = {T{0}, T{1}};
  circle_as_ellipse.extent = {c.radius, c.radius};
  return detail::gte_ellipses_collide<T>(circle_as_ellipse, e);
}

template <typename T>
[[nodiscard]] bool collision_details(ellipse<T> const& e,
                                      circle<T> const& c) {
  return collision_details(c, e);
}

} // namespace rvegen
