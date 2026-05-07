#pragma once

#include <array>
#include <memory>

namespace rvegen {

// Polymorphic AABB interface. Concrete implementations: rectangle_bounding (2D),
// box_bounding (3D). Used by shape_base for fast intersection pre-checks.
template <typename T>
class bounding_box_base {
public:
  using value_type = T;

  bounding_box_base() = default;
  bounding_box_base(bounding_box_base const&) = default;
  bounding_box_base(bounding_box_base&&) noexcept = default;
  bounding_box_base& operator=(bounding_box_base const&) = default;
  bounding_box_base& operator=(bounding_box_base&&) noexcept = default;
  virtual ~bounding_box_base() = default;

  [[nodiscard]] virtual std::array<value_type, 3> const& top_point() const = 0;
  [[nodiscard]] virtual std::array<value_type, 3> const& bottom_point() const = 0;

  // Polymorphic deep-copy. Used by shape_base's copy ctor / op= so that
  // shapes can be value-copied without each shape needing to know how to
  // clone its own bb.
  [[nodiscard]] virtual std::unique_ptr<bounding_box_base<value_type>> clone() const = 0;
};

} // namespace rvegen
