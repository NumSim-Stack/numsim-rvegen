#pragma once

#include <array>
#include <memory>

#include "bounding_box_base.h"

namespace rvegen {

// 3D AABB.
template <typename T>
class box_bounding final : public bounding_box_base<T> {
public:
  using value_type = T;

  box_bounding() = default;

  box_bounding(std::array<value_type, 3> top,
               std::array<value_type, 3> bottom) noexcept
      : _top{top}, _bottom{bottom} {}

  box_bounding(value_type top_x, value_type top_y, value_type top_z,
               value_type bottom_x, value_type bottom_y, value_type bottom_z) noexcept
      : _top{top_x, top_y, top_z}, _bottom{bottom_x, bottom_y, bottom_z} {}

  [[nodiscard]] std::array<value_type, 3> const& top_point() const override {
    return _top;
  }
  [[nodiscard]] std::array<value_type, 3> const& bottom_point() const override {
    return _bottom;
  }

  [[nodiscard]] std::array<value_type, 3>& top_point() noexcept { return _top; }
  [[nodiscard]] std::array<value_type, 3>& bottom_point() noexcept { return _bottom; }

  [[nodiscard]] std::unique_ptr<bounding_box_base<value_type>> clone() const override {
    return std::make_unique<box_bounding<value_type>>(*this);
  }

private:
  std::array<value_type, 3> _top{};
  std::array<value_type, 3> _bottom{};
};

} // namespace rvegen
