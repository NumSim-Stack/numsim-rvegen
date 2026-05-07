#pragma once

#include <array>
#include <memory>

#include "bounding_box_base.h"

namespace rvegen {

// 2D AABB. Stored as 3-component arrays with z = 0 to fit the
// bounding_box_base<T> contract.
template <typename T>
class rectangle_bounding final : public bounding_box_base<T> {
public:
  using value_type = T;

  rectangle_bounding() = default;

  rectangle_bounding(std::array<value_type, 2> top,
                     std::array<value_type, 2> bottom) noexcept
      : _top{top[0], top[1], 0}, _bottom{bottom[0], bottom[1], 0} {}

  rectangle_bounding(value_type top_x, value_type top_y,
                     value_type bottom_x, value_type bottom_y) noexcept
      : _top{top_x, top_y, 0}, _bottom{bottom_x, bottom_y, 0} {}

  [[nodiscard]] std::array<value_type, 3> const& top_point() const override {
    return _top;
  }
  [[nodiscard]] std::array<value_type, 3> const& bottom_point() const override {
    return _bottom;
  }

  [[nodiscard]] std::array<value_type, 3>& top_point() noexcept { return _top; }
  [[nodiscard]] std::array<value_type, 3>& bottom_point() noexcept { return _bottom; }

  [[nodiscard]] std::unique_ptr<bounding_box_base<value_type>> clone() const override {
    return std::make_unique<rectangle_bounding<value_type>>(*this);
  }

private:
  std::array<value_type, 3> _top{};
  std::array<value_type, 3> _bottom{};
};

} // namespace rvegen
