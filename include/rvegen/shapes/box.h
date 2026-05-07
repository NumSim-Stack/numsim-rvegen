#pragma once

#include <array>
#include <cstddef>

#include <Mathematics/AlignedBox.h>
#include <numsim-core/input_parameter_controller.h>

#include "../types.h"
#include "box_bounding.h"
#include "shape_base.h"

namespace rvegen {

// rvegen box: IS-A gte::AlignedBox<3, T> + IS-A shape_base.
template <typename T = double>
class box : public shape_base<T>, public gte::AlignedBox<3, T> {
public:
  using value_type = T;
  using size_type = std::size_t;

  box() = default;

  box(T x, T y, T z, T width, T height, T depth) {
    const auto hw = width  * T{0.5};
    const auto hh = height * T{0.5};
    const auto hd = depth  * T{0.5};
    this->min = {x - hw, y - hh, z - hd};
    this->max = {x + hw, y + hh, z + hd};
  }

  explicit box(parameter_handler_t const& handler)
      : box(handler.template get<T>("x"),
            handler.template get<T>("y"),
            handler.template get<T>("z"),
            handler.template get<T>("width"),
            handler.template get<T>("height"),
            handler.template get<T>("depth")) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<T>("x").template add<numsim_core::is_required>();
    s.template insert<T>("y").template add<numsim_core::is_required>();
    s.template insert<T>("z").template add<numsim_core::is_required>();
    s.template insert<T>("width").template add<numsim_core::is_required>();
    s.template insert<T>("height").template add<numsim_core::is_required>();
    s.template insert<T>("depth").template add<numsim_core::is_required>();
    return s;
  }

  [[nodiscard]] T operator()(size_type idx) const noexcept {
    return (this->max[idx] + this->min[idx]) * T{0.5};
  }

  [[nodiscard]] T width()  const noexcept { return this->max[0] - this->min[0]; }
  [[nodiscard]] T height() const noexcept { return this->max[1] - this->min[1]; }
  [[nodiscard]] T depth()  const noexcept { return this->max[2] - this->min[2]; }

  [[nodiscard]] T area()   const override { return T{0}; }
  [[nodiscard]] T volume() const override { return width() * height() * depth(); }

  [[nodiscard]] std::array<T, 3> max_expansion() const override {
    return {width() * T{0.5}, height() * T{0.5}, depth() * T{0.5}};
  }

  [[nodiscard]] std::array<T, 3> get_middle_point() const override {
    return {(*this)(0), (*this)(1), (*this)(2)};
  }
  void set_middle_point(std::array<T, 3> middle_point) override {
    const auto hw = width()  * T{0.5};
    const auto hh = height() * T{0.5};
    const auto hd = depth()  * T{0.5};
    this->min = {middle_point[0] - hw, middle_point[1] - hh, middle_point[2] - hd};
    this->max = {middle_point[0] + hw, middle_point[1] + hh, middle_point[2] + hd};
  }

  [[nodiscard]] bool is_inside(std::array<T, 3> const& point) const override {
    return point[0] >= this->min[0] && point[0] <= this->max[0]
        && point[1] >= this->min[1] && point[1] <= this->max[1]
        && point[2] >= this->min[2] && point[2] <= this->max[2];
  }

  void make_bounding_box() override {
    auto bb = std::make_unique<box_bounding<T>>();
    bb->top_point()    = {this->max[0], this->max[1], this->max[2]};
    bb->bottom_point() = {this->min[0], this->min[1], this->min[2]};
    this->_bounding_box = std::move(bb);
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> clone() const override {
    return std::make_unique<box<T>>(*this);
  }
};

} // namespace rvegen
