#pragma once

#include <array>
#include <cstddef>

#include <Mathematics/AlignedBox.h>
#include <numsim-core/input_parameter_controller.h>

#include "../types.h"
#include "rectangle_bounding.h"
#include "shape_base.h"

namespace rvegen {

// rvegen rectangle: IS-A gte::AlignedBox<2, T> + IS-A shape_base.
//
// The GTE storage is min/max; the rvegen ctor accepts the more familiar
// (centre, width, height) and translates. Width / height / centre are
// derived accessors.
template <typename T = double>
class rectangle
    : public numsim_core::static_indexing<rectangle<T>, shape_base<T>>,
      public gte::AlignedBox<2, T> {
public:
  using value_type = T;
  using size_type = std::size_t;

  rectangle() = default;

  rectangle(T x, T y, T width, T height) {
    const auto hw = width  * T{0.5};
    const auto hh = height * T{0.5};
    this->min = {x - hw, y - hh};
    this->max = {x + hw, y + hh};
  }

  // Schema-driven ctor.
  explicit rectangle(parameter_handler_t const& handler)
      : rectangle(handler.template get<T>("x"),
                  handler.template get<T>("y"),
                  handler.template get<T>("width"),
                  handler.template get<T>("height")) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<T>("x").template add<numsim_core::is_required>()
        .units("m")
        .description("x-coordinate of the rectangle centre");
    s.template insert<T>("y").template add<numsim_core::is_required>()
        .units("m")
        .description("y-coordinate of the rectangle centre");
    s.template insert<T>("width").template add<numsim_core::is_required>()
        .min(0.0)
        .units("m")
        .description("rectangle extent along x (must be positive)");
    s.template insert<T>("height").template add<numsim_core::is_required>()
        .min(0.0)
        .units("m")
        .description("rectangle extent along y (must be positive)");
    return s;
  }

  [[nodiscard]] T operator()(size_type idx) const noexcept {
    return (this->max[idx] + this->min[idx]) * T{0.5};
  }

  [[nodiscard]] T width()  const noexcept { return this->max[0] - this->min[0]; }
  [[nodiscard]] T height() const noexcept { return this->max[1] - this->min[1]; }

  // shape_base contract.
  [[nodiscard]] T area()   const override { return width() * height(); }
  [[nodiscard]] T volume() const override { return T{0}; }

  [[nodiscard]] std::array<T, 3> max_expansion() const override {
    return {width()  * T{0.5}, height() * T{0.5}, T{0}};
  }

  [[nodiscard]] std::array<T, 3> get_middle_point() const override {
    return {(*this)(0), (*this)(1), T{0}};
  }
  void set_middle_point(std::array<T, 3> middle_point) override {
    const auto hw = width()  * T{0.5};
    const auto hh = height() * T{0.5};
    this->min = {middle_point[0] - hw, middle_point[1] - hh};
    this->max = {middle_point[0] + hw, middle_point[1] + hh};
  }

  [[nodiscard]] bool is_inside(std::array<T, 3> const& point) const override {
    return point[0] >= this->min[0] && point[0] <= this->max[0]
        && point[1] >= this->min[1] && point[1] <= this->max[1];
  }

  void make_bounding_box() override {
    auto box = std::make_unique<rectangle_bounding<T>>();
    box->top_point()    = {this->max[0], this->max[1], T{0}};
    box->bottom_point() = {this->min[0], this->min[1], T{0}};
    this->_bounding_box = std::move(box);
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> clone() const override {
    return std::make_unique<rectangle<T>>(*this);
  }

  [[nodiscard]] numsim_core::type_id shape_id() const noexcept override {
    return rectangle::m_id;
  }
};

} // namespace rvegen
