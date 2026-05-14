#pragma once

#include <array>
#include <cstddef>

#include <tuple>

#include <Mathematics/AlignedBox.h>
#include <numsim-core/input_parameter_controller.h>

#include "../schema/field_list.h"
#include "../types.h"
#include "box_bounding.h"
#include "shape_base.h"

namespace rvegen {

// rvegen box: IS-A gte::AlignedBox<3, T> + IS-A shape_base.
template <typename T = double>
class box
    : public numsim_core::static_indexing<box<T>, shape_base<T>>,
      public gte::AlignedBox<3, T> {
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

  using fields = field_list<
      field<"x", T, true,
            numsim_core::description_label<"x-coordinate of the box centre">,
            numsim_core::unit_label<"m">>,
      field<"y", T, true,
            numsim_core::description_label<"y-coordinate of the box centre">,
            numsim_core::unit_label<"m">>,
      field<"z", T, true,
            numsim_core::description_label<"z-coordinate of the box centre">,
            numsim_core::unit_label<"m">>,
      field<"width", T, true,
            numsim_core::description_label<"box extent along x (must be positive)">,
            numsim_core::unit_label<"m">,
            min_only<T{0}>>,
      field<"height", T, true,
            numsim_core::description_label<"box extent along y (must be positive)">,
            numsim_core::unit_label<"m">,
            min_only<T{0}>>,
      field<"depth", T, true,
            numsim_core::description_label<"box extent along z (must be positive)">,
            numsim_core::unit_label<"m">,
            min_only<T{0}>>>;

  explicit box(parameter_handler_t const& handler)
      : box(fields::extract(handler)) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
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

  [[nodiscard]] numsim_core::type_id shape_id() const noexcept override {
    return box::m_id;
  }

private:
  explicit box(std::tuple<T, T, T, T, T, T> v)
      : box(std::get<0>(v), std::get<1>(v), std::get<2>(v),
            std::get<3>(v), std::get<4>(v), std::get<5>(v)) {}
};

} // namespace rvegen
