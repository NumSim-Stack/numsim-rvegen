#pragma once

#include <array>
#include <cstddef>
#include <numbers>

#include <tuple>

#include <Mathematics/Hypersphere.h>
#include <numsim-core/input_parameter_controller.h>

#include "../schema/field_list.h"
#include "../types.h"
#include "box_bounding.h"
#include "shape_base.h"

namespace rvegen {

// rvegen sphere: IS-A gte::Sphere3<T> (Hypersphere<3, T>) + IS-A shape_base.
template <typename T = double>
class sphere
    : public numsim_core::static_indexing<sphere<T>, shape_base<T>>,
      public gte::Sphere3<T> {
public:
  using value_type = T;
  using size_type = std::size_t;

  sphere() = default;

  sphere(T x, T y, T z, T r) {
    this->center = {x, y, z};
    this->radius = r;
  }

  // Declarative schema — see circle.h for the migration pattern.
  using fields = field_list<
      field<"x", T, true,
            numsim_core::description_label<"x-coordinate of the sphere centre">,
            numsim_core::unit_label<"m">>,
      field<"y", T, true,
            numsim_core::description_label<"y-coordinate of the sphere centre">,
            numsim_core::unit_label<"m">>,
      field<"z", T, true,
            numsim_core::description_label<"z-coordinate of the sphere centre">,
            numsim_core::unit_label<"m">>,
      field<"radius", T, true,
            numsim_core::description_label<"sphere radius (must be positive)">,
            numsim_core::unit_label<"m">,
            min_only<T{0}>>>;

  explicit sphere(parameter_handler_t const& handler)
      : sphere(fields::extract(handler)) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
  }

  [[nodiscard]] T operator()(size_type idx) const noexcept {
    return this->center[idx];
  }
  [[nodiscard]] T& operator()(size_type idx) noexcept {
    return this->center[idx];
  }

  // shape_base contract.
  [[nodiscard]] T area() const override { return T{0}; }
  [[nodiscard]] T volume() const override {
    return T{4} / T{3} * std::numbers::pi_v<T>
         * this->radius * this->radius * this->radius;
  }

  [[nodiscard]] std::array<T, 3> max_expansion() const override {
    return {this->radius, this->radius, this->radius};
  }

  [[nodiscard]] std::array<T, 3> get_middle_point() const override {
    return {this->center[0], this->center[1], this->center[2]};
  }
  void set_middle_point(std::array<T, 3> middle_point) override {
    this->center[0] = middle_point[0];
    this->center[1] = middle_point[1];
    this->center[2] = middle_point[2];
  }

  [[nodiscard]] bool is_inside(std::array<T, 3> const& point) const override {
    const auto dx = point[0] - this->center[0];
    const auto dy = point[1] - this->center[1];
    const auto dz = point[2] - this->center[2];
    return dx * dx + dy * dy + dz * dz <= this->radius * this->radius;
  }

  void make_bounding_box() override {
    auto box = std::make_unique<box_bounding<T>>();
    box->top_point()    = {this->center[0] + this->radius,
                           this->center[1] + this->radius,
                           this->center[2] + this->radius};
    box->bottom_point() = {this->center[0] - this->radius,
                           this->center[1] - this->radius,
                           this->center[2] - this->radius};
    this->_bounding_box = std::move(box);
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> clone() const override {
    return std::make_unique<sphere<T>>(*this);
  }

  [[nodiscard]] numsim_core::type_id shape_id() const noexcept override {
    return sphere::m_id;
  }

private:
  explicit sphere(std::tuple<T, T, T, T> v)
      : sphere(std::get<0>(v), std::get<1>(v), std::get<2>(v),
               std::get<3>(v)) {}
};

} // namespace rvegen
