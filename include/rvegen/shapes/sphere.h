#pragma once

#include <array>
#include <cstddef>
#include <numbers>

#include <Mathematics/Hypersphere.h>
#include <numsim-core/input_parameter_controller.h>

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

  // Schema-driven ctor — the rvegen addition.
  explicit sphere(parameter_handler_t const& handler)
      : sphere(handler.template get<T>("x"),
               handler.template get<T>("y"),
               handler.template get<T>("z"),
               handler.template get<T>("radius")) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<T>("x").template add<numsim_core::is_required>();
    s.template insert<T>("y").template add<numsim_core::is_required>();
    s.template insert<T>("z").template add<numsim_core::is_required>();
    s.template insert<T>("radius").template add<numsim_core::is_required>();
    return s;
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
};

} // namespace rvegen
