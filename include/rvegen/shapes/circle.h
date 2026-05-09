#pragma once

#include <array>
#include <cstddef>
#include <numbers>

#include <Mathematics/Hypersphere.h>
#include <numsim-core/input_parameter_controller.h>

#include "../types.h"
#include "rectangle_bounding.h"
#include "shape_base.h"

namespace rvegen {

// rvegen circle: IS-A gte::Circle2<T> (Hypersphere<2, T>) + IS-A shape_base.
//
// Storage and field layout come from gte::Circle2 — public members `center`
// (gte::Vector2<T>) and `radius` (T). This means rvegen::circle can be
// passed directly to any GTE intersection query that takes Circle2.
//
// rvegen adds: shape_base virtual contract (area/volume/is_inside/etc.),
// a static `parameters()` schema, and a `(parameter_handler_t const&)`
// ctor for JSON-driven construction.
template <typename T = double>
class circle
    : public numsim_core::static_indexing<circle<T>, shape_base<T>>,
      public gte::Circle2<T> {
public:
  using value_type = T;
  using size_type = std::size_t;

  circle() = default;

  circle(T x, T y, T r) {
    this->center = {x, y};
    this->radius = r;
  }

  // Schema-driven ctor — the rvegen addition.
  explicit circle(parameter_handler_t const& handler)
      : circle(handler.template get<T>("x"),
               handler.template get<T>("y"),
               handler.template get<T>("radius")) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<T>("x").template add<numsim_core::is_required>()
        .units("m")
        .description("x-coordinate of the circle centre");
    s.template insert<T>("y").template add<numsim_core::is_required>()
        .units("m")
        .description("y-coordinate of the circle centre");
    s.template insert<T>("radius").template add<numsim_core::is_required>()
        .min(0.0)
        .units("m")
        .description("circle radius (must be positive)");
    return s;
  }

  // Convenience subscript access to centre coordinates.
  [[nodiscard]] T operator()(size_type idx) const noexcept {
    return this->center[idx];
  }
  [[nodiscard]] T& operator()(size_type idx) noexcept {
    return this->center[idx];
  }

  // shape_base contract.
  [[nodiscard]] T area() const override {
    return this->radius * this->radius * std::numbers::pi_v<T>;
  }
  [[nodiscard]] T volume() const override { return T{0}; }

  [[nodiscard]] std::array<T, 3> max_expansion() const override {
    return {this->radius, this->radius, T{0}};
  }

  [[nodiscard]] std::array<T, 3> get_middle_point() const override {
    return {this->center[0], this->center[1], T{0}};
  }
  void set_middle_point(std::array<T, 3> middle_point) override {
    this->center[0] = middle_point[0];
    this->center[1] = middle_point[1];
  }

  [[nodiscard]] bool is_inside(std::array<T, 3> const& point) const override {
    const auto dx = point[0] - this->center[0];
    const auto dy = point[1] - this->center[1];
    return dx * dx + dy * dy <= this->radius * this->radius;
  }

  void make_bounding_box() override {
    auto box = std::make_unique<rectangle_bounding<T>>();
    box->top_point()    = {this->center[0] + this->radius,
                           this->center[1] + this->radius, T{0}};
    box->bottom_point() = {this->center[0] - this->radius,
                           this->center[1] - this->radius, T{0}};
    this->_bounding_box = std::move(box);
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> clone() const override {
    return std::make_unique<circle<T>>(*this);
  }

  [[nodiscard]] numsim_core::type_id shape_id() const noexcept override {
    return circle::m_id;
  }
};

} // namespace rvegen
