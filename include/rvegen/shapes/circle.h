#pragma once

#include <array>
#include <cstddef>
#include <numbers>

#include <tuple>

#include <Mathematics/Hypersphere.h>
#include <numsim-core/input_parameter_controller.h>

#include "../schema/field_list.h"
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

  // Declarative schema. Name + type + required-ness + annotations live
  // in a single declaration here; both `parameters()` and the
  // schema-driven ctor read from this list, so adding / renaming /
  // re-typing a field needs exactly one edit.
  using fields = field_list<
      field<"x", T, true,
            numsim_core::description_label<"x-coordinate of the circle centre">,
            numsim_core::unit_label<"m">>,
      field<"y", T, true,
            numsim_core::description_label<"y-coordinate of the circle centre">,
            numsim_core::unit_label<"m">>,
      field<"radius", T, true,
            numsim_core::description_label<"circle radius (must be positive)">,
            numsim_core::unit_label<"m">,
            min_only<T{0}>>>;

  // Schema-driven ctor — the rvegen addition. Delegates to the
  // tuple-taking helper which unpacks `fields::extract(handler)`.
  explicit circle(parameter_handler_t const& handler)
      : circle(fields::extract(handler)) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
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

private:
  // Tuple-taking helper used by the schema-driven ctor. Lets us delegate
  // through `fields::extract(handler)` without re-extracting per element.
  explicit circle(std::tuple<T, T, T> values)
      : circle(std::get<0>(values), std::get<1>(values),
               std::get<2>(values)) {}
};

} // namespace rvegen
