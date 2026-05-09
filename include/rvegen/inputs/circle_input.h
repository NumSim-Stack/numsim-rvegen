#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include <numsim-core/input_parameter_controller.h>

#include "../distributions/distribution_base.h"
#include "../shapes/circle.h"
#include "../types.h"
#include "shape_input_base.h"

namespace rvegen {

// Name → distribution lookup table used by JSON-driven input construction.
// The orchestrator builds this map first (one shared_ptr per registered
// distribution), then constructs inputs that look up their dependencies by
// name. shared_ptr is the owning side; inputs hold raw references whose
// stability depends on the map outliving the input.
template <typename T = double>
using distribution_map_t =
    std::unordered_map<std::string, std::shared_ptr<distribution_base<T>>>;

// Pulls position (x, y) and radius from injected distributions.
template <typename T = double>
class circle_input final : public shape_input_base<T> {
public:
  using value_type = T;

  // Direct ctor — used by tests and library users not driving via JSON.
  circle_input(distribution_base<value_type>& pos_x,
               distribution_base<value_type>& pos_y,
               distribution_base<value_type>& radius) noexcept
      : _pos_x{pos_x}, _pos_y{pos_y}, _radius{radius} {}

  // Schema-driven ctor — registry calls this. Looks up distributions by
  // name in the map. Requires the map to outlive the circle_input.
  circle_input(parameter_handler_t const& handler,
               distribution_map_t<value_type> const& distributions)
      : circle_input(
            *distributions.at(handler.template get<std::string>("pos_x_dist")),
            *distributions.at(handler.template get<std::string>("pos_y_dist")),
            *distributions.at(handler.template get<std::string>("radius_dist"))) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("pos_x_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the circle x-position">>();
    s.template insert<std::string>("pos_y_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the circle y-position">>();
    s.template insert<std::string>("radius_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the circle radius">>();
    return s;
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    return std::make_unique<circle<value_type>>(_pos_x(), _pos_y(), _radius());
  }

private:
  distribution_base<value_type>& _pos_x;
  distribution_base<value_type>& _pos_y;
  distribution_base<value_type>& _radius;
};

} // namespace rvegen
