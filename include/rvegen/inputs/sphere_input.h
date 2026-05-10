#pragma once

#include <memory>
#include <string>

#include <numsim-core/input_parameter_controller.h>

#include "../distributions/distribution_base.h"
#include "../shapes/sphere.h"
#include "../types.h"
#include "circle_input.h"  // distribution_map_t alias
#include "shape_input_base.h"

namespace rvegen {

// 3D analogue of circle_input. Pulls position (x, y, z) and radius from
// injected distributions referenced by name in the JSON config.
template <typename T = double>
class sphere_input final : public shape_input_base<T> {
public:
  using value_type = T;

  sphere_input(distribution_base<value_type>& pos_x,
               distribution_base<value_type>& pos_y,
               distribution_base<value_type>& pos_z,
               distribution_base<value_type>& radius) noexcept
      : _pos_x{pos_x}, _pos_y{pos_y}, _pos_z{pos_z}, _radius{radius} {}

  // Schema-driven ctor.
  sphere_input(parameter_handler_t const& handler,
               distribution_map_t<value_type> const& distributions)
      : sphere_input(
            *distributions.at(handler.template get<std::string>("pos_x_dist")),
            *distributions.at(handler.template get<std::string>("pos_y_dist")),
            *distributions.at(handler.template get<std::string>("pos_z_dist")),
            *distributions.at(handler.template get<std::string>("radius_dist"))) {
    if (handler.contains("phase_name")) {
      this->set_phase_name(handler.template get<std::string>("phase_name"));
    }
  }

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("pos_x_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the sphere x-position">>();
    s.template insert<std::string>("pos_y_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the sphere y-position">>();
    s.template insert<std::string>("pos_z_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the sphere z-position">>();
    s.template insert<std::string>("radius_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the sphere radius">>();
    s.template insert<std::string>("phase_name")
        .template add<numsim_core::description_label<"optional phase name stamped onto every produced shape (default: empty / unassigned)">>();
    return s;
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    auto shape = std::make_unique<sphere<value_type>>(_pos_x(), _pos_y(),
                                                       _pos_z(), _radius());
    this->tag(*shape);
    return shape;
  }

private:
  distribution_base<value_type>& _pos_x;
  distribution_base<value_type>& _pos_y;
  distribution_base<value_type>& _pos_z;
  distribution_base<value_type>& _radius;
};

} // namespace rvegen
