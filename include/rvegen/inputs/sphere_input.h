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
               distribution_base<value_type>& radius)
      : _pos_x{pos_x}, _pos_y{pos_y}, _pos_z{pos_z}, _radius{radius} {
    this->require_positive_support(_radius, "sphere_input", "radius");
  }

  // Schema-driven ctor.
  sphere_input(parameter_handler_t const& handler,
               distribution_map_t<value_type> const& distributions)
      : sphere_input(
            *distributions.at(fields::template value<"pos_x_dist">(handler)),
            *distributions.at(fields::template value<"pos_y_dist">(handler)),
            *distributions.at(fields::template value<"pos_z_dist">(handler)),
            *distributions.at(fields::template value<"radius_dist">(handler))) {
    this->read_metadata(handler);
  }

  using fields = field_list<
      field<"pos_x_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the sphere x-position">>,
      field<"pos_y_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the sphere y-position">>,
      field<"pos_z_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the sphere z-position">>,
      field<"radius_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the sphere radius">>,
      input_phase_name_field,
      input_metadata_field>;

  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
  }

  // Draw order: x, y, z, radius.
  // The radius distribution is truncated to (0, inf): non-positive draws
  // are resampled. See shape_input_base for the constructor check
  // and the redraw backstop.
  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    const value_type x = _pos_x();
    const value_type y = _pos_y();
    const value_type z = _pos_z();
    const value_type r = this->draw_positive(_radius, "sphere_input", "radius");
    auto shape = std::make_unique<sphere<value_type>>(x, y, z, r);
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
