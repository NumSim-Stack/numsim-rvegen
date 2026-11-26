#pragma once

#include <memory>
#include <string>

#include <numsim-core/input_parameter_controller.h>

#include "../distributions/distribution_base.h"
#include "../shapes/ellipse.h"
#include "../types.h"
#include "circle_input.h"  // distribution_map_t alias
#include "shape_input_base.h"

namespace rvegen {

// 2D rotated ellipse input. Pulls 5 distributions: position (x, y),
// semi-axes (a, b), and rotation (radians).
template <typename T = double>
class ellipse_input final : public shape_input_base<T> {
public:
  using value_type = T;

  ellipse_input(distribution_base<value_type>& pos_x,
                distribution_base<value_type>& pos_y,
                distribution_base<value_type>& radius_a,
                distribution_base<value_type>& radius_b,
                distribution_base<value_type>& rotation)
      : _pos_x{pos_x}, _pos_y{pos_y},
        _radius_a{radius_a}, _radius_b{radius_b},
        _rotation{rotation} {
    this->require_positive_support(_radius_a, "ellipse_input", "radius_a");
    this->require_positive_support(_radius_b, "ellipse_input", "radius_b");
  }

  ellipse_input(parameter_handler_t const& handler,
                distribution_map_t<value_type> const& distributions)
      : ellipse_input(
            *distributions.at(fields::template value<"pos_x_dist">(handler)),
            *distributions.at(fields::template value<"pos_y_dist">(handler)),
            *distributions.at(fields::template value<"radius_a_dist">(handler)),
            *distributions.at(fields::template value<"radius_b_dist">(handler)),
            *distributions.at(fields::template value<"rotation_dist">(handler))) {
    this->read_metadata(handler);
  }

  using fields = field_list<
      field<"pos_x_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the ellipse centre x">>,
      field<"pos_y_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the ellipse centre y">>,
      field<"radius_a_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the ellipse semi-axis a">>,
      field<"radius_b_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the ellipse semi-axis b">>,
      field<"rotation_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the ellipse rotation (radians)">>,
      input_phase_name_field,
      input_metadata_field>;

  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
  }

  // Draw order: x, y, radius_a, radius_b, rotation.
  // The semi-axis distributions are truncated to (0, inf): non-positive draws
  // are resampled. See shape_input_base for the constructor check
  // and the redraw backstop.
  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    const value_type x = _pos_x();
    const value_type y = _pos_y();
    const value_type a = this->draw_positive(_radius_a, "ellipse_input", "radius_a");
    const value_type b = this->draw_positive(_radius_b, "ellipse_input", "radius_b");
    const value_type rot = _rotation();
    auto shape = std::make_unique<ellipse<value_type>>(x, y, a, b, rot);
    this->tag(*shape);
    return shape;
  }

private:
  distribution_base<value_type>& _pos_x;
  distribution_base<value_type>& _pos_y;
  distribution_base<value_type>& _radius_a;
  distribution_base<value_type>& _radius_b;
  distribution_base<value_type>& _rotation;
};

} // namespace rvegen
