#pragma once

#include <memory>
#include <string>

#include <numsim-core/input_parameter_controller.h>

#include "../distributions/distribution_base.h"
#include "../shapes/rectangle.h"
#include "../types.h"
#include "circle_input.h"  // distribution_map_t alias
#include "shape_input_base.h"

namespace rvegen {

// 2D axis-aligned rectangle input. Distributions for centre (x, y) and
// extents (width, height).
template <typename T = double>
class rectangle_input final : public shape_input_base<T> {
public:
  using value_type = T;

  rectangle_input(distribution_base<value_type>& pos_x,
                  distribution_base<value_type>& pos_y,
                  distribution_base<value_type>& width,
                  distribution_base<value_type>& height)
      : _pos_x{pos_x}, _pos_y{pos_y}, _width{width}, _height{height} {
    this->require_positive_support(_width, "rectangle_input", "width");
    this->require_positive_support(_height, "rectangle_input", "height");
  }

  rectangle_input(parameter_handler_t const& handler,
                  distribution_map_t<value_type> const& distributions)
      : rectangle_input(
            *distributions.at(fields::template value<"pos_x_dist">(handler)),
            *distributions.at(fields::template value<"pos_y_dist">(handler)),
            *distributions.at(fields::template value<"width_dist">(handler)),
            *distributions.at(fields::template value<"height_dist">(handler))) {
    this->read_metadata(handler);
  }

  using fields = field_list<
      field<"pos_x_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the rectangle centre x">>,
      field<"pos_y_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the rectangle centre y">>,
      field<"width_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the rectangle width">>,
      field<"height_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the rectangle height">>,
      input_phase_name_field,
      input_metadata_field>;

  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
  }

  // Draw order: x, y, width, height.
  // The width and height distributions are truncated to (0, inf): non-positive draws
  // are resampled. See shape_input_base for the constructor check
  // and the redraw backstop.
  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    const value_type x = _pos_x();
    const value_type y = _pos_y();
    const value_type w = this->draw_positive(_width, "rectangle_input", "width");
    const value_type h = this->draw_positive(_height, "rectangle_input", "height");
    auto shape = std::make_unique<rectangle<value_type>>(x, y, w, h);
    this->tag(*shape);
    return shape;
  }

private:
  distribution_base<value_type>& _pos_x;
  distribution_base<value_type>& _pos_y;
  distribution_base<value_type>& _width;
  distribution_base<value_type>& _height;
};

} // namespace rvegen
