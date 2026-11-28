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
            lookup_distribution(distributions, handler, "pos_x_dist"),
            lookup_distribution(distributions, handler, "pos_y_dist"),
            lookup_distribution(distributions, handler, "width_dist"),
            lookup_distribution(distributions, handler, "height_dist")) {
    this->read_metadata(handler);
  }

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("pos_x_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the rectangle centre x">>();
    s.template insert<std::string>("pos_y_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the rectangle centre y">>();
    s.template insert<std::string>("width_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the rectangle width">>();
    s.template insert<std::string>("height_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the rectangle height">>();
    s.template insert<std::string>("phase_name")
        .template add<numsim_core::description_label<"optional phase name stamped onto every produced shape (default: empty / unassigned)">>();
    s.template insert<std::string>("metadata")
        .template add<numsim_core::description_label<"optional JSON-encoded string of key/value pairs merged into every produced shape's info blob (e.g. \"{\"orientation_deg\": 42.5}\")">>();
    return s;
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
