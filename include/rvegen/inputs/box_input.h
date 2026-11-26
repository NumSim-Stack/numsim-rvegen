#pragma once

#include <memory>
#include <string>

#include <numsim-core/input_parameter_controller.h>

#include "../distributions/distribution_base.h"
#include "../shapes/box.h"
#include "../types.h"
#include "circle_input.h"  // distribution_map_t alias
#include "shape_input_base.h"

namespace rvegen {

// 3D axis-aligned box input. Distributions for centre (x, y, z) and
// extents (width, height, depth).
template <typename T = double>
class box_input final : public shape_input_base<T> {
public:
  using value_type = T;

  box_input(distribution_base<value_type>& pos_x,
            distribution_base<value_type>& pos_y,
            distribution_base<value_type>& pos_z,
            distribution_base<value_type>& width,
            distribution_base<value_type>& height,
            distribution_base<value_type>& depth)
      : _pos_x{pos_x}, _pos_y{pos_y}, _pos_z{pos_z},
        _width{width}, _height{height}, _depth{depth} {
    this->require_positive_support(_width, "box_input", "width");
    this->require_positive_support(_height, "box_input", "height");
    this->require_positive_support(_depth, "box_input", "depth");
  }

  box_input(parameter_handler_t const& handler,
            distribution_map_t<value_type> const& distributions)
      : box_input(
            *distributions.at(fields::template value<"pos_x_dist">(handler)),
            *distributions.at(fields::template value<"pos_y_dist">(handler)),
            *distributions.at(fields::template value<"pos_z_dist">(handler)),
            *distributions.at(fields::template value<"width_dist">(handler)),
            *distributions.at(fields::template value<"height_dist">(handler)),
            *distributions.at(fields::template value<"depth_dist">(handler))) {
    this->read_metadata(handler);
  }

  using fields = field_list<
      field<"pos_x_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the box centre x">>,
      field<"pos_y_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the box centre y">>,
      field<"pos_z_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the box centre z">>,
      field<"width_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the box width">>,
      field<"height_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the box height">>,
      field<"depth_dist", std::string, true,
            numsim_core::description_label<"name of a distribution sampled for the box depth">>,
      input_phase_name_field,
      input_metadata_field>;

  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
  }

  // Draw order: x, y, z, width, height, depth.
  // The extent distributions are truncated to (0, inf): non-positive draws
  // are resampled. See shape_input_base for the constructor check
  // and the redraw backstop.
  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    const value_type x = _pos_x();
    const value_type y = _pos_y();
    const value_type z = _pos_z();
    const value_type w = this->draw_positive(_width, "box_input", "width");
    const value_type h = this->draw_positive(_height, "box_input", "height");
    const value_type d = this->draw_positive(_depth, "box_input", "depth");
    auto shape = std::make_unique<box<value_type>>(x, y, z, w, h, d);
    this->tag(*shape);
    return shape;
  }

private:
  distribution_base<value_type>& _pos_x;
  distribution_base<value_type>& _pos_y;
  distribution_base<value_type>& _pos_z;
  distribution_base<value_type>& _width;
  distribution_base<value_type>& _height;
  distribution_base<value_type>& _depth;
};

} // namespace rvegen
