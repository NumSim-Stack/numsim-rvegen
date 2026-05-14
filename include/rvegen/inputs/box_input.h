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
            distribution_base<value_type>& depth) noexcept
      : _pos_x{pos_x}, _pos_y{pos_y}, _pos_z{pos_z},
        _width{width}, _height{height}, _depth{depth} {}

  box_input(parameter_handler_t const& handler,
            distribution_map_t<value_type> const& distributions)
      : box_input(
            *distributions.at(handler.template get<std::string>("pos_x_dist")),
            *distributions.at(handler.template get<std::string>("pos_y_dist")),
            *distributions.at(handler.template get<std::string>("pos_z_dist")),
            *distributions.at(handler.template get<std::string>("width_dist")),
            *distributions.at(handler.template get<std::string>("height_dist")),
            *distributions.at(handler.template get<std::string>("depth_dist"))) {
    this->read_metadata(handler);
  }

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("pos_x_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the box centre x">>();
    s.template insert<std::string>("pos_y_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the box centre y">>();
    s.template insert<std::string>("pos_z_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the box centre z">>();
    s.template insert<std::string>("width_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the box width">>();
    s.template insert<std::string>("height_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the box height">>();
    s.template insert<std::string>("depth_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the box depth">>();
    s.template insert<std::string>("phase_name")
        .template add<numsim_core::description_label<"optional phase name stamped onto every produced shape (default: empty / unassigned)">>();
    s.template insert<std::string>("metadata")
        .template add<numsim_core::description_label<"optional JSON-encoded string of key/value pairs merged into every produced shape's info blob (e.g. \"{\"orientation_deg\": 42.5}\")">>();
    return s;
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    auto shape = std::make_unique<box<value_type>>(
        _pos_x(), _pos_y(), _pos_z(),
        _width(), _height(), _depth());
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
