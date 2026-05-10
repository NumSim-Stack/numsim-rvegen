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
                distribution_base<value_type>& rotation) noexcept
      : _pos_x{pos_x}, _pos_y{pos_y},
        _radius_a{radius_a}, _radius_b{radius_b},
        _rotation{rotation} {}

  ellipse_input(parameter_handler_t const& handler,
                distribution_map_t<value_type> const& distributions)
      : ellipse_input(
            *distributions.at(handler.template get<std::string>("pos_x_dist")),
            *distributions.at(handler.template get<std::string>("pos_y_dist")),
            *distributions.at(handler.template get<std::string>("radius_a_dist")),
            *distributions.at(handler.template get<std::string>("radius_b_dist")),
            *distributions.at(handler.template get<std::string>("rotation_dist"))) {
    this->read_metadata(handler);
  }

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("pos_x_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the ellipse centre x">>();
    s.template insert<std::string>("pos_y_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the ellipse centre y">>();
    s.template insert<std::string>("radius_a_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the ellipse semi-axis a">>();
    s.template insert<std::string>("radius_b_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the ellipse semi-axis b">>();
    s.template insert<std::string>("rotation_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the ellipse rotation (radians)">>();
    s.template insert<std::string>("phase_name")
        .template add<numsim_core::description_label<"optional phase name stamped onto every produced shape (default: empty / unassigned)">>();
    s.template insert<std::string>("metadata")
        .template add<numsim_core::description_label<"optional JSON-encoded string of key/value pairs merged into every produced shape's info blob (e.g. \"{\"orientation_deg\": 42.5}\")">>();
    return s;
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    auto shape = std::make_unique<ellipse<value_type>>(
        _pos_x(), _pos_y(), _radius_a(), _radius_b(), _rotation());
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
