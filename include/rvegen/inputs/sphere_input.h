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
            lookup_distribution(distributions, handler, "pos_x_dist"),
            lookup_distribution(distributions, handler, "pos_y_dist"),
            lookup_distribution(distributions, handler, "pos_z_dist"),
            lookup_distribution(distributions, handler, "radius_dist")) {
    this->read_metadata(handler);
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
    s.template insert<std::string>("metadata")
        .template add<numsim_core::description_label<"optional key/value pairs merged into every produced shape's info blob: a JSON object (e.g. {\"orientation_deg\": 42.5}), or the same object as a JSON-encoded string">>();
    return s;
  }

  // Draw order: x, y, z, radius.
  // The radius distribution is truncated to (0, inf): non-positive draws
  // are resampled. See shape_input_base for the constructor check
  // and the redraw backstop.
  [[nodiscard]] int dimension() const noexcept override { return 3; }

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
