#pragma once

#include <memory>
#include <stdexcept>
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

// Resolve the distribution named by schema key `key`. An unknown name
// raises an error naming both the key and the missing distribution
// (a std::out_of_range, as unordered_map::at threw, but with a message).
template <typename T>
[[nodiscard]] distribution_base<T>&
lookup_distribution(distribution_map_t<T> const& distributions,
                    parameter_handler_t const& handler, char const* key) {
  auto const name = handler.template get<std::string>(key);
  auto const it = distributions.find(name);
  if (it == distributions.end() || !it->second)
    throw std::out_of_range{"shape input: key '" + std::string{key} +
                             "' refers to unknown distribution '" + name +
                             "'"};
  return *it->second;
}

// Pulls position (x, y) and radius from injected distributions.
template <typename T = double>
class circle_input final : public shape_input_base<T> {
public:
  using value_type = T;

  // Direct ctor — used by tests and library users not driving via JSON.
  circle_input(distribution_base<value_type>& pos_x,
               distribution_base<value_type>& pos_y,
               distribution_base<value_type>& radius)
      : _pos_x{pos_x}, _pos_y{pos_y}, _radius{radius} {
    this->require_positive_support(_radius, "circle_input", "radius");
  }

  // Schema-driven ctor — registry calls this. Looks up distributions by
  // name in the map. Requires the map to outlive the circle_input.
  circle_input(parameter_handler_t const& handler,
               distribution_map_t<value_type> const& distributions)
      : circle_input(
            lookup_distribution(distributions, handler, "pos_x_dist"),
            lookup_distribution(distributions, handler, "pos_y_dist"),
            lookup_distribution(distributions, handler, "radius_dist")) {
    this->read_metadata(handler);
  }

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
    s.template insert<std::string>("phase_name")
        .template add<numsim_core::description_label<"optional phase name stamped onto every produced shape (default: empty / unassigned)">>();
    s.template insert<std::string>("metadata")
        .template add<numsim_core::description_label<"optional JSON-encoded string of key/value pairs merged into every produced shape's info blob (e.g. \"{\"orientation_deg\": 42.5}\")">>();
    return s;
  }

  // Draw order: x, y, radius.
  // The radius distribution is truncated to (0, inf): non-positive draws
  // are resampled. See shape_input_base for the constructor check
  // and the redraw backstop.
  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    const value_type x = _pos_x();
    const value_type y = _pos_y();
    const value_type r = this->draw_positive(_radius, "circle_input", "radius");
    auto shape = std::make_unique<circle<value_type>>(x, y, r);
    this->tag(*shape);
    return shape;
  }

private:
  distribution_base<value_type>& _pos_x;
  distribution_base<value_type>& _pos_y;
  distribution_base<value_type>& _radius;
};

} // namespace rvegen
