#pragma once

// Registry adapter for `polyline_tube_oriented_input` (#84).
//
// The input registry's ctor signature is fixed to
// `(parameter_handler_t const&, distribution_map_t const&)`, so a
// direction distribution (a different base class) cannot be passed
// next to the scalar map. Rather than widen every input's ctor, direction
// distributions travel in the SAME map wrapped in a
// `direction_distribution_holder`, a `distribution_base<T>` whose scalar
// operator() is unusable (it throws) and which carries the real
// `direction_distribution_base` via shared_ptr. The CLI/pipeline builds
// the "direction_distributions" config section into holders; this
// adapter unwraps its `direction_dist` entry.
//
// Registered name: "polyline_tube_oriented_input".

#include <memory>
#include <stdexcept>
#include <string>

#include <numsim-core/input_parameter_controller.h>

#include "../distributions/direction_distribution_base.h"
#include "../distributions/distribution_base.h"
#include "../types.h"
#include "circle_input.h"   // distribution_map_t
#include "polyline_tube_oriented_input.h"
#include "shape_input_base.h"

namespace rvegen {

template <typename T = double>
class direction_distribution_holder final : public distribution_base<T> {
public:
  explicit direction_distribution_holder(
      std::shared_ptr<direction_distribution_base<T>> dir)
      : _dir{std::move(dir)} {}

  [[nodiscard]] T operator()() override {
    throw std::logic_error{
        "direction distribution used where a scalar distribution is expected"};
  }

  [[nodiscard]] direction_distribution_base<T>& direction() const {
    return *_dir;
  }

private:
  std::shared_ptr<direction_distribution_base<T>> _dir;
};

template <typename T = double>
class polyline_tube_oriented_registered_input final
    : public shape_input_base<T> {
public:
  polyline_tube_oriented_registered_input(
      parameter_handler_t const& handler,
      distribution_map_t<T> const& d)
      : _inner{lookup_distribution(d, handler, "position_x_dist"),
               lookup_distribution(d, handler, "position_y_dist"),
               lookup_distribution(d, handler, "position_z_dist"),
               lookup_direction(d, handler.template get<std::string>("direction_dist")),
               lookup_distribution(d, handler, "length_dist"),
               lookup_distribution(d, handler, "radius_dist")} {
    this->read_metadata(handler);
  }

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    for (auto const* key : {"position_x_dist", "position_y_dist",
                            "position_z_dist", "length_dist", "radius_dist"}) {
      s.template insert<std::string>(key)
          .template add<numsim_core::is_required>()
          .template add<numsim_core::description_label<"name of a scalar distribution">>();
    }
    s.template insert<std::string>("direction_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of an entry in the top-level \"direction_distributions\" section (e.g. a bingham)">>();
    s.template insert<std::string>("phase_name")
        .template add<numsim_core::description_label<"optional phase name stamped onto every produced shape (default: empty / unassigned)">>();
    s.template insert<std::string>("metadata")
        .template add<numsim_core::description_label<"optional key/value pairs merged into every produced shape's info blob: a JSON object, or the same object as a JSON-encoded string">>();
    return s;
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    auto shape = _inner.new_shape();
    this->tag(*shape);
    return shape;
  }

private:
  static direction_distribution_base<T>&
  lookup_direction(distribution_map_t<T> const& d, std::string const& name) {
    auto const it = d.find(name);
    if (it == d.end())
      throw std::runtime_error{"polyline_tube_oriented_input: unknown "
                               "direction distribution '" + name + "'"};
    auto* holder =
        dynamic_cast<direction_distribution_holder<T>*>(it->second.get());
    if (!holder)
      throw std::runtime_error{"polyline_tube_oriented_input: '" + name +
                               "' is a scalar distribution, not a direction "
                               "distribution"};
    return holder->direction();
  }

  polyline_tube_oriented_input<T> _inner;
};

} // namespace rvegen
