#pragma once

// JSON-driven input that samples a straight 2-point polyline_tube
// between two endpoint distributions plus a radius distribution.
//
// Why 2-point and not arbitrary-N:
//   The existing rvegen input pipeline pulls a fixed bag of named
//   `distribution_base<T>&` references from `parameter_handler_t` —
//   no "list of N distributions" mechanism today. A 2-point straight
//   tube is the smallest polyline_tube that's useful, fits the
//   existing input contract, and gets the polyline_tube primitive
//   exercised end-to-end via the standard config + generator +
//   writer pipeline.
//
//   Curved / N-segment fibres come in follow-up PRs alongside a
//   `weave_generator` that lays out interlocked over/under fibres on
//   a textile grid — it'll need a different input shape (per-fibre
//   procedural centerline construction) that doesn't fit the
//   random-sampling input pattern at all.
//
// What this header lands today (phase 2 of #3, after the polyline_tube
// primitive in PR #16):
//   * `polyline_tube_input<T>` — schema-driven 7-field input
//     (start_xyz, end_xyz, radius).
//   * Registration as `"polyline_tube_input"` in
//     `register_inputs.h`.
//   * `new_shape()` produces a 2-point polyline_tube.

#include <memory>
#include <string>
#include <vector>

#include <numsim-core/input_parameter_controller.h>

#include "../distributions/distribution_base.h"
#include "../shapes/polyline_tube.h"
#include "../types.h"
#include "circle_input.h"   // distribution_map_t alias
#include "shape_input_base.h"

namespace rvegen {

template <typename T = double>
class polyline_tube_input final : public shape_input_base<T> {
public:
  using value_type = T;

  polyline_tube_input(distribution_base<value_type>& sx,
                      distribution_base<value_type>& sy,
                      distribution_base<value_type>& sz,
                      distribution_base<value_type>& ex,
                      distribution_base<value_type>& ey,
                      distribution_base<value_type>& ez,
                      distribution_base<value_type>& radius)
      : _sx{sx}, _sy{sy}, _sz{sz}, _ex{ex}, _ey{ey}, _ez{ez},
        _radius{radius} {
    this->require_positive_support(_radius, "polyline_tube_input", "radius");
  }

  // Schema-driven ctor.
  polyline_tube_input(parameter_handler_t const& handler,
                      distribution_map_t<value_type> const& distributions)
      : polyline_tube_input(
            lookup_distribution(distributions, handler, "start_x_dist"),
            lookup_distribution(distributions, handler, "start_y_dist"),
            lookup_distribution(distributions, handler, "start_z_dist"),
            lookup_distribution(distributions, handler, "end_x_dist"),
            lookup_distribution(distributions, handler, "end_y_dist"),
            lookup_distribution(distributions, handler, "end_z_dist"),
            lookup_distribution(distributions, handler, "radius_dist")) {
    this->read_metadata(handler);
  }

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("start_x_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube start x-coordinate">>();
    s.template insert<std::string>("start_y_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube start y-coordinate">>();
    s.template insert<std::string>("start_z_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube start z-coordinate">>();
    s.template insert<std::string>("end_x_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube end x-coordinate">>();
    s.template insert<std::string>("end_y_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube end y-coordinate">>();
    s.template insert<std::string>("end_z_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube end z-coordinate">>();
    s.template insert<std::string>("radius_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube radius">>();
    s.template insert<std::string>("phase_name")
        .template add<numsim_core::description_label<"optional phase name stamped onto every produced shape (default: empty / unassigned)">>();
    s.template insert<std::string>("metadata")
        .template add<numsim_core::description_label<"optional key/value pairs merged into every produced shape's info blob: a JSON object (e.g. {\"orientation_deg\": 42.5}), or the same object as a JSON-encoded string">>();
    return s;
  }

  // Draw order: start x, y, z, end x, y, z, radius.
  // The radius distribution is truncated to (0, inf): non-positive draws
  // are resampled. See shape_input_base for the constructor check
  // and the redraw backstop.
  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    const value_type sx = _sx();
    const value_type sy = _sy();
    const value_type sz = _sz();
    const value_type ex = _ex();
    const value_type ey = _ey();
    const value_type ez = _ez();
    const value_type r =
        this->draw_positive(_radius, "polyline_tube_input", "radius");
    std::vector<std::array<value_type, 3>> centerline{
        {{sx, sy, sz}}, {{ex, ey, ez}}};
    auto shape = std::make_unique<polyline_tube<value_type>>(centerline, r);
    this->tag(*shape);
    return shape;
  }

private:
  distribution_base<value_type>& _sx;
  distribution_base<value_type>& _sy;
  distribution_base<value_type>& _sz;
  distribution_base<value_type>& _ex;
  distribution_base<value_type>& _ey;
  distribution_base<value_type>& _ez;
  distribution_base<value_type>& _radius;
};

} // namespace rvegen
