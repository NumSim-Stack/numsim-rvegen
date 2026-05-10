#pragma once

// JSON-driven input that produces a 2-point straight polyline_tube
// from (position, direction, length, radius) distributions.
//
// Why this AND `polyline_tube_input` (start+end+radius):
//   `polyline_tube_input` is the most general endpoint-pair shape but
//   is awkward for the realistic textile-composite use case "place
//   fibres aligned along a direction with some scatter". With
//   independent endpoint distributions the user has to think about
//   how to constrain length and direction implicitly through the
//   start/end joint distribution — easy to get wrong.
//
//   `polyline_tube_directional_input` makes the natural decomposition
//   first-class: position controls placement, direction controls
//   orientation (typically a Bingham — see PR #19 — when the parallel
//   `direction_distribution_base` lands), length is its own scalar,
//   radius is its own scalar. 5 fields instead of 7, and produces
//   useful microstructures rather than random sticks.
//
// What this header lands today (phase 2b of #3):
//   * `polyline_tube_directional_input<T>` — schema-driven:
//     `position_x_dist`, `position_y_dist`, `position_z_dist`,
//     `direction_x_dist`, `direction_y_dist`, `direction_z_dist`,
//     `length_dist`, `radius_dist`.
//   * Direction is sampled per axis via three scalar distributions
//     and renormalised inside `new_shape()` — a stop-gap until
//     the `direction_distribution_base` follow-up lets a single
//     Bingham/Watson distribution emit a unit vector in one call.
//   * Registration as `"polyline_tube_directional_input"` in
//     `register_inputs.h`.
//
// Out of scope here, ships in follow-up PRs against #3:
//   * Native `direction_distribution_base` field accepting a
//     `bingham_distribution` directly (one sample per shape, no
//     per-axis renormalisation).

#include <cmath>
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
class polyline_tube_directional_input final : public shape_input_base<T> {
public:
  using value_type = T;

  polyline_tube_directional_input(distribution_base<value_type>& px,
                                  distribution_base<value_type>& py,
                                  distribution_base<value_type>& pz,
                                  distribution_base<value_type>& dx,
                                  distribution_base<value_type>& dy,
                                  distribution_base<value_type>& dz,
                                  distribution_base<value_type>& length,
                                  distribution_base<value_type>& radius) noexcept
      : _px{px}, _py{py}, _pz{pz},
        _dx{dx}, _dy{dy}, _dz{dz},
        _length{length}, _radius{radius} {}

  polyline_tube_directional_input(parameter_handler_t const& handler,
                                  distribution_map_t<value_type> const& d)
      : polyline_tube_directional_input(
            *d.at(handler.template get<std::string>("position_x_dist")),
            *d.at(handler.template get<std::string>("position_y_dist")),
            *d.at(handler.template get<std::string>("position_z_dist")),
            *d.at(handler.template get<std::string>("direction_x_dist")),
            *d.at(handler.template get<std::string>("direction_y_dist")),
            *d.at(handler.template get<std::string>("direction_z_dist")),
            *d.at(handler.template get<std::string>("length_dist")),
            *d.at(handler.template get<std::string>("radius_dist"))) {
    if (handler.contains("phase_name")) {
      this->set_phase_name(handler.template get<std::string>("phase_name"));
    }
  }

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("position_x_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube centre x-coordinate">>();
    s.template insert<std::string>("position_y_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube centre y-coordinate">>();
    s.template insert<std::string>("position_z_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube centre z-coordinate">>();
    s.template insert<std::string>("direction_x_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube direction x-component (renormalised inside new_shape)">>();
    s.template insert<std::string>("direction_y_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube direction y-component (renormalised inside new_shape)">>();
    s.template insert<std::string>("direction_z_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube direction z-component (renormalised inside new_shape)">>();
    s.template insert<std::string>("length_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube length">>();
    s.template insert<std::string>("radius_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the tube radius">>();
    s.template insert<std::string>("phase_name")
        .template add<numsim_core::description_label<"optional phase name stamped onto every produced shape (default: empty / unassigned)">>();
    return s;
  }

  // Degenerate-direction note: if the direction distributions
  // produce (0, 0, 0), `mag = 0` and the normalisation is skipped.
  // Both centerline endpoints collapse to the position, so the
  // resulting polyline_tube has two coincident endpoints —
  // `is_inside` then degenerates to a point-radius test (effectively
  // a sphere of \c radius centred at \c position). Callers should
  // pick direction distributions whose support excludes the origin.
  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    const value_type cx = _px(), cy = _py(), cz = _pz();
    value_type dx = _dx(), dy = _dy(), dz = _dz();
    const value_type mag = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (mag > value_type{0}) { dx /= mag; dy /= mag; dz /= mag; }
    const value_type half = _length() * value_type{0.5};
    std::vector<std::array<value_type, 3>> centerline{
        {{cx - half * dx, cy - half * dy, cz - half * dz}},
        {{cx + half * dx, cy + half * dy, cz + half * dz}}};
    auto shape =
        std::make_unique<polyline_tube<value_type>>(centerline, _radius());
    this->tag(*shape);
    return shape;
  }

private:
  distribution_base<value_type>& _px;
  distribution_base<value_type>& _py;
  distribution_base<value_type>& _pz;
  distribution_base<value_type>& _dx;
  distribution_base<value_type>& _dy;
  distribution_base<value_type>& _dz;
  distribution_base<value_type>& _length;
  distribution_base<value_type>& _radius;
};

} // namespace rvegen
