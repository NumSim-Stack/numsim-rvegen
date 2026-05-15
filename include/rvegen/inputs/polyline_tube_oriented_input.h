#pragma once

// Polyline-tube input variant that samples its direction from a
// single `direction_distribution_base` (typically a
// `bingham_distribution`) instead of three independent scalar
// distributions like `polyline_tube_directional_input` does.
//
// Why this AND `polyline_tube_directional_input`:
//   The 3-axis scalar approach in `polyline_tube_directional_input`
//   was a stop-gap for shipping #19 (Bingham distribution) and #3
//   (swept-tube generation) before the polymorphic
//   `direction_distribution_base` (#112) existed. Per-axis sampling
//   followed by post-hoc renormalisation:
//   1. Loses the joint correlation structure the Bingham encodes
//      (a Bingham draws ONE coherent unit vector; sampling x/y/z
//      independently from three marginals plus renormalisation gives
//      a distribution that isn't Bingham).
//   2. Makes the JSON config awkward — the user has to declare three
//      distributions when conceptually they want one orientation
//      distribution.
//
//   This input takes a single direction distribution by reference,
//   sampled once per produced shape. The position / length / radius
//   scalar distributions are unchanged.
//
// What this header lands today:
//   * Direct-ctor wiring — caller passes the direction distribution
//     in C++. No JSON-driven dependency injection through a registry
//     yet; a `direction_distribution_map_t` registry path is the
//     natural follow-up and parallels what `distribution_map_t`
//     does for scalar distributions today.

#include <array>
#include <memory>
#include <vector>

#include "../distributions/direction_distribution_base.h"
#include "../distributions/distribution_base.h"
#include "../shapes/polyline_tube.h"
#include "../types.h"
#include "shape_input_base.h"

namespace rvegen {

template <typename T = double>
class polyline_tube_oriented_input final : public shape_input_base<T> {
public:
  using value_type = T;

  polyline_tube_oriented_input(distribution_base<value_type>& px,
                               distribution_base<value_type>& py,
                               distribution_base<value_type>& pz,
                               direction_distribution_base<value_type>& dir,
                               distribution_base<value_type>& length,
                               distribution_base<value_type>& radius) noexcept
      : _px{px}, _py{py}, _pz{pz},
        _dir{dir}, _length{length}, _radius{radius} {}

  // `_dir()` already returns a unit vector by contract (see
  // `direction_distribution_base`), so no renormalisation is needed
  // and the joint distribution structure is preserved. Two
  // centerline points are placed symmetrically around the sampled
  // centre at ±half_length·direction.
  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    const value_type cx = _px(), cy = _py(), cz = _pz();
    const auto d = _dir();
    const value_type half = _length() * value_type{0.5};
    std::vector<std::array<value_type, 3>> centerline{
        {{cx - half * d[0], cy - half * d[1], cz - half * d[2]}},
        {{cx + half * d[0], cy + half * d[1], cz + half * d[2]}}};
    auto shape =
        std::make_unique<polyline_tube<value_type>>(centerline, _radius());
    this->tag(*shape);
    return shape;
  }

private:
  distribution_base<value_type>& _px;
  distribution_base<value_type>& _py;
  distribution_base<value_type>& _pz;
  direction_distribution_base<value_type>& _dir;
  distribution_base<value_type>& _length;
  distribution_base<value_type>& _radius;
};

} // namespace rvegen
