#pragma once

// Registers direction-distribution types (Bingham, future
// von-Mises-Fisher, generic-3×3 Bingham, ...) into a parallel
// `numsim_core::object_registry` keyed by `direction_distribution_base<T>`,
// so they can be picked by string name from a JSON config — same
// idiom as `register_distributions.h` does for scalar distributions.
//
// Usage:
//   rvegen::register_all_direction_distributions<double, std::mt19937>();
//
//   auto handler = ...;   // populated from JSON
//   auto dist = rvegen::direction_distribution_registry_t<>::instance().create(
//       "bingham", handler, engine);
//   auto v = (*dist)();   // sample a unit vector
//
// Each registered direction distribution must expose:
//   * `static parameter_controller_t parameters();`
//   * `<type>(parameter_handler_t const&, Engine&)` ctor

#include <random>

#include <numsim-core/object_registry.h>

#include "../distributions/bingham_distribution.h"
#include "../distributions/direction_distribution_base.h"
#include "../types.h"

namespace rvegen {

template <typename T = double, typename Engine = std::mt19937>
using direction_distribution_registry_t = numsim_core::object_registry<
    direction_distribution_base<T>,
    parameter_controller_t,
    parameter_handler_t const,
    Engine>;

template <typename T = double, typename Engine = std::mt19937>
inline void register_bingham_distribution() {
  direction_distribution_registry_t<T, Engine>::instance()
      .template register_type<bingham_distribution<T, Engine>>("bingham");
}

template <typename T = double, typename Engine = std::mt19937>
inline void register_all_direction_distributions() {
  register_bingham_distribution<T, Engine>();
}

} // namespace rvegen
