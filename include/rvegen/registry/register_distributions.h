#pragma once

// Registers distribution types into numsim_core::object_registry so they can
// be selected by string name from a JSON config.
//
// Usage:
//   rvegen::register_all_distributions<double, std::mt19937>();
//
//   auto schema  = rvegen::distribution_registry_t<>::instance().schema("uniform_real");
//   auto handler = ...;  // populated from JSON via parameter_visitor_nlohmann
//   auto dist    = rvegen::distribution_registry_t<>::instance().create(
//                       "uniform_real", handler, engine);
//   auto v = (*dist)();   // sample
//
// The registry is parameterised by (T, Engine) — different combinations live
// in different singletons. The default is (double, std::mt19937).

#include <random>

#include <numsim-core/object_registry.h>

#include "../distributions/constant_distribution.h"
#include "../distributions/distribution_base.h"
#include "../distributions/normal_distribution.h"
#include "../distributions/uniform_real_distribution.h"
#include "../types.h"

namespace rvegen {

// One registry singleton per (T, Engine) combination. The CtorArgs match the
// schema-driven ctor signature on each registered distribution:
// `(parameter_handler_t const&, Engine&)`.
template <typename T = double, typename Engine = std::mt19937>
using distribution_registry_t = numsim_core::object_registry<
    distribution_base<T>,
    parameter_controller_t,
    parameter_handler_t const,
    Engine>;

template <typename T = double, typename Engine = std::mt19937>
inline void register_uniform_real_distribution() {
  distribution_registry_t<T, Engine>::instance()
      .template register_type<uniform_real_distribution<T, Engine>>("uniform_real");
}

template <typename T = double, typename Engine = std::mt19937>
inline void register_normal_distribution() {
  distribution_registry_t<T, Engine>::instance()
      .template register_type<normal_distribution<T, Engine>>("normal");
}

template <typename T = double, typename Engine = std::mt19937>
inline void register_constant_distribution() {
  distribution_registry_t<T, Engine>::instance()
      .template register_type<constant_distribution<T, Engine>>("constant");
}

// Convenience aggregator. Add new distribution registrations here as they
// gain JSON-driven ctor + parameters().
template <typename T = double, typename Engine = std::mt19937>
inline void register_all_distributions() {
  register_uniform_real_distribution<T, Engine>();
  register_normal_distribution<T, Engine>();
  register_constant_distribution<T, Engine>();
}

} // namespace rvegen
