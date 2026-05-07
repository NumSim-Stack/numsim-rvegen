#pragma once

#include <numsim-core/object_registry.h>

#include "../generators/only_inside_generator.h"
#include "../generators/periodic_generator.h"
#include "../generators/random_generator.h"
#include "../generators/rve_generator_base.h"
#include "../types.h"

namespace rvegen {

// Generator registry — placement strategies configured from a
// parameter_handler. The strategy holds no engine; distributions held by
// shape_inputs do.
template <typename T = double>
using generator_registry_t = numsim_core::object_registry<
    rve_generator_base<T>,
    parameter_controller_t,
    parameter_handler_t const>;

template <typename T = double>
inline void register_only_inside_generator() {
  generator_registry_t<T>::instance()
      .template register_type<only_inside_generator<T>>("only_inside");
}

template <typename T = double>
inline void register_periodic_generator() {
  generator_registry_t<T>::instance()
      .template register_type<periodic_generator<T>>("periodic");
}

template <typename T = double>
inline void register_random_generator() {
  generator_registry_t<T>::instance()
      .template register_type<random_generator<T>>("random");
}

template <typename T = double>
inline void register_all_generators() {
  register_only_inside_generator<T>();
  register_periodic_generator<T>();
  register_random_generator<T>();
}

} // namespace rvegen
