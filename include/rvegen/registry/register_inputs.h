#pragma once

#include <numsim-core/object_registry.h>

#include "../inputs/box_input.h"
#include "../inputs/circle_input.h"  // brings in distribution_map_t
#include "../inputs/ellipse_input.h"
#include "../inputs/rectangle_input.h"
#include "../inputs/shape_input_base.h"
#include "../inputs/sphere_input.h"
#include "../types.h"

namespace rvegen {

// Input registry — CtorArgs include a distribution_map for resolving the
// "use distribution X" cross-references encoded in the JSON config.
//
// Orchestration order is: register types → build distribution_map (each
// distribution constructed via its own registry) → walk shapes section,
// using the map to resolve names.
template <typename T = double>
using input_registry_t = numsim_core::object_registry<
    shape_input_base<T>,
    parameter_controller_t,
    parameter_handler_t const,
    distribution_map_t<T> const>;

template <typename T = double>
inline void register_circle_input() {
  input_registry_t<T>::instance()
      .template register_type<circle_input<T>>("circle_input");
}

template <typename T = double>
inline void register_sphere_input() {
  input_registry_t<T>::instance()
      .template register_type<sphere_input<T>>("sphere_input");
}

template <typename T = double>
inline void register_rectangle_input() {
  input_registry_t<T>::instance()
      .template register_type<rectangle_input<T>>("rectangle_input");
}

template <typename T = double>
inline void register_box_input() {
  input_registry_t<T>::instance()
      .template register_type<box_input<T>>("box_input");
}

template <typename T = double>
inline void register_ellipse_input() {
  input_registry_t<T>::instance()
      .template register_type<ellipse_input<T>>("ellipse_input");
}

template <typename T = double>
inline void register_all_inputs() {
  register_circle_input<T>();
  register_sphere_input<T>();
  register_rectangle_input<T>();
  register_box_input<T>();
  register_ellipse_input<T>();
}

} // namespace rvegen
