#pragma once

#include <numsim-core/object_registry.h>

#include "../materials/linear_elastic.h"
#include "../materials/material_base.h"
#include "../materials/void_material.h"
#include "../types.h"

namespace rvegen {

template <typename T = double>
using material_registry_t = numsim_core::object_registry<
    material_base<T>,
    parameter_controller_t,
    parameter_handler_t const>;

template <typename T = double>
inline void register_linear_elastic() {
  material_registry_t<T>::instance()
      .template register_type<linear_elastic<T>>("linear_elastic");
}

template <typename T = double>
inline void register_void_material() {
  material_registry_t<T>::instance()
      .template register_type<void_material<T>>("void");
}

template <typename T = double>
inline void register_all_materials() {
  register_linear_elastic<T>();
  register_void_material<T>();
}

} // namespace rvegen
