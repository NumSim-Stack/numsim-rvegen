#pragma once

#include <array>

#include <numsim-core/object_registry.h>

#include "../termination/number_of_inclusions.h"
#include "../termination/termination_base.h"
#include "../termination/until_full.h"
#include "../termination/volume_fraction.h"
#include "../types.h"

namespace rvegen {

// Termination registry — CtorArgs are (parameter_handler_t const&,
// std::array<T,3> const& domain_box). Some terminations (volume_fraction)
// need the domain to compute the target; others (number_of_inclusions)
// ignore it but accept it for uniform construction.
template <typename T = double>
using termination_registry_t = numsim_core::object_registry<
    termination_base<T>,
    parameter_controller_t,
    parameter_handler_t const,
    std::array<T, 3> const>;

template <typename T = double>
inline void register_number_of_inclusions() {
  termination_registry_t<T>::instance()
      .template register_type<number_of_inclusions<T>>("number_of_inclusions");
}

template <typename T = double>
inline void register_volume_fraction() {
  termination_registry_t<T>::instance()
      .template register_type<volume_fraction<T>>("volume_fraction");
}

template <typename T = double>
inline void register_until_full() {
  termination_registry_t<T>::instance()
      .template register_type<until_full<T>>("until_full");
}

template <typename T = double>
inline void register_all_terminations() {
  register_number_of_inclusions<T>();
  register_volume_fraction<T>();
  register_until_full<T>();
}

} // namespace rvegen
