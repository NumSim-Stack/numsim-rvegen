#pragma once

// Standard type aliases used across rvegen's JSON-driven layer.
//
// Every registered type's schema + ctor speak the same parameter_handler /
// input_parameter_controller flavour; centralising the aliases avoids drift.

#include <any>
#include <limits>
#include <string>

#include <numsim-core/input_parameter_controller.h>
#include <numsim-core/parameter_handler.h>

namespace rvegen {

// The library's standard parameter handler — string keys, std::any values.
using parameter_handler_t = numsim_core::parameter_handler<std::string, std::any>;

// The library's standard input parameter controller, parameterised with
// our handler.
using parameter_controller_t =
    numsim_core::input_parameter_controller<std::string, parameter_handler_t>;

// One-sided bound helpers — wrap numsim_core::range<> with a
// std::numeric_limits<T>::max() (or ::lowest()) on the open side so call
// sites that need only a lower or only an upper bound stay compact.
//
// Usage:
//   s.template insert<std::size_t>("max_attempts")
//       .template add<numsim_core::is_required>()
//       .template add<rvegen::min_only<std::size_t{1}>>()
//       .template add<numsim_core::description_label<
//           "maximum placement attempts before giving up">>();
//
// The `decltype(Min_)` deduction means an integer literal like
// `std::size_t{1}` keeps the NTTP type aligned with the parameter's
// underlying type, avoiding a same-type mismatch between Min and Max.
template <auto Min_>
using min_only = numsim_core::range<
    Min_, std::numeric_limits<decltype(Min_)>::max()>;

template <auto Max_>
using max_only = numsim_core::range<
    std::numeric_limits<decltype(Max_)>::lowest(), Max_>;

} // namespace rvegen
