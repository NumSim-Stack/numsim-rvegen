#pragma once

// Standard type aliases used across rvegen's JSON-driven layer.
//
// Every registered type's schema + ctor speak the same parameter_handler /
// input_parameter_controller flavour; centralising the aliases avoids drift.

#include <any>
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

} // namespace rvegen
