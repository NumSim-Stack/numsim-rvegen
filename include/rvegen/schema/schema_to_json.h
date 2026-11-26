#pragma once

// schema_to_json — turns a `parameter_controller_t` (the schema every
// registered type exposes through `registry.schema(name)`) into a plain
// JSON object for introspection (Python `rvegen.<category>.schema(name)`,
// GUIs, docs):
//
//   { "<field>": { "type": "double", "required": true,
//                  "description": "...", "unit": "m",
//                  "min": 0.0, "max": 1.0, "has_default": false }, ... }
//
// "min"/"max" appear when a range hint is attached, "unit" /
// "description" when those hints are, "has_default" is true for fields
// carrying numsim_core::set_default (an absent key then is not an error).

#include <any>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <typeindex>
#include <vector>

#include <nlohmann/json.hpp>
#include <numsim-core/input_parameter_controller.h>

#include "../types.h"

namespace rvegen {

namespace detail {

using schema_base_t =
    numsim_core::input_parameter_base<std::string, parameter_handler_t>;

template <typename T>
using schema_typed_t =
    numsim_core::input_parameter<T, std::string, parameter_handler_t>;

template <typename T>
inline bool describe_if(schema_base_t const& base, char const* type_name,
                        nlohmann::json& out) {
  auto const* p = dynamic_cast<schema_typed_t<T> const*>(&base);
  if (!p) return false;
  out["type"] = type_name;
  out["required"] = false;
  out["has_default"] = false;
  for (auto const& check : p->checks()) {
    auto const* c = check.get();
    if (dynamic_cast<numsim_core::is_required<T, std::string,
                                              parameter_handler_t> const*>(c))
      out["required"] = true;
    if (dynamic_cast<numsim_core::set_default<T, std::string,
                                              parameter_handler_t> const*>(c))
      out["has_default"] = true;
    if (auto const* d = dynamic_cast<numsim_core::description_hint_base const*>(c))
      out["description"] = std::string{d->description_text()};
    if (auto const* u = dynamic_cast<numsim_core::units_hint_base const*>(c))
      out["unit"] = std::string{u->units()};
    if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
      if (auto const* r = dynamic_cast<numsim_core::range_hint_base const*>(c)) {
        out["min"] = std::any_cast<T>(r->min_value());
        out["max"] = std::any_cast<T>(r->max_value());
      }
    }
  }
  return true;
}

} // namespace detail

[[nodiscard]] inline nlohmann::json
schema_to_json(parameter_controller_t const& schema) {
  nlohmann::json result = nlohmann::json::object();
  for (auto const& [key, param] : schema) {
    nlohmann::json f = nlohmann::json::object();
    // Same type list the JSON visitor can read.
    const bool known =
        detail::describe_if<double>(*param, "double", f) ||
        detail::describe_if<float>(*param, "float", f) ||
        detail::describe_if<bool>(*param, "bool", f) ||
        detail::describe_if<std::string>(*param, "string", f) ||
        detail::describe_if<int>(*param, "int", f) ||
        detail::describe_if<long>(*param, "int64", f) ||
        detail::describe_if<long long>(*param, "int64", f) ||
        detail::describe_if<std::size_t>(*param, "size_t", f) ||
        detail::describe_if<std::uint32_t>(*param, "uint32", f) ||
        detail::describe_if<std::uint64_t>(*param, "uint64", f) ||
        detail::describe_if<std::vector<double>>(*param, "list[double]", f) ||
        detail::describe_if<std::vector<int>>(*param, "list[int]", f) ||
        detail::describe_if<std::vector<std::string>>(*param, "list[string]", f);
    if (!known) f["type"] = "unknown";
    result[key] = std::move(f);
  }
  return result;
}

} // namespace rvegen
