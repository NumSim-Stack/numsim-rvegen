#pragma once

// Convenience helper that turns a per-section JSON spec into a constructed
// instance via one of rvegen's registries. The spec must have a "type" field
// naming a registered type; the rest of the keys are the schema parameters.
//
// Used by both apps/rvegen_cli.cpp and the registry smoke tests.

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

#include "../json/parameter_visitor_nlohmann.h"
#include "../types.h"

namespace rvegen {

// Like build_from_json, with two reports:
//   * `on_unknown(type, key)` for every key in `spec` that the type's
//     schema does not know (typos would otherwise be ignored);
//   * `on_defaults(type, keys)` once, with the schema keys (sorted) that
//     `spec` left out and that got their default, when there are any.
// numsim-core can print the applied defaults itself, but only to a C
// stream; the callback lets the caller decide where such lines go.
template <typename Registry, typename OnUnknown, typename OnDefaults,
          typename... Args>
auto build_from_json_reported(Registry& registry,
                              nlohmann::json const& spec,
                              OnUnknown&& on_unknown,
                              OnDefaults&& on_defaults,
                              Args&... extra_ctor_args) {
  const auto type_name = spec.at("type").template get<std::string>();
  auto schema = registry.schema(type_name);

  parameter_handler_t handler;
  parameter_visitor_nlohmann visitor{spec};
  // log = nullptr: numsim-core would otherwise print "using default" lines
  // to C stdout, bypassing the pipeline hooks; missing required keys are
  // still named in the thrown exception.
  schema.accept(visitor, handler, nullptr);
  for (auto const& k : visitor.unknown_keys()) on_unknown(type_name, k);
  std::vector<std::string> defaulted;
  for (auto const& entry : schema) {
    auto const& key = entry.first;
    if (!spec.contains(key) && handler.contains(key)) defaulted.push_back(key);
  }
  if (!defaulted.empty()) {
    std::sort(defaulted.begin(), defaulted.end());
    on_defaults(type_name, defaulted);
  }

  return registry.create(type_name, handler, extra_ctor_args...);
}

// Like build_from_json, but calls `on_unknown(type, key)` for every key in
// `spec` that the type's schema does not know.
template <typename Registry, typename OnUnknown, typename... Args>
auto build_from_json_checked(Registry& registry,
                             nlohmann::json const& spec,
                             OnUnknown&& on_unknown,
                             Args&... extra_ctor_args) {
  return build_from_json_reported(
      registry, spec, std::forward<OnUnknown>(on_unknown),
      [](std::string const&, std::vector<std::string> const&) {},
      extra_ctor_args...);
}

template <typename Registry, typename... Args>
auto build_from_json(Registry& registry,
                     nlohmann::json const& spec,
                     Args&... extra_ctor_args) {
  return build_from_json_checked(
      registry, spec, [](std::string const&, std::string const&) {},
      extra_ctor_args...);
}

} // namespace rvegen
