#pragma once

// Convenience helper that turns a per-section JSON spec into a constructed
// instance via one of rvegen's registries. The spec must have a "type" field
// naming a registered type; the rest of the keys are the schema parameters.
//
// Used by both apps/rvegen_cli.cpp and the registry smoke tests.

#include <string>

#include "nlohmann/json.hpp"

#include "../json/parameter_visitor_nlohmann.h"
#include "../types.h"

namespace rvegen {

template <typename Registry, typename... Args>
auto build_from_json(Registry& registry,
                     nlohmann::json const& spec,
                     Args&... extra_ctor_args) {
  const auto type_name = spec.at("type").template get<std::string>();
  auto schema = registry.schema(type_name);

  parameter_handler_t handler;
  parameter_visitor_nlohmann visitor{spec};
  schema.accept(visitor, handler);

  return registry.create(type_name, handler, extra_ctor_args...);
}

} // namespace rvegen
