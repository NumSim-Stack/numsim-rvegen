#pragma once

#include <any>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include <numsim-core/input_parameter_controller.h>

#include "nlohmann/json.hpp"

namespace rvegen {

// Concrete parameter_visitor_base<std::string> that pulls values from an
// nlohmann::json object. Wires a JSON-driven config into numsim-core's
// schema-validation pipeline:
//
//   nlohmann::json j = ...;
//   parameter_visitor_nlohmann visitor{j};
//   numsim_core::parameter_handler<std::string, std::any> handler;
//   schema.accept(visitor, handler);   // populates handler, runs validation
//
// Supported types (the std::type_index in `read`):
//   double, float
//   bool
//   int, long, long long, std::size_t,
//   std::int32_t, std::int64_t, std::uint32_t, std::uint64_t
//   std::string
//   std::vector<double>, std::vector<int>, std::vector<std::string>
//
// Adding a new supported type: add an `if (tid == typeid(T))` branch to
// `read()`. Unknown types throw std::runtime_error with the type name.
class parameter_visitor_nlohmann
    : public numsim_core::parameter_visitor_base<std::string> {
public:
  using key_type = std::string;

  explicit parameter_visitor_nlohmann(nlohmann::json const& j) noexcept
      : _j{j} {}

  [[nodiscard]] bool contains(key_type const& key) const override {
    return _j.contains(key);
  }

  [[nodiscard]] std::any read(key_type const& key,
                              std::type_index tid) const override {
    auto const& node = _j.at(key);

    // Scalars — order: most-specific first, then catch-all integer types.
    if (tid == typeid(double))      return std::any{node.get<double>()};
    if (tid == typeid(float))       return std::any{node.get<float>()};
    if (tid == typeid(bool))        return std::any{node.get<bool>()};
    if (tid == typeid(std::string)) return std::any{node.get<std::string>()};

    if (tid == typeid(int))                return std::any{node.get<int>()};
    if (tid == typeid(long))               return std::any{node.get<long>()};
    if (tid == typeid(long long))          return std::any{node.get<long long>()};
    if (tid == typeid(std::size_t))        return std::any{node.get<std::size_t>()};
    if (tid == typeid(std::int32_t))       return std::any{node.get<std::int32_t>()};
    if (tid == typeid(std::int64_t))       return std::any{node.get<std::int64_t>()};
    if (tid == typeid(std::uint32_t))      return std::any{node.get<std::uint32_t>()};
    if (tid == typeid(std::uint64_t))      return std::any{node.get<std::uint64_t>()};

    // Sequences.
    if (tid == typeid(std::vector<double>))
      return std::any{node.get<std::vector<double>>()};
    if (tid == typeid(std::vector<int>))
      return std::any{node.get<std::vector<int>>()};
    if (tid == typeid(std::vector<std::string>))
      return std::any{node.get<std::vector<std::string>>()};

    throw std::runtime_error(
        std::string{"parameter_visitor_nlohmann: unsupported type for key '"} +
        key + "' (typeid: " + tid.name() + ")");
  }

private:
  nlohmann::json const& _j;
};

} // namespace rvegen
