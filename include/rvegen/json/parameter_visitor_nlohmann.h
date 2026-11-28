#pragma once

#include <any>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <set>
#include <typeindex>
#include <type_traits>
#include <typeinfo>
#include <utility>
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
// Integral targets (including the elements of std::vector<int>) are
// strict: the JSON value must be a number with an integral value. Integer
// literals and integral-valued floats (`5.0`, `1e5`, `100000.0`) are
// accepted; non-integral floats (`5.9`), strings and booleans are
// rejected, unsigned targets reject negative values, and values outside
// the target type's range are rejected. Each failure throws
// std::invalid_argument naming the key and the offending value —
// nlohmann's own get<> would silently static_cast instead (-1 → SIZE_MAX,
// 5.9 → 5).
//
// std::string targets must be JSON strings; the `metadata` key also accepts
// a JSON object (serialised to its string form). Other types throw
// std::invalid_argument naming the key.
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
    _queried.insert(key);
    return _j.contains(key);
  }

  // Keys of the JSON object the schema never asked about (typos / options
  // the type does not have). "type" is the registry discriminator, not a
  // parameter, and is never reported. Valid after schema.accept().
  [[nodiscard]] std::vector<std::string> unknown_keys() const {
    std::vector<std::string> out;
    if (!_j.is_object()) return out;
    for (auto const& [k, v] : _j.items())
      if (k != "type" && !_queried.contains(k)) out.push_back(k);
    return out;
  }

  [[nodiscard]] std::any read(key_type const& key,
                              std::type_index tid) const override {
    auto const& node = _j.at(key);

    // Scalars — order: most-specific first, then catch-all integer types.
    if (tid == typeid(double))      return std::any{node.get<double>()};
    if (tid == typeid(float))       return std::any{node.get<float>()};
    if (tid == typeid(bool))        return std::any{node.get<bool>()};
    if (tid == typeid(std::string)) return std::any{read_string(key, node)};

    if (tid == typeid(int))           return std::any{get_integral<int>(key, node)};
    if (tid == typeid(long))          return std::any{get_integral<long>(key, node)};
    if (tid == typeid(long long))     return std::any{get_integral<long long>(key, node)};
    if (tid == typeid(std::size_t))   return std::any{get_integral<std::size_t>(key, node)};
    if (tid == typeid(std::int32_t))  return std::any{get_integral<std::int32_t>(key, node)};
    if (tid == typeid(std::int64_t))  return std::any{get_integral<std::int64_t>(key, node)};
    if (tid == typeid(std::uint32_t)) return std::any{get_integral<std::uint32_t>(key, node)};
    if (tid == typeid(std::uint64_t)) return std::any{get_integral<std::uint64_t>(key, node)};

    // Sequences.
    if (tid == typeid(std::vector<double>))
      return std::any{node.get<std::vector<double>>()};
    if (tid == typeid(std::vector<int>)) {
      if (!node.is_array()) {
        throw std::invalid_argument(
            std::string{"parameter_visitor_nlohmann: key '"} + key +
            "' expects an array of integers, got " + node.dump());
      }
      std::vector<int> out;
      out.reserve(node.size());
      for (auto const& elem : node) out.push_back(get_integral<int>(key, elem));
      return std::any{std::move(out)};
    }
    if (tid == typeid(std::vector<std::string>))
      return std::any{node.get<std::vector<std::string>>()};

    throw std::runtime_error(
        std::string{"parameter_visitor_nlohmann: unsupported type for key '"} +
        key + "' (typeid: " + tid.name() + ")");
  }

private:
  // std::string targets: a JSON string, or -- for the `metadata` key only --
  // a JSON object, which is serialised to its string form so that
  // `"metadata": {"a": 1}` and `"metadata": "{\"a\": 1}"` are equivalent.
  // Any other type throws std::invalid_argument naming the key.
  [[nodiscard]] static std::string read_string(key_type const& key,
                                               nlohmann::json const& node) {
    if (node.is_string()) return node.get<std::string>();
    if (key == "metadata" && node.is_object()) return node.dump();
    throw std::invalid_argument(
        std::string{"parameter_visitor_nlohmann: key '"} + key +
        "' expects " +
        (key == "metadata" ? "a string or a JSON object" : "a string") +
        ", got " + node.dump());
  }

  // Strict JSON → integral conversion; see the class comment.
  template <typename I>
  [[nodiscard]] static I get_integral(key_type const& key,
                                      nlohmann::json const& node) {
    static_assert(std::is_integral_v<I>);
    auto fail = [&](char const* why) -> I {
      throw std::invalid_argument(
          std::string{"parameter_visitor_nlohmann: key '"} + key +
          "' expects " + why + ", got " + node.dump());
    };
    const char* expected = std::is_unsigned_v<I> ? "a non-negative integer"
                                                 : "an integer";
    if (node.is_number_float()) {
      const double d = node.get<double>();
      if (!std::isfinite(d) || d != std::trunc(d)) return fail(expected);
      constexpr double two63 = 9223372036854775808.0;
      constexpr double two64 = 18446744073709551616.0;
      if (d < 0.0) {
        if (std::is_unsigned_v<I>) return fail(expected);
        if (d < -two63) return fail("an integer within the target type's range");
        const auto v = static_cast<std::int64_t>(d);
        if (!std::in_range<I>(v)) return fail("an integer within the target type's range");
        return static_cast<I>(v);
      }
      if (d >= two64) return fail("an integer within the target type's range");
      const auto v = static_cast<std::uint64_t>(d);
      if (!std::in_range<I>(v)) return fail("an integer within the target type's range");
      return static_cast<I>(v);
    }
    if (!node.is_number_integer()) return fail(expected);
    if (node.is_number_unsigned()) {
      const auto v = node.get<std::uint64_t>();
      if (!std::in_range<I>(v)) return fail("an integer within the target type's range");
      return static_cast<I>(v);
    }
    const auto v = node.get<std::int64_t>();
    if (std::is_unsigned_v<I> && v < 0) return fail(expected);
    if (!std::in_range<I>(v)) return fail("an integer within the target type's range");
    return static_cast<I>(v);
  }

  nlohmann::json const& _j;
  mutable std::set<std::string> _queried;
};

} // namespace rvegen
