#pragma once

// field_list — compile-time schema declaration that collapses three-way
// duplication into one source of truth.
//
// Today, every registered rvegen type declares its members in three
// different places:
//   1. C++ class fields (e.g. `T _radius;`)
//   2. The `parameters()` schema (e.g. `s.insert<T>("radius").add<is_required>();`)
//   3. The schema-driven ctor's name-by-name extraction (e.g.
//      `_radius{handler.get<T>("radius")}`)
//
// Drift between the three is a recurring source of bugs. This header
// provides the scaffolding to express the schema once via a
// `field_list<field<"name", T>...>` type alias, and have both
// `parameters()` and the ctor's tuple-of-values derive from that single
// declaration.
//
// What this header lands today (phase 1 of #1):
//   * `field<Name, T, Required = true>` — single-field declaration.
//   * `field_list<Fields...>::schema()` — produces a
//     `parameter_controller_t` matching the field declarations.
//   * `field_list<Fields...>::extract(handler)` — pulls the named
//     values out of a `parameter_handler_t` into a `std::tuple<>`.
//   * Compile-time uniqueness check on field names.
//   * Standalone unit tests that exercise both, with no migration of
//     existing types.
//
// Out of scope here, ships in follow-up PRs against #1:
//   * Per-field annotations: `description_label<"...">`, `unit_label<"...">`,
//     `range<Min, Max>`, `min_only<X>` / `max_only<X>`. The current
//     scaffolding only carries name + type + required-ness; richer
//     annotations are needed before existing types (which all carry
//     descriptions and many carry units / ranges) can migrate.
//   * Migration of registered types one-by-one — circle, sphere,
//     rectangle, etc. Each migration is a self-contained PR that
//     verifies its `schema()` output matches the pre-migration schema.
//   * `make_from_tuple<Derived>(extract(handler))` ergonomic helper
//     for ctors. Trivial once annotations are sorted.
//   * Default values for `Required = false` fields. Today an optional
//     field still throws on missing key in `extract`; richer
//     annotations will let it return a default instead.

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <tuple>

#include <numsim-core/input_parameter_controller.h>

#include "../types.h"

namespace rvegen {

// One field declaration. `Name` is a compile-time string (NTTP via
// numsim_core::fixed_string), `T` is the value type, `Required`
// determines whether the field is added with `is_required` in the
// emitted schema.
//
// Default `Required = true` mirrors the rvegen convention that nearly
// every schema field is required (matches the explicit `is_required`
// add()s in every existing `parameters()` declaration). Pass
// `Required=false` for the rare optional field — note that today
// `extract()` will still throw on a missing key, so optional fields
// need defaults to be useful (annotation work is the next phase
// against #1).
template <numsim_core::fixed_string Name, typename T, bool Required = true>
struct field {
  static constexpr auto name = Name;
  using value_type = T;
  static constexpr bool required = Required;
};

namespace detail {

// Compile-time check that a list of field names is unique. Catches the
// `field_list<field<"x", T>, field<"y", T>, field<"x", T>>` typo at
// type definition rather than at runtime. O(N²) — fine for typical
// schemas (3–7 fields, never more than ~20).
template <typename... Fields>
constexpr bool field_names_unique() {
  std::array<std::string_view, sizeof...(Fields)> names{
      Fields::name.view()...};
  for (std::size_t i = 0; i < names.size(); ++i) {
    for (std::size_t j = i + 1; j < names.size(); ++j) {
      if (names[i] == names[j]) return false;
    }
  }
  return true;
}

} // namespace detail

// A pack of fields. `schema()` builds the `parameter_controller_t`;
// `extract()` pulls a tuple of values out of a `parameter_handler_t`.
//
// Note: `extract()` returns the values by value via `std::make_tuple`,
// which copies. `parameter_handler::get<T>` returns `T const&`, so a
// string-typed field is copied once per `extract()`. Negligible at
// construction time (handler.get is one-shot per shape ctor) and
// in line with how the existing inputs read from the handler today.
template <typename... Fields>
struct field_list {
  static_assert(detail::field_names_unique<Fields...>(),
                "field_list: duplicate field names");

  [[nodiscard]] static parameter_controller_t schema() {
    parameter_controller_t s;
    (insert_one<Fields>(s), ...);
    return s;
  }

  // Returns a `std::tuple<typename Fields::value_type...>` populated
  // by name-keyed lookups against the supplied handler. The intent is
  // for ctors to forward this through `std::make_from_tuple` once
  // richer field annotations are wired in (follow-up).
  [[nodiscard]] static auto extract(parameter_handler_t const& handler) {
    return std::make_tuple(
        handler.template get<typename Fields::value_type>(
            std::string{Fields::name.view()})...);
  }

private:
  template <typename F>
  static void insert_one(parameter_controller_t& s) {
    auto& slot = s.template insert<typename F::value_type>(
        std::string{F::name.view()});
    if constexpr (F::required) {
      slot.template add<numsim_core::is_required>();
    }
  }
};

} // namespace rvegen
